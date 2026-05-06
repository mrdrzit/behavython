from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from natsort import os_sorted
from PySide6.QtWidgets import QWidget, QMessageBox, QListWidgetItem
from PySide6.QtCore import QEvent, Qt, QTimer
from behavython.pipeline.workflow import run_analysis_workflow
from behavython.services.validation import is_model_installed, validate_analysis_request
from behavython.core.app_context import AppContext
from behavython.core.defaults import DEBUG_STYLE, DEFAULT_STYLE, VALIDATION_CHECKBOX_ACTIVE, VALIDATION_CHECKBOX_FADED, ANALYSIS_REQUIRED_SUFFIXES
from behavython.core.paths import FFMPEG_URL, MODELS_URLS
from behavython.pipeline.models import (
    AnalysisOptions,
    AnalysisRequest,
    DLCAnnotatedVideoRequest,
    DLCClearUnusedFilesRequest,
    DLCFrameExtractionRequest,
    DLCVideoAnalysisRequest,
    DLCAnalyzeFramesRequest,
)
from behavython.services.validation import validate_config_path, validate_video_paths
from behavython.pipeline.plugins.dlc import (
    run_clear_unused_files,
    run_dlc_video_analysis,
    run_extract_frames,
    run_create_annotated_video,
    run_analyze_frames,
)
from behavython.services.validation import validate_json_config
from behavython.gui.dialogs import ask_yes_no, show_warning, show_info, show_worker_error
from behavython.gui.dialogs import select_file, select_files, select_folder, select_save_folder
from behavython.services.logging import LoggingService
from behavython.core.utils import get_ffmpeg_path, resolve_analysis_input, resolve_output_folder, resolve_video_input, load_or_repair_dlc_yaml, group_analysis_files
from behavython.pipeline.models import (
    AnalysisInputSource,
    OutputFolderSource,
    ResolvedAnalysisInput,
    ResolvedOutputFolder,
    ResolvedVideoInput,
    VideoInputSource,
)
from behavython.gui.video_cropper import VideoCropperDialog
from behavython.tasks.task_runner import TaskRunner
from behavython.gui.dependencies import get_model_path, is_ffmpeg_installed, get_os_name, get_unix_install_instructions, dependencyDownloadDialog
from PySide6.QtCore import QThread, Signal


class ConfigLoaderThread(QThread):
    config_loaded = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, config_path, parent=None):
        super().__init__(parent)
        self.config_path = config_path

    def run(self):
        try:
            config_dict, _, _ = load_or_repair_dlc_yaml(self.config_path)
            bodyparts = config_dict.get("bodyparts", [])
            self.config_loaded.emit(bodyparts)
        except Exception as e:
            self.error_occurred.emit(str(e))


class CodecProbeThread(QThread):
    warning_detected = Signal(list)  # list of problematic files

    def __init__(self, folder_path, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path

    def run(self):
        from pathlib import Path
        from behavython.services.video_service import VideoService
        from behavython.core.defaults import ANALYSIS_REQUIRED_SUFFIXES

        folder = Path(self.folder_path)
        if not folder.exists():
            return

        extensions = ANALYSIS_REQUIRED_SUFFIXES["video"]
        videos = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions]

        problematic = []
        for vid in videos:
            if VideoService.is_legacy_codec(vid):
                problematic.append(str(vid))

        if problematic:
            self.warning_detected.emit(problematic)


class BehavythonMainWindow(QWidget):
    def __init__(self, interface, context: AppContext):
        super().__init__()
        self.interface = interface
        self.context = context
        self.progress_bar = self.interface.progress_bar
        self.logs = LoggingService(self.interface)
        self.logger = logging.getLogger("behavython")
        self.console_logger = logging.getLogger("behavython.console")

        self.runner = TaskRunner(
            threadpool=self.context.threadpool,
            on_log=self.on_worker_log,
            on_warning=self.on_worker_warning,
            on_progress=self.on_worker_progress,
            on_result=self.on_worker_result,
            on_error=self.on_worker_error,
            on_finished=self.on_worker_finished,
        )

        self._debug_click_count = 0
        self._last_debug_click_time = 0.0

        if hasattr(self.interface, "behavython_logo"):
            self.interface.behavython_logo.installEventFilter(self)

        self._connect_analysis_tab()
        self._connect_dlc_tab()
        self._connect_video_editing_tab()
        self._initialize_ui_state()

        QTimer.singleShot(200, self._check_startup_dependencies)

    @property
    def debug_mode(self) -> bool:
        return self.context.debug_mode

    @debug_mode.setter
    def debug_mode(self, value: bool) -> None:
        self.context.debug_mode = value

    def show(self) -> None:
        self.interface.show()

    def _initialize_ui_state(self) -> None:
        self.enable_analysis_buttons()
        self.enable_analysis()
        self.toggle_analyze_from_file_button()
        self.on_experiment_type_changed(self.interface.type_combobox.currentText())
        self.set_advanced_tabs_visible(False)
        self.on_crop_video_toggled()
        self.toggle_video_editing_buttons()
        if hasattr(self.interface, "extract_frames_data_process_spinbox"):
            self.interface.extract_frames_data_process_spinbox.setValue(20)
        self.toggle_assisted_analysis_ready()

    def _set_gui_log_message(self, target: str, message: str) -> None:
        self.logs.clear(target)
        if message:
            self.logs.append(target, message)

    def _check_startup_dependencies(self) -> None:
        if not is_ffmpeg_installed():
            current_os = get_os_name()

            if current_os == "Windows":
                reply = QMessageBox.question(
                    self.interface,
                    "Missing Dependency",
                    "FFmpeg is required for video extraction and processing but was not found.\n\nWould you like to download it now? (~100MB)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    dialog = dependencyDownloadDialog(target="ffmpeg", url=FFMPEG_URL, parent=self.interface)
                    dialog.start_download()

                    if not is_ffmpeg_installed():
                        show_warning(self.interface, "Warning", "FFmpeg installation incomplete. Video processing will fail.")
                    else:
                        self._set_gui_log_message("dlc", "FFmpeg installed successfully!")
                else:
                    show_warning(self.interface, "Warning", "FFmpeg is required. Video extraction and cropping will fail.")

            else:
                # macOS / Linux explicit instructions
                instructions = get_unix_install_instructions()
                QMessageBox.warning(
                    self.interface,
                    "Missing Dependency: FFmpeg",
                    f"FFmpeg is required for video processing but was not found on your system.\n\n{instructions}",
                )
        else:
            # FFmpeg is already installed, just print the path once at startup!
            ffmpeg_path = get_ffmpeg_path()
            self.logger.info(f"Startup check passed. Using FFmpeg at: {ffmpeg_path}")
            self.console_logger.info(f"Using FFmpeg at: {ffmpeg_path}")

    def _ensure_model_installed(self, model_name: str) -> bool:
        """
        Checks if a model is installed. If not, prompts the user to download it.
        Returns True if the model is ready to use, False if cancelled or failed.
        """
        if is_model_installed(model_name):
            return True

        reply = QMessageBox.question(
            self.interface,
            "Missing Neural Network",
            f"The pre-trained network '{model_name}' is required for this action but was not found.\n\nWould you like to download it now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            url = MODELS_URLS.get(model_name)
            if not url:
                show_warning(self.interface, "Download Error", f"No download URL configured for {model_name}.")
                return False

            dialog = dependencyDownloadDialog(target=model_name, url=url, parent=self.interface)
            dialog.start_download()

            if is_model_installed(model_name):
                self._set_gui_log_message("dlc", f"Model '{model_name}' installed successfully!")
                return True
            else:
                show_warning(self.interface, "Warning", f"Installation of '{model_name}' failed or was cancelled.")
                return False

        return False

    def eventFilter(self, obj, event):
        if hasattr(self.interface, "behavython_logo") and obj == self.interface.behavython_logo:
            # Catch BOTH standard clicks and double-clicks to prevent the "Double-Click Trap"
            if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
                # Method 1: Instant Modifier Click (Ctrl + Shift + Click)
                modifiers = event.modifiers()
                if (modifiers & Qt.KeyboardModifier.ControlModifier) and (modifiers & Qt.KeyboardModifier.ShiftModifier):
                    self._debug_click_count = 0  # Reset the counter cleanly
                    self.toggle_debug_mode()
                    return True

                # Method 2: Rapid 3-Click
                current_time = time.time()

                # Reset counter if more than 1 second passes between clicks
                if current_time - self._last_debug_click_time > 1.0:
                    self._debug_click_count = 0

                self._debug_click_count += 1
                self._last_debug_click_time = current_time

                # Trigger on 3 fast clicks
                if self._debug_click_count >= 3:
                    self._debug_click_count = 0
                    self.toggle_debug_mode()

                return True  # Event handled, prevent default

        return super().eventFilter(obj, event)

    def set_advanced_tabs_visible(self, visible: bool):
        self.tab_widget = self.interface.tabWidget
        data_tab_index = self.tab_widget.indexOf(self.interface.data_process_tab)

        if data_tab_index != -1:
            self.tab_widget.setTabVisible(data_tab_index, visible)

    # ------------------------------------------------------------------
    # Analysis tab
    # ------------------------------------------------------------------

    def _connect_analysis_tab(self) -> None:
        self.interface.analysis_button.clicked.connect(self.on_run_analysis_clicked)
        self.interface.clear_button.clicked.connect(self.clear_analysis_tab)
        self.interface.load_configuration_button.clicked.connect(self.on_load_configuration_clicked)
        self.interface.type_combobox.currentTextChanged.connect(self.on_experiment_type_changed)
        self.interface.plot_data_checkbox.toggled.connect(self.on_plot_enabled_changed)
        self.interface.crop_video_checkbox.toggled.connect(self.on_crop_video_toggled)

    def clear_analysis_tab(self) -> None:
        self.interface.type_combobox.setCurrentIndex(1)
        self.interface.algo_type_combobox.setCurrentIndex(0)
        self.interface.animal_combobox.setCurrentIndex(0)
        self.interface.arena_width_spinbox.setValue(30)
        self.interface.arena_height_spinbox.setValue(30)
        self.interface.frames_per_second_spinbox.setValue(30)
        self.interface.task_duration_spinbox.setValue(300)
        self.interface.crop_video_time_spinbox.setValue(0)
        self.interface.fig_max_size.setCurrentIndex(2)
        self.interface.crop_video_checkbox.setChecked(False)
        self.interface.plot_data_checkbox.setChecked(True)
        self.interface.generate_validation_video_checkbox.setChecked(False)
        self.interface.use_default_network_button.setChecked(False)
        self.interface.config_path_lineedit.clear()
        self.interface.get_config_path_button.setEnabled(True)
        self.logs.clear("resume")

    def on_load_configuration_clicked(self) -> None:
        path = select_file(self.interface, "Select configuration JSON", "JSON files (*.json)")
        if not path:
            return

        config_data = validate_json_config(path)

        if not config_data:
            self.logger.info("Invalid configuration file selected: %s", path)
            show_warning(self.interface, "Configuration file", "The selected file is not a valid JSON configuration.")
            return

        self.apply_analysis_configuration(config_data)
        self._set_gui_log_message("resume", "Configuration file loaded successfully!")
        self.logger.info("Configuration file loaded: %s", path)

    def apply_analysis_configuration(self, data: dict) -> None:
        mapping = {
            "Arena width": self.interface.arena_width_spinbox,
            "Arena height": self.interface.arena_height_spinbox,
            "Video framerate": self.interface.frames_per_second_spinbox,
            "Task Duration": self.interface.task_duration_spinbox,
            "Amount to trim": self.interface.crop_video_time_spinbox,
        }

        for key, widget in mapping.items():
            if key in data:
                try:
                    val = data[key]
                    if val is not None and str(val).strip() != "":
                        widget.setValue(int(float(val)))
                except (ValueError, TypeError):
                    self.logger.warning("Could not parse config value '%s' for %s", data[key], key)

    def build_analysis_options(self) -> AnalysisOptions:
        threshold = 0.0267 if self.interface.animal_combobox.currentIndex() == 0 else 0.0667

        return AnalysisOptions(
            arena_width=self.interface.arena_width_spinbox.value(),
            arena_height=self.interface.arena_height_spinbox.value(),
            frames_per_second=self.interface.frames_per_second_spinbox.value(),
            experiment_type=self.interface.type_combobox.currentText().lower().strip().replace(" ", "_"),
            max_fig_res=str(self.interface.fig_max_size.currentText()).replace(" ", "").replace("x", ",").split(","),
            algo_type=self.interface.algo_type_combobox.currentText().lower().strip(),
            threshold=threshold,
            task_duration=self.interface.task_duration_spinbox.value(),
            trim_amount=self.interface.crop_video_time_spinbox.value(),
            crop_video=self.interface.crop_video_checkbox.isChecked(),
            plot_options="plotting_enabled" if self.interface.plot_data_checkbox.isChecked() else "plotting_disabled",
            generate_video=self.interface.generate_validation_video_checkbox.isChecked(),
        )

    def on_run_analysis_clicked(self) -> None:
        selected_files = select_files(
            self.interface,
            "Select analysis files",
            "DLC files (*.csv *.jpg *.jpeg *.png *.mp4)",
        )
        resolved_input = self._build_analysis_input(selected_files)

        if not resolved_input.has_paths:
            message_parts = ["No valid analysis files were selected."]
            if resolved_input.warnings:
                message_parts.append("")
                message_parts.extend(resolved_input.warnings)

            show_warning(self.interface, "Analysis input", "\n".join(message_parts))
            return

        selected_output_folder = select_save_folder(self.interface, "Select output folder")
        resolved_output_folder = self._build_output_folder(selected_output_folder)

        if not resolved_output_folder.has_path:
            message_parts = ["No valid output folder was selected."]
            if resolved_output_folder.warnings:
                message_parts.append("")
                message_parts.extend(resolved_output_folder.warnings)

            show_warning(self.interface, "Analysis output", "\n".join(message_parts))
            return

        # Build options early to evaluate the experiment type
        options = self.build_analysis_options()

        # Prompt for batch configuration if the experiment requires geometry
        config_path = None
        if options.experiment_type in ["open_field", "elevated_plus_maze"]:
            groups = group_analysis_files(resolved_input.paths)
            missing_configs = any(not group["files"]["config"] for group in groups)

            if missing_configs:
                config_path = select_file(self.interface, "Select Global Fallback Configuration (Optional if all animals have configs)", "JSON Files (*.json)")
                if not config_path:
                    show_warning(self.interface, "Missing Configuration", "You must select an arena configuration JSON to run a maze analysis for animals missing individual configs.")
                    return

        request = AnalysisRequest(
            input_files=resolved_input.paths,
            output_folder=resolved_output_folder.path,
            options=options,
            config_path=config_path,
        )

        errors = validate_analysis_request(request)
        if errors:
            self.logger.info("Analysis request validation failed: %s", "; ".join(errors))
            show_warning(self.interface, "Analysis validation", "\n".join(errors))
            return

        self.logs.clear("resume")
        self.progress_bar.setValue(0)
        self.runner.submit(run_analysis_workflow, request, debug_mode=self.debug_mode)
        self.logger.info(
            "Analysis workflow submitted with %d input file(s). Output folder: %s",
            len(request.input_files),
            request.output_folder,
        )

    def on_experiment_type_changed(self, text: str = None) -> None:
        if text is None:
            text = self.interface.type_combobox.currentText()

        should_be_active = self.interface.plot_data_checkbox.isChecked()

        if hasattr(self.interface, "generate_validation_video_checkbox"):
            checkbox = self.interface.generate_validation_video_checkbox
            checkbox.setEnabled(should_be_active)

            if should_be_active:
                checkbox.setStyleSheet(VALIDATION_CHECKBOX_ACTIVE)
            else:
                checkbox.setStyleSheet(VALIDATION_CHECKBOX_FADED)
                checkbox.setChecked(False)  # Force uncheck if invalid context

    def on_plot_enabled_changed(self, *args) -> None:
        self.on_experiment_type_changed()

    def on_crop_video_toggled(self, *args) -> None:
        if self.interface.crop_video_checkbox.isChecked():
            self.interface.crop_video_time_spinbox.setEnabled(True)
            self.interface.crop_video_time_label.setEnabled(True)
        else:
            self.interface.crop_video_time_spinbox.setEnabled(False)
            self.interface.crop_video_time_label.setEnabled(False)

    # ------------------------------------------------------------------
    # DeepLabCut tab
    # ------------------------------------------------------------------

    def _connect_dlc_tab(self) -> None:
        self.interface.get_config_path_button.clicked.connect(self.on_select_dlc_config_clicked)
        self.interface.get_videos_path_button.clicked.connect(self.on_select_video_folder_clicked)
        self.interface.get_file_to_analyze_button.clicked.connect(self.on_select_video_list_clicked)
        self.interface.folder_to_create_annotated_video_button.clicked.connect(self.on_select_annotated_output_clicked)

        self.interface.config_path_lineedit.textChanged.connect(self.enable_analysis)
        self.interface.video_folder_lineedit.textChanged.connect(self.enable_analysis_buttons)
        self.interface.analyze_from_file_lineedit.textChanged.connect(self.toggle_analyze_from_file_button)

        self.interface.dlc_video_analyze_button.clicked.connect(self.on_run_dlc_analysis_clicked)
        self.interface.analyze_from_file_button.clicked.connect(self.on_run_dlc_analysis_from_file_clicked)
        self.interface.clear_unused_files_button.clicked.connect(self.on_clear_unused_files_clicked)
        self.interface.get_frames_button.clicked.connect(self.on_get_frames_clicked)
        self.interface.analyze_from_file_get_frames_button.clicked.connect(self.on_get_frames_from_file_clicked)
        self.interface.create_annotated_video_button.clicked.connect(self.on_create_annotated_video_clicked)
        self.interface.use_default_network_button.toggled.connect(self.on_use_default_network_toggled)

        # Assisted Labeling (Data Process)
        self.interface.get_config_path_data_process.clicked.connect(self.on_select_refining_dlc_config_clicked)
        self.interface.get_frames_path_data_process.clicked.connect(self.on_select_refining_video_folder_clicked)
        self.interface.start_assisted_labeling_data_process.clicked.connect(self.on_run_analyze_frames_clicked)
        self.interface.config_path_data_process_path.textChanged.connect(self.toggle_assisted_analysis_ready)
        self.interface.video_folder_data_process.textChanged.connect(self.toggle_assisted_analysis_ready)
        self.interface.extract_frames_data_process_spinbox.valueChanged.connect(self.toggle_assisted_analysis_ready)

        if hasattr(self.interface, "generate_likelihood_plots_button"):
            self.interface.generate_likelihood_plots_button.clicked.connect(self.on_generate_likelihood_plots_clicked)
            self.interface.generate_likelihood_plots_button.setEnabled(False)

    def on_select_dlc_config_clicked(self) -> None:
        path = select_file(self.interface, "Select config.yaml", "YAML files (*.yaml)")
        if path:
            self.interface.config_path_lineedit.setText(path)

    def on_select_refining_dlc_config_clicked(self) -> None:
        path = select_file(self.interface, "Select config.yaml", "YAML files (*.yaml)")
        if path:
            self.interface.config_path_data_process_path.setText(path)

    def on_select_refining_video_folder_clicked(self) -> None:
        path = select_folder(self.interface, "Select video folder")
        if path:
            self.interface.video_folder_data_process.setText(path)

    def on_select_video_folder_clicked(self) -> None:
        path = select_folder(self.interface, "Select video folder")
        if path:
            self.interface.video_folder_lineedit.setText(path)

    def on_select_video_list_clicked(self) -> None:
        path = select_file(self.interface, "Select video list file", "Text files (*.txt)")
        if path:
            self.interface.analyze_from_file_lineedit.setText(path)

    def enable_analysis(self) -> None:
        enabled = self.interface.config_path_lineedit.text().lower().endswith(".yaml")
        self.interface.dlc_video_analyze_button.setEnabled(enabled)
        self.interface.create_annotated_video_button.setEnabled(enabled)
        self.interface.folder_to_create_annotated_video_button.setEnabled(enabled)
        self._toggle_likelihood_button()

    def enable_analysis_buttons(self) -> None:
        enabled = bool(self.interface.video_folder_lineedit.text().strip())
        self.interface.get_frames_button.setEnabled(enabled)
        self.interface.clear_unused_files_button.setEnabled(enabled)
        self._toggle_likelihood_button()

    def _toggle_likelihood_button(self) -> None:
        if not hasattr(self.interface, "generate_likelihood_plots_button"):
            return

        config_ok = self.interface.config_path_lineedit.text().lower().endswith(".yaml")
        folder_path = self.interface.video_folder_lineedit.text().strip()
        folder_ok = bool(folder_path) and os.path.isdir(folder_path)

        has_data = False
        if folder_ok:
            has_data = any(f.endswith(".h5") for f in os.listdir(folder_path))

        self.interface.generate_likelihood_plots_button.setEnabled(config_ok and folder_ok and has_data)

    def on_generate_likelihood_plots_clicked(self) -> None:
        config_path = self.interface.config_path_lineedit.text().strip()
        folder_path = self.interface.video_folder_lineedit.text().strip()

        from behavython.pipeline.models import DLCLikelihoodPlotRequest
        from behavython.pipeline.plugins.dlc import run_generate_likelihood_plots

        request = DLCLikelihoodPlotRequest(config_path=config_path, folder_path=folder_path)

        self.logs.clear("dlc")
        self.progress_bar.setValue(0)
        self.runner.submit(run_generate_likelihood_plots, request, debug_mode=self.debug_mode)
        self.logger.info("Likelihood plot generation submitted for folder: %s", folder_path)

    def toggle_analyze_from_file_button(self) -> None:
        file_path = self.interface.analyze_from_file_lineedit.text().strip()
        enabled = bool(file_path) and os.path.exists(file_path) and file_path.lower().endswith(".txt")

        self.interface.analyze_from_file_button.setEnabled(enabled)
        self.interface.analyze_from_file_get_frames_button.setEnabled(enabled)

        if not file_path:
            self._set_gui_log_message("resume", "Please select a file to analyze.")
        elif not os.path.exists(file_path):
            self._set_gui_log_message("resume", "The selected file does not exist.")
        elif not file_path.lower().endswith(".txt"):
            self._set_gui_log_message("resume", "The selected file is not a .txt file.")
        else:
            self._set_gui_log_message("resume", "")

    def on_clear_unused_files_clicked(self) -> None:
        resolved_folder = self._build_output_folder(self.interface.video_folder_lineedit.text())

        if not resolved_folder.has_path:
            message_parts = ["Please select a video folder first."]
            if resolved_folder.warnings:
                message_parts.append("")
                message_parts.extend(resolved_folder.warnings)

            show_warning(self.interface, "Clear unused files", "\n".join(message_parts))
            return

        task_type = self.interface.type_combobox.currentText().lower().strip().replace(" ", "_")

        request = DLCClearUnusedFilesRequest(
            folder_path=resolved_folder.path,
            task_type=task_type,
        )

        self.logs.clear("dlc")
        self.progress_bar.setValue(0)
        self.runner.submit(run_clear_unused_files, request, debug_mode=self.debug_mode)
        self.logger.info("Clear-unused-files submitted for folder: %s", request.folder_path)

    def on_select_annotated_output_clicked(self) -> None:
        path = select_save_folder(self.interface, "Select output folder for annotated videos")
        if path:
            self.interface.folder_to_create_annotated_video_lineedit.setText(path)

    def on_create_annotated_video_clicked(self) -> None:
        annotated_folder_path = self.interface.folder_to_create_annotated_video_lineedit.text().strip()
        resolved_video_input = self._build_video_input(folder_path=annotated_folder_path)
        config_path = self.interface.config_path_lineedit.text().strip()
        input_path = annotated_folder_path
        output_path = input_path

        errors = validate_config_path(config_path) + validate_video_paths(resolved_video_input.paths)
        if not input_path:
            errors.append("Please select an output folder for the annotated video.")

        if errors:
            show_warning(self.interface, "Annotated Video Validation", "\n".join(errors))
            return

        request = DLCAnnotatedVideoRequest(config_path=config_path, video_paths=resolved_video_input.paths, output_path=output_path)

        self.logs.clear("dlc")
        self.progress_bar.setValue(0)
        self.runner.submit(run_create_annotated_video, request, debug_mode=self.debug_mode)

    def _build_video_input(self, folder_path: str | None = None, txt_path: str | None = None) -> ResolvedVideoInput:
        """
        Resolves video inputs from provided paths or falls back to UI line-edits.
        """
        # If no path is provided, fall back to the interface values
        effective_folder = folder_path if folder_path is not None else self.interface.video_folder_lineedit.text()
        effective_txt = txt_path if txt_path is not None else self.interface.analyze_from_file_lineedit.text()

        source = VideoInputSource(
            folder_path=effective_folder,
            txt_path=effective_txt,
        )

        return resolve_video_input(source)

    def _build_analysis_input(self, selected_files: list[str]) -> ResolvedAnalysisInput:
        source = AnalysisInputSource(selected_files=selected_files)
        return resolve_analysis_input(source)

    def _build_output_folder(self, selected_folder: str) -> ResolvedOutputFolder:
        source = OutputFolderSource(selected_folder=selected_folder)
        return resolve_output_folder(source)

    def _confirm_video_list(self, resolved_video_input: ResolvedVideoInput) -> bool:
        if not resolved_video_input.has_paths:
            message_parts = ["No videos could be resolved from the selected folder or txt file."]
            if resolved_video_input.warnings:
                message_parts.append("")
                message_parts.extend(resolved_video_input.warnings)

            show_warning(self.interface, "Video selection", "\n".join(message_parts))
            return False

        text = "The following videos will be processed:\n\n" + "\n".join(resolved_video_input.paths)
        return ask_yes_no(self.interface, "Confirm video list", text)

    def on_run_dlc_analysis_clicked(self) -> None:
        self._run_dlc_analysis(self._build_video_input())

    def on_run_dlc_analysis_from_file_clicked(self) -> None:
        self._run_dlc_analysis(self._build_video_input())

    def _run_dlc_analysis(self, resolved_video_input: ResolvedVideoInput) -> None:
        config_path = self.interface.config_path_lineedit.text().strip()
        errors = validate_config_path(config_path) + validate_video_paths(resolved_video_input.paths)

        if errors:
            self.logger.info("DLC validation failed: %s", "; ".join(errors))
            show_warning(self.interface, "DLC validation", "\n".join(errors))
            return

        if not self._confirm_video_list(resolved_video_input):
            self.logs.clear("dlc")
            self.logs.append("dlc", "Analysis cancelled.")
            self.logger.info("DLC analysis cancelled by user before submission.")
            return

        request = DLCVideoAnalysisRequest(
            config_path=config_path,
            video_paths=resolved_video_input.paths,
        )
        self.logs.clear("dlc")
        self.progress_bar.setValue(0)
        self.runner.submit(run_dlc_video_analysis, request, debug_mode=self.debug_mode)
        self.logger.info(
            "DLC video analysis submitted for %d video(s). Config: %s",
            len(request.video_paths),
            request.config_path,
        )

    def on_get_frames_clicked(self) -> None:
        self._run_get_frames(self._build_video_input())

    def on_get_frames_from_file_clicked(self) -> None:
        self._run_get_frames(self._build_video_input())

    def _run_get_frames(self, resolved_video_input: ResolvedVideoInput) -> None:
        errors = validate_video_paths(resolved_video_input.paths)
        if errors:
            self.logger.info("Frame extraction validation failed: %s", "; ".join(errors))
            show_warning(self.interface, "Frame extraction validation", "\n".join(errors))
            return

        request = DLCFrameExtractionRequest(video_paths=resolved_video_input.paths)
        self.logs.clear("dlc")
        self.progress_bar.setValue(0)
        self.runner.submit(run_extract_frames, request, debug_mode=self.debug_mode)
        self.logger.info("Frame extraction submitted for %d video(s).", len(request.video_paths))

    def toggle_assisted_analysis_ready(self) -> None:
        analysis_button = self.interface.start_assisted_labeling_data_process
        config_path = self.interface.config_path_data_process_path.text().strip()
        frames_folder = self.interface.video_folder_data_process.text().strip()

        is_config_valid = bool(config_path) and os.path.exists(config_path)
        is_ready = is_config_valid and bool(frames_folder) and os.path.exists(frames_folder)
        analysis_button.setEnabled(is_ready)

        if hasattr(self.interface, "bodyparts_listview_data_process"):
            self.interface.bodyparts_listview_data_process.setVisible(is_config_valid)

        if is_config_valid:
            self._populate_bodyparts_list(config_path)
        elif hasattr(self.interface, "bodyparts_listview_data_process"):
            self.interface.bodyparts_listview_data_process.clear()
            self._current_loaded_bodyparts_config = None

    def _populate_bodyparts_list(self, config_path: str) -> None:
        if not hasattr(self.interface, "bodyparts_listview_data_process"):
            return

        # Avoid reloading if it's the same config
        current_loaded = getattr(self, "_current_loaded_bodyparts_config", None)
        if current_loaded == config_path:
            return

        list_widget = self.interface.bodyparts_listview_data_process
        list_widget.clear()

        # Eager cache update to prevent redundant thread spawns
        self._current_loaded_bodyparts_config = config_path

        self._config_thread = ConfigLoaderThread(config_path, self)
        self._config_thread.config_loaded.connect(self._on_bodyparts_loaded)
        self._config_thread.error_occurred.connect(lambda err: self.logger.error(f"Failed to load bodyparts from {config_path}: {err}"))
        self._config_thread.start()

    def _on_bodyparts_loaded(self, bodyparts: list) -> None:
        if not hasattr(self.interface, "bodyparts_listview_data_process"):
            return

        list_widget = self.interface.bodyparts_listview_data_process
        list_widget.clear()

        for bp in bodyparts:
            item = QListWidgetItem(bp)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            list_widget.addItem(item)

    def on_run_analyze_frames_clicked(self) -> None:
        config_path = self.interface.config_path_data_process_path.text().strip()
        frames_folder = self.interface.video_folder_data_process.text().strip()
        num_frames = int(self.interface.extract_frames_data_process_spinbox.value() or 0)

        if not config_path or not frames_folder:
            show_warning(self.interface, "Missing Information", "Please select both a config.yaml and a video folder.")
            return

        folder_path = Path(frames_folder)
        if not folder_path.exists():
            show_warning(self.interface, "Invalid Folder", "The selected folder does not exist.")
            return

        video_extensions = ANALYSIS_REQUIRED_SUFFIXES["video"]
        image_extensions = ANALYSIS_REQUIRED_SUFFIXES["image"]
        valid_extensions = video_extensions + image_extensions

        # 1. Check for unsupported file types
        invalid_items = [item.name for item in folder_path.iterdir() if item.is_file() and item.suffix.lower() not in valid_extensions]

        if invalid_items:
            show_warning(
                self.interface,
                "Unsupported Files",
                f"The folder contains unsupported file types: "
                f"{', '.join(invalid_items[:5])}"
                f"{'...' if len(invalid_items) > 5 else ''}\n\n"
                "Please select a folder containing only video or image files.",
            )
            return

        # 2. Check for mixed content (videos + images)
        has_videos = any(item.is_file() and item.suffix.lower() in video_extensions for item in folder_path.iterdir())
        has_images = any(item.is_file() and item.suffix.lower() in image_extensions for item in folder_path.iterdir())

        if has_videos and has_images:
            show_warning(
                self.interface,
                "Mixed Content Detected",
                "The selected folder contains both video files and image files.\n\n"
                "Please select a folder containing ONLY videos (to extract frames) or ONLY images (to process directly).",
            )
            return

        mode = "image" if has_images else "video"

        target_bodyparts = None
        if hasattr(self.interface, "bodyparts_listview_data_process"):
            list_widget = self.interface.bodyparts_listview_data_process
            if list_widget.count() > 0:
                target_bodyparts = []
                for i in range(list_widget.count()):
                    item = list_widget.item(i)
                    if item.checkState() == Qt.CheckState.Checked:
                        target_bodyparts.append(item.text())

        request = DLCAnalyzeFramesRequest(
            config_path=config_path,
            frames_folder=frames_folder,
            frame_extension=self.interface.file_extension_confirmation_combobox_data_process.currentText().strip().lower(),
            number_of_frames=num_frames,
            mode=mode,
            target_bodyparts=target_bodyparts,
        )

        self.logs.clear("dlc")
        self.progress_bar.setValue(0)
        self.runner.submit(run_analyze_frames, request, debug_mode=self.debug_mode)
        self.logger.info("Assisted Labeling (analyze frames) submitted. Config: %s, Folder: %s", config_path, frames_folder)

    def toggle_debug_mode(self) -> None:
        self.debug_mode = not self.debug_mode
        self.set_advanced_tabs_visible(self.debug_mode)
        enabled = self.debug_mode

        self.logger.info("Debug mode toggled: %s", enabled)

        if enabled:
            self.interface.setStyleSheet(DEBUG_STYLE)
            self.logs.clear_all()
            self.logs.append("resume", "🛠️ Debug mode active")
            self.logs.append("dlc", "🛠️ Debug mode active")
        else:
            self.interface.setStyleSheet(DEFAULT_STYLE)
            self.logs.clear_all()
            self.logs.append("resume", "🛡️ Debug mode deactivated")
            self.logs.append("dlc", "🛡️ Debug mode deactivated")

    def on_use_default_network_toggled(self, checked: bool) -> None:
        if checked:
            if not self._ensure_model_installed("c57_network_2025_minified"):
                self.interface.use_default_network_button.blockSignals(True)
                self.interface.use_default_network_button.setChecked(False)
                self.interface.use_default_network_button.blockSignals(False)
                return

            model_path = get_model_path("c57_network_2025_minified")
            config_yaml_path = model_path / "config.yaml"

            self.interface.config_path_lineedit.setText(str(config_yaml_path))
            self.interface.get_config_path_button.setEnabled(False)

            self._set_gui_log_message("dlc", "Default C57 network loaded and ready.")
            self.logger.info("User selected default network: %s", config_yaml_path)

        else:
            # User unchecked the box. Revert to manual mode.
            self.interface.config_path_lineedit.clear()
            self.interface.get_config_path_button.setEnabled(True)
            self._set_gui_log_message("dlc", "")
            self.logger.info("User deselected default network.")

    # ------------------------------------------------------------------
    # Video Editing tab
    # ------------------------------------------------------------------

    def _connect_video_editing_tab(self) -> None:
        if hasattr(self.interface, "get_videos_path_video_editing_button"):
            self.interface.get_videos_path_video_editing_button.clicked.connect(self.on_select_video_editing_folder_clicked)
            self.interface.call_crop_dialog_button.clicked.connect(self.on_call_crop_dialog_clicked)
            self.interface.crop_videos_button.clicked.connect(self.on_execute_crop_videos_clicked)
            self.interface.videos_to_crop_folder_video_editing_lineedit.textChanged.connect(self.toggle_video_editing_buttons)
            if hasattr(self.interface, "standardize_videos_button_video_editing_button"):
                self.interface.standardize_videos_button_video_editing_button.clicked.connect(self.on_standardize_videos_clicked)
                self.interface.standardize_videos_button_video_editing_button.hide()

            if hasattr(self.interface, "codec_warning_label_video_editing_label"):
                self.interface.codec_warning_label_video_editing_label.hide()

            self.probe_thread = None
            self.problematic_videos = []

    def on_select_video_editing_folder_clicked(self) -> None:
        path = select_folder(self.interface, "Select video folder to crop")
        if path:
            self.interface.videos_to_crop_folder_video_editing_lineedit.setText(path)

    def toggle_video_editing_buttons(self) -> None:
        folder_path = self.interface.videos_to_crop_folder_video_editing_lineedit.text().strip()
        is_ready = bool(folder_path) and os.path.exists(folder_path)
        self.interface.call_crop_dialog_button.setEnabled(is_ready)
        self.interface.crop_videos_button.setEnabled(is_ready)

        if hasattr(self.interface, "codec_warning_label_video_editing_label"):
            self.interface.codec_warning_label_video_editing_label.hide()
        if hasattr(self.interface, "standardize_videos_button_video_editing_button"):
            self.interface.standardize_videos_button_video_editing_button.hide()

        if is_ready:
            self.probe_thread = CodecProbeThread(folder_path, self.interface)
            self.probe_thread.warning_detected.connect(self.on_codec_warning_detected)
            self.probe_thread.start()

    def on_codec_warning_detected(self, problematic_videos):
        self.problematic_videos = problematic_videos
        if hasattr(self.interface, "codec_warning_label_video_editing_label"):
            self.interface.codec_warning_label_video_editing_label.show()
        if hasattr(self.interface, "standardize_videos_button_video_editing_button"):
            self.interface.standardize_videos_button_video_editing_button.show()
        self._set_gui_log_message("dlc", f"WARNING: {len(problematic_videos)} legacy videos detected. Conversion highly recommended.")

    def on_standardize_videos_clicked(self) -> None:
        if not self.problematic_videos:
            return

        from behavython.services.video_service import run_standardize_videos

        request = {"videos": self.problematic_videos}

        self.logs.clear("dlc")
        self.progress_bar.setValue(0)

        # After triggering standardizing, hide the warning
        if hasattr(self.interface, "codec_warning_label_video_editing_label"):
            self.interface.codec_warning_label_video_editing_label.hide()
        if hasattr(self.interface, "standardize_videos_button_video_editing_button"):
            self.interface.standardize_videos_button_video_editing_button.hide()
        self.problematic_videos = []

        self.runner.submit(run_standardize_videos, request, debug_mode=self.debug_mode)

    def on_call_crop_dialog_clicked(self) -> None:
        folder_path = Path(self.interface.videos_to_crop_folder_video_editing_lineedit.text().strip())
        if not folder_path.exists():
            return

        video_extensions = ANALYSIS_REQUIRED_SUFFIXES["video"]
        videos = os_sorted([str(item.resolve()) for item in folder_path.iterdir() if item.is_file() and item.suffix.lower() in video_extensions])

        if not videos:
            show_warning(self.interface, "No Videos", "No video files found in the selected folder.")
            return

        project_path = folder_path / "crop_project.json"
        project_data = None
        if project_path.exists():
            import json

            try:
                with open(project_path, "r") as f:
                    project_data = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load crop project: {e}")

        dialog = VideoCropperDialog(parent=self.interface, video_list=videos, project_data=project_data, project_path=project_path)
        dialog.exec()

    def on_execute_crop_videos_clicked(self) -> None:
        folder_path = Path(self.interface.videos_to_crop_folder_video_editing_lineedit.text().strip())
        project_path = folder_path / "crop_project.json"

        if not project_path.exists():
            show_warning(self.interface, "No Crop Project", "You must set the crop coordinates first using the Cropper Dialog.")
            return

        import json

        try:
            with open(project_path, "r") as f:
                project_data = json.load(f)
        except Exception as e:
            show_warning(self.interface, "Error", f"Could not load project file: {e}")
            return

        from behavython.services.video_service import run_batch_crop

        request = {"project_data": project_data, "project_path": str(project_path)}

        self.logs.clear("dlc")
        self.progress_bar.setValue(0)
        self.runner.submit(run_batch_crop, request, debug_mode=self.debug_mode)

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def on_worker_log(self, target: str, message: str) -> None:
        self.logs.append(target, message)

    def on_worker_warning(self, title: str, text: str) -> None:
        show_warning(self.interface, title, text)

    def on_worker_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    def on_worker_result(self, result) -> None:
        kind = result.get("kind")

        if kind == "analysis":
            self.logger.info("Analysis completed. Output path: %s", result["output_path"])
            show_info(
                self.interface,
                "Analysis completed",
                f"Processed {result['rows']} animal(s). Valid: {result['valid_animals']} | Invalid: {result['invalid_animals']}",
            )
        elif kind == "dlc_analysis":
            extra = ""
            if result.get("config_was_repaired"):
                extra = f"\n\nA repaired config was used:\n{result['config_path_used']}"
            self.logger.info(
                "DLC analysis completed for %d video(s). repaired_config=%s",
                result["videos"],
                result.get("config_was_repaired", False),
            )
            show_info(self.interface, "DLC analysis completed", f"Processed {result['videos']} video(s).{extra}")
        elif kind == "dlc_skeleton":
            extra = ""
            if result.get("config_was_repaired"):
                extra = f"\n\nA repaired config was used:\n{result['config_path_used']}"
            self.logger.info(
                "Skeleton extraction completed for %d video(s). repaired_config=%s",
                result["videos"],
                result.get("config_was_repaired", False),
            )
            show_info(self.interface, "Skeleton extraction completed", f"Processed {result['videos']} video(s).{extra}")
        elif kind == "dlc_frames":
            self.logger.info("Frame extraction completed for %d video(s).", result["videos"])
            show_info(self.interface, "Frame extraction completed", f"Processed {result['videos']} video(s).")
        elif kind == "dlc_clear_unused_files":
            moved_count = len(result["moved_files"])
            self.logger.info(
                "Cleanup completed. moved_files=%d all_required_present=%s",
                moved_count,
                result["all_required_present"],
            )

            if result["all_required_present"]:
                show_info(
                    self.interface,
                    "Cleanup completed",
                    f"Moved {moved_count} file(s) to unwanted_files.\nAll required files are present.",
                )
            else:
                missing_text = "\n".join(result["missing_files"])
                show_warning(
                    self.interface,
                    "Missing files",
                    "The following files are missing:\n\n"
                    f"{missing_text}\n\n"
                    "These files are essential for the analysis to work.\n"
                    "Check the analysis folder before continuing.",
                )
        elif kind == "batch_crop":
            show_info(self.interface, "Cropping Complete", f"Successfully cropped {result['processed']} out of {result['total']} video(s).")

    def on_worker_error(self, error_info) -> None:
        self.logger.error("Worker error dialog shown: %s", error_info[1])
        show_worker_error(self.interface, error_info)

    def on_worker_finished(self) -> None:
        pass
