from __future__ import annotations

import os
import logging
from PySide6.QtWidgets import QWidget
from src.behavython.pipeline.models import analysis_options, analysis_request
from src.behavython.pipeline.workflow import run_analysis_workflow
from src.behavython.services.validation import validate_analysis_request
from src.behavython.core.app_context import AppContext
from src.behavython.core.defaults import DEBUG_STYLE, DEFAULT_STYLE
from src.behavython.pipeline.models import (
    DLCClearUnusedFilesRequest,
    DLCFrameExtractionRequest,
    DLCSkeletonExtractionRequest,
    DLCVideoAnalysisRequest,
)
from src.behavython.services.validation import validate_config_path, validate_video_paths
from src.behavython.pipeline.plugins.dlc import (
    check_dlc_folder_structure,
    run_clear_unused_files,
    run_dlc_video_analysis,
    run_extract_frames,
    run_extract_skeleton,
)
from src.behavython.services.validation import validate_json_config
from src.behavython.gui.dialogs import ask_yes_no, show_info, show_warning, show_worker_error
from src.behavython.gui.dialogs import select_file, select_files, select_folder, select_save_folder
from src.behavython.services.logging import LoggingService
from src.behavython.core.utils import resolve_analysis_input, resolve_output_folder, resolve_video_input
from src.behavython.pipeline.models import (
    AnalysisInputSource,
    OutputFolderSource,
    ResolvedAnalysisInput,
    ResolvedOutputFolder,
    ResolvedVideoInput,
    VideoInputSource,
)
from src.behavython.tasks.task_runner import TaskRunner


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

        if hasattr(self.interface, "enable_debugging_mode_checkbox"):
            self.interface.enable_debugging_mode_checkbox.stateChanged.connect(self.toggle_debug_mode)

        self._connect_analysis_tab()
        self._connect_dlc_tab()
        self._initialize_ui_state()

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

    def _set_gui_log_message(self, target: str, message: str) -> None:
        self.logs.clear(target)
        if message:
            self.logs.append(target, message)

    # ------------------------------------------------------------------
    # Analysis tab
    # ------------------------------------------------------------------

    def _connect_analysis_tab(self) -> None:
        self.interface.analysis_button.clicked.connect(self.on_run_analysis_clicked)
        self.interface.clear_button.clicked.connect(self.clear_analysis_tab)
        self.interface.load_configuration_button.clicked.connect(self.on_load_configuration_clicked)

    def clear_analysis_tab(self) -> None:
        self.interface.type_combobox.setCurrentIndex(0)
        self.interface.algo_type_combobox.setCurrentIndex(0)
        self.interface.arena_width_lineedit.setText("30")
        self.interface.arena_height_lineedit.setText("30")
        self.interface.frames_per_second_lineedit.setText("30")
        self.interface.animal_combobox.setCurrentIndex(0)
        self.interface.task_duration_lineedit.setText("300")
        self.interface.crop_video_time_lineedit.setText("0")
        self.interface.fig_max_size.setCurrentIndex(1)
        self.interface.crop_video_checkbox.setChecked(False)
        self.interface.plot_data_checkbox.setChecked(True)
        self.logs.clear("resume")

    def on_load_configuration_clicked(self) -> None:
        path = select_file(self.interface, "Select configuration JSON", "JSON files (*.json)")
        if not path:
            return

        if not validate_json_config(path):
            self.logger.info("Invalid configuration file selected: %s", path)

            show_warning(self.interface, "Configuration file", "The selected file is not a valid JSON configuration.")
            return

        data = validate_json_config(path)
        self.apply_analysis_configuration(data)
        self._set_gui_log_message("resume", "Configuration file loaded successfully!")
        self.logger.info("Configuration file loaded: %s", path)

    def apply_analysis_configuration(self, data: dict) -> None:
        mapping = {
            "Arena width": self.interface.arena_width_lineedit,
            "Arena height": self.interface.arena_height_lineedit,
            "Video framerate": self.interface.frames_per_second_lineedit,
            "Task Duration": self.interface.task_duration_lineedit,
            "Amount to trim": self.interface.crop_video_time_lineedit,
        }

        for key, widget in mapping.items():
            if key in data:
                widget.setText(str(data[key]))

    def build_analysis_options(self) -> analysis_options:
        threshold = 0.0267 if self.interface.animal_combobox.currentIndex() == 0 else 0.0667

        return analysis_options(
            arena_width=int(self.interface.arena_width_lineedit.text()),
            arena_height=int(self.interface.arena_height_lineedit.text()),
            frames_per_second=int(self.interface.frames_per_second_lineedit.text()),
            experiment_type=self.interface.type_combobox.currentText().lower().strip().replace(" ", "_"),
            max_fig_res=str(self.interface.fig_max_size.currentText()).replace(" ", "").replace("x", ",").split(","),
            algo_type=self.interface.algo_type_combobox.currentText().lower().strip(),
            threshold=threshold,
            task_duration=int(self.interface.task_duration_lineedit.text()),
            trim_amount=int(self.interface.crop_video_time_lineedit.text()),
            crop_video=self.interface.crop_video_checkbox.isChecked(),
            plot_options="plotting_enabled" if self.interface.plot_data_checkbox.isChecked() else "plotting_disabled",
        )

    def on_run_analysis_clicked(self) -> None:
        selected_files = select_files(
            self.interface,
            "Select analysis files",
            "DLC files (*.csv *.jpg *.jpeg *.png)",
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

        request = analysis_request(
            input_files=resolved_input.paths,
            output_folder=resolved_output_folder.path,
            options=self.build_analysis_options(),
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

    # ------------------------------------------------------------------
    # DeepLabCut tab
    # ------------------------------------------------------------------

    def _connect_dlc_tab(self) -> None:
        self.interface.get_config_path_button.clicked.connect(self.on_select_dlc_config_clicked)
        self.interface.get_videos_path_button.clicked.connect(self.on_select_video_folder_clicked)
        self.interface.get_file_to_analyze_button.clicked.connect(self.on_select_video_list_clicked)

        self.interface.config_path_lineedit.textChanged.connect(self.enable_analysis)
        self.interface.video_folder_lineedit.textChanged.connect(self.enable_analysis_buttons)
        self.interface.analyze_from_file_lineedit.textChanged.connect(self.toggle_analyze_from_file_button)

        self.interface.folder_structure_check_button.clicked.connect(self.on_check_folder_structure_clicked)
        self.interface.dlc_video_analyze_button.clicked.connect(self.on_run_dlc_analysis_clicked)
        self.interface.analyze_from_file_button.clicked.connect(self.on_run_dlc_analysis_from_file_clicked)
        self.interface.extract_skeleton_button.clicked.connect(self.on_extract_skeleton_clicked)
        self.interface.clear_unused_files_button.clicked.connect(self.on_clear_unused_files_clicked)
        self.interface.analyze_from_file_extract_skeleton_button.clicked.connect(self.on_extract_skeleton_from_file_clicked)
        self.interface.get_frames_button.clicked.connect(self.on_get_frames_clicked)
        self.interface.analyze_from_file_get_frames_button.clicked.connect(self.on_get_frames_from_file_clicked)

    def on_select_dlc_config_clicked(self) -> None:
        path = select_file(self.interface, "Select config.yaml", "YAML files (*.yaml)")
        if path:
            self.interface.config_path_lineedit.setText(path)

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
        self.interface.folder_structure_check_button.setEnabled(enabled)

    def enable_analysis_buttons(self) -> None:
        enabled = bool(self.interface.video_folder_lineedit.text().strip())
        self.interface.extract_skeleton_button.setEnabled(enabled)
        self.interface.get_frames_button.setEnabled(enabled)
        self.interface.clear_unused_files_button.setEnabled(enabled)

    def toggle_analyze_from_file_button(self) -> None:
        file_path = self.interface.analyze_from_file_lineedit.text().strip()
        enabled = bool(file_path) and os.path.exists(file_path) and file_path.lower().endswith(".txt")

        self.interface.analyze_from_file_button.setEnabled(enabled)
        self.interface.analyze_from_file_extract_skeleton_button.setEnabled(enabled)
        self.interface.analyze_from_file_get_frames_button.setEnabled(enabled)

        if not file_path:
            self._set_gui_log_message("resume", "Please select a file to analyze.")
        elif not os.path.exists(file_path):
            self._set_gui_log_message("resume", "The selected file does not exist.")
        elif not file_path.lower().endswith(".txt"):
            self._set_gui_log_message("resume", "The selected file is not a .txt file.")
        else:
            self._set_gui_log_message("resume", "")

    def on_check_folder_structure_clicked(self) -> None:
        config_path = self.interface.config_path_lineedit.text().strip()
        ok, messages = check_dlc_folder_structure(config_path)

        self.logs.clear("dlc")
        for message in messages:
            self.logs.append("dlc", message)

        if not ok:
            self.logger.info("DLC folder structure check failed: %s", " | ".join(messages))
            show_warning(self.interface, "Folder structure", "\n".join(messages))
        else:
            self.logger.info("DLC folder structure check passed for config: %s", config_path)

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

    def _build_video_input(self) -> ResolvedVideoInput:
        source = VideoInputSource(
            folder_path=self.interface.video_folder_lineedit.text(),
            txt_path=self.interface.analyze_from_file_lineedit.text(),
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

    def on_extract_skeleton_clicked(self) -> None:
        self._run_extract_skeleton(self._build_video_input())

    def on_extract_skeleton_from_file_clicked(self) -> None:
        self._run_extract_skeleton(self._build_video_input())

    def _run_extract_skeleton(self, resolved_video_input: ResolvedVideoInput) -> None:
        config_path = self.interface.config_path_lineedit.text().strip()
        errors = validate_config_path(config_path) + validate_video_paths(resolved_video_input.paths)

        if errors:
            self.logger.info("Skeleton extraction validation failed: %s", "; ".join(errors))
            show_warning(self.interface, "Skeleton validation", "\n".join(errors))
            return

        request = DLCSkeletonExtractionRequest(
            config_path=config_path,
            video_paths=resolved_video_input.paths,
        )
        self.logs.clear("dlc")
        self.progress_bar.setValue(0)
        self.runner.submit(run_extract_skeleton, request, debug_mode=self.debug_mode)
        self.logger.info(
            "Skeleton extraction submitted for %d video(s). Config: %s",
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

    def toggle_debug_mode(self) -> None:
        if not hasattr(self.interface, "enable_debugging_mode_checkbox"):
            return

        enabled = self.interface.enable_debugging_mode_checkbox.isChecked()
        self.debug_mode = enabled
        self.logger.info("Debug mode toggled: %s", enabled)

        if enabled:
            self.interface.setStyleSheet(DEBUG_STYLE)
            self.logs.clear_all()
            self.logs.append("resume", "🐞 Debug mode active")
            self.logs.append("dlc", "🐞 Debug mode active")
        else:
            self.interface.setStyleSheet(DEFAULT_STYLE)
            self.logs.clear_all()
            self.logs.append("resume", "🔧 Debug mode deactivated")
            self.logs.append("dlc", "🔧 Debug mode deactivated")

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
            moved_count = len(result["movedFiles"])
            self.logger.info(
                "Cleanup completed. moved_files=%d all_required_present=%s",
                moved_count,
                result["allRequiredPresent"],
            )

            if result["allRequiredPresent"]:
                show_info(
                    self.interface,
                    "Cleanup completed",
                    f"Moved {moved_count} file(s) to unwanted_files.\nAll required files are present.",
                )
            else:
                missing_text = "\n".join(result["missingFiles"])
                show_warning(
                    self.interface,
                    "Missing files",
                    "The following files are missing:\n\n"
                    f"{missing_text}\n\n"
                    "These files are essential for the analysis to work.\n"
                    "Check the analysis folder before continuing.",
                )

    def on_worker_error(self, error_info) -> None:
        self.logger.error("Worker error dialog shown: %s", error_info[1])
        show_worker_error(self.interface, error_info)

    def on_worker_finished(self) -> None:
        pass
