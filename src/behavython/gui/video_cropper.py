from __future__ import annotations

from PySide6.QtWidgets import QGraphicsScene, QGraphicsRectItem, QGraphicsItem, QListWidgetItem, QGraphicsPixmapItem
from PySide6.QtGui import QPen, QColor, QBrush, QPainter, QPixmap
from PySide6.QtCore import Qt, QRectF, QPointF, QObject, QEvent, QTimer
from PySide6 import QtUiTools

import logging
import tempfile
from pathlib import Path
from behavython.services.video_service import VideoService

from behavython.core.paths import GUI_UI_ROOT

logger = logging.getLogger("behavython.console")


class ViewportEventFilter(QObject):
    def __init__(self, view, controller, parent=None):
        super().__init__(parent)
        self.view = view
        self.controller = controller

    def eventFilter(self, watched, event):
        if event.type() == QEvent.MouseMove:
            scene_pos = self.view.mapToScene(event.pos())
            self.controller.on_mouse_move(scene_pos)
        elif event.type() == QEvent.MouseButtonPress:
            # mapToScene handles the viewport-to-scene transformation correctly
            scene_pos = self.view.mapToScene(event.pos())
            if self.controller.on_mouse_press(scene_pos, event.button()):
                return True
        elif event.type() == QEvent.MouseButtonRelease:
            scene_pos = self.view.mapToScene(event.pos())
            if self.controller.on_mouse_release(scene_pos, event.button()):
                return True
        elif event.type() == QEvent.Wheel:
            scene_pos = self.view.mapToScene(event.position().toPoint())
            if hasattr(self.controller, "on_wheel") and self.controller.on_wheel(scene_pos, event.angleDelta().y()):
                return True
        return super().eventFilter(watched, event)


class ResizableRectItem(QGraphicsRectItem):
    """
    Logic/Math for the resizable rectangle, completely separated from the UI.
    """

    handleTopLeft = 1
    handleTopRight = 2
    handleBottomLeft = 3
    handleBottomRight = 4

    handleSize = +8.0

    handleCursors = {
        handleTopLeft: Qt.SizeFDiagCursor,
        handleTopRight: Qt.SizeBDiagCursor,
        handleBottomLeft: Qt.SizeBDiagCursor,
        handleBottomRight: Qt.SizeFDiagCursor,
    }

    def __init__(self, *args):
        super().__init__(*args)
        self.handles = {}
        self.handleSelected = None
        self.mousePressPos = None
        self.mousePressRect = None

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        self.setPen(QPen(QColor(255, 0, 0), 2))
        self.setBrush(QBrush(QColor(255, 0, 0, 50)))
        self.setCursor(Qt.SizeAllCursor)
        self.updateHandlesPos()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            new_pos = value
            rect = self.rect()
            scene_rect = self.scene().sceneRect()

            x = new_pos.x()
            y = new_pos.y()

            if x + rect.left() < scene_rect.left():
                x = scene_rect.left() - rect.left()
            elif x + rect.right() > scene_rect.right():
                x = scene_rect.right() - rect.right()

            if y + rect.top() < scene_rect.top():
                y = scene_rect.top() - rect.top()
            elif y + rect.bottom() > scene_rect.bottom():
                y = scene_rect.bottom() - rect.bottom()

            return QPointF(x, y)

        return super().itemChange(change, value)

    def handleAt(self, point):
        for (
            k,
            v,
        ) in self.handles.items():
            if v.contains(point):
                return k
        return None

    def hoverMoveEvent(self, moveEvent):
        if self.isSelected():
            handle = self.handleAt(moveEvent.pos())
            cursor = Qt.SizeAllCursor if handle is None else self.handleCursors[handle]
            self.setCursor(cursor)
        super().hoverMoveEvent(moveEvent)

    def hoverLeaveEvent(self, moveEvent):
        self.setCursor(Qt.SizeAllCursor)
        super().hoverLeaveEvent(moveEvent)

    def mousePressEvent(self, mouseEvent):
        self.handleSelected = self.handleAt(mouseEvent.pos())
        if self.handleSelected:
            self.mousePressPos = mouseEvent.scenePos()
            self.mousePressRect = self.rect()
            mouseEvent.accept()
        else:
            self.setCursor(Qt.ClosedHandCursor)
            super().mousePressEvent(mouseEvent)

    def mouseMoveEvent(self, mouseEvent):
        if self.handleSelected is not None:
            self.interactiveResize(mouseEvent.scenePos())
        else:
            super().mouseMoveEvent(mouseEvent)

    def mouseReleaseEvent(self, mouseEvent):
        super().mouseReleaseEvent(mouseEvent)
        self.handleSelected = None
        self.setCursor(Qt.SizeAllCursor)
        self.update()

    def interactiveResize(self, mouseScenePos):
        rect = QRectF(self.mousePressRect)
        diff = self.mapFromScene(mouseScenePos) - self.mapFromScene(self.mousePressPos)

        if self.handleSelected == self.handleTopLeft:
            rect.setTopLeft(self.mousePressRect.topLeft() + diff)
        elif self.handleSelected == self.handleTopRight:
            rect.setTopRight(self.mousePressRect.topRight() + diff)
        elif self.handleSelected == self.handleBottomLeft:
            rect.setBottomLeft(self.mousePressRect.bottomLeft() + diff)
        elif self.handleSelected == self.handleBottomRight:
            rect.setBottomRight(self.mousePressRect.bottomRight() + diff)

        if rect.width() < 10:
            if self.handleSelected in (self.handleTopLeft, self.handleBottomLeft):
                rect.setLeft(rect.right() - 10)
            else:
                rect.setRight(rect.left() + 10)

        if rect.height() < 10:
            if self.handleSelected in (self.handleTopLeft, self.handleTopRight):
                rect.setTop(rect.bottom() - 10)
            else:
                rect.setBottom(rect.top() + 10)

        # Clamp to scene bounds
        if self.scene():
            scene_bounds = self.mapRectFromScene(self.scene().sceneRect())
            rect = rect.intersected(scene_bounds)

        self.setRect(rect)
        self.updateHandlesPos()

    def updateHandlesPos(self):
        s = self.handleSize
        b = self.boundingRect()
        self.handles[self.handleTopLeft] = QRectF(b.left(), b.top(), s, s)
        self.handles[self.handleTopRight] = QRectF(b.right() - s, b.top(), s, s)
        self.handles[self.handleBottomLeft] = QRectF(b.left(), b.bottom() - s, s, s)
        self.handles[self.handleBottomRight] = QRectF(b.right() - s, b.bottom() - s, s, s)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        for handle, rect in self.handles.items():
            if self.isSelected():
                painter.drawRect(rect)


class VideoCropperDialog:
    """
    Controller for the batch video cropper dialog.
    """

    def __init__(self, parent=None, video_list=None, project_data=None, project_path=None):
        ui_path = GUI_UI_ROOT / "video_cropper.ui"
        loader = QtUiTools.QUiLoader()

        self.dialog = loader.load(str(ui_path), parent)

        self.project_path = project_path
        logger.info("Started crop video interface.")

        # Database to hold state
        # format: { "video_name.mp4": { "coordinates": dict, "coordinates_set": bool, "video_cropped": bool } }
        self.crop_database = {}
        if project_data:
            self.crop_database = project_data
        elif video_list:
            for vid in video_list:
                self.crop_database[vid] = {"coordinates": None, "coordinates_set": False, "video_cropped": False}

        # Initialize the list widget
        self.populate_list()

        # Connect UI
        self.dialog.video_list.currentRowChanged.connect(self.on_video_selected)
        self.dialog.btn_set_coords.clicked.connect(self.on_set_coords)
        self.dialog.btn_copy_checked.clicked.connect(self.on_copy_checked)
        self.dialog.btn_save_project.clicked.connect(self.on_save_project)

        self.scene = QGraphicsScene()
        self.dialog.graphicsView.setScene(self.scene)
        self.dialog.graphicsView.setRenderHint(QPainter.Antialiasing)

        # We start with a generic background, but eventually we will load the video frame here
        # For now, just a dummy size.
        video_width, video_height = 1280, 720
        self.scene.setSceneRect(0, 0, video_width, video_height)
        self.background_rect = self.scene.addRect(0, 0, video_width, video_height, QPen(Qt.black), QBrush(QColor(50, 50, 80)))
        self.background_rect.setZValue(0)

        self.crop_box = None
        self.is_drawing = False
        self.draw_start_pos = None
        self.current_video = None

        # Mouse tracking for coordinates and drawing
        self.dialog.graphicsView.viewport().setMouseTracking(True)
        self.event_filter = ViewportEventFilter(self.dialog.graphicsView, self)
        self.dialog.graphicsView.viewport().installEventFilter(self.event_filter)

        self.final_db = None

        # Automatically scale to fit the window after the UI lays out
        QTimer.singleShot(0, lambda: self.dialog.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio))

    def populate_list(self):
        from natsort import os_sorted

        self.dialog.video_list.clear()
        for vid in os_sorted(self.crop_database.keys()):
            data = self.crop_database[vid]
            item = QListWidgetItem(vid)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.update_item_color(item, data)
            self.dialog.video_list.addItem(item)

    def update_item_color(self, item, data):
        # Green - Ready to cut
        # Yellow - Needs attention
        # White - Finished
        if data["video_cropped"]:
            item.setBackground(QColor(255, 255, 255))  # White
            item.setForeground(QColor(0, 0, 0))  # Black text for readability
        elif data["coordinates_set"]:
            item.setBackground(QColor(44, 161, 83))  # Green
            item.setForeground(QColor(255, 255, 255))
        else:
            item.setBackground(QColor(200, 160, 40))  # Yellow
            item.setForeground(QColor(255, 255, 255))

    def on_video_selected(self, index):
        if index < 0:
            return
        item = self.dialog.video_list.item(index)
        self.current_video = item.text()

        # 1. Attempt to load the real video frame
        video_path = Path(self.current_video)
        temp_img = Path(tempfile.gettempdir()) / f"crop_preview_{video_path.name}.jpg"

        if video_path.exists():
            success, width, height = VideoService.extract_preview_frame(video_path, temp_img)
            if success:
                # Store dimensions in the database for later use in cropping
                self.crop_database[self.current_video]["orig_w"] = width
                self.crop_database[self.current_video]["orig_h"] = height

                pixmap = QPixmap(str(temp_img))
                if not pixmap.isNull():
                    # Remove old background if it exists
                    if hasattr(self, "video_pixmap_item") and self.video_pixmap_item:
                        self.scene.removeItem(self.video_pixmap_item)
                    elif self.background_rect:
                        self.scene.removeItem(self.background_rect)
                        self.background_rect = None

                    # Add new image background
                    self.video_pixmap_item = QGraphicsPixmapItem(pixmap)
                    self.video_pixmap_item.setZValue(0)
                    self.scene.addItem(self.video_pixmap_item)
                    self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

        # 2. Fit view to the new scene dimensions
        self.dialog.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        # 3. Restore the crop box if it exists
        if self.crop_box:
            self.scene.removeItem(self.crop_box)
            self.crop_box = None

        data = self.crop_database[self.current_video]
        if data["coordinates_set"] and data["coordinates"]:
            c = data["coordinates"]
            self.crop_box = ResizableRectItem(0, 0, c["width"], c["height"])
            self.crop_box.setPos(c["x"], c["y"])
            self.crop_box.setTransformOriginPoint(self.crop_box.rect().center())
            self.crop_box.setRotation(c["rotation"])
            self.crop_box.setZValue(1)
            self.scene.addItem(self.crop_box)
            self.crop_box.setSelected(True)

    def on_set_coords(self):
        if self.current_video and self.crop_box:
            # Save local position to exactly restore it
            c = {
                "x": int(self.crop_box.pos().x() + self.crop_box.rect().x()),
                "y": int(self.crop_box.pos().y() + self.crop_box.rect().y()),
                "width": int(self.crop_box.rect().width()),
                "height": int(self.crop_box.rect().height()),
                "rotation": self.crop_box.rotation(),
                "orig_w": self.crop_database[self.current_video].get("orig_w"),
                "orig_h": self.crop_database[self.current_video].get("orig_h"),
            }
            self.crop_database[self.current_video]["coordinates"] = c
            self.crop_database[self.current_video]["coordinates_set"] = True
            logger.info(f"Coordinates set for {self.current_video}.")

            # Update UI color
            current_item = self.dialog.video_list.currentItem()
            self.update_item_color(current_item, self.crop_database[self.current_video])

    def on_copy_checked(self):
        if self.current_video and self.crop_box:
            # We must set current first to make sure we have the latest
            self.on_set_coords()
            c = self.crop_database[self.current_video]["coordinates"]

            # Copy to checked items
            for i in range(self.dialog.video_list.count()):
                item = self.dialog.video_list.item(i)
                if item.checkState() == Qt.Checked:
                    vid = item.text()
                    # Don't overwrite finished videos
                    if not self.crop_database[vid]["video_cropped"]:
                        self.crop_database[vid]["coordinates"] = c.copy()
                        self.crop_database[vid]["coordinates_set"] = True
                        self.update_item_color(item, self.crop_database[vid])
                        logger.info(f"Propagated coordinates to {vid}.")

    def on_mouse_press(self, scene_pos: QPointF, button):
        if button == Qt.LeftButton:
            # Check what is under the mouse
            item = self.scene.itemAt(scene_pos, self.dialog.graphicsView.transform())

            # We allow starting a new drawing if we click on the background, the video frame, or nothing
            if item is None or item == self.background_rect or isinstance(item, QGraphicsPixmapItem):
                # Start drawing a new crop box
                if self.crop_box is not None:
                    self.scene.removeItem(self.crop_box)
                    self.crop_box = None

                self.is_drawing = True
                self.draw_start_pos = scene_pos
                self.crop_box = ResizableRectItem(scene_pos.x(), scene_pos.y(), 0, 0)
                self.crop_box.setZValue(1)
                self.scene.addItem(self.crop_box)
                self.crop_box.setSelected(True)
                return True  # We consumed the event
        return False

    def on_mouse_move(self, scene_pos: QPointF):
        x, y = int(scene_pos.x()), int(scene_pos.y())
        if hasattr(self.dialog, "mouse_position_lineedit"):
            # Clamp display to scene coordinates
            scene_rect = self.scene.sceneRect()
            x = max(int(scene_rect.left()), min(int(scene_rect.right()), x))
            y = max(int(scene_rect.top()), min(int(scene_rect.bottom()), y))
            self.dialog.mouse_position_lineedit.setText(f"X: {x}, Y: {y}")

        if self.is_drawing and self.crop_box is not None:
            rect = QRectF(self.draw_start_pos, scene_pos).normalized()

            # Clamp to scene bounds
            scene_bounds = self.scene.sceneRect()
            rect = rect.intersected(scene_bounds)

            self.crop_box.setRect(rect)
            self.crop_box.updateHandlesPos()

    def on_mouse_release(self, scene_pos: QPointF, button):
        if button == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            if self.crop_box is not None:
                rect = self.crop_box.rect()
                # If the user just clicked without dragging, ignore it
                if rect.width() < 10 or rect.height() < 10:
                    self.scene.removeItem(self.crop_box)
                    self.crop_box = None
            return True
        return False

    def on_wheel(self, scene_pos: QPointF, delta: int):
        from PySide6.QtWidgets import QApplication

        if self.crop_box is not None and self.crop_box.sceneBoundingRect().contains(scene_pos):
            step = 1.0
            if QApplication.keyboardModifiers() & Qt.ControlModifier:
                step = 5.0

            angle = step if delta > 0 else -step
            self.crop_box.setTransformOriginPoint(self.crop_box.rect().center())
            self.crop_box.setRotation(self.crop_box.rotation() + angle)
            return True
        return False

    def on_save_project(self):
        self.final_db = self.crop_database
        if self.project_path:
            import json

            try:
                with open(self.project_path, "w") as f:
                    json.dump(self.final_db, f, indent=4)
                logger.info(f"Crop coordinates saved to {self.project_path}")

                from PySide6.QtWidgets import QMessageBox

                QMessageBox.information(self.dialog, "Saved", "Crop coordinates were saved successfully!")
            except Exception as e:
                logger.error(f"Failed to save project: {e}")
                from PySide6.QtWidgets import QMessageBox

                QMessageBox.critical(self.dialog, "Error", f"Failed to save coordinates:\n{e}")
        else:
            # Fallback for standalone testing
            logger.info("Project saved (standalone mode).")

    def accept(self):
        # We save the state and exit
        self.final_db = self.crop_database
        self.dialog.accept()

    def exec(self):
        return self.dialog.exec()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add 'src' to sys.path so we can import 'behavython'
    src_path = Path(__file__).resolve().parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Mock some videos
    dummy_videos = ["mouse_1_trial_1.mp4", "mouse_1_trial_2.mp4", "mouse_2_trial_1.mp4", "mouse_3_trial_1.mp4"]

    # We pretend trial_2 is already cropped
    dummy_project = {vid: {"coordinates": None, "coordinates_set": False, "video_cropped": False} for vid in dummy_videos}
    dummy_project["mouse_1_trial_2.mp4"]["video_cropped"] = True

    cropper = VideoCropperDialog(project_data=dummy_project)
    if cropper.exec():
        print("Final Project Database:")
        import json

        print(json.dumps(cropper.final_db, indent=2))
    sys.exit(0)
