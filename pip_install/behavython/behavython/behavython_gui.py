# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'behavython_gui.ui'
##
## Created by: Qt User Interface Compiler version 6.3.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

import os
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QLabel, QLineEdit, QMainWindow, QProgressBar,
    QPushButton, QRadioButton, QSizePolicy, QTabWidget,
    QTextEdit, QWidget)

from behavython.behavython_plot_widget import plot_viewer
current_path = os.path.dirname(os.path.abspath(__file__))

icon_path = os.path.join(current_path, "VY.ico")
class Ui_behavython(object):
    def setupUi(self, behavython):
        if not behavython.objectName():
            behavython.setObjectName(u"behavython")
        behavython.resize(1280, 720)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(behavython.sizePolicy().hasHeightForWidth())
        behavython.setSizePolicy(sizePolicy)
        behavython.setMinimumSize(QSize(1280, 720))
        behavython.setMaximumSize(QSize(1280, 720))
        behavython.setWindowTitle(u"Behavython")
        icon = QIcon()
        icon.addFile(os.path.join(current_path, "logo", "VY.ico"), QSize(), QIcon.Normal, QIcon.Off)
        behavython.setWindowIcon(icon)
        behavython.setStyleSheet(u"QMainWindow{background-color:#4B4B4B;color:#FFFFFF;}")
        self.widget = QWidget(behavython)
        self.widget.setObjectName(u"widget")
        self.widget.setStyleSheet(u"QWidget{	background-color:#353535;}\n"
"QScrollBar:horizontal {\n"
"    background-color: #353535;\n"
"    height: 10px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"    background-color: #2C53A1;\n"
"    min-width: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-page:horizontal,\n"
"QScrollBar::sub-page:horizontal {\n"
"    background-color: #353535;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"    background-color: #353535;\n"
"    width: 10px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"    background-color: #2C53A1;\n"
"    min-height: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-page:vertical,\n"
"QScrollBar::sub-page:vertical {\n"
"    background-color: #353535;\n"
"}")
        self.plot_viewer = plot_viewer(self.widget)
        self.plot_viewer.setObjectName(u"plot_viewer")
        self.plot_viewer.setGeometry(QRect(330, 10, 931, 691))
        self.tabWidget = QTabWidget(self.widget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(20, 80, 310, 611))
        self.tabWidget.setStyleSheet(u"QTabWidget::pane {\n"
"    border-top: 2px solid #2C53A1;\n"
"    position: absolute;\n"
"    top: -0.5em;\n"
"}\n"
"\n"
"QTabWidget::tab-bar {\n"
"    alignment: center;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"    background: #606060;\n"
"    border: 2px solid #606060;\n"
"    border-bottom-color: #606060;\n"
"    border-top-left-radius: 8px;\n"
"    border-top-right-radius: 8px;\n"
"    border-bottom-left-radius: 8px;\n"
"    border-bottom-right-radius: 8px;\n"
"    width: 130px;\n"
"    height: 15px;\n"
"    padding: 0px;\n"
"    color: #FFFFFF;\n"
"    font: 7pt \"DejaVu Sans\";\n"
"    font-weight: bold;\n"
"    margin-left: 2px;\n"
"    margin-right: 2px;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"    background: #606060;\n"
"    border: 2px solid #2C53A1;\n"
"    border-bottom-color: #2C53A1;\n"
"}")
        self.tabWidget.setTabShape(QTabWidget.Rounded)
        self.tabWidget.setIconSize(QSize(4, 4))
        self.tabWidget.setUsesScrollButtons(True)
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.analysis_tab = QWidget()
        self.analysis_tab.setObjectName(u"analysis_tab")
        self.interface_frame = QFrame(self.analysis_tab)
        self.interface_frame.setObjectName(u"interface_frame")
        self.interface_frame.setGeometry(QRect(10, 10, 291, 600))
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.interface_frame.sizePolicy().hasHeightForWidth())
        self.interface_frame.setSizePolicy(sizePolicy1)
        self.interface_frame.setMaximumSize(QSize(500, 600))
        self.interface_frame.setStyleSheet(u"QLabel{}QCheckBox{color:#FFFFFF;font:8pt\"DejaVu Sans\";font-weight:bold;}QCheckBox::indicator{width:15px;height:15px;background-color:#606060;border-radius:4px;}QCheckBox::indicator:checked{background-color:#A21F27;}")
        self.analysis_button = QPushButton(self.interface_frame)
        self.analysis_button.setObjectName(u"analysis_button")
        self.analysis_button.setGeometry(QRect(20, 330, 261, 21))
        self.analysis_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.analysis_button.setStyleSheet(u"QPushButton{font:10pt\"DejaVu Sans\";font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:10pt\"DejaVu Sans\";font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}")
        self.analysis_button.setText(u"ANALYZE")
        self.label_1 = QLabel(self.interface_frame)
        self.label_1.setObjectName(u"label_1")
        self.label_1.setGeometry(QRect(20, 20, 261, 16))
        self.label_1.setStyleSheet(u"QLabel{border:#969696;background-color:#2C53A1;border-radius:8px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_1.setText(u"SETTINGS")
        self.label_1.setAlignment(Qt.AlignCenter)
        self.type_combobox = QComboBox(self.interface_frame)
        self.type_combobox.addItem("")
        self.type_combobox.addItem("")
        self.type_combobox.addItem(u"      Plus Maze")
        self.type_combobox.addItem(u"      Open Field")
        self.type_combobox.setObjectName(u"type_combobox")
        self.type_combobox.setGeometry(QRect(160, 40, 121, 16))
        self.type_combobox.setCursor(QCursor(Qt.PointingHandCursor))
        self.type_combobox.setStyleSheet(u"QComboBox{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QListView{color:#FFFFFF;font:8pt\"Century Gothic\";font-weight:bold;background-color:#969696;border:0px;}QListView{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QComboBox::drop-down{width:20px;border:5px;}QComboBox::down-arrow{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #FFFFFF;width:0.5px;height:1px;border-radius:2px;}QComboBox::down-arrow:hover{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #A21F27;width:0.5px;height:1px;border-radius:2px;}QAbstractItemView{border:2px solid #969696;selection-background-color:#2C53A1;}")
        self.type_combobox.setCurrentText(u"          NJR")
        self.clear_button = QPushButton(self.interface_frame)
        self.clear_button.setObjectName(u"clear_button")
        self.clear_button.setGeometry(QRect(20, 360, 261, 21))
        self.clear_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.clear_button.setStyleSheet(u"QPushButton{font:14pt\"DejaVu Sans\";font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:10pt\"DejaVu Sans\";font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}")
        self.clear_button.setText(u"RESET")
        self.arena_width_lineedit = QLineEdit(self.interface_frame)
        self.arena_width_lineedit.setObjectName(u"arena_width_lineedit")
        self.arena_width_lineedit.setGeometry(QRect(160, 80, 81, 16))
        self.arena_width_lineedit.setStyleSheet(u"QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.arena_width_lineedit.setText(u"63")
        self.arena_width_lineedit.setAlignment(Qt.AlignCenter)
        self.arena_height_lineedit = QLineEdit(self.interface_frame)
        self.arena_height_lineedit.setObjectName(u"arena_height_lineedit")
        self.arena_height_lineedit.setGeometry(QRect(160, 100, 81, 16))
        self.arena_height_lineedit.setStyleSheet(u"QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.arena_height_lineedit.setText(u"39")
        self.arena_height_lineedit.setAlignment(Qt.AlignCenter)
        self.frames_per_second_lineedit = QLineEdit(self.interface_frame)
        self.frames_per_second_lineedit.setObjectName(u"frames_per_second_lineedit")
        self.frames_per_second_lineedit.setGeometry(QRect(160, 120, 81, 16))
        self.frames_per_second_lineedit.setStyleSheet(u"QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.frames_per_second_lineedit.setText(u"30")
        self.frames_per_second_lineedit.setAlignment(Qt.AlignCenter)
        self.label_2 = QLabel(self.interface_frame)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(20, 40, 112, 16))
        self.label_2.setMaximumSize(QSize(114, 16))
        self.label_2.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_2.setText(u"Experiment type")
        self.label_2.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_3 = QLabel(self.interface_frame)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(20, 80, 114, 16))
        self.label_3.setMaximumSize(QSize(114, 16))
        self.label_3.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_3.setText(u"Arena width")
        self.label_3.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_4 = QLabel(self.interface_frame)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(20, 100, 114, 16))
        self.label_4.setMaximumSize(QSize(114, 16))
        self.label_4.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_4.setText(u"Arena height")
        self.label_4.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_5 = QLabel(self.interface_frame)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(20, 120, 131, 16))
        self.label_5.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_5.setText(u"Frames per second")
        self.label_5.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_6 = QLabel(self.interface_frame)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(240, 80, 41, 16))
        self.label_6.setStyleSheet(u"QLabel{border:#969696;background-color:#606060;border-bottom-right-radius:6px;border-top-right-radius:6px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_6.setText(u"cm")
        self.label_6.setAlignment(Qt.AlignCenter)
        self.label_7 = QLabel(self.interface_frame)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(240, 100, 41, 16))
        self.label_7.setStyleSheet(u"QLabel{border:#969696;background-color:#606060;border-bottom-right-radius:6px;border-top-right-radius:6px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_7.setText(u"cm")
        self.label_7.setAlignment(Qt.AlignCenter)
        self.label_8 = QLabel(self.interface_frame)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(240, 120, 41, 16))
        self.label_8.setStyleSheet(u"QLabel{border:#969696;background-color:#606060;border-bottom-right-radius:6px;border-top-right-radius:6px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_8.setText(u"fps")
        self.label_8.setAlignment(Qt.AlignCenter)
        self.label_10 = QLabel(self.interface_frame)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(20, 140, 131, 16))
        self.label_10.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_10.setText(u"Experimental animal")
        self.label_10.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.resume_lineedit = QTextEdit(self.interface_frame)
        self.resume_lineedit.setObjectName(u"resume_lineedit")
        self.resume_lineedit.setGeometry(QRect(20, 400, 261, 151))
        self.resume_lineedit.setStyleSheet(u"QTextEdit{color:#FFFFFF;font:5pt\"DejaVu Sans\";font-weight:bold}\n"
"QScrollBar:horizontal {\n"
"    background-color: #353535;\n"
"    height: 10px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"    background-color: #2C53A1;\n"
"    min-width: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-page:horizontal,\n"
"QScrollBar::sub-page:horizontal {\n"
"    background-color: #353535;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"    background-color: #353535;\n"
"    width: 10px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"    background-color: #2C53A1;\n"
"    min-height: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-page:vertical,\n"
"QScrollBar::sub-page:vertical {\n"
"    background-color: #353535;\n"
"}")
        self.resume_lineedit.setFrameShape(QFrame.NoFrame)
        self.resume_lineedit.setFrameShadow(QFrame.Plain)
        self.resume_lineedit.setReadOnly(True)
        self.progress_bar = QProgressBar(self.interface_frame)
        self.progress_bar.setObjectName(u"progress_bar")
        self.progress_bar.setGeometry(QRect(20, 560, 261, 23))
        self.progress_bar.setStyleSheet(u"QProgressBar { border: 2px solid #969696;border-radius:5px;text-align:center;color:#FFFFFF;font:10pt\"DejaVu Sans\";font-weight:bold;}QProgressBar::chunk{background-color:#2C53A1;width:20px;}\n"
"")
        self.progress_bar.setValue(0)
        self.animal_combobox = QComboBox(self.interface_frame)
        self.animal_combobox.addItem(u"     Mouse")
        self.animal_combobox.addItem(u"        Rat")
        self.animal_combobox.setObjectName(u"animal_combobox")
        self.animal_combobox.setGeometry(QRect(160, 140, 121, 16))
        self.animal_combobox.setCursor(QCursor(Qt.PointingHandCursor))
        self.animal_combobox.setStyleSheet(u"QComboBox{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QListView{color:#FFFFFF;font:8pt\"Century Gothic\";font-weight:bold;background-color:#969696;border:0px;}QListView{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QComboBox::drop-down{width:20px;border:5px;}QComboBox::down-arrow{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #FFFFFF;width:0.5px;height:1px;border-radius:2px;}QComboBox::down-arrow:hover{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #A21F27;width:0.5px;height:1px;border-radius:2px;}QAbstractItemView{border:2px solid #969696;selection-background-color:#2C53A1;}")
        self.animal_combobox.setCurrentText(u"     Mouse")
        self.label_12 = QLabel(self.interface_frame)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(20, 230, 131, 16))
        self.label_12.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_12.setText(u"Desired figure size")
        self.label_12.setAlignment(Qt.AlignCenter)
        self.fig_max_size = QComboBox(self.interface_frame)
        self.fig_max_size.addItem("")
        self.fig_max_size.addItem(u" 1280 x 720")
        self.fig_max_size.addItem("")
        self.fig_max_size.addItem("")
        self.fig_max_size.setObjectName(u"fig_max_size")
        self.fig_max_size.setGeometry(QRect(160, 230, 121, 16))
        sizePolicy2 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.fig_max_size.sizePolicy().hasHeightForWidth())
        self.fig_max_size.setSizePolicy(sizePolicy2)
        self.fig_max_size.setCursor(QCursor(Qt.PointingHandCursor))
        self.fig_max_size.setAutoFillBackground(False)
        self.fig_max_size.setStyleSheet(u"QComboBox{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QListView{color:#FFFFFF;font:8pt\"Century Gothic\";font-weight:bold;background-color:#969696;border:0px;}QListView{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QComboBox::drop-down{width:20px;border:5px;}QComboBox::down-arrow{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #FFFFFF;width:0.5px;height:1px;border-radius:2px;}QComboBox::down-arrow:hover{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #A21F27;width:0.5px;height:1px;border-radius:2px;}QAbstractItemView{border:2px solid #969696;selection-background-color:#2C53A1;}")
        self.fig_max_size.setCurrentText(u" 1280 x 720")
        self.fig_max_size.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        self.label_13 = QLabel(self.interface_frame)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(20, 290, 261, 31))
        font = QFont()
        font.setFamilies([u"DejaVu Sans"])
        font.setPointSize(6)
        font.setBold(True)
        font.setItalic(False)
        self.label_13.setFont(font)
        self.label_13.setStyleSheet(u"QLabel{border:#969696;background-color:#2C53A1;border-radius:8px;color:#FFFFFF;font:6pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_13.setText(u"Obs.: Hover the mouse over the attribute\n"
"to see the help text")
        self.label_13.setAlignment(Qt.AlignCenter)
        self.save_button = QRadioButton(self.interface_frame)
        self.save_button.setObjectName(u"save_button")
        self.save_button.setGeometry(QRect(50, 250, 91, 17))
        self.save_button.setStyleSheet(u"QRadioButton{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}QRadioButton::indicator::checked{border:2px solid #969696;border-radius:5px;background-color:#2C53A1;width:8px;height:8px;}QRadioButton::indicator::unchecked{border:2px solid #969696;border-radius:5px;background-color:#969696;width:8px;height:8px;}")
        self.save_button.setChecked(True)
        self.save_button.setAutoExclusive(True)
        self.only_plot_button = QRadioButton(self.interface_frame)
        self.only_plot_button.setObjectName(u"only_plot_button")
        self.only_plot_button.setGeometry(QRect(170, 250, 81, 17))
        self.only_plot_button.setStyleSheet(u"QRadioButton{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}QRadioButton::indicator::checked{border:2px solid #969696;border-radius:5px;background-color:#2C53A1;width:8px;height:8px;}QRadioButton::indicator::unchecked{border:2px solid #969696;border-radius:5px;background-color:#969696;width:8px;height:8px;}")
        self.only_plot_button.setChecked(False)
        self.label_14 = QLabel(self.interface_frame)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(20, 210, 261, 16))
        self.label_14.setStyleSheet(u"QLabel{border:#969696;background-color:#2C53A1;border-radius:8px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_14.setText(u"ADVANCED SETTINGS")
        self.label_14.setAlignment(Qt.AlignCenter)
        self.algo_type_combobox = QComboBox(self.interface_frame)
        self.algo_type_combobox.addItem(u"  DeepLabCut")
        self.algo_type_combobox.addItem(u"      Bonsai")
        self.algo_type_combobox.setObjectName(u"algo_type_combobox")
        self.algo_type_combobox.setGeometry(QRect(160, 60, 121, 16))
        self.algo_type_combobox.setCursor(QCursor(Qt.PointingHandCursor))
        self.algo_type_combobox.setStyleSheet(u"QComboBox{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QListView{color:#FFFFFF;font:8pt\"Century Gothic\";font-weight:bold;background-color:#969696;border:0px;}QListView{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QComboBox::drop-down{width:20px;border:5px;}QComboBox::down-arrow{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #FFFFFF;width:0.5px;height:1px;border-radius:2px;}QComboBox::down-arrow:hover{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #A21F27;width:0.5px;height:1px;border-radius:2px;}QAbstractItemView{border:2px solid #969696;selection-background-color:#2C53A1;}")
        self.algo_type_combobox.setCurrentText(u"  DeepLabCut")
        self.label_9 = QLabel(self.interface_frame)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(20, 60, 112, 16))
        self.label_9.setMaximumSize(QSize(114, 16))
        self.label_9.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_9.setText(u"Analysis Engine")
        self.label_9.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_11 = QLabel(self.interface_frame)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(20, 160, 114, 16))
        self.label_11.setMaximumSize(QSize(114, 16))
        self.label_11.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_11.setText(u"Task duration")
        self.label_11.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.task_duration_lineedit = QLineEdit(self.interface_frame)
        self.task_duration_lineedit.setObjectName(u"task_duration_lineedit")
        self.task_duration_lineedit.setGeometry(QRect(160, 160, 91, 16))
        self.task_duration_lineedit.setStyleSheet(u"QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;border-bottom-right-radius:6px;border-top-right-radius:6px}")
        self.task_duration_lineedit.setText(u"300")
        self.task_duration_lineedit.setAlignment(Qt.AlignCenter)
        self.crop_video_checkbox = QCheckBox(self.interface_frame)
        self.crop_video_checkbox.setObjectName(u"crop_video_checkbox")
        self.crop_video_checkbox.setGeometry(QRect(110, 270, 88, 17))
        self.crop_video_checkbox.setChecked(True)
        self.crop_video_time_lineedit = QLineEdit(self.interface_frame)
        self.crop_video_time_lineedit.setObjectName(u"crop_video_time_lineedit")
        self.crop_video_time_lineedit.setGeometry(QRect(160, 180, 91, 16))
        self.crop_video_time_lineedit.setStyleSheet(u"QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;border-bottom-right-radius:6px;border-top-right-radius:6px}")
        self.crop_video_time_lineedit.setText(u"15")
        self.crop_video_time_lineedit.setAlignment(Qt.AlignCenter)
        self.label_15 = QLabel(self.interface_frame)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(20, 180, 114, 16))
        self.label_15.setMaximumSize(QSize(114, 16))
        self.label_15.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_15.setText(u"How much to trim")
        self.label_15.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_16 = QLabel(self.interface_frame)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(240, 180, 41, 16))
        self.label_16.setStyleSheet(u"QLabel{border:#969696;background-color:#606060;border-bottom-right-radius:6px;border-top-right-radius:6px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_16.setText(u"sec")
        self.label_16.setAlignment(Qt.AlignCenter)
        self.label_17 = QLabel(self.interface_frame)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setGeometry(QRect(240, 160, 41, 16))
        self.label_17.setStyleSheet(u"QLabel{border:#969696;background-color:#606060;border-bottom-right-radius:6px;border-top-right-radius:6px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.label_17.setText(u"sec")
        self.label_17.setAlignment(Qt.AlignCenter)
        self.tabWidget.addTab(self.analysis_tab, "")
        self.dlc_tab = QWidget()
        self.dlc_tab.setObjectName(u"dlc_tab")
        self.network_settings = QFrame(self.dlc_tab)
        self.network_settings.setObjectName(u"network_settings")
        self.network_settings.setGeometry(QRect(10, 0, 291, 600))
        sizePolicy1.setHeightForWidth(self.network_settings.sizePolicy().hasHeightForWidth())
        self.network_settings.setSizePolicy(sizePolicy1)
        self.network_settings.setMaximumSize(QSize(500, 600))
        self.network_settings.setStyleSheet(u"QLabel{}QCheckBox{color:#FFFFFF;font:6pt\"DejaVu Sans\";font-weight:bold;}QCheckBox::indicator{width:15px;height:15px;background-color:#606060;border-radius:4px;}QCheckBox::indicator:checked{background-color:#A21F27;}")
        self.network_tab_title = QLabel(self.network_settings)
        self.network_tab_title.setObjectName(u"network_tab_title")
        self.network_tab_title.setGeometry(QRect(20, 20, 261, 16))
        self.network_tab_title.setStyleSheet(u"QLabel{border:#969696;background-color:#2C53A1;border-radius:8px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.network_tab_title.setText(u"NETWORK SETTINGS")
        self.network_tab_title.setAlignment(Qt.AlignCenter)
        self.config_path_lineedit = QLineEdit(self.network_settings)
        self.config_path_lineedit.setObjectName(u"config_path_lineedit")
        self.config_path_lineedit.setGeometry(QRect(0, 70, 285, 16))
        self.config_path_lineedit.setMinimumSize(QSize(285, 0))
        self.config_path_lineedit.setMaximumSize(QSize(500, 20))
        self.config_path_lineedit.setStyleSheet(u"QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;border-bottom-right-radius:6px;border-top-right-radius:6px}")
        self.config_path_lineedit.setText(u"")
        self.config_path_lineedit.setFrame(False)
        self.config_path_lineedit.setCursorPosition(0)
        self.config_path_lineedit.setAlignment(Qt.AlignCenter)
        self.config_path_lineedit.setReadOnly(True)
        self.config_path = QLabel(self.network_settings)
        self.config_path.setObjectName(u"config_path")
        self.config_path.setGeometry(QRect(0, 50, 200, 16))
        sizePolicy.setHeightForWidth(self.config_path.sizePolicy().hasHeightForWidth())
        self.config_path.setSizePolicy(sizePolicy)
        self.config_path.setMinimumSize(QSize(200, 16))
        self.config_path.setMaximumSize(QSize(200, 16))
        self.config_path.setStyleSheet(u"QLabel{color:#FFFFFF;font:6pt\"DejaVu Sans\";font-weight:bold;}")
        self.config_path.setText(u"Path to the config.yaml file ")
        self.config_path.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.video_folder_lineedit = QLineEdit(self.network_settings)
        self.video_folder_lineedit.setObjectName(u"video_folder_lineedit")
        self.video_folder_lineedit.setGeometry(QRect(0, 120, 285, 16))
        self.video_folder_lineedit.setMinimumSize(QSize(285, 0))
        self.video_folder_lineedit.setMaximumSize(QSize(500, 20))
        self.video_folder_lineedit.setStyleSheet(u"QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;border-bottom-right-radius:6px;border-top-right-radius:6px}")
        self.video_folder_lineedit.setText(u"")
        self.video_folder_lineedit.setFrame(False)
        self.video_folder_lineedit.setCursorPosition(0)
        self.video_folder_lineedit.setAlignment(Qt.AlignCenter)
        self.video_folder_lineedit.setReadOnly(True)
        self.video_folder = QLabel(self.network_settings)
        self.video_folder.setObjectName(u"video_folder")
        self.video_folder.setGeometry(QRect(0, 97, 195, 16))
        sizePolicy.setHeightForWidth(self.video_folder.sizePolicy().hasHeightForWidth())
        self.video_folder.setSizePolicy(sizePolicy)
        self.video_folder.setMinimumSize(QSize(130, 16))
        self.video_folder.setMaximumSize(QSize(200, 42))
        self.video_folder.setFont(font)
        self.video_folder.setStyleSheet(u"QLabel{color:#FFFFFF;font:6pt\"DejaVu Sans\";font-weight:bold;}")
        self.video_folder.setText(u"Folder containing the videos to analyze")
        self.video_folder.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.video_folder.setWordWrap(True)
        self.folder_structure_check = QLabel(self.network_settings)
        self.folder_structure_check.setObjectName(u"folder_structure_check")
        self.folder_structure_check.setGeometry(QRect(3, 142, 280, 16))
        sizePolicy3 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.folder_structure_check.sizePolicy().hasHeightForWidth())
        self.folder_structure_check.setSizePolicy(sizePolicy3)
        self.folder_structure_check.setMinimumSize(QSize(280, 16))
        self.folder_structure_check.setMaximumSize(QSize(214, 16))
        self.folder_structure_check.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.folder_structure_check.setText(u"Check if the folder structure is correct")
        self.folder_structure_check.setAlignment(Qt.AlignCenter)
        self.folder_structure_check_button = QPushButton(self.network_settings)
        self.folder_structure_check_button.setObjectName(u"folder_structure_check_button")
        self.folder_structure_check_button.setGeometry(QRect(61, 160, 171, 31))
        self.folder_structure_check_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.folder_structure_check_button.setStyleSheet(u"QPushButton{font:8pt\"DejaVu Sans\";font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:9pt\"DejaVu Sans\";font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}")
        self.folder_structure_check_button.setText(u"CHECK")
        self.get_frames_label = QLabel(self.network_settings)
        self.get_frames_label.setObjectName(u"get_frames_label")
        self.get_frames_label.setGeometry(QRect(0, 319, 161, 30))
        sizePolicy3.setHeightForWidth(self.get_frames_label.sizePolicy().hasHeightForWidth())
        self.get_frames_label.setSizePolicy(sizePolicy3)
        self.get_frames_label.setMinimumSize(QSize(100, 30))
        self.get_frames_label.setMaximumSize(QSize(214, 16))
        self.get_frames_label.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.get_frames_label.setText(u"Get a frame from every video")
        self.get_frames_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.get_frames_label.setWordWrap(True)
        self.get_frames_button = QPushButton(self.network_settings)
        self.get_frames_button.setObjectName(u"get_frames_button")
        self.get_frames_button.setGeometry(QRect(160, 319, 131, 31))
        self.get_frames_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.get_frames_button.setStyleSheet(u"QPushButton{font:14pt\"DejaVu Sans\";font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:8pt\"DejaVu Sans\";font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}")
        self.get_frames_button.setText(u"GET FRAME")
        self.data_refining_title = QLabel(self.network_settings)
        self.data_refining_title.setObjectName(u"data_refining_title")
        self.data_refining_title.setGeometry(QRect(10, 259, 271, 16))
        self.data_refining_title.setStyleSheet(u"QLabel{border:#969696;background-color:#2C53A1;border-radius:8px;color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.data_refining_title.setText(u"DATA REFINING")
        self.data_refining_title.setAlignment(Qt.AlignCenter)
        self.extract_skeleton_title = QLabel(self.network_settings)
        self.extract_skeleton_title.setObjectName(u"extract_skeleton_title")
        self.extract_skeleton_title.setGeometry(QRect(0, 287, 151, 16))
        sizePolicy3.setHeightForWidth(self.extract_skeleton_title.sizePolicy().hasHeightForWidth())
        self.extract_skeleton_title.setSizePolicy(sizePolicy3)
        self.extract_skeleton_title.setMinimumSize(QSize(100, 16))
        self.extract_skeleton_title.setMaximumSize(QSize(214, 16))
        self.extract_skeleton_title.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.extract_skeleton_title.setText(u"Extract skeleton data")
        self.extract_skeleton_title.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.extract_skeleton_button = QPushButton(self.network_settings)
        self.extract_skeleton_button.setObjectName(u"extract_skeleton_button")
        self.extract_skeleton_button.setGeometry(QRect(160, 280, 131, 31))
        self.extract_skeleton_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.extract_skeleton_button.setStyleSheet(u"QPushButton{font:14pt\"DejaVu Sans\";font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:8pt\"DejaVu Sans\";font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}")
        self.extract_skeleton_button.setText(u"EXTRACT DATA")
        self.clear_unused_files_button = QPushButton(self.network_settings)
        self.clear_unused_files_button.setObjectName(u"clear_unused_files_button")
        self.clear_unused_files_button.setGeometry(QRect(160, 359, 131, 31))
        self.clear_unused_files_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.clear_unused_files_button.setStyleSheet(u"QPushButton{font:14pt\"DejaVu Sans\";font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:8pt\"DejaVu Sans\";font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}")
        self.clear_unused_files_button.setText(u"CLEANUP FOLDER")
        self.clear_unused_files_title = QLabel(self.network_settings)
        self.clear_unused_files_title.setObjectName(u"clear_unused_files_title")
        self.clear_unused_files_title.setGeometry(QRect(0, 365, 151, 16))
        sizePolicy3.setHeightForWidth(self.clear_unused_files_title.sizePolicy().hasHeightForWidth())
        self.clear_unused_files_title.setSizePolicy(sizePolicy3)
        self.clear_unused_files_title.setMinimumSize(QSize(100, 16))
        self.clear_unused_files_title.setMaximumSize(QSize(214, 16))
        self.clear_unused_files_title.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.clear_unused_files_title.setText(u"Remove unnecessary files")
        self.clear_unused_files_title.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.clear_unused_files_lineedit = QTextEdit(self.network_settings)
        self.clear_unused_files_lineedit.setObjectName(u"clear_unused_files_lineedit")
        self.clear_unused_files_lineedit.setGeometry(QRect(0, 396, 291, 191))
        font1 = QFont()
        font1.setFamilies([u"DejaVu Sans"])
        font1.setPointSize(8)
        font1.setBold(False)
        font1.setItalic(False)
        self.clear_unused_files_lineedit.setFont(font1)
        self.clear_unused_files_lineedit.setStyleSheet(u"QTextEdit{color:#FFFFFF;font:8pt\"DejaVu Sans\";font-weight:light}\n"
"QScrollBar:horizontal {\n"
"    background-color: #353535;\n"
"    height: 10px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"    background-color: #2C53A1;\n"
"    min-width: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-page:horizontal,\n"
"QScrollBar::sub-page:horizontal {\n"
"    background-color: #353535;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"    background-color: #353535;\n"
"    width: 10px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"    background-color: #2C53A1;\n"
"    min-height: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-page:vertical,\n"
"QScrollBar::sub-page:vertical {\n"
"    background-color: #353535;\n"
"}")
        self.clear_unused_files_lineedit.setFrameShape(QFrame.StyledPanel)
        self.clear_unused_files_lineedit.setFrameShadow(QFrame.Plain)
        self.clear_unused_files_lineedit.setReadOnly(True)
        self.dlc_video_analyze_title = QLabel(self.network_settings)
        self.dlc_video_analyze_title.setObjectName(u"dlc_video_analyze_title")
        self.dlc_video_analyze_title.setGeometry(QRect(3, 197, 280, 16))
        sizePolicy3.setHeightForWidth(self.dlc_video_analyze_title.sizePolicy().hasHeightForWidth())
        self.dlc_video_analyze_title.setSizePolicy(sizePolicy3)
        self.dlc_video_analyze_title.setMinimumSize(QSize(280, 16))
        self.dlc_video_analyze_title.setMaximumSize(QSize(214, 16))
        self.dlc_video_analyze_title.setStyleSheet(u"QLabel{color:#FFFFFF;font:7pt\"DejaVu Sans\";font-weight:bold;}")
        self.dlc_video_analyze_title.setText(u"Analyze videos using the network supplied")
        self.dlc_video_analyze_title.setAlignment(Qt.AlignCenter)
        self.dlc_video_analyze_button = QPushButton(self.network_settings)
        self.dlc_video_analyze_button.setObjectName(u"dlc_video_analyze_button")
        self.dlc_video_analyze_button.setGeometry(QRect(61, 216, 171, 31))
        font2 = QFont()
        font2.setFamilies([u"DejaVu Sans"])
        font2.setPointSize(9)
        font2.setBold(True)
        font2.setItalic(False)
        self.dlc_video_analyze_button.setFont(font2)
        self.dlc_video_analyze_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.dlc_video_analyze_button.setStyleSheet(u"QPushButton{font:14pt\"DejaVu Sans\";font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:9pt\"DejaVu Sans\";font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}")
        self.dlc_video_analyze_button.setText(u"ANALYZE")
        self.get_config_path_button = QPushButton(self.network_settings)
        self.get_config_path_button.setObjectName(u"get_config_path_button")
        self.get_config_path_button.setGeometry(QRect(204, 44, 81, 20))
        font3 = QFont()
        font3.setFamilies([u"DejaVu Sans"])
        font3.setPointSize(8)
        font3.setBold(True)
        font3.setItalic(False)
        self.get_config_path_button.setFont(font3)
        self.get_config_path_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.get_config_path_button.setStyleSheet(u"QPushButton{font:10pt\"DejaVu Sans\";font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:8pt\"DejaVu Sans\";font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}")
        self.get_config_path_button.setText(u"GET FILE")
        self.get_videos_path_button = QPushButton(self.network_settings)
        self.get_videos_path_button.setObjectName(u"get_videos_path_button")
        self.get_videos_path_button.setGeometry(QRect(205, 94, 81, 20))
        self.get_videos_path_button.setFont(font3)
        self.get_videos_path_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.get_videos_path_button.setStyleSheet(u"QPushButton{font:6pt\"DejaVu Sans\";font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:8pt\"DejaVu Sans\";font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}")
        self.get_videos_path_button.setText(u"GET FOLDER")
        self.tabWidget.addTab(self.dlc_tab, "")
        self.behavython_logo = QLabel(self.widget)
        self.behavython_logo.setObjectName(u"behavython_logo")
        self.behavython_logo.setGeometry(QRect(60, 10, 231, 61))
        self.behavython_logo.setPixmap(QPixmap(os.path.join(current_path, "logo", "logo.png")))
        self.behavython_logo.setScaledContents(True)
        self.behavython_logo.setAlignment(Qt.AlignCenter)
        behavython.setCentralWidget(self.widget)

        self.retranslateUi(behavython)

        self.tabWidget.setCurrentIndex(0)
        self.type_combobox.setCurrentIndex(0)
        self.animal_combobox.setCurrentIndex(0)
        self.fig_max_size.setCurrentIndex(1)
        self.algo_type_combobox.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(behavython)
    # setupUi

    def retranslateUi(self, behavython):
        self.type_combobox.setItemText(0, QCoreApplication.translate("behavython", u"          NJR", None))
        self.type_combobox.setItemText(1, QCoreApplication.translate("behavython", u"Social Recognition", None))

#if QT_CONFIG(tooltip)
        self.label_10.setToolTip(QCoreApplication.translate("behavython", u"<html><head/><body><p><span style=\" font-size:8pt; font-weight:400;\">Select the experimental animal used.</span></p><p align=\"justify\"><span style=\" font-size:8pt; font-weight:400; font-style:italic;\">Currently the calculations are being made considering these animals:<br/>Mouse = C57BL/6<br/>Rat = Wistar</span></p></body></html>", None))
#endif // QT_CONFIG(tooltip)

#if QT_CONFIG(tooltip)
        self.animal_combobox.setToolTip(QCoreApplication.translate("behavython", u"<html><head/><body><p><span style=\" font-size:8pt; font-weight:400;\">Select the experimental animal used.</span></p><p align=\"justify\"><span style=\" font-size:8pt; font-weight:400; font-style:italic;\">Currently the calculations are being made considering these animals:<br/>Mouse = C57BL/6<br/>Rat = Wistar</span></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.label_12.setToolTip(QCoreApplication.translate("behavython", u"<html><head/><body><p><span style=\" font-size:8pt; font-weight:400;\">Set the maximum resolution for the resulting figures computed.</span></p><p><span style=\" font-size:8pt; font-weight:400; font-style:italic;\">Note that, to maintain aspect ratio of the original file, the final resolution might have &quot;uncommon&quot; sizes and might not match 1:1 the maximum resolution set (if the original image also had &quot;uncommon&quot; resolution).</span></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.fig_max_size.setItemText(0, QCoreApplication.translate("behavython", u"  640 x 480", None))
        self.fig_max_size.setItemText(2, QCoreApplication.translate("behavython", u"1920 x 1080", None))
        self.fig_max_size.setItemText(3, QCoreApplication.translate("behavython", u"2560 x 1440", None))

#if QT_CONFIG(tooltip)
        self.fig_max_size.setToolTip(QCoreApplication.translate("behavython", u"<html><head/><body><p><span style=\" font-size:8pt; font-weight:400;\">Set the maximum resolution for the resulting figures computed.</span></p><p><span style=\" font-size:8pt; font-weight:400; font-style:italic;\">Note that, to maintain aspect ratio of the original file, the final resolution might have &quot;uncommon&quot; sizes and might not match 1:1 the maximum resolution set (if the original image also had &quot;uncommon&quot; resolution).</span></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.save_button.setText(QCoreApplication.translate("behavython", u"Save data", None))
        self.only_plot_button.setText(QCoreApplication.translate("behavython", u"Only plot", None))

        self.crop_video_checkbox.setText(QCoreApplication.translate("behavython", u"Crop video", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.analysis_tab), QCoreApplication.translate("behavython", u"ANALYSIS", None))
        self.config_path_lineedit.setPlaceholderText(QCoreApplication.translate("behavython", u"%USERPROFILE%\\path\\to\\project_folder\\network_name_date\\config.yaml", None))
        self.video_folder_lineedit.setPlaceholderText(QCoreApplication.translate("behavython", u"%USERPROFILE%\\path\\to\\video\\folder\\", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.dlc_tab), QCoreApplication.translate("behavython", u"DEEPLABCUT", None))
        self.behavython_logo.setText("")
        pass
    # retranslateUi

