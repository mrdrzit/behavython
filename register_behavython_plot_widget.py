# File: registerwigglywidget.py
from behavython_plot_widget import plot_viewer
from PySide6.QtDesigner import QPyDesignerCustomWidgetCollection


TOOLTIP = "Plot viewer widget"
DOM_XML = """
<?xml version="1.0" encoding="UTF-8"?>
<ui version="4.0">
 <class>behavython</class>
 <widget class="QMainWindow" name="behavython">
  <property name="geometry">
   <rect>
    <x>0</x>
    <y>0</y>
    <width>1280</width>
    <height>720</height>
   </rect>
  </property>
  <property name="sizePolicy">
   <sizepolicy hsizetype="Fixed" vsizetype="Fixed">
    <horstretch>0</horstretch>
    <verstretch>0</verstretch>
   </sizepolicy>
  </property>
  <property name="minimumSize">
   <size>
    <width>1280</width>
    <height>720</height>
   </size>
  </property>
  <property name="maximumSize">
   <size>
    <width>1280</width>
    <height>720</height>
   </size>
  </property>
  <property name="windowTitle">
   <string notr="true">Behavython</string>
  </property>
  <property name="windowIcon">
   <iconset>
    <normaloff>logo/VY.ico</normaloff>logo/VY.ico</iconset>
  </property>
  <property name="styleSheet">
   <string notr="true">QMainWindow{background-color:#4B4B4B;color:#FFFFFF;}</string>
  </property>
  <widget class="QWidget" name="widget">
   <property name="styleSheet">
    <string notr="true">QWidget{	background-color:#353535;}
QScrollBar:horizontal {
    background-color: #353535;
    height: 10px;
}

QScrollBar::handle:horizontal {
    background-color: #2C53A1;
    min-width: 20px;
}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background-color: #353535;
}

QScrollBar:vertical {
    background-color: #353535;
    width: 10px;
}

QScrollBar::handle:vertical {
    background-color: #2C53A1;
    min-height: 20px;
}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background-color: #353535;
}</string>
   </property>
   <widget class="plot_viewer" name="plot_viewer" native="true">
    <property name="geometry">
     <rect>
      <x>330</x>
      <y>10</y>
      <width>931</width>
      <height>691</height>
     </rect>
    </property>
   </widget>
   <widget class="QTabWidget" name="tabWidget">
    <property name="geometry">
     <rect>
      <x>20</x>
      <y>80</y>
      <width>311</width>
      <height>611</height>
     </rect>
    </property>
    <property name="styleSheet">
     <string notr="true">QTabWidget::pane {border-top:2px solid #2C53A1;position: absolute;top: -0.5em;}QTabWidget::tab-bar{alignment: center;}QTabBar::tab{background:#606060;border:2px solid #606060;border-bottom-color: #606060;border-top-left-radius:8px;border-top-right-radius:8px;border-bottom-left-radius:8px;border-bottom-right-radius:8px;width: 45ex;height:7ex;padding:0px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;margin-left:2px;margin-right:2px;}QTabBar::tab:selected{background:#606060; border: 2px solid #2C53A1;border-bottom-color:#2C53A1;}</string>
    </property>
    <property name="currentIndex">
     <number>1</number>
    </property>
    <property name="tabBarAutoHide">
     <bool>false</bool>
    </property>
    <widget class="QWidget" name="analysis_tab">
     <attribute name="title">
      <string>ANALYSIS</string>
     </attribute>
     <widget class="QFrame" name="interface_frame">
      <property name="geometry">
       <rect>
        <x>10</x>
        <y>10</y>
        <width>291</width>
        <height>600</height>
       </rect>
      </property>
      <property name="sizePolicy">
       <sizepolicy hsizetype="Preferred" vsizetype="Preferred">
        <horstretch>0</horstretch>
        <verstretch>0</verstretch>
       </sizepolicy>
      </property>
      <property name="maximumSize">
       <size>
        <width>500</width>
        <height>600</height>
       </size>
      </property>
      <property name="styleSheet">
       <string notr="true">QLabel{}QCheckBox{color:#FFFFFF;font:8pt&quot;DejaVu Sans&quot;;font-weight:bold;}QCheckBox::indicator{width:15px;height:15px;background-color:#606060;border-radius:4px;}QCheckBox::indicator:checked{background-color:#A21F27;}</string>
      </property>
      <widget class="QPushButton" name="analysis_button">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>330</y>
         <width>261</width>
         <height>21</height>
        </rect>
       </property>
       <property name="cursor">
        <cursorShape>PointingHandCursor</cursorShape>
       </property>
       <property name="styleSheet">
        <string notr="true">QPushButton{font:10pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:10pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}</string>
       </property>
       <property name="text">
        <string notr="true">ANALYZE</string>
       </property>
      </widget>
      <widget class="QLabel" name="label_1">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>20</y>
         <width>261</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{border:#969696;background-color:#2C53A1;border-radius:8px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">SETTINGS</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QComboBox" name="type_combobox">
       <property name="geometry">
        <rect>
         <x>160</x>
         <y>40</y>
         <width>121</width>
         <height>16</height>
        </rect>
       </property>
       <property name="cursor">
        <cursorShape>PointingHandCursor</cursorShape>
       </property>
       <property name="styleSheet">
        <string notr="true">QComboBox{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QListView{color:#FFFFFF;font:8pt&quot;Century Gothic&quot;;font-weight:bold;background-color:#969696;border:0px;}QListView{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QComboBox::drop-down{width:20px;border:5px;}QComboBox::down-arrow{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #FFFFFF;width:0.5px;height:1px;border-radius:2px;}QComboBox::down-arrow:hover{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #A21F27;width:0.5px;height:1px;border-radius:2px;}QAbstractItemView{border:2px solid #969696;selection-background-color:#2C53A1;}</string>
       </property>
       <property name="currentText">
        <string notr="true">          NJR</string>
       </property>
       <property name="currentIndex">
        <number>0</number>
       </property>
       <item>
        <property name="text">
         <string>          NJR</string>
        </property>
       </item>
       <item>
        <property name="text">
         <string>Social Recognition</string>
        </property>
       </item>
       <item>
        <property name="text">
         <string notr="true">      Plus Maze</string>
        </property>
       </item>
       <item>
        <property name="text">
         <string notr="true">      Open Field</string>
        </property>
       </item>
      </widget>
      <widget class="QPushButton" name="clear_button">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>360</y>
         <width>261</width>
         <height>21</height>
        </rect>
       </property>
       <property name="cursor">
        <cursorShape>PointingHandCursor</cursorShape>
       </property>
       <property name="styleSheet">
        <string notr="true">QPushButton{font:14pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:10pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}</string>
       </property>
       <property name="text">
        <string notr="true">RESET</string>
       </property>
      </widget>
      <widget class="QLineEdit" name="arena_width_lineedit">
       <property name="geometry">
        <rect>
         <x>160</x>
         <y>80</y>
         <width>81</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">65</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QLineEdit" name="arena_height_lineedit">
       <property name="geometry">
        <rect>
         <x>160</x>
         <y>100</y>
         <width>81</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">65</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QLineEdit" name="frames_per_second_lineedit">
       <property name="geometry">
        <rect>
         <x>160</x>
         <y>120</y>
         <width>81</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">30</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QLabel" name="label_2">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>40</y>
         <width>112</width>
         <height>16</height>
        </rect>
       </property>
       <property name="maximumSize">
        <size>
         <width>114</width>
         <height>16</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Experiment type</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter</set>
       </property>
      </widget>
      <widget class="QLabel" name="label_3">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>80</y>
         <width>114</width>
         <height>16</height>
        </rect>
       </property>
       <property name="maximumSize">
        <size>
         <width>114</width>
         <height>16</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Arena width</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter</set>
       </property>
      </widget>
      <widget class="QLabel" name="label_4">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>100</y>
         <width>114</width>
         <height>16</height>
        </rect>
       </property>
       <property name="maximumSize">
        <size>
         <width>114</width>
         <height>16</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Arena height</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter</set>
       </property>
      </widget>
      <widget class="QLabel" name="label_5">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>120</y>
         <width>131</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Frames per second</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter</set>
       </property>
      </widget>
      <widget class="QLabel" name="label_6">
       <property name="geometry">
        <rect>
         <x>240</x>
         <y>80</y>
         <width>41</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{border:#969696;background-color:#606060;border-bottom-right-radius:6px;border-top-right-radius:6px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">cm</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QLabel" name="label_7">
       <property name="geometry">
        <rect>
         <x>240</x>
         <y>100</y>
         <width>41</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{border:#969696;background-color:#606060;border-bottom-right-radius:6px;border-top-right-radius:6px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">cm</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QLabel" name="label_8">
       <property name="geometry">
        <rect>
         <x>240</x>
         <y>120</y>
         <width>41</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{border:#969696;background-color:#606060;border-bottom-right-radius:6px;border-top-right-radius:6px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">fps</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QLabel" name="label_10">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>140</y>
         <width>131</width>
         <height>16</height>
        </rect>
       </property>
       <property name="toolTip">
        <string>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;&lt;span style=&quot; font-size:8pt; font-weight:400;&quot;&gt;Select the experimental animal used.&lt;/span&gt;&lt;/p&gt;&lt;p align=&quot;justify&quot;&gt;&lt;span style=&quot; font-size:8pt; font-weight:400; font-style:italic;&quot;&gt;Currently the calculations are being made considering these animals:&lt;br/&gt;Mouse = C57BL/6&lt;br/&gt;Rat = Wistar&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</string>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Experimental animal</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter</set>
       </property>
      </widget>
      <widget class="QTextEdit" name="resume_lineedit">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>400</y>
         <width>261</width>
         <height>151</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QTextEdit{color:#FFFFFF;font:5pt&quot;DejaVu Sans&quot;;font-weight:bold}
QScrollBar:horizontal {
    background-color: #353535;
    height: 10px;
}

QScrollBar::handle:horizontal {
    background-color: #2C53A1;
    min-width: 20px;
}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background-color: #353535;
}

QScrollBar:vertical {
    background-color: #353535;
    width: 10px;
}

QScrollBar::handle:vertical {
    background-color: #2C53A1;
    min-height: 20px;
}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background-color: #353535;
}</string>
       </property>
       <property name="frameShape">
        <enum>QFrame::NoFrame</enum>
       </property>
       <property name="frameShadow">
        <enum>QFrame::Plain</enum>
       </property>
       <property name="readOnly">
        <bool>true</bool>
       </property>
      </widget>
      <widget class="QProgressBar" name="progress_bar">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>560</y>
         <width>261</width>
         <height>23</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QProgressBar { border: 2px solid #969696;border-radius:5px;text-align:center;color:#FFFFFF;font:10pt&quot;DejaVu Sans&quot;;font-weight:bold;}QProgressBar::chunk{background-color:#2C53A1;width:20px;}
</string>
       </property>
       <property name="value">
        <number>0</number>
       </property>
      </widget>
      <widget class="QComboBox" name="animal_combobox">
       <property name="geometry">
        <rect>
         <x>160</x>
         <y>140</y>
         <width>121</width>
         <height>16</height>
        </rect>
       </property>
       <property name="cursor">
        <cursorShape>PointingHandCursor</cursorShape>
       </property>
       <property name="toolTip">
        <string>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;&lt;span style=&quot; font-size:8pt; font-weight:400;&quot;&gt;Select the experimental animal used.&lt;/span&gt;&lt;/p&gt;&lt;p align=&quot;justify&quot;&gt;&lt;span style=&quot; font-size:8pt; font-weight:400; font-style:italic;&quot;&gt;Currently the calculations are being made considering these animals:&lt;br/&gt;Mouse = C57BL/6&lt;br/&gt;Rat = Wistar&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</string>
       </property>
       <property name="styleSheet">
        <string notr="true">QComboBox{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QListView{color:#FFFFFF;font:8pt&quot;Century Gothic&quot;;font-weight:bold;background-color:#969696;border:0px;}QListView{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QComboBox::drop-down{width:20px;border:5px;}QComboBox::down-arrow{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #FFFFFF;width:0.5px;height:1px;border-radius:2px;}QComboBox::down-arrow:hover{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #A21F27;width:0.5px;height:1px;border-radius:2px;}QAbstractItemView{border:2px solid #969696;selection-background-color:#2C53A1;}</string>
       </property>
       <property name="currentText">
        <string notr="true">     Mouse</string>
       </property>
       <property name="currentIndex">
        <number>0</number>
       </property>
       <item>
        <property name="text">
         <string notr="true">     Mouse</string>
        </property>
       </item>
       <item>
        <property name="text">
         <string notr="true">        Rat</string>
        </property>
       </item>
      </widget>
      <widget class="QLabel" name="label_12">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>230</y>
         <width>131</width>
         <height>16</height>
        </rect>
       </property>
       <property name="toolTip">
        <string>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;&lt;span style=&quot; font-size:8pt; font-weight:400;&quot;&gt;Set the maximum resolution for the resulting figures computed.&lt;/span&gt;&lt;/p&gt;&lt;p&gt;&lt;span style=&quot; font-size:8pt; font-weight:400; font-style:italic;&quot;&gt;Note that, to maintain aspect ratio of the original file, the final resolution might have &amp;quot;uncommon&amp;quot; sizes and might not match 1:1 the maximum resolution set (if the original image also had &amp;quot;uncommon&amp;quot; resolution).&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</string>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Desired figure size</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QComboBox" name="fig_max_size">
       <property name="geometry">
        <rect>
         <x>160</x>
         <y>230</y>
         <width>121</width>
         <height>16</height>
        </rect>
       </property>
       <property name="sizePolicy">
        <sizepolicy hsizetype="Maximum" vsizetype="Maximum">
         <horstretch>0</horstretch>
         <verstretch>0</verstretch>
        </sizepolicy>
       </property>
       <property name="cursor">
        <cursorShape>PointingHandCursor</cursorShape>
       </property>
       <property name="toolTip">
        <string>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;&lt;span style=&quot; font-size:8pt; font-weight:400;&quot;&gt;Set the maximum resolution for the resulting figures computed.&lt;/span&gt;&lt;/p&gt;&lt;p&gt;&lt;span style=&quot; font-size:8pt; font-weight:400; font-style:italic;&quot;&gt;Note that, to maintain aspect ratio of the original file, the final resolution might have &amp;quot;uncommon&amp;quot; sizes and might not match 1:1 the maximum resolution set (if the original image also had &amp;quot;uncommon&amp;quot; resolution).&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</string>
       </property>
       <property name="autoFillBackground">
        <bool>false</bool>
       </property>
       <property name="styleSheet">
        <string notr="true">QComboBox{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QListView{color:#FFFFFF;font:8pt&quot;Century Gothic&quot;;font-weight:bold;background-color:#969696;border:0px;}QListView{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QComboBox::drop-down{width:20px;border:5px;}QComboBox::down-arrow{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #FFFFFF;width:0.5px;height:1px;border-radius:2px;}QComboBox::down-arrow:hover{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #A21F27;width:0.5px;height:1px;border-radius:2px;}QAbstractItemView{border:2px solid #969696;selection-background-color:#2C53A1;}</string>
       </property>
       <property name="currentText">
        <string notr="true">  640 x 480</string>
       </property>
       <property name="currentIndex">
        <number>0</number>
       </property>
       <property name="sizeAdjustPolicy">
        <enum>QComboBox::AdjustToContentsOnFirstShow</enum>
       </property>
       <item>
        <property name="text">
         <string>  640 x 480</string>
        </property>
       </item>
       <item>
        <property name="text">
         <string notr="true"> 1280 x 720</string>
        </property>
       </item>
       <item>
        <property name="text">
         <string>1920 x 1080</string>
        </property>
       </item>
       <item>
        <property name="text">
         <string>2560 x 1440</string>
        </property>
       </item>
      </widget>
      <widget class="QLabel" name="label_13">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>290</y>
         <width>261</width>
         <height>31</height>
        </rect>
       </property>
       <property name="font">
        <font>
         <family>DejaVu Sans</family>
         <pointsize>6</pointsize>
         <weight>75</weight>
         <italic>false</italic>
         <bold>true</bold>
        </font>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{border:#969696;background-color:#2C53A1;border-radius:8px;color:#FFFFFF;font:6pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Obs.: Hover the mouse over the attribute
to see the help text</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QRadioButton" name="save_button">
       <property name="geometry">
        <rect>
         <x>50</x>
         <y>250</y>
         <width>91</width>
         <height>17</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QRadioButton{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}QRadioButton::indicator::checked{border:2px solid #969696;border-radius:5px;background-color:#2C53A1;width:8px;height:8px;}QRadioButton::indicator::unchecked{border:2px solid #969696;border-radius:5px;background-color:#969696;width:8px;height:8px;}</string>
       </property>
       <property name="text">
        <string>Save data</string>
       </property>
       <property name="checked">
        <bool>true</bool>
       </property>
       <property name="autoExclusive">
        <bool>true</bool>
       </property>
      </widget>
      <widget class="QRadioButton" name="only_plot_button">
       <property name="geometry">
        <rect>
         <x>170</x>
         <y>250</y>
         <width>81</width>
         <height>17</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QRadioButton{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}QRadioButton::indicator::checked{border:2px solid #969696;border-radius:5px;background-color:#2C53A1;width:8px;height:8px;}QRadioButton::indicator::unchecked{border:2px solid #969696;border-radius:5px;background-color:#969696;width:8px;height:8px;}</string>
       </property>
       <property name="text">
        <string>Only plot</string>
       </property>
       <property name="checked">
        <bool>false</bool>
       </property>
      </widget>
      <widget class="QLabel" name="label_14">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>210</y>
         <width>261</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{border:#969696;background-color:#2C53A1;border-radius:8px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">ADVANCED SETTINGS</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QComboBox" name="algo_type_combobox">
       <property name="geometry">
        <rect>
         <x>160</x>
         <y>60</y>
         <width>121</width>
         <height>16</height>
        </rect>
       </property>
       <property name="cursor">
        <cursorShape>PointingHandCursor</cursorShape>
       </property>
       <property name="styleSheet">
        <string notr="true">QComboBox{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QListView{color:#FFFFFF;font:8pt&quot;Century Gothic&quot;;font-weight:bold;background-color:#969696;border:0px;}QListView{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;background-color:#606060;border:0px;border-radius:6px;}QComboBox::drop-down{width:20px;border:5px;}QComboBox::down-arrow{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #FFFFFF;width:0.5px;height:1px;border-radius:2px;}QComboBox::down-arrow:hover{border-left:2px solid none;border-right:2px solid none;border-top:2px solid #A21F27;width:0.5px;height:1px;border-radius:2px;}QAbstractItemView{border:2px solid #969696;selection-background-color:#2C53A1;}</string>
       </property>
       <property name="currentText">
        <string notr="true">  DeepLabCut</string>
       </property>
       <property name="currentIndex">
        <number>0</number>
       </property>
       <item>
        <property name="text">
         <string notr="true">  DeepLabCut</string>
        </property>
       </item>
       <item>
        <property name="text">
         <string notr="true">      Bonsai</string>
        </property>
       </item>
      </widget>
      <widget class="QLabel" name="label_9">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>60</y>
         <width>112</width>
         <height>16</height>
        </rect>
       </property>
       <property name="maximumSize">
        <size>
         <width>114</width>
         <height>16</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Analysis Engine</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter</set>
       </property>
      </widget>
      <widget class="QLabel" name="label_11">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>160</y>
         <width>114</width>
         <height>16</height>
        </rect>
       </property>
       <property name="maximumSize">
        <size>
         <width>114</width>
         <height>16</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Task duration</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter</set>
       </property>
      </widget>
      <widget class="QLineEdit" name="crop_video_lineedit">
       <property name="geometry">
        <rect>
         <x>160</x>
         <y>160</y>
         <width>91</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;border-bottom-right-radius:6px;border-top-right-radius:6px}</string>
       </property>
       <property name="text">
        <string notr="true">300</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QCheckBox" name="crop_video_checkbox">
       <property name="geometry">
        <rect>
         <x>110</x>
         <y>270</y>
         <width>88</width>
         <height>17</height>
        </rect>
       </property>
       <property name="text">
        <string>Crop video</string>
       </property>
      </widget>
      <widget class="QLineEdit" name="crop_video_time_lineedit">
       <property name="geometry">
        <rect>
         <x>160</x>
         <y>180</y>
         <width>91</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;border-bottom-right-radius:6px;border-top-right-radius:6px}</string>
       </property>
       <property name="text">
        <string notr="true">15</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QLabel" name="label_15">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>180</y>
         <width>114</width>
         <height>16</height>
        </rect>
       </property>
       <property name="maximumSize">
        <size>
         <width>114</width>
         <height>16</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">How much to trim</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter</set>
       </property>
      </widget>
      <widget class="QLabel" name="label_16">
       <property name="geometry">
        <rect>
         <x>240</x>
         <y>180</y>
         <width>41</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{border:#969696;background-color:#606060;border-bottom-right-radius:6px;border-top-right-radius:6px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">sec</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QLabel" name="label_17">
       <property name="geometry">
        <rect>
         <x>240</x>
         <y>160</y>
         <width>41</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{border:#969696;background-color:#606060;border-bottom-right-radius:6px;border-top-right-radius:6px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">sec</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
     </widget>
    </widget>
    <widget class="QWidget" name="dlc_tab">
     <attribute name="title">
      <string>DEEPLABCUT</string>
     </attribute>
     <widget class="QFrame" name="network_settings">
      <property name="geometry">
       <rect>
        <x>10</x>
        <y>0</y>
        <width>291</width>
        <height>600</height>
       </rect>
      </property>
      <property name="sizePolicy">
       <sizepolicy hsizetype="Preferred" vsizetype="Preferred">
        <horstretch>0</horstretch>
        <verstretch>0</verstretch>
       </sizepolicy>
      </property>
      <property name="maximumSize">
       <size>
        <width>500</width>
        <height>600</height>
       </size>
      </property>
      <property name="styleSheet">
       <string notr="true">QLabel{}QCheckBox{color:#FFFFFF;font:6pt&quot;DejaVu Sans&quot;;font-weight:bold;}QCheckBox::indicator{width:15px;height:15px;background-color:#606060;border-radius:4px;}QCheckBox::indicator:checked{background-color:#A21F27;}</string>
      </property>
      <widget class="QLabel" name="network_tab_title">
       <property name="geometry">
        <rect>
         <x>20</x>
         <y>20</y>
         <width>261</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{border:#969696;background-color:#2C53A1;border-radius:8px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">NETWORK SETTINGS</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QLineEdit" name="config_path_lineedit">
       <property name="geometry">
        <rect>
         <x>0</x>
         <y>70</y>
         <width>285</width>
         <height>16</height>
        </rect>
       </property>
       <property name="minimumSize">
        <size>
         <width>285</width>
         <height>0</height>
        </size>
       </property>
       <property name="maximumSize">
        <size>
         <width>500</width>
         <height>20</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;border-bottom-right-radius:6px;border-top-right-radius:6px}</string>
       </property>
       <property name="text">
        <string notr="true"/>
       </property>
       <property name="frame">
        <bool>false</bool>
       </property>
       <property name="cursorPosition">
        <number>0</number>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
       <property name="readOnly">
        <bool>true</bool>
       </property>
       <property name="placeholderText">
        <string>%USERPROFILE%\path\to\project_folder\network_name_date\config.yaml</string>
       </property>
      </widget>
      <widget class="QLabel" name="config_path">
       <property name="geometry">
        <rect>
         <x>0</x>
         <y>50</y>
         <width>200</width>
         <height>16</height>
        </rect>
       </property>
       <property name="sizePolicy">
        <sizepolicy hsizetype="Fixed" vsizetype="Fixed">
         <horstretch>0</horstretch>
         <verstretch>0</verstretch>
        </sizepolicy>
       </property>
       <property name="minimumSize">
        <size>
         <width>200</width>
         <height>16</height>
        </size>
       </property>
       <property name="maximumSize">
        <size>
         <width>200</width>
         <height>16</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:6pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Path to the config.yaml file </string>
       </property>
       <property name="alignment">
        <set>Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter</set>
       </property>
      </widget>
      <widget class="QLineEdit" name="video_folder_lineedit">
       <property name="geometry">
        <rect>
         <x>0</x>
         <y>120</y>
         <width>285</width>
         <height>16</height>
        </rect>
       </property>
       <property name="minimumSize">
        <size>
         <width>285</width>
         <height>0</height>
        </size>
       </property>
       <property name="maximumSize">
        <size>
         <width>500</width>
         <height>20</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLineEdit{border:#969696;background-color:#606060;border-bottom-left-radius:6px;border-top-left-radius:6px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;border-bottom-right-radius:6px;border-top-right-radius:6px}</string>
       </property>
       <property name="text">
        <string notr="true"/>
       </property>
       <property name="frame">
        <bool>false</bool>
       </property>
       <property name="cursorPosition">
        <number>0</number>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
       <property name="readOnly">
        <bool>true</bool>
       </property>
       <property name="placeholderText">
        <string>%USERPROFILE%\path\to\video\folder\</string>
       </property>
      </widget>
      <widget class="QLabel" name="video_folder">
       <property name="geometry">
        <rect>
         <x>0</x>
         <y>97</y>
         <width>195</width>
         <height>16</height>
        </rect>
       </property>
       <property name="sizePolicy">
        <sizepolicy hsizetype="Fixed" vsizetype="Fixed">
         <horstretch>0</horstretch>
         <verstretch>0</verstretch>
        </sizepolicy>
       </property>
       <property name="minimumSize">
        <size>
         <width>130</width>
         <height>16</height>
        </size>
       </property>
       <property name="maximumSize">
        <size>
         <width>200</width>
         <height>42</height>
        </size>
       </property>
       <property name="font">
        <font>
         <family>DejaVu Sans</family>
         <pointsize>6</pointsize>
         <weight>75</weight>
         <italic>false</italic>
         <bold>true</bold>
        </font>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:6pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Folder containing the videos to analyze</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter</set>
       </property>
       <property name="wordWrap">
        <bool>true</bool>
       </property>
      </widget>
      <widget class="QLabel" name="folder_structure_check">
       <property name="geometry">
        <rect>
         <x>3</x>
         <y>142</y>
         <width>280</width>
         <height>16</height>
        </rect>
       </property>
       <property name="sizePolicy">
        <sizepolicy hsizetype="MinimumExpanding" vsizetype="MinimumExpanding">
         <horstretch>0</horstretch>
         <verstretch>0</verstretch>
        </sizepolicy>
       </property>
       <property name="minimumSize">
        <size>
         <width>280</width>
         <height>16</height>
        </size>
       </property>
       <property name="maximumSize">
        <size>
         <width>214</width>
         <height>16</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Check if the folder structure is correct</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QPushButton" name="folder_structure_check_button">
       <property name="geometry">
        <rect>
         <x>61</x>
         <y>160</y>
         <width>171</width>
         <height>31</height>
        </rect>
       </property>
       <property name="cursor">
        <cursorShape>PointingHandCursor</cursorShape>
       </property>
       <property name="styleSheet">
        <string notr="true">QPushButton{font:8pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:9pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}</string>
       </property>
       <property name="text">
        <string notr="true">CHECK</string>
       </property>
      </widget>
      <widget class="QLabel" name="get_frames_label">
       <property name="geometry">
        <rect>
         <x>0</x>
         <y>319</y>
         <width>161</width>
         <height>30</height>
        </rect>
       </property>
       <property name="sizePolicy">
        <sizepolicy hsizetype="MinimumExpanding" vsizetype="MinimumExpanding">
         <horstretch>0</horstretch>
         <verstretch>0</verstretch>
        </sizepolicy>
       </property>
       <property name="minimumSize">
        <size>
         <width>100</width>
         <height>30</height>
        </size>
       </property>
       <property name="maximumSize">
        <size>
         <width>214</width>
         <height>16</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Get a frame from every video</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter</set>
       </property>
       <property name="wordWrap">
        <bool>true</bool>
       </property>
      </widget>
      <widget class="QPushButton" name="get_frames_button">
       <property name="geometry">
        <rect>
         <x>160</x>
         <y>319</y>
         <width>131</width>
         <height>31</height>
        </rect>
       </property>
       <property name="cursor">
        <cursorShape>PointingHandCursor</cursorShape>
       </property>
       <property name="styleSheet">
        <string notr="true">QPushButton{font:14pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:8pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}</string>
       </property>
       <property name="text">
        <string notr="true">GET FRAME</string>
       </property>
      </widget>
      <widget class="QLabel" name="data_refining_title">
       <property name="geometry">
        <rect>
         <x>10</x>
         <y>259</y>
         <width>271</width>
         <height>16</height>
        </rect>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{border:#969696;background-color:#2C53A1;border-radius:8px;color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">DATA REFINING</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QLabel" name="extract_skeleton_title">
       <property name="geometry">
        <rect>
         <x>0</x>
         <y>287</y>
         <width>151</width>
         <height>16</height>
        </rect>
       </property>
       <property name="sizePolicy">
        <sizepolicy hsizetype="MinimumExpanding" vsizetype="MinimumExpanding">
         <horstretch>0</horstretch>
         <verstretch>0</verstretch>
        </sizepolicy>
       </property>
       <property name="minimumSize">
        <size>
         <width>100</width>
         <height>16</height>
        </size>
       </property>
       <property name="maximumSize">
        <size>
         <width>214</width>
         <height>16</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Extract skeleton data</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter</set>
       </property>
      </widget>
      <widget class="QPushButton" name="extract_skeleton_button">
       <property name="geometry">
        <rect>
         <x>160</x>
         <y>280</y>
         <width>131</width>
         <height>31</height>
        </rect>
       </property>
       <property name="cursor">
        <cursorShape>PointingHandCursor</cursorShape>
       </property>
       <property name="styleSheet">
        <string notr="true">QPushButton{font:14pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:8pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}</string>
       </property>
       <property name="text">
        <string notr="true">EXTRACT DATA</string>
       </property>
      </widget>
      <widget class="QPushButton" name="clear_unused_files_button">
       <property name="geometry">
        <rect>
         <x>160</x>
         <y>359</y>
         <width>131</width>
         <height>31</height>
        </rect>
       </property>
       <property name="cursor">
        <cursorShape>PointingHandCursor</cursorShape>
       </property>
       <property name="styleSheet">
        <string notr="true">QPushButton{font:14pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:8pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}</string>
       </property>
       <property name="text">
        <string notr="true">CLEANUP FOLDER</string>
       </property>
      </widget>
      <widget class="QLabel" name="clear_unused_files_title">
       <property name="geometry">
        <rect>
         <x>0</x>
         <y>365</y>
         <width>151</width>
         <height>16</height>
        </rect>
       </property>
       <property name="sizePolicy">
        <sizepolicy hsizetype="MinimumExpanding" vsizetype="MinimumExpanding">
         <horstretch>0</horstretch>
         <verstretch>0</verstretch>
        </sizepolicy>
       </property>
       <property name="minimumSize">
        <size>
         <width>100</width>
         <height>16</height>
        </size>
       </property>
       <property name="maximumSize">
        <size>
         <width>214</width>
         <height>16</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Remove unnecessary files</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter</set>
       </property>
      </widget>
      <widget class="QTextEdit" name="clear_unused_files_lineedit">
       <property name="geometry">
        <rect>
         <x>0</x>
         <y>396</y>
         <width>291</width>
         <height>191</height>
        </rect>
       </property>
       <property name="font">
        <font>
         <family>DejaVu Sans</family>
         <pointsize>8</pointsize>
         <weight>50</weight>
         <italic>false</italic>
         <bold>false</bold>
        </font>
       </property>
       <property name="styleSheet">
        <string notr="true">QTextEdit{color:#FFFFFF;font:8pt&quot;DejaVu Sans&quot;;font-weight:light}
QScrollBar:horizontal {
    background-color: #353535;
    height: 10px;
}

QScrollBar::handle:horizontal {
    background-color: #2C53A1;
    min-width: 20px;
}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background-color: #353535;
}

QScrollBar:vertical {
    background-color: #353535;
    width: 10px;
}

QScrollBar::handle:vertical {
    background-color: #2C53A1;
    min-height: 20px;
}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background-color: #353535;
}</string>
       </property>
       <property name="frameShape">
        <enum>QFrame::StyledPanel</enum>
       </property>
       <property name="frameShadow">
        <enum>QFrame::Plain</enum>
       </property>
       <property name="readOnly">
        <bool>true</bool>
       </property>
      </widget>
      <widget class="QLabel" name="dlc_video_analyze_title">
       <property name="geometry">
        <rect>
         <x>3</x>
         <y>197</y>
         <width>280</width>
         <height>16</height>
        </rect>
       </property>
       <property name="sizePolicy">
        <sizepolicy hsizetype="MinimumExpanding" vsizetype="MinimumExpanding">
         <horstretch>0</horstretch>
         <verstretch>0</verstretch>
        </sizepolicy>
       </property>
       <property name="minimumSize">
        <size>
         <width>280</width>
         <height>16</height>
        </size>
       </property>
       <property name="maximumSize">
        <size>
         <width>214</width>
         <height>16</height>
        </size>
       </property>
       <property name="styleSheet">
        <string notr="true">QLabel{color:#FFFFFF;font:7pt&quot;DejaVu Sans&quot;;font-weight:bold;}</string>
       </property>
       <property name="text">
        <string notr="true">Analyze videos using the network supplied</string>
       </property>
       <property name="alignment">
        <set>Qt::AlignCenter</set>
       </property>
      </widget>
      <widget class="QPushButton" name="dlc_video_analyze_button">
       <property name="geometry">
        <rect>
         <x>61</x>
         <y>216</y>
         <width>171</width>
         <height>31</height>
        </rect>
       </property>
       <property name="font">
        <font>
         <family>DejaVu Sans</family>
         <pointsize>9</pointsize>
         <weight>75</weight>
         <italic>false</italic>
         <bold>true</bold>
        </font>
       </property>
       <property name="cursor">
        <cursorShape>PointingHandCursor</cursorShape>
       </property>
       <property name="styleSheet">
        <string notr="true">QPushButton{font:14pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:9pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}</string>
       </property>
       <property name="text">
        <string notr="true">ANALYZE</string>
       </property>
      </widget>
      <widget class="QPushButton" name="get_config_path_button">
       <property name="geometry">
        <rect>
         <x>204</x>
         <y>44</y>
         <width>81</width>
         <height>20</height>
        </rect>
       </property>
       <property name="font">
        <font>
         <family>DejaVu Sans</family>
         <pointsize>8</pointsize>
         <weight>75</weight>
         <italic>false</italic>
         <bold>true</bold>
        </font>
       </property>
       <property name="cursor">
        <cursorShape>PointingHandCursor</cursorShape>
       </property>
       <property name="styleSheet">
        <string notr="true">QPushButton{font:10pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:8pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}</string>
       </property>
       <property name="text">
        <string notr="true">GET FILE</string>
       </property>
      </widget>
      <widget class="QPushButton" name="get_videos_path_button">
       <property name="geometry">
        <rect>
         <x>205</x>
         <y>94</y>
         <width>81</width>
         <height>20</height>
        </rect>
       </property>
       <property name="font">
        <font>
         <family>DejaVu Sans</family>
         <pointsize>8</pointsize>
         <weight>75</weight>
         <italic>false</italic>
         <bold>true</bold>
        </font>
       </property>
       <property name="cursor">
        <cursorShape>PointingHandCursor</cursorShape>
       </property>
       <property name="styleSheet">
        <string notr="true">QPushButton{font:6pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton{border:2px solid #A21F27;border-radius:8px;background-color:#2C53A1;color:#FFFFFF;font:8pt&quot;DejaVu Sans&quot;;font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;border-radius:8px;background-color:#A21F27;color:#FFFFFF;}</string>
       </property>
       <property name="text">
        <string notr="true">GET FOLDER</string>
       </property>
      </widget>
     </widget>
    </widget>
   </widget>
   <widget class="QLabel" name="behavython_logo">
    <property name="geometry">
     <rect>
      <x>60</x>
      <y>10</y>
      <width>231</width>
      <height>61</height>
     </rect>
    </property>
    <property name="text">
     <string/>
    </property>
    <property name="pixmap">
     <pixmap>logo/logo.png</pixmap>
    </property>
    <property name="scaledContents">
     <bool>true</bool>
    </property>
    <property name="alignment">
     <set>Qt::AlignCenter</set>
    </property>
   </widget>
  </widget>
 </widget>
 <customwidgets>
  <customwidget>
   <class>plot_viewer</class>
   <extends>QWidget</extends>
   <header location="global">behavython_plot_widget</header>
   <container>1</container>
  </customwidget>
 </customwidgets>
 <resources/>
 <connections/>
</ui>

"""

QPyDesignerCustomWidgetCollection.registerCustomWidget(
    plot_viewer, module="behavython_plot_widget", tool_tip=TOOLTIP, xml=DOM_XML
)