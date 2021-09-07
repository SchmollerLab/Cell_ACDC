import os, sys

import pandas as pd
import numpy as np

from PyQt5.QtGui import QIcon, QFont, QFontMetrics, QPixmap
from PyQt5.QtCore import Qt, QSize, QEvent
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QTextEdit, QPushButton,
    QListWidget, QListWidgetItem, QCheckBox, QFrame, QStyleFactory,
    QLabel, QPushButton, QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator,
    QScrollArea, QComboBox, QHBoxLayout, QToolButton
)

script_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.dirname(script_path)
sys.path.append(src_path)

import gui
from prompts import QPushButton

# NOTE: Enable icons
import qrc_resources

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.yeastacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

class welcomeWin(QWidget):
    def __init__(self, parent=None, mainWin=None, app=None):
        self.parent = parent
        self.mainWin = mainWin
        self.app = app
        super().__init__(parent)
        self.setWindowTitle('Welcome')
        self.setWindowIcon(QIcon(":assign-motherbud.svg"))
        self.loadSettings()

        self.itemsDict = {}

        # Initialize Permanent items
        self.mainLayout = QGridLayout()
        self.addtreeSelector()
        self.addShowGuideCheckbox()

        # Create all pages of the guide as one frame for each page
        self.addWelcomePage()
        self.addQuickStartPage()

        self.setStyleSheet(
            """
            QTreeWidget::item:hover {background-color:#E6E6E6;}
            QTreeWidget::item:selected {background-color:#CFEB9B;}
            QTreeWidget::item:selected {color:black;}
            QTreeView {
                selection-background-color: #CFEB9B;
                selection-color: white;
                show-decoration-selected: 1;
            }
            QTreeWidget::item {padding: 5px;}
            QPushButton {
                font-size:11pt
                font-family:"Ubuntu"
            }
            """
        )

        self.setLayout(self.mainLayout)

    def loadSettings(self):
        temp_path = os.path.join(src_path, 'temp')
        csv_path = os.path.join(temp_path, 'settings.csv')
        if os.path.exists(csv_path):
            self.df_settings = pd.read_csv(csv_path, index_col='setting')
            if 'showWelcomeGuide' not in self.df_settings.index:
                self.df_settings.at['showWelcomeGuide', 'value'] = 'True'
        else:
            idx = ['showWelcomeGuide']
            values = ['True']
            self.df_settings = pd.DataFrame({'setting': idx,
                                             'value': values}
                                           ).set_index('setting')
            self.df_settings.to_csv(csv_path)
        self.df_settings_path = csv_path

    def addtreeSelector(self):
        treeSelector = QTreeWidget()

        treeSelector.setFrameStyle(QFrame.NoFrame)

        self.welcomeItem = QTreeWidgetItem(treeSelector)
        self.welcomeItem.setIcon(0, QIcon(':home.svg'))
        self.welcomeItem.setText(0, 'Welcome')
        treeSelector.addTopLevelItem(self.welcomeItem)

        self.quickStartItem = QTreeWidgetItem(treeSelector)
        self.quickStartItem.setIcon(0, QIcon(':quickStart.svg'))
        self.quickStartItem.setText(0, 'Quick Start')
        treeSelector.addTopLevelItem(self.quickStartItem)

        self.settingsItem = QTreeWidgetItem(treeSelector)
        self.settingsItem.setIcon(0, QIcon(':cog.svg'))
        self.settingsItem.setText(0, 'Settings')
        treeSelector.addTopLevelItem(self.settingsItem)

        self.manualItem = QTreeWidgetItem(treeSelector)
        self.manualItem.setIcon(0, QIcon(':book.svg'))
        self.manualItem.setText(0, 'User Manual')
        treeSelector.addTopLevelItem(self.manualItem)
        self.manualDataPrepItem = QTreeWidgetItem(self.manualItem)
        self.manualDataPrepItem.setText(0, '    Data Prep module')
        self.manualItem.addChild(self.manualDataPrepItem)
        self.manualSegmItem = QTreeWidgetItem(self.manualItem)
        self.manualSegmItem.setText(0, '    Segmentation module')
        self.manualItem.addChild(self.manualDataPrepItem)
        self.manualGUIItem = QTreeWidgetItem(self.manualItem)
        self.manualGUIItem.setText(0, '    GUI: segmentation and tracking')
        self.manualItem.addChild(self.manualDataPrepItem)

        self.contributeItem = QTreeWidgetItem(treeSelector)
        self.contributeItem.setIcon(0, QIcon(':contribute.svg'))
        self.contributeItem.setText(0, 'Contribute')
        treeSelector.addTopLevelItem(self.contributeItem)
        # treeSelector.setSpacing(3)

        # treeSelector.setCurrentItem(self.welcomeItem, 0)
        treeSelector.setHeaderHidden(True)

        self.treeSelector = treeSelector
        self.mainLayout.addWidget(treeSelector, 0, 0)

        treeSelector.currentItemChanged.connect(self.treeItemChanged)

    def treeItemChanged(self, currentItem, prevItem=None):
        currentItem.setExpanded(True)
        itemText = currentItem.text(0)
        for key, frame in self.itemsDict.items():
            if key == itemText:
                frame.show()
            else:
                frame.hide()


    def addWelcomePage(self):
        self.welcomeFrame = QFrame(self)
        welcomeLayout = QGridLayout()

        welcomeTextWidget = QLabel()

        # welcomeTextWidget = QTextEdit()
        # welcomeTextWidget.setReadOnly(True)
        # welcomeTextWidget.setFrameStyle(QFrame.NoFrame)
        # welcomeTextWidget.viewport().setAutoFillBackground(False)

        htmlTxt = (
        """
        <html>
        <head>
        <title></title>
        <style type="text/css">
        blockquote {
         margin: 5;
         padding: 0;
        }
        </style>
        </head>
        <body>
        <blockquote>
        <p style="font-size:18pt; font-family:ubuntu">
            <b>Welcome to Yeast-ACDC</b>
        </p>
        <p style="font-size:12pt; font-family:ubuntu">
            Welcome to your new image analysis tool!
        </p>
        <p style="font-size:12pt; font-family:ubuntu">
            Yeast-ACDC is open-source software for
            <b>segmentation</b>, <b>tracking,</b> and<br>
            <b>cell cycle annotation</b> of microscopy imaging data.
        </p>
        <p style="font-size:12pt; font-family:ubuntu">
            For more info see our paper here or navigate the menus on the left.
        </p>
        <p style="font-size:12pt; font-family:ubuntu">
            If it is your <b>first time here</b> we reccomend reading the
            <a href=\"quickStart">Quick Start guide</a>
            and/or the
            <a href=\"userManual">User Manual</a>.
        </p>
        <p style="font-size:12pt; font-family:ubuntu; line-height:1.2">
            Alternatively, you can launch a <b>Wizard</b> that will guide you through the
            <b>conversion</b> of<br> one or more <b>raw microscopy</b> files into the required structure
            or you can test the main GUI
            <br>
        </p>
        </blockquote>
        </body>
        </html>
        """
        )

        # welcomeTextWidget.setHtml(htmlTxt)
        welcomeTextWidget.setText(htmlTxt)
        welcomeTextWidget.linkActivated.connect(self.linkActivated_cb)

        welcomeLayout.addWidget(welcomeTextWidget, 0, 0, 1, 5,
                                alignment=Qt.AlignTop)

        startWizardButton = QPushButton(' Launch Wizard')
        startWizardButton.setIcon(QIcon(':wizard.svg'))

        welcomeLayout.addWidget(startWizardButton, 1, 0)

        testMyImageButton = QPushButton(' Test segmentation with my image')
        testMyImageButton.setIcon(QIcon(':image.svg'))
        testMyImageButton.clicked.connect(self.openGUIsingleImage)

        welcomeLayout.addWidget(testMyImageButton, 1, 1)

        testTimeLapseExample = QPushButton(
            text='Download and test with a time-lapse example')
        testTimeLapseExample.setIcon(QIcon(':download.svg'))
        testTimeLapseExample.clicked.connect(self.openGUItimeLapseExample)

        welcomeLayout.addWidget(testTimeLapseExample, 1, 2)

        test3DzStackImage = QPushButton(
            text='Download and test with a 3D z-stack example')
        test3DzStackImage.setIcon(QIcon(':download.svg'))
        test3DzStackImage.clicked.connect(self.openGUIzStackExample)

        welcomeLayout.addWidget(test3DzStackImage, 1, 3)

        welcomeLayout.setRowStretch(2, 1)
        welcomeLayout.setColumnStretch(5, 1)

        self.welcomeFrame.setLayout(welcomeLayout)
        self.mainLayout.addWidget(self.welcomeFrame, 0, 1)
        self.itemsDict[self.welcomeItem.text(0)] = self.welcomeFrame

    def addQuickStartPage(self):
        self.QuickStartViewBox = QWidget(self)

        self.QSscrollArea = QScrollArea()
        self.QSscrollArea.setWidgetResizable(True)
        self.QSscrollArea.setFrameStyle(QFrame.NoFrame)
        self.QSscrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.QSscrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        QuickStartLayout = QGridLayout()

        fs = 11 # font size

        row = 0
        QuickStartTextWidget = QLabel()

        htmlHead = (
        """
        <html>
        <head>

        <style type="text/css">
        blockquote {
         margin: 5;
         padding: 0;
        }
        li{
          margin: 7px 0;
        }
        </style>

        </head>
        """
        )

        htmlTxt = (
        f"""
        {htmlHead}
        <body>
        <blockquote>
        <p>
            <span style="font-size:14pt; font-family:ubuntu">
                <b>NOTE: This Quick Start is NOT an exhaustive manual.</b><br>
            </span>
            <span style="font-size:12pt; font-family:ubuntu">
                It is only meant to get you started as quickly as possible.<br>
            </span>
        </p>
        </blockquote>
        </body>
        </html>
        """
        )

        QuickStartTextWidget.setText(htmlTxt)
        QuickStartTextWidget.linkActivated.connect(self.linkActivated_cb)

        QuickStartLayout.addWidget(QuickStartTextWidget, row, 0,
                                   alignment=Qt.AlignTop)

        row += 1
        QuickStartTextWidget = QLabel()
        htmlTxt = (
        f"""
        {htmlHead}
        <body>
        <blockquote>
        <p style="font-size:{fs}pt; font-family:ubuntu">
            Yeast-ACDC is made of three main modules:
            <ul>
                <li>
                    <b>Data Prep</b>: used for selection of z-slice of z-projection,
                    and/or cropping images.
                    <a href=\"DataPrepMore">More details...</a>
                </li>
                <li>
                    <b>Segmentation</b>: automatic segmentation of multiple Positions
                    using YeaZ or Cellpose.
                    <a href=\"segmMore">More details...</a>
                </li>
                <li>
                    <b>GUI</b>: used for visualizing and/or correcting results of the
                    segmentation module<br> or segmenting one image/frame
                    at the time.
                    <a href=\"guiMore">More details...</a>
                </li>
            </ul>
            <br>
        </p>
        </blockquote>
        </body>
        </html>
        """
        )

        QuickStartTextWidget.setText(htmlTxt)
        QuickStartTextWidget.linkActivated.connect(self.linkActivated_cb)

        QuickStartLayout.addWidget(QuickStartTextWidget, row, 0,
                                   alignment=Qt.AlignTop)

        row += 1
        QS_tipTxtLabel = QLabel()

        htmlTxt = (
        f"""
        {htmlHead}
        <body>
        <blockquote>
        <p style="font-size:{fs}pt; font-family:ubuntu">
            <b>GUI tips:</b>
            <ul>
                <li>
                    Most of the <b>functions</b> are available from the <b>toolbar</b>
                    on the top of the window:
                </li>
        </p>
        </blockquote>
        """
        )

        QS_tipTxtLabel.setText(htmlTxt)
        QuickStartLayout.addWidget(QS_tipTxtLabel, row, 0,
                                   alignment=Qt.AlignTop)

        row += 1
        pixmap = QPixmap(os.path.join(script_path, 'images', 'toolbar.png'))
        label = QLabel()
        # padding: top, left, bottom, right
        label.setStyleSheet("padding:5px 0px 10px 40px;")
        label.setPixmap(pixmap)
        QuickStartLayout.addWidget(label, row, 0, alignment=Qt.AlignTop)

        row += 1
        QS_tipTxtLabel = QLabel()

        htmlTxt = (
        f"""
        {htmlHead}
        <body>
        <blockquote>
        <p style="font-size:{fs}pt; font-family:ubuntu">
            <ul>
                <li>
                    Activate a function with a <b>SINGLE-click</b> on the button.
                </li>
            </ul>
        </p>
        </blockquote>
        """
        )

        QS_tipTxtLabel.setText(htmlTxt)
        QS_tipTxtLabel.setStyleSheet('padding-bottom: 10px')
        QuickStartLayout.addWidget(QS_tipTxtLabel, row, 0,
                                   alignment=Qt.AlignTop)

        row += 1
        QS_tipTxtLabel = QLabel()

        htmlTxt = (
        f"""
        {htmlHead}
        <body>
        <blockquote>
        <p style="font-size:{fs}pt; font-family:ubuntu">
            <ul>
                <li>
                    When you <b>hover a button</b> with mouse cursor you get a <b>tool tip</b>
                    on how to use that function:
                </li>
            </ul>
        </p>
        </blockquote>
        """
        )

        QS_tipTxtLabel.setText(htmlTxt)
        QuickStartLayout.addWidget(QS_tipTxtLabel, row, 0,
                                   alignment=Qt.AlignTop)

        row += 1
        pixmap = QPixmap(os.path.join(script_path, 'images', 'toolTip.png'))
        label = QLabel()
        label.setStyleSheet("padding:5px 0px 10px 40px;")
        label.setPixmap(pixmap)
        QuickStartLayout.addWidget(label, row, 0, alignment=Qt.AlignTop)

        row += 1
        QS_tipTxtLabel = QLabel()

        htmlTxt = (
        f"""
        {htmlHead}
        <body>
        <blockquote>
        <p style="font-size:{fs}pt; font-family:ubuntu">
            <ul>
                <li>
                    The Tool tip will tell you whether you need <b>RIGHT-click</b>
                    or <b>LEFT-click</b> for that function.
                </li>
            </ul>
        </p>
        </blockquote>
        """
        )

        QS_tipTxtLabel.setText(htmlTxt)
        QS_tipTxtLabel.setStyleSheet('padding-bottom: 10px')
        QuickStartLayout.addWidget(QS_tipTxtLabel, row, 0,
                                   alignment=Qt.AlignTop)

        row += 1
        QS_tipTxtLabel = QLabel()

        htmlTxt = (
        f"""
        {htmlHead}
        <body>
        <blockquote>
        <p style="font-size:{fs}pt; font-family:ubuntu">
            <ul>
                <li>
                    Functions that are <b>NOT activated by a toolbar button</b>:
                    <blockquote>
                    &nbsp;&nbsp; - Scrolling wheel button --> <b>delete</b> segmented object<br>
                    &nbsp;&nbsp; - "H" key --> <b>automatic zoom</b> on the segmented objects<br>
                    &nbsp;&nbsp; - "Ctrl+P" --> Visualize cell cycle annotations in a table
                    </blockquote>
                </li>
            </ul>
        </p>
        </blockquote>
        """
        )

        QS_tipTxtLabel.setText(htmlTxt)
        QS_tipTxtLabel.setStyleSheet('padding-bottom: 10px')
        QuickStartLayout.addWidget(QS_tipTxtLabel, row, 0,
                                   alignment=Qt.AlignTop)

        row += 1
        QS_tipTxtLabel = QLabel()

        htmlTxt = (
        f"""
        {htmlHead}
        <body>
        <blockquote>
        <p style="font-size:{fs}pt; font-family:ubuntu">
            <ul>
                <li>
                    To segment in the GUI use the "Segment" menu.
                </li>
            </ul>
        </p>
        </blockquote>
        """
        )

        QS_tipTxtLabel.setText(htmlTxt)
        QS_tipTxtLabel.setStyleSheet('padding-bottom: 10px')
        QuickStartLayout.addWidget(QS_tipTxtLabel, row, 0,
                                   alignment=Qt.AlignTop)

        row += 1
        QS_tipTxtLabel = QLabel()


        htmlTxt = (
        f"""
        {htmlHead}
        <body>
        <blockquote>
        <p style="font-size:{fs}pt; font-family:ubuntu">
            <ul>
                <li>
                    For <b>time-lapse data</b> the default mode is "Viewer".
                    Toggle the other modes with the following selector on the toolbar:
                </li>
            </ul>
        </p>
        </blockquote>
        """
        )

        QS_tipTxtLabel.setText(htmlTxt)
        QuickStartLayout.addWidget(QS_tipTxtLabel, row, 0,
                                   alignment=Qt.AlignTop)

        row += 1
        modeComboBox = QComboBox()
        modeComboBox.addItems(['Segmentation and Tracking',
                                'Cell cycle analysis',
                                'Viewer'])
        modeComboBox.setCurrentText('Viewer')
        modeComboBox.setFocusPolicy(Qt.StrongFocus)
        modeComboBox.installEventFilter(self)
        modeComboBoxLabel = QLabel('    Mode: ')
        layout = QHBoxLayout()
        layout.addWidget(modeComboBoxLabel)
        layout.addWidget(modeComboBox)
        layout.addStretch(1)
        # Left, top, right, bottom
        layout.setContentsMargins(40, 5, 0, 10)
        QuickStartLayout.addLayout(layout, row, 0)

        row += 1
        QS_tipTxtLabel = QLabel()
        htmlTxt = (
        f"""
        {htmlHead}
        <body>
        <blockquote>
        <p style="font-size:{fs}pt; font-family:ubuntu">
            <ul>
                <li>
                    To navigate frames of <b>time-lapse data</b> use left and
                    right arrow on the keyboard OR the slider below left image.
                </li>
            </ul>
        </p>
        </blockquote>
        """
        )

        QS_tipTxtLabel.setText(htmlTxt)
        QS_tipTxtLabel.setStyleSheet('padding-bottom: 10px')
        QuickStartLayout.addWidget(QS_tipTxtLabel, row, 0,
                                   alignment=Qt.AlignTop)


        row += 1
        QS_tipTxtLabel = QLabel()
        htmlTxt = (
        f"""
        {htmlHead}
        <body>
        <blockquote>
        <p style="font-size:{fs}pt; font-family:ubuntu">
            <ul>
                <li>
                    To visualize the frames of time-lapse data in a second
                    window click on the "Slideshow" button on the toolbar:
                </li>
            </ul>
        </p>
        </blockquote>
        """
        )
        QS_tipTxtLabel.setText(htmlTxt)
        QS_tipTxtLabel.setStyleSheet('padding-bottom: 8px')

        viewerButton = QToolButton()
        viewerButton.setIcon(QIcon(':eye-plus.svg'))
        viewerButton.setIconSize(QSize(24, 24));

        layout = QHBoxLayout()
        layout.addWidget(QS_tipTxtLabel, alignment=Qt.AlignBottom)
        layout.addWidget(viewerButton)
        layout.addStretch(1)
        QuickStartLayout.addLayout(layout, row, 0)



        row += 1
        QS_tipTxtLabel = QLabel()

        htmlTxt = (
        f"""
        {htmlHead}
        <body>
        <blockquote>
        <p style="font-size:{fs}pt; font-family:ubuntu">
            <ul>
                <li>
                    Settings such as <b>Font Size</b>, <b>Overlay color</b> can be
                    edited from the "Edit" menu.
                </li>
            </ul>
            <br>
        </p>
        </blockquote>
        """
        )

        QS_tipTxtLabel.setText(htmlTxt)
        QS_tipTxtLabel.setStyleSheet('padding-top: 2px')
        QuickStartLayout.addWidget(QS_tipTxtLabel, row, 0,
                                   alignment=Qt.AlignTop)

        row +=1
        QuickStartTextWidget = QLabel()
        htmlTxt = (
        f"""
        {htmlHead}
        <body>
        <blockquote>
        <p style="font-size:{fs}pt; font-family:ubuntu; line-height:1.2">
            &nbsp;&nbsp;If you feel ready now you can start <b>testing out the main GUI</b>
            with one of your images<br>
            &nbsp;&nbsp;or test with one of our example images:
        </p>
        </blockquote>
        </body>
        </html>
        """
        )

        QuickStartTextWidget.setText(htmlTxt)
        QuickStartLayout.addWidget(QuickStartTextWidget, row, 0,
                                   alignment=Qt.AlignTop)

        row += 1
        layout = QHBoxLayout()
        testMyImage = QPushButton(
            text='Test with my image')
        testMyImage.setIcon(QIcon(':image.svg'))
        layout.addWidget(testMyImage)
        testMyImage.clicked.connect(self.openGUIsingleImage)

        testTimeLapseExample = QPushButton(
            text='Download and test with a time-lapse example')
        testTimeLapseExample.setIcon(QIcon(':download.svg'))
        layout.addWidget(testTimeLapseExample)
        testMyImage.clicked.connect(self.openGUItimeLapseExample)

        test3DzStackImage = QPushButton(
            text='Download and test with a 3D z-stack example')
        test3DzStackImage.setIcon(QIcon(':download.svg'))
        layout.addWidget(test3DzStackImage)
        test3DzStackImage.clicked.connect(self.openGUIzStackExample)

        layout.addStretch(1)

        layout.setContentsMargins(15, 0, 0, 10)
        QuickStartLayout.addLayout(layout, row, 0)

        QuickStartLayout.setRowStretch(QuickStartLayout.rowCount(), 1)

        self.QuickStartLayout = QuickStartLayout

        self.QuickStartViewBox.setLayout(QuickStartLayout)
        self.QSscrollArea.setWidget(self.QuickStartViewBox)

        self.mainLayout.addWidget(self.QSscrollArea, 0, 1)
        self.itemsDict[self.quickStartItem.text(0)] = self.QSscrollArea


    def linkActivated_cb(self, link):
        if link == 'DataPrepMore':
            pass
        elif link == 'segmMore':
            pass
        elif link == 'guiMore':
            pass
        elif link == 'quickStart':
            self.showPage(self.quickStartItem)
        elif link == 'userManual':
            self.showPage(self.manualItem)

    def addShowGuideCheckbox(self):
        checkBox = QCheckBox('Show Welcome Guide when opening Yeast-ACDC')
        checked = self.df_settings.at['showWelcomeGuide', 'value'] == 'True'
        checkBox.setChecked(checked)
        self.mainLayout.addWidget(checkBox, 1, 1, alignment=Qt.AlignRight)

        checkBox.stateChanged.connect(self.showWelcomeGuideCheckBox_cb)

    def showWelcomeGuideCheckBox_cb(self, state):
        self.df_settings.at['showWelcomeGuide', 'value'] = str(state==2)
        self.saveSettings()

    def saveSettings(self):
        self.df_settings.to_csv(self.df_settings_path)


    def openGUIsingleImage(self):
        if self.mainWin is not None:
            self.mainWin.launchGui()
            self.mainWin.guiWin.openFile()
        else:
            self.guiWin = gui.guiWin(self.app)
            self.guiWin.showAndSetSize()
            self.guiWin.openFile()


    def showAndSetSize(self):
        self.show()
        font = QFont()
        font.setPointSize(10)
        font.setFamily('Ubuntu')
        self.treeSelector.setFont(font)
        w = 0
        it = QTreeWidgetItemIterator(self.treeSelector)
        while it:
            currentItem = it.value()
            if currentItem is None:
                break
            w += QFontMetrics(currentItem.font(0)).maxWidth()+5
            # w = QFontMetrics()
            it += 1
        self.treeSelector.setFixedWidth(w)

        # Resize to remove need for horizontal scroolbar

        QSviewBoxWidth = self.QuickStartViewBox.minimumSizeHint().width()
        w = (self.QSscrollArea.pos().x() + QSviewBoxWidth
             + 5*self.QuickStartLayout.columnCount() + 20)

        winGeometry = self.geometry()
        l, t, h = winGeometry.left(), winGeometry.top(), winGeometry.height()
        w0 = winGeometry.width()
        Dw = w - w0
        Dh = 1.5
        self.setGeometry(l-int(Dw/2), int(t-(1-Dh)), w, int(h*Dh))

    def openGUItimeLapseExample(self, checked=False):
        pass

    def openGUIzStackExample(self, checked=False):
        pass

    def showPage(self, currentItem):
        self.treeSelector.setCurrentItem(currentItem, 0)

    def eventFilter(self, object, event):
        # Disable wheel scroll on widgets to allow scroll only on scrollarea
        if event.type() == QEvent.Wheel:
            event.ignore()
            return True
        return False

if __name__ == '__main__':
    app = QApplication([])
    win = welcomeWin(app=app)
    win.showAndSetSize()
    win.showPage(win.quickStartItem)
    app.setStyle(QStyleFactory.create('Fusion'))
    sys.exit(app.exec_())
