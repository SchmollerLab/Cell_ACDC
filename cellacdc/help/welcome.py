import os, sys
from functools import partial
import webbrowser

import pandas as pd
import numpy as np

from qtpy.QtGui import (
    QIcon, QFont, QFontMetrics, QPixmap, QPalette, QColor
)
from qtpy.QtCore import (
    Qt, QSize, QEvent, Signal, QObject, QThread, QTimer
)
from qtpy.QtWidgets import (
    QApplication, QWidget, QGridLayout, QTextEdit, QPushButton,
    QListWidget, QListWidgetItem, QCheckBox, QFrame, QStyleFactory,
    QLabel, QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator,
    QScrollArea, QComboBox, QHBoxLayout, QToolButton, QMainWindow,
    QProgressBar, QAction
)

script_path = os.path.dirname(os.path.realpath(__file__))

from .. import gui, dataStruct, myutils, cite_url, html_utils, urls, widgets
from .. import _palettes

# NOTE: Enable icons
from .. import qrc_resources, cellacdc_path, settings_folderpath

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

class downloadWorker(QObject):
    finished = Signal()
    progress = Signal(int, int)

    def __init__(self, which):
        QObject.__init__(self)
        self.which = which

    def run(self):
        self.exp_path = myutils.download_examples(
            self.which, progress=self.progress
        )
        self.finished.emit()

class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)


class welcomeWin(QWidget):
    def __init__(self, parent=None, mainWin=None, app=None):
        self.parent = parent
        self.mainWin = mainWin
        self.app = app
        super().__init__(parent)
        self.setWindowTitle('Welcome')
        self.setWindowIcon(QIcon(":icon.ico"))
        self.loadSettings()

        self.itemsDict = {}
        self.QPbar = None

        # Initialize Permanent items
        self.mainLayout = QGridLayout()
        self.addtreeSelector()
        self.addShowGuideCheckbox()

        # Create all pages of the guide as one frame for each page
        self.addWelcomePage()
        self.addQuickStartPage()
        self.addManualPage()
        self.addContributePage()

        self.setStyleSheet(
            """
            QTreeWidget::item:hover {background-color:#E6E6E6;}
            QTreeWidget::item:hover {color:black;}
            QTreeWidget::item:selected {background-color:#CFEB9B;}
            QTreeWidget::item:selected {color:black;}
            QTreeView {
                selection-background-color: #CFEB9B;
                show-decoration-selected: 1;
            }
            QTreeWidget::item {padding: 5px;}
            QPushButton {
                font-size:13px
                font-family:"Ubuntu"
            }
            """
        )

        self.setLayout(self.mainLayout)
        # self.setDebuggingTools()

    def setDebuggingTools(self):
        self.debugButton = QPushButton('debug')
        self.debugButton.clicked.connect(self.debug)
        self.mainLayout.addWidget(self.debugButton, 2, 0)
        # self.debugAction.hide()

    def loadSettings(self):
        csv_path = os.path.join(settings_folderpath, 'settings.csv')
        if os.path.exists(csv_path):
            self.df_settings = pd.read_csv(csv_path, index_col='setting')
            if 'showWelcomeGuide' not in self.df_settings.index:
                self.df_settings.at['showWelcomeGuide', 'value'] = 'Yes'
        else:
            idx = ['showWelcomeGuide']
            values = ['Yes']
            self.df_settings = pd.DataFrame({'setting': idx,
                                             'value': values}
                                           ).set_index('setting')
            self.df_settings.to_csv(csv_path)
        self.df_settings_path = csv_path

    def addtreeSelector(self):
        treeSelector = QTreeWidget()

        treeSelector.setFrameStyle(QFrame.Shape.NoFrame)

        self.welcomeItem = QTreeWidgetItem(treeSelector)
        self.welcomeItem.setIcon(0, QIcon(':home.svg'))
        self.welcomeItem.setText(0, 'Welcome')
        treeSelector.addTopLevelItem(self.welcomeItem)

        self.quickStartItem = QTreeWidgetItem(treeSelector)
        self.quickStartItem.setIcon(0, QIcon(':quickStart.svg'))
        self.quickStartItem.setText(0, 'Quick Start')
        treeSelector.addTopLevelItem(self.quickStartItem)

        # self.settingsItem = QTreeWidgetItem(treeSelector)
        # self.settingsItem.setIcon(0, QIcon(':cog.svg'))
        # self.settingsItem.setText(0, 'Settings')
        # treeSelector.addTopLevelItem(self.settingsItem)

        self.manualItem = QTreeWidgetItem(treeSelector)
        self.manualItem.setIcon(0, QIcon(':book.svg'))
        # textLabel = QLabel()
        # textLabel.setText("""
        # <p style="font-size:13px; font-family:ubuntu">
        #     User Manual
        # </p>
        # """)
        self.manualItem.setText(0, 'User Manual')
        treeSelector.addTopLevelItem(self.manualItem)
        # treeSelector.setItemWidget(self.manualItem, 0, textLabel)


        # self.manualDataPrepItem = QTreeWidgetItem(self.manualItem)
        # self.manualDataPrepItem.setText(0, '    Data Prep module')
        # self.manualItem.addChild(self.manualDataPrepItem)
        # self.manualSegmItem = QTreeWidgetItem(self.manualItem)
        # self.manualSegmItem.setText(0, '    Segmentation module')
        # self.manualItem.addChild(self.manualSegmItem)
        # self.manualGUIItem = QTreeWidgetItem(self.manualItem)
        # self.manualGUIItem.setText(0, '    GUI: segmentation and tracking')
        # self.manualItem.addChild(self.manualGUIItem)

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
        self.welcomeLayout = welcomeLayout

        welcomeTextWidget = QLabel()

        # welcomeTextWidget = QTextEdit()
        # welcomeTextWidget.setReadOnly(True)
        # welcomeTextWidget.setFrameStyle(QFrame.Shape.NoFrame)
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
        <p style="font-size:20px; font-family:ubuntu">
            <b>Welcome to Cell-ACDC</b>
        </p>
        <p style="font-size:15px; font-family:ubuntu">
            Welcome to your new image analysis tool!
        </p>
        <p style="font-size:15px; font-family:ubuntu">
            Cell-ACDC is open-source software for
            <b>segmentation</b>, <b>tracking,</b> and<br>
            <b>cell cycle annotation</b> of microscopy imaging data.
        </p>
        <p style="font-size:15px; font-family:ubuntu">
            You can check out our <a href=\"paper">publication</a>
            or Twitter <a href=\"tweet">thread</a>.
        </p>
        <p style="font-size:15px; font-family:ubuntu">
            If it is your <b>first time here</b> we recommend reading the
            <a href=\"quickStart">Quick Start guide</a>
            and/or the
            <a href=\"userManual">User Manual</a>.
        </p>
        <p style="font-size:15px; font-family:ubuntu; line-height:1.2">
            Alternatively, you can launch a <b>Wizard</b> that will guide you through the
            <b>conversion</b> of<br> one or more <b>raw microscopy</b> files into the required structure
            or you can test the main GUI
        </p>
        <p style="font-size:13px; font-family:ubuntu; line-height:1.2">
            <i>Note that if you are looking for the <b>main launcher</b> to launch any
            of the Cell-ACDC modules you simply need to <b>close this welcome guide</b>.
            </i>
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
        startWizardButton.clicked.connect(self.launchDataStruct)

        welcomeLayout.addWidget(startWizardButton, 1, 0)

        testMyImageButton = QPushButton(' Test segmentation with my image/video')
        testMyImageButton.setIcon(QIcon(':image.svg'))
        testMyImageButton.clicked.connect(self.openGUIsingleImage)

        welcomeLayout.addWidget(testMyImageButton, 1, 1)

        testTimeLapseButton = QPushButton(
            text='Download and test with a time-lapse example')
        testTimeLapseButton.setIcon(QIcon(':download.svg'))
        testTimeLapseButton.clicked.connect(self.testTimeLapseExample)

        welcomeLayout.addWidget(testTimeLapseButton, 1, 2)

        test3DzStackButton = QPushButton(
            text='Download and test with a 3D z-stack example')
        test3DzStackButton.setIcon(QIcon(':download.svg'))
        test3DzStackButton.clicked.connect(self.test3DzStacksExample)

        welcomeLayout.addWidget(test3DzStackButton, 1, 3)

        self.infoTextWidget = QLabel()
        welcomeLayout.addWidget(self.infoTextWidget, 2, 0, 1, 5)

        welcomeLayout.setRowStretch(4, 1)
        welcomeLayout.setColumnStretch(5, 1)

        self.welcomeFrame.setLayout(welcomeLayout)
        self.mainLayout.addWidget(self.welcomeFrame, 0, 1)
        self.itemsDict[self.welcomeItem.text(0)] = self.welcomeFrame

    def addQuickStartPage(self):
        self.QuickStartViewBox = QWidget(self)

        self.quickStartScrollArea = QScrollArea()
        self.quickStartScrollArea.setWidgetResizable(True)
        self.quickStartScrollArea.setFrameStyle(QFrame.Shape.NoFrame)
        self.quickStartScrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.quickStartScrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        QuickStartLayout = QGridLayout()

        fs = 13 # font size

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
            <span style="font-size:16px; font-family:ubuntu">
                <b>NOTE: This Quick Start is NOT an exhaustive manual.</b><br>
            </span>
            <span style="font-size:15px; font-family:ubuntu">
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
        <p style="font-size:{fs}px; font-family:ubuntu">
            Cell-ACDC is made of three main modules:
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
        <p style="font-size:{fs}px; font-family:ubuntu">
            <b>GUI tips and tricks:</b>
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
        pixmap = QPixmap(':toolbar.png')
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
        <p style="font-size:{fs}px; font-family:ubuntu">
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
        <p style="font-size:{fs}px; font-family:ubuntu">
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
        pixmap = QPixmap(':toolTip.png')
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
        <p style="font-size:{fs}px; font-family:ubuntu">
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
        <p style="font-size:{fs}px; font-family:ubuntu">
            <ul style="line-height:150%">
                <li>
                    Functions that are <b>NOT activated by a toolbar button</b>:
                    <blockquote>

                    &nbsp;&nbsp; - Click <code>scrolling wheel button</code> on Windows,
                    <code>Cmd+Click</code> on macOS --> <b>delete</b> segmented object<br>

                    &nbsp;&nbsp; - <code>H key</code> -->
                    <b>automatic zoom</b> on the segmented objects<br>

                    &nbsp;&nbsp; - Double press <code>H key</code> -->
                    <b>zoom out</b><br>

                    &nbsp;&nbsp; - <code>Shift+S</code> -->
                    Shuffle colormap to change labels' color<br>

                    &nbsp;&nbsp; - <code>Z+left/right arrow</code> -->
                    Change visualized z-slice (for z-stack data)<br>

                    &nbsp;&nbsp; - <code>Ctrl+P</code> -->
                    Visualize cell cycle annotations in a table<br>

                    &nbsp;&nbsp; - <code>Ctrl+L</code> -->
                    relabel object IDs sequentially<br>

                    &nbsp;&nbsp; - Double press <code>Spacebar</code> -->
                    hide/show annotations on left image<br>

                    &nbsp;&nbsp; - <code>Alt+Click&Drag</code> -->
                    pan/move image even if other tools active<br>
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
        <p style="font-size:{fs}px; font-family:ubuntu">
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
        <p style="font-size:{fs}px; font-family:ubuntu">
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
        <p style="font-size:{fs}px; font-family:ubuntu">
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
        <p style="font-size:{fs}px; font-family:ubuntu">
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
        <p style="font-size:{fs}px; font-family:ubuntu">
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
        <p style="font-size:{fs}px; font-family:ubuntu; line-height:1.2">
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
            text='Test segmentation with my image/video')
        testMyImage.setIcon(QIcon(':image.svg'))
        layout.addWidget(testMyImage)
        testMyImage.clicked.connect(self.openGUIsingleImage)

        testTimeLapseButton = QPushButton(
            text='Download and test with a time-lapse example')
        testTimeLapseButton.setIcon(QIcon(':download.svg'))
        layout.addWidget(testTimeLapseButton)
        testTimeLapseButton.clicked.connect(self.testTimeLapseExample)

        test3DzStackButton = QPushButton(
            text='Download and test with a 3D z-stack example')
        test3DzStackButton.setIcon(QIcon(':download.svg'))
        layout.addWidget(test3DzStackButton)
        test3DzStackButton.clicked.connect(self.test3DzStacksExample)

        layout.addStretch(1)

        layout.setContentsMargins(15, 0, 0, 10)
        QuickStartLayout.addLayout(layout, row, 0)

        QuickStartLayout.setRowStretch(QuickStartLayout.rowCount(), 1)

        self.QuickStartLayout = QuickStartLayout

        self.QuickStartViewBox.setLayout(QuickStartLayout)
        self.quickStartScrollArea.setWidget(self.QuickStartViewBox)

        self.mainLayout.addWidget(self.quickStartScrollArea, 0, 1)
        self.itemsDict[self.quickStartItem.text(0)] = self.quickStartScrollArea

    def addManualPage(self):
        self.manualFrame = QFrame(self)
        manualLayout = QGridLayout()

        openManualButton = widgets.showInFileManagerButton(
            ' Download and open user manual... '
        )
        openManualButton.clicked.connect(myutils.showUserManual)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(openManualButton)
        buttonLayout.addStretch()

        manualLayout.addLayout(buttonLayout, 0, 0, alignment=Qt.AlignTop)

        self.manualFrame.setLayout(manualLayout)
        self.mainLayout.addWidget(self.manualFrame, 0, 1)
        self.itemsDict[self.manualItem.text(0)] = self.manualFrame

    def addContributePage(self):
        self.contributeFrame = QFrame(self)

        layout = QGridLayout()

        contribute_href = html_utils.href_tag('here', urls.contribute_url)
        github_href = html_utils.href_tag('GitHub page', urls.github_url)
        issues_href = html_utils.href_tag('Issues', urls.issues_url)
        forum_href = html_utils.href_tag('Discussions', urls.forum_url)
        resources_href = html_utils.href_tag('here', urls.resources_url)
        my_contact_href = html_utils.href_tag('my email', urls.my_contact_url)
        user_manual_href = html_utils.href_tag('User Manual', urls.user_manual_url)

        text = (f"""
        <p style="font-size:15px; font-family:ubuntu">
            Here at Cell-ACDC we want to keep a <b>community-centred approach</b>.<br><br>
            
            If the software is <b>any useful</b> to you it's only thanks to the <b>great feedback</b><br>
            we received and we keep receiving from many users.<br><br>
            
            The easiest, and yet very effective, way to contribute is to simply <b>give us any feedback</b>.<br><br>
            
            It can be <b>ideas</b> you have that can improve the user experience, or a <b>request</b> for a specific feature.<br><br>
            
            We also welcome <b>contributions to the code</b>, for example by adding your own custom model or tracker.<br>
            This way you allow users without programming skills to use your models! Check out the instructions<br>
            on how to contribute to the code {contribute_href}.<br><br>

            You can find instructions on how to add segmentation or tracking models on our {user_manual_href} at the section<br>
            called <code>Adding segmentation models to the pipeline</code>.<br><br>

            Additionally, please <b>report any issue</b> you have, because this will greatly help also the other users.<br><br>

            Finally, do not hesitate to <b>ask any question</b> you have about the software.<br><br>

            The best way to talk to us is on our {github_href}. You can use the {issues_href} page to report an issue, propose a new feature<br>
            or simply ask a question.<br><br>

            Alternatively, you can participate or open a new discussion on our {forum_href} page.<br><br>

            Of course, you are also <b>free to contact me</b> directly at {my_contact_href}.<br><br>

            Additional resources {resources_href}.
        </p>
        """)

        label = QLabel()
        label.setText(text)
        label.setOpenExternalLinks(True)

        layout.addWidget(label, 0, 0, alignment=Qt.AlignTop)
        self.contributeFrame.setLayout(layout)
        self.mainLayout.addWidget(self.contributeFrame, 0, 1)
        self.itemsDict[self.contributeItem.text(0)] = self.contributeFrame


    def linkActivated_cb(self, link):
        if link == 'DataPrepMore':
            pass
        elif link == 'paper':
            url = cite_url
            webbrowser.open(url)
        elif link == 'tweet':
            url = 'https://twitter.com/frank_pado/status/1443957038841794561?s=20'
            webbrowser.open(url)
        elif link == 'segmMore':
            pass
        elif link == 'guiMore':
            pass
        elif link == 'quickStart':
            self.showPage(self.quickStartItem)
        elif link == 'userManual':
            self.showPage(self.manualItem)

    def addShowGuideCheckbox(self):
        checkBox = QCheckBox('Show Welcome Guide when opening Cell-ACDC')
        checked = self.df_settings.at['showWelcomeGuide', 'value'] == 'Yes'
        checkBox.setChecked(checked)
        self.mainLayout.addWidget(checkBox, 1, 1, alignment=Qt.AlignRight)

        checkBox.stateChanged.connect(self.showWelcomeGuideCheckBox_cb)

    def showWelcomeGuideCheckBox_cb(self, state):
        if state == 0:
            show = 'No'
        else:
            show = 'Yes'
        self.df_settings.loc['showWelcomeGuide'] = (
            self.df_settings.loc['showWelcomeGuide'].astype(str)
        )
        self.df_settings.at['showWelcomeGuide', 'value'] = show
        self.saveSettings()

    def saveSettings(self):
        self.df_settings.to_csv(self.df_settings_path)

    def openGUIsingleImage(self):
        if self.mainWin is not None:
            self.mainWin.launchGui()
            guiWin = self.mainWin.guiWins[-1]
            QTimer.singleShot(200, guiWin.openFile)
            self.mainWin.guiWins[-1].openFile()
        else:
            self.guiWin = gui.guiWin(self.app)
            self.guiWin.showAndSetSize()
            self.guiWin.openFile()

    def openGUIfolder(self, exp_path):
        if self.mainWin is not None:
            self.mainWin.launchGui()
            guiWin = self.mainWin.guiWins[-1]
            QTimer.singleShot(
                200, partial(guiWin.openFolder, exp_path=exp_path)
            )
        else:
            self.guiWin = gui.guiWin(self.app)
            self.guiWin.showAndSetSize()
            self.guiWin.openFolder(exp_path=exp_path)

    def launchDataStruct(self, checked=True):
        self.dataStructWin = dataStruct.createDataStructWin(
            mainWin=self
        )
        self.dataStructWin.show()
        self.dataStructWin.main()

    def addPbar(self):
        self.QPbar = widgets.ProgressBar(self)
        self.QPbar.setValue(0)
        self.welcomeLayout.addWidget(self.QPbar, 3, 0, 1, 3)

    def testTimeLapseExample(self, checked=True):
        _, example_path, _, _ = myutils.get_examples_path('time_lapse_2D')
        txt = (
        f"""
        <p style="font-size:11px; font-family:ubuntu">
            <br><b>Downloading example</b> to {example_path}...
        </p>
        """
        )
        self.infoTextWidget.setText(txt)

        if self.QPbar is None:
            self.addPbar()

        self.thread = QThread()
        self.worker = downloadWorker('time_lapse_2D')
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self.downloadProgress)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.openGUIexample)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def downloadProgress(self, file_size, len_chunk):
        if file_size != -1:
            self.QPbar.setMaximum(file_size)
        elif len_chunk != -1:
            self.QPbar.setValue(self.QPbar.value()+len_chunk)
        elif len_chunk == 0:
            self.QPbar.setValue(self.QPbar.maximum())

    def openGUIexample(self):
        txt = (
        f"""
        <p style="font-size:11px; font-family:ubuntu">
            <br><b>Example downloaded</b> to {self.worker.exp_path}.<br>
            Opening GUI...
        </p>
        """
        )
        self.infoTextWidget.setText(txt)
        self.QPbar.setValue(self.QPbar.maximum())
        self.openGUIfolder(self.worker.exp_path)


    def test3DzStacksExample(self, checked=True):
        _, example_path, _, _ = myutils.get_examples_path('snapshots_3D')
        txt = (
        f"""
        <p style="font-size:11px; font-family:ubuntu">
            <br><b>Downloading example</b> to {example_path}...
        </p>
        """
        )
        self.infoTextWidget.setText(txt)

        if self.QPbar is None:
            self.addPbar()

        self.thread = QThread()
        self.worker = downloadWorker('snapshots_3D')
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self.downloadProgress)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.openGUIexample)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def debug(self):
        print(self.quickStartScrollArea.horizontalScrollBar().isVisible())

    def showAndSetSize(self):
        font = QFont()
        font.setPixelSize(13)
        font.setFamily('Ubuntu')
        self.treeSelector.setFont(font)
        self.showPage(self.quickStartItem)

        self.show()

        self.treeSelector.setFixedWidth(self.treeSelector.width())

        self.timer = QTimer()
        self.timer.timeout.connect(self.resizeScrollbar)
        self.timer.start(1)

    def resizeScrollbar(self):
        if self.quickStartScrollArea.horizontalScrollBar().isVisible():
            self.resize(self.width()+5, self.height())
        else:
            self.timer.stop()
            self.moveWindow()

    def moveWindow(self):
        screenWidth = self.screen().size().width()
        screenHeight = self.screen().size().height()
        screenLeft = self.screen().geometry().x()
        screenTop = self.screen().geometry().y()
        screenRight = screenLeft + screenWidth

        winGeometry = self.geometry()
        w = winGeometry.width()
        l, t, h = winGeometry.left(), winGeometry.top(), winGeometry.height()
        w0 = winGeometry.width()
        Dw = w - w0
        Dh = 1.5
        left = screenLeft + 10
        top = screenTop + 70
        width = w
        height = int(h*Dh)
        if height > 0.9*screenHeight:
            height = int(0.9*screenHeight)

        self.setGeometry(left, top, width, height)
        if self.mainWin is not None:
            mainWinWidth = self.mainWin.width()
            welcomeWinRight = left+width
            if welcomeWinRight+mainWinWidth > screenRight:
                # The right edge of the welcome window is out of screen bounds
                # Keep it in the screen
                welcomeWinRight = screenRight-mainWinWidth
            self.mainWin.move(welcomeWinRight, top)


    def showPage(self, currentItem):
        self.treeSelector.setCurrentItem(currentItem, 0)

    def eventFilter(self, object, event):
        # Disable wheel scroll on widgets to allow scroll only on scrollarea
        if event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return False

if __name__ == '__main__':
    app = QApplication([])
    win = welcomeWin(app=app)
    win.showAndSetSize()
    win.showPage(win.welcomeItem)
    # win.showPage(win.quickStartItem)
    app.setStyle(QStyleFactory.create('Fusion'))
    sys.exit(app.exec_())
