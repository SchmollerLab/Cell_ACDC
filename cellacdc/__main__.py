#!/usr/bin/env python
import os
import logging

from cellacdc import dataReStruct
from . import exception_handler
from . import qrc_resources
if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

from PyQt5 import QtGui, QtWidgets, QtCore

# Handle high resolution displays:
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

# Needed by pyqtgraph with display resolution scaling
QtWidgets.QApplication.setAttribute(
    QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
)

class AcdcSPlashScreen(QtWidgets.QSplashScreen):
    def __init__(self):
        super().__init__()
        cellacdc_path = os.path.dirname(os.path.abspath(__file__))
        resources_path = os.path.join(cellacdc_path, 'resources')
        logo_path = os.path.join(resources_path, 'logo.png')
        self.setPixmap(QtGui.QPixmap(logo_path))
    
    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        pass

# Create the application
app = QtWidgets.QApplication([])

app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
app.setPalette(app.style().standardPalette())
app.setWindowIcon(QtGui.QIcon(":icon.ico"))

# Launch splashscreen
splashScreen = AcdcSPlashScreen()
splashScreen.setWindowIcon(QtGui.QIcon(":icon.ico"))
splashScreen.setWindowFlags(
    QtCore.Qt.WindowStaysOnTopHint 
    | QtCore.Qt.SplashScreen 
    | QtCore.Qt.FramelessWindowHint
)
splashScreen.show()
splashScreen.raise_()

print('Importing modules...')
import sys
import re
import traceback

import pandas as pd
import psutil

from functools import partial

from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QPushButton, QLabel, QAction,
    QMenu, QHBoxLayout, QFileDialog
)
from PyQt5.QtCore import (
    Qt, QProcess, pyqtSignal, pyqtSlot, QTimer, QSize,
    QSettings, QUrl, QCoreApplication
)
from PyQt5.QtGui import (
    QFontDatabase, QIcon, QDesktopServices, QFont, QMouseEvent, 
    QPixmap
)

# acdc modules
try:
    # We try to import from cellacdc instead of "from ." to check
    # if cellacdc was installed with pip or not
    from cellacdc import (
        dataPrep, segm, gui, dataStruct, utils, help, qrc_resources, myutils,
        cite_url, html_utils, widgets, apps, dataReStruct
    )
    from cellacdc.help import about
    from cellacdc.utils import concat as utilsConcat
    from cellacdc.utils import convert as utilsConvert
    from cellacdc.utils import rename as utilsRename
    from cellacdc.utils import align as utilsAlign
    from cellacdc.utils import compute as utilsCompute
    from cellacdc.utils import repeat as utilsRepeat
    from cellacdc.utils import toImageJroi as utilsToImageJroi
    from cellacdc.utils import acdcToSymDiv as utilsSymDiv
    from cellacdc.utils import trackSubCellObjects as utilsTrackSubCell
    from cellacdc.utils import computeMultiChannel as utilsComputeMultiCh
    from cellacdc.utils import applyTrackFromTable as utilsApplyTrackFromTab
    from cellacdc.info import utilsInfo
    from cellacdc import is_win, is_linux, temp_path
    from cellacdc import printl
except ModuleNotFoundError as e:
    src_path = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.dirname(src_path)
    print('='*30)
    traceback.print_exc()
    print('----------------------------------------')
    print(
        'Cellacdc NOT INSTALLED. '
        'Run the following command to install: '
        f'pip install -e "{main_path}"'
    )
    print('----------------------------------------')
    exit('Execution aborted due to an error. See above for details.')

try:
    import spotmax
    from spotmax import _run as spotmaxRun
    spotmax_filepath = os.path.dirname(os.path.abspath(spotmax.__file__))
    spotmax_logo_path = os.path.join(spotmax_filepath, 'resources', 'logo.svg')
    SPOTMAX = True
except Exception as e:
    if not isinstance(e, ModuleNotFoundError):
        traceback.print_exc()
    SPOTMAX = False

class mainWin(QMainWindow):
    def __init__(self, app, parent=None):
        self.app = app
        self.welcomeGuide = None
        super().__init__(parent)
        self.setWindowTitle("Cell-ACDC")
        self.setWindowIcon(QIcon(":icon.ico"))
        self.setAcceptDrops(True)

        logger, logs_path, log_path, log_filename = myutils.setupLogger(
            module='main'
        )
        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path

        if not is_linux:
            self.loadFonts()

        self.createActions()
        self.createMenuBar()
        self.connectActions()

        mainContainer = QtWidgets.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()
        mainLayout.addStretch()

        welcomeLabel = QLabel(html_utils.paragraph(
            '<b>Welcome to Cell-ACDC!</b>',
            center=True, font_size='18px'
        ))
        # padding: top, left, bottom, right
        welcomeLabel.setStyleSheet("padding:0px 0px 5px 0px;")
        mainLayout.addWidget(welcomeLabel)

        label = QLabel(html_utils.paragraph(
            'Press any of the following buttons<br>'
            'to <b>launch</b> the respective module',
            center=True, font_size='14px'
        ))
        # padding: top, left, bottom, right
        label.setStyleSheet("padding:0px 0px 10px 0px;")
        mainLayout.addWidget(label)

        mainLayout.addStretch()

        iconSize = 26

        dataStructButton = widgets.setPushButton(
            '  0. Create data structure from microscopy/image file(s)...  '
        )
        dataStructButton.setIconSize(QSize(iconSize,iconSize))
        font = QFont()
        font.setPixelSize(13)
        dataStructButton.setFont(font)
        dataStructButton.clicked.connect(self.launchDataStruct)
        self.dataStructButton = dataStructButton
        mainLayout.addWidget(dataStructButton)

        dataPrepButton = QPushButton('  1. Launch data prep module...')
        dataPrepButton.setIcon(QIcon(':prep.svg'))
        dataPrepButton.setIconSize(QSize(iconSize,iconSize))
        font = QFont()
        font.setPixelSize(13)
        dataPrepButton.setFont(font)
        dataPrepButton.clicked.connect(self.launchDataPrep)
        self.dataPrepButton = dataPrepButton
        mainLayout.addWidget(dataPrepButton)

        segmButton = QPushButton('  2. Launch segmentation module...')
        segmButton.setIcon(QIcon(':segment.svg'))
        segmButton.setIconSize(QSize(iconSize,iconSize))
        segmButton.setFont(font)
        segmButton.clicked.connect(self.launchSegm)
        self.segmButton = segmButton
        mainLayout.addWidget(segmButton)

        guiButton = QPushButton('  3. Launch GUI...')
        guiButton.setIcon(QIcon(':icon.ico'))
        guiButton.setIconSize(QSize(iconSize,iconSize))
        guiButton.setFont(font)
        guiButton.clicked.connect(self.launchGui)
        self.guiButton = guiButton
        mainLayout.addWidget(guiButton)

        if SPOTMAX:
            spotmaxButton = QPushButton('  4. Launch spotMAX...')
            spotmaxButton.setIcon(QIcon(spotmax_logo_path))
            spotmaxButton.setIconSize(QSize(iconSize,iconSize))
            spotmaxButton.setFont(font)
            self.spotmaxButton = spotmaxButton
            spotmaxButton.clicked.connect(self.launchSpotmaxGui)
            mainLayout.addWidget(spotmaxButton)

        font = QFont()
        font.setPixelSize(13)

        closeLayout = QHBoxLayout()
        restartButton = QPushButton(
            QIcon(":reload.svg"),
            '  Restart Cell-ACDC'
        )
        restartButton.setFont(font)
        restartButton.setIconSize(QSize(iconSize, iconSize))
        restartButton.clicked.connect(self.close)
        self.restartButton = restartButton
        closeLayout.addWidget(restartButton)

        closeButton = QPushButton(QIcon(":exit.png"), '  Exit')
        closeButton.setIconSize(QSize(iconSize, iconSize))
        self.closeButton = closeButton
        # closeButton.setIconSize(QSize(24,24))
        closeButton.setFont(font)
        closeButton.clicked.connect(self.close)
        closeLayout.addWidget(closeButton)

        mainLayout.addLayout(closeLayout)
        mainContainer.setLayout(mainLayout)

        self.start_JVM = True

        self.guiWins = []
        self.spotmaxWins = []
        self.dataPrepWin = None
        self._version = None
        self.progressWin = None
    
    def dragEnterEvent(self, event) -> None:
        printl(event)
    
    def log(self, text):
        self.logger.info(text)
        
        if self.progressWin is None:
            return
    
        self.progressWin.log(text)

    def setVersion(self, version):
        self._version = version

    def loadFonts(self):
        font = QFont()
        # font.setFamily('Ubuntu')
        QFontDatabase.addApplicationFont(":Ubuntu-Regular.ttf")
        QFontDatabase.addApplicationFont(":Ubuntu-Bold.ttf")
        QFontDatabase.addApplicationFont(":Ubuntu-Italic.ttf")
        QFontDatabase.addApplicationFont(":Ubuntu-BoldItalic.ttf")
        QFontDatabase.addApplicationFont(":Calibri-Regular.ttf")
        QFontDatabase.addApplicationFont(":Calibri-Bold.ttf")
        QFontDatabase.addApplicationFont(":Calibri-Italic.ttf")
        QFontDatabase.addApplicationFont(":Calibri-BoldItalic.ttf")
        QFontDatabase.addApplicationFont(":ArialMT-Regular.ttf")
        QFontDatabase.addApplicationFont(":ArialMT-Bold.otf")
        QFontDatabase.addApplicationFont(":ArialMT-Italic.otf")
        QFontDatabase.addApplicationFont(":ArialMT-BoldItalic.otf")

    def launchWelcomeGuide(self, checked=False):
        cellacdc_path = os.path.dirname(os.path.realpath(__file__))
        temp_path = os.path.join(cellacdc_path, 'temp')
        csv_path = os.path.join(temp_path, 'settings.csv')
        self.settings_csv_path = csv_path
        if not os.path.exists(csv_path):
            idx = ['showWelcomeGuide']
            values = ['Yes']
            self.df_settings = pd.DataFrame({'setting': idx,
                                             'value': values}
                                           ).set_index('setting')
            self.df_settings.to_csv(csv_path)
        self.df_settings = pd.read_csv(csv_path, index_col='setting')
        if 'showWelcomeGuide' not in self.df_settings.index:
            self.df_settings.at['showWelcomeGuide', 'value'] = 'Yes'
            self.df_settings.to_csv(csv_path)

        show = (
            self.df_settings.at['showWelcomeGuide', 'value'] == 'Yes'
            or self.sender() is not None
        )
        if not show:
            return

        self.welcomeGuide = help.welcome.welcomeWin(mainWin=self)
        self.welcomeGuide.showAndSetSize()
        self.welcomeGuide.showPage(self.welcomeGuide.welcomeItem)

    def setColorsAndText(self):
        self.moduleLaunchedColor = '#ead935'
        defaultColor = self.guiButton.palette().button().color().name()
        self.defaultPushButtonColor = defaultColor
        self.defaultTextDataStructButton = self.dataStructButton.text()
        self.defaultTextGuiButton = self.guiButton.text()
        self.defaultTextDataPrepButton = self.dataPrepButton.text()
        self.defaultTextSegmButton = self.segmButton.text()

    def createMenuBar(self):
        menuBar = self.menuBar()

        self.recentPathsMenu = QMenu("&Recent paths", self)
        # On macOS an empty menu would not appear --> add dummy action
        self.recentPathsMenu.addAction('dummy macos')
        menuBar.addMenu(self.recentPathsMenu)

        utilsMenu = QMenu("&Utilities", self)

        convertMenu = utilsMenu.addMenu('Convert file formats')
        convertMenu.addAction(self.npzToNpyAction)
        convertMenu.addAction(self.npzToTiffAction)
        convertMenu.addAction(self.TiffToNpzAction)
        convertMenu.addAction(self.h5ToNpzAction)
        convertMenu.addAction(self.toImageJroiAction)

        trackingMenu = utilsMenu.addMenu('Tracking')
        trackingMenu.addAction(self.trackSubCellFeaturesAction)
        trackingMenu.addAction(self.applyTrackingFromTableAction)

        measurementsMenu = utilsMenu.addMenu('Measurements')
        measurementsMenu.addAction(self.calcMetricsAcdcDf)
        measurementsMenu.addAction(self.combineMetricsMultiChannelAction) 

        utilsMenu.addAction(self.toSymDivAction)
        utilsMenu.addAction(self.concatAcdcDfsAction)                      
        utilsMenu.addAction(self.batchConverterAction)
        utilsMenu.addAction(self.repeatDataPrepAction)
        utilsMenu.addAction(self.alignAction)
        utilsMenu.addAction(self.renameAction)

        self.utilsMenu = utilsMenu

        utilsMenu.addSeparator()
        utilsHelpAction = utilsMenu.addAction('Help...')
        utilsHelpAction.triggered.connect(self.showUtilsHelp)
    
        menuBar.addMenu(utilsMenu)

        napariMenu = QMenu("&napari", self)
        napariMenu.addAction(self.arboretumAction)
        napariMenu.triggered.connect(self.launchNapariUtil)
        menuBar.addMenu(napariMenu)

        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.welcomeGuideAction)
        helpMenu.addAction(self.userManualAction)
        helpMenu.addAction(self.citeAction)
        helpMenu.addAction(self.contributeAction)
        helpMenu.addAction(self.showLogsAction)
        helpMenu.addSeparator()
        helpMenu.addAction(self.aboutAction)

        menuBar.addMenu(helpMenu)
    
    def showUtilsHelp(self):
        treeInfo = {}
        for action in self.utilsMenu.actions():
            if action.menu() is not None:
                menu = action.menu()
                for sub_action in menu.actions():
                    treeInfo = self._addActionToTree(
                        sub_action, treeInfo, parentMenu=menu
                    )
            else:
                treeInfo = self._addActionToTree(action, treeInfo)
         
        self.utilsHelpWin = apps.TreeSelectorDialog(
            title='Utilities help', 
            infoTxt="Double click on a utility's name to get help about it<br>",
            parent=self, multiSelection=False, widthFactor=2, heightFactor=1.5
        )
        self.utilsHelpWin.addTree(treeInfo)
        self.utilsHelpWin.sigItemDoubleClicked.connect(self._showUtilHelp)
        self.utilsHelpWin.exec_()
    
    def _showUtilHelp(self, item):
        if item.parent() is None:
            return
        utilityName = item.text(0)
        infoText = html_utils.paragraph(utilsInfo[utilityName])
        runUtilityButton = widgets.playPushButton('Run utility...')
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.information(
            self.utilsHelpWin, f'"{utilityName}" help', infoText,
            buttonsTexts=(runUtilityButton, 'Close'), showDialog=False
        )
        runUtilityButton.utilityName = utilityName
        runUtilityButton.clicked.connect(self._runUtility)
        msg.exec_()
    
    def _runUtility(self):
        self.utilsHelpWin.ok_cb()
        utilityName = self.sender().utilityName
        for action in self.utilsMenu.actions():
            if action.menu() is not None:
                menu = action.menu()
                for sub_action in menu.actions():
                    if sub_action.text() == utilityName:
                        sub_action.trigger()
                        break
                else:
                    continue
                break
            else:
                action.trigger()
                break
    
    def _addActionToTree(self, action, treeInfo, parentMenu=None):
        if action.isSeparator():
            return treeInfo
        
        text = action.text()
        if text not in utilsInfo:
            return treeInfo
        
        if parentMenu is None:
            treeInfo[text] = []
        elif parentMenu.title() not in treeInfo:
            treeInfo[parentMenu.title()] = [text]
        else:
            treeInfo[parentMenu.title()].append(text)
        return treeInfo

    def createActions(self):
        self.npzToNpyAction = QAction('Convert .npz file(s) to .npy...')
        self.npzToTiffAction = QAction('Convert .npz file(s) to .tif...')
        self.TiffToNpzAction = QAction('Convert .tif file(s) to _segm.npz...')
        self.h5ToNpzAction = QAction('Convert .h5 file(s) to _segm.npz...')
        self.toImageJroiAction = QAction(
            'Convert _segm.npz file(s) to ImageJ ROIs...'
        )
        self.trackSubCellFeaturesAction = QAction(
            'Track sub-cellular objects (assign same ID as the cell they belong to)...'
        )    
        self.applyTrackingFromTableAction = QAction(
            'Apply tracking info from tabular data...'
        )
        self.batchConverterAction = QAction(
            'Create required data structure from image files...'
        )
        self.repeatDataPrepAction = QAction(
            'Re-apply data prep steps to selected channels...'
        )
        # self.TiffToHDFAction = QAction('Convert .tif file(s) to .h5py...')
        self.concatAcdcDfsAction = QAction(
            'Concatenate acdc output tables from multiple Positions...'
        )
        self.calcMetricsAcdcDf = QAction(
            'Compute measurements for one or more experiments...'
        )
        self.combineMetricsMultiChannelAction = QAction(
            'Combine measurements from multiple segmentation files...'
        )
        self.toSymDivAction = QAction(
            'Add lineage tree table to one or more experiments...'
        )
        self.renameAction = QAction('Rename files by appending additional text...')
        self.alignAction = QAction('Align or revert alignment...')

        self.arboretumAction = QAction(
            'View lineage tree in napari-arboretum...'
        )

        self.welcomeGuideAction = QAction('Welcome Guide')
        self.userManualAction = QAction('User manual...')
        self.aboutAction = QAction('About Cell-ACDC')
        self.citeAction = QAction('Cite us...')
        self.contributeAction = QAction('Contribute...')
        self.showLogsAction = QAction('Show log files...')

    def connectActions(self):
        self.alignAction.triggered.connect(self.launchAlignUtil)
        self.concatAcdcDfsAction.triggered.connect(self.launchConcatUtil)
        self.npzToNpyAction.triggered.connect(self.launchConvertFormatUtil)
        self.npzToTiffAction.triggered.connect(self.launchConvertFormatUtil)
        self.TiffToNpzAction.triggered.connect(self.launchConvertFormatUtil)
        self.h5ToNpzAction.triggered.connect(self.launchConvertFormatUtil)
        self.toImageJroiAction.triggered.connect(self.launchToImageJroiUtil)
        self.trackSubCellFeaturesAction.triggered.connect(
            self.launchTrackSubCellFeaturesUtil
        )
        self.combineMetricsMultiChannelAction.triggered.connect(
            self.launchCombineMeatricsMultiChanneliUtil
        )
        
        self.batchConverterAction.triggered.connect(
                self.launchImageBatchConverter
            )
        self.repeatDataPrepAction.triggered.connect(
                self.launchRepeatDataPrep
            )
        self.welcomeGuideAction.triggered.connect(self.launchWelcomeGuide)
        self.toSymDivAction.triggered.connect(self.launchToSymDicUtil)
        self.calcMetricsAcdcDf.triggered.connect(self.launchCalcMetricsUtil)
        self.aboutAction.triggered.connect(self.showAbout)
        self.renameAction.triggered.connect(self.launchRenameUtil)

        self.userManualAction.triggered.connect(myutils.showUserManual)
        self.contributeAction.triggered.connect(self.showContribute)
        self.citeAction.triggered.connect(
            partial(QDesktopServices.openUrl, QUrl(cite_url))
        )
        self.recentPathsMenu.aboutToShow.connect(self.populateOpenRecent)
        self.showLogsAction.triggered.connect(self.showLogFiles)
        self.applyTrackingFromTableAction.triggered.connect(
            self.launchApplyTrackingFromTableUtil
        )
    
    def showLogFiles(self):
        logs_path = myutils.get_logs_path()
        myutils.showInExplorer(logs_path)

    def populateOpenRecent(self):
        # Step 0. Remove the old options from the menu
        self.recentPathsMenu.clear()
        # Step 1. Read recent Paths
        recentPaths_path = os.path.join(temp_path, 'recentPaths.csv')
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            if 'opened_last_on' in df.columns:
                df = df.sort_values('opened_last_on', ascending=False)
            recentPaths = df['path'].to_list()
        else:
            recentPaths = []
        # Step 2. Dynamically create the actions
        actions = []
        for path in recentPaths:
            action = QAction(path, self)
            action.triggered.connect(partial(myutils.showInExplorer, path))
            actions.append(action)
        # Step 3. Add the actions to the menu
        self.recentPathsMenu.addActions(actions)

    def showContribute(self):
        self.launchWelcomeGuide()
        self.welcomeGuide.showPage(self.welcomeGuide.contributeItem)

    def showAbout(self):
        self.aboutWin = about.QDialogAbout(parent=self)
        self.aboutWin.show()
    
    def getSelectedPosPath(self, utilityName):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            After you click "Ok" on this dialog you will be asked
            to <b>select one position folder</b> that contains timelapse
            data.
        """)
        msg.information(
            self, f'{utilityName}', txt,
            buttonsTexts=('Cancel', 'Ok')
        )
        if msg.cancel:
            print(f'{utilityName} aborted by the user.')
            return
        
        mostRecentPath = myutils.getMostRecentPath()
        exp_path = QFileDialog.getExistingDirectory(
            self, 'Select Position_n folder',
            mostRecentPath
        )
        if not exp_path:
            print(f'{utilityName} aborted by the user.')
            return
        
        myutils.addToRecentPaths(exp_path)
        baseFolder = os.path.basename(exp_path)
        isPosFolder = re.search('Position_(\d+)$', baseFolder) is not None
        isImagesFolder = baseFolder == 'Images'
        if isImagesFolder:
            posPath = os.path.dirname(exp_path)
            posFolders = [os.path.basename(posPath)]
            exp_path = os.path.dirname(posPath)
        elif isPosFolder:
            posPath = exp_path
            posFolders = [os.path.basename(posPath)]
            exp_path = os.path.dirname(exp_path)
        else:
            posFolders = myutils.get_pos_foldernames(exp_path)
            if not posFolders:
                msg = widgets.myMessageBox()
                msg.addShowInFileManagerButton(
                    exp_path, txt='Show selected folder...'
                )
                _ls = "\n".join(os.listdir(exp_path))
                msg.setDetailedText(f'Files present in the folder:\n{_ls}')
                txt = html_utils.paragraph(f"""
                    The selected folder:<br><br>
                    <code>{exp_path}</code><br><br>
                    does not contain any valid Position folders.<br>
                """)
                msg.warning(
                    self, 'Not valid folder', txt,
                    buttonsTexts=('Cancel', 'Try again')
                )
                if msg.cancel:
                    print(f'{utilityName} aborted by the user.')
                    return

        if len(posFolders) > 1:
            win = apps.QDialogCombobox(
                'Select position folder', posFolders, 'Select position folder',
                'Positions: ', parent=self
            )
            win.exec_()
            posPath = os.path.join(exp_path, win.selectedItemText)
        
        return posPath

    def getSelectedExpPaths(self, utilityName):
        self.logger.info('Asking to select experiment folders...')
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            After you click "Ok" on this dialog you will be asked
            to <b>select the experiment folders</b>, one by one.<br><br>
            Next, you will be able to <b>choose specific Positions</b>
            from each selected experiment.
        """)
        msg.information(
            self, f'{utilityName}', txt,
            buttonsTexts=('Cancel', 'Ok')
        )
        if msg.cancel:
            self.logger.info(f'{utilityName} aborted by the user.')
            return

        expPaths = {}
        mostRecentPath = myutils.getMostRecentPath()
        while True:
            exp_path = QFileDialog.getExistingDirectory(
                self, 'Select experiment folder containing Position_n folders',
                mostRecentPath
            )
            if not exp_path:
                break
            myutils.addToRecentPaths(exp_path)
            baseFolder = os.path.basename(exp_path)
            isPosFolder = (
                re.search('Position_(\d+)$', baseFolder) is not None
                and os.path.exists(os.path.join(exp_path, 'Images'))
            )
            isImagesFolder = baseFolder == 'Images'
            if isImagesFolder:
                posPath = os.path.dirname(exp_path)
                posFolders = [os.path.basename(posPath)]
                exp_path = os.path.dirname(posPath)
            elif isPosFolder:
                posPath = exp_path
                posFolders = [os.path.basename(posPath)]
                exp_path = os.path.dirname(exp_path)
            else:
                posFolders = myutils.get_pos_foldernames(exp_path)
                if not posFolders:
                    msg = widgets.myMessageBox()
                    msg.addShowInFileManagerButton(
                        exp_path, txt='Show selected folder...'
                    )
                    _ls = "\n".join(os.listdir(exp_path))
                    msg.setDetailedText(f'Files present in the folder:\n{_ls}')
                    txt = html_utils.paragraph(f"""
                        The selected folder:<br><br>
                        <code>{exp_path}</code><br><br>
                        does not contain any valid Position folders.<br>
                    """)
                    msg.warning(
                        self, 'Not valid folder', txt,
                        buttonsTexts=('Cancel', 'Try again')
                    )
                    if msg.cancel:
                        self.logger.info(f'{utilityName} aborted by the user.')
                        return
                    continue

            expPaths[exp_path] = posFolders
            mostRecentPath = exp_path
            msg = widgets.myMessageBox(wrapText=False)
            txt = html_utils.paragraph("""
                Do you want to select <b>additional experiment folders</b>?
            """)
            noButton, yesButton = msg.question(
                self, 'Select additional experiments?', txt,
                buttonsTexts=('No', 'Yes')
            )
            if msg.clickedButton == noButton:
                break

        if len(expPaths) > 1 or len(posFolders) > 1:
            infoPaths = self.getInfoPosStatus(expPaths)
            selectPosWin = apps.selectPositionsMultiExp(
                expPaths, infoPaths=infoPaths
            )
            selectPosWin.exec_()
            if selectPosWin.cancel:
                print(f'{utilityName} aborted by the user.')
                return
            selectedExpPaths = selectPosWin.selectedPaths
        else:
            selectedExpPaths = expPaths
        
        return selectedExpPaths
    
    def launchApplyTrackingFromTableUtil(self):
        posPath = self.getSelectedPosPath('Apply tracking info from tabular data')
        if posPath is None:
            return
        
        title = 'Apply tracking info from tabular data utility'
        infoText = 'Launching apply tracking info from tabular data...'
        self.applyTrackWin = (
            utilsApplyTrackFromTab.ApplyTrackingInfoFromTableUtil(
                self.app, title, infoText, parent=self, 
                callbackOnFinished=self.applyTrackingFromTableFinished
            )
        )
        self.applyTrackWin.show()
        func = partial(self._runApplyTrackingFromTableUtil, posPath, self.applyTrackWin)
        QTimer.singleShot(200, func)

    def _runApplyTrackingFromTableUtil(self, posPath, win):
        success = win.run(posPath)
        if not success:
            self.logger.info(
                'Apply tracking info from tabular data ABORTED by the user.'
            )
            win.close()          
        
    def applyTrackingFromTableFinished(self):
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph(
            'Apply tracking info from tabular data completed.'
        )
        msg.information(self, 'Process completed', txt)
        self.logger.info('Apply tracking info from tabular data completed.')
        self.applyTrackWin.close()

    
    def launchNapariUtil(self, action):
        myutils.install_package('napari', parent=self)
        if action == self.arboretumAction:
            self._launchArboretum()

    def _launchArboretum(self):
        myutils.install_package('napari_arboretum', parent=self)

        from cellacdc.napari_utils import arboretum
        
        posPath = self.getSelectedPosPath('napari-arboretum')
        if posPath is None:
            return

        title = 'napari-arboretum utility'
        infoText = 'Launching napari-arboretum to visualize lineage tree...'
        self.arboretumWindow = arboretum.NapariArboretumDialog(
            posPath, self.app, title, infoText, parent=self
        )
        self.arboretumWindow.show()
    
    def launchToImageJroiUtil(self):
        myutils.install_package('roifile', parent=self)

        import roifile

        selectedExpPaths = self.getSelectedExpPaths(
            'From _segm.npz to ImageJ ROIs'
        )
        if selectedExpPaths is None:
            return
        
        title = 'Convert _segm.npz file(s) to ImageJ ROIs'
        infoText = 'Launching to ImageJ ROIs process...'
        progressDialogueTitle = 'Converting _segm.npz file(s) to ImageJ ROIs'
        self.toImageJroiWin = utilsToImageJroi.toImageRoiUtil(
            selectedExpPaths, self.app, title, infoText, progressDialogueTitle,
            parent=self
        )
        self.toImageJroiWin.show()
    
    def launchCombineMeatricsMultiChanneliUtil(self):
        self.logger.info('Launching "Combine measurements from multiple tables" utility...')
        selectedExpPaths = self.getSelectedExpPaths(
            'Combine measurements from multiple channels'
        )
        if selectedExpPaths is None:
            return
        
        title = 'Compute measurements from multiple channels'
        infoText = 'Launching compute measurements from multiple channels process...'
        progressDialogueTitle = 'Compute measurements from multiple channels'
        self.multiChannelWin = utilsComputeMultiCh.ComputeMetricsMultiChannel(
            selectedExpPaths, self.app, title, infoText, progressDialogueTitle,
            parent=self
        )
        self.multiChannelWin.show()

    def launchTrackSubCellFeaturesUtil(self):
        selectedExpPaths = self.getSelectedExpPaths(
            'Track sub-cellular objects'
        )
        if selectedExpPaths is None:
            return
        
        win = apps.TrackSubCellObjectsDialog()
        win.exec_()
        if win.cancel:
            return
        
        title = 'Track sub-cellular objects'
        infoText = 'Launching sub-cellular objects tracker...'
        progressDialogueTitle = 'Tracking sub-cellular objects'
        self.trackSubCellObjWin = utilsTrackSubCell.TrackSubCellFeatures(
            selectedExpPaths, self.app, title, infoText, progressDialogueTitle,
            win.trackSubCellObjParams, parent=self
        )
        self.trackSubCellObjWin.show()

    
    def launchCalcMetricsUtil(self):
        selectedExpPaths = self.getSelectedExpPaths('Compute measurements utility')
        if selectedExpPaths is None:
            return

        self.calcMeasWin = utilsCompute.computeMeasurmentsUtilWin(
            selectedExpPaths, self.app, parent=self
        )
        self.calcMeasWin.show()
    
    def launchToSymDicUtil(self):
        selectedExpPaths = self.getSelectedExpPaths('Lineage tree utility')
        if selectedExpPaths is None:
            return

        self.toSymDivWin = utilsSymDiv.AcdcToSymDivUtil(
            selectedExpPaths, self.app, parent=self
        )
        self.toSymDivWin.show()
    
    def getInfoPosStatus(self, expPaths):
        infoPaths = {}
        for exp_path, posFoldernames in expPaths.items():
            posFoldersInfo = {}
            for pos in posFoldernames:
                pos_path = os.path.join(exp_path, pos)
                status = myutils.get_pos_status(pos_path)
                posFoldersInfo[pos] = status
            infoPaths[exp_path] = posFoldersInfo
        return infoPaths

    def launchRenameUtil(self):
        isUtilnabled = self.sender().isEnabled()
        if isUtilnabled:
            self.sender().setDisabled(True)
            self.renameWin = utilsRename.renameFilesWin(
                parent=self,
                actionToEnable=self.sender(),
                mainWin=self
            )
            self.renameWin.show()
            self.renameWin.main()
        else:
            self.renameWin.setWindowState(Qt.WindowActive)
            self.renameWin.raise_()

    def launchConvertFormatUtil(self, checked=False):
        s = self.sender().text()
        m = re.findall(r'Convert \.(\w+) file\(s\) to (.*)\.(\w+)...', s)
        from_, info, to = m[0]
        isConvertEnabled = self.sender().isEnabled()
        if isConvertEnabled:
            self.sender().setDisabled(True)
            self.convertWin = utilsConvert.convertFileFormatWin(
                parent=self,
                actionToEnable=self.sender(),
                mainWin=self, from_=from_, to=to,
                info=info
            )
            self.convertWin.show()
            self.convertWin.main()
        else:
            # self.convertWin.setWindowState(Qt.WindowNoState)
            self.convertWin.setWindowState(Qt.WindowActive)
            self.convertWin.raise_()
    
    def launchImageBatchConverter(self):
        self.batchConverterWin = utilsConvert.ImagesToPositions(parent=self)
        self.batchConverterWin.show()
    
    def launchRepeatDataPrep(self):
        self.batchConverterWin = utilsRepeat.repeatDataPrepWindow(parent=self)
        self.batchConverterWin.show()

    def launchDataStruct(self, checked=False):
        self.dataStructButton.setStyleSheet(
            f'QPushButton {{background-color: {self.moduleLaunchedColor};}}'
        )
        self.dataStructButton.setText(
            '0. Creating data structure running...'
        )

        QTimer.singleShot(100, self._showDataStructWin)

    # def attemptDataStructSeparateProcess(self):
    #     self.dataStructButton.setStyleSheet(
    #         f'QPushButton {{background-color: {self.moduleLaunchedColor};}}'
    #     )
    #     self.dataStructButton.setText(
    #         '0. Creating data structure running...'
    #     )
    #
    #     cellacdc_path = os.path.dirname(os.path.realpath(__file__))
    #     dataStruct_path = os.path.join(cellacdc_path, 'dataStruct.py')
    #
    #     # Due to javabridge limitation only one 'start_vm' can be called in
    #     # each process. To get around with this every data structure conversion
    #     # is launched in a separate process
    #     try:
    #         subprocess.run(
    #             [sys.executable, dataStruct_path], check=True, text=True,
    #             shell=False
    #         )
    #     except Exception as e:
    #         print('=========================================')
    #         traceback.print_exc()
    #         print('=========================================')
    #         err = ("""
    #         <p style="font-size:12px">
    #             Launching data structure module in a separate process failed.<br><br>
    #             Please restart Cell-ACDC if you need to use this module again.
    #         <p>
    #         """)
    #         self.dataStructButton.setStyleSheet(
    #             f'QPushButton {{background-color: {self.defaultPushButtonColor};}}')
    #         self.dataStructButton.setText(
    #             '0. Restart Cell-ACDC to enable module 0 again.')
    #         self.dataStructButton.setToolTip(
    #             'Due to an interal limitation of the Java Virtual Machine\n'
    #             'moduel 0 can be launched only once.\n'
    #             'To use it again close and reopen Cell-ACDC'
    #         )
    #         self.dataStructButton.setDisabled(True)
    #         return
    #
    #     self.dataStructButton.setStyleSheet(
    #         f'QPushButton {{background-color: {self.defaultPushButtonColor};}}')
    #     self.dataStructButton.setText(
    #         '0. Create data structure from microscopy file(s)...')
    #     self.dataStructButton.setDisabled(False)

    def _showDataStructWin(self):
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        bioformats_url = 'https://www.openmicroscopy.org/bio-formats/'
        txt = html_utils.paragraph(f"""
            Cell-ACDC can use <b>Bio-Formats</b> to read microscopy files 
            (more info {html_utils.href_tag('here', bioformats_url)}).<br><br>
            Bio-Formats requires Java and a python package called <code>javabridge</code>,<br>
            that will be automatically installed if missing.<br><br>
            We recommend using Bio-Formats, since it can read the metadata of the file,<br> 
            such as pixel size, numerical aperture etc.<br><br>
            However, if you <b>already pre-processed your microsocpy files into .tif 
            files</b>,<br>
            you can choose to simply re-structure them into the Cell-ACDC compatible 
            format.<br><br>
            How do you want to proceed?          
        """)
        useBioFormatsButton = QPushButton(
            QIcon(':ome.svg'), ' Use Bio-Formats ', msg
        )
        restructButton = QPushButton(
            QIcon(':folders.svg'), ' Re-structure image files ', msg
        )
        _, useBioFormatsButton, restructButton = msg.question(
            self, 'How to structure files', txt, 
            buttonsTexts=('Cancel', useBioFormatsButton, restructButton)
        )
        if msg.cancel:
            self.logger.info('Creating data structure process aborted by the user.')
            self.restoreDefaultButtons()
            return
        
        useBioFormats = msg.clickedButton == useBioFormatsButton
        if self.dataStructButton.isEnabled() and useBioFormats:
            self.dataStructButton.setText(
                '0. Restart Cell-ACDC to enable module 0 again.')
            self.dataStructButton.setToolTip(
                'Due to an interal limitation of the Java Virtual Machine\n'
                'moduel 0 can be launched only once.\n'
                'To use it again close and reopen Cell-ACDC'
            )
            self.dataStructButton.setDisabled(True)
            self.dataStructWin = dataStruct.createDataStructWin(
                parent=self, version=self._version
            )
            self.dataStructWin.show()
            self.dataStructWin.main()
        elif msg.clickedButton == restructButton:
            self.progressWin = apps.QDialogWorkerProgress(
                title='Re-structure image files log', parent=self,
                pbarDesc='Re-structuring image files running...'
            )
            self.progressWin.sigClosed.connect(self.progressWinClosed)
            self.progressWin.show(app)
            self.workerName = 'Re-structure image files'
            success = dataReStruct.run(self)
            if not success:
                self.progressWin.workerFinished = True
                self.progressWin.close()
                self.restoreDefaultButtons()
                self.logger.info('Re-structuring files NOT completed.')
    
    def progressWinClosed(self):
        self.progressWin = None
    
    def workerInitProgressbar(self, totalIter):
        if self.progressWin is None:
            return

        self.progressWin.mainPbar.setValue(0)
        if totalIter == 1:
            totalIter = 0
        self.progressWin.mainPbar.setMaximum(totalIter)
    
    def workerFinished(self, worker=None):
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph(
            f'{self.workerName} process finished.'
        )
        msg.information(self, 'Process finished', txt)

        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
        
        self.restoreDefaultButtons()
    
    @exception_handler
    def workerCritical(self, error):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
        raise error        
    
    def workerUpdateProgressbar(self, step):
        if self.progressWin is None:
            return

        self.progressWin.mainPbar.update(step)
    
    def workerProgress(self, text, loggerLevel='INFO'):
        if self.progressWin is not None:
            self.progressWin.logConsole.append(text)
        self.logger.log(getattr(logging, loggerLevel), text)

    def restoreDefaultButtons(self):
        self.dataStructButton.setStyleSheet(
            f'QPushButton {{background-color: {self.defaultPushButtonColor};}}'
        )
        self.dataStructButton.setText(
            '0. Create data structure from microscopy/image file(s)...'
        )


    def launchDataPrep(self, checked=False):
        c = self.dataPrepButton.palette().button().color().name()
        launchedColor = self.moduleLaunchedColor
        defaultColor = self.defaultPushButtonColor
        defaultText = self.defaultTextDataPrepButton
        if c != self.moduleLaunchedColor:
            self.dataPrepButton.setStyleSheet(
                f'QPushButton {{background-color: {launchedColor};}}')
            self.dataPrepButton.setText(
                'DataPrep is running. Click to restore window.'
            )
            self.dataPrepWin = dataPrep.dataPrepWin(
                buttonToRestore=(self.dataPrepButton, defaultColor, defaultText),
                mainWin=self, version=self._version
            )
            self.dataPrepWin.sigClose.connect(self.dataPrepClosed)
            self.dataPrepWin.show()
        else:
            # self.dataPrepWin.setWindowState(Qt.WindowNoState)
            self.dataPrepWin.setWindowState(Qt.WindowActive)
            self.dataPrepWin.raise_()
    
    def dataPrepClosed(self):
        self.logger.info('Data prep window closed.')
        del self.dataPrepWin

    def launchSegm(self, checked=False):
        c = self.segmButton.palette().button().color().name()
        launchedColor = self.moduleLaunchedColor
        defaultColor = self.defaultPushButtonColor
        defaultText = self.defaultTextSegmButton
        if c != self.moduleLaunchedColor:
            self.segmButton.setStyleSheet(
                f'QPushButton {{background-color: {launchedColor};}}')
            self.segmButton.setText('Segmentation is running. '
                                    'Check progress in the terminal/console')
            self.segmWin = segm.segmWin(
                buttonToRestore=(self.segmButton, defaultColor, defaultText),
                mainWin=self, version=self._version
            )
            self.segmWin.show()
            self.segmWin.main()
        else:
            # self.segmWin.setWindowState(Qt.WindowNoState)
            self.segmWin.setWindowState(Qt.WindowActive)
            self.segmWin.raise_()


    def launchGui(self, checked=False):
        self.logger.info('Opening GUI...')
        guiWin = gui.guiWin(self.app, mainWin=self, version=self._version)
        self.guiWins.append(guiWin)
        guiWin.sigClosed.connect(self.guiClosed)
        guiWin.run()
        
    def guiClosed(self, guiWin):
        self.guiWins.remove(guiWin)
    
    def launchSpotmaxGui(self, checked=False):
        self.logger.info('Launching spotMAX...')
        spotmaxWin = spotmaxRun.run_gui(app=self.app)
        spotmaxWin.sigClosed.connect(self.spotmaxGuiClosed)
        self.spotmaxWins.append(spotmaxWin)
    
    def spotmaxGuiClosed(self, spotmaxWin):
        self.spotmaxWins.remove(spotmaxWin)
        
    def guiClosed(self, guiWin):
        try:
            self.guiWins.remove(guiWin)
        except ValueError:
            pass

    def launchAlignUtil(self, checked=False):
        selectedExpPaths = self.getSelectedExpPaths(
            'Align frames in X and Y with phase cross-correlation'
        )
        if selectedExpPaths is None:
            return
        
        title = 'Align frames'
        infoText = 'Aligning frames in X and Y with phase cross-correlation...'
        progressDialogueTitle = 'Align frames'
        self.concatWindow = utilsAlign.alignWin(
            selectedExpPaths, self.app, title, infoText, progressDialogueTitle,
            parent=self
        )
        self.concatWindow.show()

    def launchConcatUtil(self, checked=False):
        selectedExpPaths = self.getSelectedExpPaths(
            'Concatenate acdc_output files'
        )
        if selectedExpPaths is None:
            return
        
        title = 'Concatenate acdc_output files'
        infoText = 'Launching concatenate acdc_output files process...'
        progressDialogueTitle = 'Concatenate acdc_output files'
        self.concatWindow = utilsConcat.concatWin(
            selectedExpPaths, self.app, title, infoText, progressDialogueTitle,
            parent=self
        )
        self.concatWindow.show()
    
    def show(self):
        super().show()
        h = self.dataPrepButton.geometry().height()
        f = 1.8
        self.dataStructButton.setMinimumHeight(int(h*f))
        self.dataPrepButton.setMinimumHeight(int(h*f))
        self.segmButton.setMinimumHeight(int(h*f))
        self.guiButton.setMinimumHeight(int(h*f))
        if hasattr(self, 'spotmaxButton'):
            self.spotmaxButton.setMinimumHeight(int(h*f))
        self.restartButton.setMinimumHeight(int(int(h*f)))
        self.closeButton.setMinimumHeight(int(int(h*f)))
        # iconWidth = int(self.closeButton.iconSize().width()*1.3)
        # self.closeButton.setIconSize(QSize(iconWidth, iconWidth))
        self.setColorsAndText()
        self.readSettings()

    def saveWindowGeometry(self):
        settings = QSettings('schmollerlab', 'acdc_main')
        settings.setValue("geometry", self.saveGeometry())

    def readSettings(self):
        settings = QSettings('schmollerlab', 'acdc_main')
        if settings.value('geometry') is not None:
            self.restoreGeometry(settings.value("geometry"))

    def checkOpenModules(self):
        c1 = self.dataPrepButton.palette().button().color().name()
        c2 = self.segmButton.palette().button().color().name()
        c3 = self.guiButton.palette().button().color().name()
        launchedColor = self.moduleLaunchedColor

        openModules = []
        if c1 == launchedColor:
            openModules.append(self.dataPrepWin)
        if c2 == launchedColor:
            openModules.append(self.segmWin)
        if self.guiWins:
            openModules.extend(self.guiWins)
        if self.spotmaxWins:
            openModules.extend(self.spotmaxWins)

        if not openModules:
            return True, openModules

        msg = widgets.myMessageBox()
        warn_txt = html_utils.paragraph(
            'There are still <b>other Cell-ACDC windows open</b>.<br><br>'
            'Are you sure you want to close everything?'
        )
        _, yesButton = msg.warning(
           self, 'Modules still open!', warn_txt, buttonsTexts=('Cancel', 'Yes')
        )

        return msg.clickedButton == yesButton, openModules

    def closeEvent(self, event):
        if self.welcomeGuide is not None:
            self.welcomeGuide.close()

        self.saveWindowGeometry()

        acceptClose, openModules = self.checkOpenModules()
        if acceptClose:
            for openModule in openModules:
                openModule.setWindowState(Qt.WindowActive)
                openModule.raise_()
                openModule.close()
                if openModule.isVisible():
                    event.ignore()
                    return
        else:
            event.ignore()
            return

        if self.sender() == self.restartButton:
            try:
                restart()
            except Exception as e:
                traceback.print_exc()
                print('-----------------------------------------')
                print('Failed to restart Cell-ACDC. Please restart manually')
        else:
            self.logger.info('**********************************************')
            self.logger.info(f'Cell-ACDC closed. {myutils.get_salute_string()}')
            self.logger.info('**********************************************')

def restart():
    QCoreApplication.quit()
    process = QtCore.QProcess()
    process.setProgram(sys.argv[0])
    # process.setStandardOutputFile(QProcess.nullDevice())
    status = process.startDetached()
    if status:
        print('Restarting Cell-ACDC...')

def run():
    from cellacdc.config import parser_args
    print('Launching application...')

    if not splashScreen.isVisible():
        splashScreen.show()
    
    win = mainWin(app)
    try:
        myutils.check_matplotlib_version(qparent=win)
    except Exception as e:
        pass
    version, success = myutils.read_version(
        logger=win.logger.info, return_success=True
    )
    if not success:
        error = myutils.install_package(
            'setuptools_scm', pypi_name='setuptools-scm'
        )
        if error:
            win.logger.info(error)
        else:
            version = myutils.read_version(logger=win.logger.info)
    win.setVersion(version)
    win.launchWelcomeGuide()
    win.show()
    try:
        win.welcomeGuide.showPage(win.welcomeGuide.welcomeItem)
    except AttributeError:
        pass
    win.logger.info('**********************************************')
    win.logger.info(f'Welcome to Cell-ACDC v{version}!')
    win.logger.info('**********************************************')
    win.logger.info('----------------------------------------------')
    win.logger.info('NOTE: If application is not visible, it is probably minimized\n'
          'or behind some other open window.')
    win.logger.info('----------------------------------------------')
    splashScreen.close()
    # splashScreenApp.quit()
    sys.exit(app.exec_())

def main():
    # Keep compatibility with users that installed older versions
    # where the entry point was main()
    run()

if __name__ == "__main__":
    run()
else:
    splashScreen.hide()