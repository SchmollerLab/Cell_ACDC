import os
import sys
import traceback
import re
import logging

import pandas as pd

from functools import partial

from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import (
    QMainWindow, QVBoxLayout, QPushButton, QLabel, QAction,
    QMenu, QHBoxLayout, QFileDialog, QGroupBox, QCheckBox, QSplashScreen
)
from qtpy.QtCore import (
    Qt, QProcess, Signal, Slot, QTimer, QSize,
    QSettings, QUrl, QCoreApplication
)
from qtpy.QtGui import (
    QFontDatabase, QIcon, QDesktopServices, QFont, QColor, 
    QPalette, QGuiApplication, QPixmap
)
import qtpy.compat

from . import (
    dataPrep, segm, gui, dataStruct, load, help, myutils,
    cite_url, html_utils, widgets, apps, dataReStruct
)
from .help import about
from .utils import concat as utilsConcat
from .utils import convert as utilsConvert
from .utils import rename as utilsRename
from .utils import align as utilsAlign
from .utils import compute as utilsCompute
from .utils import repeat as utilsRepeat
from .utils import toImageJroi as utilsToImageJroi
from .utils.resize import util as utilsResizePositionsUtil
from .utils import fromImageJroiToSegm as utilsFromImageJroi
from .utils import toObjCoords as utilsToObjCoords
from .utils import acdcToSymDiv as utilsSymDiv
from .utils import trackSubCellObjects as utilsTrackSubCell
from .utils import createConnected3Dsegm as utilsConnected3Dsegm
from .utils import filterObjFromCoordsTable as utilsFilterObjsFromTable
from .utils import stack2Dinto3Dsegm as utilsStack2Dto3D
from .utils import computeMultiChannel as utilsComputeMultiCh
from .utils import applyTrackFromTable as utilsApplyTrackFromTab
from .utils import applyTrackFromTrackMateXML as utilsApplyTrackFromTrackMate
from .info import utilsInfo
from . import is_win, is_linux, settings_folderpath, issues_url, is_mac
from . import settings_csv_path
from . import path
from . import printl
from . import _warnings
from . import exception_handler
from . import user_profile_path
from . import cellacdc_path

from . import qrc_resources

try:
    import spotmax
    from spotmax import _run as spotmaxRun
    spotmax_filepath = os.path.dirname(os.path.abspath(spotmax.__file__))
    spotmax_logo_path = os.path.join(
        spotmax_filepath, 'resources', 'spotMAX_logo.svg'
    )
    SPOTMAX_INSTALLED = True
except Exception as e:
    # traceback.print_exc()
    if not isinstance(e, ModuleNotFoundError):
        traceback.print_exc()
    SPOTMAX_INSTALLED = False

def restart():
    QCoreApplication.quit()
    process = QtCore.QProcess()
    process.setProgram(sys.argv[0])
    # process.setStandardOutputFile(QProcess.nullDevice())
    status = process.startDetached()
    if status:
        print('Restarting Cell-ACDC...')

class mainWin(QMainWindow):
    def __init__(self, app, parent=None):
        self.checkConfigFiles()
        self.app = app
        scheme = self.getColorScheme()
        self.welcomeGuide = None
        self._do_restart = False
        
        super().__init__(parent)
        self.setWindowTitle("Cell-ACDC")
        self.setWindowIcon(QIcon(":icon.ico"))
        self.setAcceptDrops(True)
        
        self.checkUserDataFolderPath = True

        logger, logs_path, log_path, log_filename = myutils.setupLogger(
            module='main'
        )
        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path        
        
        if not is_linux:
            self.loadFonts()

        self.addStatusBar(scheme)
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
        
        modulesButtonsGroupBox = QGroupBox()
        modulesButtonsGroupBox.setTitle('Modules')
        modulesButtonsGroupBoxLayout = QVBoxLayout()
        modulesButtonsGroupBox.setLayout(modulesButtonsGroupBoxLayout)
        
        dataStructButton = widgets.setPushButton(
            '  0. Create data structure from microscopy/image file(s)...  '
        )
        dataStructButton.setIconSize(QSize(iconSize,iconSize))
        font = QFont()
        font.setPixelSize(13)
        dataStructButton.setFont(font)
        dataStructButton.clicked.connect(self.launchDataStruct)
        self.dataStructButton = dataStructButton
        modulesButtonsGroupBoxLayout.addWidget(dataStructButton)

        dataPrepButton = QPushButton('  1. Launch data prep module...')
        dataPrepButton.setIcon(QIcon(':prep.svg'))
        dataPrepButton.setIconSize(QSize(iconSize,iconSize))
        font = QFont()
        font.setPixelSize(13)
        dataPrepButton.setFont(font)
        dataPrepButton.clicked.connect(self.launchDataPrep)
        self.dataPrepButton = dataPrepButton
        modulesButtonsGroupBoxLayout.addWidget(dataPrepButton)

        segmButton = QPushButton('  2. Launch segmentation module...')
        segmButton.setIcon(QIcon(':segment.svg'))
        segmButton.setIconSize(QSize(iconSize,iconSize))
        segmButton.setFont(font)
        segmButton.clicked.connect(self.launchSegm)
        self.segmButton = segmButton
        modulesButtonsGroupBoxLayout.addWidget(segmButton)

        guiButton = QPushButton('  3. Launch GUI...')
        guiButton.setIcon(QIcon(':icon.ico'))
        guiButton.setIconSize(QSize(iconSize,iconSize))
        guiButton.setFont(font)
        guiButton.clicked.connect(self.launchGui)
        self.guiButton = guiButton
        modulesButtonsGroupBoxLayout.addWidget(guiButton)

        if SPOTMAX_INSTALLED:
            spotmaxButton = QPushButton('  4. Launch spotMAX...')
            spotmaxButton.setIcon(QIcon(spotmax_logo_path))
            spotmaxButton.setIconSize(QSize(iconSize,iconSize))
            spotmaxButton.setFont(font)
            self.spotmaxButton = spotmaxButton
            spotmaxButton.clicked.connect(self.launchSpotmaxGui)
            modulesButtonsGroupBoxLayout.addWidget(spotmaxButton)
        
        mainLayout.addWidget(modulesButtonsGroupBox)
        mainLayout.addSpacing(10)
        
        controlsButtonsGroupBox = QGroupBox()
        controlsButtonsGroupBox.setTitle('Controls')
        controlsButtonsGroupBoxLayout = QVBoxLayout()
        controlsButtonsGroupBox.setLayout(controlsButtonsGroupBoxLayout)
        
        showAllWindowsButton = QPushButton('  Restore open windows')
        showAllWindowsButton.setIcon(QIcon(':eye.svg'))
        showAllWindowsButton.setIconSize(QSize(iconSize,iconSize))
        showAllWindowsButton.setFont(font)
        self.showAllWindowsButton = showAllWindowsButton
        showAllWindowsButton.clicked.connect(self.showAllWindows)
        # showAllWindowsButton.setDisabled(True)

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
        self.restartButton.hide()
        closeLayout.addWidget(restartButton)
        
        closeLayout.addWidget(showAllWindowsButton)

        closeButton = QPushButton(QIcon(":close.svg"), '  Close application')
        closeButton.setIconSize(QSize(iconSize, iconSize))
        self.closeButton = closeButton
        # closeButton.setIconSize(QSize(24,24))
        closeButton.setFont(font)
        closeButton.clicked.connect(self.close)
        closeLayout.addWidget(closeButton)

        controlsButtonsGroupBoxLayout.addLayout(closeLayout)
        
        mainLayout.addWidget(controlsButtonsGroupBox)
        
        mainContainer.setLayout(mainLayout)

        self.guiWins = []
        self.spotmaxWins = []
        self.dataPrepWins = []
        self._version = None
        self.progressWin = None
        self.forceClose = False
    
    def addStatusBar(self, scheme):
        self.statusbar = self.statusBar()
        # Permanent widget
        label = QLabel('Dark mode')
        widget = QtWidgets.QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)
        layout.addWidget(label)
        self.darkModeToggle = widgets.Toggle(label_text='Dark mode')
        self.darkModeToggle.ignoreEvent = False
        self.darkModeToggle.warnMessageBox  = True
        if scheme == 'dark':
            self.darkModeToggle.ignoreEvent = True
            self.darkModeToggle.setChecked(True)
        self.darkModeToggle.toggled.connect(self.onDarkModeToggled)
        layout.addWidget(self.darkModeToggle)
        self.statusBarLayout = layout
        self.statusbar.addWidget(widget)
    
    def getColorScheme(self):
        from ._palettes import get_color_scheme
        return get_color_scheme()
    
    def onDarkModeToggled(self, checked):
        if self.darkModeToggle.ignoreEvent:
            self.darkModeToggle.ignoreEvent = False
            return
        from ._palettes import getPaletteColorScheme
        scheme = 'dark' if checked else 'light'
        load.rename_qrc_resources_file(scheme)
        if not os.path.exists(settings_csv_path):
            df_settings = pd.DataFrame(
                {'setting': [], 'value': []}).set_index('setting')
        else:
            df_settings = pd.read_csv(settings_csv_path, index_col='setting')
        df_settings.at['colorScheme', 'value'] = scheme
        df_settings.to_csv(settings_csv_path)
        if self.darkModeToggle.warnMessageBox:
            _warnings.warnRestartCellACDCcolorModeToggled(
                scheme, app_name='Cell-ACDC', parent=self
            )
            self.darkModeToggle.warnMessageBox = True
        self.setStatusBarRestartCellACDC()
    
    def setStatusBarRestartCellACDC(self):
        self.statusBarLayout.addWidget(QLabel(html_utils.paragraph(
            '<i>Restart Cell-ACDC for the change to take effect</i>', 
            font_color='red'
        )))
    
    def checkConfigFiles(self):
        print('Loading configuration files...')
        paths_to_check = [
            gui.favourite_func_metrics_csv_path, 
            # gui.custom_annot_path, 
            gui.shortcut_filepath, 
            os.path.join(settings_folderpath, 'recentPaths.csv'), 
            load.last_entries_metadata_path, 
            load.additional_metadata_path, 
            load.last_selected_groupboxes_measurements_path
        ]
        for path in paths_to_check:
            load.remove_duplicates_file(path)
    
    def dragEnterEvent(self, event) -> None:
        ...
    
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
        QFontDatabase.addApplicationFont(":Helvetica-Regular.ttf")
        QFontDatabase.addApplicationFont(":Helvetica-Bold.ttf")
        QFontDatabase.addApplicationFont(":Helvetica-Italic.ttf")
        QFontDatabase.addApplicationFont(":Helvetica-BoldItalic.ttf")

    def launchWelcomeGuide(self, checked=False):
        if not os.path.exists(settings_csv_path):
            idx = ['showWelcomeGuide']
            values = ['Yes']
            self.df_settings = pd.DataFrame(
                {'setting': idx, 'value': values}).set_index('setting')
            self.df_settings.to_csv(settings_csv_path)
        self.df_settings = pd.read_csv(settings_csv_path, index_col='setting')
        if 'showWelcomeGuide' not in self.df_settings.index:
            self.df_settings.at['showWelcomeGuide', 'value'] = 'Yes'
            self.df_settings.to_csv(settings_csv_path)

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
        self.moduleLaunchedColor = '#f1dd00'
        self.moduleLaunchedQColor = QColor(self.moduleLaunchedColor)
        defaultColor = self.guiButton.palette().button().color().name()
        self.defaultButtonPalette = self.guiButton.palette()
        self.defaultPushButtonColor = defaultColor
        self.defaultTextDataStructButton = self.dataStructButton.text()
        self.defaultTextGuiButton = self.guiButton.text()
        self.defaultTextDataPrepButton = self.dataPrepButton.text()
        self.defaultTextSegmButton = self.segmButton.text()
        self.moduleLaunchedPalette = self.guiButton.palette()
        self.moduleLaunchedPalette.setColor(
            QPalette.Button, self.moduleLaunchedQColor
        )
        self.moduleLaunchedPalette.setColor(
            QPalette.ButtonText, QColor(0, 0, 0)
        )

    def createMenuBar(self):
        menuBar = self.menuBar()

        self.recentPathsMenu = QMenu("&Recent paths", self)
        # On macOS an empty menu would not appear --> add dummy action
        self.recentPathsMenu.addAction('dummy macos')
        menuBar.addMenu(self.recentPathsMenu)

        utilsMenu = menuBar.addMenu("&Utilities")

        convertMenu = utilsMenu.addMenu('Convert file formats')
        convertMenu.addAction(self.npzToNpyAction)
        convertMenu.addAction(self.npzToTiffAction)
        convertMenu.addAction(self.TiffToNpzAction)
        convertMenu.addAction(self.h5ToNpzAction)
        convertMenu.addAction(self.toImageJroiAction)
        convertMenu.addAction(self.fromImageJroiAction)
        convertMenu.addAction(self.toObjsCoordsAction)

        segmMenu = utilsMenu.addMenu('Segmentation')
        segmMenu.addAction(self.createConnected3Dsegm)
        segmMenu.addAction(self.stack2Dto3DsegmAction)
        segmMenu.addAction(self.filterObjsFromTableAction)

        trackingMenu = utilsMenu.addMenu('Tracking and lineage')
        trackingMenu.addAction(self.trackSubCellFeaturesAction)
        trackingMenu.addAction(self.applyTrackingFromTableAction)
        trackingMenu.addAction(self.applyTrackingFromTrackMateXMLAction)
        trackingMenu.addAction(self.toSymDivAction)        
        
        self.trackingMenu = trackingMenu

        measurementsMenu = utilsMenu.addMenu('Measurements')
        measurementsMenu.addAction(self.calcMetricsAcdcDf)
        measurementsMenu.addAction(self.combineMetricsMultiChannelAction) 
        
        concatMenu = utilsMenu.addMenu('Concatenate')
        concatMenu.addAction(self.concatAcdcDfsAction)    
        if SPOTMAX_INSTALLED:
            concatMenu.addAction(self.concatSpotmaxDfsAction) 

        dataPrepMenu = utilsMenu.addMenu('Pre-processing')
                 
        dataPrepMenu.addAction(self.batchConverterAction)
        dataPrepMenu.addAction(self.repeatDataPrepAction)
        dataPrepMenu.addAction(self.alignAction)
        dataPrepMenu.addAction(self.resizeImagesAction)
        
        utilsMenu.addAction(self.renameAction)

        self.utilsMenu = utilsMenu

        utilsMenu.addSeparator()
        utilsHelpAction = utilsMenu.addAction('Help...')
        utilsHelpAction.triggered.connect(self.showUtilsHelp)
    
        menuBar.addMenu(utilsMenu)
        
        self.settingsMenu = QMenu("&Settings", self)
        self.settingsMenu.addAction(self.changeUserProfileFolderPathAction)
        self.settingsMenu.addAction(self.resetUserProfileFolderPathAction)
        menuBar.addMenu(self.settingsMenu)

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
        if SPOTMAX_INSTALLED:
            helpMenu.addAction(self.aboutSmaxAction)
        
        utilsMenu.addAction(self.debugAction)
        self.debugAction.setVisible(False)

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
    
    def resetUserProfileFolderPath(self):
        from . import user_profile_path, user_home_path
        
        if os.path.samefile(user_profile_path, user_home_path):
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                'The user profile data is already in the default folder.'
            )
            msg.warning(self, 'Reset user profile data', txt)
            return
        
        acdc_folders = load.get_all_acdc_folders(user_profile_path)
        acdc_folders_format = [
            f'&nbsp;&nbsp;&nbsp;{folder}' for folder in acdc_folders
        ]
        acdc_folders_format = '<br>'.join(acdc_folders_format)
        
        txt = (f"""
            Current user profile path:<br><br>
            <code>{user_profile_path}</code><br><br>
            The user profile contains the following Cell-ACDC folders:<br><br>
            <code>{acdc_folders_format}</code><br><br>
            After clicking "Ok" you <b>Cell-ACDC will migrate</b> 
            the user profile data to the following folder:<br><br>
            <code>{user_home_path}</code>.<br>
        """)
        
        txt = html_utils.paragraph(txt)
        
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(
            self, 'Reset default user profile folder path', txt, 
            buttonsTexts=('Cancel', 'Ok')
        )
        if msg.cancel:
            self.logger.info('Resetting user profile folder path cancelled.')
            return
        
        
        new_user_profile_path = user_home_path
        
        self.startMigrateUserProfileWorker(
            user_profile_path, new_user_profile_path, acdc_folders
        )
    
    def changeUserProfileFolderPath(self):        
        acdc_folders = load.get_all_acdc_folders(user_profile_path)
        acdc_folders_format = [
            f'&nbsp;&nbsp;&nbsp;{folder}' for folder in acdc_folders
        ]
        acdc_folders_format = '<br>'.join(acdc_folders_format)
        
        txt = (f"""
            Current user profile path:<br><br>
            <code>{user_profile_path}</code><br><br>
            The user profile contains the following Cell-ACDC folders:<br><br>
            <code>{acdc_folders_format}</code><br><br>
            After clicking "Ok" you will be <b>asked to select the folder</b> where 
            you want to <b>migrate</b> the user profile data.<br>
        """)
        
        txt = html_utils.paragraph(txt)
        
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(
            self, 'Change user profile folder path', txt, 
            buttonsTexts=('Cancel', 'Ok')
        )
        if msg.cancel:
            self.logger.info('Changing user profile folder path cancelled.')
            return

        from qtpy.compat import getexistingdirectory
        new_user_profile_path = getexistingdirectory(
            parent=self,
            caption='Select folder for user profile data', 
            basedir=user_profile_path
        )
        if not new_user_profile_path:
            self.logger.info('Changing user profile folder path cancelled.')
            return
        
        if os.path.samefile(user_profile_path, new_user_profile_path):
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                'The user profile data is already in the selected folder.'
            )
            msg.warning(self, 'Change user profile data folder', txt)
            return
        
        self.startMigrateUserProfileWorker(
            user_profile_path, new_user_profile_path, acdc_folders
        )
        
    def startMigrateUserProfileWorker(self, src_path, dst_path, acdc_folders):
        self.progressWin = apps.QDialogWorkerProgress(
            title='Migrate user profile data', parent=self,
            pbarDesc='Migrating user profile data...',
            showInnerPbar=True
        )
        self.progressWin.sigClosed.connect(self.progressWinClosed)
        self.progressWin.show(self.app)
        
        from . import workers
        self.workerName = 'Migrating user profile data'
        self._thread = QtCore.QThread()
        self.migrateWorker = workers.MigrateUserProfileWorker(
            src_path, dst_path, acdc_folders
        )
        self.migrateWorker.moveToThread(self._thread)
        self.migrateWorker.finished.connect(self._thread.quit)
        self.migrateWorker.finished.connect(self.migrateWorker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        
        self.migrateWorker.progress.connect(self.workerProgress)
        self.migrateWorker.critical.connect(self.workerCritical)
        self.migrateWorker.finished.connect(self.migrateWorkerFinished)
        
        self.migrateWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.migrateWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.migrateWorker.signals.sigInitInnerPbar.connect(
            self.workerInitInnerPbar
        )
        self.migrateWorker.signals.sigUpdateInnerPbar.connect(
            self.workerUpdateInnerPbar
        )
        
        self._thread.started.connect(self.migrateWorker.run)
        self._thread.start()
    
    def workerInitProgressbar(self, totalIter):
        self.progressWin.mainPbar.setValue(0)
        if totalIter == 1:
            totalIter = 0
        self.progressWin.mainPbar.setMaximum(totalIter)

    def workerUpdateProgressbar(self, step):
        self.progressWin.mainPbar.update(step)
    
    def workerInitInnerPbar(self, totalIter):
        self.progressWin.innerPbar.setValue(0)
        if totalIter == 1:
            totalIter = 0
        self.progressWin.innerPbar.setMaximum(totalIter)
    
    def workerUpdateInnerPbar(self, step):
        self.progressWin.innerPbar.update(step)
    
    def migrateWorkerFinished(self, worker):
        self.workerFinished()
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            To make this change effective, please <b>restart</b> Cell-ACDC.<br><br>
            Thanks!
        """)
        self.statusBarLayout.addWidget(QLabel(html_utils.paragraph(
            '<i>Restart Cell-ACDC for the change to take effect</i>', 
            font_color='red'
        )))
        msg.information(self, 'Restart Cell-ACDC', txt)
    
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
        self.changeUserProfileFolderPathAction = QAction(
            'Change user profile path...'
        )
        self.resetUserProfileFolderPathAction = QAction(
            'Reset default user profile path'
        )
        self.npzToNpyAction = QAction('Convert .npz file(s) to .npy...')
        self.npzToTiffAction = QAction('Convert .npz file(s) to .tif...')
        self.TiffToNpzAction = QAction('Convert .tif file(s) to _segm.npz...')
        self.h5ToNpzAction = QAction('Convert .h5 file(s) to _segm.npz...')
        self.toImageJroiAction = QAction(
            'Convert Cell-ACDC segmentation file(s) (segm.npz) to ImageJ ROIs...'
        )
        self.fromImageJroiAction = QAction(
            'Convert ImageJ ROIs to Cell-ACDC segmentation file(s) (segm.npz)...'
        )
        self.toObjsCoordsAction = QAction(
            'Convert .npz segmentation file(s) to object coordinates (CSV)...'
        )
        self.createConnected3Dsegm = QAction(
            'Create connected 3D segmentation mask from z-slices segmentation...'
        ) 
        self.filterObjsFromTableAction = QAction(
            'Filter segmented objects using a table of coordinates (e.g., centroids)...'
        ) 
        self.stack2Dto3DsegmAction = QAction(
            'Stack 2D segmentation objects into 3D objects...'
        )  
        self.trackSubCellFeaturesAction = QAction(
            'Track and/or count sub-cellular objects (assign same ID as the '
            'cell they belong to)...'
        )    
        self.applyTrackingFromTableAction = QAction(
            'Apply tracking info from tabular data...'
        )
        self.applyTrackingFromTrackMateXMLAction = QAction(
            'Apply tracking info from TrackMate XML file...'
        )
        self.batchConverterAction = QAction(
            'Create required data structure from image files...'
        )
        self.repeatDataPrepAction = QAction(
            'Re-apply data prep steps to selected channels...'
        )
        # self.TiffToHDFAction = QAction('Convert .tif file(s) to .h5py...')
        self.concatAcdcDfsAction = QAction(
            'Concatenate acdc output tables from multiple Positions and experiments...'
        )
        if SPOTMAX_INSTALLED:
            self.concatSpotmaxDfsAction = QAction(
                'Concatenate spotMAX output tables from multiple Positions and experiments...'
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

        self.resizeImagesAction = QAction(
            'Reisize images (downscale or upscale) in one or more experiments...'
        )
        self.welcomeGuideAction = QAction('Welcome Guide')
        self.userManualAction = QAction('User documentation...')
        self.aboutAction = QAction('About Cell-ACDC')
        self.citeAction = QAction('Cite us...')
        self.contributeAction = QAction('Contribute...')
        self.showLogsAction = QAction('Show log files...')
        
        if SPOTMAX_INSTALLED:
            self.aboutSmaxAction = QAction('About SpotMAX')
        
        self.debugAction = QAction('Daje de mac')

    def connectActions(self):
        self.changeUserProfileFolderPathAction.triggered.connect(
            self.changeUserProfileFolderPath
        )
        self.resetUserProfileFolderPathAction.triggered.connect(
            self.resetUserProfileFolderPath
        )
        self.alignAction.triggered.connect(self.launchAlignUtil)
        self.concatAcdcDfsAction.triggered.connect(self.launchConcatUtil)
        if SPOTMAX_INSTALLED:
            self.concatSpotmaxDfsAction.triggered.connect(
                self.launchConcatSpotmaxUtil
            )
        self.npzToNpyAction.triggered.connect(self.launchConvertFormatUtil)
        self.npzToTiffAction.triggered.connect(self.launchConvertFormatUtil)
        self.TiffToNpzAction.triggered.connect(self.launchConvertFormatUtil)
        self.h5ToNpzAction.triggered.connect(self.launchConvertFormatUtil)
        self.fromImageJroiAction.triggered.connect(
            self.launchFromImageJroiToSegmUtil
        )
        self.resizeImagesAction.triggered.connect(self.launchResizeUtil)
        self.toImageJroiAction.triggered.connect(self.launchToImageJroiUtil)
        self.toObjsCoordsAction.triggered.connect(
            self.launchToObjectsCoordsUtil
        )
        self.createConnected3Dsegm.triggered.connect(
            self.launchConnected3DsegmActionUtil
        )
        self.filterObjsFromTableAction.triggered.connect(
            self.launchFilterObjsFromTableActionUtil
        )
        self.stack2Dto3DsegmAction.triggered.connect(
            self.launchStack2Dto3DsegmActionUtil
        )
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
        if SPOTMAX_INSTALLED:
            self.aboutSmaxAction.triggered.connect(self.showAboutSmax)

        self.userManualAction.triggered.connect(myutils.browse_docs)
        self.contributeAction.triggered.connect(self.showContribute)
        self.citeAction.triggered.connect(
            partial(QDesktopServices.openUrl, QUrl(cite_url))
        )
        self.recentPathsMenu.aboutToShow.connect(self.populateOpenRecent)
        self.showLogsAction.triggered.connect(self.showLogFiles)
        self.applyTrackingFromTableAction.triggered.connect(
            self.launchApplyTrackingFromTableUtil
        )
        self.applyTrackingFromTrackMateXMLAction.triggered.connect(
            self.launchApplyTrackingFromTrackMateXML
        )
        
        self.debugAction.triggered.connect(self._debug)
    
    def showLogFiles(self):
        logs_path = myutils.get_logs_path()
        myutils.showInExplorer(logs_path)

    def populateOpenRecent(self):
        # Step 0. Remove the old options from the menu
        self.recentPathsMenu.clear()
        # Step 1. Read recent Paths
        recentPaths_path = os.path.join(settings_folderpath, 'recentPaths.csv')
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
    
    def showAboutSmax(self):
        from spotmax.dialogs import AboutSpotMAXDialog
        win = AboutSpotMAXDialog(parent=self)
        win.exec_()
    
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
        else:
            posPath = os.path.join(exp_path, posFolders[0])
        
        return posPath

    def getSelectedExpPaths(self, utilityName, exp_folderpath=None):
        # self._debug()
        
        if exp_folderpath is None:
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
        warn_exp_already_selected = True
        while True:
            if exp_folderpath is None:
                exp_path = qtpy.compat.getexistingdirectory(
                    parent=self, 
                    caption='Select experiment folder containing Position_n folders',
                    basedir=mostRecentPath,
                    # options=QFileDialog.DontUseNativeDialog
                )
                if not exp_path:
                    break
                myutils.addToRecentPaths(exp_path)
            else: 
                exp_path = exp_folderpath
            selected_path = exp_path
            baseFolder = os.path.basename(exp_path)
            isPosFolder = myutils.is_pos_folderpath(exp_path)
            isImagesFolder = baseFolder == 'Images'
            if isImagesFolder:
                posPath = os.path.dirname(exp_path)
                posFolders = [os.path.basename(posPath)]
                exp_path = os.path.dirname(posPath)
                selected_exp_paths = {exp_path:posFolders}
            elif isPosFolder:
                posPath = exp_path
                posFolders = [os.path.basename(posPath)]
                exp_path = os.path.dirname(exp_path)
                selected_exp_paths = {exp_path:posFolders}
            else:
                self.logger.info(f'Scanning selected folder "{exp_path}"...')
                selected_exp_paths = path.get_posfolderpaths_walk(exp_path)
                if not selected_exp_paths:
                    cancel = self.warnNoValidExpPaths(exp_path)
                    if cancel:
                        self.logger.info(f'{utilityName} aborted by the user.')
                        return
                    continue
            
            is_multi_pos = False
            for exp_path, pos_folders in selected_exp_paths.items():
                if exp_path in expPaths:
                    if warn_exp_already_selected:
                        proceed = self.warnExpPathAlreadySelected(
                            selected_path, exp_path
                        )
                        if not proceed:
                            self.logger.info(f'{utilityName} aborted by the user.')
                            return
                        warn_exp_already_selected = False
                    expPaths[exp_path].extend(pos_folders)
                else:
                    expPaths[exp_path] = pos_folders
                
                if len(pos_folders) > 1 and not is_multi_pos:
                    is_multi_pos = True
            
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
        
        if not expPaths:
            self.logger.info(f'{utilityName} aborted by the user.')
            return

        if len(expPaths) > 1 or is_multi_pos:
            infoPaths = self.getInfoPosStatus(expPaths)
            selectPosWin = apps.selectPositionsMultiExp(
                expPaths, 
                infoPaths=infoPaths, 
                parent=self
            )
            selectPosWin.exec_()
            if selectPosWin.cancel:
                self.logger.info(f'{utilityName} aborted by the user.')
                return
            selectedExpPaths = selectPosWin.selectedPaths
        else:
            selectedExpPaths = expPaths
        
        return selectedExpPaths
    
    def warnNoValidExpPaths(self, selected_path):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            The selected folder does 
            <b>not contain any valid experiment folders</b>.
        """)
        command = selected_path.replace('\\', os.sep)
        command = selected_path.replace('/', os.sep)
        msg.warning(
            self, 'No valid folders found', txt,
            buttonsTexts=('Cancel', 'Try again'), 
            commands=(command,), 
            path_to_browse=selected_path
        )
        return msg.cancel
    
    def warnExpPathAlreadySelected(self, selected_path, exp_path):
        selected_text = myutils.to_relative_path(selected_path)
        exp_text = myutils.to_relative_path(exp_path)
        txt = html_utils.paragraph(f""" 
            The experiment folder of the selected path was already previously selected.<br><br>
            Are you adding Position folders one by one? If yes, you do not 
            need to do that.<br><br>
            Simply select the parent folder containing the Position 
            folders and<br>
            Cell-ACDC will ask you later which Positions you want 
            to concatenate.<br><br>
            Do you want to continue?<br><br>
            Selected path: <code>{selected_text}</code><br><br>
            Experiment path: <code>{exp_text}</code>
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Folder already selected!', txt, 
            buttonsTexts=('Cancel', 'Yes'), 
            path_to_browse=selected_path
        )
        return not msg.cancel
    
    def _debug(self):
        printl('ciao')
        from qtpy.QtWidgets import QDialog, QTreeWidget, QTreeWidgetItem
        secondWin = QDialog(self)

        layout = QVBoxLayout()

        widget = QTreeWidget()
        item = QTreeWidgetItem(['ciao'])
        widget.addTopLevelItem(item)
        
        layout.addWidget(widget)
        secondWin.setLayout(layout)
        secondWin.exec_()
    
    def askRestartAcdc(self):
        txt = html_utils.paragraph(
            'Are you sure you want to restart Cell-ACDC?<br>'
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Restart?', txt, buttonsTexts=('Cancel', 'Yes'))
        return msg.cancel
    
    def keyPressEvent(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl_shift = modifiers == Qt.ControlModifier | Qt.ShiftModifier
        if ctrl_shift and event.key() == Qt.Key_R:
            cancel = self.askRestartAcdc()
            if not cancel:
                event.ignore()
                self._do_restart = True
                self.close()
                return
        return super().keyPressEvent(event)
    
    def launchApplyTrackingFromTrackMateXML(self):
        posPath = self.getSelectedPosPath('Apply tracking info from tabular data')
        if posPath is None:
            return
        
        title = 'Apply tracking info from TrackMate XML file utility'
        infoText = 'Launching apply tracking info from from TrackMate XML data...'
        self.applyTrackMateXMLWin = (
            utilsApplyTrackFromTrackMate.ApplyTrackingInfoFromTrackMateUtil(
                self.app, title, infoText, parent=self, 
                callbackOnFinished=self.applyTrackingFromTackmateXMLFinished
            )
        )
        self.applyTrackMateXMLWin.show()
        func = partial(
            self._runApplyTrackingFromTrackMateXML, posPath, 
            self.applyTrackMateXMLWin
        )
        QTimer.singleShot(200, func)
    
    def _runApplyTrackingFromTrackMateXML(self, posPath, win):
        success = win.run(posPath)
        if not success:
            self.logger.info(
                'Apply tracking info from TrackMate XML ABORTED by the user.'
            )
            win.close()  
    
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
        func = partial(
            self._runApplyTrackingFromTableUtil, posPath, self.applyTrackWin
        )
        QTimer.singleShot(200, func)

    def _runApplyTrackingFromTableUtil(self, posPath, win):
        success = win.run(posPath)
        if not success:
            self.logger.info(
                'Apply tracking info from tabular data ABORTED by the user.'
            )
            win.close()          
        
    def applyTrackingFromTackmateXMLFinished(self):
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph(
            'Apply tracking info from TrackMate XML data completed.'
        )
        msg.information(self, 'Process completed', txt)
        self.logger.info('Apply tracking info from TrackMate XML data completed.')
        self.applyTrackMateXMLWin.close()
    
    def applyTrackingFromTableFinished(self):
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph(
            'Apply tracking info from tabular data completed.'
        )
        msg.information(self, 'Process completed', txt)
        self.logger.info('Apply tracking info from tabular data completed.')
        self.applyTrackWin.close()
    
    def launchNapariUtil(self, action):
        myutils.check_install_package('napari', parent=self)
        if action == self.arboretumAction:
            self._launchArboretum()

    def _launchArboretum(self):
        myutils.check_install_package('napari_arboretum', parent=self)

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
    
    def launchToObjectsCoordsUtil(self):
        self.logger.info(f'Launching utility "{self.sender().text()}"')

        selectedExpPaths = self.getSelectedExpPaths(
            'From _segm.npz to objects coordinates (CSV)'
        )
        if selectedExpPaths is None:
            return
        
        title = 'Convert _segm.npz file(s) to objects coordinates (CSV)'
        infoText = 'Launching to to objects coordinates process...'
        progressDialogueTitle = (
            'Converting _segm.npz file(s) to to objects coordinates (CSV)'
        )
        self.toObjCoordsWin = utilsToObjCoords.toObjCoordsUtil(
            selectedExpPaths, self.app, title, infoText, progressDialogueTitle,
            parent=self
        )
        self.toObjCoordsWin.show()
    
    def launchFromImageJroiToSegmUtil(self):
        self.logger.info(f'Launching utility "{self.sender().text()}"')
        myutils.check_install_package('roifile', parent=self)

        import roifile

        selectedExpPaths = self.getSelectedExpPaths(
            'From ImageJ ROIs to _segm.npz'
        )
        if selectedExpPaths is None:
            return
        
        title = 'Convert ImageJ ROIs to _segm.npz file(s)'
        infoText = 'Launching ImageJ ROIs conversion process...'
        progressDialogueTitle = 'Converting ImageJ ROIs to _segm.npz file(s)'
        self.toImageJroiWin = utilsFromImageJroi.fromImageJRoiToSegmUtil(
            selectedExpPaths, self.app, title, infoText, progressDialogueTitle,
            parent=self
        )
        self.toImageJroiWin.show() 
    
    def launchResizeUtil(self):
        self.logger.info(f'Launching utility "{self.sender().text()}"')
        
        selectedExpPaths = self.getSelectedExpPaths(
            'From _segm.npz to ImageJ ROIs'
        )
        if selectedExpPaths is None:
            return
        
        title = 'Resize images'
        infoText = 'Launching resizing images process...'
        progressDialogueTitle = 'Resize images'
        self.resizeUtilWin = utilsResizePositionsUtil.ResizePositionsUtil(
            selectedExpPaths, self.app, title, infoText, progressDialogueTitle,
            parent=self
        )
        self.resizeUtilWin.show()
    
    def launchToImageJroiUtil(self):
        self.logger.info(f'Launching utility "{self.sender().text()}"')
        myutils.check_install_package('roifile', parent=self)

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
        self.logger.info(f'Launching utility "{self.sender().text()}"')
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
    
    def launchConnected3DsegmActionUtil(self):
        self.logger.info(f'Launching utility "{self.sender().text()}"')
        selectedExpPaths = self.getSelectedExpPaths(
            'Create connected 3D segmentation mask'
        )
        if selectedExpPaths is None:
            return
        
        title = 'Create connected 3D segmentation mask'
        infoText = 'Launching connected 3D segmentation mask creation process...'
        progressDialogueTitle = 'Creating connected 3D segmentation mask'
        self.connected3DsegmWin = utilsConnected3Dsegm.CreateConnected3Dsegm(
            selectedExpPaths, self.app, title, infoText, progressDialogueTitle,
            parent=self
        )
        self.connected3DsegmWin.show()
    
    def launchFilterObjsFromTableActionUtil(self):
        self.logger.info(f'Launching utility "{self.sender().text()}"')
        selectedExpPaths = self.getSelectedExpPaths(
            'Create connected 3D segmentation mask'
        )
        if selectedExpPaths is None:
            return
        
        title = 'Filter segmented objects from coordinates'
        infoText = 'Launching Filter segmented objects from coordinates process...'
        progressDialogueTitle = 'Filtering objects'
        self.filterObjsFromTableWin = (
                utilsFilterObjsFromTable.FilterObjsFromCoordsTable(
                selectedExpPaths, self.app, title, infoText, 
                progressDialogueTitle, parent=self
            )
        )
        self.filterObjsFromTableWin.show()
    
    def launchStack2Dto3DsegmActionUtil(self):
        self.logger.info(f'Launching utility "{self.sender().text()}"')
        selectedExpPaths = self.getSelectedExpPaths(
            'Create 3D segmentation mask from 2D'
        )
        if selectedExpPaths is None:
            return
        
        SizeZwin = apps.NumericEntryDialog(
            title='Number of z-slices', 
            instructions='Enter number of z-slices required',
            currentValue=1, parent=self, 
            stretch=True
        )
        SizeZwin.exec_()
        if SizeZwin.cancel:
            return
        
        title = 'Create stacked 3D segmentation mask'
        infoText = 'Launching stacked 3D segmentation mask creation process...'
        progressDialogueTitle = 'Creating stacked 3D segmentation mask'
        self.stack2DsegmWin = utilsStack2Dto3D.Stack2DsegmTo3Dsegm(
            selectedExpPaths, self.app, title, infoText, progressDialogueTitle,
            SizeZwin.value, parent=self
        )
        self.stack2DsegmWin.show()

    def launchTrackSubCellFeaturesUtil(self):
        self.logger.info(f'Launching utility "{self.sender().text()}"')
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
        self.logger.info(f'Launching utility "{self.sender().text()}"')
        selectedExpPaths = self.getSelectedExpPaths('Compute measurements utility')
        if selectedExpPaths is None:
            return

        self.calcMeasWin = utilsCompute.computeMeasurmentsUtilWin(
            selectedExpPaths, self.app, parent=self
        )
        self.calcMeasWin.show()
    
    def launchToSymDicUtil(self):
        self.logger.info(f'Launching utility "{self.sender().text()}"')
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
            geometry = self.renameWin.saveGeometry()
            self.renameWin.setWindowState(Qt.WindowActive)
            self.renameWin.restoreGeometry(geometry)

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
            geometry = self.convertWin.saveGeometry()
            self.convertWin.setWindowState(Qt.WindowActive)
            self.convertWin.restoreGeometry(geometry)
    
    def launchImageBatchConverter(self):
        self.batchConverterWin = utilsConvert.ImagesToPositions(parent=self)
        self.batchConverterWin.show()
    
    def launchRepeatDataPrep(self):
        self.batchConverterWin = utilsRepeat.repeatDataPrepWindow(parent=self)
        self.batchConverterWin.show()

    def launchDataStruct(self, checked=False):
        self.dataStructButton.setPalette(self.moduleLaunchedPalette)
        self.dataStructButton.setText(
            '0. Creating data structure running...'
        )

        QTimer.singleShot(100, self._showDataStructWin)

    def _showDataStructWin(self):
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        bioformats_url = 'https://www.openmicroscopy.org/bio-formats/'
        bioformats_href = html_utils.href_tag('<b>Bio-Formats</b>', bioformats_url)
        aicsimageio_url = 'https://allencellmodeling.github.io/aicsimageio/#'
        aicsimageio_href = html_utils.href_tag('<b>AICSImageIO</b>', aicsimageio_url)
        issues_href = f'<a href="{issues_url}">GitHub page</a>'
        txt = html_utils.paragraph(f"""
            To process microsocpy files, Cell-ACDC uses the {bioformats_href}.<br><br>
            <b>Bio-Formats requires Java</b> and a python package called <code>javabridge</code>,<br>
            that will be automatically installed if missing.<br><br>
            We recommend using Bio-Formats, since it can read the metadata of the file,<br> 
            such as pixel size, numerical aperture etc.<br><br>
            Alternatively, if you <b>already pre-processed your microsocpy files into .tif 
            files</b>,<br>
            you can choose to simply re-structure them into the Cell-ACDC compatible 
            format.<br><br>
            If nothing works, open an issue on our {issues_href} and we 
            will be happy to help you out.<br><br>
            How do you want to proceed?          
        """)
        # useAICSImageIO = QPushButton(
        #     QIcon(':AICS_logo.svg'), ' Use AICSImageIO ', msg
        # )
        useBioFormatsButton = QPushButton(
            QIcon(':ome.svg'), ' Use Bio-Formats ', msg
        )
        restructButton = QPushButton(
            QIcon(':folders.svg'), ' Re-structure image files ', msg
        )
        msg.question(
            self, 'How to structure files', txt, 
            buttonsTexts=(
                'Cancel', 
                useBioFormatsButton, 
                # useAICSImageIO, 
                restructButton
            )
        )
        if msg.cancel:
            self.logger.info('Creating data structure process aborted by the user.')
            self.restoreDefaultButtons()
            return
        
        useBioFormats = msg.clickedButton == useBioFormatsButton
        if self.dataStructButton.isEnabled() and useBioFormats:
            if is_win:
                self.dataStructButton.setPalette(self.defaultButtonPalette)
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
            elif is_mac:
                self.dataStructWin = (
                    dataStruct.InitFijiMacro(self)
                )
                self.dataStructWin.run()
                self.restoreDefaultButtons()
        elif msg.clickedButton == restructButton:
            self.progressWin = apps.QDialogWorkerProgress(
                title='Re-structure image files log', parent=self,
                pbarDesc='Re-structuring image files running...'
            )
            self.progressWin.sigClosed.connect(self.progressWinClosed)
            self.progressWin.show(self.app)
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
        self.dataStructButton.setText(
            '0. Create data structure from microscopy/image file(s)...'
        )
        self.dataStructButton.setPalette(self.defaultButtonPalette)

    def launchDataPrep(self, checked=False):
        dataPrepWin = dataPrep.dataPrepWin(
            mainWin=self, version=self._version
        )
        dataPrepWin.sigClose.connect(self.dataPrepClosed)
        dataPrepWin.show()
        self.dataPrepWins.append(dataPrepWin)
    
    def dataPrepClosed(self, dataPrepWin):
        try:
            self.dataPrepWins.remove(dataPrepWin)
        except ValueError:
            pass

    def launchSegm(self, checked=False):
        c = self.segmButton.palette().button().color().name()
        launchedColor = self.moduleLaunchedColor
        defaultColor = self.defaultPushButtonColor
        defaultText = self.defaultTextSegmButton
        if c != self.moduleLaunchedColor:
            self.segmButton.setPalette(self.moduleLaunchedPalette)
            self.segmButton.setText('Segmentation is running. '
                                    'Check progress in the terminal/console')
            self.segmWin = segm.segmWin(
                buttonToRestore=(self.segmButton, defaultColor, defaultText),
                mainWin=self, version=self._version
            )
            self.segmWin.sigClosed.connect(self.segmWinClosed) 
            self.segmWin.show()
            self.segmWin.main()
        else:
            geometry = self.segmWin.saveGeometry()
            self.segmWin.setWindowState(Qt.WindowActive)
            self.segmWin.restoreGeometry(geometry)

    def segmWinClosed(self):
        self.segmButton.setPalette(self.defaultButtonPalette)

    def launchGui(self, checked=False):
        self.logger.info('Opening GUI...')
        guiWin = gui.guiWin(
            self.app, mainWin=self, version=self._version, 
            launcherSlot=self.launchGui
        )
        self.guiWins.append(guiWin)
        guiWin.sigClosed.connect(self.guiClosed)
        guiWin.run()
    
    def launchSpotmaxGui(self, checked=False):
        from spotmax import icon_path, logo_path
        # logoDialog = apps.LogoDialog(logo_path, icon_path, parent=self)
        
        splashScreen = QSplashScreen()
        splashScreen.setPixmap(QPixmap(logo_path))
        splashScreen.show()

        QTimer.singleShot(300, partial(self._launchSpotMaxGui, splashScreen))
    
    def _launchSpotMaxGui(self, splashScreen):
        self.logger.info('Launching spotMAX...')
        spotmaxWin = spotmaxRun.run_gui(
            app=self.app, mainWin=self, launcherSlot=self.launchSpotmaxGui, 
            
        )
        spotmaxWin.sigClosed.connect(self.spotmaxGuiClosed)
        self.spotmaxWins.append(spotmaxWin)
        splashScreen.close()
    
    def spotmaxGuiClosed(self, spotmaxWin):
        self.spotmaxWins.remove(spotmaxWin)
        
    def guiClosed(self, guiWin):
        try:
            self.guiWins.remove(guiWin)
        except ValueError:
            pass

    def launchAlignUtil(self, checked=False):
        self.logger.info(f'Launching utility "{self.sender().text()}"')
        selectedExpPaths = self.getSelectedExpPaths(
            'Align frames in X and Y with phase cross-correlation'
        )
        if selectedExpPaths is None:
            return
        
        title = 'Align frames'
        infoText = 'Aligning frames in X and Y with phase cross-correlation...'
        progressDialogueTitle = 'Align frames'
        self.alignWindow = utilsAlign.alignWin(
            selectedExpPaths, self.app, title, infoText, progressDialogueTitle,
            parent=self
        )
        self.alignWindow.show()

    def launchConcatUtil(self, checked=False, exp_folderpath=None):
        self.logger.info(
            f'Launching utility "Concatenate tables from multipe positions"'
        )
        selectedExpPaths = self.getSelectedExpPaths(
            'Concatenate acdc_output files', exp_folderpath=exp_folderpath
        )
        if selectedExpPaths is None:
            return
        
        title = 'Concatenate acdc_output files'
        infoText = 'Launching concatenate acdc_output files process...'
        progressDialogueTitle = 'Concatenate acdc_output files'
        self.concatWindow = utilsConcat.ConcatWin(
            selectedExpPaths, self.app, title, infoText, progressDialogueTitle,
            parent=self
        )
        self.concatWindow.show()
    
    def launchConcatSpotmaxUtil(self, checked=False, exp_folderpath=None):
        self.logger.info(
            f'Launching utility "Concatenate tables from multipe positions"'
        )
        selectedExpPaths = self.getSelectedExpPaths(
            'Concatenate spotMAX output files', exp_folderpath=exp_folderpath
        )
        if selectedExpPaths is None:
            return
        
        title = 'Concatenate spotMAX output files'
        infoText = 'Launching concatenate spotMAX output files process...'
        progressDialogueTitle = 'Concatenate spotMAX output files'
        self.concatWindow = utilsConcat.ConcatWin(
            selectedExpPaths, self.app, title, infoText, progressDialogueTitle,
            parent=self
        )
        self.concatWindow.show()
    
    def showEvent(self, event):
        self.showAllWindows()
        # self.setFocus()
        self.activateWindow()
        if not self.checkUserDataFolderPath:
            return
        self.checkMigrateUserDataFolderPath()
    
    def checkMigrateUserDataFolderPath(self):
        from . import user_home_path
        user_home_acdc_folders = load.get_all_acdc_folders(user_home_path)
        if not user_home_acdc_folders:
            self.checkUserDataFolderPath = False
            return

        if 'doNotAskMigrate' in self.df_settings.index:
            if str(self.df_settings.at['doNotAskMigrate', 'value']) == 'Yes':
                return
        
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'Starting from version 1.4.0, Cell-ACDC default <b>user profile path</b> '
            f'has been <b>changed</b> to <code>{user_profile_path}</code><br><br>'
            'Since you have some profile data saved in the old path, Cell-ACDC '
            'can now migrate everything to the new folder.<br><br>'
            'Do you want to <b>migrate now?</b><br>'
        )
        acdc_folders_format = [
            f'&nbsp;&nbsp;&nbsp;<code>{os.path.join(user_home_path, folder)}</code>' 
            for folder in user_home_acdc_folders
        ]
        acdc_folders_format = '<br>'.join(acdc_folders_format)
        detailsText = (
            f'Folders found in the previous location:<br><br>{acdc_folders_format}'
        )
        doNotAskAgainCheckbox = QCheckBox('Do not ask again')
        msg.warning(
            self, 'Migrate old user profile', txt, 
            buttonsTexts=('Cancel', 'Yes'),
            detailsText=detailsText,
            widgets=doNotAskAgainCheckbox
        )
        if doNotAskAgainCheckbox.isChecked():
            self.df_settings.at['doNotAskMigrate', 'value'] = 'Yes'
            self.df_settings.to_csv(settings_csv_path)
        if msg.cancel:
            self.logger.info(
                'Migrating old user profile cancelled.'
            )
            self.checkUserDataFolderPath = False
            return
        self.startMigrateUserProfileWorker(
            user_home_path, user_profile_path, user_home_acdc_folders
        )
        self.checkUserDataFolderPath = False
        
    def showAllWindows(self):
        openModules = self.getOpenModules()
        for win in openModules:
            if not win.isMinimized():
                continue
            geometry = win.saveGeometry()
            win.setWindowState(Qt.WindowNoState)
            win.restoreGeometry(geometry)
        self.raise_()
        # self.setFocus()
        self.activateWindow()

    def show(self):
        self.setColorsAndText()
        super().show()
        h = self.dataPrepButton.geometry().height()
        f = 1.5
        self.dataStructButton.setMinimumHeight(int(h*f))
        self.dataPrepButton.setMinimumHeight(int(h*f))
        self.segmButton.setMinimumHeight(int(h*f))
        self.guiButton.setMinimumHeight(int(h*f))
        if hasattr(self, 'spotmaxButton'):
            self.spotmaxButton.setMinimumHeight(int(h*f))
        self.showAllWindowsButton.setMinimumHeight(int(h*f))
        self.restartButton.setMinimumHeight(int(int(h*f)))
        self.closeButton.setMinimumHeight(int(int(h*f)))
        # iconWidth = int(self.closeButton.iconSize().width()*1.3)
        # self.closeButton.setIconSize(QSize(iconWidth, iconWidth))
        self.setColorsAndText()
        self.readSettings()
        if self.app.toggle_dark_mode:
            self.darkModeToggle.warnMessageBox  = False
            self.darkModeToggle.setChecked(True)

    def saveWindowGeometry(self):
        settings = QSettings('schmollerlab', 'acdc_main')
        settings.setValue("geometry", self.saveGeometry())

    def readSettings(self):
        settings = QSettings('schmollerlab', 'acdc_main')
        if settings.value('geometry') is not None:
            self.restoreGeometry(settings.value("geometry"))
    
    def getOpenModules(self):
        c2 = self.segmButton.palette().button().color().name()
        launchedColor = self.moduleLaunchedColor

        openModules = []
        if self.dataPrepWins:
            openModules.extend(self.dataPrepWins)
        if c2 == launchedColor:
            openModules.append(self.segmWin)
        if self.guiWins:
            openModules.extend(self.guiWins)
        if self.spotmaxWins:
            openModules.extend(self.spotmaxWins)
        return openModules


    def checkOpenModules(self):
        openModules = self.getOpenModules()

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

        if not self.forceClose:
            acceptClose, openModules = self.checkOpenModules()
            if acceptClose:
                for openModule in openModules:
                    geometry = openModule.saveGeometry()
                    openModule.setWindowState(Qt.WindowActive)
                    openModule.restoreGeometry(geometry)
                    openModule.close()
                    if openModule.isVisible():
                        event.ignore()
                        return
            else:
                event.ignore()
                return

        if self._do_restart:
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
