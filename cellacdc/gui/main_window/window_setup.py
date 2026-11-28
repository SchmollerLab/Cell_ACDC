"""Start-up and generic services of the main GUI window: run(),
initGlobalAttr, session settings, help, about and log menus, worker
progress callbacks, view range helpers and close prompts. Mixins of
guiWin (gui/main_window/window.py) only: the measurements viewer never
composes them.
"""
import sys
import os
from datetime import timedelta
from collections import defaultdict
import pandas as pd
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QButtonGroup
from cellacdc import base_cca_dict
from cellacdc import cca_df_colnames
from cellacdc import html_utils
from cellacdc.myutils import filesystem
from cellacdc.shared.widgets import message_box as widgets_message_box
from ..widgets import (
    edit_toolbars as gui_widgets_edit_toolbars,
    main_window_items as gui_widgets_main_window_items,
)
from cellacdc import settings_csv_path
from cellacdc import autopilot
from cellacdc.myutils.app_info import setupLogger
from cellacdc.help import welcome, about
from cellacdc.plot import imshow


class WindowGuiElements:
    """Start-up and generic services of the main GUI window: run() and
    initGlobalAttr, last session settings, help, about and log file
    menus, progress, log and debug callbacks of workers, view range
    store/reset and the close-all-windows prompt.
    """

    def run(self, module='acdc_gui', logs_path=None):                
        self.setWindowIcon()
        self.setWindowTitle()
        
        self.is_win = sys.platform.startswith("win")
        if self.is_win:
            self.openFolderText = 'Show in Explorer...'
        else:
            self.openFolderText = 'Reveal in Finder...'

        self.is_error_state = False
        logger, logs_path, log_path, log_filename = setupLogger(
            module=module, logs_path=logs_path, caller=self._appName
        )
        if self._version is not None:
            logger.info(f'Initializing GUI v{self._version}')
        else:
            logger.info(f'Initializing GUI...')
        
        self.module = module
        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path

        from cellacdc import regionprops as acdc_regionprops
        
        self._acdcRegionProps = acdc_regionprops.acdcRegionprops
        self._acdc_regionprops = acdc_regionprops

        self.initProfileModels()
        self.loadLastSessionSettings()

        self.newWindows = []
        self.progressWin = None
        self.slideshowWin = None
        self.ccaTableWin = None
        self.exportToImageWindow = None
        self.customAnnotButton = None
        self.ccaCheckerRunning = False
        self.isCancellingModeInit = False
        self.isDataLoaded = False
        self.highlightedID = 0
        self.hoverLabelID = 0
        self.expandingID = -1
        self.count = 0
        self.isDilation = True
        self.flag = True
        self.currentPropsID = 0
        self.isSegm3D = False
        self.newSegmEndName = ''
        self.closeGUI = False
        self.warnKeyPressedMsg = None
        self.AutoPilotProfile = autopilot.AutoPilotProfile()
        self.storeStateWorker = None
        self.AutoPilot = None
        self.widgetsWithShortcut = {}
        self.widgetsForActions = {}
        self.customRightClickItems = dict()
        self.invertBwAlreadyCalledOnce = False
        self.zoomOutKeyValue = Qt.Key_H
        self.preprocWorker = None
        self.preprocessDialog = None
        self.viewOriginalLabels = True
        self.keepDisabled = False
        self.whitelistAddNewIDsFrame = None
        self.whitelistOriginalIDs = None
        self.whyNavigateDisabled = set()
        self.autoSaveTimer = QTimer()
        self.dirtyPointsLayerTableEndNames = set()
        self.mouseBindings = dict()
        self.defaultMouseShortcuts = dict()
        self.widgetsPersistentShortcut = dict()
        self.doubleSpaceBarState = False
        self.protected_new_IDs = dict()
        
        self._setup_vars_combine()
        if 'autoSaveIntevalValue' not in self.df_settings.index:
            autoSaveIntevalValue = 2
            autoSaveIntervalUnit = 'minutes'
        else:
            autoSaveIntevalValue = float(
                self.df_settings.at['autoSaveIntevalValue', 'value']
            )
            autoSaveIntervalUnit = str(
                self.df_settings.at['autoSaveIntervalUnit', 'value']
            )
        
        self.autoSaveIntevalValueUnit = (
            autoSaveIntevalValue, autoSaveIntervalUnit
        )
        self.logger.info(
            'Autosave interval set to: '
            f'{autoSaveIntevalValue} {autoSaveIntervalUnit}'
        )

        self.checkableButtons = []
        self.leftClickButtons = []
        self.toolsActiveInProj3Dsegm = set()
        self.customAnnotDict = {}

        # Keep a list of functions that are not functional in 3D, yet
        self.functionsNotTested3D = []

        self.isSnapshot = False
        self.debugFlag = False
        self.pos_i = 0
        self.save_until_frame_i = 0
        self.countKeyPress = 0
        self.countRightClicks = 0
        self.xHoverImg, self.yHoverImg = None, None
        
        # Keep track on what frames the on first visit tools already ran
        self.lastFrameRanOnFirstVisitTools = 0
        
        # Buttons added to QButtonGroup will be mutually exclusive
        self.checkableQButtonsGroup = QButtonGroup(self)
        self.checkableQButtonsGroup.setExclusive(False)

        self.lazyLoader = None

        self.gui_createCursors()
        self.gui_createActions()
        self.gui_createMenuBar()

        self.gui_createToolBars()
        self.gui_createControlsToolbar()
        self.gui_createShowPropsButton()
        self.gui_createRegionPropsDockWidget()
        self.gui_createQuickSettingsWidgets()
        self.setTooltips()
        self.gui_populateToolSettingsMenu()
        self.gui_editRightClickMenuButtons()

        self.autoSaveGarbageWorkers = []
        self.autoSaveActiveWorkers = []

        self.gui_connectActions()
        self.gui_createStatusBar()
        # self.gui_createTerminalWidget()

        self.gui_createGraphicsPlots()
        self.gui_addGraphicsItems()

        self.gui_createBottomWidgets()
        self.gui_createImg2Widgets()
        self.gui_createBottomWidgetsToBottomLayout()
        self.searchWidget.registerSearchControls()

        mainContainer = gui_widgets_main_window_items.GuiCentralWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = self.gui_createMainLayout()
        self.mainLayout = mainLayout

        mainContainer.setLayout(mainLayout)

        self.isEditActionsConnected = False

        self.readRecentPaths()
        self.initShortcuts()
        self.show()
        QTimer.singleShot(100, self.resizeRangeWelcomeText)
        
        self.app.installEventFilter(self)
        
        self.logger.info('GUI ready.')

    def initGlobalAttr(self):
        self.setOverlayColors()

        self.initImgCmap()

        # Colormap
        self.setLut()

        self.fluoDataChNameActions = []

        self.splineHoverON = False
        self.tempSegmentON = False
        self.draggingDelROI = False
        self.xyOnCtrlPressedFirstTime = None
        self.typingEditID = False
        self.prevAnnotOptions = None
        self.ghostObject = None
        self.autoContourHoverON = False
        self.navigateScrollBarStartedMoving = True
        self.zSliceScrollBarStartedMoving = True
        self.labelRoiRunning = False
        self.isRangeReset = True
        self.editIDmergeIDs = True
        self.doNotAskAgainExistingID = False
        self.doubleRightClickTimeElapsed = False
        self.isRealTimeTrackerInitialized = False
        self.isWarningCcaIntegrity = False
        self.isDoubleRightClick = False
        self.isExportingVideo = False
        self.pointsLayersNeverToggled = True
        self.highlightedIDopts = None
        self.timestampStartTimedelta = timedelta(seconds=0)
        self.keptObjectsIDs = gui_widgets_edit_toolbars.KeptObjectIDsList(
            self.keptIDsLineEdit, self.keepIDsConfirmAction
        )
        self._ZprojWidgersEnabledState = None
        self.imgValueFormatter = 'd'
        self.rawValueFormatter = 'd'
        self.lastHoverID = -1
        self.annotOptionsToRestore = defaultdict(dict)
        self.rescaleIntensChannelHowMapper = {
            self.user_ch_name: 'Rescale each 2D image'
        }
        self.timestampDialog = None
        self.scaleBarDialog = None
        self.countObjsWindow = None
        self.editCcaWidget = None
        
        self.annotateSingleMothBudPairState = {}
        self.initLabelRoiModelDialog = None

        # Second channel used by cellpose
        self.secondChannelName = None

        self.ax1_viewRange = None
        self.measurementsWin = None

        self.model_kwargs = None
        self.segmModelName = None
        self.labelRoiModel = None
        # model_name -> ModelExecution (environment chosen once per session)
        self.modelExecutions = {}
        # Kept when loading new data: the magic prompts toolbar keeps its
        # initialized model (and execution) across data loads
        self.promptModelExecutions = getattr(
            self, 'promptModelExecutions', {}
        )
        # model_idx -> (settings key, model_name, model) for lost IDs
        self.segForLostIDsModels = {}
        self.autoSegmDoNotAskAgain = False
        self.labelRoiGarbageWorkers = []
        self.labelRoiActiveWorkers = []

        self.clickedOnBud = False
        self.postProcessSegmWin = None

        self.UserEnforced_DisabledTracking = False
        self.UserEnforced_Tracking = False

        self.ax1BrushHoverID = 0
        
        self.disabled_cca_warnings = set()

        self.last_pos_i = -1
        self.last_frame_i = -1

        # Plots items
        self.isMouseDragImg2 = False
        self.isMouseDragImg1 = False
        self.isMovingLabel = False
        self.isRightClickDragImg1 = False
        self.clickObjYc, self.clickObjXc = None, None

        # Brush strokes prompts (e.g., ConvPaint)
        # {pos_i: {frame_i: uint8 array with shape segm_data.shape[1:]}}
        self.brushStrokesAnnotData = {}
        # {(pos_i, frame_i): deque of (z or None, array copy, image source)}
        self.brushStrokesUndo = {}
        # {(pos_i, frame_i): z-slice index or z-projection name} displayed
        # when the 2D brush strokes of z-stack data were painted, i.e., the
        # image they are paired with for training (see
        # `prompt_utils.get_brush_strokes_image_source`)
        self.brushStrokesImageSources = {}
        self.isBrushStrokeDragImg1 = False
        self.brushStrokePaintValue = 0
        self.brushStrokeLastYX = None
        # (pos_i, frame_i, z, image source) being painted by the current
        # stroke
        self.brushStrokeDragTarget = None
        self.brushStrokesChangedSinceTraining = True
        self.brushStrokesDiskMasks = {}
        self.brushStrokesWarnedNotPaintable = False
        # Cells selected to be added to the brush strokes
        self.brushStrokesSelectCellsToolbar.resetSelection()

        # Box prompts (e.g., micro-sam)
        # {pos_i: {frame_i: [{'id', 'z', 'ymin', 'xmin', 'ymax', 'xmax'}]}}
        # where 'z' is None for 2D data and max coordinates are excluded
        self.magicPromptsBoxesData = {}
        # {(pos_i, frame_i): deque of copies of the frame boxes}
        self.magicPromptsBoxesUndo = {}
        # `widgets.PromptBoxROI` items of the displayed boxes
        self.removeMagicPromptsBoxItems()
        # ROI and start point of the box being drawn
        self.magicPromptsDrawingBox = None
        self.isBoxPromptDragImg1 = False

        self.cca_df_colnames = cca_df_colnames
        self.cca_df_dtypes = [
            str, int, int, str, int, int, bool, bool, int
        ]
        self.cca_df_default_values = list(base_cca_dict.values())
        self.cca_df_int_cols = [
            col for col in cca_df_colnames if type(base_cca_dict[col]) == int
        ]
        self.lin_tree_df_bool_col = [
            col for col in cca_df_colnames 
            if isinstance(base_cca_dict[col], bool)
        ]

        self.lin_tree_col_checks = [
            'generation_num',
        ]

        # self.lin_tree_df_colnames = set(base_cca_df.keys()) | set(lineage_tree_cols)
        # # self.lin_tree_df_dtypes = [ #dk if i need this, for now ignored
        # #     str, int, int, str, int, int, bool, bool, int
        # # ]
        # self.lin_tree_df_default_values = list(base_cca_df.values()) + lineage_tree_cols_std_val
        self.lin_tree_df_int_cols = [
            'generation_num',
            'relative_ID',
            'emerg_frame_i',
            'division_frame_i',
            'corrected_on_frame_i'
        ]
        self.lin_tree_df_bool_col = [
            'is_history_known',
        ]

        self.lin_tree_col_checks = [
            'generation_num',
        ]

        self.lin_tree_df_colnames = self.lin_tree_df_int_cols + self.lin_tree_df_bool_col + self.lin_tree_col_checks
        self.SegForLostIDsSettings = {}

    def initProfileModels(self):
        self.logger.info('Initiliazing profilers...')
        
        from cellacdc._profile.spline_to_obj import model
        
        self.splineToObjModel = model.Model()

        self.splineToObjModel.fit()

    def loadLastSessionSettings(self):
        self.settings_csv_path = settings_csv_path
        if os.path.exists(settings_csv_path):
            self.df_settings = pd.read_csv(
                settings_csv_path, index_col='setting'
            )
            if 'is_bw_inverted' not in self.df_settings.index:
                self.df_settings.at['is_bw_inverted', 'value'] = 'No'
            else:
                self.df_settings.loc['is_bw_inverted'] = (
                    self.df_settings.loc['is_bw_inverted'].astype(str)
                )
            if 'fontSize' not in self.df_settings.index:
                self.df_settings.at['fontSize', 'value'] = 24
            if 'overlayColor' not in self.df_settings.index:
                self.df_settings.at['overlayColor', 'value'] = '255-255-0'
            if 'how_normIntensities' not in self.df_settings.index:
                raw = 'Do not normalize. Display raw image'
                self.df_settings.at['how_normIntensities', 'value'] = raw
        else:
            idx = ['is_bw_inverted', 'fontSize', 'overlayColor', 'how_normIntensities']
            values = ['No', 12, '255-255-0', 'raw']
            self.df_settings = pd.DataFrame({
                'setting': idx,'value': values}
            ).set_index('setting')
        
        if 'isLabelsVisible' not in self.df_settings.index:
            self.df_settings.at['isLabelsVisible', 'value'] = 'No'
        
        if 'isNextFrameVisible' not in self.df_settings.index:
            self.df_settings.at['isNextFrameVisible', 'value'] = 'No'
        
        if 'isRightImageVisible' not in self.df_settings.index:
            self.df_settings.at['isRightImageVisible', 'value'] = 'Yes'
        
        if 'colorScheme' in self.df_settings.index:
            col = 'colorScheme'
            self._colorScheme = self.df_settings.at[col, 'value']
        else:
            self._colorScheme = 'light'
        
        self.doNotShowAgainMissingCca = False
        if 'doNotShowAgainMissingCca' not in self.df_settings.index:
            self.df_settings.at['doNotShowAgainMissingCca', 'value'] = 'No'
        else:
            val = self.df_settings.at['doNotShowAgainMissingCca', 'value']
            self.doNotShowAgainMissingCca = val=='Yes'

    def showImageDebug(self, display_info):
        title = ''
        img_titles = None
        imgs = []
        labs = None
        if isinstance(display_info, dict):
            title = display_info.get('title', '')
            imgs = list(display_info.get('images', []))
            img_titles = display_info.get('img_titles', None)

            if img_titles is not None:
                img_titles = list(img_titles)
                if len(img_titles) < len(imgs):
                    img_titles.extend([''] * (len(imgs) - len(img_titles)))

            if 'img_seg_pairs' in display_info:
                new_imgs, labs, new_titles = [], [], []
                pairs = display_info['img_seg_pairs']
                i_to_skip = set()
                for i, img in enumerate(imgs):
                    if i in i_to_skip:
                        continue

                    title_i = img_titles[i] if img_titles is not None else f'image_{i}'
                    if i in pairs:
                        lab_i = pairs[i]
                        if 0 <= lab_i < len(imgs):
                            i_to_skip.add(lab_i)
                            lab = imgs[lab_i]
                            lab_title = (
                                img_titles[lab_i]
                                if img_titles is not None else f'image_{lab_i}'
                            )
                            img_title = f"{title_i} ({lab_title}) "
                        else:
                            lab = None
                            img_title = title_i
                    else:
                        lab = None
                        img_title = title_i

                    new_imgs.append(img)
                    labs.append(lab)
                    new_titles.append(img_title)

                imgs = new_imgs
                img_titles = new_titles

        if not imgs:
            self.logger.warning(
                'showImageDebug called without images. Ignoring debug display.'
            )
            return

        win = imshow(
            *imgs,
            window_title=str(title),
            figure_title=str(title),
            axis_titles=img_titles,
            labels_overlays=labs,
            parent=self,
            block=True,
            show_contours=True,
            show_IDs=True,
            win_stay_on_top=False,
        )
        if not hasattr(self, '_debug_imshow_windows'):
            self._debug_imshow_windows = []
        self._debug_imshow_windows = [
            w for w in self._debug_imshow_windows
            if w is not None
        ]
        self._debug_imshow_windows.append(win)
        
        try:
            self._ackSegForLostIDsWorker('image_debug')
        except:
            pass

    def workerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        self.logger.info('Worker process ended.')
        self.updateAllImages()
        self.titleLabel.setText('Done', color='w')

    def workerInitInnerPbar(self, totalIter):
        self.progressWin.innerPbar.setValue(0)
        if totalIter == 1:
            totalIter = 0
        self.progressWin.innerPbar.setMaximum(totalIter)

    def workerUpdateInnerPbar(self, step):
        self.progressWin.innerPbar.update(step)

    def workerDebug(self, item):
        tracked_video, worker = item
        from cellacdc.plot import imshow
        imshow(tracked_video)
        worker.waitCond.wakeAll()

    def storeViewRange(self):
        if not hasattr(self, 'isRangeReset'):
            return
        
        if not self.isRangeReset:
            return
        self.ax1_viewRange = self.ax1.viewRange()
        self.isRangeReset = False

    def workerLog(self, text):
        self.logger.info(text)

    def debugShowImg(self, img):
        imshow(img)

    def removeAxLimits(self):
        self.ax1.vb.state['limits']['xLimits'] = [-1E307, +1E307]
        self.ax1.vb.state['limits']['yLimits'] = [-1E307, +1E307]

    def autoRange(self):
        if self.labelsGrad.showLabelsImgAction.isChecked():
            self.ax2.autoRange()
        self.ax1.autoRange()

    def resetRange(self):
        if self.ax1_viewRange is None:
            return
        xRange, yRange = self.ax1_viewRange
        if self.labelsGrad.showLabelsImgAction.isChecked():
            self.ax2.vb.setRange(xRange=xRange, yRange=yRange)
        self.ax1.vb.setRange(xRange=xRange, yRange=yRange)
        self.ax1_viewRange = None
        self.isRangeReset = True

    def resizeRangeWelcomeText(self):
        xRange, yRange = self.ax1.viewRange()
        deltaX = xRange[1] - xRange[0]
        deltaY = yRange[1] - yRange[0]
        self.ax1.setXRange(0, deltaX)
        self.ax1.setYRange(0, deltaY)
        self.ax1.setLimits(
            xMin=0, xMax=deltaX, yMin=0, yMax=deltaY
        )
        # self.ax1.setXRange(0, 0)
        # self.ax1.setYRange(0, 0)

    def showInExplorer_cb(self):
        posData = self.data[self.pos_i]
        path = posData.images_path
        filesystem.showInExplorer(path)

    def showAbout(self):
        self.aboutWin = about.QDialogAbout(parent=self)
        self.aboutWin.show()

    def openLogFile(self):
        self.logger.info(f'Opening log file "{self.log_path}"...')
        filesystem.showInExplorer(self.log_path)

    def showLogFiles(self):
        log_files_path = os.path.dirname(self.log_path)
        self.logger.info(f'Opening log files folder "{log_files_path}"...')
        filesystem.showInExplorer(log_files_path)

    def showTipsAndTricks(self):
        self.welcomeWin = welcome.welcomeWin()
        self.welcomeWin.showAndSetSize()
        self.welcomeWin.showPage(self.welcomeWin.quickStartItem)

    def about(self):
        pass

    def clearMemory(self):
        if not hasattr(self, 'data'):
            return
        self.logger.info('Clearing memory...')
        for posData in self.data:
            try:
                del posData.img_data
            except Exception as e:
                pass
            try:
                del posData.segm_data
            except Exception as e:
                pass
            try:
                del posData.ol_data_dict
            except Exception as e:
                pass
            try:
                del posData.fluo_data_dict
            except Exception as e:
                pass
            try:
                del posData.ol_data
            except Exception as e:
                pass
        del self.data

    def askCloseAllWindows(self):
        txt = html_utils.paragraph("""
            There are other open windows that were created from this window.
            <br><br>
            If you proceed, the <b>other windows will be closed too.<br>
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Open windows', txt, 
            buttonsTexts=('Cancel', 'Ok, close now')
        )
        return msg.cancel
