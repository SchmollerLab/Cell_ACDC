"""Loading data into the main GUI: open file or folder, load worker and
lazy loader, initialization after loading, recovery of unsaved data.
Mixins of guiWin (gui/main_window/window.py) only: the measurements
viewer never composes them.
"""
import os
import shutil
from datetime import datetime
import uuid
import psutil
import zipfile
from functools import partial
from tqdm import tqdm
from natsort import natsorted
import numpy as np
import pandas as pd
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.draw
import skimage.exposure
import skimage.transform
import skimage.segmentation
from qtpy.QtCore import Qt, QEventLoop, QTimer, QThread, QMutex, QWaitCondition
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QPushButton, QFileDialog
from cellacdc import _window_stacking
from cellacdc import exception_handler
from cellacdc import cca_df_colnames
from cellacdc import load, prompts, apps, workers, html_utils
from cellacdc import myutils
from cellacdc.data_prep import main_window as dataPrep
from cellacdc import widgets
from cellacdc import colors
from cellacdc import user_manual_url
from cellacdc import autopilot
from cellacdc import urls
from cellacdc import path as acdc_path


class LoadDataGuiElements:
    """Load data worker and lazy loader, initialization of the GUI after
    loading (loadingDataCompleted, initPosAttr, reInitGui), recovery of
    unsaved data and memory checks.
    """

    def gui_createLazyLoader(self):
        if not self.lazyLoader is None:
            return

        self.lazyLoaderThread = QThread()
        self.lazyLoaderMutex = QMutex()
        self.lazyLoaderWaitCond = QWaitCondition()
        self.waitReadH5cond = QWaitCondition()
        self.readH5mutex = QMutex()
        self.lazyLoader = workers.LazyLoader(
            self.lazyLoaderMutex, self.lazyLoaderWaitCond, 
            self.waitReadH5cond, self.readH5mutex
        )
        self.lazyLoader.moveToThread(self.lazyLoaderThread)
        self.lazyLoader.wait = True

        self.lazyLoader.signals.finished.connect(self.lazyLoaderThread.quit)
        self.lazyLoader.signals.finished.connect(self.lazyLoader.deleteLater)
        self.lazyLoaderThread.finished.connect(self.lazyLoaderThread.deleteLater)

        self.lazyLoader.signals.progress.connect(self.workerProgress)
        self.lazyLoader.signals.sigLoadingNewChunk.connect(self.loadingNewChunk)
        self.lazyLoader.sigLoadingFinished.connect(self.lazyLoaderFinished)
        self.lazyLoader.signals.critical.connect(self.lazyLoaderCritical)
        self.lazyLoader.signals.finished.connect(self.lazyLoaderWorkerClosed)

        self.lazyLoaderThread.started.connect(self.lazyLoader.run)
        self.lazyLoaderThread.start()

    def loadingNewChunk(self, chunk_range):
        coord0_chunk, coord1_chunk = chunk_range
        desc = (
            f'Loading new window, range = ({coord0_chunk}, {coord1_chunk})...'
        )
        self.progressWin = apps.QDialogWorkerProgress(
            title='Loading data...', parent=self, pbarDesc=desc
        )
        self.progressWin.mainPbar.setMaximum(0)
        self.progressWin.show(self.app)

    def lazyLoaderFinished(self):
        self.logger.info('Load chunk data worker done.')
        if self.lazyLoader.updateImgOnFinished:
            self.updateAllImages()

        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None

    @exception_handler
    def lazyLoaderCritical(self, error):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
            self.lazyLoader.pause()
        raise error

    def lazyLoaderWorkerClosed(self):
        if self.lazyLoader.salute:
            self.logger.info('Cell-ACDC GUI closed.')     
            self.sigClosed.emit(self)
        
        self.lazyLoader = None

    def loadSelectedData(self, user_ch_file_paths, user_ch_name):
        data = []
        numPos = len(user_ch_file_paths)
        self.user_ch_file_paths = user_ch_file_paths
        
        self.logger.info(f'Reading {user_ch_name} channel metadata...')
        # Get information from first loaded position
        posData = load.loadData(user_ch_file_paths[0], user_ch_name, log_func=self.logger.info)
        posData.getBasenameAndChNames(qparent=self)
        posData.buildPaths()

        if posData.ext != '.h5':
            self.lazyLoader.salute = False
            self.lazyLoader.exit = True
            self.lazyLoaderWaitCond.wakeAll()
            self.waitReadH5cond.wakeAll()

        # Get end name of every existing segmentation file
        existingSegmEndNames = set()
        for filePath in user_ch_file_paths:
            _posData = load.loadData(filePath, user_ch_name, log_func=self.logger.info)
            _posData.getBasenameAndChNames(qparent=self)
            segm_files = load.get_segm_files(_posData.images_path)
            _existingEndnames = load.get_endnames(
                _posData.basename, segm_files
            )
            existingSegmEndNames.update(_existingEndnames)

        selectedSegmEndName = ''
        self.newSegmEndName = ''
        if self.isNewFile or not existingSegmEndNames:
            self.isNewFile = True
            # Remove the 'segm_' part to allow filenameDialog to check if
            # a new file is existing (since we only ask for the part after
            # 'segm_')
            existingEndNames = [
                n.replace('segm', '', 1).replace('_', '', 1)
                for n in existingSegmEndNames
            ]
            if posData.basename.endswith('_'):
                basename = f'{posData.basename}segm'
            else:
                basename = f'{posData.basename}_segm'
            win = apps.filenameDialog(
                basename=basename,
                hintText='Insert a <b>filename</b> for the segmentation file:',
                existingNames=existingEndNames
            )
            win.exec_()
            if win.cancel:
                self.loadingDataAborted()
                return
            self.newSegmEndName = win.entryText
        else:
            if len(existingSegmEndNames) > 0:
                win = apps.SelectSegmFileDialog(
                    existingSegmEndNames, self.exp_path, parent=self,
                    addNewFileButton=True, basename=posData.basename
                )
                win.exec_()
                if win.cancel:
                    self.loadingDataAborted()
                    return
                if win.newSegmEndName is None:
                    selectedSegmEndName = win.selectedItemText
                    self.AutoPilotProfile.storeSelectedSegmFile(
                        selectedSegmEndName
                    )
                else:
                    self.newSegmEndName = win.newSegmEndName
                    self.isNewFile = True
            elif len(existingSegmEndNames) == 1:
                selectedSegmEndName = list(existingSegmEndNames)[0]

        posData.loadImgData()
        
        required_ram = posData.getBytesImageData()
        if required_ram >= 5e8:
            # Disable autosave for data > 500MB
            self.autoSaveToggle.setChecked(False)

        proceed = self.checkMemoryRequirements(required_ram)
        if not proceed:
            self.loadingDataAborted()
            return
        
        posData.loadOtherFiles(
            load_segm_data=True,
            load_metadata=True,
            create_new_segm=self.isNewFile,
            new_endname=self.newSegmEndName,
            end_filename_segm=selectedSegmEndName,
            load_segm_info_ini=True,
            loadTrackedLostCentroids=True,
        )
        self.selectedSegmEndName = selectedSegmEndName
        self.labelBoolSegm = posData.labelBoolSegm
        posData.labelSegmData()

        print('')
        self.logger.info(
            f'Segmentation filename: {posData.segm_npz_path}'
        )

        proceed = posData.askInputMetadata(
            self.num_pos,
            ask_SizeT=self.num_pos==1,
            ask_TimeIncrement=True,
            ask_PhysicalSizes=True,
            singlePos=False,
            save=True, 
            warnMultiPos=True
        )
        if not proceed:
            self.loadingDataAborted()
            return
        
        self.AutoPilotProfile.storeOkAskInputMetadata()

        if posData.isSegm3D is None:
            self.isSegm3D = False
        else:
            self.isSegm3D = posData.isSegm3D
        self.SizeT = posData.SizeT
        self.SizeZ = posData.SizeZ
        self.TimeIncrement = posData.TimeIncrement
        self.PhysicalSizeZ = posData.PhysicalSizeZ
        self.PhysicalSizeY = posData.PhysicalSizeY
        self.PhysicalSizeX = posData.PhysicalSizeX
        self.loadSizeS = posData.loadSizeS
        self.loadSizeT = posData.loadSizeT
        self.loadSizeZ = posData.loadSizeZ

        self.overlayLabelsItems = {}
        self.drawModeOverlayLabelsChannels = {}

        self.existingSegmEndNames = existingSegmEndNames
        self.createOverlayLabelsContextMenu(existingSegmEndNames)
        self.overlayLabelsButtonAction.setVisible(True)
        self.createOverlayLabelsItems(existingSegmEndNames)
        self.setNonFunctionalButtonsDisabled()

        self.isH5chunk = (
            posData.ext == '.h5'
            and (self.loadSizeT != self.SizeT
                or self.loadSizeZ != self.SizeZ)
        )

        required_ram = posData.checkH5memoryFootprint()*self.loadSizeS
        if required_ram > 0:
            proceed = self.checkMemoryRequirements(required_ram)
            if not proceed:
                self.loadingDataAborted()
                return

        if posData.SizeT == 1:
            self.isSnapshot = True
        else:
            self.isSnapshot = False

        self.progressWin = apps.QDialogWorkerProgress(
            title='Loading data...', parent=self,
            pbarDesc=f'Loading "{user_ch_file_paths[0]}"...'
        )
        self.progressWin.show(self.app)

        func = partial(
            self.startLoadDataWorker, user_ch_file_paths, user_ch_name,
            posData
        )


        QTimer.singleShot(150, func)

    @exception_handler
    def startLoadDataWorker(
            self, user_ch_file_paths, user_ch_name, firstPosData
        ):
        self.funcDescription = 'loading data'
        
        self.guiTabControl.propsQGBox.idSB.setValue(0)

        self.thread = QThread()
        self.loadDataMutex = QMutex()
        self.loadDataWaitCond = QWaitCondition()

        self.loadDataWorker = workers.loadDataWorker(
            self, user_ch_file_paths, user_ch_name, firstPosData
        )

        self.loadDataWorker.moveToThread(self.thread)
        self.loadDataWorker.signals.finished.connect(self.thread.quit)
        self.loadDataWorker.signals.finished.connect(
            self.loadDataWorker.deleteLater
        )
        self.thread.finished.connect(self.thread.deleteLater)

        self.loadDataWorker.signals.finished.connect(
            self.loadDataWorkerFinished
        )
        self.loadDataWorker.signals.progress.connect(self.workerProgress)
        self.loadDataWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.loadDataWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.loadDataWorker.signals.critical.connect(
            self.workerCritical
        )
        self.loadDataWorker.signals.dataIntegrityCritical.connect(
            self.loadDataWorkerDataIntegrityCritical
        )
        self.loadDataWorker.signals.dataIntegrityWarning.connect(
            self.loadDataWorkerDataIntegrityWarning
        )
        self.loadDataWorker.signals.sigPermissionError.connect(
            self.workerPermissionError
        )
        self.loadDataWorker.signals.sigWarnMismatchSegmDataShape.connect(
            self.askMismatchSegmDataShape
        )
        self.loadDataWorker.signals.sigRecovery.connect(
            self.askRecoverNotSavedData
        )

        self.thread.started.connect(self.loadDataWorker.run)
        self.thread.start()

    def askRecoverNotSavedData(self, posData):
        last_modified_time_unsaved = 'NEVER'
        if os.path.exists(posData.segm_npz_temp_path):
            recovered_file_path = posData.segm_npz_temp_path
            if os.path.exists(posData.segm_npz_path):
                last_modified_time_unsaved = (
                    datetime.fromtimestamp(
                        os.path.getmtime(posData.segm_npz_path)
                    ).strftime("%a %d. %b. %y - %H:%M:%S")
                )
        else:
            posData.setTempPaths()
            if os.path.exists(posData.unsaved_acdc_df_autosave_path):
                zip_path = posData.unsaved_acdc_df_autosave_path
                with zipfile.ZipFile(zip_path, mode='r') as zip:
                    csv_names = natsorted(set(zip.namelist()))
                iso_key = csv_names[-1][:-4]
                most_recent_unsaved_acdc_df_datetime = datetime.strptime(
                    iso_key, load.ISO_TIMESTAMP_FORMAT
                )
                last_modified_time_unsaved = (
                    most_recent_unsaved_acdc_df_datetime
                ).strftime("%a %d. %b. %y - %H:%M:%S")
        
        if os.path.exists(posData.acdc_output_csv_path):
            acdc_df_mtime = os.path.getmtime(posData.acdc_output_csv_path)
            timestamp = datetime.fromtimestamp(acdc_df_mtime)
            last_modified_time_saved = timestamp.strftime(
                "%a %d. %b. %y - %H:%M:%S"
            )
        else:
            last_modified_time_saved = 'Null'
        
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph("""
            Cell-ACDC detected <b>unsaved data</b>.<br><br>
            Do you want to <b>load and recover</b> the unsaved data or 
            load the data that was <b>last saved by the user</b>?
        """)
        details = (f"""
            The unsaved data was created on {last_modified_time_unsaved}\n\n
            The user saved the data last time on {last_modified_time_saved}
        """)
        msg.setDetailedText(details)
        loadUnsavedButton = widgets.reloadPushButton('Recover unsaved data')
        loadSavedButton = widgets.savePushButton('Load saved data')
        infoButton = widgets.infoPushButton('More info...')
        loadSafeNpzButton = ''
        if posData.isSafeNpzOverwritePresent():
            loadSafeNpzButton = widgets.reloadPushButton(
                'Load .safe.npz file from crash'
            )
            buttons = (
                loadSavedButton, loadUnsavedButton, loadSafeNpzButton, 
                infoButton
            )
        else:
            buttons = (loadSavedButton, loadUnsavedButton, infoButton)
        msg.question(
            self.progressWin, 'Recover unsaved data?', txt, 
            buttonsTexts=('Cancel', *buttons), 
            showDialog=False
        )
        infoButton.disconnect()
        infoButton.clicked.connect(partial(self.showInfoAutosave, posData))
        msg.exec_()
        if msg.cancel:
            self.loadDataWorker.abort = True
        elif msg.clickedButton == loadUnsavedButton:
            self.loadDataWorker.loadUnsaved = True
        elif msg.clickedButton == loadSafeNpzButton:
            self.loadDataWorker.loadSafeOverwriteNpz = True
            
        self.loadDataWorker.waitCond.wakeAll()
        # self.AutoPilotProfile.storeLoadSavedData()

    def showInfoAutosave(self, posData):
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = (f"""
            Cell-ACDC either detected unsaved data in a previous session and it 
            stored it because the <b>Autosave</b><br>
            function was active, or it crashed during saving.<br><br>
            You can toggle Autosave ON and OFF from the menu on the top menubar 
            <code>File --> Autosave</code>.
        """)
        txt = (f"""
            {txt}<br><br>
            If Cell-ACDC crashed during saving, the segmentation file ending 
            with <code>.new.npz</code><br>
            is present and you might be able to recover the data from there. 
        """) 
        
        txt = (f"""
            {txt}<br><br>
            You can find additional recovered data in the following folder:
        """)  
        txt = html_utils.paragraph(txt)
        msg.information(
            self, 'Autosave info', txt, 
            path_to_browse=posData.recoveryFolderPath, 
            commands=(posData.recoveryFolderPath,)
        )

    def askMismatchSegmDataShape(self, posData):
        msg = widgets.myMessageBox(wrapText=False)
        title = 'Segm. data shape mismatch'
        f = '3D' if self.isSegm3D else '2D'
        f = f'{f} over time' if posData.SizeT > 1 else f
        r = '2D' if self.isSegm3D else '3D'
        r = f'{r} over time' if posData.SizeT > 1 else r
        text = html_utils.paragraph(f"""
            The segmentation masks of the first Position that you loaded is 
            <b>{f}</b>,<br>
            while {posData.pos_foldername} is <b>{r}</b>.<br><br>
            The loaded segmentation masks <b>must be</b> either <b>all 3D</b> 
            or <b>all 2D</b>.<br><br>
            Do you want to skip loading this position or cancel the process?
        """)
        _, skipPosButton = msg.warning(
            self, title, text, buttonsTexts=('Cancel', 'Skip this Position')
        )
        if skipPosButton == msg.clickedButton:
            self.loadDataWorker.skipPos = True
        self.loadDataWorker.waitCond.wakeAll()

    def workerPermissionError(self, txt, waitCond):
        msg = widgets.myMessageBox(parent=self)
        msg.setIcon(iconName='SP_MessageBoxCritical')
        msg.setWindowTitle('Permission denied')
        msg.addText(txt)
        msg.addButton('  Ok  ')
        msg.exec_()
        waitCond.wakeAll()

    def loadDataWorkerDataIntegrityCritical(self):
        errTitle = 'All loaded positions contains frames over time!'
        self.titleLabel.setText(errTitle, color='r')

        msg = widgets.myMessageBox(parent=self)

        err_msg = html_utils.paragraph(f"""
            {errTitle}.<br><br>
            To load data that contains frames over time you have to select
            only ONE position.
        """)
        msg.setIcon(iconName='SP_MessageBoxCritical')
        msg.setWindowTitle('Loaded multiple positions with frames!')
        msg.addText(err_msg)
        msg.addButton('Ok')
        msg.show(block=True)

    @exception_handler
    def loadDataWorkerFinished(self, data):
        self.funcDescription = 'loading data worker finished'
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None

        if data is None or data=='abort':
            self.loadingDataAborted()
            return
        
        if data[0].onlyEditMetadata:
            self.loadingDataAborted()
            return

        self.pos_i = 0
        self.data = data
        self.gui_createGraphicsItems()
        return True

    @exception_handler
    def loadingDataCompleted(self):
        self.isDataLoading = True
        posData = self.data[self.pos_i]
        
        files_format = '\n'.join([
            f'  - {file}' for file in posData.images_folder_files
        ])
        sep = '-'*100
        self.logger.info(
            f'{sep}\nFiles present in the first Position folder loaded:\n\n'
            f'{files_format}\n{sep}'
        )
        self.logger.info(f'Basename of the first Position: {posData.basename}')
        self.secondLevelToolbar.setVisible(True)
        self.updateImageValueFormatter()
        self.checkManageVersions()
        self.initManualBackgroundImage()
        self.initPixelSizePropsDockWidget()

        title_path = acdc_path.unexpanduser(posData.exp_path)
        self.setWindowTitle(
            f'Cell-ACDC v{self._acdc_version} - GUI - "{title_path}"'
        )
        
        self.setupPreprocessing()
        self.setupCombiningChannels()

        if self.isSegm3D:
            self.segmNdimIndicator.setText('3D')
        else:
            self.segmNdimIndicator.setText('2D')
            
        self.segmNdimIndicatorAction.setVisible(True)

        self.guiTabControl.addChannels([posData.user_ch_name])
        self.showPropsDockButton.setDisabled(False)

        self.bottomScrollArea.show()
        self.gui_createStoreStateWorker()
        self.init_segmInfo_df()
        self.connectScrollbars()
        self.initPosAttr()
        
        self.logger.info('Pre-computing min and max values of the images...')
        self.img1.preComputedMinMaxValues(self.data)
        self.img2.minMaxValuesMapper = self.img1.minMaxValuesMapper
        
        self.initMetrics()
        self.initFluoData()
        self.createChannelNamesActions()
        self.addActionsLutItemContextMenu(self.imgGrad)
        
        # Scrollbar for opacity of img1 (when overlaying)
        self.img1.alphaScrollbar = self.addAlphaScrollbar(
            self.user_ch_name, self.img1
        )

        self.navigateScrollBar.setSliderPosition(posData.frame_i+1)

        # Connect events at the end of loading data process
        self.gui_connectGraphicsEvents()
        if not self.isEditActionsConnected:
            self.gui_connectEditActions()
            self.normalizeToFloatAction.setChecked(True)
            
        self.navSpinBox.connectValueChanged(self.navigateSpinboxValueChanged)

        self.setFramesSnapshotMode()
        if self.isSnapshot:
            self.navSizeLabel.setText(f'/{len(self.data)}') 
        else:
            self.navSizeLabel.setText(f'/{posData.SizeT}')

        self.enableZstackWidgets(posData.SizeZ > 1)

        if 'showFrameIndex' in self.df_settings.index:
            checked = self.df_settings.at['showFrameIndex', 'value'] == 'Yes'
            if checked:
                self.showFrameIndexAction.setChecked(True)
        
        if 'showZsliceIndex' in self.df_settings.index:
            checked = self.df_settings.at['showZsliceIndex', 'value'] == 'Yes'
            if checked:
                self.showZsliceIndexAction.setChecked(True)
        
        self.exportToVideoAction.setDisabled(
            posData.SizeZ == 1 and posData.SizeT == 1
        )

        self.img1BottomGroupbox.show()

        isLabVisible = self.df_settings.at['isLabelsVisible', 'value'] == 'Yes'
        isRightImgVisible = (
            self.df_settings.at['isRightImageVisible', 'value'] == 'Yes'
        )
        isNextFrameVisible = (
            self.df_settings.at['isNextFrameVisible', 'value'] == 'Yes'
        )
        isNextFrameActive = (
            isNextFrameVisible and self.labelsGrad.showNextFrameAction.isEnabled()
        )
        self.updateScrollbars()
        self.openFolderAction.setEnabled(True)
        self.editTextIDsColorAction.setDisabled(False)
        self.imgPropertiesAction.setEnabled(True)
        self.navigateToolBar.setVisible(True)
        self.labelsGrad.showLabelsImgAction.setChecked(isLabVisible)
        self.labelsGrad.showRightImgAction.setChecked(isRightImgVisible)
        self.labelsGrad.showNextFrameAction.setChecked(isNextFrameActive)
        if isRightImgVisible or isNextFrameActive:
            self.rightBottomGroupbox.setChecked(True)
        
        isTwoImagesLayout = (
            isRightImgVisible or isLabVisible or isNextFrameActive
        )
        self.setTwoImagesLayout(isTwoImagesLayout)
        
        self.setBottomLayoutStretch()
        
        if isNextFrameActive:
            self.rightBottomGroupbox.show()
            self.rightBottomGroupbox.setChecked(True)

        self.readSavedCustomAnnot()
        self.addCustomAnnotButtonAllLoadedPos()
        self.setStatusBarLabel()

        self.initLookupTableLab()
        # First-load `invertBw`, then the LUT saved for the channel
        self.restoreImageChannelLutAfterLoading()
        self.restoreSavedSettings()

        self.initContoursImage()
        self.initTextAnnot()

        self.update_rp()
        self.updateAllImages()
        if posData.SizeT > 1:
            self.rightImageFramesScrollbar.setValueNoSignal(posData.frame_i+2)
        self.setMetricsFunc()

        self.gui_createLabelRoiItem()
        self.gui_createZoomRectItem()

        self.titleLabel.setText(
            'Data successfully loaded.',
            color=self.titleColor
        )

        self.setNonFunctionalButtonsDisabled()
        self.setVisible3DsegmWidgets()

        if len(self.data) == 1 and posData.SizeZ > 1 and posData.SizeT == 1:
            self.zSliceCheckbox.setChecked(True)
        else:
            self.zSliceCheckbox.setChecked(False)

        self.labelRoiCircItemLeft.setImageShape(self.currentLab2D.shape)
        self.labelRoiCircItemRight.setImageShape(self.currentLab2D.shape)

        self.retainSpaceSlidersToggled(self.retainSpaceSlidersAction.isChecked())

        self.stopAutomaticLoadingPos()
        self.viewAllCustomAnnotAction.setChecked(True)

        self.updateImageValueFormatter()

        posData.loadWhitelist()

        self.setFocusGraphics()
        self.setFocusMain()

        # Overwrite axes viewbox context menu
        self.ax1.vb.menu = self.imgGrad.gradient.menu
        self.ax2.vb.menu = self.labelsGrad.menu

        self.isDataLoaded = True
        self.isDataLoading = False
        
        self.initImgGradRescaleIntensitiesHowPreference()
        
        self.rescaleIntensitiesLut(setImage=False)
        
        self.gui_createAutoSaveWorker()
        
        self.initLoadedDelROI()
        
        self.separateBudToolbar.setSegm3D(self.isSegm3D)
        
        QTimer.singleShot(100, self.resizeGuiAndAutoRange)

    def restoreSavedSettings(self):        
        if 'addNewIDsWhitelistToggle' in self.df_settings.index:
            self.addNewIDsWhitelistToggle = (
                self.df_settings.at['addNewIDsWhitelistToggle', 'value']
                ) == 'Yes'
        else:
            self.addNewIDsWhitelistToggle = True
        
        self.restoreSavedAnnotationOptions()

    def init_segmInfo_df(self):
        for posData in self.data:
            if posData is None:
                # posData is None when computing measurements with the utility
                # and with timelapse data
                continue
            posData.init_segmInfo_df()

    def getAllPosChNames(self):
        """Ordered union of the channel names found in all loaded Positions
        (order of first appearance)."""
        allChNames = list(self.ch_names)
        for _posData in self.data:
            chNames = getattr(_posData, 'chNames', None)
            if not chNames:
                continue
            for chName in chNames:
                if chName not in allChNames:
                    allChNames.append(chName)
        return allChNames

    def initPosAttr(self):
        exp_path = self.data[self.pos_i].exp_path
        pos_foldernames = myutils.get_pos_foldernames(exp_path)
        if len(pos_foldernames) == 1:
            self.loadPosAction.setDisabled(True)
        else:
            self.loadPosAction.setDisabled(False)

        for p, posData in enumerate(self.data):
            self.pos_i = p
            posData.curvPlotItems = []
            posData.curvAnchorsItems = []
            posData.curvHoverItems = []
            posData.trackedLostIDs = set()

            posData.HDDmaxID = np.max(posData.segm_data)

            # Decision on what to do with changes to future frames attr
            posData.doNotShowAgain_EditID = False
            posData.UndoFutFrames_EditID = False
            posData.applyFutFrames_EditID = False

            posData.doNotShowAgain_RipID = False
            posData.UndoFutFrames_RipID = False
            posData.applyFutFrames_RipID = False

            posData.doNotShowAgain_DelID = False
            posData.UndoFutFrames_DelID = False
            posData.applyFutFrames_DelID = False

            posData.doNotShowAgain_keepID = False
            posData.UndoFutFrames_keepID = False
            posData.applyFutFrames_keepID = False
            
            posData.doNotShowAgainAssignNewID = False
            posData.UndoFutFramesAssignNewID = False
            posData.applyFutFramesAssignNewID = False

            posData.includeUnvisitedInfo = {
                'Delete ID': False, 'Edit ID': False, 'Keep ID': False
            }
            
            posData.acdcTracker2stepsAnnotInfo = {}

            posData.doNotShowAgain_BinID = False
            posData.UndoFutFrames_BinID = False
            posData.applyFutFrames_BinID = False

            posData.disableAutoActivateViewerWindow = False
            posData.new_IDs = []
            posData.lost_IDs = []
            posData.multiBud_mothIDs = [2]
            posData.UndoRedoStates = [[] for _ in range(posData.SizeT)]
            posData.UndoRedoCcaStates = [[] for _ in range(posData.SizeT)]

            posData.ol_data_dict = {}
            posData.ol_data = None

            posData.ol_labels_data = None       
            
            missing_frames = posData.SizeT - len(posData.allData_li)
            if missing_frames > 0:
                posData.allData_li.extend([None] * missing_frames)
            for i in range(posData.SizeT):
                if posData.allData_li[i] is None:
                    posData.allData_li[i] = (
                        myutils.get_empty_stored_data_dict()
                    )
            
            posData.lutLevels = {channel: {} for channel in self.ch_names}

            posData.ccaStatus_whenEmerged = {}

            posData.frame_i = 0
            posData.brushID = 0
            posData.binnedIDs = set()
            posData.ripIDs = set()
            posData.cca_df = None
            if posData.last_tracked_i is not None:
                last_tracked_num = posData.last_tracked_i+1
                # Load previous session data
                # Keep track of which ROIs have already been added
                # in previous frame
                delROIshapes = [[] for _ in range(posData.SizeT)]
                for i in range(last_tracked_num):
                    posData.frame_i = i
                    self.get_data()
                    self.store_data(
                        enforce=True, autosave=False, store_cca_df_copy=True
                    )
                    to_print = []
                    try:
                        to_print.append(posData.cca_df[['will_divide']])
                    except Exception as err:
                        pass
                    
                    try:
                        to_print.append(
                            posData.allData_li[posData.frame_i]['acdc_df'][cca_df_colnames].iloc[:, :-2]
                        )
                    except Exception as err:
                        pass

                self.initSingleMotherBudPairPartialCcaInfo()

                # Ask whether to resume from last frame
                if last_tracked_num>1:
                    msg = widgets.myMessageBox()
                    txt = html_utils.paragraph(
                        'Cell-ACDC detected a previous session ended '
                        f'at frame {last_tracked_num}.<br><br>'
                        f'Do you want to <b>resume from frame '
                        f'{last_tracked_num}?</b>'
                    )
                    noButton, yesButton = msg.question(
                        self, 'Start from last session?', txt,
                        buttonsTexts=(' No ', 'Yes')
                    )
                    self.AutoPilotProfile.storeClickMessageBox(
                        'Start from last session?', msg.clickedButton.text()
                    )
                    if msg.clickedButton == yesButton:
                        posData.frame_i = posData.last_tracked_i
                        self.lastFrameRanOnFirstVisitTools = posData.frame_i
                    else:
                        posData.frame_i = 0
                
            posData.img_data_min_max = (
                posData.img_data.min(), posData.img_data.max()
            )    

        # Back to first position
        self.pos_i = 0
        self.get_data(debug=False)
        self.store_data(autosave=False)
        # self.updateAllImages()

        # Link Y and X axis of both plots to scroll zoom and pan together
        self.ax2.vb.setYLink(self.ax1.vb)
        self.ax2.vb.setXLink(self.ax1.vb)

        self.setAllIDs()

    def reInitGui(self):
        cancel = self.checkAskSavePointsLayers()
        if cancel:
            return False
        
        # The overlay modes keep the user's check states (saved in the
        # settings): `overlay_cb` applies them to the new data's items

        # LUT of the channel to load: restored in `loadingDataCompleted`
        self.resetImageChannelLut()

        self.secondLevelToolbar.setVisible(False)
        
        self.gui_createLazyLoader()

        try:
            self.navSpinBox.valueChanged.disconnect()
        except Exception as e:
            pass
        
        try: 
            self.scaleBar.removeFromAxis(self.ax1)
        except Exception as e:
            pass

        self.showFrameIndexAction.setVisible(False)
        self.showZsliceIndexAction.setVisible(False)
        self.editObjectTrackSettingsAction.setVisible(False)
        
        if self.isDataLoaded:
            self.saveAnnotationOptions()

        self.lineage_tree = None
        self.clearLineageTreeBackup()
        self.getDistanceListMissingIDsCachedFrame = None
        self.isZmodifier = False
        self.zKeptDown = False
        self.askRepeatSegment3D = True
        self.askZrangeSegm3D = True
        self.isDataLoaded = False
        self.retainSizeLutItems = False
        self.setMeasWinState = None
        self.addPointsWin = None
        self.showPropsDockButton.setDisabled(True)
        self.removeOverlayItems()
        self.lutItemsLayout.addItem(self.imgGrad, row=0, col=0)
        
        self.reinitWidgetsPos()
        self.removeAllItems()
        self.reinitCustomAnnot()
        self.reinitPointsLayers()
        self.gui_createPlotItems()
        self.setUncheckedAllButtons()
        self.setUncheckedPointsLayers()
        self.restoreDefaultColors()
        self.reinitStoredSegmModels()
        self.removeAxLimits()
        self.updateSingleMotherBudPairLastAnnotatedFrame()
        self.curvToolButton.setChecked(False)
        
        self.wandControlsToolbar.setVisible(False)
        self.wandToolButton.setChecked(False)
        self.segmNdimIndicatorAction.setVisible(False)
        
        self.navigateToolBar.hide()
        self.ccaToolBar.hide()
        self.editToolBar.hide()
        self.brushEraserToolBar.hide()
        self.modeToolBar.hide()

        self.modeComboBox.setCurrentText('Viewer')
        
        alpha = self.imgGrad.labelsAlphaSlider.value()
        self.labelsLayerImg1.setOpacity(alpha)
        self.labelsLayerRightImg.setOpacity(alpha)
        self.lastTrackedFrameLabel.setText('')
        
        self.promptSegmentPointsLayerToolbar.isPointsLayerInit = False
        
        for action in self.askHowFutureFramesActions.values():
            action.setChecked(True)
            action.setDisabled(True)
        
        try:
            self.addScaleBarAction.setChecked(False)
        except Exception as err:
            pass

        try:
            self.addTimestampAction.setChecked(False)
        except Exception as err:
            pass
        
        return True

    def reinitWidgetsPos(self):
        pass

    def loadingDataAborted(self):
        self.openFolderAction.setEnabled(True)
        self.titleLabel.setText('Loading data aborted.')

    def cleanUpOnError(self):
        self.onEscape()
        caller = 'Cell-ACDC'
        if self.module.startswith('spotmax'):
            caller = 'spotMAX'
        txt = f'WARNING: {caller} is in error state. Please, restart.'
        _hl = '*'*100
        self.titleLabel.setText(txt, color='r')
        self.logger.info(f'{_hl}\n{txt}\n{_hl}')

    @exception_handler
    def loadDataWorkerDataIntegrityWarning(self, pos_foldername):
        err_msg = (
            'WARNING: Segmentation mask file ("..._segm.npz") not found. '
            'You could run segmentation module first.'
        )
        self.workerProgress(err_msg, 'INFO')
        self.titleLabel.setText(err_msg, color='r')
        abort = False
        msg = widgets.myMessageBox(parent=self)
        warn_msg = html_utils.paragraph(f"""
            The folder {pos_foldername} <b>does not contain a
            pre-computed segmentation mask</b>.<br><br>
            You can continue with a blank mask or cancel and
            pre-compute the mask with the segmentation module.<br><br>
            Do you want to continue?
        """)
        msg.setIcon(iconName='SP_MessageBoxWarning')
        msg.setWindowTitle('Segmentation file not found')
        msg.addText(warn_msg)
        msg.addButton('Ok')
        continueWithBlankSegm = msg.addButton(' Cancel ')
        msg.show(block=True)
        if continueWithBlankSegm == msg.clickedButton:
            abort = True
        self.loadDataWorker.abort = abort
        self.loadDataWaitCond.wakeAll()

    def warnMemoryNotSufficient(self, total_ram, available_ram, required_ram):
        total_ram = myutils._bytes_to_GB(total_ram)
        available_ram = myutils._bytes_to_GB(available_ram)
        required_ram = myutils._bytes_to_GB(required_ram)
        required_perc = round(100*required_ram/available_ram)
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(f"""
            The total amount of data that you requested to load is about
            <b>{required_ram:.2f} GB</b> ({required_perc}% of the available memory)
            but there are only <b>{available_ram:.2f} GB</b> available.<br><br>
            For <b>optimal operation</b>, we recommend loading <b>maximum 30%</b>
            of the available memory. To do so, try to close open apps to
            free up some memory. Another option is to crop the images
            using the data prep module.<br><br>
            If you choose to continue, the <b>system might freeze</b>
            or your OS could simply kill the process.<br><br>
            What do you want to do?
        """)
        cancelButton, continueButton = msg.warning(
            self, 'Memory not sufficient', txt,
            buttonsTexts=('Cancel', 'Continue anyway')
        )
        if msg.clickedButton == continueButton:
            # Disable autosaving since it would keep a copy of the data and 
            # we cannot afford it with low memory
            self.autoSaveToggle.setChecked(False)
            return True
        else:
            return False

    def checkMemoryRequirements(self, required_ram):
        memory = psutil.virtual_memory()
        total_ram = memory.total
        available_ram = memory.available
        if required_ram/available_ram > 0.3:
            proceed = self.warnMemoryNotSufficient(
                total_ram, available_ram, required_ram
            )
            return proceed
        else:
            return True

    def getSecondChannelData(self):
        if self.secondChannelName is None:
            return

        posData = self.data[self.pos_i]

        fluo_ch = self.secondChannelName
        fluo_path, filename = self.getPathAndFilenameNoExtFromChName(fluo_ch, posData)
        if filename in posData.fluo_data_dict:
            fluo_data = posData.fluo_data_dict[filename]
        else:
            fluo_data, bkgrData = self.load_fluo_data(fluo_path)
            posData.fluo_data_dict[filename] = fluo_data
            posData.fluo_bkgrData_dict[filename] = bkgrData
        
        if self.labelRoiTrangeCheckbox.isChecked():
            start_frame_i = self.labelRoiStartFrameNoSpinbox.value()-1
            stop_frame_n = self.labelRoiStopFrameNoSpinbox.value()
            tRangeLen = stop_frame_n-start_frame_i
        else:
            tRangeLen = 1
        
        if tRangeLen > 1:
            # fluo_img_data = fluo_data[start_frame_i:stop_frame_n]
            if self.isSegm3D or posData.SizeZ == 1:
                return fluo_data
            else:
                T, Z, Y, X = fluo_data.shape
                secondChannelData = np.zeros((T, Y, X), dtype=fluo_data.dtype)
                for frame_i, fluo_img in enumerate(fluo_data):
                    secondChannelData[frame_i] = self.get_2Dimg_from_3D(
                        fluo_data, frame_i=frame_i
                    )
                return secondChannelData
        else:
            if posData.SizeT > 1:
                fluo_img_data = fluo_data[posData.frame_i]
            else:
                fluo_img_data = fluo_data
                
            if self.isSegm3D or posData.SizeZ == 1:
                return fluo_img_data
            else:
                return self.get_2Dimg_from_3D(fluo_img_data)
    
        # annotationMenu = lutItem.gradient.menu.addMenu('Annotations settings')
        # ID_menu = annotationMenu.addMenu('IDs')
        # self.annotSettingsIDmenu = QActionGroup(annotationMenu)
        # labID_action = QAction("Show label's ID")
        # labID_action.setCheckable(True)
        # labID_action.setChecked(True)
        # labID_action.toggled.connect(self.annotLabelIDtreeToggled)
        # treeID_action = QAction("Show tree's ID")
        # treeID_action.setCheckable(True)
        # treeID_action.toggled.connect(self.annotLabelIDtreeToggled)
        # self.annotSettingsIDmenu.addAction(labID_action)
        # self.annotSettingsIDmenu.addAction(treeID_action)
        # ID_menu.addAction(labID_action)
        # ID_menu.addAction(treeID_action)

        # ID_menu = annotationMenu.addMenu('Generation number')
        # self.annotSettingsGenNumMenu = QActionGroup(annotationMenu)
        # gen_num_action = QAction("Show default generation number")
        # gen_num_action.setCheckable(True)
        # gen_num_action.setChecked(True)
        # gen_num_action.toggled.connect(self.annotGenNumTreeToggled)
        # tree_gen_num_action = QAction("Show tree generation number")
        # tree_gen_num_action.setCheckable(True)
        # tree_gen_num_action.toggled.connect(self.annotGenNumTreeToggled)
        # self.annotSettingsGenNumMenu.addAction(gen_num_action)
        # self.annotSettingsGenNumMenu.addAction(tree_gen_num_action)
        # ID_menu.addAction(gen_num_action)
        # ID_menu.addAction(tree_gen_num_action)

    def zSliceAbsent(self, filename, posData):
        self.app.restoreOverrideCursor()
        SizeZ = posData.SizeZ
        chNames = posData.chNames
        filenamesPresent = posData.segmInfo_df.index.get_level_values(0).unique()
        chNamesPresent = [
            ch for ch in chNames
            for file in filenamesPresent
            if file.endswith(ch) or file.endswith(f'{ch}_aligned')
        ]
        win = apps.QDialogZsliceAbsent(filename, SizeZ, chNamesPresent)
        win.exec_()
        if win.cancel:
            self.worker.abort = True
            self.waitCond.wakeAll()
            return
        
        parts = filename[len(posData.basename):].split(';;')
        if len(parts) == 2:
            channel_name = parts[1]
        else:
            channel_name = parts[0]
            
        if win.useMiddleSlice:
            for _posData in self.data:
                if _posData is None:
                    continue
                
                _, _filename = self.getPathAndFilenameNoExtFromChName(channel_name, _posData)
                df = myutils.getDefault_SegmInfo_df(_posData, _filename)
                _posData.segmInfo_df = pd.concat([df, _posData.segmInfo_df])
                unique_idx = ~_posData.segmInfo_df.index.duplicated()
                _posData.segmInfo_df = _posData.segmInfo_df[unique_idx]
                _posData.segmInfo_df.to_csv(_posData.segmInfo_df_csv_path)
        elif win.useSameAsCh:
            for _posData in self.data:
                if _posData is None:
                    continue
                _, srcFilename = self.getPathAndFilenameNoExtFromChName(
                    win.selectedChannel, _posData
                )
                cellacdc_df = _posData.segmInfo_df.loc[srcFilename].copy()
                _, dstFilename = self.getPathAndFilenameNoExtFromChName(
                    channel_name, _posData
                )
                if dstFilename is None:
                    self.worker.abort = True
                    self.waitCond.wakeAll()
                    return
                dst_df = myutils.getDefault_SegmInfo_df(_posData, dstFilename)
                for z_info in cellacdc_df.itertuples():
                    frame_i = z_info.Index
                    zProjHow = z_info.which_z_proj
                    if zProjHow == 'single z-slice':
                        src_idx = (srcFilename, frame_i)
                        if _posData.segmInfo_df.at[src_idx, 'resegmented_in_gui']:
                            col = 'z_slice_used_gui'
                        else:
                            col = 'z_slice_used_dataPrep'
                        z_slice = _posData.segmInfo_df.at[src_idx, col]
                        dst_idx = (dstFilename, frame_i)
                        dst_df.at[dst_idx, 'z_slice_used_dataPrep'] = z_slice
                        dst_df.at[dst_idx, 'z_slice_used_gui'] = z_slice
                _posData.segmInfo_df = pd.concat([dst_df, _posData.segmInfo_df])
                unique_idx = ~_posData.segmInfo_df.index.duplicated()
                _posData.segmInfo_df = _posData.segmInfo_df[unique_idx]
                _posData.segmInfo_df.to_csv(_posData.segmInfo_df_csv_path)
        elif win.runDataPrep:
            user_ch_file_paths = []
            for _posData in self.data:
                if _posData is None:
                    continue
                user_ch_path = load.get_filename_from_channel(
                    _posData.images_path, channel_name
                )
                if user_ch_path is None:
                    self.worker.abort = True
                    self.waitCond.wakeAll()
                    return
                user_ch_file_paths.append(user_ch_path)
                exp_path = os.path.dirname(_posData.pos_path)

            dataPrepWin = dataPrep.dataPrepWin()
            _window_stacking.set_window_flags_on_top(dataPrepWin, Qt.Window)
            dataPrepWin.titleText = (
            """
            Select z-slice (or projection) for each frame/position.<br>
            Once happy, close the window.
            """)
            dataPrepWin.show()
            dataPrepWin.initLoading()
            dataPrepWin.SizeT = self.data[0].SizeT
            dataPrepWin.SizeZ = self.data[0].SizeZ
            dataPrepWin.metadataAlreadyAsked = True
            self.logger.info(f'Loading channel {channel_name} data...')
            dataPrepWin.loadFiles(
                exp_path, user_ch_file_paths, channel_name
            )
            dataPrepWin.startAction.setDisabled(True)
            dataPrepWin.onlySelectingZslice = True

            loop = QEventLoop(self)
            dataPrepWin.loop = loop
            loop.exec_()

        self.waitCond.wakeAll()


class OpenFileGuiElements:
    """File menu: new file, open file or folder, versions, recent files and
    automatic loading of positions.
    """

    def checkManageVersions(self):
        posData = self.data[self.pos_i]
        posData.setTempPaths(createFolder=False)
        loaded_acdc_df_filename = os.path.basename(posData.acdc_output_csv_path)
        
        if os.path.exists(posData.recoveryFolderpath()):
            self.manageVersionsAction.setDisabled(False)
            self.manageVersionsAction.setToolTip(
                f'Load an older version of the `{loaded_acdc_df_filename}` file '
                '(table with annotations and measurements).'
            )
        else:
            self.manageVersionsAction.setDisabled(True)

    @exception_handler
    def _createEmptyData(self):
        self.MostRecentPath = self.getMostRecentPath()
        exp_path = QFileDialog.getExistingDirectory(
            self,
            'Select experiment folder where to create empty data',
            self.MostRecentPath
        )
        if not exp_path:
            return
        
        pos_path = os.path.join(exp_path, 'Position_1')
        images_path = os.path.join(pos_path, 'Images')
        if os.path.exists(images_path):
            raise FileExistsError(f'The following path already exists "{images_path}"')

        os.makedirs(images_path, exist_ok=True)
        
        basename = 'test_empty_'
        tif_filename = f'{basename}channel_1.tif'
        tif_filepath = os.path.join(images_path, tif_filename)
        empty_img = np.zeros((256,256), dtype=np.uint8)
        empty_img[0,0] = 255
        skimage.io.imsave(tif_filepath, empty_img)
        
        metadata_filename = f'{basename}metadata.csv'
        metadata_filepath = os.path.join(images_path, metadata_filename)
        df_metadata = pd.DataFrame({
            'Description': ['basename'],
            'values': [basename]
        })
        df_metadata.to_csv(metadata_filepath, index=False)
        
        self.isNewFile = True
        self._openFolder(exp_path=images_path)


    def segmNdimIndicatorClicked(self):
        ndimText = self.segmNdimIndicator.text()
        if ndimText == '2D':
            alternativeNdimText = '3D'
            toggleText = 'activate'
        else:
            alternativeNdimText = '2D'
            toggleText = 'de-activate'
        msg = widgets.myMessageBox(wrapText=False)
        important_txt = ("""
            The toggle to activate 3D segmentation is visible only when 
            the <code>Number of z-slices</code> is greater than 1.
        """)
        txt = html_utils.paragraph(f"""
            This indicator shows that you are working with {ndimText} 
            segmentation masks.<br><br>
            
            If instead, you want to work with {alternativeNdimText} segmentation, 
            you need to initialize a new segmentation file.<br><br>
            
            To do so, go the menu on the top menubar <code>File --> 
            New Segmentation File...</code> and,<br>
            at the dialog where you insert the metadata (Number of z-slices, 
            pixel size, etc.),<br> 
            <b>{toggleText}</b> the parameter called <code>Work with 3D 
            segmentation masks (z-stack)</code><br> 
            as indicated in the screenshot below<br>.
            {html_utils.to_admonition(important_txt, admonition_type='note')}
            <br>
        """)
        msg.information(
            self, 'Segmentation nmber of dimensions info', txt,
            image_paths=':toggle_3D_screenshot.png'
        )
        self.segmNdimIndicator.setChecked(True)

    def newFile(self):
        self.newSegmEndName = ''
        self.isNewFile = True
        msg = widgets.myMessageBox(parent=self, showCentered=False)
        msg.setWindowTitle('File or folder?')
        msg.addText(html_utils.paragraph(f"""
            Do you want to load an <b>image file</b> or <b>Position 
            folder(s)</b>?
        """))
        loadPosButton = QPushButton('Load Position folder', msg)
        loadPosButton.setIcon(QIcon(":folder-open.svg"))
        loadFileButton = QPushButton('Load image file', msg)
        loadFileButton.setIcon(QIcon(":image.svg"))
        helpButton = widgets.helpPushButton('Help...')
        msg.addButton(helpButton)
        helpButton.disconnect()
        helpButton.clicked.connect(self.helpNewFile)
        msg.addCancelButton(connect=True)
        msg.addButton(loadFileButton)
        msg.addButton(loadPosButton)
        loadPosButton.setDefault(True)
        msg.exec_()
        if msg.cancel:
            return
        
        if msg.clickedButton == loadPosButton:
            self._openFolder()
        else:
            self._openFile()

    def openNewWindow(self):
        self.logger.info('Opening a new window...')
        if self.launcherSlot is not None:
            self.launcherSlot()
            return

        winClass = self.__class__
        win = winClass(
            self.app, parent=self, mainWin=self.mainWin, version=self._version
        )
        win.run()
        self.newWindows.append(win)

    def helpNewFile(self):
        msg = widgets.myMessageBox(showCentered=False)
        href = f'<a href="{user_manual_url}">user manual</a>'
        txt = html_utils.paragraph(f"""
            Cell-ACDC can open both a single image file or files structured 
            into Position folders.<br><br>
            If you are just testing out you can load a single image file, but 
            in general <b>we reccommend structuring your data into Position 
            folders.</b><br><br>
            More info about Position folders in the {href} at the section 
            called "Create required data structure from microscopy file(s)".
        """)
        msg.information(
            self, 'Help on Position folders', txt
        )

    def openFile(self, checked=False, file_path=None):
        self.logger.info(f'Opening FILE "{file_path}"')

        self.isNewFile = False
        self._openFile(file_path=file_path)

    def manageVersions(self):
        posData = self.data[self.pos_i]
        selectVersion = apps.SelectAcdcDfVersionToRestore(posData, parent=self)
        selectVersion.exec_()

        if selectVersion.cancel:
            return

        undoId = uuid.uuid4()
        if posData.cca_df is not None:
            self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)
        
        selectedTime = selectVersion.selectedTimestamp

        self.modeComboBox.setCurrentText('Viewer')
        self.logger.info(f'Loading file from {selectedTime}...')

        acdc_df = load.read_acdc_df_from_archive(
            selectVersion.archiveFilePath, selectVersion.selectedKey
        )
        posData.acdc_df = acdc_df
        frames = acdc_df.index.get_level_values(0)
        last_visited_frame_i = frames.max()
        current_frame_i = posData.frame_i
        pbar = tqdm(total=last_visited_frame_i+1, ncols=100)
        for frame_i in range(last_visited_frame_i+1):
            posData.frame_i = frame_i
            self.get_data()
            if posData.cca_df is not None:
                self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)
            if posData.allData_li[frame_i]['labels'] is None:
                pbar.update()
                continue
        
            if frame_i not in frames:
                acdc_df_i = pd.DataFrame(columns=acdc_df.columns)
                acdc_df_i.drop(self.cca_df_colnames, axis=1, errors='ignore')
                acdc_df_i.index.name = 'Cell_ID'
            else:
                acdc_df_i = acdc_df.loc[frame_i].dropna(axis=1, how='all')
            
            posData.allData_li[frame_i]['acdc_df'] = acdc_df_i
            pbar.update()
        pbar.close()
        
        # Back to current frame
        posData.frame_i = current_frame_i
        self.get_data(debug=False)
        self.updateAllImages()
        self.logger.info('Annotations correctly recovered.')

    def askUserChannelName(self, filename_no_ext):
        basename = filename_no_ext
        underscore_splits = filename_no_ext.split('_')
        if len(underscore_splits) > 1:
            channel_name = underscore_splits[-1]
            basename = '_'.join(underscore_splits[:-1])
        else:
            channel_name = 'channel_1'
        
        txt = html_utils.paragraph(f"""
            Provide some text (e.g., the channel name) to identify the image file.
        """)
        
        win = apps.QLineEditDialog(
            title='Provide id text for image file',
            msg=txt,
            defaultTxt=channel_name,
            parent=self,
            allowEmpty=False,
            allowText=True
        )
        win.exec_()
        if win.cancel:
            return True, ''
        
        return False, win.enteredValue

    def askUserCreationImagesFolder(self, images_path, ext):
        msg = widgets.myMessageBox(wrapText=False)
        txt = (f"""
            Cell-ACDC requires a specific folder structure to load the data.<br><br>
            Specifically, it requires the <b>image(s) to be located in a
            folder called <code>Images</code></b>.<br><br>
            The <b>file format</b> of the images must be <b>TIFF, H5, or NPZ</b> 
            (.tif, .h5, or .npz extension).<br><br>
            However, Cell-ACDC can also create a <b>symbolic link</b> to the image file.<br><br> 
            A symbolic link is a special type of file that acts as a pointer or alias,<br>
            referring to another file by its path rather than its content.<br><br>
            With the symbolic link, no data will be copied or moved, but the 
            source data file's location<br>
            cannot change, otherwise the link will be broken.<br><br>
            Note that any new file created by Cell-ACDC will be saved in the newly created <code>Images</code> folder (see below).<br><br>            
            You can choose to let Cell-ACDC create the required data structure 
            from your file,<br>
            or you can stop the 
            process and manually place the image(s) into a folder called 
            <code>Images</code>.<br><br>
            If you choose to proceed, Cell-ACDC will create the following 
            folder:
            <copiable>{images_path}</copiable>
            <br>
            How do you want to proceed?
        """)
        txt = html_utils.paragraph(txt)
        symlinkButton = widgets.SegmentPushButton(
            'Create symbolic link to the image'
        )
        copyButton = None
        moveButton = None
        if ext == '.tif' or ext == '.npz':
            copyButton = widgets.copyPushButton(
                'Copy the image into the new folder'
            )
            moveButton = widgets.movePushButton(
                'Move the image into the new folder'
            )
            buttonsTexts = (
                'Cancel', 
                copyButton, 
                moveButton,
                symlinkButton
            )
        else:
            copyButton = widgets.copyPushButton(
                'Create grayscale TIFF file in the new folder'
            )
            buttonsTexts=('Cancel', copyButton, symlinkButton)
        
        msg.information(
            self, 'Creating Images folder', txt, buttonsTexts=buttonsTexts
        )
        do_copy = msg.clickedButton == copyButton
        do_move = msg.clickedButton == moveButton
        use_symlink = msg.clickedButton == symlinkButton
        
        return msg.cancel, do_copy, do_move, use_symlink

    @exception_handler
    def _openFile(self, file_path=None):
        """
        Function used for loading an image file directly.
        """
        if file_path is None:
            self.MostRecentPath = self.getMostRecentPath()
            file_path = QFileDialog.getOpenFileName(
                self, 'Select image file', self.MostRecentPath,
                "Image/Video Files (*.png *.tif *.tiff *.jpg *.jpeg *.mov *.avi *.mp4)"
                ";;All Files (*)")[0]
            if not file_path:
                return
        
        filename, ext = os.path.splitext(os.path.basename(file_path))
        ext = ext.lower()
        dirpath = os.path.dirname(file_path)
        dirname = os.path.basename(dirpath)
        filename = filename.rstrip('_')
        channel_name = None
        do_copy = True
        do_move = False
        use_symlink = False
        if dirname != 'Images':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            acdc_folder = f'{timestamp}_acdc'
            exp_path = os.path.join(dirpath, acdc_folder, 'Images')
            out = self.askUserCreationImagesFolder(exp_path, ext)
            cancel, do_copy, do_move, use_symlink = out
            if cancel:
                self.logger.info('Loading image file cancelled.')
                return
            
            cancel, channel_name = self.askUserChannelName(filename)
            if cancel:
                self.logger.info('Loading image file cancelled.')
                return
            
            os.makedirs(exp_path, exist_ok=True)
        else:
            exp_path = dirpath

        if channel_name is not None:
            # Check if user wants to use the existing channel name
            underscore_splits = filename.split('_')
            if len(underscore_splits) > 1:
                default_ch_name = underscore_splits[-1]
                if channel_name == default_ch_name:
                    filename = '_'.join(underscore_splits[:-1])
                    
            basename = f'{filename}_'
            new_filename = f'{filename}_{channel_name}{ext}'
            df_metadata = pd.DataFrame({
                'Description': ['basename', 'channel_0_name'],
                'values': [basename, channel_name]
            })
            metadata_csv_filename = f'{basename}metadata.csv'
            metadata_csv_filepath = os.path.join(
                exp_path, metadata_csv_filename
            )
            df_metadata.to_csv(metadata_csv_filepath, index=False)
        else:
            new_filename = f'{filename}{ext}'
        
        action_text = ''
        if do_copy:
            action_text = 'Copying'
        elif do_move:
            action_text = 'Moving'
        elif use_symlink:
            action_text = 'Creating symbolic link'
        
        if use_symlink:
            symlink_filepath = load.save_symlink_ini_from_image_filepath(
                file_path, exp_path, channel_name
            )
            self._openFolder(
                exp_path=exp_path, 
                imageFilePath=symlink_filepath,
                user_ch_name=channel_name
            )
        elif ext == '.tif' or ext == '.npz' or ext == '.tiff':
            new_filepath = os.path.join(exp_path, new_filename)
            if not os.path.exists(new_filepath):
                self.logger.info(f'{action_text} file to Images folder...')
                if do_copy:
                    shutil.copy2(file_path, new_filepath)
                else:
                    shutil.move(file_path, new_filepath)
            self._openFolder(exp_path=exp_path, imageFilePath=new_filepath)
        else:
            self.logger.info(f'{action_text} file to .tif format...')
            data = load.loadData(file_path, '', log_func=self.logger.info)
            data.loadImgData()
            img = data.img_data
            if img.ndim == 3 and (img.shape[-1] == 3 or img.shape[-1] == 4):
                self.logger.info('Converting RGB image to grayscale...')
                data.img_data = colors.image_2d_rgb_or_rgba_to_uint8_grayscale(
                    data.img_data
                )
            new_filename_no_ext, ext = os.path.splitext(new_filename)
            tif_filename = f'{new_filename_no_ext}.tif'
            tif_path = os.path.join(exp_path, tif_filename)
            if data.img_data.ndim == 3:
                SizeT = data.img_data.shape[0]
                SizeZ = 1
            elif data.img_data.ndim == 4:
                SizeT = data.img_data.shape[0]
                SizeZ = data.img_data.shape[1]
            else:
                SizeT = 1
                SizeZ = 1
            is_imageJ_dtype = (
                data.img_data.dtype == np.uint8
                or data.img_data.dtype == np.uint32
                or data.img_data.dtype == np.uint32
                or data.img_data.dtype == np.float32
            )
            if not is_imageJ_dtype:
                data.img_data = skimage.img_as_ubyte(data.img_data)

            myutils.to_tiff(tif_path, data.img_data)
            self._openFolder(exp_path=exp_path, imageFilePath=tif_path)

    def criticalNoTifFound(self, images_path):
        err_title = 'No .tif files found in folder.'
        err_msg = html_utils.paragraph(
            'The following folder<br><br>'
            f'<code>{images_path}</code><br><br>'
            '<b>does not contain .tif or .h5 files</b>.<br><br>'
            'Only .tif or .h5 files can be loaded with "Open Folder" button.<br><br>'
            'Try with <code>File --> Open image/video file...</code> '
            'and directly select the file you want to load.'
        )
        msg = widgets.myMessageBox()
        msg.addShowInFileManagerButton(images_path)
        msg.critical(self, err_title, err_msg)

    def openFolder(
            self, checked=False, exp_path=None, imageFilePath=''
        ):
        if exp_path is None:
            self.logger.info('Asking to select a folder path...')
        else:
            self.logger.info(f'Opening FOLDER "{exp_path}"...')

        self.isNewFile = False
        if hasattr(self, 'data') and self.titleLabel.text != 'Saved!':
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                'Do you want to <b>save</b> before loading another dataset?'
            )
            _, no, yes = msg.question(
                self, 'Save?', txt,
                buttonsTexts=('Cancel', 'No', 'Yes')
            )
            if msg.clickedButton == yes:
                func = partial(self._openFolder, exp_path, imageFilePath)
                cancel = self.saveData(finishedCallback=func)
                return
            elif msg.cancel:
                self.store_data()
                return
            else:
                self.store_data(autosave=False)
                
        self._openFolder(
            exp_path=exp_path, imageFilePath=imageFilePath
        )

    @exception_handler
    def _openFolder(
            self, 
            checked=False, 
            exp_path=None, 
            imageFilePath='', 
            user_ch_name=None
        ):
        """Main function to load data.

        Parameters
        ----------
        checked : bool
            kwarg needed because openFolder can be called by openFolderAction.
        exp_path : string or None
            Path selected by the user either directly, through openFile,
            or drag and drop image file.
        imageFilePath : string
            Path of the image file that was either drag and dropped or opened
            from File --> Open image/video file (openFileAction).

        Returns
        -------
            None
        """

        if exp_path is None:
            self.MostRecentPath = self.getMostRecentPath()
            exp_path = QFileDialog.getExistingDirectory(
                self,
                'Select experiment folder containing Position_n folders '
                'or specific Position_n folder',
                self.MostRecentPath
            )

        if not exp_path:
            self.openFolderAction.setEnabled(True)
            return

        proceed = self.reInitGui()
        if not proceed:
            self.openFolderAction.setEnabled(True)
            return

        self.openFolderAction.setEnabled(False)

        if self.slideshowWin is not None:
            self.slideshowWin.close()

        if self.ccaTableWin is not None:
            self.ccaTableWin.close()

        self.exp_path = exp_path
        self.logger.info(f'Loading from {self.exp_path}')
        self.addToRecentPaths(exp_path, logger=self.logger)
        self.addPathToOpenRecentMenu(exp_path)

        folder_type = myutils.determine_folder_type(exp_path)
        is_pos_folder, is_images_folder, exp_path = folder_type

        self.titleLabel.setText('Loading data...', color=self.titleColor)

        skip_channels = []
        ch_name_selector = prompts.select_channel_name(
            which_channel='segm', allow_abort=False
        )
        if not is_pos_folder and not is_images_folder and not imageFilePath:
            images_paths = self._loadFromExperimentFolder(exp_path)
            if not images_paths:
                self.loadingDataAborted()
                return

        elif is_pos_folder and not imageFilePath:
            pos_foldername = os.path.basename(exp_path)
            exp_path = os.path.dirname(exp_path)
            images_paths = [os.path.join(exp_path, pos_foldername, 'Images')]

        elif is_images_folder and not imageFilePath:
            images_paths = [exp_path]
            pos_path = os.path.dirname(exp_path)
            exp_path = os.path.dirname(pos_path)
            
        elif imageFilePath:
            # images_path = exp_path because called by openFile func
            filenames = myutils.listdir(exp_path)
            ch_names, basenameNotFound = (
                ch_name_selector.get_available_channels(filenames, exp_path)
            )
            filename = os.path.basename(imageFilePath)
            self.ch_names = ch_names
            if user_ch_name is None:
                user_ch_name = [
                    chName for chName in ch_names if filename.find(chName)!=-1
                ][0]
            images_paths = [exp_path]
            pos_path = os.path.dirname(exp_path)
            exp_path = os.path.dirname(pos_path)

        self.images_paths = images_paths

        # Get info from first position selected
        images_path = self.images_paths[0]
        filenames = myutils.listdir(images_path)
        if ch_name_selector.is_first_call and user_ch_name is None:
            ch_names, _ = ch_name_selector.get_available_channels(
                filenames, images_path
            )
            self.ch_names = ch_names
            if not ch_names:
                self.openFolderAction.setEnabled(True)
                self.criticalNoTifFound(images_path)
                return
            if len(ch_names) > 1:
                CbLabel='Select channel name to load: '
                ch_name_selector.QtPrompt(
                    self, ch_names, CbLabel=CbLabel
                )
                if ch_name_selector.was_aborted:
                    self.openFolderAction.setEnabled(True)
                    return
                skip_channels.extend([
                    ch for ch in ch_names if ch!=ch_name_selector.channel_name
                ])
            else:
                ch_name_selector.channel_name = ch_names[0]
            ch_name_selector.setUserChannelName()
            user_ch_name = ch_name_selector.user_ch_name
        else:
            # File opened directly with self.openFile
            ch_name_selector.channel_name = user_ch_name

        user_ch_file_paths = []
        not_allowed_ends = ['btrack_tracks.h5']
        for images_path in self.images_paths:
            channel_file_path = load.get_filename_from_channel(
                images_path, user_ch_name, skip_channels=skip_channels,
                not_allowed_ends=not_allowed_ends, logger=self.logger.info
            )
            if not channel_file_path:
                self.criticalImgPathNotFound(images_path)
                return
            user_ch_file_paths.append(channel_file_path)

        ch_name_selector.setUserChannelName()
        self.user_ch_name = user_ch_name
        self.img1.channelName = user_ch_name

        self.AutoPilotProfile.storeSelectedChannel(self.user_ch_name)

        self.initGlobalAttr()
        self.createOverlayContextMenu()
        self.createUserChannelNameAction()
        self.gui_createOverlayColors()
        self.gui_createOverlayItems()
        lastRow = self.bottomLeftLayout.rowCount()
        self.bottomLeftLayout.setRowStretch(lastRow+1, 1)

        self.num_pos = len(user_ch_file_paths)
        proceed = self.loadSelectedData(user_ch_file_paths, user_ch_name)
        if not proceed:
            self.openFolderAction.setEnabled(True)
            return

    def _loadFromExperimentFolder(self, exp_path):
        select_folder = load.select_exp_folder()
        values = select_folder.get_values_segmGUI(exp_path)
        if not values:
            self.criticalInvalidPosFolder(exp_path)
            self.openFolderAction.setEnabled(True)
            return []

        if len(values) > 1:
            select_folder.QtPrompt(self, values, allow_cancel=False)
            if select_folder.cancel:
                return []
        else:
            select_folder.cancel = False
            select_folder.selected_pos = select_folder.pos_foldernames

        images_paths = []
        for pos in select_folder.selected_pos:
            images_paths.append(os.path.join(exp_path, pos, 'Images'))
        return images_paths

    def criticalInvalidPosFolder(self, exp_path):
        href = html_utils.href_tag('here', urls.data_structure_docs_url)
        txt = html_utils.paragraph(f"""
            The selected folder:<br><br>
            
            <code>{exp_path}</code><br><br>
            
            is <b>not a valid folder</b>.<br><br>
            
            Select a folder that contains the Position_n folders, 
            or a specific Position.<br><br>
            
            If you are trying to load a single image file go to 
            <code>File --> Open image/video file...</code>.<br><br>
            
            To load a folder containing multiple .tif files the folder must 
            be called either <code>Position_n</code><br>
            (with <code>n</code> being an integer) or <code>Images</code>.<br><br>
            
            For more information about the correct folder structure see {href}.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        helpButton = widgets.helpPushButton('Help...')
        msg.addButton(helpButton)
        helpButton.clicked.disconnect()
        helpButton.clicked.connect(
            partial(myutils.browse_url, urls.data_structure_docs_url)
        )
        msg.addShowInFileManagerButton(exp_path)
        msg.critical(
            self, 'Incompatible folder', txt
        )

    def criticalImgPathNotFound(self, images_path):
        self.logger.info(
            'The following folder does not contain valid image files: '
            f'"{images_path}"\n\n'
            'Check that all the positions loaded contain the same channel name. '
            'Make sure to double check for spelling mistakes or types in the '
            'channel names.'
        )
        msg = widgets.myMessageBox()
        msg.addShowInFileManagerButton(images_path)
        err_msg = html_utils.paragraph(f"""
            The folder<br><br>
            <code>{images_path}</code><br><br>
            <b>does not contain any valid image file!</b><br><br>
            Valid file formats are .h5, .tif, _aligned.h5, _aligned.npz.
        """)
        okButton = msg.critical(
            self, 'No valid files found!', err_msg, buttonsTexts=('Ok',)
        )

    def loadPosTriggered(self):
        if not self.isDataLoaded:
            return
        
        self.startAutomaticLoadingPos()

    def startAutomaticLoadingPos(self):
        self.AutoPilot = autopilot.AutoPilot(self)
        self.AutoPilot.execLoadPos()

    def stopAutomaticLoadingPos(self):
        if self.AutoPilot is None:
            return
        
        if self.AutoPilot.timer.isActive():
            self.AutoPilot.timer.stop()
        self.AutoPilot = None

    def openRecentFile(self, path):
        self.logger.info(f'Opening recent folder: {path}')
        self.addToRecentPaths(path, logger=self.logger)
        self.openFolder(exp_path=path)
