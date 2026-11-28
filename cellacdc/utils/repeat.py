import os
import sys
import traceback
import shutil

from qtpy.QtCore import Qt, QThread, QSize
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QListWidgetItem 
)
from qtpy import QtGui

from .. import exception_handler
from .. import html_utils, load
from cellacdc.myutils import app_info, filesystem, image_utils
from cellacdc.shared.dialogs import file_selection as dialogs_file_selection
from cellacdc.shared.widgets import (
    basic_widgets as widgets_basic_widgets,
    item_views as widgets_item_views,
    message_box as widgets_message_box,
    push_buttons as widgets_push_buttons,
)
from qtpy.QtCore import (
    Signal, QObject, QMutex, QWaitCondition
)
from cellacdc import prompts, io
from cellacdc import core

class repeatDataPrepWindow(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        name = 'repeat data prep'
        
        logger, logs_path, log_path, log_filename = app_info.setupLogger(
            module=name
        )

        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path

        self.logger.info(f'Initializing {name}...')

        self.cancel = True

        self.setWindowTitle(f'Cell-ACDC {name}')
        self.funcDescription = f'Cell-ACDC {name}'

        instructions = [
            'Press <b>start<b> button',
            '<b>Select experiment</b> folder or specific Position folder',
            'Select which <b>channels</b> or <b>un-prepped .tif file</b> to apply data prep to',
            '<b>Wait</b> until process ends'
        ]

        txt = html_utils.paragraph(f"""
            This utility is used to <b>re-apply data prep</b> steps such as 
            cropping and aligning.<br><br>
            This is needed when you are adding <b>new channels</b> 
            to already prepped data.<br><br>
            How to use it:
            {html_utils.to_list(instructions, ordered=True)}
        """)

        layout = QVBoxLayout()
        textLayout = QHBoxLayout()

        pixmap = QtGui.QIcon(":cog_play.svg").pixmap(QSize(64,64))
        iconLabel = QLabel()
        iconLabel.setPixmap(pixmap)

        textLayout.addWidget(iconLabel, alignment=Qt.AlignTop)
        textLayout.addSpacing(20)
        textLayout.addWidget(QLabel(txt))
        textLayout.addStretch(1)

        buttonsLayout = QHBoxLayout()
        stopButton = widgets_push_buttons.stopPushButton('Stop process')
        startButton = widgets_push_buttons.playPushButton('    Start     ')
        cancelButton = widgets_push_buttons.cancelPushButton('Close')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(startButton)
        buttonsLayout.addWidget(stopButton)

        self.startButton = startButton
        self.stopButton = stopButton

        progressBarLayout = QHBoxLayout()
        self.progressBar = widgets_basic_widgets.ProgressBarWithETA(parent=self)
        progressBarLayout.addWidget(self.progressBar)
        progressBarLayout.addWidget(self.progressBar.ETA_label)
        self.logConsole = widgets_basic_widgets.QLogConsole(parent=self)

        layout.addLayout(textLayout)
        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)
        layout.addSpacing(20)
        layout.addLayout(progressBarLayout)
        layout.addWidget(self.logConsole)

        self.setLayout(layout)

        cancelButton.clicked.connect(self.close)
        startButton.clicked.connect(self.start)
        stopButton.clicked.connect(self.stop)
    
    def showEvent(self, event: QtGui.QShowEvent) -> None:
        self.startButton.setFixedWidth(self.stopButton.width())
        self.stopButton.hide()
        return super().showEvent(event)

    @exception_handler    
    def start(self):
        self.startButton.hide()
        self.stopButton.show()

        MostRecentPath = filesystem.getMostRecentPath()
        exp_path = QFileDialog.getExistingDirectory(
            self, 'Select experiment folder or specific Position folder', 
            MostRecentPath
        )
        if not exp_path:
            self.logger.info('No path selected. Process stopped.')
            self.stop()
            return
        
        filesystem.addToRecentPaths(exp_path, logger=self.logger)
        
        folder_type = filesystem.determine_folder_type(exp_path)
        is_pos_folder, is_images_folder, exp_path = folder_type

        if is_pos_folder:
            pos_path = exp_path
            exp_path = os.path.dirname(pos_path)
            posFoldernames = [os.path.basename(pos_path)]
        elif is_images_folder:
            pos_path = os.path.dirname(exp_path)
            exp_path = os.path.dirname(pos_path)
            posFoldernames = [os.path.basename(pos_path)]
        else:
            select_folder = load.select_exp_folder()
            values = select_folder.get_values_dataprep(exp_path)
            if not values:
                self.criticalNotValidFolder(exp_path)
                self.stop()
                return
            if len(values) > 1:
                select_folder.QtPrompt(
                    self, values, allow_cancel=False, toggleMulti=True,
                    CbLabel="Select Position folder(s) to process:"
                )
                if select_folder.cancel:
                    self.logger.info(
                        'Process aborted by the user '
                        '(cancelled at Postion selection)'
                    )
                    self.stop()
                    return
                posFoldernames = select_folder.selected_pos
            else:
                posFoldernames = select_folder.pos_foldernames

        self.workerProgress(f'Selected folder: "{exp_path}"')
        self.workerProgress(' ')
        posListFormat = '\n'.join(posFoldernames)
        self.workerProgress(f'Selected Positions:\n{posListFormat}')
        self.workerProgress(' ')

        self.workerInitProgressBar(len(posFoldernames))
        
        self.thread = QThread()
        self.worker = reapplyDataPrepWorker(exp_path, posFoldernames)

        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.progress.connect(self.workerProgress)
        self.worker.initPbar.connect(self.workerInitProgressBar)
        self.worker.updatePbar.connect(self.workerUpdateProgressBar)
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.workerFinished)
        self.worker.sigCriticalNoChannels.connect(self.criticalNoChannelsFound)
        self.worker.sigSelectChannels.connect(self.selectChannels)

        self.thread.started.connect(self.worker.run)
        self.thread.start()
    
    def selectChannels(self, ch_name_selector, ch_names, imagesPath, basename):
        if basename is not None:
            self.ch_names = ch_names
            self.basename = basename
            self.imagesPath = imagesPath
            browseButton = widgets_push_buttons.browseFileButton(
                'Select .tif file to add and prep', start_dir=imagesPath,
                title='Select .tif file to add and prep', ext={'TIFF files': '.tif'}
            )
            browseButton.sigPathSelected.connect(self.selectTifFileToAdd)
            additionalButtons = (browseButton,)
        else:
            additionalButtons = []
        self.selectChannelWindow = widgets_item_views.QDialogListbox(
            'Select channel',
            'Select channel names to process:\n',
            ch_names, multiSelection=True, parent=self, 
            additionalButtons=additionalButtons
        )
        self.selectChannelWindow.exec_()
        if self.selectChannelWindow.cancel:
            self.worker.abort = True
        self.worker.selectedChannels = self.selectChannelWindow.selectedItemsText
        self.worker.waitCond.wakeAll()
    
    def selectTifFileToAdd(self, tif_file_path):
        tif_filename = os.path.splitext(os.path.basename(tif_file_path))[0]
        win = dialogs_file_selection.filenameDialog(
            ext='.tif', basename=self.basename, 
            title='Insert a name for new channel', 
            hintText='Insert a name for the new channel', 
            allowEmpty=False, defaultEntry=tif_filename, 
            existingNames=self.ch_names, parent=self.selectChannelWindow
        )
        win.exec_()
        if win.cancel:
            return
        
        newTifFilePath = os.path.join(self.imagesPath, win.filename)
        try:
            self.logger.info(f'Copying and renaming "{tif_filename}.tif" file...')
            shutil.copy2(tif_file_path, newTifFilePath)
        except Exception as e:
            error = traceback.format_exc()
            self.warnCopyTifFileFailed(tif_file_path, newTifFilePath, error)
            return
        newChannel = win.entryText
        newItem = QListWidgetItem(newChannel)
        self.selectChannelWindow.areItemsSelected.append(newItem)
        self.selectChannelWindow.listBox.addItem(newItem)
        self.selectChannelWindow.listBox.clearSelection()
        newItem.setSelected(True)
    
    def warnCopyTifFileFailed(self, tif_file_path, newTifFilePath, error):
        tifFilename = os.path.basename(tif_file_path)
        msg = widgets_message_box.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph(f"""
            Unfortunately, <b>copying and renaming</b> the file 
            <code>{tifFilename}</code> <b>failed</b> (see error detail below).<br><br>
            Please, <b>manually copy the file</b> to the following location and then try 
            again.<br><br>
            Copy to: <code>{newTifFilePath}</code>
        """)
        msg.setDetailedText(error)
        msg.critical(self.selectChannelWindow, 'Copy .tif file failed', txt)
    
    def criticalNotValidFolder(self, path: os.PathLike):
        txt = html_utils.paragraph(
            'The selected folder:<br><br>'
            f'<code>{path}</code><br><br>'
            'is <b>not a valid folder</b>. '
            'Select a folder that contains the Position_n folders'
        )
        msg = widgets_message_box.myMessageBox()
        msg.addShowInFileManagerButton(path)
        msg.critical(
            self, 'Incompatible folder', txt,
            buttonsTexts=('Ok',)
        )
    
    def criticalNoChannelsFound(self, images_path):
        err_title = 'Channel names not found'
        err_msg = html_utils.paragraph(
            'The following folder<br><br>'
            '<code>{images_path}</code><br><br>'
            '<b>does not valid channel files</b>.<br>'
        )
        msg = widgets_message_box.myMessageBox()
        msg.addShowInFileManagerButton(images_path)
        msg.critical(self, err_title, err_msg)
        self.logger.info(err_title)
        self.stop()
    
    def stop(self):
        self.startButton.show()
        self.stopButton.hide()

        if hasattr(self, 'worker'):
            self.worker.abort = True
    
    @exception_handler
    def workerInitProgressBar(self, maximum):
        self.progressBar.setValue(0)
        self.progressBar.setMaximum(maximum)
    
    @exception_handler
    def workerUpdateProgressBar(self):
        self.progressBar.update(1)
    
    @exception_handler
    def workerProgress(self, txt):
        self.logger.info(txt)
        self.logConsole.append(txt)
    
    @exception_handler
    def workerProgressBar(self, txt):
        self.logger.info(txt)
        self.logConsole.write(txt)
    
    @exception_handler
    def workerCritical(self, error):
        raise error
    
    @exception_handler
    def workerFinished(self):
        self.startButton.show()
        self.stopButton.hide()

        if self.worker.abort:
            msg = widgets_message_box.myMessageBox()
            msg.warning(
                self, 'Process stopped', 
                html_utils.paragraph('Data prep process stopped!')
            )
        else:
            msg = widgets_message_box.myMessageBox()
            msg.information(
                self, 'Process completed', 
                html_utils.paragraph('Data prep process completed!')
            )


class reapplyDataPrepWorker(QObject):
    finished = Signal()
    debug = Signal(object)
    critical = Signal(object)
    progress = Signal(str)
    initPbar = Signal(int)
    updatePbar = Signal()
    sigCriticalNoChannels = Signal(str)
    sigSelectChannels = Signal(object, object, object, str)

    def __init__(self, expPath, posFoldernames):
        super().__init__()
        self.expPath = expPath
        self.posFoldernames = posFoldernames
        self.abort = False
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
    
    def raiseSegmInfoNotFound(self, path):
        raise FileNotFoundError(
            'The following file is required for the alignment of 4D data '
            f'but it was not found: "{path}"'
        )
    
    def saveBkgrData(self, imageData, posData, isAligned=False):
        bkgrROI_data = {}
        for r, roi in enumerate(posData.bkgrROIs):
            xl, yt = [int(round(c)) for c in roi.pos()]
            w, h = [int(round(c)) for c in roi.size()]
            if not yt+h>yt or not xl+w>xl:
                # Prevent 0 height or 0 width roi
                continue
            is4D = posData.SizeT > 1 and posData.SizeZ > 1
            is3Dz = posData.SizeT == 1 and posData.SizeZ > 1
            is3Dt = posData.SizeT > 1 and posData.SizeZ == 1
            is2D = posData.SizeT == 1 and posData.SizeZ == 1
            if is4D:
                bkgr_data = imageData[:, :, yt:yt+h, xl:xl+w]
            elif is3Dz or is3Dt:
                bkgr_data = imageData[:, yt:yt+h, xl:xl+w]
            elif is2D:
                bkgr_data = imageData[yt:yt+h, xl:xl+w]
            bkgrROI_data[f'roi{r}_data'] = bkgr_data

        if not bkgrROI_data:
            return

        if isAligned:
            bkgr_data_fn = f'{posData.filename}_aligned_bkgrRoiData.npz'
        else:
            bkgr_data_fn = f'{posData.filename}_bkgrRoiData.npz'
        bkgr_data_path = os.path.join(posData.images_path, bkgr_data_fn)
        self.progress.emit('Saving background data to:')
        self.progress.emit(bkgr_data_path)
        io.savez_compressed(bkgr_data_path, **bkgrROI_data)

    def run(self):
        ch_name_selector = prompts.select_channel_name(
            which_channel='segm', allow_abort=False
        )
        for p, pos in enumerate(self.posFoldernames):
            if self.abort:
                break
            
            self.progress.emit(f'Processing {pos}...')
                
            posPath = os.path.join(self.expPath, pos)
            imagesPath = os.path.join(posPath, 'Images')

            ls = filesystem.listdir(imagesPath)
            if p == 0:
                ch_names, basenameNotFound = (
                    ch_name_selector.get_available_channels(ls, imagesPath)
                )
                if not ch_names:
                    self.sigCriticalNoChannels.emit(imagesPath)
                    break
                self.mutex.lock()
                if len(self.posFoldernames) == 1:
                    # User selected only one pos --> allow selecting and adding
                    # and external .tif file that will be renamed with the basename
                    basename = ch_name_selector.basename
                else:
                    basename = None
                self.sigSelectChannels.emit(
                    ch_name_selector, ch_names, imagesPath, basename
                )
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()
                if self.abort:
                    break
            
                self.progress.emit(
                    f'Selected channels: {self.selectedChannels}'
                )
            
            for chName in self.selectedChannels:
                filePath = load.get_filename_from_channel(imagesPath, chName)
                posData = load.loadData(filePath, chName)
                posData.getBasenameAndChNames()
                posData.buildPaths()
                posData.loadImgData()
                posData.loadOtherFiles(
                    load_segm_data=False, 
                    getTifPath=True,
                    load_metadata=True,
                    load_shifts=True,
                    load_dataPrep_ROIcoords=True,
                    loadBkgrROIs=True
                )

                imageData = posData.img_data

                prepped = False
                isAligned = False
                # Align
                if posData.loaded_shifts is not None:
                    self.progress.emit('Aligning frames...')
                    shifts = posData.loaded_shifts
                    if imageData.ndim == 4:
                        align_func = core.align_frames_3D
                    else:
                        align_func = core.align_frames_2D 
                    imageData, _ = align_func(imageData, user_shifts=shifts)
                    prepped = True
                    isAligned = True
                
                # Crop and save background
                if posData.dataPrep_ROIcoords is not None:
                    df = posData.dataPrep_ROIcoords
                    isCropped = int(df.at['cropped', 'value']) == 1
                    if isCropped:
                        self.saveBkgrData(imageData, posData, isAligned)
                        self.progress.emit('Cropping...')
                        x0 = int(df.at['x_left', 'value']) 
                        y0 = int(df.at['y_top', 'value']) 
                        x1 = int(df.at['x_right', 'value']) 
                        y1 = int(df.at['y_bottom', 'value']) 
                        if imageData.ndim == 4:
                            imageData = imageData[:, :, y0:y1, x0:x1]
                        elif imageData.ndim == 3:
                            imageData = imageData[:, y0:y1, x0:x1]
                        elif imageData.ndim == 2:
                            imageData = imageData[y0:y1, x0:x1]
                        prepped = True
                    else:
                        filename = os.path.basename(posData.dataPrepBkgrROis_path)
                        self.progress.emit(
                            f'WARNING: the file "{filename}" was not found. '
                            'I cannot crop the data.'
                        )
                    
                if prepped:              
                    self.progress.emit('Saving prepped data...')
                    io.savez_compressed(posData.align_npz_path, imageData)
                    if hasattr(posData, 'tif_path'):
                        image_utils.to_tiff(
                            posData.tif_path, imageData
                        )

            self.updatePbar.emit()
            if self.abort:
                break
        self.finished.emit()
