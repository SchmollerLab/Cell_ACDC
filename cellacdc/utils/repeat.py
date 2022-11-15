import os
import sys
import traceback

from PyQt5.QtCore import Qt, QThread, QSize
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog
)
from PyQt5 import QtGui

from .. import myutils, html_utils, workers, widgets, load, apps

class repeatDataPrepWindow(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        name = 'repeat data prep'
        
        logger, logs_path, log_path, log_filename = myutils.setupLogger(
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
            'Select which <b>channels</b> to reapply data prep to',
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
        stopButton = widgets.stopPushButton('Stop process')
        startButton = widgets.playPushButton('    Start     ')
        cancelButton = widgets.cancelPushButton('Close')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(startButton)
        buttonsLayout.addWidget(stopButton)

        self.startButton = startButton
        self.stopButton = stopButton

        progressBarLayout = QHBoxLayout()
        self.progressBar = widgets.QProgressBarWithETA(parent=self)
        progressBarLayout.addWidget(self.progressBar)
        progressBarLayout.addWidget(self.progressBar.ETA_label)
        self.logConsole = widgets.QLogConsole(parent=self)

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

    @myutils.exception_handler    
    def start(self):
        self.startButton.hide()
        self.stopButton.show()

        MostRecentPath = myutils.getMostRecentPath()
        exp_path = QFileDialog.getExistingDirectory(
            self, 'Select experiment folder or specific Position folder', 
            MostRecentPath
        )
        if not exp_path:
            self.logger.info('No path selected. Process stopped.')
            self.stop()
            return
        
        myutils.addToRecentPaths(exp_path, logger=self.logger)
        
        is_pos_folder = os.path.basename(exp_path).find('Position_') != -1
        is_images_folder = os.path.basename(exp_path) == 'Images'
        contains_images_folder = os.path.exists(
            os.path.join(exp_path, 'Images')
        )
        if contains_images_folder and not is_pos_folder:
            is_images_folder = True
            exp_path = os.path.join(exp_path, 'Images')

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
                    self, values, allow_abort=False, toggleMulti=True,
                    CbLabel="Select Position folder(s) to process:"
                )
                if select_folder.was_aborted:
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
        self.worker = workers.reapplyDataPrepWorker(exp_path, posFoldernames)

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
    
    def selectChannels(self, ch_name_selector, ch_names):
        win = apps.QDialogListbox(
            'Select channel',
            'Select channel names to process:\n',
            ch_names, multiSelection=True, parent=self
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True
        self.worker.selectedChannels = win.selectedItemsText
        self.worker.waitCond.wakeAll()
    
    def criticalNotValidFolder(self, path: os.PathLike):
        txt = html_utils.paragraph(
            'The selected folder:<br><br>'
            f'<code>{path}</code><br><br>'
            'is <b>not a valid folder</b>. '
            'Select a folder that contains the Position_n folders'
        )
        msg = widgets.myMessageBox()
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
        msg = widgets.myMessageBox()
        msg.addShowInFileManagerButton(images_path)
        msg.critical(self, err_title, err_msg)
        self.logger.info(err_title)
        self.stop()
    
    def stop(self):
        self.startButton.show()
        self.stopButton.hide()

        if hasattr(self, 'worker'):
            self.worker.abort = True
    
    @myutils.exception_handler
    def workerInitProgressBar(self, maximum):
        self.progressBar.setValue(0)
        self.progressBar.setMaximum(maximum)
    
    @myutils.exception_handler
    def workerUpdateProgressBar(self):
        self.progressBar.update(1)
    
    @myutils.exception_handler
    def workerProgress(self, txt):
        self.logger.info(txt)
        self.logConsole.append(txt)
    
    @myutils.exception_handler
    def workerProgressBar(self, txt):
        self.logger.info(txt)
        self.logConsole.write(txt)
    
    @myutils.exception_handler
    def workerCritical(self, error):
        raise error
    
    @myutils.exception_handler
    def workerFinished(self):
        self.startButton.show()
        self.stopButton.hide()

        if self.worker.abort:
            msg = widgets.myMessageBox()
            msg.warning(
                self, 'Process stopped', 
                html_utils.paragraph('Data prep process stopped!')
            )
        else:
            msg = widgets.myMessageBox()
            msg.information(
                self, 'Process completed', 
                html_utils.paragraph('Data prep process completed!')
            )