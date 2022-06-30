import os
import sys
import traceback

from PyQt5.QtCore import Qt, QThread, QSize
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel
)
from PyQt5 import QtGui

from .. import myutils, html_utils, workers, widgets

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
            'Press <b>start<b> button'
            '<b>Select experiment</b> folder or specific Position folder',
            'Select which <b>channels</b> to repeat data prep to',
            '<b>Wait</b> until process ends'
        ]

        txt = html_utils.paragraph(f"""
            This utility does bla bla
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

        self.thread = QThread()
        self.worker = workers.BLABLA

        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.progress.connect(self.workerProgress)
        self.worker.initPbar.connect(self.workerInitProgressBar)
        self.worker.updatePbar.connect(self.workerUpdateProgressBar)
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.workerFinished)

        self.thread.started.connect(self.worker.run)
        self.thread.start()
    
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
                self, 'Conversion process stopped', 
                html_utils.paragraph('Conversion process stopped!')
            )
        else:
            msg = widgets.myMessageBox()
            msg.information(
                self, 'Conversion completed', 
                html_utils.paragraph('Conversion process completed!')
            )