import sys
import os
import shutil
import re
import traceback
import time
import datetime
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from natsort import natsorted

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QVBoxLayout, QPushButton, QLabel, QStyleFactory,
    QWidget, QMessageBox, QPlainTextEdit
)
from PyQt5.QtCore import (
    Qt, QObject, pyqtSignal, QThread, QMutex, QWaitCondition
)
from PyQt5 import QtGui

import qrc_resources
import apps, myutils

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except:
        pass


class bioFormatsWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    criticalError = pyqtSignal(str, str, str)
    filesExisting = pyqtSignal(str)
    confirmMetadata = pyqtSignal(
        str, float, str, int, int, int, int,
        float, str, float, float, float,
        str, list, list
    )
    # aborted = pyqtSignal()

    def __init__(self, exp_path, rawFilenames, mutex, waitCond, rawDataStruct):
        QObject.__init__(self)
        self.exp_path = exp_path
        self.rawFilenames = rawFilenames
        self.mutex = mutex
        self.waitCond = waitCond
        self.askReplacePosFiles = True
        self.overWriteMetadata = False
        self.trustMetadataReader = False
        self.rawDataStruct = rawDataStruct

    def readMetadata(self, exp_path, filename):
        rawFilePath = os.path.join(exp_path, filename)

        self.progress.emit('Reading OME metadata...')

        try:
            metadataXML = bioformats.get_omexml_metadata(rawFilePath)
            metadata = bioformats.OMEXML(metadataXML)
            self.metadataXML = metadataXML
        except Exception as e:
            self.isCriticalError = True
            self.criticalError.emit(
                'reading image data or metadata',
                traceback.format_exc(), filename
            )
            return True

        try:
            LensNA = float(metadata.instrument().Objective.LensNA)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: LensNA not found in metadata.')
            self.progress.emit(
                '===================================================')
            LensNA = 1.4

        try:
            SizeS = int(metadata.get_image_count())
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: SizeS not found in metadata.')
            self.progress.emit(
                '===================================================')
            SizeS = 1

        try:
            DimensionOrder = metadata.image().Pixels.DimensionOrder
            if DimensionOrder is None:
                raise
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: DimensionOrder not found in metadata.')
            self.progress.emit(
                '===================================================')
            DimensionOrder = ''

        try:
            SizeZ = int(metadata.image().Pixels.SizeZ)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: SizeZ not found in metadata.')
            self.progress.emit(
                '===================================================')
            SizeZ = 1

        try:
            SizeT = int(metadata.image().Pixels.SizeT)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: SizeT not found in metadata.')
            self.progress.emit(
                '===================================================')
            SizeT = 1

        try:
            Pixels = metadata.image().Pixels
            TimeIncrement = float(Pixels.node.get('TimeIncrement'))
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: TimeIncrement not found in metadata.')
            self.progress.emit(
                '===================================================')
            TimeIncrement = 180.0

        try:
            Pixels = metadata.image().Pixels
            TimeIncrementUnit = Pixels.node.get('TimeIncrementUnit')
            if TimeIncrementUnit is None:
                raise
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: TimeIncrementUnit not found in metadata.')
            self.progress.emit(
                '===================================================')
            TimeIncrementUnit = 's'

        try:
            SizeC = int(metadata.image().Pixels.SizeC)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: SizeC not found in metadata.')
            self.progress.emit(
                '===================================================')
            SizeC = 1

        try:
            PhysicalSizeX = float(metadata.image().Pixels.PhysicalSizeX)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: PhysicalSizeX not found in metadata.')
            self.progress.emit(
                '===================================================')
            PhysicalSizeX = 1.0

        try:
            PhysicalSizeY = float(metadata.image().Pixels.PhysicalSizeY)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: PhysicalSizeY not found in metadata.')
            self.progress.emit(
                '===================================================')
            PhysicalSizeY = 1.0

        try:
            PhysicalSizeZ = float(metadata.image().Pixels.PhysicalSizeZ)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: PhysicalSizeZ not found in metadata.')
            self.progress.emit(
                '===================================================')
            PhysicalSizeZ = 1.0

        try:
            Pixels = metadata.image().Pixels
            PhysicalSizeUnit = Pixels.node.get('PhysicalSizeXUnit')
            if PhysicalSizeUnit is None:
                raise
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: PhysicalSizeUnit not found in metadata.')
            self.progress.emit(
                '===================================================')
            PhysicalSizeUnit = 'μm'

        try:
            chNames = ['name_not_found']*SizeC
            for c in range(SizeC):
                try:
                    chNames[c] = metadata.image().Pixels.Channel(c).Name
                except Exception as e:
                    pass
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: chNames not found in metadata.')
            self.progress.emit(
                '===================================================')
            chNames = ['not_found']*SizeC

        try:
            emWavelens = [500.0]*SizeC
            for c in range(SizeC):
                Channel = metadata.image().Pixels.Channel(c)
                emWavelen = Channel.node.get("EmissionWavelength")
                try:
                    emWavelens[c] = float(emWavelen)
                except Exception as e:
                    emWavelens[c] = 0.0
        except Exception as e:
            traceback.print_exc()
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: EmissionWavelength not found in metadata.')
            self.progress.emit(
                '===================================================')
            emWavelens = [500.0]*SizeC

        if self.trustMetadataReader:
            self.LensNA = LensNA
            self.DimensionOrder = DimensionOrder
            self.SizeT = SizeT
            self.SizeZ = SizeZ
            self.SizeC = SizeC
            self.SizeS = SizeS
            self.TimeIncrement = TimeIncrement
            self.PhysicalSizeX = PhysicalSizeX
            self.PhysicalSizeY = PhysicalSizeY
            self.PhysicalSizeZ = PhysicalSizeZ
            # self.chNames = chNames
            self.emWavelens = emWavelens
        else:
            self.mutex.lock()
            self.confirmMetadata.emit(
                filename, LensNA, DimensionOrder, SizeT, SizeZ, SizeC, SizeS,
                TimeIncrement, TimeIncrementUnit, PhysicalSizeX, PhysicalSizeY,
                PhysicalSizeZ, PhysicalSizeUnit, chNames, emWavelens
            )
            self.waitCond.wait(self.mutex)
            self.mutex.unlock()
            if self.metadataWin.cancel:
                return True
            elif self.metadataWin.overWrite:
                self.overWriteMetadata = True
            elif self.metadataWin.trust:
                self.trustMetadataReader = True

            self.LensNA = self.metadataWin.LensNA
            self.DimensionOrder = self.metadataWin.DimensionOrder
            self.SizeT = self.metadataWin.SizeT
            self.SizeZ = self.metadataWin.SizeZ
            self.SizeC = self.metadataWin.SizeC
            self.SizeS = self.metadataWin.SizeS
            self.TimeIncrement = self.metadataWin.TimeIncrement
            self.PhysicalSizeX = self.metadataWin.PhysicalSizeX
            self.PhysicalSizeY = self.metadataWin.PhysicalSizeY
            self.PhysicalSizeZ = self.metadataWin.PhysicalSizeZ
            self.chNames = self.metadataWin.chNames
            self.saveChannels = self.metadataWin.saveChannels
            self.emWavelens = self.metadataWin.emWavelens

    def saveToPosFolder(self, p, exp_path, filename, series):
        foldername = os.path.basename(exp_path)
        rawFilePath = os.path.join(exp_path, filename)

        if foldername == 'raw_microscopy_files':
            exp_path = os.path.dirname(exp_path)

        pos_path = os.path.join(exp_path, f'Position_{p+1}')
        images_path = os.path.join(pos_path, 'Images')

        if os.path.exists(images_path) and self.askReplacePosFiles:
            self.askReplacePosFiles = False
            self.mutex.lock()
            self.filesExisting.emit(pos_path)
            self.waitCond.wait(self.mutex)
            self.mutex.unlock()
            if self.cancel:
                return True

        if os.path.exists(images_path):
            shutil.rmtree(images_path)
        os.makedirs(images_path)

        self.saveData(images_path, rawFilePath, filename, p, series)

    def saveData(self, images_path, rawFilePath, filename, p, series):
        s0p = str(p+1).zfill(self.numPosDigits)
        self.progress.emit(
            f'Position {p+1}/{self.numPos}: saving data to {images_path}...'
        )
        filenameNOext, ext = os.path.splitext(filename)

        metadataXML_path = os.path.join(
            images_path, f'{filenameNOext}_s{s0p}_metadataXML.txt'
        )
        with open(metadataXML_path, 'w', encoding="utf-8") as txt:
            txt.write(self.metadataXML)

        metadata_csv_path = os.path.join(
            images_path, f'{filenameNOext}_s{s0p}_metadata.csv'
        )
        df = pd.DataFrame({
            'LensNA': self.LensNA,
            'DimensionOrder': self.DimensionOrder,
            'SizeT': self.SizeT,
            'SizeZ': self.SizeZ,
            'TimeIncrement': self.TimeIncrement,
            'PhysicalSizeZ': self.PhysicalSizeZ,
            'PhysicalSizeY': self.PhysicalSizeY,
            'PhysicalSizeX': self.PhysicalSizeX
        }, index=['values']).T
        df.index.name = 'Description'

        ch_metadata = self.chNames.copy()
        description = [f'channel_{c}_name' for c in range(self.SizeC)]
        ch_metadata.extend(self.emWavelens)
        description.extend([f'channel_{c}_emWavelen' for c in range(self.SizeC)])

        df_channelNames = pd.DataFrame({
            'Description': description,
            'values': ch_metadata
        }).set_index('Description')

        df = pd.concat([df, df_channelNames])

        df.to_csv(metadata_csv_path)

        with bioformats.ImageReader(rawFilePath) as reader:
            for c, (chName, saveCh) in enumerate(zip(self.chNames, self.saveChannels)):
                if not saveCh:
                    continue

                self.progress.emit(
                    f'  Saving channel {c+1}/{len(self.chNames)} ({chName})'
                )
                imgData_ch = []
                for t in range(self.SizeT):
                    imgData_z = []
                    for z in range(self.SizeZ):
                        imgData = reader.read(
                            c=c, z=z, t=t, series=series, rescale=False
                        )
                        imgData_z.append(imgData)
                    imgData_z = np.array(imgData_z, dtype=imgData.dtype)
                    imgData_ch.append(imgData_z)
                imgData_ch = np.array(imgData_ch, dtype=imgData.dtype)
                imgData_ch = np.squeeze(imgData_ch)
                tifPath = os.path.join(
                    images_path, f'{filenameNOext}_s{s0p}_{chName}.tif'
                )
                myutils.imagej_tiffwriter(
                    tifPath, imgData_ch, {}, self.SizeT, self.SizeZ
                )

    def run(self):
        exp_path = self.exp_path
        javabridge.start_vm(class_path=bioformats.JARS)
        self.progress.emit('Java VM already running.')
        self.aborted = False
        self.isCriticalError = False
        for p, filename in enumerate(self.rawFilenames):
            if self.rawDataStruct == 0:
                if not self.overWriteMetadata:
                    abort = self.readMetadata(exp_path, filename)
                    if abort:
                        self.aborted = True
                        break
                    for p in range(self.SizeS):
                        self.numPos = self.SizeS
                        self.numPosDigits = len(str(self.numPos))
                        abort = self.saveToPosFolder(p, exp_path, filename, p)
                        if abort:
                            self.aborted = True
                            break

            elif self.rawDataStruct == 1:
                if not self.overWriteMetadata:
                    abort = self.readMetadata(exp_path, filename)
                    if abort:
                        self.aborted = True
                        break
                self.numPos = len(self.rawFilenames)
                self.numPosDigits = len(str(self.numPos))
                abort = self.saveToPosFolder(p, exp_path, filename, 0)
                if abort:
                    self.aborted = True
                    break

            foldername = os.path.basename(self.exp_path)
            if foldername != 'raw_microscopy_files':
                rawFilePath = os.path.join(self.exp_path, filename)
                raw_path = os.path.join(exp_path, 'raw_microscopy_files')
                if not os.path.exists(raw_path):
                    os.mkdir(raw_path)
                dst = os.path.join(raw_path, filename)
                shutil.move(rawFilePath, dst)

        self.finished.emit()
        javabridge.kill_vm()

class createDataStructWin(QMainWindow):
    def __init__(self, parent=None, allowExit=False,
                 buttonToRestore=None, mainWin=None,
                 start_JVM=True):
        self.start_JVM = start_JVM
        self.allowExit = allowExit
        self.processFinished = False
        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin
        super().__init__(parent)
        self.setWindowTitle("Cell-ACDC - From raw microscopy file to tifs")
        self.setWindowIcon(QtGui.QIcon(":assign-motherbud.svg"))

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()

        label = QLabel(
            'Creating data structure from raw microscopy file(s)...')

        label.setStyleSheet("padding:5px 10px 10px 10px;")
        label.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        label.setFont(font)
        mainLayout.addWidget(label)

        informativeHtml = (
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
        <p style="font-size:11pt; line-height:1.2">
            This <b>wizard</b> will guide you through the <b>creation of the required
            data structure</b><br> starting from the raw microscopy file(s)
        </p>
        <p style="font-size:10pt; line-height:1.2">
            Follow the instructions in the pop-up windows.<br>
            Note that pop-ups might be minimized or behind other open windows.<br>
            Progess is displayed in the terminal/console.
        </p>
        <p style="font-size:10pt; line-height:1.2">
            Progress will be displayed below.
        </p>
        </blockquote>
        </body>
        </html>
        """
        )

        informativeText = QLabel(self)

        informativeText.setTextFormat(Qt.RichText)
        informativeText.setText(informativeHtml)
        informativeText.setStyleSheet("padding:5px 0px 10px 0px;")
        mainLayout.addWidget(informativeText)

        self.logWin = QPlainTextEdit()
        self.logWin.setReadOnly(True)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.logWin.setFont(font)
        mainLayout.addWidget(self.logWin)

        abortButton = QPushButton('Abort process')
        abortButton.clicked.connect(self.close)
        mainLayout.addWidget(abortButton)

        mainLayout.setContentsMargins(20, 0, 20, 20)
        mainContainer.setLayout(mainLayout)

        global bioformats, javabridge
        print('Checking if Java is installed...')
        try:
            import javabridge
            import bioformats
        except Exception as e:
            myutils.download_java()

        try:
            import javabridge
            import bioformats
        except Exception as e:
            print('===============================================================')
            print('Automatic download of Java failed. Please download the portable '
            'version of Java SE Runtime Environment and extract it into '
            '"/Cell_ACDC/src/java/<OS name folder>"')
            print('===============================================================')
            raise FileNotFoundError('Dowload of Java failed. See above for details.')

    def getMostRecentPath(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'temp', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            if 'opened_last_on' in df.columns:
                df = df.sort_values('opened_last_on', ascending=False)
            self.MostRecentPath = df.iloc[0]['path']
            if not isinstance(self.MostRecentPath, str):
                self.MostRecentPath = ''
        else:
            self.MostRecentPath = ''

    def addToRecentPaths(self, exp_path):
        if not os.path.exists(exp_path):
            return
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'temp', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            recentPaths = df['path'].to_list()
            if 'opened_last_on' in df.columns:
                openedOn = df['opened_last_on'].to_list()
            else:
                openedOn = [np.nan]*len(recentPaths)
            if exp_path in recentPaths:
                pop_idx = recentPaths.index(exp_path)
                recentPaths.pop(pop_idx)
                openedOn.pop(pop_idx)
            recentPaths.insert(0, exp_path)
            openedOn.insert(0, datetime.datetime.now())
            # Keep max 20 recent paths
            if len(recentPaths) > 20:
                recentPaths.pop(-1)
                openedOn.pop(-1)
        else:
            recentPaths = [exp_path]
            openedOn = [datetime.datetime.now()]
        df = pd.DataFrame({'path': recentPaths,
                           'opened_last_on': pd.Series(openedOn,
                                                       dtype='datetime64[ns]')})
        df.index.name = 'index'
        df.to_csv(recentPaths_path)

    def main(self):
        self.log('Asking how raw data is structured...')
        rawDataStruct, abort = self.askRawDataStruct()
        if abort:
            if self.allowExit:
                exit('Execution aborted by the user.')
            else:
                self.close()
                return

        self.rawDataStruct = rawDataStruct

        self.log('Instructing to move raw data...')
        proceed = self.instructMoveRawFiles()
        if not proceed:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        self.log(
            'Asking to select the folder that contains the microscopy files...'
        )
        self.getMostRecentPath()
        exp_path = QFileDialog.getExistingDirectory(
            self, 'Select folder containing the microscopy files', self.MostRecentPath)
        self.addToRecentPaths(exp_path)

        if exp_path == '':
            if self.allowExit:
                exit('Execution aborted by the user.')
            else:
                self.close()
                return

        self.log(
            'Checking file format of loaded files...'
        )
        ls = self.checkFileFormat(exp_path)
        if not ls:
            if self.allowExit:
                exit('Folder selected does not contain files.')
            else:
                self.close()
                return

        self.log(
            'Starting a Java Virtual Machine...'
        )

        # Set up separate thread for bioFormatsWorker class
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.thread = QThread()
        self.worker = bioFormatsWorker(
            exp_path, ls, self.mutex, self.waitCond, rawDataStruct
        )
        self.worker.moveToThread(self.thread)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.taskEnded)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.log)
        self.worker.criticalError.connect(self.criticalBioFormats)
        self.worker.confirmMetadata.connect(self.askConfirmMetadata)
        self.worker.filesExisting.connect(self.askReplacePosFilesFiles)
        self.thread.started.connect(self.worker.run)

        self.thread.start()

    def taskEnded(self):
        if self.worker.aborted and not self.worker.isCriticalError:
            msg = QMessageBox(self)
            abort = msg.critical(
               self, 'Conversion task aborted.',
               'Conversion task aborted.',
               msg.Close
            )
            self.close()
            if self.allowExit:
                exit('Conversion task ended.')
        elif not self.worker.aborted:
            msg = QMessageBox(self)
            abort = msg.information(
               self, 'Conversion task ended.',
               'Conversion task ended.\n\n'
               f'Files saved to "{self.worker.exp_path}"',
               msg.Close
            )
            self.close()
            if self.allowExit:
                exit('Conversion task ended.')

    def log(self, text):
        self.logWin.appendPlainText(text)

    def askRawDataStruct(self):
        win = apps.QDialogCombobox(
            'Raw data structure',
            ['Single microscopy file with one or more positions',
             'Multiple microscopy files, one for each position',
             'Multiple microscopy files, one for each channel',
             'NONE of the above'],
             '<p style="font-size:10pt">'
             'Select how you have your <b>raw_microscopy_files arranged</b>'
             '</p>',
             CbLabel='', parent=self
        )
        win.exec_()
        return win.selectedItemIdx, win.cancel

    def instructMoveRawFiles(self):
        msg = QMessageBox(self)
        msg.setWindowTitle('Move microscopy files')
        msg.setIcon(msg.Information)
        msg.setTextFormat(Qt.RichText)
        msg.setText(
        """
        Put all of the raw_microscopy_files from the <b>same experiment</b>
        into an <b>empty folder</b>.<br><br>

        Note that there should be no other files in this folder.
        """
        )
        doneButton = QPushButton('Done')
        cancelButton = QPushButton('Cancel')
        msg.addButton(doneButton, msg.YesRole)
        msg.addButton(cancelButton, msg.NoRole)
        msg.exec_()
        if msg.clickedButton() == doneButton:
            return True
        else:
            return False

    def checkFileFormat(self, exp_path):
        ls = natsorted(os.listdir(exp_path))
        files = [
            filename for filename in ls
            if os.path.isfile(os.path.join(exp_path, filename))
        ]
        all_ext = [
            os.path.splitext(filename)[1] for filename in ls
            if os.path.isfile(os.path.join(exp_path, filename))
        ]
        counter = Counter(all_ext)
        unique_ext = list(counter.keys())
        is_ext_unique = len(unique_ext) == 1
        most_common_ext, _ = counter.most_common(1)[0]
        if not is_ext_unique:
            msg = QMessageBox()
            proceedWithMostCommon = msg.warning(
                self, 'Multiple extensions detected',
                f'The folder {exp_path}\n'
                'contains files with different file extensions '
                f'(extensions detected: {unique_ext})\n\n'
                f'However, the most common extension is {most_common_ext}, '
                'do you want to proceed with\n'
                f'loading only files with extension {most_common_ext}?',
                msg.Yes | msg.Cancel
            )
            if proceedWithMostCommon == msg.Yes:
                files = [
                    filename for filename in files
                    if os.path.splitext(filename)[1] == most_common_ext
                ]
            else:
                return []

        if self.rawDataStruct == 0 and len(files) > 1:
            files = self.warnMultipleFiles(files)

        return files

    def warnMultipleFiles(self, files):
        win = apps.QDialogCombobox(
            'Multiple microscopy files detected!', files,
             '<p style="font-size:10pt">'
             'You selected "Single microscopy file", but the <b>folder contains multiple files</b>.<br>'
             '</p>',
             CbLabel='Select which file to load: ', parent=self,
             iconPixmap=QtGui.QPixmap(':warning.svg')
        )
        win.exec_()
        if win.cancel:
            return []
        else:
            files = [win.selectedItemText]

    def criticalBioFormats(self, actionTxt, tracebackFormat, filename):
        msg = QMessageBox(self)
        msg.setIcon(msg.Critical)
        msg.setWindowTitle('Critical error Bio-Formats')
        msg.setDefaultButton(msg.Ok)

        url = 'https://docs.openmicroscopy.org/bio-formats/6.7.0/supported-formats.html'
        seeHere = f'<a href=\"{url}">here</a>'

        _, ext = os.path.splitext(filename)
        txt = (
            f"""
            <p "font-size:10pt">
            Error while {actionTxt} with Bio-Formats.<br><br>

            This is most likely because the <b>file format {ext} is not fully supported</b>
            by the Bio-Formats library.<br><br>

            See {seeHere} for details about supported formats.<br><br>

            Try loading file in Fiji and create the data structure manually.<br><br>
            Alternatively, if you are trying to load a video file, you can try
            to open the main GUI and then go to "File --> Open image/video file..."<br><br>
            You were trying to read file: {filename}
            </p>
            """
        )
        msg.setTextFormat(Qt.RichText)
        msg.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg.setText(txt)
        msg.setDetailedText(tracebackFormat)
        msg.exec_()
        self.close()

    def askConfirmMetadata(
            self, filename, LensNA, DimensionOrder, SizeT, SizeZ, SizeC, SizeS,
            TimeIncrement, TimeIncrementUnit, PhysicalSizeX, PhysicalSizeY,
            PhysicalSizeZ, PhysicalSizeUnit, chNames, emWavelens
        ):
        self.metadataWin = apps.QDialogMetadataXML(
            title=f'Metadata for {filename}',
            LensNA=LensNA, DimensionOrder=DimensionOrder,
            SizeT=SizeT, SizeZ=SizeZ, SizeC=SizeC, SizeS=SizeS,
            TimeIncrement=TimeIncrement, TimeIncrementUnit=TimeIncrementUnit,
            PhysicalSizeX=PhysicalSizeX, PhysicalSizeY=PhysicalSizeY,
            PhysicalSizeZ=PhysicalSizeZ, PhysicalSizeUnit=PhysicalSizeUnit,
            chNames=chNames, emWavelens=emWavelens,
            parent=self, rawDataStruct=self.rawDataStruct
        )
        self.metadataWin.exec_()
        self.worker.metadataWin = self.metadataWin
        self.waitCond.wakeAll()

    def askReplacePosFilesFiles(self, pos_path):
        msg = QMessageBox(self)
        abort = msg.warning(
           self, 'Replace files?',
           f'The folder "{pos_path}" already exists.\n\n'
           'Do you want to replace it?',
           msg.YesToAll | msg.Cancel
        )
        if abort == msg.Cancel:
            self.worker.cancel = True
        else:
            self.worker.cancel = False
        self.waitCond.wakeAll()

    def doAbort(self):
        msg = QMessageBox(self)
        closeAnswer = msg.warning(
           self, 'Abort execution?', 'Do you really want to abort process?',
           msg.Yes | msg.No
        )
        if closeAnswer == msg.Yes:
            if self.allowExit:
                exit('Execution aborted by the user')
            else:
                print('Segmentation routine aborted by the user.')
                return True
        else:
            return False

    def closeEvent(self, event):
        if self.buttonToRestore is not None:
            button, color, text = self.buttonToRestore
            button.setText(text)
            button.setDisabled(True)
            button.setToolTip(
                'Button is disabled because due to an internal limitation '
                'of the Java Virtual Machine you cannot start another process.\n\n'
                'To launch another conversion process you need to RESTART Cell-ACDC'
            )
            button.setStyleSheet(
                f'QPushButton {{background-color: {color};}}')
            self.mainWin.setWindowState(Qt.WindowNoState)
            self.mainWin.setWindowState(Qt.WindowActive)
            self.mainWin.raise_()


if __name__ == "__main__":
    print('Launching segmentation script...')
    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # Create the application
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    win = createDataStructWin(allowExit=True)
    win.show()
    win.setWindowState(Qt.WindowActive)
    win.raise_()
    print('Done. If window asking to select a folder is not visible, it is '
          'behind some other open window.')
    win.main()
    sys.exit(app.exec_())
