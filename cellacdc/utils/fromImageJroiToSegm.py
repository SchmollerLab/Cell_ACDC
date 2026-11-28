from .. import myutils, workers, widgets, html_utils
from .. import apps

from .base import NewThreadMultipleExpBaseUtil
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QLineEdit, QGridLayout, QGroupBox
from cellacdc.shared.widgets import numeric_inputs as widgets_numeric_inputs

class fromImageJRoiToSegmUtil(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, parent=None):
        module = myutils.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.qparent = parent
        self.expPaths = expPaths
    
    def runWorker(self):
        self.worker = workers.FromImajeJroiToSegmNpzWorker(self)
        self.worker.sigSelectRoisProps.connect(self.selectRoisProps)
        super().runWorker(self.worker)
    
    def selectRoisProps(self, roi_filepath, TZYX_shape, is_multi_pos):
        win = apps.ImageJRoisToSegmManager(
            roi_filepath, TZYX_shape, 
            addUseSamePropsForNextPosButton=is_multi_pos,
            parent=self.qparent
        )
        win.exec_()
        self.worker.abort = win.cancel
        if win.cancel:
            self.worker.waitCond.wakeAll()
            return
        
        self.worker.IDsToRoisMapper = win.IDsToRoisMapper
        self.worker.rescaleRoisSizes = win.rescaleSizes
        self.worker.repeatRoisZslicesRange = win.repeatRoisZslicesRange
        self.worker.useSamePropsForNextPos = win.useSamePropsForNextPos
        self.worker.areAllRoisSelected = win.areAllRoisSelected
        self.worker.waitCond.wakeAll()
    
    def showEvent(self, event):
        self.runWorker()
    
    def workerFinished(self, worker):
        super().workerFinished(worker)
        txt = 'Converting from ImageJ ROIs to Cell-ACDC segmentation file(s) completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.information(self, 'Process completed', html_utils.paragraph(txt))
        self.close()


class RescaleImageJroisGroupbox(QGroupBox):
    def __init__(self, TZYX_out_shape, parent=None):
        super().__init__(parent)
        
        self.setTitle('Rescale ROIs')
        self.setCheckable(True)
        
        gridLayout = QGridLayout()
        
        dims = ('Z', 'Y', 'X')
        self.widgets = {}
        for row, SizeD in enumerate(TZYX_out_shape[1:]):
            if SizeD == 1:
                continue
            
            dim = dims[row]
            inputSpinbox = widgets_numeric_inputs.SpinBox()
            inputSpinbox.setMinimum(1)
            inputSpinbox.setValue(SizeD)
            
            outZwidget = QLineEdit()
            outZwidget.setReadOnly(True)
            outZwidget.setAlignment(Qt.AlignCenter)
            # outZwidget.setValue(SizeD)
            outZwidget.setText(str(SizeD))

            row0 = row*2
            row1 = row0+1
            gridLayout.addWidget(QLabel(f'{dim}-dimension: '), row1, 0)
            
            gridLayout.addWidget(QLabel('Input size'), row0, 1)
            gridLayout.addWidget(inputSpinbox, row1, 1)
            
            gridLayout.addWidget(QLabel('Output size'), row0, 2)
            gridLayout.addWidget(outZwidget, row1, 2)
            
            self.widgets[dim] = (inputSpinbox, SizeD)
        
        self.setLayout(gridLayout)
    
    def inputOutputSizes(self):
        if not self.isChecked():
            return
        
        sizes = {
            dim: (spinbox.value(), int(SizeD))
            for dim, (spinbox, SizeD) in self.widgets.items()
        }
        return sizes
