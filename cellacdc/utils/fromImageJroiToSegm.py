from .. import myutils, workers, widgets, html_utils
from .. import apps

from .base import NewThreadMultipleExpBaseUtil
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QLineEdit, QGridLayout, QGroupBox
from cellacdc.shared.widgets import numeric_inputs as widgets_numeric_inputs
from qtpy.QtWidgets import QVBoxLayout, QAbstractItemView, QHeaderView
from cellacdc.shared.dialogs import base as dialogs_base

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


class ImageJRoisToSegmManager(dialogs_base.QBaseDialog):
    def __init__(
            self, rois_filepath, TZYX_shape, 
            addUseSamePropsForNextPosButton=False, parent=None
        ):
        import roifile
        
        self.cancel = True
        super().__init__(parent)
        
        self.setWindowTitle('ROI Manager')
        
        mainLayout = QVBoxLayout()
        
        rois = roifile.roiread(rois_filepath)
        if not isinstance(rois, list):
            rois = [rois]
        self.rois = {roi.name: roi for roi in rois}
        
        roisNamesTreeWidget = widgets.TreeWidget()
        roisNamesTreeWidget.setHeaderLabels(['ROI name', 'Cell_ID'])
        roisNamesTreeWidget.header().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        # roisNamesTreeWidget.header().setStretchLastSection(False)
        for r, roi in enumerate(rois):
            item = widgets.TreeWidgetItem()
            item.setText(0, roi.name)
            item.setText(1, str(r+1))
            roisNamesTreeWidget.addTopLevelItem(item)
        roisNamesTreeWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        roisNamesTreeWidget.selectAll()
        mainLayout.addWidget(QLabel('Select ROIs to convert'))
        mainLayout.addWidget(roisNamesTreeWidget)
        self.roisNamesTreeWidget = roisNamesTreeWidget
        mainLayout.addSpacing(10)
        mainLayout.addWidget(widgets.QHLine())
        mainLayout.addSpacing(5)
        
        self.repeatRoiZstackGroupbox = None
        
        SizeT, SizeZ, SizeY, SizeX = TZYX_shape
        if SizeZ > 1:
            self.repeatRoiZstackGroupbox = QGroupBox()
            repeatRoiZstackLayout = QGridLayout()
            self.repeatRoiZstackGroupbox.setCheckable(True)
            self.repeatRoiZstackGroupbox.setTitle(
                'ROI is 2D'
            )
            self.repeatRoiZstackGroupbox.setChecked(False)
            self.repeatRoiZstackGroupbox.setLayout(repeatRoiZstackLayout)
            self.lowZspinbox = widgets.SpinBox()
            self.lowZspinbox.setMinimum(0)
            self.lowZspinbox.setMaximum(SizeZ-1)
            
            self.highZspinbox = widgets.SpinBox()
            self.highZspinbox.setMinimum(0)
            self.highZspinbox.setMaximum(SizeZ-1)
            self.highZspinbox.setValue(SizeZ-1)
            
            repeatRoiZstackLayout.addWidget(
                QLabel('Repeat 2D ROIs over z-range: '), 1, 0)
            
            repeatRoiZstackLayout.addWidget(QLabel('Start z-slice'), 0, 1)
            repeatRoiZstackLayout.addWidget(self.lowZspinbox, 1, 1)
            
            repeatRoiZstackLayout.addWidget(QLabel('Stop z-slice'), 0, 2)
            repeatRoiZstackLayout.addWidget(self.highZspinbox, 1, 2)
        
        if self.repeatRoiZstackGroupbox is not None:
            mainLayout.addWidget(self.repeatRoiZstackGroupbox)
            mainLayout.addSpacing(5)
            mainLayout.addWidget(widgets.QHLine())
            mainLayout.addSpacing(10)
        
        self.rescaleRoisGroupbox = widgets.RescaleImageJroisGroupbox(TZYX_shape)
        self.rescaleRoisGroupbox.setChecked(False)
        mainLayout.addWidget(self.rescaleRoisGroupbox)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()
        
        self.useSamePropsForNextPos = False
        if addUseSamePropsForNextPosButton:
            useSamePropsForNextPosButton = widgets.reloadPushButton(
                'Keep the same preferences for all next Positions'
            )
            buttonsLayout.insertWidget(3, useSamePropsForNextPosButton)
            useSamePropsForNextPosButton.clicked.connect(
                self.useSamePropsForNextPosClicked
            )

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def useSamePropsForNextPosClicked(self):
        self.useSamePropsForNextPos = True
        self.ok_cb()
    
    def warnRoiSelectionEmpty(self):
        txt = html_utils.paragraph(f"""
            You did not select any ROI.<br><br>
            <b>ROIs selection cannot be empty</b>. Thank you for your patience! 
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'ROIs selection empty', txt)
    
    def ok_cb(self):
        selectedRois = self.roisNamesTreeWidget.selectedItems()
        if not selectedRois:
            self.useSamePropsForNextPos = False
            self.warnRoiSelectionEmpty()
            return
        
        self.IDsToRoisMapper = {}
        for item in selectedRois:
            roiName = item.text(0)
            ID = int(item.text(1))
            self.IDsToRoisMapper[ID] = self.rois[roiName]
        
        numRois = self.roisNamesTreeWidget.topLevelItemCount()
        self.areAllRoisSelected = len(self.IDsToRoisMapper) == numRois
        
        self.rescaleSizes = self.rescaleRoisGroupbox.inputOutputSizes()
        self.repeatRoisZslicesRange = None
        if self.repeatRoiZstackGroupbox is not None:
            if self.repeatRoiZstackGroupbox.isChecked():
                self.repeatRoisZslicesRange = (
                    self.lowZspinbox.value(), 
                    self.highZspinbox.value()+1
                )
        
        self.cancel = False
        self.close()
