from .. import html_utils
from cellacdc.myutils import filesystem
from cellacdc.shared.widgets import (
    basic_widgets as widgets_basic_widgets,
    item_views as widgets_item_views,
    message_box as widgets_message_box,
    push_buttons as widgets_push_buttons,
)

from .base import NewThreadMultipleExpBaseUtil
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QLineEdit, QGridLayout, QGroupBox
from cellacdc.shared.widgets import numeric_inputs as widgets_numeric_inputs
from qtpy.QtWidgets import QVBoxLayout, QAbstractItemView, QHeaderView
from cellacdc.shared.dialogs import base as dialogs_base
import os
from qtpy.QtCore import Signal
from cellacdc import load, io
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base
import numpy as np
import skimage
import skimage.io
import skimage.measure

class fromImageJRoiToSegmUtil(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, parent=None):
        module = filesystem.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.qparent = parent
        self.expPaths = expPaths
    
    def runWorker(self):
        self.worker = FromImajeJroiToSegmNpzWorker(self)
        self.worker.sigSelectRoisProps.connect(self.selectRoisProps)
        super().runWorker(self.worker)
    
    def selectRoisProps(self, roi_filepath, TZYX_shape, is_multi_pos):
        win = ImageJRoisToSegmManager(
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
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
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
        
        roisNamesTreeWidget = widgets_item_views.TreeWidget()
        roisNamesTreeWidget.setHeaderLabels(['ROI name', 'Cell_ID'])
        roisNamesTreeWidget.header().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        # roisNamesTreeWidget.header().setStretchLastSection(False)
        for r, roi in enumerate(rois):
            item = widgets_item_views.TreeWidgetItem()
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
        mainLayout.addWidget(widgets_basic_widgets.QHLine())
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
            self.lowZspinbox = widgets_numeric_inputs.SpinBox()
            self.lowZspinbox.setMinimum(0)
            self.lowZspinbox.setMaximum(SizeZ-1)
            
            self.highZspinbox = widgets_numeric_inputs.SpinBox()
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
            mainLayout.addWidget(widgets_basic_widgets.QHLine())
            mainLayout.addSpacing(10)
        
        self.rescaleRoisGroupbox = RescaleImageJroisGroupbox(TZYX_shape)
        self.rescaleRoisGroupbox.setChecked(False)
        mainLayout.addWidget(self.rescaleRoisGroupbox)
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()
        
        self.useSamePropsForNextPos = False
        if addUseSamePropsForNextPosButton:
            useSamePropsForNextPosButton = widgets_push_buttons.reloadPushButton(
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
        msg = widgets_message_box.myMessageBox(wrapText=False)
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


class FromImajeJroiToSegmNpzWorker(utils_base.BaseWorkerUtil):
    sigSelectRoisProps = Signal(str, object, bool)
    
    def __init__(self, mainWin):
        super().__init__(mainWin)
    
    def emitSelectRoisProps(self, roi_filepath, TZYX_shape, is_multi_pos):
        self.mutex.lock()
        self.sigSelectRoisProps.emit(roi_filepath, TZYX_shape, is_multi_pos)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort
    
    @workers_base.worker_exception_handler
    def run(self):
        import roifile
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            cancel = self.emitSelectFilesWithText(
                exp_path, pos_foldernames, 'imagej_rois', ext=('.zip', '.roi')
            )
            if cancel:
                self.signals.finished.emit(self)
                return
            
            self.askRoiPreferences = True
            self.repeatRoisZslicesRange = None
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.signals.finished.emit(self)
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')
                endFilenameRoi = self.mainWin.endFilenameWithText
                ls = filesystem.listdir(images_path)
                rois_filepaths = [
                    os.path.join(images_path, f) for f in ls 
                    if (
                        f.endswith(f'{endFilenameRoi}.zip')
                        or f.endswith(f'{endFilenameRoi}.roi')
                    )
                ]
                
                if not rois_filepaths:
                    self.logger.log(
                        '[WARNING]: The following Position folder does not '
                        f'contain any file ending with {endFilenameRoi}. '
                        f'Skipping it. "{os.path.join(exp_path, pos)}")'
                    )
                    continue
                    
                rois_filepath = rois_filepaths[0]
                
                if self.askRoiPreferences:
                    is_multi_pos = len(pos_foldernames) > 1
                    self.logger.log('Loading image data to get image shape...')
                    TZYX_shape = load.get_tzyx_shape(images_path)
                    cancel = self.emitSelectRoisProps(
                        rois_filepath, TZYX_shape, is_multi_pos
                    )
                    if cancel:
                        self.signals.finished.emit(self)
                        return
                    
                    self.askRoiPreferences = not self.useSamePropsForNextPos
                elif self.areAllRoisSelected:
                    rois = roifile.roiread(rois_filepath)
                    if not isinstance(rois, list):
                        rois = [rois]
                    self.IDsToRoisMapper = {
                        i+1: roi for i, roi in enumerate(rois)
                    }
                else:
                    # Use same ID of previous position
                    rois = roifile.roiread(rois_filepath)
                    if not isinstance(rois, list):
                        rois = [rois]
                    self.IDsToRoisMapper = {
                        ID: rois[i]
                        for i, ID in enumerate(self.IDsToRoisMapper.keys())
                    }

                self.logger.log('Generating segm mask from ROIs...')
                segm_data = from_imagej_rois_to_segm_data(
                    TZYX_shape, self.IDsToRoisMapper, self.rescaleRoisSizes, 
                    self.repeatRoisZslicesRange
                ) 
                
                segm_filepath = (rois_filepath
                    .replace('imagej_rois', 'segm')
                    .replace('.zip', '.npz')
                    .replace('.roi', '.npz')
                )
                self.logger.log(f'Saving segm mask to "{segm_filepath}"...')
                io.savez_compressed(segm_filepath, segm_data)
        
        self.signals.finished.emit(self)


def from_imagej_rois_to_segm_data(
        TZYX_shape, 
        ID_to_roi_mapper, 
        rescale_rois_sizes, 
        repeat_2d_rois_zslices_range: tuple[int, int] | None
    ):
    SizeT, SizeZ, SizeY, SizeX = TZYX_shape
    segm_data = np.zeros(TZYX_shape, dtype=np.uint32)
    for ID, roi in ID_to_roi_mapper.items():
        name = roi.name
        is_3d_roi = (
            roi.z_position > 0 or
            roi.position > 0
        )
        zz = [0]
        if repeat_2d_rois_zslices_range is not None and SizeZ > 1:
            # 2D roi in 3D segm data --> place 2D roi on each z-slice
            zz = range(*repeat_2d_rois_zslices_range)
        elif SizeZ > 1:
            if roi.z_position > 0:
                z = roi.z_position - 1
            else:
                # Fallback to position if z_position not encoded
                z = roi.position - 1
            zz = [z]
        
        tt = [0]*len(zz)
        if SizeT > 1:
            tt = [roi.t_position - 1]*len(zz)
        
        coordinates = roi.coordinates()
        xx = coordinates[:, 0]
        yy = coordinates[:, 1]
        if rescale_rois_sizes is not None:        
            rescale_z = rescale_rois_sizes['Z']
            rescale_y = rescale_rois_sizes['Y']
            rescale_x = rescale_rois_sizes['X']
            
            factor_z = rescale_z[1]/rescale_z[0]
            factor_y = rescale_y[1]/rescale_y[0]
            factor_x = rescale_x[1]/rescale_x[0]
            
            xx = np.clip(np.round(xx * factor_x).astype(int), 0, SizeX-1)
            yy = np.clip(np.round(yy * factor_y).astype(int), 0, SizeY-1)

        for t, z in zip(tt, zz):
            if rescale_rois_sizes is not None:
                z = round(z*factor_z)
                z = z if z<SizeZ else SizeZ
                z = z if z>=0 else 0
            
            rr, cc = skimage.draw.polygon(yy, xx)
            segm_data[t, z, rr, cc] = ID
    
    return np.squeeze(segm_data)
