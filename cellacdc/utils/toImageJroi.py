from .. import myutils, workers, widgets, html_utils

from .base import NewThreadMultipleExpBaseUtil
import os
from cellacdc import load
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base
import skimage
import skimage.io
import skimage.measure
from cellacdc import core

class toImageRoiUtil(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, parent=None):
        module = myutils.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.expPaths = expPaths
    
    def runWorker(self):
        self.worker = workers.ToImajeJroiWorker(self)
        super().runWorker(self.worker)
    
    def showEvent(self, event):
        self.runWorker()
    
    def workerFinished(self, worker):
        super().workerFinished(worker)
        txt = 'Converting to ImageJ ROIs completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.information(self, 'Process completed', html_utils.paragraph(txt))
        self.close()


class ToImajeJroiWorker(utils_base.BaseWorkerUtil):
    def __init__(self, mainWin):
        super().__init__(mainWin)
        
    @workers_base.worker_exception_handler
    def run(self):
        from roifile import ImagejRoi, roiwrite

        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.signals.finished.emit(self)
                return
            
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.signals.finished.emit(self)
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')
                endFilenameSegm = self.mainWin.endFilenameSegm
                ls = myutils.listdir(images_path)
                
                files_path = [
                    os.path.join(images_path, f) for f in ls 
                    if f.endswith(f'{endFilenameSegm}.npz')
                ]
                
                if not files_path:
                    self.logger.log(
                        '[WARNING]: The following Position folder does not '
                        f'contain any file ending with {endFilenameSegm}. '
                        f'Skipping it. "{os.path.join(exp_path, pos)}")'
                    )
                    continue
                
                file_path = files_path[0]
                
                posData = load.loadData(file_path, '')

                self.signals.sigUpdatePbarDesc.emit(f'Processing {posData.pos_path}')

                posData.getBasenameAndChNames()
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm
                )
    
                if posData.SizeT > 1:
                    rois = []
                    max_ID = posData.segm_data.max()
                    for t, lab in enumerate(posData.segm_data):
                        rois_t = myutils.from_lab_to_imagej_rois(
                            lab, ImagejRoi, t=t, SizeT=posData.SizeT, 
                            max_ID=max_ID
                        )
                        rois.extend(rois_t)
                else:
                    rois = myutils.from_lab_to_imagej_rois(
                        posData.segm_data, ImagejRoi
                    )

                roi_filepath = posData.segm_npz_path.replace('.npz', '.zip')
                roi_filepath = roi_filepath.replace('_segm', '_imagej_rois')

                try:
                    os.remove(roi_filepath)
                except Exception as e:
                    pass

                roiwrite(roi_filepath, rois)
                        
        self.signals.finished.emit(self)


def lab2d_to_rois(ImagejRoi, lab2D, ndigits, t=None, z=None):
    rp = skimage.measure.regionprops(lab2D)
    rois = []
    for obj in rp:
        cont = core.get_obj_contours(obj=obj)
        yc, xc = obj.centroid
        x_str = str((int(xc))).zfill(ndigits)
        y_str = str((int(yc))).zfill(ndigits)
        name = f'{x_str}-{y_str}'
        if z is not None:
            z_str = str(z).zfill(ndigits)
            name = f'{z_str}-{name}'
        
        if t is not None:
            t_str = str(t).zfill(ndigits)
            name = f'{t_str}-{name}'
        
        name = f'id={obj.label}-{name}'
        
        roi = ImagejRoi.frompoints(
            cont, name=name, t=t, z=z, index=obj.label
        )
        rois.append(roi)
    return rois

def from_lab_to_imagej_rois(lab, ImagejRoi, t=0, SizeT=1, max_ID=None):
    if max_ID is None:
        max_ID = lab.max()
    
    if SizeT == 1:
        t = None
    
    SizeY, SizeX = lab.shape[-2:]
    ndigitsT = len(str(SizeT))
    ndigitsY = len(str(SizeY))
    ndigitsX = len(str(SizeX))
    
    if lab.ndim == 3:
        rois = []
        SizeZ = len(lab)
        ndigitsZ = len(str(SizeZ))
        ndigits = max(ndigitsT, ndigitsZ, ndigitsY, ndigitsX)
        for z, lab2D in enumerate(lab):
            z_rois = lab2d_to_rois(ImagejRoi, lab2D, ndigits, t=t, z=z)
        rois.extend(z_rois)
    else:
        ndigits = max(ndigitsT, ndigitsY, ndigitsX)
        rois = lab2d_to_rois(ImagejRoi, lab, ndigits, t=t)
    return rois
