from .. import myutils, workers, widgets, html_utils

from .base import NewThreadMultipleExpBaseUtil
import os
import numpy as np
import pandas as pd
from cellacdc import load
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base
import skimage
import skimage.io
import skimage.measure

class toObjCoordsUtil(NewThreadMultipleExpBaseUtil):
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
        self.worker = workers.ToObjCoordsWorker(self)
        super().runWorker(self.worker)
    
    def showEvent(self, event):
        self.runWorker()
    
    def workerFinished(self, worker):
        super().workerFinished(worker)
        txt = 'Converting to object coordinates completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.information(self, 'Process completed', html_utils.paragraph(txt))
        self.close()


class ToObjCoordsWorker(utils_base.BaseWorkerUtil):
    def __init__(self, mainWin):
        super().__init__(mainWin)
        
    @workers_base.worker_exception_handler
    def run(self):
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
                file_path = [
                    os.path.join(images_path, f) for f in ls 
                    if f.endswith(f'{endFilenameSegm}.npz')
                ][0]
                
                posData = load.loadData(file_path, '')

                self.signals.sigUpdatePbarDesc.emit(f'Processing {posData.pos_path}')

                posData.getBasenameAndChNames()
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm
                )

                if posData.SizeT == 1:
                    posData.segm_data = posData.segm_data[np.newaxis]
                
                dfs = []
                n_frames = len(posData.segm_data)
                self.signals.initProgressBar.emit(n_frames)
                for frame_i, lab in enumerate(posData.segm_data):
                    df_coords_i = myutils.from_lab_to_obj_coords(lab)
                    dfs.append(df_coords_i)
                    self.signals.progressBar.emit(1)
                df_filepath = posData.segm_npz_path.replace('.npz', '.csv')
                df_filepath = df_filepath.replace('_segm', '_objects_coordinates')

                keys = list(range(len(posData.segm_data)))
                df = pd.concat(dfs, keys=keys, names=['frame_i'])
                
                self.signals.initProgressBar.emit(0)
                df.to_csv(df_filepath)

        self.signals.finished.emit(self)


def from_lab_to_obj_coords(lab):
    rp = skimage.measure.regionprops(lab)
    dfs = []
    keys = []
    for obj in rp:
        keys.append(obj.label)
        obj_coords = obj.coords
        ndim = obj_coords.shape[1]
        if ndim == 3:
            columns = ['z', 'y', 'x']
        else:
            columns = ['y', 'x']
        df_obj = pd.DataFrame(data=obj_coords, columns=columns)
        dfs.append(df_obj)
    df = pd.concat(dfs, keys = keys, names=['Cell_ID', 'idx']).droplevel('idx')
    return df
