"""Data structure workers: CreateSymLinkToPosWinWorker creates the
symlinked Positions of the CreateSymLinkToPosWin window in
data_structure/data_struct.py.

These workers use worker_exception_handler of shared/workers/base.py,
which emits (worker, error) on the critical signal of the worker and of
its signals object. data_struct.py defines a different
worker_exception_handler for its own bioFormatsWorker, which emits only
the error on the worker's critical signal: never mix the two.
"""
import os
from qtpy.QtCore import QObject
from cellacdc import load
from cellacdc.shared.workers import base as workers_base


class CreateSymLinkToPosWinWorker(QObject):
    def __init__(
            self, 
            exp_pos_mapper, 
            dst_folderpath, 
            exp_segm_files_to_copy_mapper
        ):
        QObject.__init__(self)
        self.signals = workers_base.signals()
        self.logger = workers_base.workerLogger(self.signals.progress)
        
        self._exp_pos_mapper = exp_pos_mapper
        self._dst_folderpath = dst_folderpath
        self._exp_segm_files_to_copy_mapper = exp_segm_files_to_copy_mapper
    
    def log_current_pos(self, src_pos_path, dst_pos_path):
        sep = '-'*100
        text = (
            f'{sep}\nLinking following positions:\n\n'
            f'{src_pos_path}\n'
            f'--> {dst_pos_path}\n'
        )
        self.logger.log(text)
    
    @workers_base.worker_exception_handler
    def run(self):
        for exp_path, pos_foldernames in self._exp_pos_mapper.items():
            exp_foldername = os.path.basename(exp_path)
            dst_exp_path = os.path.join(self._dst_folderpath, exp_foldername)
            if os.path.exists(dst_exp_path):
                raise FileExistsError(
                    'The destination folder path already exists. '
                    'Please, select a destination folder where '
                    'the experiment folders do not exist. '
                    f'Existing folder: "{dst_exp_path}"'
                )
            
            segm_endanmes_to_copy = self._exp_segm_files_to_copy_mapper.get(
                exp_path, []
            )
            for pos in pos_foldernames:
                src_pos_path = os.path.join(exp_path, pos)
                dst_pos_path = os.path.join(dst_exp_path, pos)
                self.log_current_pos(src_pos_path, dst_pos_path)
                load.create_symlinked_pos_folder(
                    src_pos_path, 
                    dst_pos_path, 
                    segm_endanmes_to_copy
                )
                self.signals.progressBar.emit(1)
                
        self.signals.finished.emit(self)
