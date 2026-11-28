"""Launcher workers: MigrateUserProfileWorker moves the user profile folder
to a new location (the launcher action in launcher/main_window.py
imports this module inside the function that starts it).
"""
import os
from qtpy.QtCore import Signal, QObject
from cellacdc import load
from cellacdc.path import copy_or_move_tree
from cellacdc.shared.workers import base as workers_base


class MigrateUserProfileWorker(QObject):
    finished = Signal(object)
    critical = Signal(object)
    progress = Signal(str)
    debug = Signal(object)

    def __init__(self, src_path, dst_path, acdc_folders):
        QObject.__init__(self)
        self.signals = workers_base.signals()
        self.src_path = src_path
        self.dst_path = dst_path
        self.acdc_folders = acdc_folders
    
    @workers_base.worker_exception_handler
    def run(self):
        import shutil
        from cellacdc import models_path

        self.progress.emit(
            'Migrating user profile data from '
            f'"{self.src_path}" to "{self.dst_path}"...'
        )
        acdc_folders = self.acdc_folders
        self.signals.initProgressBar.emit(2*len(acdc_folders))
        dst_folder = os.path.basename(self.dst_path)
        folders_to_remove = []
        for acdc_folder in acdc_folders:
            if acdc_folder == dst_folder:
                # Skip the destination folder that would be picked up if the 
                # user called it with acdc at the start of the name
                self.signals.progressBar.emit(2)
                continue
            src = os.path.join(self.src_path, acdc_folder)
            dst = os.path.join(self.dst_path, acdc_folder)
            self.progress.emit(f'Copying {src} to {dst}...')
            files_failed_move = copy_or_move_tree(
                src, dst, copy=False,
                sigInitPbar=self.signals.sigInitInnerPbar, 
                sigUpdatePbar=self.signals.sigUpdateInnerPbar
            )
            folders_to_remove.append(src)
            self.signals.progressBar.emit(1)
        
        for to_remove in folders_to_remove:
            try:
                self.progress.emit(f'Removing "{to_remove}"...')
                shutil.rmtree(to_remove)
            except Exception as err:
                self.progress.emit(
                    '--------------------------------------------------------\n'
                    f'[WARNING]: Removal of the folder "{to_remove}" failed. '
                    'Please remove manually.\n'
                    '--------------------------------------------------------'
                )
            finally:
                self.signals.progressBar.emit(1)
        
        # Update model's paths
        load.migrate_models_paths(self.dst_path)        
        
        # Store user profile data folder path
        from cellacdc import user_profile_path_txt
        os.makedirs(os.path.dirname(user_profile_path_txt), exist_ok=True)
        with open(user_profile_path_txt, 'w') as txt:
            txt.write(self.dst_path)
        
        self.finished.emit(self)
