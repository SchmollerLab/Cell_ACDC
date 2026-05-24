"""Decorators shared by guiWin and mixins."""

from __future__ import annotations

import os
import traceback
from functools import wraps

from qtpy.QtCore import QTimer

from . import html_utils, widgets


def get_data_exception_handler(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        try:
            if func.__code__.co_argcount == 1 and func.__defaults__ is None:
                result = func(self)
            elif func.__code__.co_argcount > 1 and func.__defaults__ is None:
                result = func(self, *args)
            else:
                result = func(self, *args, **kwargs)
        except Exception as e:
            try:
                if self.progressWin is not None:
                    self.progressWin.workerFinished = True
                    self.progressWin.close()
                    self.progressWin = None
            except AttributeError:
                pass
            result = None
            posData = self.data[self.pos_i]
            acdc_df_filename = os.path.basename(posData.acdc_output_csv_path)
            segm_filename = os.path.basename(posData.segm_npz_path)
            traceback_str = traceback.format_exc()
            self.logger.exception(traceback_str)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            msg.addShowInFileManagerButton(self.logs_path, txt='Show log file...')
            msg.setDetailedText(traceback_str)
            err_msg = html_utils.paragraph(f"""
                Error in function <code>{func.__name__}</code>.<br><br>
                One possbile explanation is that either the
                <code>{acdc_df_filename}</code> file<br>
                or the segmentation file <code>{segm_filename}</code><br>
                <b>are being synchronized by a cloud service (e.g., Google Drive 
                or OneDrive) or they are corrupted/damaged</b>.<br><br>
                <b>Try moving these files</b> (one by one) outside of the
                <code>{os.path.dirname(posData.relPath)}</code> folder
                <br>and reloading the data.<br><br>
                More details below or in the terminal/console.<br><br>
                Note that the <b>error details</b> from this session are
                also <b>saved in the following file</b>:<br><br>
                {self.log_path}<br><br>
                Please <b>send the log file</b> when reporting a bug, thanks!
                <b>Please restart Cell-ACDC, we apologise for any inconvenience.</b><br><br>

            """)

            msg.critical(self, 'Critical error', err_msg)
            self.is_error_state = True
            raise e
        return result
    return inner_function


def resetViewRange(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        self.storeViewRange()
        if func.__code__.co_argcount == 1 and func.__defaults__ is None:
            result = func(self)
        elif func.__code__.co_argcount > 1 and func.__defaults__ is None:
            result = func(self, *args)
        else:
            result = func(self, *args, **kwargs)
        QTimer.singleShot(200, self.resetRange)
        return result
    return inner_function
