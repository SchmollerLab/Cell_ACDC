"""Main GUI (guiWin-only) cell cycle annotation integrity checker:
CcaIntegrityCheckerWorker checks the cell cycle annotations in the
background while the user annotates.
"""
import time
from collections import deque
import pandas as pd
import traceback
from qtpy.QtCore import Signal, QObject
from cellacdc import html_utils
from cellacdc import printl
from cellacdc import core
from cellacdc import cca_df_colnames
from cellacdc.shared.workers import base as workers_base


class CcaIntegrityCheckerWorker(QObject):
    finished = Signal(object)
    critical = Signal(object)
    progress = Signal(str, object)
    sigDone = Signal()
    sigWarning = Signal(str, str)
    sigFixWillDivide = Signal(str, list)
    
    def __init__(self, mutex, waitCond):
        QObject.__init__(self)
        self.logger = workers_base.workerLogger(self.progress)
        self.mutex = mutex
        self.waitCond = waitCond
        self.exit = False
        self.isFinished = False
        self.abortChecking = False
        self.isChecking = False
        self.isPaused = False
        self.debug = False
        self.dataQ = deque(maxlen=10)
    
    def pause(self):
        if self.debug:
            self.logger.log('Cell cycle annotations checker is idle.')
        self.mutex.lock()
        # Re-check under the mutex. Without this, a wake-up sent by `_stop`
        # or `_enqueue` between the check in `run` and the `wait` below
        # would be lost and the worker would sleep forever
        if not self.exit and len(self.dataQ) == 0:
            self.isPaused = True
            self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        self.isPaused = False
    
    def enqueue(self, posData):
        # First stop previous checking
        if self.isChecking:
            self.abortChecking = True
        self._enqueue(posData)
    
    def _enqueue(self, posData):
        if self.exit:
            # Worker is stopping --> do not enqueue, otherwise
            # `abortChecking` would be reset and the current check would
            # run until the end
            return

        if self.debug:
            self.logger.log('Enqueing posData...')
        self.mutex.lock()
        self.dataQ.append(posData)
        if len(self.dataQ) == 1:
            # Wake worker upon inserting first element
            self.abortChecking = False
            self.waitCond.wakeAll()
        self.mutex.unlock()
    
    def clearQueue(self):
        self.dataQ.clear()
    
    def _stop(self):
        self.mutex.lock()
        self.exit = True
        self.waitCond.wakeAll()
        self.mutex.unlock()
    
    def abort(self):
        self.abortChecking = True
        self.mutex.lock()
        self.dataQ.clear()
        self.mutex.unlock()
        self._stop()
    
    def _check_equality_num_mothers_buds_in_S(self, checker, frame_i):
        num_moth_S, num_buds = checker.get_num_mothers_and_buds_in_S()
        
        if num_moth_S == num_buds:
            return True
        
        category = 'number of buds different from number of mothers in S phase'
        ul_items = [
            f'Number of buds = {num_buds}', 
            f'Number of mothers in S phase = {num_moth_S}'
        ]
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} the number of buds and number of '
            'mother cells in S phase are different!'
            f'{html_utils.to_list(ul_items)}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_mothers_multiple_buds(self, checker, frame_i):
        mother_IDs_with_multiple_buds = (
            checker.get_mother_IDs_with_multiple_buds()
        )
        if len(mother_IDs_with_multiple_buds) == 0:
            return True

        category = 'mother cells with multiple buds'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'the following mother cells have <b>multiple buds</b> assigned to it'
            f'<br><br>{mother_IDs_with_multiple_buds}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_cells_without_G1(self, checker, global_cca_df):
        IDs_cycles_without_G1 = (
            checker.get_IDs_cycles_without_G1(global_cca_df)
        )
        if len(IDs_cycles_without_G1) == 0:
            return True

        category = 'cell cycles without G1'
        txt = html_utils.paragraph(
            'Cell-ACDC requires that every cell cycle has at least '
            'one frame in G1.<br>'
            'The following pairs of <code>(ID, generation number)</code> '
            'do not satisfy this condition:<br><br>'
            f'{IDs_cycles_without_G1}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_will_divide_is_true(self, checker, global_cca_df):
        # NOTE: unfortunately this function performs pandas manipulations 
        # that are either not thread-safe or in any case are freezing the 
        # GUI. For now we don't run this until we find a solution
        return True
    
        IDs_will_divide_wrong = (
            checker.get_IDs_gen_num_will_divide_wrong(global_cca_df)
        )
        if len(IDs_will_divide_wrong) == 0:
            return True

        txt = html_utils.paragraph(
            'Cell-ACDC found that `will_divide` is annotated as True on the '
            'following <code>(ID, generation number)</code> cell<br>'
            'despite the fact that division is still not annotated on '
            'these cells <br><br>:'
            f'{IDs_will_divide_wrong}'
        )
        self.sigFixWillDivide.emit(txt, IDs_will_divide_wrong)
        return False
    
    def _check_buds_gen_num_zero(self, checker, frame_i):
        bud_IDs_gen_num_nonzero = (
            checker.get_bud_IDs_gen_num_nonzero()
        )
        if len(bud_IDs_gen_num_nonzero) == 0:
            return True

        category = 'buds whose generation number is not zero'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'the following bud IDs have generation number different from 0:'
            f'<br><br>{bud_IDs_gen_num_nonzero}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_mothers_gen_num_greater_one(self, checker, frame_i):
        moth_IDs_gen_num_non_greater_one = (
            checker.get_moth_IDs_gen_num_non_greater_one()
        )
        if len(moth_IDs_gen_num_non_greater_one) == 0:
            return True

        category = 'mothers whose generation number is < 1'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'the following mother cells have generation number &lt; 1:'
            f'<br><br>{moth_IDs_gen_num_non_greater_one}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_buds_G1(self, checker, frame_i):
        buds_G1 = (
            checker.get_buds_G1()
        )
        if len(buds_G1) == 0:
            return True

        category = 'buds in G1'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'the following bud IDs are in G1 (buds must be in S):'
            f'<br><br>{buds_G1}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_cell_S_rel_ID_zero(self, checker, frame_i):
        cell_S_rel_ID_zero = (
            checker.get_cell_S_rel_ID_zero()
        )
        if len(cell_S_rel_ID_zero) == 0:
            return True

        category = 'buds in G1'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'the following cell IDs in S phase do not have '
            '<code>relative_ID > 0</code>:'
            f'<br><br>{cell_S_rel_ID_zero}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_ID_rel_ID_mismatches(self, checker, frame_i):
        ID_rel_ID_mismatches = checker.get_ID_rel_ID_mismatches()
        if len(ID_rel_ID_mismatches) == 0:
            return True

        items = [
            f'Cell ID {ID} has relative ID = {relID}, '
            f'while cell ID {relID} has relative ID = {relID_of_relID}'
            for ID, relID, relID_of_relID in ID_rel_ID_mismatches
        ]
        category = '`ID-relative_ID` mismatches'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'there are the following `ID-relative_ID` mismatches:'
            f'{html_utils.to_list(items)}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _check_lonely_cells_in_S(self, checker, frame_i):
        lonely_cells_in_S = checker.get_lonely_cells_in_S()
        if len(lonely_cells_in_S) == 0:
            return True

        category = 'Lovely cells in S phase'
        txt = html_utils.paragraph(
            f'At frame n. {frame_i+1} '
            'the following cell IDs are in `S` phase but their `relative_ID` '
            f'does not exist:<br><br>'
            f'{lonely_cells_in_S}'
        )
        self.sigWarning.emit(txt, category)
        return False
    
    def _get_cca_df_copy(self, acdc_df):
        try:
            cca_df = pd.DataFrame(
                data=acdc_df[cca_df_colnames].values,
                columns=cca_df_colnames,
                index=acdc_df.index
            )
            return cca_df
        except KeyError as error:
            return 
        
    def check(self, posData):    
        self.isChecking = True
        checkpoints = (
            '_check_lonely_cells_in_S',
            '_check_equality_num_mothers_buds_in_S',
            '_check_mothers_multiple_buds',
            '_check_buds_gen_num_zero',
            '_check_mothers_gen_num_greater_one',
            '_check_buds_G1',
            '_check_cell_S_rel_ID_zero',
            '_check_ID_rel_ID_mismatches'
        )
        cca_dfs = []
        keys = []
        check_integrity_globally = True
        for frame_i, data_dict in enumerate(posData.allData_li):
            if self.abortChecking:
                check_integrity_globally = False
                break
            
            lab = data_dict['labels']
            if lab is None:
                break
            
            cca_df = data_dict.get('cca_df_checker')
            if cca_df is None:
                # There are no annotations at frame_i --> stop
                break
            
            IDs = data_dict['regionprops'].IDs
            checker = core.CcaIntegrityChecker(cca_df, lab, IDs)
            
            for checkpoint in checkpoints:
                proceed = getattr(self, checkpoint)(checker, frame_i)
                if not proceed:
                    break
            
            if not proceed:
                check_integrity_globally = False
                break
            
            cca_dfs.append(cca_df)
            keys.append(frame_i)
        
        if check_integrity_globally and len(cca_dfs)>1:
            global_checkpoints = [
                '_check_cells_without_G1',
                # '_check_will_divide_is_true'
            ]
            # Check integrity globally
            global_cca_df = pd.concat(cca_dfs, keys=keys, names=['frame_i'])
            for checkpoint in global_checkpoints:
                proceed = getattr(self, checkpoint)(checker, global_cca_df)
                if not proceed:
                    break
        
        self.abortChecking = False
        self.isChecking = False
    
        # Throttle checks, but return early when stopped so that the thread
        # can be quickly joined (see `guiWin.stopCcaIntegrityCheckerWorker`)
        for _ in range(10):
            if self.exit:
                break
            time.sleep(0.1)

    def run(self):
        try:
            self._run()
        except Exception as error:
            # Do not use `worker_exception_handler` because it unlocks
            # `self.mutex`, which this worker never holds when raising.
            # Unlocking a mutex that is not locked crashes on Qt5 and it
            # could release the lock held by the GUI thread
            printl(traceback.format_exc())
            self.dataQ.clear()
            self.critical.emit((self, error))

        # Always emit `finished`, also after an exception, otherwise the
        # thread's event loop is never quit and the QThread would be
        # destroyed while still running --> hard crash (GitHub issue #1299)
        self.isFinished = True
        self.finished.emit(self)

    def _run(self):
        while True:
            if self.exit:
                self.logger.log('Closing cell cycle integrity checker worker...')
                break
            elif not len(self.dataQ) == 0:
                if self.debug:
                    self.logger.log(
                        'Checking integrity of cell cycle annotations '
                        f'({len(self.dataQ)})...'
                    )
                try:
                    data = self.dataQ.pop()
                except IndexError:
                    # Queue cleared by the GUI thread in the meantime
                    continue
                self.check(data)
                if len(self.dataQ) == 0:
                    self.sigDone.emit()
            else:
                self.pause()
