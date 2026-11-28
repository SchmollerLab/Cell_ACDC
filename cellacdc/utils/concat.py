from functools import partial

from qtpy.QtWidgets import QFileDialog

from cellacdc import measurements

from .. import apps, myutils, workers, widgets, html_utils
from .. import printl

from .base import NewThreadMultipleExpBaseUtil
import re
import os
import shutil
from collections import defaultdict
import pandas as pd
from qtpy.QtCore import Signal
from cellacdc import load
from cellacdc import cca_df_colnames, lineage_tree_cols, default_annot_df
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base

class ConcatWin(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, parent=None
        ):
        module = myutils.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.expPaths = expPaths
        if title.find('spotMAX') != -1:
            self.worker_func = workers.ConcatSpotmaxDfsWorker
            self._infoText = 'spotMAX'
        else:
            self.worker_func = workers.ConcatAcdcDfsWorker
            self._infoText = 'acdc_output'
    
    def runWorker(self, format='CSV'):
        self.worker = self.worker_func(self, format=format)
        self.worker.sigAskFolder.connect(self.askFolderWhereToSaveAllExp)
        self.worker.sigAborted.connect(self.workerAborted)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigSetMeasurements.connect(self.askSetMeasurements)
        self.worker.signals.sigAskCopyCca.connect(self.askCopyCcaFromAcdcOutput)
        super().runWorker(self.worker)
    
    def askCopyCcaFromAcdcOutput(self, images_path):
        acdc_output_tables = []
        for file in myutils.listdir(images_path):
            if not file.endswith('.csv'):
                continue
            
            idx = file.find('acdc_output')
            if idx == -1:
                continue
            
            acdc_output_tables.append(file[idx:])
        
        if not acdc_output_tables:
            self.worker.waitCond.wakeAll()
            return
        
        txt = html_utils.paragraph(
            'Do you want to <b>copy cell cycle annotations</b><br>'
            'from one of the tables below?<br><br>'
            'If yes, please select from which table you want to copy from:'
        )
        noButton = widgets.noPushButton('No, do not copy')
        # The empty-selection check applies only to OK: "No" needs no table
        selectTableNameWin = widgets.QDialogListbox(
            'Copy cell cycle annotations?', txt,
            acdc_output_tables, multiSelection=False, parent=self,
            additionalButtons=(noButton,), allowEmptySelection=False
        )
        # ok_cb reports the button passed to it, never `sender()`
        noButton.clicked.connect(
            partial(selectTableNameWin.ok_cb, button=noButton)
        )
        selectTableNameWin.exec_()
        if selectTableNameWin.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return

        if selectTableNameWin.clickedButton == noButton:
            self.worker.waitCond.wakeAll()
            return

        selectedTableNames = selectTableNameWin.selectedItemsText
        self.worker.setAcdcOutputEndname(selectedTableNames[0])
        self.worker.waitCond.wakeAll()
    
    def askSetMeasurements(self, kwargs):
        loadedChNames = kwargs['loadedChNames']
        notLoadedChNames = kwargs['notLoadedChNames']
        isZstack = kwargs['isZstack']
        isSegm3D = kwargs['isSegm3D']
        self.setMeasurementsWin = apps.SetMeasurementsDialog(
            loadedChNames, notLoadedChNames, isZstack, isSegm3D,
            is_concat=True, parent=self
        )
        existing_colnames = kwargs['existing_colnames']
        self.setMeasurementsWin.addNonMeasurementColumns(
            existing_colnames
        )
        self.setMeasurementsWin.setDisabledNotExistingMeasurements(
            existing_colnames
        )
        self.setMeasurementsWin.sigClosed.connect(self.setMeasurements)
        self.setMeasurementsWin.sigCancel.connect(self.setMeasurementsCancelled)
        self.setMeasurementsWin.show()
    
    def setMeasurements(self):
        selectedColumns = []
        if hasattr(self.setMeasurementsWin, 'nonMeasurementsGroupbox'):
            if self.setMeasurementsWin.nonMeasurementsGroupbox.isChecked():
                groupbox = self.setMeasurementsWin.nonMeasurementsGroupbox
                for checkBox in groupbox.checkBoxes: 
                    if not checkBox.isEnabled():
                        continue
                    if not checkBox.isChecked():
                        continue
                    colname = checkBox.text()
                    selectedColumns.append(colname) 
                
        for chNameGroupbox in self.setMeasurementsWin.chNameGroupboxes:
            chName = chNameGroupbox.chName
            if not chNameGroupbox.isChecked():
                # Skip entire channel
                continue
            
            for checkBox in chNameGroupbox.checkBoxes:
                if not checkBox.isEnabled():
                    continue
                
                if not checkBox.isChecked():
                    continue
                colname = checkBox.text()
                selectedColumns.append(colname)
        
        if self.setMeasurementsWin.sizeMetricsQGBox.isChecked():
            for checkBox in self.setMeasurementsWin.sizeMetricsQGBox.checkBoxes:
                if not checkBox.isEnabled():
                    continue
                
                if not checkBox.isChecked():
                    continue
                colname = checkBox.text()
                selectedColumns.append(colname)
        
        selectedPropsNames = []
        if self.setMeasurementsWin.regionPropsQGBox.isChecked():
            for checkBox in self.setMeasurementsWin.regionPropsQGBox.checkBoxes:
                if not checkBox.isEnabled():
                    continue
                
                if not checkBox.isChecked():
                    continue
                colname = checkBox.text()
                selectedPropsNames.append(colname)
            selectedRpCols = measurements.get_regionprops_columns(
                self.setMeasurementsWin.existing_colnames, selectedPropsNames
            )
            selectedColumns.extend(selectedRpCols)
        
        checkMixedChannel = (
            self.setMeasurementsWin.mixedChannelsCombineMetricsQGBox is not None
            and self.setMeasurementsWin.mixedChannelsCombineMetricsQGBox.isChecked()
        )
        if checkMixedChannel:
            win = self.setMeasurementsWin
            checkBoxes = win.mixedChannelsCombineMetricsQGBox.checkBoxes
            for checkBox in checkBoxes:
                if not checkBox.isEnabled():
                    continue
                
                if not checkBox.isChecked():
                    continue
                colname = checkBox.text()
                selectedColumns.append(colname)
            
        self.worker.selectedColumns = selectedColumns
        self.worker.abort = False
        self.worker.waitCond.wakeAll()
    
    def setMeasurementsCancelled(self):
        self.worker.abort = True
        self.worker.waitCond.wakeAll()
    
    def showEvent(self, event):
        formats = (
            'CSV (Comma Separated Values)', 
            'XLS (Excel)'
        )
        selectFormatWin = widgets.QDialogListbox(
            'Select output file format',
            'Select format of the output file\n',
            formats, multiSelection=False, parent=self,
            allowEmptySelection=False
        )
        selectFormatWin.exec_()
        if selectFormatWin.cancel:
            return
        
        if selectFormatWin.selectedItemsText[0].startswith('CSV'):
            self._ext = '.csv'
        else:
            self._ext = '.xlsx'
            myutils.check_install_package(
                'OpenPyXL', 
                import_pkg_name='openpyxl', 
                pypi_name='XlsxWriter'
            )
        self.runWorker(format=selectFormatWin.selectedItemsText[0])
    
    def askAppendName(self, basename, existingEndnames):
        win = apps.filenameDialog(
            basename=basename,
            hintText='Insert a name for the <b>concatenated table</b> file:',
            existingNames=existingEndnames, 
            allowEmpty=True,
            ext=self._ext
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True     
        else:   
            self.worker.concat_df_filename = win.filename
        self.worker.waitCond.wakeAll()
    
    def askFolderWhereToSaveAllExp(self, allExp_filename):
        txt = ("""
            After clicking "Ok" you will be asked to <b>select a folder where you
            want to save the file</b><br>
            with the <b>concatenated tables</b> from the multiple experiments selected<br>
            
        """)
        if allExp_filename:
            txt = f'{txt}(the filename will be <code>{allExp_filename}</code>)'
        
        txt = html_utils.paragraph(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.information(self, 'Select folder', txt)
        if msg.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
        
    
        mostRecentPath = myutils.getMostRecentPath()
        save_to_dir = QFileDialog.getExistingDirectory(
            self, f'Select folder where to save multiple experiments table', 
            mostRecentPath
        )
        if not save_to_dir:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
        
        self.worker.allExpSaveFolder = save_to_dir

        self.worker.waitCond.wakeAll()

    def workerAborted(self):
        self.worker.signals.finished.emit(self)
        self.workerFinished(self.worker, aborted=True)
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = f'Concatenating <code>{self._infoText}</code> tables aborted.'
        else:
            txt = f'Concatenating <code>{self._infoText}</code> tables completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()


class ConcatAcdcDfsWorker(utils_base.BaseWorkerUtil):
    sigAborted = Signal()
    sigAskFolder = Signal(str)
    sigSetMeasurements = Signal(object)
    sigAskAppendName = Signal(str, list)

    def __init__(self, mainWin, format='CSV'):
        super().__init__(mainWin)
        if format.startswith('CSV'):
            self._to_format = 'to_csv'
        elif format.startswith('XLS'):
            self._to_format = 'to_excel'
    
    def emitSetMeasurements(self, kwargs):
        self.mutex.lock()
        self.sigSetMeasurements.emit(kwargs)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
    
    def emitAskAppendName(self, allPos_acdc_df_basename):
        # Ask appendend name
        self.mutex.lock()
        self.sigAskAppendName.emit(allPos_acdc_df_basename, [])
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    @workers_base.worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        
        self.signals.initProgressBar.emit(0)
        acdc_dfs_allexp = []
        acdc_objs_count_dfs_allexp = {}
        keys_exp = []
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)
            
            if i == 0:
                abort = self.emitSelectAcdcOutputFiles(
                    exp_path, pos_foldernames, infoText=' to combine',
                    allowSingleSelection=True, multiSelection=False
                )
                if abort:
                    self.sigAborted.emit()
                    return

            selectedAcdcOutputEndname = self.mainWin.selectedAcdcOutputEndnames[0]
            selectedAcdcObjsCountEndname = selectedAcdcOutputEndname.replace(
                'acdc_output', 'acdc_objects_count'
            )

            self.signals.initProgressBar.emit(len(pos_foldernames))
            acdc_dfs = []
            acdc_objs_count_dfs = {}
            keys = []
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigAborted.emit()
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')

                ls = myutils.listdir(images_path)

                acdc_output_file = [
                    f for f in ls 
                    if f.endswith(f'{selectedAcdcOutputEndname}.csv')
                ]
                if not acdc_output_file:
                    self.logger.log(
                        f'{pos} does not contain any '
                        f'{selectedAcdcOutputEndname}.csv file. '
                        'Skipping it.'
                    )
                    self.signals.progressBar.emit(1)
                    continue
                
                acdc_objs_count_file = [
                    f for f in ls 
                    if f.endswith(f'{selectedAcdcObjsCountEndname}.csv')
                ]
                if acdc_objs_count_file:
                    df_count_filepath = os.path.join(
                        images_path, acdc_objs_count_file[0]
                    )
                    df_count = pd.read_csv(df_count_filepath)
                    acdc_objs_count_dfs[pos] = df_count
                
                acdc_df_filepath = os.path.join(images_path, acdc_output_file[0])
                acdc_df = pd.read_csv(acdc_df_filepath).set_index('Cell_ID')
                acdc_dfs.append(acdc_df)
                keys.append(pos)

                self.signals.progressBar.emit(1)

            self.signals.initProgressBar.emit(0)
            acdc_df_allpos = pd.concat(
                acdc_dfs, keys=keys, names=['Position_n', 'Cell_ID']
            )
            acdc_df_allpos['experiment_folderpath'] = exp_path
            
            basename, chNames = myutils.getBasenameAndChNames(
                images_path, useExt=('.tif', '.h5')
            )
            df_metadata = load.load_metadata_df(images_path)
            SizeZ = df_metadata.at['SizeZ', 'values']
            SizeZ = int(float(SizeZ))
            existing_colnames = acdc_df_allpos.columns
            isSegm3D = any([col.endswith('3D') for col in existing_colnames])
            
            if i == 0:
                kwargs = {
                    'loadedChNames': chNames, 
                    'notLoadedChNames': [],
                    'isZstack': SizeZ > 1,
                    'isSegm3D': isSegm3D,
                    'existing_colnames': existing_colnames
                }
                self.emitSetMeasurements(kwargs)
                if self.abort:
                    self.sigAborted.emit()
                    return
            
            selected_cols = [
                col for col in self.selectedColumns 
                if col in acdc_df_allpos.columns
            ]
            acdc_df_allpos = acdc_df_allpos[selected_cols]
            acdc_dfs_allexp.append(acdc_df_allpos)
            exp_name = os.path.basename(exp_path)
            keys_exp.append((exp_path, exp_name))

            allpos_dir = os.path.join(exp_path, 'AllPos_acdc_output')
            if not os.path.exists(allpos_dir):
                os.mkdir(allpos_dir)
            
            allPos_acdc_df_basename = f'AllPos_{selectedAcdcOutputEndname}'
            if i == 0:
                self.emitAskAppendName(allPos_acdc_df_basename)
                if self.abort:
                    self.sigAborted.emit()
                    return
            
            acdc_objs_count_df_allpos_filename = (
                self.concat_df_filename.replace(
                    'acdc_output', 'acdc_objects_count'
                )
            )
            
            acdc_dfs_allpos_filepath = os.path.join(
                allpos_dir, self.concat_df_filename
            )

            self.logger.log(
                'Saving all positions concatenated file to '
                f'"{acdc_dfs_allpos_filepath}"'
            )
            to_format_func = getattr(acdc_df_allpos, self._to_format)
            to_format_func(acdc_dfs_allpos_filepath)
            self.acdc_dfs_allpos_filepath = acdc_dfs_allpos_filepath
            
            if not acdc_objs_count_dfs:
                continue
            
            acdc_objs_count_df_allpos = pd.concat(
                acdc_objs_count_dfs, names=['Position_n']
            )
            acdc_objs_count_df_allpos['experiment_folderpath'] = exp_path
            
            acdc_objs_count_df_allpos_filepath = os.path.join(
                allpos_dir, acdc_objs_count_df_allpos_filename
            )
            
            self.logger.log(
                'Saving all positions objects count file to '
                f'"{acdc_objs_count_df_allpos_filepath}"'
            )
            to_format_func = getattr(acdc_objs_count_df_allpos, self._to_format)
            to_format_func(acdc_objs_count_df_allpos_filepath)
            
            acdc_objs_count_dfs_allexp[(exp_path, exp_name)] = (
                acdc_objs_count_df_allpos
            )
        
        if len(keys_exp) <= 1:
            self.signals.finished.emit(self)
            return
        
        allExp_filename = f'multiExp_{self.concat_df_filename}'
        self.mutex.lock()
        self.sigAskFolder.emit(allExp_filename)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        if self.abort:
            self.sigAborted.emit()
            return
        
        acdc_df_allexp = pd.concat(
            acdc_dfs_allexp, keys=keys_exp, 
            names=['experiment_folderpath', 'experiment_foldername']
        )
        acdc_dfs_allexp_filepath = os.path.join(
            self.allExpSaveFolder, allExp_filename
        )
        self.logger.log(
            'Saving multiple experiments concatenated file to '
            f'"{acdc_dfs_allexp_filepath}"'
        )
        to_format_func = getattr(acdc_df_allexp, self._to_format)
        to_format_func(acdc_dfs_allexp_filepath)
        
        if acdc_objs_count_dfs_allexp:
            allexp_count_df_filename = (
                f'multiExp_{acdc_objs_count_df_allpos_filename}'
            )
            acdc_objs_count_df_allexp = pd.concat(
                acdc_objs_count_dfs_allexp,
                names=['experiment_folderpath', 'experiment_foldername']
            )
            acdc_objs_count_df_allexp_filepath = os.path.join(
                self.allExpSaveFolder, allexp_count_df_filename
            )
            self.logger.log(
                'Saving multiple experiments concatenated file to '
                f'"{acdc_objs_count_df_allexp_filepath}"'
            )
            to_format_func = getattr(acdc_objs_count_df_allexp, self._to_format)
            to_format_func(acdc_objs_count_df_allexp_filepath)

        self.signals.finished.emit(self)


class ConcatSpotmaxDfsWorker(utils_base.BaseWorkerUtil):
    sigAborted = Signal()
    sigAskFolder = Signal(str)
    sigSetMeasurements = Signal(object)
    sigAskAppendName = Signal(str, list)

    def __init__(self, mainWin, format='CSV'):
        super().__init__(mainWin)
        if format.startswith('CSV'):
            self._final_ext = '.csv'
        elif format.startswith('XLS'):
            self._final_ext = '.xlsx'
        self.acdcOutputEndname = None
    
    def emitSetMeasurements(self, kwargs):
        self.mutex.lock()
        self.sigSetMeasurements.emit(kwargs)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
    
    def emitAskAppendName(self, allPos_spotmax_df_basename):
        # Ask appendend name
        self.mutex.lock()
        self.sigAskAppendName.emit(allPos_spotmax_df_basename, [])
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def emitAskCopyCca(self, images_path):
        self.mutex.lock()
        self.signals.sigAskCopyCca.emit(images_path)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
    
    def setAcdcOutputEndname(self, acdcOutputEndname):
        self.acdcOutputEndname = acdcOutputEndname
    
    def getAcdcDf(self, images_path):
        if self.acdcOutputEndname is None:
            return
        
        for file in myutils.listdir(images_path):
            if not file.endswith(self.acdcOutputEndname):
                continue
            
            filepath = os.path.join(images_path, file)
            acdc_df = pd.read_csv(filepath, index_col=['frame_i', 'Cell_ID'])
            return acdc_df
    
    def copyCcaColsFromAcdcDf(self, df, acdc_df, debug=False):
        if acdc_df is None:
            return df
        
        if debug:
            printl(acdc_df.columns.to_list(), pretty=True)
        
        idx = df.index.intersection(acdc_df.index)
        for col in cca_df_colnames:
            if col not in acdc_df.columns:
                continue
        
            if col not in self.selectedColumns:
                continue
            
            df.loc[idx, col] = acdc_df.loc[idx, col]
        
        for col in lineage_tree_cols:
            if col not in acdc_df.columns:
                continue
            
            if col not in self.selectedColumns:
                continue
            
            df.loc[idx, col] = acdc_df.loc[idx, col]
        
        for col in default_annot_df.keys():
            if col not in acdc_df.columns:
                continue
            
            if col not in self.selectedColumns:
                continue
            
            df.loc[idx, col] = acdc_df.loc[idx, col]
        
        for col in self.selectedColumns:
            if col not in acdc_df.columns:
                continue
            
            df.loc[idx, col] = acdc_df.loc[idx, col]
            
            if debug and col == 'cell_vol_fl':
                printl(df[[col]])
        
        return df
    
    def emitAskFolderWhereToSaveMultiExp(self):
        self.mutex.lock()
        self.sigAskFolder.emit('')
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        if self.abort:
            self.sigAborted.emit()
            return

        return self.allExpSaveFolder
    
    def askSelectMeasurements(self, exp_path, posFoldernames):
        acdc_dfs = []
        keys = []
        for p, pos in enumerate(posFoldernames):
            if self.abort:
                self.sigAborted.emit()
                return False

            images_path = os.path.join(exp_path, pos, 'Images')
            acdc_df = self.getAcdcDf(images_path)
            if acdc_df is None:
                continue
            
            acdc_dfs.append(acdc_df)
            keys.append(pos)
        
        if not acdc_dfs:
            return True
        
        acdc_df_allpos = pd.concat(
            acdc_dfs, keys=keys, names=['Position_n', 'frame_i', 'Cell_ID']
        )
        acdc_df_allpos['experiment_folderpath'] = exp_path
        basename, chNames = myutils.getBasenameAndChNames(
            images_path, useExt=('.tif', '.h5')
        )
        df_metadata = load.load_metadata_df(images_path)
        SizeZ = df_metadata.at['SizeZ', 'values']
        SizeZ = int(float(SizeZ))
        existing_colnames = acdc_df_allpos.columns
        isSegm3D = any([col.endswith('3D') for col in existing_colnames])
        
        kwargs = {
            'loadedChNames': chNames, 
            'notLoadedChNames': [],
            'isZstack': SizeZ > 1,
            'isSegm3D': isSegm3D,
            'existing_colnames': existing_colnames
        }
        self.emitSetMeasurements(kwargs)
        if self.abort:
            self.sigAborted.emit()
            return False
        
        return True
    
    @workers_base.worker_exception_handler
    def run(self):
        from spotmax import DFs_FILENAMES, DF_REF_CH_FILENAME
        from spotmax.utils import get_runs_num_and_desc
        import spotmax.io
        
        self.selectedColumns = None
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        spotmax_dfs_spots_allexp = defaultdict(lambda: defaultdict(list))
        spotmax_dfs_aggr_allexp = defaultdict(lambda: defaultdict(list))
        ref_ch_dfs_allexp = defaultdict(lambda: defaultdict(list))
        runNumberAlreadyAsked = False
        copyFromCcaAlreadyAsked = False
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)
            
            all_runs = get_runs_num_and_desc(
                exp_path, pos_foldernames=pos_foldernames
            )
            if not all_runs:
                self.logger.log(
                    '[WARNING] The following experiment does not contain '
                    f'valid spotMAX output files. Skipping it. "{exp_path}"'
                )
                continue
            
            if not runNumberAlreadyAsked:
                abort = self.emitSelectSpotmaxRun(
                    exp_path, pos_foldernames, all_runs, 
                    infoText=' to combine',
                    allowSingleSelection=True, 
                    multiSelection=False
                )
                if abort:
                    self.sigAborted.emit()
                    return
                runNumberAlreadyAsked = True
            
            selectedSpotmaxRuns = self.mainWin.selectedSpotmaxRuns

            self.signals.initProgressBar.emit(len(pos_foldernames))
            dfs_spots = defaultdict(list)
            dfs_aggr = defaultdict(list)
            dfs_ref_ch = defaultdict(list)
            pos_runs = defaultdict(list)
            pos_runs_ref_ch = defaultdict(list)
            pos_ini_filepaths = {}
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigAborted.emit()
                    return
                
                pos_path = os.path.join(exp_path, pos)
                spotmax_output_path = os.path.join(pos_path, 'spotMAX_output')
                
                if not os.path.exists(spotmax_output_path):
                    self.logger.log(
                        '[WARNING] The following Position folder does not contain '
                        f'valid spotMAX output files. Skipping it. "{pos_path}"'
                    )
                    continue
                
                images_path = os.path.join(exp_path, pos, 'Images')
                
                if not copyFromCcaAlreadyAsked:
                    self.emitAskCopyCca(images_path)
                    if self.abort:
                        self.sigAborted.emit()
                        return
                    
                    self.askSelectMeasurements(exp_path, pos_foldernames)
                    if self.abort:
                        return  
                    copyFromCcaAlreadyAsked = True            
                    
                acdc_df = self.getAcdcDf(images_path)
                
                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                
                for run_desc in selectedSpotmaxRuns:
                    run, desc = run_desc.split('_...')
                    ini_filename = f'{run}_analysis_parameters{desc}.ini'
                    ini_filepath = os.path.join(
                        spotmax_output_path, ini_filename
                    )
                    if not os.path.exists(ini_filepath):
                        self.logger.log(
                            '[WARNING] The following Position folder does not contain '
                            f'the spotMAX output file for run number {run}. '
                            f'Skipping it. "{pos_path}"'
                        )
                        continue
                        
                    pos_ini_filepaths[(run, desc)] = ini_filepath
                    for _, pattern_filename in DFs_FILENAMES.items():
                        run_filename = pattern_filename.replace('*rn*', run)
                        run_filename = run_filename.replace('*desc*', desc)
                        aggr_filename = f'{run_filename}_aggregated.csv'
                        aggr_filepath = os.path.join(
                            spotmax_output_path, aggr_filename
                        )
                        if not os.path.exists(aggr_filepath):
                            continue                        
                        
                        df_spots_filename = f'{run_filename}.h5'
                        spots_filepath = os.path.join(
                            spotmax_output_path, df_spots_filename
                        )
                        ext_spots = '.h5'
                        if not os.path.exists(spots_filepath):
                            df_spots_filename = f'{run_filename}.csv'
                            spots_filepath = os.path.join(
                                spotmax_output_path, df_spots_filename
                            )
                            ext_spots = '.csv'
                        
                        if not os.path.exists(spots_filepath):
                            continue
                        
                        analysis_step = re.findall(
                            r'\*rn\*(.*)\*desc\*', pattern_filename
                        )[0]
                        key = (run, analysis_step, desc, ext_spots)
                        try:
                            df_spots = spotmax.io.load_spots_table(
                                spotmax_output_path, df_spots_filename
                            ).reset_index().set_index(['frame_i', 'Cell_ID'])
                            df_spots = self.copyCcaColsFromAcdcDf(
                                df_spots, acdc_df, debug=False
                            )
                            df_spots = (
                                df_spots.reset_index()
                                .set_index(['frame_i', 'Cell_ID', 'spot_id'])
                            )
                            dfs_spots[key].append(df_spots)
                        except Exception as err:
                            self.logger.log(str(err), level='ERROR')
                            self.logger.log(
                                'WARNING: Error when reading single-spots '
                                'tables (possibly because there are no spots). '
                                'Skipping this Position.', 
                                level='WARNING'
                            )
                            pass
                        
                        df_aggregated = pd.read_csv(
                            aggr_filepath, index_col=['frame_i', 'Cell_ID']
                        )
                        df_aggregated = self.copyCcaColsFromAcdcDf(
                            df_aggregated, acdc_df
                        )
                        dfs_aggr[key].append(df_aggregated)
                        pos_runs[key].append(pos)
                    
                    ref_ch_id_text = re.findall(
                        r'\*rn\*(.*)\*desc\*', DF_REF_CH_FILENAME
                    )[0]
                    ref_ch_filename = (
                        DF_REF_CH_FILENAME.replace('*rn*', run)
                    )
                    ref_ch_filename = (
                        ref_ch_filename.replace('*desc*', desc)
                    )
                    ref_ch_filepath = os.path.join(
                        spotmax_output_path, ref_ch_filename
                    )
                    if not os.path.exists(ref_ch_filepath):
                        continue
                    
                    df_ref_ch = pd.read_csv(
                        ref_ch_filepath, index_col=['frame_i', 'Cell_ID']
                    )
                    df_ref_ch = self.copyCcaColsFromAcdcDf(df_ref_ch, acdc_df)
                    ref_ch_key = (run, ref_ch_id_text, desc)
                    dfs_ref_ch[ref_ch_key].append(df_ref_ch)
                    pos_runs_ref_ch[ref_ch_key].append(pos)

                self.signals.progressBar.emit(1)        
            
            self.signals.initProgressBar.emit(0)
            
            self.logger.log('Saving concatenated files...')
            
            allpos_folderpath = os.path.join(exp_path, 'spotMAX_multipos_output')
            os.makedirs(allpos_folderpath, exist_ok=True)
            
            exp_name = os.path.basename(exp_path)
            for key, dfs in dfs_spots.items():
                pos_keys = pos_runs[key]
                run, analysis_step, desc, ext_spots = key
                
                if ext_spots == '.csv':
                    ext_spots = self._final_ext
                filename = f'multipos_{run}{analysis_step}{desc}{ext_spots}'
                all_exp_key = filename
                df_spots_concat = spotmax.io.save_concat_dfs(
                    dfs, pos_keys, allpos_folderpath, filename, ext_spots, 
                    names=['Position_n'], return_concat_df=True
                )
                df_spots_concat['experiment_foldername'] = exp_name
                df_spots_concat['experiment_folderpath'] = exp_path
                spotmax_dfs_spots_allexp[all_exp_key]['dfs'].append(
                    df_spots_concat
                )
                spotmax_dfs_spots_allexp[all_exp_key]['keys'].append(
                    exp_path
                )
                ini_filepath = pos_ini_filepaths[(run, desc)]
                ini_filename = os.path.basename(ini_filepath)
                dst_ini_filepath = os.path.join(allpos_folderpath, ini_filename)
                if not os.path.exists(dst_ini_filepath):
                    shutil.copy2(ini_filepath, dst_ini_filepath)
                    
                spotmax_dfs_spots_allexp[all_exp_key]['ini_filepath'].append(
                    dst_ini_filepath
                )
            
            for key, dfs in dfs_aggr.items():
                pos_keys = pos_runs[key]
                run, analysis_step, desc, _ = key
                filename = (
                    f'multipos_{run}{analysis_step}{desc}'
                    f'_aggregated{self._final_ext}'
                )
                all_exp_aggr_key = filename
                df_aggr_concat = spotmax.io.save_concat_dfs(
                    dfs, pos_keys, allpos_folderpath, filename, self._final_ext, 
                    names=['Position_n'], return_concat_df=True
                )
                spotmax_dfs_aggr_allexp[all_exp_aggr_key]['dfs'].append(
                    df_aggr_concat
                )
                spotmax_dfs_aggr_allexp[all_exp_aggr_key]['keys'].append(
                    (exp_path, exp_name)
                )
            
            for key, dfs in dfs_ref_ch.items():
                run, ref_ch_id_text, desc = key
                pos_keys = pos_runs_ref_ch[key]
                filename = (
                    f'multipos_{run}{ref_ch_id_text}{desc}{self._final_ext}'
                )
                all_exp_ref_ch_key = filename
                df_ref_ch_concat = spotmax.io.save_concat_dfs(
                    dfs, pos_keys, allpos_folderpath, filename, self._final_ext,
                    names=['Position_n'], return_concat_df=True
                )
                ref_ch_dfs_allexp[all_exp_ref_ch_key]['dfs'].append(
                    df_ref_ch_concat
                )
                ref_ch_dfs_allexp[all_exp_ref_ch_key]['keys'].append(
                    (exp_path, exp_name)
                )
        
        multiexp_dst_folderpath = ''
        if len(expPaths) == 1:
            self.signals.finished.emit(self)
            return
        
        multiexp_dst_folderpath = self.emitAskFolderWhereToSaveMultiExp()
        printl(multiexp_dst_folderpath)
        if multiexp_dst_folderpath is None:
            return
        
        self.logger.log(
            f'Saving multi-experiment files to "{multiexp_dst_folderpath}"...'
        )
        names = ['experiment_folderpath', 'experiment_foldername']
        for filename, items in spotmax_dfs_spots_allexp.items():
            keys = items['keys']
            dfs = items['dfs']
            multiexp_filename = f'multiexp_{filename}'
            extension = os.path.splitext(filename)[-1]
            spotmax.io.save_concat_dfs(
                dfs, keys, multiexp_dst_folderpath, 
                multiexp_filename, 
                extension,
                names=['experiment_folderpath']
            )
            ini_filepath = items['ini_filepath'][0]
            ini_filename = os.path.basename(ini_filepath)
            dst_ini_filepath = os.path.join(
                multiexp_dst_folderpath, ini_filename
            )
            if not os.path.exists(dst_ini_filepath):
                shutil.copy2(ini_filepath, dst_ini_filepath)
            
        for filename, items in spotmax_dfs_aggr_allexp.items():
            keys = items['keys']
            dfs = items['dfs']
            printl(keys, pretty=True)
            multiexp_filename = f'multiexp_{filename}'
            extension = os.path.splitext(filename)[-1]
            spotmax.io.save_concat_dfs(
                dfs, keys, multiexp_dst_folderpath, 
                multiexp_filename, 
                extension,
                names=names
            )
        
        for filename, items in ref_ch_dfs_allexp.items():
            keys = items['keys']
            dfs = items['dfs']
            multiexp_filename = f'multiexp_{filename}'
            extension = os.path.splitext(filename)[-1]
            spotmax.io.save_concat_dfs(
                dfs, keys, multiexp_dst_folderpath, 
                multiexp_filename, 
                extension,
                names=names
            )
        
        self.signals.finished.emit(self)
