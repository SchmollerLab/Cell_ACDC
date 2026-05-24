"""Background Qt workers: metrics."""

import re
import os
import shutil
import time
import json
import concurrent.futures
from functools import partial
from collections import defaultdict, deque
import itertools

from typing import Union, List, Dict, Callable, Any, Tuple, Iterable

from functools import wraps
import numpy as np
import pandas as pd
import h5py
import traceback

import skimage.io
import skimage.measure
import skimage.exposure

import queue

from tqdm import tqdm

from qtpy.QtCore import Signal, QObject, QMutex, QWaitCondition

from cellacdc import html_utils

from .. import load, myutils, core, prompts, printl, config, segm_re_pattern, io
from .. import transformation, measurements, cca_functions
from ..path import copy_or_move_tree
from .. import features, plot
from .. import core
from .. import cca_df_colnames, lineage_tree_cols, default_annot_df
from .. import cca_df_colnames_with_tree
from .. import cli
from ..utils import resize
from .. import segm_utils

DEBUG = False

from ._base import (
    BaseWorkerUtil,
)

class ComputeMetricsWorker(QObject):
    progressBar = Signal(int, int, float)

    def __init__(self, mainWin):
        QObject.__init__(self)
        self.signals = signals()
        self.abort = False
        self.setup_done = False
        self.logger = workerLogger(self.signals.progress)
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.mainWin = mainWin

    def emitSelectSegmFiles(self, exp_path, pos_foldernames):
        self.mutex.lock()
        self.signals.sigSelectSegmFiles.emit(exp_path, pos_foldernames)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        if self.abort:
            return True
        else:
            return False

    @worker_exception_handler
    def run(self):
        np.seterr(invalid="ignore")
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.standardMetricsErrors = {}
            self.customMetricsErrors = {}
            self.regionPropsErrors = {}
            tot_pos = len(pos_foldernames)
            self.allPosDataInputs = []
            posDatas = []
            self.logger.log("-" * 30)
            expFoldername = os.path.basename(exp_path)

            if i == 0:
                abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
                if abort:
                    self.signals.finished.emit(self)
                    return

            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.signals.finished.emit(self)
                    return

                self.logger.log(
                    f"Processing experiment n. {i + 1}/{tot_exp}, "
                    f"{pos} ({p + 1}/{tot_pos})"
                )

                pos_path = os.path.join(exp_path, pos)
                images_path = os.path.join(pos_path, "Images")
                basename, chNames = myutils.getBasenameAndChNames(
                    images_path, useExt=(".tif", ".h5")
                )

                self.signals.sigUpdatePbarDesc.emit(f"Loading {pos_path}...")

                # Use first found channel, it doesn't matter for metrics
                chName = chNames[0]
                file_path = myutils.getChannelFilePath(images_path, chName)

                # Load data
                posData = load.loadData(file_path, chName)
                posData.getBasenameAndChNames(useExt=(".tif", ".h5"))
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=False,
                    load_acdc_df=True,
                    load_metadata=True,
                    loadSegmInfo=True,
                    load_customCombineMetrics=True,
                )

                posDatas.append(posData)

                self.allPosDataInputs.append(
                    {
                        "file_path": file_path,
                        "chName": chName,
                        "combineMetricsConfig": posData.combineMetricsConfig,
                        "combineMetricsPath": posData.custom_combine_metrics_path,
                    }
                )

            if any([posData.SizeT > 1 for posData in posDatas]):
                self.mutex.lock()
                self.signals.sigAskStopFrame.emit(posDatas)
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()
                if self.abort:
                    self.signals.finished.emit(self)
                    return
                for p, posData in enumerate(posDatas):
                    self.allPosDataInputs[p]["stopFrameNum"] = posData.stopFrameNum
            else:
                for p, posData in enumerate(posDatas):
                    self.allPosDataInputs[p]["stopFrameNum"] = 1

            self.kernel = cli.ComputeMeasurementsKernel(
                self.logger,
                self.mainWin.log_path,
                False,
            )

            from cellacdc.workflow.pipelines.batch import run_gui_measurements_batch
            from cellacdc.workflow.runnable import RunnableConfig

            run_gui_measurements_batch(
                kernel=self.kernel,
                paths=[inp["file_path"] for inp in self.allPosDataInputs],
                stop_frame_numbers=[
                    inp["stopFrameNum"] for inp in self.allPosDataInputs
                ],
                end_filename_segm=self.mainWin.endFilenameSegm,
                compute_metrics_worker=self,
                config=RunnableConfig(logger_func=self.logger.log),
            )

            if self.kernel.setup_done or self.abort:
                return

            self.logger.log("*" * 30)

            self.mutex.lock()
            self.signals.sigErrorsReport.emit(
                self.standardMetricsErrors,
                self.customMetricsErrors,
                self.regionPropsErrors,
            )
            self.waitCond.wait(self.mutex)
            self.mutex.unlock()

        self.signals.finished.emit(self)

    def emitSigComputeVolume(self, posData, stop_frame_n):
        # Recreate allData_li attribute of the gui
        posData.allData_li = []
        for frame_i, lab in enumerate(posData.segm_data[:stop_frame_n]):
            data_dict = {"labels": lab, "regionprops": skimage.measure.regionprops(lab)}
            posData.allData_li.append(data_dict)
        self.mutex.lock()
        self.signals.sigComputeVolume.emit(stop_frame_n, posData)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def emitSigPermissionErrorAndSave(
        self, posData, traceback_str, all_frames_acdc_df, custom_annot_columns
    ):
        self.mutex.lock()
        self.signals.sigPermissionError.emit(
            traceback_str, posData.acdc_output_csv_path
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        load.save_acdc_df_file(
            all_frames_acdc_df,
            posData.acdc_output_csv_path,
            custom_annot_columns=custom_annot_columns,
        )

    def emitSigInitMetricsDialog(self, posData):
        self.mainWin.gui.data = [posData]
        self.mainWin.gui.pos_i = 0
        self.mainWin.gui.isSegm3D = posData.getIsSegm3D()
        self.mutex.lock()
        self.signals.sigInitAddMetrics.emit(posData, self.allPosDataInputs)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def emitSigAskRunNow(self):
        self.mutex.lock()
        self.signals.sigAskRunNow.emit(self)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()


class ComputeMetricsMultiChannelWorker(BaseWorkerUtil):
    sigAskAppendName = Signal(str, list, list)
    sigCriticalNotEnoughSegmFiles = Signal(str)
    sigAborted = Signal()
    sigHowCombineMetrics = Signal(str, list, list, list)

    def __init__(self, mainWin):
        super().__init__(mainWin)

    def emitHowCombineMetrics(
        self,
        imagesPath,
        selectedAcdcOutputEndnames,
        existingAcdcOutputEndnames,
        allChNames,
    ):
        self.mutex.lock()
        self.sigHowCombineMetrics.emit(
            imagesPath,
            selectedAcdcOutputEndnames,
            existingAcdcOutputEndnames,
            allChNames,
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def loadAcdcDfs(self, imagesPath, selectedAcdcOutputEndnames):
        for end in selectedAcdcOutputEndnames:
            filePath, _ = load.get_path_from_endname(end, imagesPath)
            acdc_df = pd.read_csv(filePath)
            yield acdc_df

    def run_iter_exp(self, exp_path, pos_foldernames, i, tot_exp):
        tot_pos = len(pos_foldernames)

        abort = self.emitSelectAcdcOutputFiles(
            exp_path,
            pos_foldernames,
            infoText=" to combine",
            allowSingleSelection=False,
        )
        if abort:
            self.sigAborted.emit()
            return

        # Ask appendend name
        self.mutex.lock()
        self.sigAskAppendName.emit(
            f"{self.mainWin.basename_pos1}acdc_output",
            self.mainWin.existingAcdcOutputEndnames,
            self.mainWin.selectedAcdcOutputEndnames,
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        if self.abort:
            self.sigAborted.emit()
            return

        selectedAcdcOutputEndnames = self.mainWin.selectedAcdcOutputEndnames
        existingAcdcOutputEndnames = self.mainWin.existingAcdcOutputEndnames
        appendedName = self.appendedName

        self.signals.initProgressBar.emit(len(pos_foldernames))
        for p, pos in enumerate(pos_foldernames):
            if self.abort:
                self.sigAborted.emit()
                return

            self.logger.log(
                f"Processing experiment n. {i + 1}/{tot_exp}, {pos} ({p + 1}/{tot_pos})"
            )

            imagesPath = os.path.join(exp_path, pos, "Images")
            basename, chNames = myutils.getBasenameAndChNames(
                imagesPath, useExt=(".tif", ".h5")
            )

            if p == 0:
                abort = self.emitHowCombineMetrics(
                    imagesPath,
                    selectedAcdcOutputEndnames,
                    existingAcdcOutputEndnames,
                    chNames,
                )
                if abort:
                    self.sigAborted.emit()
                    return
                acdcDfs = self.acdcDfs.values()
                # Update selected acdc_dfs since the user could have
                # loaded additional ones inside the emitHowCombineMetrics
                # dialog
                selectedAcdcOutputEndnames = self.acdcDfs.keys()
            else:
                acdcDfs = self.loadAcdcDfs(imagesPath, selectedAcdcOutputEndnames)

            dfs = []
            for i, acdc_df in enumerate(acdcDfs):
                dfs.append(acdc_df.add_suffix(f"_table{i + 1}"))
            combined_df = pd.concat(dfs, axis=1)

            newAcdcDf = pd.DataFrame(index=combined_df.index)
            for newColname, equation in self.equations.items():
                newAcdcDf[newColname] = combined_df.eval(equation)

            newAcdcDfPath = os.path.join(
                imagesPath, f"{basename}acdc_output_{appendedName}.csv"
            )
            newAcdcDf.to_csv(newAcdcDfPath)

            equationsIniPath = os.path.join(
                imagesPath, f"{basename}equations_{appendedName}.ini"
            )
            equationsConfig = config.ConfigParser()
            if os.path.exists(equationsIniPath):
                equationsConfig.read(equationsIniPath)
            equationsConfig = self.addEquationsToConfigPars(
                equationsConfig, selectedAcdcOutputEndnames, self.equations
            )
            with open(equationsIniPath, "w") as configfile:
                equationsConfig.write(configfile)

            self.signals.progressBar.emit(1)

        return True

    def addEquationsToConfigPars(self, cp, selectedAcdcOutputEndnames, equations):
        section = [
            f"df{i + 1}:{end}" for i, end in enumerate(selectedAcdcOutputEndnames)
        ]
        section = ";".join(section)
        if section not in cp:
            cp[section] = {}

        for metricName, expression in equations.items():
            cp[section][metricName] = expression

        return cp

    @worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        self.errors = {}
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            try:
                result = self.run_iter_exp(exp_path, pos_foldernames, i, tot_exp)
                if result is None:
                    return
            except Exception as e:
                traceback_str = traceback.format_exc()
                self.errors[e] = traceback_str
                self.logger.log(traceback_str)

        self.signals.finished.emit(self)


class ConcatAcdcDfsWorker(BaseWorkerUtil):
    sigAborted = Signal()
    sigAskFolder = Signal(str)
    sigSetMeasurements = Signal(object)
    sigAskAppendName = Signal(str, list)

    def __init__(self, mainWin, format="CSV"):
        super().__init__(mainWin)
        if format.startswith("CSV"):
            self._to_format = "to_csv"
        elif format.startswith("XLS"):
            self._to_format = "to_excel"

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

    @worker_exception_handler
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
                    exp_path,
                    pos_foldernames,
                    infoText=" to combine",
                    allowSingleSelection=True,
                    multiSelection=False,
                )
                if abort:
                    self.sigAborted.emit()
                    return

            selectedAcdcOutputEndname = self.mainWin.selectedAcdcOutputEndnames[0]
            selectedAcdcObjsCountEndname = selectedAcdcOutputEndname.replace(
                "acdc_output", "acdc_objects_count"
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
                    f"Processing experiment n. {i + 1}/{tot_exp}, "
                    f"{pos} ({p + 1}/{tot_pos})"
                )

                images_path = os.path.join(exp_path, pos, "Images")

                ls = myutils.listdir(images_path)

                acdc_output_file = [
                    f for f in ls if f.endswith(f"{selectedAcdcOutputEndname}.csv")
                ]
                if not acdc_output_file:
                    self.logger.log(
                        f"{pos} does not contain any "
                        f"{selectedAcdcOutputEndname}.csv file. "
                        "Skipping it."
                    )
                    self.signals.progressBar.emit(1)
                    continue

                acdc_objs_count_file = [
                    f for f in ls if f.endswith(f"{selectedAcdcObjsCountEndname}.csv")
                ]
                if acdc_objs_count_file:
                    df_count_filepath = os.path.join(
                        images_path, acdc_objs_count_file[0]
                    )
                    df_count = pd.read_csv(df_count_filepath)
                    acdc_objs_count_dfs[pos] = df_count

                acdc_df_filepath = os.path.join(images_path, acdc_output_file[0])
                acdc_df = pd.read_csv(acdc_df_filepath).set_index("Cell_ID")
                acdc_dfs.append(acdc_df)
                keys.append(pos)

                self.signals.progressBar.emit(1)

            self.signals.initProgressBar.emit(0)
            acdc_df_allpos = pd.concat(
                acdc_dfs, keys=keys, names=["Position_n", "Cell_ID"]
            )
            acdc_df_allpos["experiment_folderpath"] = exp_path

            basename, chNames = myutils.getBasenameAndChNames(
                images_path, useExt=(".tif", ".h5")
            )
            df_metadata = load.load_metadata_df(images_path)
            SizeZ = df_metadata.at["SizeZ", "values"]
            SizeZ = int(float(SizeZ))
            existing_colnames = acdc_df_allpos.columns
            isSegm3D = any([col.endswith("3D") for col in existing_colnames])

            if i == 0:
                kwargs = {
                    "loadedChNames": chNames,
                    "notLoadedChNames": [],
                    "isZstack": SizeZ > 1,
                    "isSegm3D": isSegm3D,
                    "existing_colnames": existing_colnames,
                }
                self.emitSetMeasurements(kwargs)
                if self.abort:
                    self.sigAborted.emit()
                    return

            selected_cols = [
                col for col in self.selectedColumns if col in acdc_df_allpos.columns
            ]
            acdc_df_allpos = acdc_df_allpos[selected_cols]
            acdc_dfs_allexp.append(acdc_df_allpos)
            exp_name = os.path.basename(exp_path)
            keys_exp.append((exp_path, exp_name))

            allpos_dir = os.path.join(exp_path, "AllPos_acdc_output")
            if not os.path.exists(allpos_dir):
                os.mkdir(allpos_dir)

            allPos_acdc_df_basename = f"AllPos_{selectedAcdcOutputEndname}"
            if i == 0:
                self.emitAskAppendName(allPos_acdc_df_basename)
                if self.abort:
                    self.sigAborted.emit()
                    return

            acdc_objs_count_df_allpos_filename = self.concat_df_filename.replace(
                "acdc_output", "acdc_objects_count"
            )

            acdc_dfs_allpos_filepath = os.path.join(allpos_dir, self.concat_df_filename)

            self.logger.log(
                "Saving all positions concatenated file to "
                f'"{acdc_dfs_allpos_filepath}"'
            )
            to_format_func = getattr(acdc_df_allpos, self._to_format)
            to_format_func(acdc_dfs_allpos_filepath)
            self.acdc_dfs_allpos_filepath = acdc_dfs_allpos_filepath

            if not acdc_objs_count_dfs:
                continue

            acdc_objs_count_df_allpos = pd.concat(
                acdc_objs_count_dfs, names=["Position_n"]
            )
            acdc_objs_count_df_allpos["experiment_folderpath"] = exp_path

            acdc_objs_count_df_allpos_filepath = os.path.join(
                allpos_dir, acdc_objs_count_df_allpos_filename
            )

            self.logger.log(
                "Saving all positions objects count file to "
                f'"{acdc_objs_count_df_allpos_filepath}"'
            )
            to_format_func = getattr(acdc_objs_count_df_allpos, self._to_format)
            to_format_func(acdc_objs_count_df_allpos_filepath)

            acdc_objs_count_dfs_allexp[(exp_path, exp_name)] = acdc_objs_count_df_allpos

        if len(keys_exp) <= 1:
            self.signals.finished.emit(self)
            return

        allExp_filename = f"multiExp_{self.concat_df_filename}"
        self.mutex.lock()
        self.sigAskFolder.emit(allExp_filename)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        if self.abort:
            self.sigAborted.emit()
            return

        acdc_df_allexp = pd.concat(
            acdc_dfs_allexp,
            keys=keys_exp,
            names=["experiment_folderpath", "experiment_foldername"],
        )
        acdc_dfs_allexp_filepath = os.path.join(self.allExpSaveFolder, allExp_filename)
        self.logger.log(
            "Saving multiple experiments concatenated file to "
            f'"{acdc_dfs_allexp_filepath}"'
        )
        to_format_func = getattr(acdc_df_allexp, self._to_format)
        to_format_func(acdc_dfs_allexp_filepath)

        if acdc_objs_count_dfs_allexp:
            allexp_count_df_filename = f"multiExp_{acdc_objs_count_df_allpos_filename}"
            acdc_objs_count_df_allexp = pd.concat(
                acdc_objs_count_dfs_allexp,
                names=["experiment_folderpath", "experiment_foldername"],
            )
            acdc_objs_count_df_allexp_filepath = os.path.join(
                self.allExpSaveFolder, allexp_count_df_filename
            )
            self.logger.log(
                "Saving multiple experiments concatenated file to "
                f'"{acdc_objs_count_df_allexp_filepath}"'
            )
            to_format_func = getattr(acdc_objs_count_df_allexp, self._to_format)
            to_format_func(acdc_objs_count_df_allexp_filepath)

        self.signals.finished.emit(self)


class ConcatSpotmaxDfsWorker(BaseWorkerUtil):
    sigAborted = Signal()
    sigAskFolder = Signal(str)
    sigSetMeasurements = Signal(object)
    sigAskAppendName = Signal(str, list)

    def __init__(self, mainWin, format="CSV"):
        super().__init__(mainWin)
        if format.startswith("CSV"):
            self._final_ext = ".csv"
        elif format.startswith("XLS"):
            self._final_ext = ".xlsx"
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
            acdc_df = pd.read_csv(filepath, index_col=["frame_i", "Cell_ID"])
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

            if debug and col == "cell_vol_fl":
                printl(df[[col]])

        return df

    def emitAskFolderWhereToSaveMultiExp(self):
        self.mutex.lock()
        self.sigAskFolder.emit("")
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

            images_path = os.path.join(exp_path, pos, "Images")
            acdc_df = self.getAcdcDf(images_path)
            if acdc_df is None:
                continue

            acdc_dfs.append(acdc_df)
            keys.append(pos)

        if not acdc_dfs:
            return True

        acdc_df_allpos = pd.concat(
            acdc_dfs, keys=keys, names=["Position_n", "frame_i", "Cell_ID"]
        )
        acdc_df_allpos["experiment_folderpath"] = exp_path
        basename, chNames = myutils.getBasenameAndChNames(
            images_path, useExt=(".tif", ".h5")
        )
        df_metadata = load.load_metadata_df(images_path)
        SizeZ = df_metadata.at["SizeZ", "values"]
        SizeZ = int(float(SizeZ))
        existing_colnames = acdc_df_allpos.columns
        isSegm3D = any([col.endswith("3D") for col in existing_colnames])

        kwargs = {
            "loadedChNames": chNames,
            "notLoadedChNames": [],
            "isZstack": SizeZ > 1,
            "isSegm3D": isSegm3D,
            "existing_colnames": existing_colnames,
        }
        self.emitSetMeasurements(kwargs)
        if self.abort:
            self.sigAborted.emit()
            return False

        return True

    @worker_exception_handler
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

            all_runs = get_runs_num_and_desc(exp_path, pos_foldernames=pos_foldernames)
            if not all_runs:
                self.logger.log(
                    "[WARNING] The following experiment does not contain "
                    f'valid spotMAX output files. Skipping it. "{exp_path}"'
                )
                continue

            if not runNumberAlreadyAsked:
                abort = self.emitSelectSpotmaxRun(
                    exp_path,
                    pos_foldernames,
                    all_runs,
                    infoText=" to combine",
                    allowSingleSelection=True,
                    multiSelection=False,
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
                spotmax_output_path = os.path.join(pos_path, "spotMAX_output")

                if not os.path.exists(spotmax_output_path):
                    self.logger.log(
                        "[WARNING] The following Position folder does not contain "
                        f'valid spotMAX output files. Skipping it. "{pos_path}"'
                    )
                    continue

                images_path = os.path.join(exp_path, pos, "Images")

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
                    f"Processing experiment n. {i + 1}/{tot_exp}, "
                    f"{pos} ({p + 1}/{tot_pos})"
                )

                for run_desc in selectedSpotmaxRuns:
                    run, desc = run_desc.split("_...")
                    ini_filename = f"{run}_analysis_parameters{desc}.ini"
                    ini_filepath = os.path.join(spotmax_output_path, ini_filename)
                    if not os.path.exists(ini_filepath):
                        self.logger.log(
                            "[WARNING] The following Position folder does not contain "
                            f"the spotMAX output file for run number {run}. "
                            f'Skipping it. "{pos_path}"'
                        )
                        continue

                    pos_ini_filepaths[(run, desc)] = ini_filepath
                    for _, pattern_filename in DFs_FILENAMES.items():
                        run_filename = pattern_filename.replace("*rn*", run)
                        run_filename = run_filename.replace("*desc*", desc)
                        aggr_filename = f"{run_filename}_aggregated.csv"
                        aggr_filepath = os.path.join(spotmax_output_path, aggr_filename)
                        if not os.path.exists(aggr_filepath):
                            continue

                        df_spots_filename = f"{run_filename}.h5"
                        spots_filepath = os.path.join(
                            spotmax_output_path, df_spots_filename
                        )
                        ext_spots = ".h5"
                        if not os.path.exists(spots_filepath):
                            df_spots_filename = f"{run_filename}.csv"
                            spots_filepath = os.path.join(
                                spotmax_output_path, df_spots_filename
                            )
                            ext_spots = ".csv"

                        if not os.path.exists(spots_filepath):
                            continue

                        analysis_step = re.findall(
                            r"\*rn\*(.*)\*desc\*", pattern_filename
                        )[0]
                        key = (run, analysis_step, desc, ext_spots)
                        try:
                            df_spots = (
                                spotmax.io.load_spots_table(
                                    spotmax_output_path, df_spots_filename
                                )
                                .reset_index()
                                .set_index(["frame_i", "Cell_ID"])
                            )
                            df_spots = self.copyCcaColsFromAcdcDf(
                                df_spots, acdc_df, debug=False
                            )
                            df_spots = df_spots.reset_index().set_index(
                                ["frame_i", "Cell_ID", "spot_id"]
                            )
                            dfs_spots[key].append(df_spots)
                        except Exception as err:
                            self.logger.log(str(err), level="ERROR")
                            self.logger.log(
                                "WARNING: Error when reading single-spots "
                                "tables (possibly because there are no spots). "
                                "Skipping this Position.",
                                level="WARNING",
                            )
                            pass

                        df_aggregated = pd.read_csv(
                            aggr_filepath, index_col=["frame_i", "Cell_ID"]
                        )
                        df_aggregated = self.copyCcaColsFromAcdcDf(
                            df_aggregated, acdc_df
                        )
                        dfs_aggr[key].append(df_aggregated)
                        pos_runs[key].append(pos)

                    ref_ch_id_text = re.findall(
                        r"\*rn\*(.*)\*desc\*", DF_REF_CH_FILENAME
                    )[0]
                    ref_ch_filename = DF_REF_CH_FILENAME.replace("*rn*", run)
                    ref_ch_filename = ref_ch_filename.replace("*desc*", desc)
                    ref_ch_filepath = os.path.join(spotmax_output_path, ref_ch_filename)
                    if not os.path.exists(ref_ch_filepath):
                        continue

                    df_ref_ch = pd.read_csv(
                        ref_ch_filepath, index_col=["frame_i", "Cell_ID"]
                    )
                    df_ref_ch = self.copyCcaColsFromAcdcDf(df_ref_ch, acdc_df)
                    ref_ch_key = (run, ref_ch_id_text, desc)
                    dfs_ref_ch[ref_ch_key].append(df_ref_ch)
                    pos_runs_ref_ch[ref_ch_key].append(pos)

                self.signals.progressBar.emit(1)

            self.signals.initProgressBar.emit(0)

            self.logger.log("Saving concantenated files...")

            allpos_folderpath = os.path.join(exp_path, "spotMAX_multipos_output")
            os.makedirs(allpos_folderpath, exist_ok=True)

            exp_name = os.path.basename(exp_path)
            for key, dfs in dfs_spots.items():
                pos_keys = pos_runs[key]
                run, analysis_step, desc, ext_spots = key

                if ext_spots == ".csv":
                    ext_spots = self._final_ext
                filename = f"multipos_{run}{analysis_step}{desc}{ext_spots}"
                all_exp_key = filename
                df_spots_concat = spotmax.io.save_concat_dfs(
                    dfs,
                    pos_keys,
                    allpos_folderpath,
                    filename,
                    ext_spots,
                    names=["Position_n"],
                    return_concat_df=True,
                )
                df_spots_concat["experiment_foldername"] = exp_name
                df_spots_concat["experiment_folderpath"] = exp_path
                spotmax_dfs_spots_allexp[all_exp_key]["dfs"].append(df_spots_concat)
                spotmax_dfs_spots_allexp[all_exp_key]["keys"].append(exp_path)
                ini_filepath = pos_ini_filepaths[(run, desc)]
                ini_filename = os.path.basename(ini_filepath)
                dst_ini_filepath = os.path.join(allpos_folderpath, ini_filename)
                if not os.path.exists(dst_ini_filepath):
                    shutil.copy2(ini_filepath, dst_ini_filepath)

                spotmax_dfs_spots_allexp[all_exp_key]["ini_filepath"].append(
                    dst_ini_filepath
                )

            for key, dfs in dfs_aggr.items():
                pos_keys = pos_runs[key]
                run, analysis_step, desc, _ = key
                filename = (
                    f"multipos_{run}{analysis_step}{desc}_aggregated{self._final_ext}"
                )
                all_exp_aggr_key = filename
                df_aggr_concat = spotmax.io.save_concat_dfs(
                    dfs,
                    pos_keys,
                    allpos_folderpath,
                    filename,
                    self._final_ext,
                    names=["Position_n"],
                    return_concat_df=True,
                )
                spotmax_dfs_aggr_allexp[all_exp_aggr_key]["dfs"].append(df_aggr_concat)
                spotmax_dfs_aggr_allexp[all_exp_aggr_key]["keys"].append(
                    (exp_path, exp_name)
                )

            for key, dfs in dfs_ref_ch.items():
                run, ref_ch_id_text, desc = key
                pos_keys = pos_runs_ref_ch[key]
                filename = f"multipos_{run}{ref_ch_id_text}{desc}{self._final_ext}"
                all_exp_ref_ch_key = filename
                df_ref_ch_concat = spotmax.io.save_concat_dfs(
                    dfs,
                    pos_keys,
                    allpos_folderpath,
                    filename,
                    self._final_ext,
                    names=["Position_n"],
                    return_concat_df=True,
                )
                ref_ch_dfs_allexp[all_exp_ref_ch_key]["dfs"].append(df_ref_ch_concat)
                ref_ch_dfs_allexp[all_exp_ref_ch_key]["keys"].append(
                    (exp_path, exp_name)
                )

        multiexp_dst_folderpath = ""
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
        names = ["experiment_folderpath", "experiment_foldername"]
        for filename, items in spotmax_dfs_spots_allexp.items():
            keys = items["keys"]
            dfs = items["dfs"]
            multiexp_filename = f"multiexp_{filename}"
            extension = os.path.splitext(filename)[-1]
            spotmax.io.save_concat_dfs(
                dfs,
                keys,
                multiexp_dst_folderpath,
                multiexp_filename,
                extension,
                names=["experiment_folderpath"],
            )
            ini_filepath = items["ini_filepath"][0]
            ini_filename = os.path.basename(ini_filepath)
            dst_ini_filepath = os.path.join(multiexp_dst_folderpath, ini_filename)
            if not os.path.exists(dst_ini_filepath):
                shutil.copy2(ini_filepath, dst_ini_filepath)

        for filename, items in spotmax_dfs_aggr_allexp.items():
            keys = items["keys"]
            dfs = items["dfs"]
            printl(keys, pretty=True)
            multiexp_filename = f"multiexp_{filename}"
            extension = os.path.splitext(filename)[-1]
            spotmax.io.save_concat_dfs(
                dfs,
                keys,
                multiexp_dst_folderpath,
                multiexp_filename,
                extension,
                names=names,
            )

        for filename, items in ref_ch_dfs_allexp.items():
            keys = items["keys"]
            dfs = items["dfs"]
            multiexp_filename = f"multiexp_{filename}"
            extension = os.path.splitext(filename)[-1]
            spotmax.io.save_concat_dfs(
                dfs,
                keys,
                multiexp_dst_folderpath,
                multiexp_filename,
                extension,
                names=names,
            )

        self.signals.finished.emit(self)


class CcaIntegrityCheckerWorker(QObject):
    finished = Signal(object)
    critical = Signal(object)
    progress = Signal(str, object)
    sigDone = Signal()
    sigWarning = Signal(str, str)
    sigFixWillDivide = Signal(str, list)

    def __init__(self, mutex, waitCond):
        QObject.__init__(self)
        self.logger = workerLogger(self.progress)
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
            self.logger.log("Cell cycle annotations checker is idle.")
        self.mutex.lock()
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
        if self.debug:
            self.logger.log("Enqueing posData...")
        self.dataQ.append(posData)
        if len(self.dataQ) == 1:
            # Wake worker upon inserting first element
            self.abortChecking = False
            self.waitCond.wakeAll()

    def clearQueue(self):
        self.dataQ.clear()

    def _stop(self):
        self.exit = True
        self.waitCond.wakeAll()

    def abort(self):
        self.abortChecking = True
        while not len(self.dataQ) == 0:
            data = self.dataQ.pop()
            del data
        self._stop()

    def _check_equality_num_mothers_buds_in_S(self, checker, frame_i):
        num_moth_S, num_buds = checker.get_num_mothers_and_buds_in_S()

        if num_moth_S == num_buds:
            return True

        category = "number of buds different from number of mothers in S phase"
        ul_items = [
            f"Number of buds = {num_buds}",
            f"Number of mothers in S phase = {num_moth_S}",
        ]
        txt = html_utils.paragraph(
            f"At frame n. {frame_i + 1} the number of buds and number of "
            "mother cells in S phase are different!"
            f"{html_utils.to_list(ul_items)}"
        )
        self.sigWarning.emit(txt, category)
        return False

    def _check_mothers_multiple_buds(self, checker, frame_i):
        mother_IDs_with_multiple_buds = checker.get_mother_IDs_with_multiple_buds()
        if len(mother_IDs_with_multiple_buds) == 0:
            return True

        category = "mother cells with multiple buds"
        txt = html_utils.paragraph(
            f"At frame n. {frame_i + 1} "
            "the following mother cells have <b>multiple buds</b> assigned to it"
            f"<br><br>{mother_IDs_with_multiple_buds}"
        )
        self.sigWarning.emit(txt, category)
        return False

    def _check_cells_without_G1(self, checker, global_cca_df):
        IDs_cycles_without_G1 = checker.get_IDs_cycles_without_G1(global_cca_df)
        if len(IDs_cycles_without_G1) == 0:
            return True

        category = "cell cycles without G1"
        txt = html_utils.paragraph(
            "Cell-ACDC requires that every cell cycle has at least "
            "one frame in G1.<br>"
            "The following pairs of <code>(ID, generation number)</code> "
            "do not satisfy this condition:<br><br>"
            f"{IDs_cycles_without_G1}"
        )
        self.sigWarning.emit(txt, category)
        return False

    def _check_will_divide_is_true(self, checker, global_cca_df):
        # NOTE: unfortunately this function performs pandas manipulations
        # that are either not thread-safe or in any case are freezing the
        # GUI. For now we don't run this until we find a solution
        return True

        IDs_will_divide_wrong = checker.get_IDs_gen_num_will_divide_wrong(global_cca_df)
        if len(IDs_will_divide_wrong) == 0:
            return True

        txt = html_utils.paragraph(
            "Cell-ACDC found that `will_divide` is annotated as True on the "
            "following <code>(ID, generation number)</code> cell<br>"
            "despite the fact that division is still not annotated on "
            "these cells <br><br>:"
            f"{IDs_will_divide_wrong}"
        )
        self.sigFixWillDivide.emit(txt, IDs_will_divide_wrong)
        return False

    def _check_buds_gen_num_zero(self, checker, frame_i):
        bud_IDs_gen_num_nonzero = checker.get_bud_IDs_gen_num_nonzero()
        if len(bud_IDs_gen_num_nonzero) == 0:
            return True

        category = "buds whose generation number is not zero"
        txt = html_utils.paragraph(
            f"At frame n. {frame_i + 1} "
            "the following bud IDs have generation number different from 0:"
            f"<br><br>{bud_IDs_gen_num_nonzero}"
        )
        self.sigWarning.emit(txt, category)
        return False

    def _check_mothers_gen_num_greater_one(self, checker, frame_i):
        moth_IDs_gen_num_non_greater_one = (
            checker.get_moth_IDs_gen_num_non_greater_one()
        )
        if len(moth_IDs_gen_num_non_greater_one) == 0:
            return True

        category = "mothers whose generation number is < 1"
        txt = html_utils.paragraph(
            f"At frame n. {frame_i + 1} "
            "the following mother cells have generation number &lt; 1:"
            f"<br><br>{moth_IDs_gen_num_non_greater_one}"
        )
        self.sigWarning.emit(txt, category)
        return False

    def _check_buds_G1(self, checker, frame_i):
        buds_G1 = checker.get_buds_G1()
        if len(buds_G1) == 0:
            return True

        category = "buds in G1"
        txt = html_utils.paragraph(
            f"At frame n. {frame_i + 1} "
            "the following bud IDs are in G1 (buds must be in S):"
            f"<br><br>{buds_G1}"
        )
        self.sigWarning.emit(txt, category)
        return False

    def _check_cell_S_rel_ID_zero(self, checker, frame_i):
        cell_S_rel_ID_zero = checker.get_cell_S_rel_ID_zero()
        if len(cell_S_rel_ID_zero) == 0:
            return True

        category = "buds in G1"
        txt = html_utils.paragraph(
            f"At frame n. {frame_i + 1} "
            "the following cell IDs in S phase do not have "
            "<code>relative_ID > 0</code>:"
            f"<br><br>{cell_S_rel_ID_zero}"
        )
        self.sigWarning.emit(txt, category)
        return False

    def _check_ID_rel_ID_mismatches(self, checker, frame_i):
        ID_rel_ID_mismatches = checker.get_ID_rel_ID_mismatches()
        if len(ID_rel_ID_mismatches) == 0:
            return True

        items = [
            f"Cell ID {ID} has relative ID = {relID}, "
            f"while cell ID {relID} has relative ID = {relID_of_relID}"
            for ID, relID, relID_of_relID in ID_rel_ID_mismatches
        ]
        category = "`ID-relative_ID` mismatches"
        txt = html_utils.paragraph(
            f"At frame n. {frame_i + 1} "
            "there are the following `ID-relative_ID` mismatches:"
            f"{html_utils.to_list(items)}"
        )
        self.sigWarning.emit(txt, category)
        return False

    def _check_lonely_cells_in_S(self, checker, frame_i):
        lonely_cells_in_S = checker.get_lonely_cells_in_S()
        if len(lonely_cells_in_S) == 0:
            return True

        category = "Lovely cells in S phase"
        txt = html_utils.paragraph(
            f"At frame n. {frame_i + 1} "
            "the following cell IDs are in `S` phase but their `relative_ID` "
            f"does not exist:<br><br>"
            f"{lonely_cells_in_S}"
        )
        self.sigWarning.emit(txt, category)
        return False

    def _get_cca_df_copy(self, acdc_df):
        try:
            cca_df = pd.DataFrame(
                data=acdc_df[cca_df_colnames].values,
                columns=cca_df_colnames,
                index=acdc_df.index,
            )
            return cca_df
        except KeyError as error:
            return

    def check(self, posData):
        self.isChecking = True
        checkpoints = (
            "_check_lonely_cells_in_S",
            "_check_equality_num_mothers_buds_in_S",
            "_check_mothers_multiple_buds",
            "_check_buds_gen_num_zero",
            "_check_mothers_gen_num_greater_one",
            "_check_buds_G1",
            "_check_cell_S_rel_ID_zero",
            "_check_ID_rel_ID_mismatches",
        )
        cca_dfs = []
        keys = []
        check_integrity_globally = True
        for frame_i, data_dict in enumerate(posData.allData_li):
            if self.abortChecking:
                check_integrity_globally = False
                break

            lab = data_dict["labels"]
            if lab is None:
                break

            cca_df = data_dict.get("cca_df_checker")
            if cca_df is None:
                # There are no annotations at frame_i --> stop
                break

            IDs = data_dict["IDs"]
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

        if check_integrity_globally and len(cca_dfs) > 1:
            global_checkpoints = [
                "_check_cells_without_G1",
                # '_check_will_divide_is_true'
            ]
            # Check integrity globally
            global_cca_df = pd.concat(cca_dfs, keys=keys, names=["frame_i"])
            for checkpoint in global_checkpoints:
                proceed = getattr(self, checkpoint)(checker, global_cca_df)
                if not proceed:
                    break

        self.abortChecking = False
        self.isChecking = False
        time.sleep(1)

    @worker_exception_handler
    def run(self):
        while True:
            if self.exit:
                self.logger.log("Closing cell cycle integrity checker worker...")
                break
            elif not len(self.dataQ) == 0:
                if self.debug:
                    self.logger.log(
                        "Checking integrity of cell cycle annotations "
                        f"({len(self.dataQ)})..."
                    )
                data = self.dataQ.pop()
                self.check(data)
                if len(self.dataQ) == 0:
                    self.sigDone.emit()
            else:
                self.pause()
        self.isFinished = True
        self.finished.emit(self)


class GenerateMotherBudTotalTableWorker(BaseWorkerUtil):
    def __init__(
        self, parentWin, input_csv_filepath, selected_options, out_csv_filepath
    ):
        super().__init__(parentWin)
        self.input_csv_filepath = input_csv_filepath
        self.selected_options = selected_options
        self.out_csv_filepath = out_csv_filepath

    @worker_exception_handler
    def run(self):
        self.logger.log(f'Loading table "{self.input_csv_filepath}"...')
        self.signals.initProgressBar.emit(0)

        input_df = pd.read_csv(self.input_csv_filepath)

        self.logger.log("Generating output table...")
        out_df = cca_functions.generate_mother_bud_total_df(
            input_df, **self.selected_options
        )

        self.logger.log(f'Saving output table to "{self.out_csv_filepath}"...')

        out_df.to_csv(self.out_csv_filepath)

        self.signals.finished.emit(self)


class CountObjectsInSegm(BaseWorkerUtil):
    sigAskAppendName = Signal(str, list)
    sigAborted = Signal()

    def __init__(self, mainWin):
        super().__init__(mainWin)

    @worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            self.mainWin.infoText = f"Select <b>segmentation file to count</b>"
            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.sigAborted.emit()
                return

            self.signals.initProgressBar.emit(len(pos_foldernames))
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigAborted.emit()
                    return

                self.logger.log(
                    f"Processing experiment n. {i + 1}/{tot_exp}, "
                    f"{pos} ({p + 1}/{tot_pos})"
                )

                images_path = os.path.join(exp_path, pos, "Images")
                endFilenameSegm = self.mainWin.endFilenameSegm
                ls = myutils.listdir(images_path)
                file_path = [
                    os.path.join(images_path, f)
                    for f in ls
                    if f.endswith(f"{endFilenameSegm}.npz")
                ][0]

                posData = load.loadData(file_path, "")

                self.signals.sigUpdatePbarDesc.emit(f"Processing {posData.pos_path}")

                posData.getBasenameAndChNames()
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_acdc_df=False,
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm,
                )
                if posData.segm_data.ndim == 3:
                    posData.segm_data = posData.segm_data[np.newaxis]

                self.logger.log("Counting objects...")

                countMapper = posData.countObjectsInSegm()
                countMapper.pop("In current frame", None)
                df_count_endname = posData.saveObjCounts(countMapper)

                self.logger.log(
                    "Saved object counts table to file ending with: "
                    f'"{df_count_endname}"'
                )

                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)

# Sibling imports (deferred to avoid import cycles)
from ._base import (
    signals,
    workerLogger,
    worker_exception_handler,
)

