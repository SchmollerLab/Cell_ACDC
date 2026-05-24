"""View adapter for measurement setup and dialogs."""

from __future__ import annotations

import pandas as pd

from cellacdc import apps, cli, favourite_func_metrics_csv_path, widgets


class MeasurementsMixin:
    """Qt-facing adapter around measurement view-model contracts."""

    """Headless measurement calculation and setup rules."""

    def _favourite_metric_functions(self):
        try:
            df_favourite_funcs = pd.read_csv(favourite_func_metrics_csv_path)
            return df_favourite_funcs["favourite_func_name"].to_list()
        except Exception:
            return None

    def _log_removed_measurements(self, del_cols, del_rps):
        del_cols_format = [f"  *  {colname}" for colname in del_cols]
        del_rps_format = [f"  *  {colname}" for colname in del_rps]
        del_cols_format.extend(del_rps_format)
        del_cols_format = "\n".join(del_cols_format)
        self.logger.info(del_cols_format)

    def _remove_existing_unchecked_measurements(self):
        self.logger.info("Removing existing unchecked measurements...")
        del_cols = self.measurementsWin.existingUncheckedColnames
        del_rps = self.measurementsWin.existingUncheckedRps
        self._log_removed_measurements(del_cols, del_rps)
        for pos_data in self.data:
            for data_dict in pos_data.allData_li:
                data_dict["acdc_df"] = self.drop_unchecked_measurements(
                    data_dict["acdc_df"],
                    del_cols,
                    del_rps,
                )

    def _set_metrics(self, measurements_win):
        self._measurements_kernel.set_metrics_from_set_measurements_dialog(
            measurements_win
        )
        for ch_name in self._measurements_kernel.chNamesToProcess:
            if ch_name not in self.notLoadedChNames:
                continue

            success = self.loadFluo_cb(fluo_channels=[ch_name])
            if not success:
                continue

    def add_combine_metric(self):
        pos_data = self.data[self.pos_i]
        is_zstack = pos_data.SizeZ > 1
        win = apps.combineMetricsEquationDialog(
            self.ch_names,
            is_zstack,
            self.isSegm3D,
            parent=self,
        )
        win.sigOk.connect(self.save_combine_metrics_to_pos_data)
        win.exec_()
        win.sigOk.disconnect()

    def add_custom_metric(self, checked=False):
        txt = self.custom_metrics_instructions()
        metrics_path = self.metrics_examples_path()
        msg = widgets.myMessageBox()
        msg.addShowInFileManagerButton(metrics_path, "Show example...")
        title = "Add custom metrics instructions"
        msg.information(self, title, txt, buttonsTexts=("Ok",))

    def all_acdc_df_columns(self, all_pos_data):
        columns = set()
        for pos_data in all_pos_data:
            for data_dict in pos_data.allData_li:
                acdc_df = data_dict["acdc_df"]
                if acdc_df is None:
                    continue
                columns.update(acdc_df.columns)
        return columns

    def custom_metrics_instructions(self):
        return measurements.add_metrics_instructions()

    def drop_unchecked_measurements(self, acdc_df, columns, regionprops):
        if acdc_df is None:
            return None
        acdc_df = acdc_df.drop(columns=columns, errors="ignore")
        for col_rp in regionprops:
            drop_df_rp = acdc_df.filter(regex=rf"{col_rp}.*", axis=1)
            drop_cols_rp = drop_df_rp.columns
            acdc_df = acdc_df.drop(columns=drop_cols_rp, errors="ignore")
        return acdc_df

    def init_metrics(self):
        self.logger.info("Initializing measurements...")
        pos_data = self.data[self.pos_i]
        self._measurements_kernel = cli.ComputeMeasurementsKernel(
            self.logger, self.log_path, False
        )
        self._measurements_kernel.init_args(pos_data.chNames, pos_data.getSegmEndname())
        self._measurements_kernel._init_metrics(pos_data, self.isSegm3D)

    def init_metrics_to_save(self, pos_data):
        self._measurements_kernel._init_metrics_to_save(pos_data)

    def metrics_examples_path(self):
        return measurements.metrics_path

    def not_loaded_channels(self, all_channel_names, loaded_channel_names):
        return [c for c in all_channel_names if c not in loaded_channel_names]

    def rotational_volume(
        self,
        obj,
        physical_size_y=1,
        physical_size_x=1,
        logger=None,
    ):
        return _calc_rot_vol(
            obj,
            physical_size_y,
            physical_size_x,
            logger=logger,
        )

    def save_combine_metrics_to_pos_data(self, window):
        for pos_data in self.data:
            equations_dict, is_mixed_channels = window.getEquationsDict()
            for new_col_name, equation in equations_dict.items():
                pos_data.addEquationCombineMetrics(
                    equation, new_col_name, is_mixed_channels
                )
                pos_data.saveCombineMetrics()

        if self.measurementsWin is None:
            return

        self.measurementsWinState = self.measurementsWin.state()
        self.measurementsWin.close()
        self.show_set_measurements()
        self.measurementsWin.restoreState(self.measurementsWinState)

    def set_measurements(self):
        if self.measurementsWin.delExistingCols:
            self._remove_existing_unchecked_measurements()
        self.setMeasWinState = self.measurementsWin.state()
        self.logger.info("Setting measurements...")
        self._set_metrics(self.measurementsWin)
        self.logger.info("Metrics successfully set.")
        self.measurementsWin = None

    def set_measurements_cancelled(self):
        self.measurementsWin = None

    def set_metrics_func(self):
        pos_data = self.data[self.pos_i]
        self._measurements_kernel._set_metrics_func_from_posData(pos_data)

    def show_set_measurements(self, checked=False, qparent=None):
        qparent = qparent if qparent is not None else self
        if self.measurementsWin is not None:
            self.measurementsWin.show()
            self.measurementsWin.raise_()
            self.measurementsWin.activateWindow()
            return

        favourite_funcs = self._favourite_metric_functions()
        pos_data = self.data[self.pos_i]
        all_pos_acdc_df_cols = self.all_acdc_df_columns(self.data)
        loaded_ch_names = pos_data.setLoadedChannelNames(returnList=True)
        pos_data.fluo_data_dict.pop(self.user_ch_name, None)
        if self.user_ch_name not in loaded_ch_names:
            loaded_ch_names.insert(0, self.user_ch_name)
        not_loaded_ch_names = self.not_loaded_channels(
            self.ch_names,
            loaded_ch_names,
        )
        self.notLoadedChNames = not_loaded_ch_names
        self.measurementsWin = apps.SetMeasurementsDialog(
            loaded_ch_names,
            not_loaded_ch_names,
            pos_data.SizeZ > 1,
            self.isSegm3D,
            favourite_funcs=favourite_funcs,
            allPos_acdc_df_cols=list(all_pos_acdc_df_cols),
            acdc_df_path=pos_data.images_path,
            posData=pos_data,
            addCombineMetricCallback=self.add_combine_metric,
            allPosData=self.data,
            parent=qparent,
            state=self.setMeasWinState,
        )
        self.measurementsWin.sigCancel.connect(self.set_measurements_cancelled)
        self.measurementsWin.sigClosed.connect(self.set_measurements)
        self.measurementsWin.show()
