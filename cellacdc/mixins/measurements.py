"""View adapter for measurement setup and dialogs."""

from __future__ import annotations

import pandas as pd

from cellacdc import apps, cli, favourite_func_metrics_csv_path, widgets


class MeasurementsView:
    """Qt-facing adapter around measurement view-model contracts."""

    """Headless measurement calculation and setup rules."""

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

    def custom_metrics_instructions(self):
        return measurements.add_metrics_instructions()

    def metrics_examples_path(self):
        return measurements.metrics_path

    def all_acdc_df_columns(self, all_pos_data):
        columns = set()
        for pos_data in all_pos_data:
            for data_dict in pos_data.allData_li:
                acdc_df = data_dict['acdc_df']
                if acdc_df is None:
                    continue
                columns.update(acdc_df.columns)
        return columns

    def not_loaded_channels(self, all_channel_names, loaded_channel_names):
        return [c for c in all_channel_names if c not in loaded_channel_names]

    def drop_unchecked_measurements(self, acdc_df, columns, regionprops):
        if acdc_df is None:
            return None
        acdc_df = acdc_df.drop(columns=columns, errors='ignore')
        for col_rp in regionprops:
            drop_df_rp = acdc_df.filter(regex=fr'{col_rp}.*', axis=1)
            drop_cols_rp = drop_df_rp.columns
            acdc_df = acdc_df.drop(columns=drop_cols_rp, errors='ignore')
        return acdc_df


    def __init__(self, host):
        self.host = host
    def init_metrics_to_save(self, pos_data):
        self.host._measurements_kernel._init_metrics_to_save(pos_data)

    def init_metrics(self):
        self.host.logger.info('Initializing measurements...')
        pos_data = self.host.data[self.host.pos_i]
        self.host._measurements_kernel = cli.ComputeMeasurementsKernel(
            self.host.logger, self.host.log_path, False
        )
        self.host._measurements_kernel.init_args(
            pos_data.chNames, pos_data.getSegmEndname()
        )
        self.host._measurements_kernel._init_metrics(
            pos_data, self.host.isSegm3D
        )

    def show_set_measurements(self, checked=False, qparent=None):
        qparent = qparent if qparent is not None else self.host
        if self.host.measurementsWin is not None:
            self.host.measurementsWin.show()
            self.host.measurementsWin.raise_()
            self.host.measurementsWin.activateWindow()
            return

        favourite_funcs = self._favourite_metric_functions()
        pos_data = self.host.data[self.host.pos_i]
        all_pos_acdc_df_cols = self.all_acdc_df_columns(
            self.host.data
        )
        loaded_ch_names = pos_data.setLoadedChannelNames(returnList=True)
        pos_data.fluo_data_dict.pop(self.host.user_ch_name, None)
        if self.host.user_ch_name not in loaded_ch_names:
            loaded_ch_names.insert(0, self.host.user_ch_name)
        not_loaded_ch_names = self.not_loaded_channels(
            self.host.ch_names,
            loaded_ch_names,
        )
        self.host.notLoadedChNames = not_loaded_ch_names
        self.host.measurementsWin = apps.SetMeasurementsDialog(
            loaded_ch_names,
            not_loaded_ch_names,
            pos_data.SizeZ > 1,
            self.host.isSegm3D,
            favourite_funcs=favourite_funcs,
            allPos_acdc_df_cols=list(all_pos_acdc_df_cols),
            acdc_df_path=pos_data.images_path,
            posData=pos_data,
            addCombineMetricCallback=self.add_combine_metric,
            allPosData=self.host.data,
            parent=qparent,
            state=self.host.setMeasWinState,
        )
        self.host.measurementsWin.sigCancel.connect(
            self.set_measurements_cancelled
        )
        self.host.measurementsWin.sigClosed.connect(self.set_measurements)
        self.host.measurementsWin.show()

    def set_measurements_cancelled(self):
        self.host.measurementsWin = None

    def set_measurements(self):
        if self.host.measurementsWin.delExistingCols:
            self._remove_existing_unchecked_measurements()
        self.host.setMeasWinState = self.host.measurementsWin.state()
        self.host.logger.info('Setting measurements...')
        self._set_metrics(self.host.measurementsWin)
        self.host.logger.info('Metrics successfully set.')
        self.host.measurementsWin = None

    def add_custom_metric(self, checked=False):
        txt = self.custom_metrics_instructions()
        metrics_path = self.metrics_examples_path()
        msg = widgets.myMessageBox()
        msg.addShowInFileManagerButton(metrics_path, 'Show example...')
        title = 'Add custom metrics instructions'
        msg.information(self.host, title, txt, buttonsTexts=('Ok',))

    def add_combine_metric(self):
        pos_data = self.host.data[self.host.pos_i]
        is_zstack = pos_data.SizeZ > 1
        win = apps.combineMetricsEquationDialog(
            self.host.ch_names,
            is_zstack,
            self.host.isSegm3D,
            parent=self.host,
        )
        win.sigOk.connect(self.save_combine_metrics_to_pos_data)
        win.exec_()
        win.sigOk.disconnect()

    def save_combine_metrics_to_pos_data(self, window):
        for pos_data in self.host.data:
            equations_dict, is_mixed_channels = window.getEquationsDict()
            for new_col_name, equation in equations_dict.items():
                pos_data.addEquationCombineMetrics(
                    equation, new_col_name, is_mixed_channels
                )
                pos_data.saveCombineMetrics()

        if self.host.measurementsWin is None:
            return

        self.host.measurementsWinState = self.host.measurementsWin.state()
        self.host.measurementsWin.close()
        self.show_set_measurements()
        self.host.measurementsWin.restoreState(
            self.host.measurementsWinState
        )

    def set_metrics_func(self):
        pos_data = self.host.data[self.host.pos_i]
        self.host._measurements_kernel._set_metrics_func_from_posData(
            pos_data
        )

    def _set_metrics(self, measurements_win):
        self.host._measurements_kernel.set_metrics_from_set_measurements_dialog(
            measurements_win
        )
        for ch_name in self.host._measurements_kernel.chNamesToProcess:
            if ch_name not in self.host.notLoadedChNames:
                continue

            success = self.host.loadFluo_cb(fluo_channels=[ch_name])
            if not success:
                continue

    def _remove_existing_unchecked_measurements(self):
        self.host.logger.info('Removing existing unchecked measurements...')
        del_cols = self.host.measurementsWin.existingUncheckedColnames
        del_rps = self.host.measurementsWin.existingUncheckedRps
        self._log_removed_measurements(del_cols, del_rps)
        for pos_data in self.host.data:
            for data_dict in pos_data.allData_li:
                data_dict['acdc_df'] = (
                    self.drop_unchecked_measurements(
                        data_dict['acdc_df'],
                        del_cols,
                        del_rps,
                    )
                )

    def _log_removed_measurements(self, del_cols, del_rps):
        del_cols_format = [f'  *  {colname}' for colname in del_cols]
        del_rps_format = [f'  *  {colname}' for colname in del_rps]
        del_cols_format.extend(del_rps_format)
        del_cols_format = '\n'.join(del_cols_format)
        self.host.logger.info(del_cols_format)

    def _favourite_metric_functions(self):
        try:
            df_favourite_funcs = pd.read_csv(favourite_func_metrics_csv_path)
            return df_favourite_funcs['favourite_func_name'].to_list()
        except Exception:
            return None