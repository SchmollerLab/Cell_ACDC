"""View adapter for measurement setup and dialogs."""

from __future__ import annotations

import pandas as pd

from cellacdc import apps, cli, favourite_func_metrics_csv_path, widgets


class Measurements:
    """Extracted from guiWin."""

    def _setMetrics(self, measurementsWin):
        self._measurements_kernel.set_metrics_from_set_measurements_dialog(
            measurementsWin
        )
        for ch in self._measurements_kernel.chNamesToProcess:
            if ch not in self.notLoadedChNames:
                continue
            
            success = self.loadFluo_cb(fluo_channels=[ch])
            if not success:
                continue

    def addCombineMetric(self):
        posData = self.data[self.pos_i]
        isZstack = posData.SizeZ > 1
        win = apps.combineMetricsEquationDialog(
            self.ch_names, isZstack, self.isSegm3D, parent=self
        )
        win.sigOk.connect(self.saveCombineMetricsToPosData)
        win.exec_()
        win.sigOk.disconnect()

    def addCustomMetric(self, checked=False):
        txt = measurements.add_metrics_instructions()
        metrics_path = measurements.metrics_path
        msg = widgets.myMessageBox()
        msg.addShowInFileManagerButton(metrics_path, 'Show example...')
        title = 'Add custom metrics instructions'
        msg.information(self, title, txt, buttonsTexts=('Ok',))

    def initMetricsToSave(self, posData):
        self._measurements_kernel._init_metrics_to_save(posData)

    def initMetrics(self):
        self.logger.info('Initializing measurements...')
        posData = self.data[self.pos_i]
        self._measurements_kernel = cli.ComputeMeasurementsKernel(
            self.logger, self.log_path, False
        )
        self._measurements_kernel.init_args(
            posData.chNames, posData.getSegmEndname()
        )
        self._measurements_kernel._init_metrics(posData, self.isSegm3D)

    def showSetMeasurements(self, checked=False, qparent=None):
        qparent = qparent if qparent is not None else self
        if self.measurementsWin is not None:
            self.measurementsWin.show()
            self.measurementsWin.raise_()
            self.measurementsWin.activateWindow()
            return

        try:
            df_favourite_funcs = pd.read_csv(favourite_func_metrics_csv_path)
            favourite_funcs = df_favourite_funcs['favourite_func_name'].to_list()
        except Exception as e:
            favourite_funcs = None

        posData = self.data[self.pos_i]
        allPos_acdc_df_cols = set()
        for _posData in self.data:
            for frame_i, data_dict in enumerate(_posData.allData_li):
                acdc_df = data_dict['acdc_df']
                if acdc_df is None:
                    continue
                
                allPos_acdc_df_cols.update(acdc_df.columns)
        loadedChNames = posData.setLoadedChannelNames(returnList=True)
        posData.fluo_data_dict.pop(self.user_ch_name, None)
        if self.user_ch_name not in loadedChNames:
            loadedChNames.insert(0, self.user_ch_name)
        notLoadedChNames = [c for c in self.ch_names if c not in loadedChNames]
        self.notLoadedChNames = notLoadedChNames
        self.measurementsWin = apps.SetMeasurementsDialog(
            loadedChNames, notLoadedChNames, posData.SizeZ > 1, self.isSegm3D,
            favourite_funcs=favourite_funcs, 
            allPos_acdc_df_cols=list(allPos_acdc_df_cols),
            acdc_df_path=posData.images_path, posData=posData,
            addCombineMetricCallback=self.addCombineMetric,
            allPosData=self.data, 
            parent=qparent, 
            state=self.setMeasWinState
        )
        self.measurementsWin.sigCancel.connect(self.setMeasurementsCancelled)
        self.measurementsWin.sigClosed.connect(self.setMeasurements)
        self.measurementsWin.show()

    def setMeasurementsCancelled(self):
        self.measurementsWin = None

    def setMeasurements(self):
        posData = self.data[self.pos_i]
        if self.measurementsWin.delExistingCols:
            self.logger.info('Removing existing unchecked measurements...')
            delCols = self.measurementsWin.existingUncheckedColnames
            delRps = self.measurementsWin.existingUncheckedRps
            delCols_format = [f'  *  {colname}' for colname in delCols]
            delRps_format = [f'  *  {colname}' for colname in delRps]
            delCols_format.extend(delRps_format)
            delCols_format = '\n'.join(delCols_format)
            self.logger.info(delCols_format)
            for _posData in self.data:
                for frame_i, data_dict in enumerate(_posData.allData_li):
                    acdc_df = data_dict['acdc_df']
                    if acdc_df is None:
                        continue
                    
                    acdc_df = acdc_df.drop(columns=delCols, errors='ignore')
                    for col_rp in delRps:
                        drop_df_rp = acdc_df.filter(regex=fr'{col_rp}.*', axis=1)
                        drop_cols_rp = drop_df_rp.columns
                        acdc_df = acdc_df.drop(columns=drop_cols_rp, errors='ignore')
                    _posData.allData_li[frame_i]['acdc_df'] = acdc_df
        self.setMeasWinState = self.measurementsWin.state()
        self.logger.info('Setting measurements...')
        self._setMetrics(self.measurementsWin)
        self.logger.info('Metrics successfully set.')
        self.measurementsWin = None

    def saveCombineMetricsToPosData(self, window):
        for posData in self.data:
            equationsDict, isMixedChannels = window.getEquationsDict()
            for newColName, equation in equationsDict.items():
                posData.addEquationCombineMetrics(
                    equation, newColName, isMixedChannels
                )
                posData.saveCombineMetrics()
        
        if self.measurementsWin is None:
            return
        
        self.measurementsWinState = self.measurementsWin.state()
        self.measurementsWin.close()
        self.showSetMeasurements()
        self.measurementsWin.restoreState(self.measurementsWinState)

    def setMetricsFunc(self):
        posData = self.data[self.pos_i]
        self._measurements_kernel._set_metrics_func_from_posData(posData)
