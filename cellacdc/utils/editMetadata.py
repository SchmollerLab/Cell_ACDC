import pprint

import pandas as pd

from .. import apps, myutils, workers, widgets, html_utils

from .base import NewThreadMultipleExpBaseUtil
import os
import traceback
from qtpy.QtCore import Signal
from cellacdc import load
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base

MISSING_VALUE_TEXT = '<missing>'
EMPTY_VALUE_TEXT = '<empty>'

# Sentinel for keys that are absent from a metadata.csv file. It must be
# different from the normalized empty value (None) so that a missing field
# and a present-but-empty field are reported as a difference.
_MISSING_KEY = object()

def _normalize_metadata_value(value):
    """Normalize a value read from metadata.csv so that values like '1',
    1.0, and '1.0' are considered equal when comparing positions.
    Empty values (None, NaN, or blank strings) are normalized to None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        value = str(value).strip()
        return value if value else None

def _get_normalized_value(values, key):
    if key not in values:
        return _MISSING_KEY
    return _normalize_metadata_value(values[key])

def _get_display_value(values, key):
    if key not in values:
        return MISSING_VALUE_TEXT
    if _normalize_metadata_value(values[key]) is None:
        return EMPTY_VALUE_TEXT
    return values[key]

def get_metadata_differences(scannedMetadata):
    """Find the metadata keys whose values are not the same in all files.

    Parameters
    ----------
    scannedMetadata : dict
        Mapping of {metadata_csv_path: {key: value}}

    Returns
    -------
    dict
        Mapping of {key: {metadata_csv_path: value}} only for the keys
        whose values differ (or are missing) in at least one file. Missing
        keys are reported as `MISSING_VALUE_TEXT` and empty values as
        `EMPTY_VALUE_TEXT`.
    """
    allKeys = []
    for values in scannedMetadata.values():
        for key in values.keys():
            if key not in allKeys:
                allKeys.append(key)

    differences = {}
    for key in allKeys:
        normalizedValues = {
            _get_normalized_value(values, key)
            for values in scannedMetadata.values()
        }
        if len(normalizedValues) == 1:
            continue

        differences[key] = {
            path: _get_display_value(values, key)
            for path, values in scannedMetadata.items()
        }
    return differences

def _display_key(key):
    if key.startswith('__'):
        return f'{key[2:]} (custom field)'
    return key

def _to_int(value, default=1):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default

def _to_float(value, default=1.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

class EditMetadataUtil(NewThreadMultipleExpBaseUtil):
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

    def runWorker(self):
        self.worker = workers.EditMetadataWorker(self)
        self.worker.sigAskEditMetadata.connect(self.askEditMetadata)
        super().runWorker(self.worker)

    def showEvent(self, event):
        self.runWorker()

    def _abortWorker(self):
        self.worker.abort = True
        self.worker.waitCond.wakeAll()

    def warnNoMetadataFound(self, missingMetadataPaths):
        txt = html_utils.paragraph(
            'None of the selected Positions contain a '
            '<code>metadata.csv</code> file.<br><br>'
            'There is nothing to edit.'
        )
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.setDetailedText('\n'.join(missingMetadataPaths))
        msg.warning(self, 'Metadata not found', txt)

    def warnMetadataDifferences(
            self, differences, missingMetadataPaths, numFiles
        ):
        txt = ''
        detailsLines = []
        if differences:
            differentKeys = html_utils.to_list(
                [_display_key(key) for key in differences.keys()]
            )
            txt = (
                'The <b>metadata is NOT the same</b> in all the selected '
                f'Positions ({numFiles} metadata files).<br><br>'
                'The following fields have different values:'
                f'{differentKeys}'
            )
            for key, pathsValues in differences.items():
                detailsLines.append(f'{_display_key(key)}:')
                for path, value in pathsValues.items():
                    detailsLines.append(f'  - {value}  -->  {path}')
                detailsLines.append('')

        if missingMetadataPaths:
            if txt:
                txt = f'{txt}<br>'
            txt = (
                f'{txt}<b>{len(missingMetadataPaths)} Position(s) do not '
                'have a <code>metadata.csv</code> file</b> and will be '
                'skipped.<br><br>'
            )
            detailsLines.append('Positions without metadata.csv file:')
            detailsLines.extend(
                [f'  - {path}' for path in missingMetadataPaths]
            )

        txt = (
            f'{txt}'
            'You can still <b>apply the same metadata to all</b> the '
            'Positions.<br><br>'
            'The values shown in the next dialogue are taken from the '
            'first Position:<br><br>'
            f'<copiable>{self._referencePath}</copiable><br><br>'
            'See the details below for the values in each file.<br><br>'
            'Do you want to continue?'
        )
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.setDetailedText('\n'.join(detailsLines))
        msg.warning(
            self, 'Metadata is not the same in all Positions',
            html_utils.paragraph(txt),
            buttonsTexts=('Cancel', 'Continue')
        )
        return not msg.cancel

    def askEditMetadata(self, scannedMetadata, missingMetadataPaths):
        if not scannedMetadata:
            self.warnNoMetadataFound(missingMetadataPaths)
            self._abortWorker()
            return

        # Reference values are taken from the first scanned Position
        self._referencePath = list(scannedMetadata.keys())[0]
        refValues = scannedMetadata[self._referencePath]

        differences = get_metadata_differences(scannedMetadata)
        if differences or missingMetadataPaths:
            self.logger.info(
                'Metadata is not the same in all selected Positions. '
                f'Differences:\n{pprint.pformat(differences)}'
            )
            proceed = self.warnMetadataDifferences(
                differences, missingMetadataPaths, len(scannedMetadata)
            )
            if not proceed:
                self._abortWorker()
                return

        SizeT_metadata = None
        if 'SizeT' in refValues:
            SizeT_metadata = _to_int(refValues['SizeT'])
        SizeZ_metadata = None
        if 'SizeZ' in refValues:
            SizeZ_metadata = _to_int(refValues['SizeZ'])

        # Empty custom fields are read as NaN by pandas --> show them as
        # empty text instead of 'nan' in the dialog
        additionalValues = {
            key: '' if _normalize_metadata_value(value) is None else value
            for key, value in refValues.items()
            if key.startswith('__')
        }

        win = apps.QDialogMetadata(
            _to_int(refValues.get('SizeT')),
            _to_int(refValues.get('SizeZ')),
            _to_float(refValues.get('TimeIncrement')),
            _to_float(refValues.get('PhysicalSizeZ')),
            _to_float(refValues.get('PhysicalSizeY')),
            _to_float(refValues.get('PhysicalSizeX')),
            True, True, True,
            parent=self,
            askSegm3D=False,
            additionalValues=additionalValues,
            SizeT_metadata=SizeT_metadata,
            SizeZ_metadata=SizeZ_metadata
        )
        # Every selected Position is always updated --> only one
        # "apply" button is needed
        win.okButton.setText(
            f'Apply to all selected Positions ({len(scannedMetadata)})'
        )
        win.okButton.setToolTip(
            'Save the metadata to the metadata.csv file of all the '
            'selected Positions'
        )
        if win.okAllButton is not None:
            win.okAllButton.hide()
        if win.selectButton is not None:
            win.selectButton.hide()
        win.exec_()
        if win.cancel:
            self._abortWorker()
            return

        # Ignore custom fields without a name
        newAdditionalValues = {
            key: value for key, value in win._additionalValues.items()
            if len(key) > 2
        }
        self.worker.newMetadata = {
            'SizeT': win.SizeT,
            'SizeZ': win.SizeZ,
            'TimeIncrement': win.TimeIncrement,
            'PhysicalSizeZ': win.PhysicalSizeZ,
            'PhysicalSizeY': win.PhysicalSizeY,
            'PhysicalSizeX': win.PhysicalSizeX,
            'additionalValues': newAdditionalValues
        }
        self.logger.info(
            'Metadata to apply to all selected Positions:\n'
            f'{pprint.pformat(self.worker.newMetadata)}'
        )
        self.worker.waitCond.wakeAll()

    def workerFinished(self, worker):
        aborted = worker is None or worker.abort
        if aborted:
            self.logger.info('Editing metadata cancelled.')
            txt = 'Editing metadata cancelled.'
        else:
            numEditedFiles = worker.numEditedFiles
            self.logger.info(
                f'Editing metadata completed ({numEditedFiles} files updated).'
            )
            txt = (
                'Editing metadata completed.<br><br>'
                f'Number of updated metadata files: <b>{numEditedFiles}</b>'
            )
        super().workerFinished(worker)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process cancelled', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        self.close()


class EditMetadataWorker(utils_base.BaseWorkerUtil):
    sigAskEditMetadata = Signal(object, object)

    def __init__(self, mainWin):
        super().__init__(mainWin)
        # Set by the main thread in the slot connected to sigAskEditMetadata
        self.newMetadata = None
        self.numEditedFiles = 0

    def _scanMetadata(self, expPaths):
        scannedMetadata = {}
        missingMetadataPaths = []
        tot_pos = sum(len(pos_foldernames) for pos_foldernames in expPaths.values())
        self.signals.initProgressBar.emit(tot_pos)
        self.signals.sigUpdatePbarDesc.emit('Scanning metadata files...')
        for exp_path, pos_foldernames in expPaths.items():
            for pos in pos_foldernames:
                if self.abort:
                    return None, None

                images_path = os.path.join(exp_path, pos, 'Images')
                metadata_csv_path = load.get_metadata_csv_filepath(images_path)
                if metadata_csv_path is None:
                    self.logger.log(
                        f'[WARNING]: metadata.csv file not found in '
                        f'"{images_path}"'
                    )
                    missingMetadataPaths.append(images_path)
                    self.signals.progressBar.emit(1)
                    continue

                self.logger.log(f'Reading "{metadata_csv_path}"...')
                df_metadata = load.read_metadata_csv(metadata_csv_path)
                scannedMetadata[metadata_csv_path] = (
                    load.get_editable_metadata_values(df_metadata)
                )
                self.signals.progressBar.emit(1)

        return scannedMetadata, missingMetadataPaths

    def _saveMetadataCsv(self, df_metadata, metadata_csv_path):
        while True:
            try:
                df_metadata.to_csv(metadata_csv_path)
                return
            except PermissionError:
                traceback_str = traceback.format_exc()
                self.mutex.lock()
                self.signals.sigPermissionError.emit(
                    traceback_str, metadata_csv_path
                )
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()

    def _updateMetadataCsv(self, metadata_csv_path, newMetadata):
        df_metadata = load.read_metadata_csv(metadata_csv_path)
        for key in load.EDITABLE_METADATA_KEYS:
            df_metadata.at[key, 'values'] = newMetadata[key]

        additionalValues = newMetadata['additionalValues']
        for key, value in additionalValues.items():
            df_metadata.at[key, 'values'] = value

        # Remove custom fields that the user removed from the dialog
        idx_to_drop = [
            key for key in df_metadata.index
            if key.startswith('__') and key not in additionalValues
        ]
        df_metadata = df_metadata.drop(idx_to_drop)

        self._saveMetadataCsv(df_metadata, metadata_csv_path)

    @workers_base.worker_exception_handler
    def run(self):
        expPaths = self.mainWin.expPaths

        # 1. Scan metadata files of all selected Positions
        scannedMetadata, missingMetadataPaths = self._scanMetadata(expPaths)
        if self.abort:
            self.signals.finished.emit(self)
            return

        # 2. Ask (in main thread) the new metadata to apply
        self.mutex.lock()
        self.sigAskEditMetadata.emit(scannedMetadata, missingMetadataPaths)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        if self.abort or self.newMetadata is None:
            self.abort = True
            self.signals.finished.emit(self)
            return

        # 3. Write new metadata to every metadata.csv file
        self.signals.initProgressBar.emit(len(scannedMetadata))
        self.signals.sigUpdatePbarDesc.emit('Saving metadata files...')
        for metadata_csv_path in scannedMetadata.keys():
            if self.abort:
                self.signals.finished.emit(self)
                return

            self.logger.log(f'Updating "{metadata_csv_path}"...')
            self._updateMetadataCsv(metadata_csv_path, self.newMetadata)
            self.numEditedFiles += 1
            self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)
