import pprint

import pandas as pd

from .. import apps, myutils, workers, widgets, html_utils

from .base import NewThreadMultipleExpBaseUtil

MISSING_VALUE_TEXT = '<missing>'

def _normalize_metadata_value(value):
    """Normalize a value read from metadata.csv so that values like '1',
    1.0, and '1.0' are considered equal when comparing positions."""
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
        return str(value).strip()

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
        whose values differ (or are missing) in at least one file.
    """
    allKeys = []
    for values in scannedMetadata.values():
        for key in values.keys():
            if key not in allKeys:
                allKeys.append(key)

    differences = {}
    for key in allKeys:
        normalizedValues = {
            _normalize_metadata_value(values.get(key))
            for values in scannedMetadata.values()
        }
        if len(normalizedValues) == 1:
            continue

        differences[key] = {
            path: values.get(key, MISSING_VALUE_TEXT)
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

        additionalValues = {
            key: value for key, value in refValues.items()
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
