"""Setup dialog of premade measurements viewer configurations: maps channel
placeholders and typed inputs, with Calculate/Load of background values
(background_values.py, utils/computeBackgroundValues.py).
"""
import configparser
import html
import numpy as np
from qtpy.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QVBoxLayout
)
from cellacdc import apps, background_values, html_utils
from cellacdc import widgets
from . import predictor_editors as viewer_dialogs_predictor_editors


class ChannelPlaceholderMappingDialog(apps.QBaseDialog):
    """Map each channel placeholder of a premade configuration to a channel
    and enter the values of its input placeholders.

    `placeholders` maps placeholder names to descriptions; `channels` are the
    available channel names. `inputs` are the template's validated
    `input_placeholders` (see `MeasurementsViewerWin.validateInputPlaceholders`),
    each shown with its default value and an info button for its `info` text.
    `validate(mapping, values)` (optional, e.g. resolving the template)
    may raise ValueError to refuse a selection: its message is shown and
    the dialog stays open.
    `channelHints` (`{placeholder: [name parts]}`, optional) improve the
    preselected channels: each placeholder gets an unused channel whose name
    contains the placeholder name, else the first of its hints (in order,
    case-insensitive), else the first unused channel.
    `backgroundValues` is the template's validated `background_values`
    section (`MeasurementsViewerWin.validateBackgroundValuesSpec`): its
    linked inputs can be filled from a summary table of the "Compute
    background intensity" utility, loaded from a file (start folder
    `tableStartDir`) or computed by
    `computeBackgroundValues(channels, backgroundValues, parent=dialog)`,
    which returns the summary table (None if cancelled). The table row of
    each linked channel placeholder is the row of its selected channel
    (the user picks another row when the table has none). Changing the
    channel of a linked placeholder after filling refills its values from
    the same table. While the background values are computed or a table
    row is asked, the dialog is disabled.
    After `exec_`, `mapping` is `{placeholder: channel}` and `values` is
    `{input name: typed value}` unless `cancel` is True.
    """

    def __init__(
            self, placeholders, channels, parent=None, inputs=None,
            validate=None, backgroundValues=None,
            computeBackgroundValues=None, tableStartDir='', channelHints=None
        ):
        super().__init__(parent)
        self.cancel = True
        self.mapping = None
        self.values = None
        self.validate = validate
        self.inputs = dict(inputs or {})
        self.backgroundValues = backgroundValues
        self.computeBackgroundValues = computeBackgroundValues
        self.tableStartDir = tableStartDir
        self.backgroundSourceLabel = None
        # Last table used to fill the values: {'summary', 'source', 'mapping'}
        self.backgroundFill = None
        self._closed = False
        self.setWindowTitle('Set up premade configuration')
        channels = sorted(channels)
        layout = QVBoxLayout()
        layout.addWidget(QLabel(html_utils.paragraph(
            'This premade configuration uses placeholder channels.<br>'
            'Select the channel of your experiment for each of them:'
        )))
        grid = QGridLayout()
        self.comboboxes = {}
        guesses = self.guessChannels(placeholders, channels, channelHints)
        for row, (name, description) in enumerate(placeholders.items()):
            text = f'<b>{name}</b>'
            if description:
                text += f'<br><i>{description}</i>'
            label = QLabel(html_utils.paragraph(text))
            combobox = QComboBox()
            combobox.addItems(channels)
            combobox.setCurrentText(guesses[name])
            grid.addWidget(label, row, 0)
            grid.addWidget(combobox, row, 1)
            self.comboboxes[name] = combobox
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        self.inputLineEdits = {}
        if self.inputs:
            layout.addSpacing(10)
            layout.addWidget(QLabel(html_utils.paragraph(
                'Enter the values requested by the configuration:'
            )))
            inputsGrid = QGridLayout()
            for row, (name, spec) in enumerate(self.inputs.items()):
                text = f'<b>{name}</b>'
                if spec.get('description'):
                    text += f'<br><i>{spec["description"]}</i>'
                inputsGrid.addWidget(QLabel(html_utils.paragraph(text)), row, 0)
                lineEdit = QLineEdit(str(spec['default']))
                lineEdit.setToolTip(f'Type: {spec["type"]}')
                inputsGrid.addWidget(lineEdit, row, 1)
                if spec.get('info'):
                    infoButton = widgets.infoPushButton(
                        info_text=html_utils.paragraph(spec['info']),
                        info_title=name
                    )
                    inputsGrid.addWidget(infoButton, row, 2)
                self.inputLineEdits[name] = lineEdit
            inputsGrid.setColumnStretch(1, 1)
            layout.addLayout(inputsGrid)
            if self.backgroundValues is not None:
                layout.addLayout(self._backgroundValuesLayout())
        layout.addSpacing(10)
        buttons = widgets.CancelOkButtonsLayout()
        buttons.okButton.clicked.connect(self.ok_cb)
        buttons.cancelButton.clicked.connect(self.close)
        layout.addLayout(buttons)
        self.setLayout(layout)

    @staticmethod
    def guessChannels(placeholders, channels, channelHints=None):
        """Preselected channel of each placeholder: an unused channel whose
        name contains the placeholder name, else the one containing the
        first matching hint of `channelHints[placeholder]`, else the first
        unused channel (case-insensitive)."""
        channelHints = channelHints or {}
        guesses = dict.fromkeys(placeholders)

        def unused(ch):
            return ch not in guesses.values()

        # Placeholder names first (most specific), then the hints
        for name in placeholders:
            guesses[name] = next(
                (ch for ch in channels
                 if name.lower() in ch.lower() and unused(ch)),
                None
            )
        for name in placeholders:
            if guesses[name] is not None:
                continue
            for hint in channelHints.get(name, []):
                guesses[name] = next(
                    (ch for ch in channels
                     if hint.lower() in ch.lower() and unused(ch)),
                    None
                )
                if guesses[name] is not None:
                    break
        for name in placeholders:
            if guesses[name] is None:
                guesses[name] = next(
                    (ch for ch in channels if unused(ch)),
                    channels[0] if channels else ''
                )
        return guesses

    def currentMapping(self):
        return {
            name: combobox.currentText()
            for name, combobox in self.comboboxes.items()
        }

    def _backgroundValuesLayout(self):
        spec = self.backgroundValues
        layout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()
        if self.computeBackgroundValues is not None:
            computeButton = widgets.calcPushButton(
                'Calculate background values...'
            )
            computeButton.setToolTip(
                'Run the "Compute background intensity" utility on the '
                'Positions of\nthis experiment with the settings of this '
                f'configuration (background\n{spec["statistic"]}, '
                f'{spec["low_percentile"]:g}th and '
                f'{spec["high_percentile"]:g}th percentiles) and fill in '
                'the values'
            )
            computeButton.clicked.connect(self.computeBackgroundValuesClicked)
            buttonsLayout.addWidget(computeButton)
            self.computeBackgroundButton = computeButton
        loadButton = widgets.LoadPushButton(
            'Load calculated background values...'
        )
        loadButton.setToolTip(
            'Fill in the values from the summary '
            'table saved by the "Compute background\n'
            'intensity" utility (<name>_summary.csv in the experiment folder)'
        )
        loadButton.clicked.connect(self.loadBackgroundValuesTable)
        buttonsLayout.addWidget(loadButton)
        self.loadBackgroundTableButton = loadButton
        buttonsLayout.addStretch(1)
        layout.addLayout(buttonsLayout)
        self.backgroundSourceLabel = QLabel()
        self.backgroundSourceLabel.setWordWrap(True)
        self.backgroundSourceLabel.hide()
        layout.addWidget(self.backgroundSourceLabel)
        linked = {link['channel'] for link in spec['inputs'].values()}
        for placeholder in linked:
            self.comboboxes[placeholder].currentTextChanged.connect(
                self.linkedChannelChanged
            )
        return layout

    def linkedBackgroundMapping(self):
        mapping = self.currentMapping()
        return {
            link['channel']: mapping[link['channel']]
            for link in self.backgroundValues['inputs'].values()
        }

    def linkedChannelChanged(self, text):
        """Refill the values from the last table when the channel of a
        linked placeholder changes (they belong to the previous channel)."""
        fill = self.backgroundFill
        if fill is None or self.linkedBackgroundMapping() == fill['mapping']:
            return
        self.fillBackgroundValues(fill['summary'], fill['source'])

    def closeEvent(self, event):
        self._closed = True
        super().closeEvent(event)

    def linkedChannels(self):
        """Selected channels of the placeholders linked to background
        values (each once, in link order)."""
        mapping = self.currentMapping()
        return list(dict.fromkeys(
            mapping[link['channel']]
            for link in self.backgroundValues['inputs'].values()
        ))

    def _warnBackgroundValues(self, title, text):
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, title, html_utils.paragraph(text))

    def computeBackgroundValuesClicked(self):
        # The selection dialogs are not modal: keep this dialog from being
        # used (OK, Cancel, a second run) until the computation returns
        self.setEnabled(False)
        try:
            summary = self.computeBackgroundValues(
                self.linkedChannels(), self.backgroundValues, parent=self
            )
        finally:
            self.setEnabled(True)
        if summary is None or self._closed:
            return
        if summary.empty:
            self._warnBackgroundValues(
                'No background values',
                'No Position was processed, no value was filled in.'
            )
            return
        self.fillBackgroundValues(summary, 'the computed summary table')

    def loadBackgroundValuesTable(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Select summary table of the background values',
            self.tableStartDir, 'CSV files (*.csv);;All files (*)'
        )
        if not filepath:
            return
        try:
            summary = background_values.read_summary_table(filepath)
        except ValueError as err:
            self._warnBackgroundValues(
                'Invalid table',
                f'{err}<br><br>File: <code>{html.escape(filepath)}</code>'
            )
            return
        self.fillBackgroundValues(
            summary, f'<code>{html.escape(filepath)}</code>'
        )

    def askTableChannel(self, placeholder, channel, tableChannels):
        """Ask which row (channel) of the table to use for `placeholder`,
        mapped to `channel` that the table does not have. Returns the
        channel or None to skip its values."""
        win = widgets.QDialogListbox(
            'Select channel of the table',
            f'The table has no values for the channel "{channel}" '
            f'({placeholder}).\n'
            'Select the channel of the table to use instead:',
            tableChannels, multiSelection=False, parent=self,
            allowEmptySelection=False
        )
        win.exec_()
        if win.cancel or not win.selectedItemsText:
            return None
        return win.selectedItemsText[0]

    def fillBackgroundValues(self, summary, source):
        """Fill the inputs linked to background values from a summary table
        (`background_values.read_summary_table`); `source` (HTML) is shown
        in the dialog. Values that are missing (NaN) are left unchanged.
        Returns `{input name: value}` of the filled inputs."""
        spec = self.backgroundValues
        mapping = self.currentMapping()
        linkedMapping = self.linkedBackgroundMapping()
        rows = {
            str(row['channel']): row for _, row in summary.iterrows()
        }
        tableRows = {}
        # The row prompts are not modal: OK would accept the values before
        # they are filled in, and a channel changed meanwhile would get the
        # values of the previous one
        with viewer_dialogs_predictor_editors.disabled_while_nested(self):
            for link in spec['inputs'].values():
                placeholder = link['channel']
                if placeholder in tableRows:
                    continue
                channel = mapping[placeholder]
                if channel not in rows:
                    channel = self.askTableChannel(
                        placeholder, channel, list(rows)
                    )
                tableRows[placeholder] = channel
        if self._closed:
            return {}
        filled, empty, skipped = {}, [], []
        for name, link in spec['inputs'].items():
            channel = tableRows[link['channel']]
            if channel is None:
                skipped.append(name)
                continue
            try:
                value = float(rows[channel][link['value']])
            except (TypeError, ValueError):
                value = np.nan
            if not np.isfinite(value):
                empty.append(
                    f'<b>{html.escape(name)}</b> (<code>'
                    f'{html.escape(link["value"])}</code> of '
                    f'{html.escape(channel)})'
                )
                continue
            self.inputLineEdits[name].setText(f'{value:.10g}')
            filled[name] = value
        self.backgroundFill = {
            'summary': summary, 'source': source, 'mapping': linkedMapping,
        }
        notes = self._backgroundSettingsNotes(
            {ch: rows[ch] for ch in tableRows.values() if ch is not None},
            uses_percentiles=any(
                link['value'] != 'constant_background'
                for link in spec['inputs'].values()
            )
        )
        used = ', '.join(
            f'{html.escape(placeholder)} = {html.escape(channel)}'
            for placeholder, channel in tableRows.items()
            if channel is not None
        )
        self.backgroundSourceLabel.setText(html_utils.paragraph(
            f'Filled in {len(filled)} of {len(spec["inputs"])} values from '
            f'{source}' + (f' (table rows: {used})' if used else '') + '.',
            font_size='11px'
        ))
        self.backgroundSourceLabel.show()
        if empty or skipped or notes:
            items = []
            if empty:
                items.append(
                    'The table has <b>no value</b> for: '
                    f'{", ".join(empty)}. Min/max values are empty when no '
                    'acdc_output table with the mean intensity of the '
                    'channel was selected.'
                )
            if skipped:
                items.append(
                    'Not filled in (no table row selected): '
                    + ', '.join(f'<b>{html.escape(n)}</b>' for n in skipped)
                )
            items.extend(notes)
            self._warnBackgroundValues(
                'Background values',
                f'Filled in {len(filled)} of {len(spec["inputs"])} values. '
                'The other values are left unchanged.'
                + html_utils.to_list(items)
            )
        return filled

    def _backgroundSettingsNotes(self, rows, uses_percentiles=True):
        """Differences between the settings of the table rows and of the
        configuration (statistic, percentiles and ignored zero pixels)."""
        spec = self.backgroundValues
        notes = []
        for channel, row in rows.items():
            differences = []
            statistic = str(row['background_statistic'])
            if statistic != spec['statistic']:
                differences.append(
                    f'background {html.escape(statistic)} '
                    f'(configuration: {spec["statistic"]})'
                )
            low, high = row['low_percentile'], row['high_percentile']
            if uses_percentiles and not (
                    np.isclose(low, spec['low_percentile'])
                    and np.isclose(high, spec['high_percentile'])):
                differences.append(
                    f'{low:g}th/{high:g}th percentiles (configuration: '
                    f'{spec["low_percentile"]:g}th/'
                    f'{spec["high_percentile"]:g}th)'
                )
            excludeZero = configparser.ConfigParser.BOOLEAN_STATES.get(
                str(row.get('exclude_zero_pixels', '')).strip().lower()
            )
            if (excludeZero is not None
                    and excludeZero != spec['exclude_zero_pixels']):
                ignored = 'ignored' if excludeZero else 'included'
                expected = (
                    'ignored' if spec['exclude_zero_pixels'] else 'included'
                )
                differences.append(
                    f'pixels with intensity 0 {ignored} '
                    f'(configuration: {expected})'
                )
            if differences:
                notes.append(
                    f'The values of {html.escape(str(channel))} were computed '
                    f'with {" and ".join(differences)}.'
                )
        return notes

    @staticmethod
    def parseInput(text, kind):
        """Convert the entered `text` to `kind` ('float', 'int' or 'str')."""
        if kind == 'str':
            return text
        text = text.strip()
        if kind == 'int':
            return int(text)
        value = float(text)
        if not np.isfinite(value):
            raise ValueError('not a finite number')
        return value

    def currentValues(self):
        """Return `{name: typed value}`; raise ValueError on invalid input."""
        values = {}
        for name, lineEdit in self.inputLineEdits.items():
            kind = self.inputs[name]['type']
            try:
                values[name] = self.parseInput(lineEdit.text(), kind)
            except ValueError:
                raise ValueError(
                    f'"{html.escape(lineEdit.text())}" is not a valid {kind} '
                    f'for <b>{html.escape(name)}</b>.'
                )
        return values

    def ok_cb(self):
        try:
            inputValues = self.currentValues()
        except ValueError as err:
            msg = widgets.myMessageBox(wrapText=False)
            msg.warning(
                self, 'Invalid value', html_utils.paragraph(str(err))
            )
            return
        mapping = self.currentMapping()
        values = list(mapping.values())
        duplicated = sorted({ch for ch in values if values.count(ch) > 1})
        if duplicated:
            msg = widgets.myMessageBox(wrapText=False)
            msg.warning(
                self, 'Channel used twice',
                html_utils.paragraph(
                    'Each placeholder must be mapped to a different channel. '
                    'Used more than once: '
                    f'<b>{html.escape(", ".join(duplicated))}</b>'
                )
            )
            return
        if self.validate is not None:
            try:
                self.validate(mapping, inputValues)
            except ValueError as err:
                msg = widgets.myMessageBox(wrapText=False)
                msg.warning(
                    self, 'Invalid selection',
                    html_utils.paragraph(html.escape(str(err)))
                )
                return
        self.mapping = mapping
        self.values = inputValues
        self.cancel = False
        self.close()
