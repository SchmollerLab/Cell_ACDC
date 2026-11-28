"""Measurements viewer configurations (ViewerConfigurationGuiElements):
saving, selecting and loading them, premade templates and mapping their
input onto experiments. Defines the configuration folder path, the
premade templates folder path
(measurements_viewer/premade_configurations/) and the premade tag. Spec:
measurements_viewer/__init__.py.
"""
import os
import copy
import json
from datetime import datetime
from functools import partial
from qtpy.QtWidgets import QFileDialog, QApplication
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import widgets
from cellacdc import cellacdc_path
from cellacdc import settings_folderpath
from cellacdc import apps
from cellacdc import html_utils
from cellacdc.apps_measurements_viewer import ChannelPlaceholderMappingDialog
from cellacdc.timepoint_predictors import TimepointIdentifier
from cellacdc.measurements_viewer_utils import (
    VISITED_COLUMN, get_images_paths, get_channels_and_segm_endnames,
    analysis_last_frame, blank_unanalyzed_outputs, flag_column_conflicts,
    flag_column_conflicts_text, load_display_positions
)


MEASUREMENTS_CONFIGURATIONS_PATH = os.path.join(
    settings_folderpath, 'measurements_viewer_configurations'
)
PREMADE_CONFIGURATIONS_PATH = os.path.join(
    cellacdc_path, 'measurements_viewer', 'premade_configurations'
)
PREMADE_TAG = '[Premade] '


class ViewerConfigurationGuiElements:
    """The viewer configuration (JSON): building, saving and selecting it,
    premade templates (channel mapping, background summary, copy into
    the settings folder), mapping its input onto an experiment and the
    configuration loading flow. The widget-free validation lives in
    measurements_viewer/viewer_utils.py.
    """

    def selectConfigurationFilepath(self):
        from cellacdc.apps import QTreeDialog

        paths = {}
        types = {}
        for folder, prefix, kind in (
                (MEASUREMENTS_CONFIGURATIONS_PATH, '', 'Saved'),
                (PREMADE_CONFIGURATIONS_PATH, PREMADE_TAG, 'Premade')):
            if not os.path.isdir(folder):
                continue
            for filename in sorted(os.listdir(folder)):
                filepath = os.path.join(folder, filename)
                if filename.lower().endswith('.json') and os.path.isfile(filepath):
                    paths[f'{prefix}{filename}'] = filepath
                    types[f'{prefix}{filename}'] = kind

        browseButton = widgets.browseFileButton(
            'Browse configuration...', self,
            ext={'JSON': ['.json']},
            title='Load measurements viewer configuration',
            start_dir=MEASUREMENTS_CONFIGURATIONS_PATH,
        )
        skipButton = widgets.SkipPushButton('Skip/New configuration', self)
        dialog = QTreeDialog(
            [
                (name, datetime.fromtimestamp(os.path.getmtime(path)).strftime(
                    '%Y/%m/%d %H:%M'
                ), types[name])
                for name, path in paths.items()
            ],
            headerLabels=['Name', 'Date Modified', 'Type'],
            title='Load measurements viewer configuration',
            infoText='Select a configuration to load, or skip loading:',
            parent=self,
            path_to_browse=MEASUREMENTS_CONFIGURATIONS_PATH if paths else None,
            additional_buttons=(browseButton,skipButton),
        )
        dialog.treeWidget.setCurrentItem(dialog.treeWidget.topLevelItem(0))
        browsed = []

        def selected(path):
            if path:
                browsed.append(path)
                dialog.close()

        browseButton.sigPathSelected.connect(selected)
        skipButton.clicked.connect(lambda: setattr(dialog, 'cancel', True) or dialog.close())
        dialog.exec_()
        if browsed:
            return browsed[0]
        if dialog.cancel:
            return None
        return paths[dialog.selectedText]

    def configuration(self):
        posData = self.posData()
        segm_filename = os.path.splitext(
            os.path.basename(posData.segm_npz_path)
        )[0]
        segm_endname = segm_filename[len(posData.basename):].lstrip('_')
        # The overlay is turned off only while a zy/zx plane is viewed
        # (`setOverlayEnabledForPlane`)
        overlay_channels = self.activeOverlayChannels()
        visible_channels = [
            channel for channel, button in self.allOverlayToolbuttons.items()
            if button.isChecked()
            and (channel == self.user_ch_name or channel in overlay_channels)
        ] or [self.user_ch_name]
        currentInput = {
            'images_paths': [
                os.path.abspath(pos.images_path) for pos in self.data
            ],
            'channel': self.user_ch_name,
            'segm_endname': segm_endname,
            'overlay_channels': overlay_channels,
            'visible_channels': visible_channels,
        }
        inputs = copy.deepcopy(self.configurationInputs)
        # Same experiment (same Images folders, channel and segmentation),
        # possibly with another channel order or other overlay/visible
        # channels: update that option in place instead of adding a
        # duplicate. An option equal up to channel order is preferred, then
        # the loaded option (several options may share the experiment).
        currentKey = self.inputOptionKey(currentInput)
        currentIdentity = self.inputIdentityKey(currentInput)
        loadedKey = self.loadedInputKey
        matches = [
            i for i, option in enumerate(inputs)
            if self.inputOptionKey(option) == currentKey
        ] or [
            i for i, option in enumerate(inputs)
            if loadedKey is not None
            and self.inputOptionKey(option) == loadedKey
            and self.inputIdentityKey(option) == currentIdentity
        ] or [
            i for i, option in enumerate(inputs)
            if self.inputIdentityKey(option) == currentIdentity
        ]
        if matches:
            inputs[matches[0]] = copy.deepcopy(currentInput)
        else:
            inputs.append(currentInput)
        timeCourseColumns = (
            self.checkedColumns() if self._measurementsMode == 0
            else sorted(self._timeCourseColumns)
        )
        return {
            'version': 2,
            **currentInput,
            'input_options': inputs,
            'measurement_processing': copy.deepcopy(self.measurementProcessingSteps),
            'combined_measurements': copy.deepcopy(self.combinedMeasurementSteps),
            'measurements_mode': self.measurementsModeCombobox.currentIndex(),
            'time_course_columns': timeCourseColumns,
            'time_course_y_axes': self.timeCourseYAxesConfiguration(
                timeCourseColumns
            ),
            'measurements_axes': self.measurementsAxes.copy(),
            'normalize_measurements': self.normalizeCheckbox.isChecked(),
            'measurements_colors': {
                col: pg.mkColor(color).name()
                for col, color in self.measurementsColors.items()
            },
            'conditional_formatting': copy.deepcopy(self.conditionalFormatting),
            'timepoint_identifiers': [
                identifier.to_dict() for identifier in self.timepointIdentifiers
            ],
            'custom_annotations': self.customAnnotConfiguration(),
            'phase_translator_rules': copy.deepcopy(self.phaseTranslatorRules),
            'phase_colors': dict(self.phaseColors),
            'auto_apply_phases': self.autoApplyPhases,
            'mask_unanalyzed_frames': self.maskUnanalyzedFrames,
            'measurement_steps_last_frames': dict(
                self.measurementStepsLastFrames
            ),
            # 'cell_constants': [dict(constant) for constant in self.cellConstants],
        }

    def configurationError(self, operation, error):
        self.logger.error('Could not %s configuration: %s', operation, error)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Configuration failed',
            f'Could not {operation} configuration:\n\n{error}'
        )

    def saveConfiguration(self, checked=False, save_as=False):
        if self.measurementProcessingThread is not None:
            self.configurationError('save', 'Wait for measurement processing to finish.')
            return False
        if not self.isDataLoaded:
            self.configurationError('save', 'Load image data before saving.')
            return False
        filepath = self.configurationFilepath
        if save_as or filepath is None:
            try:
                os.makedirs(MEASUREMENTS_CONFIGURATIONS_PATH, exist_ok=True)
            except OSError as err:
                self.configurationError('save', err)
                return False
            current_files = sorted(os.listdir(MEASUREMENTS_CONFIGURATIONS_PATH))
            helpText = html_utils.paragraph(
                '''Select a filename for the measurements viewer configuration.
                Which allows you to later reload it, preserving displayed measurements.'''
                )
            date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            msg = apps.filenameDialog(
                ext='json', title='Save measurements viewer configuration as',
                parent=self, helpText=helpText,existingNames=current_files,
                allowEmpty=False, addDoNotSaveButton=True, defaultEntry=date
            )
            msg.exec_()
            if msg.cancel:
                return False
            if msg.doNotSave:
                return True
            filename = msg.filename
            if not filename:
                return False
            filepath = os.path.join(MEASUREMENTS_CONFIGURATIONS_PATH, filename)
            if not filepath.lower().endswith('.json'):
                filepath += '.json'
            # fileNameDialog handles overwrite confirmation internally
        try:
            configuration = self.configuration()
            self.validateConfiguration(configuration)
            self.writeConfigurationFile(filepath, configuration)
        except (OSError, ValueError, TypeError, AttributeError) as err:
            self.configurationError('save', err)
            return False
        self.configurationFilepath = filepath
        self.configurationInputs = copy.deepcopy(configuration['input_options'])
        self.loadedInputKey = self.inputOptionKey(configuration)
        self.savedConfiguration = configuration
        self.logger.info(f'Configuration saved to "{filepath}"')
        # Possibly a new file name
        self.setStatusBarLabel(log=False)
        self.statusBar().showMessage(f'Configuration saved to "{filepath}".', 5000)
        return True

    def saveConfigurationAs(self, checked=False):
        return self.saveConfiguration(save_as=True)

    def selectChannelMapping(
            self, placeholders, channels, inputs=None, validate=None,
            backgroundValues=None, experimentPositions=None,
            channelHints=None
        ):
        """Return `(channel mapping, input values)` or None if cancelled.
        `validate(mapping, values)` may raise ValueError to refuse a
        selection (message shown, dialog stays open). With the template's
        validated `backgroundValues` section, the linked inputs can be
        filled by running the background utility on the Positions of
        `experimentPositions` (`{exp_path: [pos_foldername, ...]}`, see
        `computeBackgroundSummary`) or from a saved summary table."""
        experimentPositions = experimentPositions or {}
        compute = None
        if backgroundValues is not None:
            compute = partial(
                self.computeBackgroundSummary, experimentPositions
            )
        dialog = ChannelPlaceholderMappingDialog(
            placeholders, channels, parent=self, inputs=inputs,
            validate=validate, backgroundValues=backgroundValues,
            computeBackgroundValues=compute,
            tableStartDir=next(iter(experimentPositions), ''),
            channelHints=channelHints
        )
        dialog.exec_()
        if dialog.cancel:
            return None
        return dialog.mapping, dialog.values

    def computeBackgroundSummary(
            self, experimentPositions, channels, settings, parent=None
        ):
        """Run the "Compute background intensity" utility on the
        experiments of `experimentPositions` (`{exp_path: [preselected
        Positions]}`, the experiment selected for the premade configuration)
        with `channels` and `settings` (validated `background_values`
        section of a premade template) preset in its parameters dialog,
        which is still shown. Only the Positions are asked (not the
        experiments). Returns the summary table (empty if no Position was
        processed) or None if cancelled."""
        from cellacdc.utils import computeBackgroundValues as utilsBackground
        parent = parent or self
        expPaths = utilsBackground.select_positions(
            parent, experimentPositions
        )
        if not expPaths:
            return None
        presetParams = {
            'channels': list(channels),
            'statistic': settings['statistic'],
            'exclude_zero_pixels': settings['exclude_zero_pixels'],
            'use_acdc_output': True,
            'low_percentile': settings['low_percentile'],
            'high_percentile': settings['high_percentile'],
        }
        win = utilsBackground.ComputeBackgroundValuesUtil(
            expPaths, QApplication.instance(), 'Compute background intensity',
            'Launching compute background intensity process...',
            'Computing background intensity', parent=parent,
            presetParams=presetParams,
            presetNote=(
                'The settings are preset by the configuration that is being '
                'set up. After the computation, its normalization values '
                'are filled in from the summary table.'
            )
        )
        # Keep a reference: the worker thread may still be finishing when
        # the dialog closes
        self.backgroundValuesUtilWin = win
        win.exec_()
        return win.summaryDf

    def selectPremadeBaseChannel(self, channels, default=None):
        """Ask which channel to display as base layer (default listed first)."""
        items = list(channels)
        if default in items:
            items.remove(default)
            items.insert(0, default)
        return self.selectItem(
            'Select channel',
            'Select channel to display (base layer).\n'
            'The mapped channels that are not displayed as base are '
            'added as overlays:', items
        )

    def askPremadeCopyFilename(self, defaultName):
        """Return the filepath for the copy of a premade configuration."""
        os.makedirs(MEASUREMENTS_CONFIGURATIONS_PATH, exist_ok=True)
        current_files = sorted(os.listdir(MEASUREMENTS_CONFIGURATIONS_PATH))
        helpText = html_utils.paragraph(
            'The premade configuration with your channels is saved as a new '
            'configuration, which you can later load and modify.'
        )
        msg = apps.filenameDialog(
            ext='json', title='Save configuration from premade template',
            parent=self, helpText=helpText, existingNames=current_files,
            allowEmpty=False, defaultEntry=defaultName
        )
        msg.exec_()
        if msg.cancel or not msg.filename:
            return None
        filepath = os.path.join(MEASUREMENTS_CONFIGURATIONS_PATH, msg.filename)
        if not filepath.lower().endswith('.json'):
            filepath += '.json'
        # fileNameDialog handles overwrite confirmation internally
        return filepath

    def instantiatePremadeConfiguration(self, configuration, source):
        """Create a configuration from a premade template.

        Asks for an experiment folder and a channel for each placeholder,
        replaces the `{NAME}` tokens, resolves the experiment input and writes
        the result to the configurations folder. Returns
        `(filepath, configuration, selectedInput)` or None if cancelled.
        """
        placeholders = configuration.get('channel_placeholders')
        self.validateChannelPlaceholders(placeholders)
        inputs = configuration.get('input_placeholders', {})
        self.validateInputPlaceholders(inputs, placeholders)
        backgroundSpec = self.validateBackgroundValuesSpec(
            configuration.get('background_values'), placeholders, inputs
        )
        channelHints = self.validateChannelHints(
            configuration.get('channel_hints'), placeholders
        )
        folder = QFileDialog.getExistingDirectory(
            self, 'Select experiment, Position or Images folder',
            self.getMostRecentPath()
        )
        if not folder:
            return None
        paths = get_images_paths(folder)
        if not paths:
            raise ValueError(f'No Images folders found in "{folder}".')
        available = [get_channels_and_segm_endnames(path) for path in paths]
        channels = set(available[0][0]).intersection(
            *(set(item[0]) for item in available)
        )
        if len(channels) < len(placeholders):
            raise ValueError(
                f'The premade configuration needs {len(placeholders)} '
                f'channels, but the positions have only {len(channels)} '
                'channels in common.'
            )
        # Experiments and Positions of the selected folder (background
        # values are computed from them)
        experimentPositions = {}
        for path in paths:
            pos_path = os.path.dirname(path)
            experimentPositions.setdefault(
                os.path.dirname(pos_path), []
            ).append(os.path.basename(pos_path))
        selection = self.selectChannelMapping(
            placeholders, sorted(channels), inputs=inputs,
            validate=partial(self.resolvePremadeTemplate, configuration),
            backgroundValues=backgroundSpec,
            experimentPositions=experimentPositions, channelHints=channelHints
        )
        if selection is None:
            return None
        mapping, values = selection
        result = self.resolvePremadeTemplate(configuration, mapping, values)
        result.setdefault('segm_endname', '')
        result.setdefault('overlay_channels', [])
        result.setdefault('visible_channels', [result.get('channel', '')])
        result.setdefault('version', 2)
        result.setdefault('time_course_columns', [])
        result.setdefault('measurements_axes', {'X': '', 'Y': ''})
        result.setdefault('measurements_mode', 0)
        result.setdefault('normalize_measurements', False)
        default = result.get('channel') or None
        offered = sorted(channels)
        if default is None:
            # No display channel in the template: offer the channels that
            # are not mapped (e.g. phase contrast) first
            mapped = set(mapping.values())
            offered.sort(key=lambda ch: ch in mapped)
        base = self.selectPremadeBaseChannel(offered, default)
        if base is None:
            return None
        self.setConfigurationBaseChannel(result, base)
        selectedInput = self.newExperimentInput(result, paths)
        if selectedInput is None:
            return None
        result.update(selectedInput)
        # z-stack tables have no `<channel>_mean`: use the column of the
        # background utility (`_mean_3D` / `_mean_meanProj`) instead
        meanSuffixes = self.premadeMeanSuffixes(
            configuration, mapping, result['images_paths'],
            result['segm_endname']
        )
        if meanSuffixes:
            plain = self.resolvePremadeTemplate(configuration, mapping, values)
            adapted = self.resolvePremadeTemplate(
                configuration, mapping, values, meanSuffixes=meanSuffixes
            )
            # Only the sections with mean columns (the channel and input
            # sections chosen above have none)
            result.update({
                key: value for key, value in adapted.items()
                if value != plain.get(key)
            })
            self.logger.info(
                'Premade configuration: using the mean intensity columns '
                + ', '.join(
                    f'"{mapping[name]}{suffix}"'
                    for name, suffix in meanSuffixes.items()
                )
                + ' (the tables have no "<channel>_mean" column)'
            )
        result['input_options'] = [copy.deepcopy(selectedInput)]
        result['premade'] = {
            'source': os.path.basename(source), 'channel_mapping': mapping
        }
        if inputs:
            result['premade']['input_values'] = dict(values or {})
        self.validateConfiguration(result)
        stem = os.path.splitext(os.path.basename(source))[0]
        defaultName = '_'.join([stem, *mapping.values()])
        filepath = self.askPremadeCopyFilename(defaultName)
        if filepath is None:
            return None
        self.writeConfigurationFile(filepath, result)
        return filepath, result, selectedInput

    def markConfigurationSaved(self):
        try:
            self.savedConfiguration = self.configuration()
        except Exception:
            self.logger.exception('Could not snapshot the configuration')
            self.savedConfiguration = None

    def hasUnsavedConfiguration(self):
        if not self.isDataLoaded:
            return False
        try:
            return self.configuration() != self.savedConfiguration
        except Exception:
            self.logger.exception('Could not compare the configuration')
            return True

    def selectConfigurationInput(self, configuration):
        options = self.configurationInputOptions(configuration)
        def overlaysLabel(option):
            overlays = option.get('overlay_channels', [])
            return f'; overlays: {", ".join(overlays)}' if overlays else ''

        labels = [
            f'{i + 1}. {option["images_paths"][0]} '
            f'({len(option["images_paths"])} positions; '
            f'{option["channel"]}; {option["segm_endname"]}'
            f'{overlaysLabel(option)})'
            for i, option in enumerate(options)
        ]
        new = 'Load a new experiment...'
        selected = self.selectItem(
            'Select experiment', 'Load a saved experiment or a new one:',
            [*labels, new]
        )
        if selected is None:
            return None
        if selected != new:
            return options[labels.index(selected)]
        folder = QFileDialog.getExistingDirectory(
            self, 'Select experiment, Position or Images folder',
            self.getMostRecentPath()
        )
        if not folder:
            return None
        paths = get_images_paths(folder)
        if not paths:
            raise ValueError(f'No Images folders found in "{folder}".')
        return self.newExperimentInput(configuration, paths)

    def newExperimentInput(self, configuration, paths):
        """Map the configuration's input onto new Images folders.

        The configured channel, segmentation and overlays are reused when
        every position has them; otherwise the user picks a replacement
        (channel/segmentation) or confirms dropping them (overlays). Channels
        of the new experiment that the configuration does not use are then
        offered as optional extra overlays. The base channel becomes visible
        when none of the configured visible channels remains.
        """
        available = [get_channels_and_segm_endnames(path) for path in paths]
        channels = set(available[0][0]).intersection(*(set(item[0]) for item in available))
        segmentations = set(available[0][1]).intersection(*(set(item[1]) for item in available))
        if not channels or not segmentations:
            raise ValueError('The positions have no common channels or segmentations.')
        channel = configuration['channel']
        if not channel:
            channel = self.selectDisplayChannel(
                channels, configuration['overlay_channels']
            )
            if channel is None:
                return None
        elif channel not in channels:
            channel = self.selectItem(
                'Select channel',
                f'Channel "{channel}" is not in every position of the new '
                'experiment. Select channel to display:', sorted(channels)
            )
            if channel is None:
                return None
        segm = configuration['segm_endname']
        if segm not in segmentations:
            segm = self.selectItem(
                'Select segmentation',
                f'Segmentation "{segm}" is not in every position of the new '
                'experiment. Select segmentation to load:', sorted(segmentations)
            )
            if segm is None:
                return None
        configured = list(configuration['overlay_channels'])
        overlays = [c for c in configured if c in channels and c != channel]
        missing = [c for c in configured if c not in channels]
        if missing:
            msg = widgets.myMessageBox(wrapText=False)
            cancel, _ = msg.warning(
                self, 'Overlay channels missing',
                html_utils.paragraph(
                    'These overlay channels of the configuration are not in '
                    'every position of the new experiment and will be '
                    f'skipped:<br><br><b>{", ".join(missing)}</b>'
                ),
                buttonsTexts=('Cancel', 'Continue')
            )
            if msg.cancel or msg.clickedButton == cancel:
                return None
        used = {configuration['channel'], channel, *configured}
        added = self.selectConfigurationOverlayChannels(
            sorted(channels - used),
            text=(
                'The new experiment has channels that the configuration does '
                'not use.<br>Select the ones to add as overlay channels '
                '(or leave the selection empty):'
            )
        )
        if added is None:
            return None
        overlays += added
        visible = [
            c for c in configuration['visible_channels']
            if c == channel or c in overlays
        ]
        if channel != configuration['channel'] and channel not in visible:
            visible.insert(0, channel)
        visible += [c for c in added if c not in visible]
        if not visible:
            # E.g., base hidden and every visible overlay missing
            visible = [channel]
        return {
            'images_paths': [os.path.abspath(path) for path in paths],
            'channel': channel, 'segm_endname': segm,
            'overlay_channels': overlays,
            'visible_channels': visible,
        }

    def selectDisplayChannel(self, channels, overlays=()):
        """Ask for the display (base layer) channel of a configuration
        without one. Channels that are not overlays are offered first; an
        overlay chosen as base is removed from the overlays by the caller."""
        items = sorted(channels, key=lambda ch: (ch in overlays, ch))
        return self.selectItem(
            'Select channel',
            'The configuration does not specify a channel to display.\n'
            'Select channel to display (base layer):', items
        )

    def resolveInputChannel(self, selectedInput):
        """Return `selectedInput` with a display channel, asking for one if
        it is empty (None if cancelled). The chosen channel is removed from
        the overlays and shown first."""
        if selectedInput['channel']:
            return selectedInput
        available = [
            set(get_channels_and_segm_endnames(path)[0])
            for path in selectedInput['images_paths']
        ]
        channels = set.intersection(*available) if available else set()
        if not channels:
            raise ValueError('The positions have no common channels.')
        channel = self.selectDisplayChannel(
            channels, selectedInput['overlay_channels']
        )
        if channel is None:
            return None
        resolved = copy.deepcopy(selectedInput)
        self.setConfigurationBaseChannel(resolved, channel)
        return resolved

    def selectConfigurationOverlayChannels(self, channels, text=None):
        if not channels:
            return []
        dialog = widgets.QDialogListbox(
            'Select overlay channels',
            text or 'Select optional overlay channels (or leave the selection empty):',
            channels, multiSelection=True, parent=self, preSelectedItems=()
        )
        dialog.exec_()
        if dialog.cancel:
            return None
        return list(dialog.selectedItemsText)

    def askMeasurementReplay(self, outputs, data):
        """Return 'recalculate', 'existing', or None if the user cancels."""
        existing = {
            column for posData in data
            if getattr(posData, 'acdc_df', None) is not None
            for column in outputs if column in posData.acdc_df
        }
        incomplete = [
            os.path.basename(posData.pos_path) for posData in data
            if getattr(posData, 'acdc_df', None) is None
            or any(column not in posData.acdc_df for column in outputs)
        ]
        txt = (
            'This configuration contains <b>measurement processing</b> and/or '
            '<b>combined measurements</b> steps.<br><br>'
        )
        if incomplete:
            txt += (
                f'Their output columns are missing in {len(incomplete)} of '
                f'{len(data)} positions, so they must be calculated to use '
                'existing values.'
            )
        else:
            txt += 'Their output columns already exist in every position.'
        if existing:
            txt += (
                '<br><br>Calculating will <b>overwrite</b> the existing '
                'output columns.'
            )
        txt = html_utils.paragraph(
            f'{txt}<br><br>Do you want to calculate them?'
        )
        details = 'Output columns:\n' + '\n'.join(outputs)
        if incomplete:
            details += '\n\nPositions with missing columns:\n' + '\n'.join(incomplete)
        msg = widgets.myMessageBox(wrapText=False)
        cancel, existingButton, recalculate = msg.question(
            self, 'Calculate measurements?', txt, detailsText=details,
            buttonsTexts=('Cancel', 'Use existing values', 'Calculate')
        )
        if msg.cancel or msg.clickedButton == cancel:
            return None
        if msg.clickedButton == recalculate:
            return 'recalculate'
        if incomplete:
            raise ValueError(
                'Cannot use existing values: workflow columns are missing in '
                + ', '.join(incomplete) + '.'
            )
        return 'existing'

    def loadConfiguration(self, checked=False, filepath=None):
        if self.loadingInProgress():
            return False
        self._isLoadingData = True
        try:
            return self._loadConfiguration(filepath=filepath)
        finally:
            self._isLoadingData = False

    def _loadConfiguration(self, filepath=None):
        if self.refuseLoadingWhileProcessing():
            return False
        if filepath is None:
            filepath = self.selectConfigurationFilepath()
        if not filepath:
            return False
        # Ask before anything is replaced (setTimepointIdentifiers removes the
        # old tables' time point columns) and before reading the file, so a
        # configuration saved over the same file is the one that is loaded.
        if not self.askSaveBeforeLoading():
            return False
        # Processing may have been started from the non-blocking dialogs
        if self.refuseLoadingWhileProcessing():
            return False
        self.logger.info(f'Loading configuration "{filepath}"...')
        try:
            with open(filepath, encoding='utf-8') as file:
                configuration = json.load(file)
            if (isinstance(configuration, dict)
                    and 'channel_placeholders' in configuration):
                premade = self.instantiatePremadeConfiguration(
                    configuration, filepath
                )
                if premade is None:
                    return False
                filepath, configuration, selectedInput = premade
            else:
                self.validateConfiguration(configuration)
                selectedInput = self.selectConfigurationInput(configuration)
                if selectedInput is not None:
                    selectedInput = self.resolveInputChannel(selectedInput)
                if selectedInput is not None:
                    # Report a bad input instead of applying it (it would
                    # also block saving the configuration later)
                    self.validateConfigurationInput(selectedInput)
            if selectedInput is None:
                return False
            inputs = self.configurationInputOptions(configuration)
            selectedKey = self.inputOptionKey(selectedInput)
            if all(self.inputOptionKey(option) != selectedKey
                   for option in inputs):
                inputs.append(copy.deepcopy(selectedInput))
            configuration.update(selectedInput)
            for path in configuration['images_paths']:
                if not os.path.isdir(path):
                    raise FileNotFoundError(f'Images folder not found: "{path}".')
                channels, segmentations = get_channels_and_segm_endnames(path)
                required = {
                    configuration['channel'], *configuration['overlay_channels']
                }
                missing = required.difference(channels)
                if missing:
                    raise ValueError(f'Missing channels in "{path}": {sorted(missing)}.')
                if configuration['segm_endname'] not in segmentations:
                    raise ValueError(f'Segmentation not found in "{path}".')
            legacyViewed = {
                os.path.normcase(os.path.abspath(path)): IDs
                for path, IDs in configuration.get('viewed_cells', {}).items()
            }
            self.logger.info(
                f'Loading channel "{configuration["channel"]}" of '
                f'{len(configuration["images_paths"])} position(s): '
                + ', '.join(
                    f'"{path}"' for path in configuration['images_paths']
                )
            )
            # In a worker thread (including the overlay channels, taken by
            # `load_fluo_data`): the window stays responsive. Errors are
            # re-raised here.
            data = self.runInThread(
                load_display_positions, configuration['images_paths'],
                configuration['channel'], configuration['segm_endname'],
                skip_errors=False, reportProgress=True,
                overlay_channels=configuration['overlay_channels'],
                text=(
                    f'Loading {len(configuration["images_paths"])} '
                    'position(s)...'
                )
            )
            # Before anything is replaced: `setData` would turn such columns
            # into 0/1 (`customAnnotColumnConflict` for loaded annotations)
            conflicts = flag_column_conflicts(
                [state.get('name') for state
                 in configuration.get('custom_annotations', [])],
                [getattr(posData, 'acdc_df', None) for posData in data]
            )
            if conflicts:
                raise ValueError(
                    flag_column_conflicts_text(conflicts) + '\n\nRename '
                    'these annotations (e.g., on the data the configuration '
                    'was made for) and save the configuration first.'
                )
            # Before the measurement steps and the predictions (of
            # `setData`), which use only the analyzed frames when masking
            mask, maskAsked = self.resolveMaskUnanalyzedFrames(
                data, stored=configuration.get('mask_unanalyzed_frames')
            )
            if mask is None:
                return False
            replay = None
            outputs = self.measurementStepOutputs(configuration)
            if outputs:
                replay = self.askMeasurementReplay(outputs, data)
                if replay is None:
                    return False
            stepsLastFrames = dict(
                configuration.get('measurement_steps_last_frames', {})
            )
            # As in the file (also the premade copy, written before)
            fileStepsLastFrames = dict(stepsLastFrames)
            if replay == 'recalculate':
                lastFrames = [
                    analysis_last_frame(posData, mask) for posData in data
                ]
                def replaySteps():
                    for posData, last in zip(data, lastFrames):
                        self.replayMeasurementSteps(
                            posData, configuration, overwrite=True,
                            last_frame=last
                        )
                self.runInThread(
                    replaySteps, text='Calculating the measurement steps...'
                )
                for posData, last in zip(data, lastFrames):
                    self.recordStepsLastFrame(
                        stepsLastFrames, posData, last, replace=True
                    )
                    # Saved NaN outputs are read back as 0 (tables with
                    # cell cycle columns): not an edit of the file
                    blank_unanalyzed_outputs(
                        getattr(posData, 'savedAcdcDf', None), outputs, last
                    )
            elif replay == 'existing':
                # Saved outputs are empty (read back as 0 from a table with
                # cell cycle columns) after the frame they were calculated
                # up to, and must be ignored after the current one
                self.blankSavedStepOutputs(
                    data, outputs, mask, stepsLastFrames
                )
            for posData in data:
                legacyIDs = legacyViewed.get(
                    os.path.normcase(os.path.abspath(posData.pos_path)), []
                ) if legacyViewed else []
                if legacyIDs and posData.acdc_df is not None:
                    if VISITED_COLUMN not in posData.acdc_df:
                        posData.acdc_df[VISITED_COLUMN] = False
                    rows = posData.acdc_df.index.get_level_values('Cell_ID').isin(legacyIDs)
                    posData.acdc_df.loc[rows, VISITED_COLUMN] = True
        except Exception as err:
            self.logger.exception('Loading measurements viewer workflow failed')
            self.configurationError('load', err)
            return False

        self.isSegm3D = None
        # Remove the previous custom annotations while their data are still
        # loaded: setData (getAcdcDf) adds a column to the new tables for
        # every registered annotation, and setTimepointIdentifiers checks
        # the new auto shortcuts against the registered annotation actions.
        self.setUncheckedAllCustomAnnotButtons()
        self.reinitCustomAnnot()
        # self.setCellConstants(configuration.get('cell_constants', []))
        self.setTimepointIdentifiers([
            TimepointIdentifier.from_dict(identifier)
            for identifier in configuration.get('timepoint_identifiers', [])
        ])
        self.configurationInputs = inputs
        self.loadedInputKey = selectedKey
        self.measurementProcessingSteps = copy.deepcopy(
            configuration.get('measurement_processing', [])
        )
        self.combinedMeasurementSteps = copy.deepcopy(
            configuration.get('combined_measurements', [])
        )
        # Before the deferred predictions of the displayed cell run (on exit
        # of the block below), so they auto-apply the new phase rules.
        self.phaseTranslatorRules = copy.deepcopy(
            configuration.get('phase_translator_rules', [])
        )
        self.phaseColors = dict(configuration.get('phase_colors', {}))
        self.autoApplyPhases = configuration.get('auto_apply_phases', False)
        self.setMaskUnanalyzedFrames(mask, update=False)
        self.measurementStepsLastFrames = stepsLastFrames
        with self.deferPredictorPredictions():
            # Session prediction state of the replaced data is rebuilt from
            # the visited column of the loaded tables (as in
            # `loadImagesPaths`).
            self.resetAutoPredictionTracking()
            self.setData(data)
            self.setCustomAnnotations(configuration.get('custom_annotations', []))
            # Exported assignments and time points of visited cells must
            # survive auto-prediction.
            self.seedVisitedCustomAnnotPredictions(
                self.customAnnotNames(), data
            )
            self.seedVisitedTimepointPredictions(data=data)
        self.configurationFilepath = None
        # `setData` showed the previous configuration file
        self.setStatusBarLabel(log=False)
        overlays = configuration['overlay_channels']
        if overlays:
            loaded = self.loadOverlayData(overlays)
            # Release the preloaded data that `load_fluo_data` did not take
            for posData in data:
                posData.__dict__.pop('_viewerPreloadedFluo', None)
            if not loaded:
                # The data stay loaded, without configuration file
                self.setStatusBarLabel()
                self.configurationError('load', 'Could not load overlay channels.')
                return False
            self.setCheckedOverlayContextMenusActions(overlays)
            self.overlayButton.setChecked(True)
        for channel, button in self.allOverlayToolbuttons.items():
            button.setChecked(channel in configuration['visible_channels'])
        if self.overlayButton.isChecked() and self.overlayToolbar.isTransparent():
            # As `overlayChannelToolbuttonClicked`: the RGBA composite of
            # the transparency modes follows the visible channels (the
            # opacities of the normal mode would hide it)
            self.setOverlayImages()
        else:
            self.setOverlayItemsOpacities()

        columns = {item.text() for item in self.columnsModel.measurementItems()}
        selected = set(configuration['time_course_columns'])
        axes = configuration['measurements_axes']
        missing = (selected | set(axes.values())).difference(columns, {''})
        self.measurementsModeCombobox.blockSignals(True)
        self.measurementsModeCombobox.setCurrentIndex(configuration['measurements_mode'])
        self.measurementsModeCombobox.blockSignals(False)
        # Do not replace the restored time-course selection with the old checks.
        self._measurementsMode = 1
        self._timeCourseColumns = selected & columns
        self.measurementsAxes = {
            axis: value if value in columns else '' for axis, value in axes.items()
        }
        self.conditionalFormatting = copy.deepcopy(
            configuration.get('conditional_formatting', {})
        )
        self.measurementsColors = {
            col: pg.mkColor(color) for col, color
            in configuration.get('measurements_colors', {}).items()
        }
        self.loadTimeCourseYAxes(configuration.get('time_course_y_axes', {}))
        self.measurementsModeChanged()
        self.normalizeCheckbox.setChecked(configuration['normalize_measurements'])
        self.configurationFilepath = filepath
        self.markConfigurationSaved()
        if maskAsked and isinstance(self.savedConfiguration, dict):
            # The answer is not in the file yet: offer to save it on closing
            self.savedConfiguration.pop('mask_unanalyzed_frames', None)
        if (stepsLastFrames != fileStepsLastFrames
                and isinstance(self.savedConfiguration, dict)):
            # Steps (re)calculated or blanked up to other frames: the new
            # record (it protects the saved step outputs) is not in the file
            self.savedConfiguration['measurement_steps_last_frames'] = (
                fileStepsLastFrames
            )
        self.logger.info(f'Configuration loaded: "{filepath}"')
        self.setStatusBarLabel()
        if missing:
            self.configurationError(
                'restore all plot selections',
                f'Columns no longer available: {", ".join(sorted(missing))}.'
            )
        return True
