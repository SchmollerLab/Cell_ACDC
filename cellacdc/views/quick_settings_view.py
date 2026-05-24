"""View adapter for quick settings and side-panel widgets."""

from __future__ import annotations

from dataclasses import dataclass
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFormLayout, QLabel, QVBoxLayout

from cellacdc import apps, settings_csv_path, widgets


class QuickSettingsView:
    """Qt-facing adapter around quick-settings view-model contracts."""

    """Headless quick-settings decision rules."""

    def font_size_setting(
        self,
        saved_font_size,
        *,
        has_px_mode: bool,
    ) -> FontSizeSetting:
        saved_font_size = str(saved_font_size)
        if saved_font_size.find('pt') != -1:
            saved_font_size = saved_font_size[:-2]
        font_size = int(saved_font_size)
        if has_px_mode:
            return FontSizeSetting(value=font_size)
        return FontSizeSetting(
            value=2*font_size,
            add_px_mode_setting=True,
        )

    def should_update_all_contours(self, *, is_data_loaded: bool) -> bool:
        return is_data_loaded


    def __init__(self, host):
        self.host = host
    def create_show_props_button(self, side='left'):
        self.host.leftSideDocksLayout = QVBoxLayout()
        self.host.leftSideDocksLayout.setSpacing(0)
        self.host.leftSideDocksLayout.setContentsMargins(0, 0, 0, 0)
        self.host.rightSideDocksLayout = QVBoxLayout()
        self.host.rightSideDocksLayout.setSpacing(0)
        self.host.rightSideDocksLayout.setContentsMargins(0, 0, 0, 0)
        self.host.showPropsDockButton = widgets.expandCollapseButton()
        self.host.showPropsDockButton.setDisabled(True)
        self.host.showPropsDockButton.setFocusPolicy(Qt.NoFocus)
        self.host.showPropsDockButton.setToolTip('Show object properties')
        if side == 'left':
            self.host.leftSideDocksLayout.addWidget(
                self.host.showPropsDockButton
            )
        else:
            self.host.rightSideDocksLayout.addWidget(
                self.host.showPropsDockButton
            )

    def create_widgets(self):
        self.host.quickSettingsLayout = QVBoxLayout()
        self.host.quickSettingsGroupbox = widgets.GroupBox()
        self.host.quickSettingsGroupbox.setTitle('Quick settings')

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint
        )
        layout.setFormAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._add_view_preprocessed_toggle(layout)
        self._add_combined_channels_toggle(layout)
        self._add_autosave_toggles(layout)
        self._add_autosave_interval_control(layout)
        self._add_cca_integrity_checker_toggle(layout)
        self._add_lost_objects_toggle(layout)
        self._add_realtime_tracking_toggle(layout)
        self._add_show_all_contours_toggle(layout)
        self._add_font_size_control(layout)

        self.host.quickSettingsGroupbox.setLayout(layout)
        self.host.quickSettingsLayout.addWidget(
            self.host.quickSettingsGroupbox
        )
        self.host.quickSettingsLayout.addStretch(1)

    def show_all_contours_toggled(self):
        if not self.should_update_all_contours(
            is_data_loaded=self.host.isDataLoaded
        ):
            return

        self.host.computeAllContours()
        self.host.updateAllImages()

    def _add_view_preprocessed_toggle(self, layout):
        self.host.viewPreprocDataToggle = widgets.Toggle()
        tooltip = (
            'View pre-processed data. See menu `Image --> Pre-processing...`\n'
            'on the top menubar.'
        )
        self.host.viewPreprocDataToggle.setChecked(False)
        self.host.viewPreprocDataToggle.setToolTip(tooltip)
        label = QLabel('View pre-processed image')
        label.setToolTip(tooltip)
        layout.addRow(label, self.host.viewPreprocDataToggle)

    def _add_combined_channels_toggle(self, layout):
        self.host.viewCombineChannelDataToggle = widgets.Toggle()
        tooltip = (
            'View combined channel. See menu `Image --> combing channels...`\n'
            'on the top menubar.'
        )
        self.host.viewCombineChannelDataToggle.setChecked(False)
        self.host.viewCombineChannelDataToggle.setToolTip(tooltip)
        label = QLabel('View combined channels')
        label.setToolTip(tooltip)
        layout.addRow(label, self.host.viewCombineChannelDataToggle)

    def _add_autosave_toggles(self, layout):
        self.host.autoSaveToggle = widgets.Toggle()
        tooltip = (
            'Automatically store a copy of the segmentation data '
            'in the `.recovery` folder after every edit.'
        )
        self.host.autoSaveToggle.setChecked(True)
        self.host.autoSaveToggle.setToolTip(tooltip)
        label = QLabel('Autosave segmentation')
        label.setToolTip(tooltip)
        layout.addRow(label, self.host.autoSaveToggle)

        self.host.autoSaveAnnotToggle = widgets.Toggle()
        tooltip = (
            'Automatically store a copy of the annotations (acdc_output CSV '
            'file) in the `.recovery` folder after every edit.'
        )
        self.host.autoSaveAnnotToggle.setChecked(True)
        self.host.autoSaveAnnotToggle.setToolTip(tooltip)
        label = QLabel('Autosave annotations')
        label.setToolTip(tooltip)
        layout.addRow(label, self.host.autoSaveAnnotToggle)

    def _add_autosave_interval_control(self, layout):
        self.host.autoSaveIntervalEditButton = widgets.editPushButton(
            flat=True, hoverable=True
        )
        self.host.autoSaveIntervalLabel = QLabel('Autosave interval')
        self.host.autoSaveIntervalSetTooltip()
        layout.addRow(
            self.host.autoSaveIntervalLabel,
            self.host.autoSaveIntervalEditButton,
        )

        self.host.autoSaveIntervalDialog = apps.AutoSaveIntervalDialog(
            parent=self.host
        )
        self.host.autoSaveIntervalDialog.setValues(
            *self.host.autoSaveIntevalValueUnit
        )

    def _add_cca_integrity_checker_toggle(self, layout):
        self.host.ccaIntegrCheckerToggle = widgets.Toggle()
        tooltip = (
            'Toggle background cell cycle annotations integrity checker ON/OFF'
        )
        self.host.ccaIntegrCheckerToggle.setChecked(False)
        self.host.ccaIntegrCheckerToggle.setToolTip(tooltip)
        label = QLabel('Cc annot. checker')
        label.setToolTip(tooltip)
        layout.addRow(label, self.host.ccaIntegrCheckerToggle)
        idx = 'is_cca_integrity_checker_activated'
        if idx in self.host.df_settings.index:
            val = int(self.host.df_settings.at[idx, 'value'])
            self.host.ccaIntegrCheckerToggle.setChecked(not val)

    def _add_lost_objects_toggle(self, layout):
        self.host.annotLostObjsToggle = widgets.Toggle()
        tooltip = 'Toggle annotation of lost objects mode ON/OFF'
        self.host.annotLostObjsToggle.setChecked(True)
        self.host.annotLostObjsToggle.setToolTip(tooltip)
        label = QLabel('Annot. lost objects')
        label.setToolTip(tooltip)
        layout.addRow(label, self.host.annotLostObjsToggle)

    def _add_realtime_tracking_toggle(self, layout):
        self.host.realTimeTrackingToggle = widgets.Toggle()
        self.host.realTimeTrackingToggle.setChecked(True)
        self.host.realTimeTrackingToggle.setDisabled(True)
        label = QLabel('Real-time tracking')
        label.setDisabled(True)
        self.host.realTimeTrackingToggle.label = label
        layout.addRow(label, self.host.realTimeTrackingToggle)

    def _add_show_all_contours_toggle(self, layout):
        self.host.showAllContoursToggle = widgets.Toggle()
        tooltip = (
            'If active, all contours will be displayed, including inner '
            'contours(e.g. holes and sub-objects)'
        )
        self.host.showAllContoursToggle.setToolTip(tooltip)
        label = QLabel('Show all contours')
        label.setToolTip(tooltip)
        layout.addRow(label, self.host.showAllContoursToggle)
        self.host.showAllContoursToggle.toggled.connect(
            self.show_all_contours_toggled
        )

    def _add_font_size_control(self, layout):
        self.host.fontSizeSpinBox = widgets.SpinBox()
        self.host.fontSizeSpinBox.setMinimum(1)
        self.host.fontSizeSpinBox.setMaximum(99)
        layout.addRow('Font size', self.host.fontSizeSpinBox)
        font_size_setting = self.font_size_setting(
            self.host.df_settings.at['fontSize', 'value'],
            has_px_mode='pxMode' in self.host.df_settings.index,
        )
        self.host.fontSize = font_size_setting.value
        if font_size_setting.add_px_mode_setting:
            self.host.df_settings.at['pxMode', 'value'] = 1
            self.host.df_settings.to_csv(settings_csv_path)
        self.host.fontSizeSpinBox.setValue(self.host.fontSize)
        self.host.fontSizeSpinBox.editingFinished.connect(
            self.host.changeFontSize
        )
        self.host.fontSizeSpinBox.sigUpClicked.connect(
            self.host.changeFontSize
        )
        self.host.fontSizeSpinBox.sigDownClicked.connect(
            self.host.changeFontSize
        )