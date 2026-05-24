"""View adapter for quick settings and side-panel widgets."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFormLayout, QLabel, QVBoxLayout

from cellacdc import apps, settings_csv_path, widgets


class QuickSettingsMixin:
    """Qt-facing adapter around quick-settings view-model contracts."""

    """Headless quick-settings decision rules."""

    def _add_autosave_interval_control(self, layout):
        self.autoSaveIntervalEditButton = widgets.editPushButton(
            flat=True, hoverable=True
        )
        self.autoSaveIntervalLabel = QLabel("Autosave interval")
        self.autoSaveIntervalSetTooltip()
        layout.addRow(
            self.autoSaveIntervalLabel,
            self.autoSaveIntervalEditButton,
        )

        self.autoSaveIntervalDialog = apps.AutoSaveIntervalDialog(parent=self)
        self.autoSaveIntervalDialog.setValues(*self.autoSaveIntevalValueUnit)

    def _add_autosave_toggles(self, layout):
        self.autoSaveToggle = widgets.Toggle()
        tooltip = (
            "Automatically store a copy of the segmentation data "
            "in the `.recovery` folder after every edit."
        )
        self.autoSaveToggle.setChecked(True)
        self.autoSaveToggle.setToolTip(tooltip)
        label = QLabel("Autosave segmentation")
        label.setToolTip(tooltip)
        layout.addRow(label, self.autoSaveToggle)

        self.autoSaveAnnotToggle = widgets.Toggle()
        tooltip = (
            "Automatically store a copy of the annotations (acdc_output CSV "
            "file) in the `.recovery` folder after every edit."
        )
        self.autoSaveAnnotToggle.setChecked(True)
        self.autoSaveAnnotToggle.setToolTip(tooltip)
        label = QLabel("Autosave annotations")
        label.setToolTip(tooltip)
        layout.addRow(label, self.autoSaveAnnotToggle)

    def _add_cca_integrity_checker_toggle(self, layout):
        self.ccaIntegrCheckerToggle = widgets.Toggle()
        tooltip = "Toggle background cell cycle annotations integrity checker ON/OFF"
        self.ccaIntegrCheckerToggle.setChecked(False)
        self.ccaIntegrCheckerToggle.setToolTip(tooltip)
        label = QLabel("Cc annot. checker")
        label.setToolTip(tooltip)
        layout.addRow(label, self.ccaIntegrCheckerToggle)
        idx = "is_cca_integrity_checker_activated"
        if idx in self.df_settings.index:
            val = int(self.df_settings.at[idx, "value"])
            self.ccaIntegrCheckerToggle.setChecked(not val)

    def _add_combined_channels_toggle(self, layout):
        self.viewCombineChannelDataToggle = widgets.Toggle()
        tooltip = (
            "View combined channel. See menu `Image --> combing channels...`\n"
            "on the top menubar."
        )
        self.viewCombineChannelDataToggle.setChecked(False)
        self.viewCombineChannelDataToggle.setToolTip(tooltip)
        label = QLabel("View combined channels")
        label.setToolTip(tooltip)
        layout.addRow(label, self.viewCombineChannelDataToggle)

    def _add_font_size_control(self, layout):
        self.fontSizeSpinBox = widgets.SpinBox()
        self.fontSizeSpinBox.setMinimum(1)
        self.fontSizeSpinBox.setMaximum(99)
        layout.addRow("Font size", self.fontSizeSpinBox)
        font_size_setting = self.font_size_setting(
            self.df_settings.at["fontSize", "value"],
            has_px_mode="pxMode" in self.df_settings.index,
        )
        self.fontSize = font_size_setting.value
        if font_size_setting.add_px_mode_setting:
            self.df_settings.at["pxMode", "value"] = 1
            self.df_settings.to_csv(settings_csv_path)
        self.fontSizeSpinBox.setValue(self.fontSize)
        self.fontSizeSpinBox.editingFinished.connect(self.changeFontSize)
        self.fontSizeSpinBox.sigUpClicked.connect(self.changeFontSize)
        self.fontSizeSpinBox.sigDownClicked.connect(self.changeFontSize)

    def _add_lost_objects_toggle(self, layout):
        self.annotLostObjsToggle = widgets.Toggle()
        tooltip = "Toggle annotation of lost objects mode ON/OFF"
        self.annotLostObjsToggle.setChecked(True)
        self.annotLostObjsToggle.setToolTip(tooltip)
        label = QLabel("Annot. lost objects")
        label.setToolTip(tooltip)
        layout.addRow(label, self.annotLostObjsToggle)

    def _add_realtime_tracking_toggle(self, layout):
        self.realTimeTrackingToggle = widgets.Toggle()
        self.realTimeTrackingToggle.setChecked(True)
        self.realTimeTrackingToggle.setDisabled(True)
        label = QLabel("Real-time tracking")
        label.setDisabled(True)
        self.realTimeTrackingToggle.label = label
        layout.addRow(label, self.realTimeTrackingToggle)

    def _add_show_all_contours_toggle(self, layout):
        self.showAllContoursToggle = widgets.Toggle()
        tooltip = (
            "If active, all contours will be displayed, including inner "
            "contours(e.g. holes and sub-objects)"
        )
        self.showAllContoursToggle.setToolTip(tooltip)
        label = QLabel("Show all contours")
        label.setToolTip(tooltip)
        layout.addRow(label, self.showAllContoursToggle)
        self.showAllContoursToggle.toggled.connect(self.show_all_contours_toggled)

    def _add_view_preprocessed_toggle(self, layout):
        self.viewPreprocDataToggle = widgets.Toggle()
        tooltip = (
            "View pre-processed data. See menu `Image --> Pre-processing...`\n"
            "on the top menubar."
        )
        self.viewPreprocDataToggle.setChecked(False)
        self.viewPreprocDataToggle.setToolTip(tooltip)
        label = QLabel("View pre-processed image")
        label.setToolTip(tooltip)
        layout.addRow(label, self.viewPreprocDataToggle)

    def create_show_props_button(self, side="left"):
        self.leftSideDocksLayout = QVBoxLayout()
        self.leftSideDocksLayout.setSpacing(0)
        self.leftSideDocksLayout.setContentsMargins(0, 0, 0, 0)
        self.rightSideDocksLayout = QVBoxLayout()
        self.rightSideDocksLayout.setSpacing(0)
        self.rightSideDocksLayout.setContentsMargins(0, 0, 0, 0)
        self.showPropsDockButton = widgets.expandCollapseButton()
        self.showPropsDockButton.setDisabled(True)
        self.showPropsDockButton.setFocusPolicy(Qt.NoFocus)
        self.showPropsDockButton.setToolTip("Show object properties")
        if side == "left":
            self.leftSideDocksLayout.addWidget(self.showPropsDockButton)
        else:
            self.rightSideDocksLayout.addWidget(self.showPropsDockButton)

    def create_widgets(self):
        self.quickSettingsLayout = QVBoxLayout()
        self.quickSettingsGroupbox = widgets.GroupBox()
        self.quickSettingsGroupbox.setTitle("Quick settings")

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
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

        self.quickSettingsGroupbox.setLayout(layout)
        self.quickSettingsLayout.addWidget(self.quickSettingsGroupbox)
        self.quickSettingsLayout.addStretch(1)

    def font_size_setting(
        self,
        saved_font_size,
        *,
        has_px_mode: bool,
    ) -> FontSizeSetting:
        saved_font_size = str(saved_font_size)
        if saved_font_size.find("pt") != -1:
            saved_font_size = saved_font_size[:-2]
        font_size = int(saved_font_size)
        if has_px_mode:
            return FontSizeSetting(value=font_size)
        return FontSizeSetting(
            value=2 * font_size,
            add_px_mode_setting=True,
        )

    def should_update_all_contours(self, *, is_data_loaded: bool) -> bool:
        return is_data_loaded

    def show_all_contours_toggled(self):
        if not self.should_update_all_contours(is_data_loaded=self.isDataLoaded):
            return

        self.computeAllContours()
        self.updateAllImages()
