"""View adapter for quick settings and side-panel widgets."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFormLayout, QLabel, QVBoxLayout

from cellacdc import apps, settings_csv_path, widgets


class QuickSettings:
    """Extracted from guiWin."""

    def gui_createQuickSettingsWidgets(self):
        self.quickSettingsLayout = QVBoxLayout()
        self.quickSettingsGroupbox = widgets.GroupBox()
        self.quickSettingsGroupbox.setTitle('Quick settings')

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint
        )
        layout.setFormAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        self.viewPreprocDataToggle = widgets.Toggle()
        viewPreprocDataToggleTooltip = (
            'View pre-processed data. See menu `Image --> Pre-processing...`\n'
            'on the top menubar.'
        )
        self.viewPreprocDataToggle.setChecked(False)
        self.viewPreprocDataToggle.setToolTip(viewPreprocDataToggleTooltip)
        viewPreprocDataToggleLabel = QLabel('View pre-processed image')
        viewPreprocDataToggleLabel.setToolTip(viewPreprocDataToggleTooltip)
        layout.addRow(viewPreprocDataToggleLabel, self.viewPreprocDataToggle)

        self.viewCombineChannelDataToggle = widgets.Toggle()
        viewCombineChannelDataToggleTooltip = (
            'View combined channel. See menu `Image --> combing channels...`\n'
            'on the top menubar.'
        )
        self.viewCombineChannelDataToggle.setChecked(False)
        self.viewCombineChannelDataToggle.setToolTip(
            viewCombineChannelDataToggleTooltip
        )
        viewCombineChannelDataToggleLabel = QLabel('View combined channels')
        viewCombineChannelDataToggleLabel.setToolTip(
            viewCombineChannelDataToggleTooltip
        )
        layout.addRow(
            viewCombineChannelDataToggleLabel, 
            self.viewCombineChannelDataToggle
        )

        self.autoSaveToggle = widgets.Toggle()
        autoSaveTooltip = (
            'Automatically store a copy of the segmentation data '
            'in the `.recovery` folder after every edit.'
        )
        self.autoSaveToggle.setChecked(True)
        self.autoSaveToggle.setToolTip(autoSaveTooltip)
        autoSaveLabel = QLabel('Autosave segmentation')
        autoSaveLabel.setToolTip(autoSaveTooltip)
        layout.addRow(autoSaveLabel, self.autoSaveToggle)
        
        self.autoSaveAnnotToggle = widgets.Toggle()
        autoSaveAnnotTooltip = (
            'Automatically store a copy of the annotations (acdc_output CSV file) '
            'in the `.recovery` folder after every edit.'
        )
        self.autoSaveAnnotToggle.setChecked(True)
        self.autoSaveAnnotToggle.setToolTip(autoSaveAnnotTooltip)
        autoSaveAnnotLabel = QLabel('Autosave annotations')
        autoSaveAnnotLabel.setToolTip(autoSaveAnnotTooltip)
        layout.addRow(autoSaveAnnotLabel, self.autoSaveAnnotToggle)
        
        self.autoSaveIntervalEditButton = widgets.editPushButton(
            flat=True, hoverable=True
        )
        self.autoSaveIntervalLabel = QLabel('Autosave interval')        
        self.autoSaveIntervalSetTooltip()
        layout.addRow(
            self.autoSaveIntervalLabel, self.autoSaveIntervalEditButton
        )
        
        self.autoSaveIntervalDialog = apps.AutoSaveIntervalDialog(parent=self)
        self.autoSaveIntervalDialog.setValues(*self.autoSaveIntevalValueUnit)
        
        self.ccaIntegrCheckerToggle = widgets.Toggle()
        ccaIntegrCheckerToggleTooltip = (
            'Toggle background cell cycle annotations integrity checker ON/OFF'
        )
        self.ccaIntegrCheckerToggle.setChecked(False)
        self.ccaIntegrCheckerToggle.setToolTip(ccaIntegrCheckerToggleTooltip)
        label = QLabel('Cc annot. checker')
        label.setToolTip(ccaIntegrCheckerToggleTooltip)
        layout.addRow(label, self.ccaIntegrCheckerToggle)
        if 'is_cca_integrity_checker_activated' in self.df_settings.index:
            idx = 'is_cca_integrity_checker_activated'
            val = int(self.df_settings.at[idx, 'value'])
            self.ccaIntegrCheckerToggle.setChecked(not val)
        
        self.annotLostObjsToggle = widgets.Toggle()
        annotLostObjsToggleTooltip = (
            'Toggle annotation of lost objects mode ON/OFF'
        )
        self.annotLostObjsToggle.setChecked(True)
        self.annotLostObjsToggle.setToolTip(annotLostObjsToggleTooltip)
        label = QLabel('Annot. lost objects')
        label.setToolTip(annotLostObjsToggleTooltip)
        layout.addRow(label, self.annotLostObjsToggle)

        self.realTimeTrackingToggle = widgets.Toggle()
        self.realTimeTrackingToggle.setChecked(True)
        self.realTimeTrackingToggle.setDisabled(True)
        label = QLabel('Real-time tracking')
        label.setDisabled(True)
        self.realTimeTrackingToggle.label = label
        layout.addRow(label, self.realTimeTrackingToggle)
        
        self.showAllContoursToggle = widgets.Toggle()
        showAllContoursTooltip = (
            'If active, all contours will be displayed, including inner contours'
            '(e.g. holes and sub-objects)'
        )
        self.showAllContoursToggle.setToolTip(showAllContoursTooltip)
        showAllContourLabel = QLabel('Show all contours')
        showAllContourLabel.setToolTip(showAllContoursTooltip)
        layout.addRow(showAllContourLabel, self.showAllContoursToggle)
        self.showAllContoursToggle.toggled.connect(
            self.showAllContoursToggled
        )

        # Font size
        self.fontSizeSpinBox = widgets.SpinBox()
        self.fontSizeSpinBox.setMinimum(1)
        self.fontSizeSpinBox.setMaximum(99)
        layout.addRow('Font size', self.fontSizeSpinBox) 
        savedFontSize = str(self.df_settings.at['fontSize', 'value'])
        if savedFontSize.find('pt') != -1:
            savedFontSize = savedFontSize[:-2]
        self.fontSize = int(savedFontSize)
        if 'pxMode' not in self.df_settings.index:
            # Users before introduction of pxMode had pxMode=False, but now 
            # the new default is True. This requires larger font size.
            self.fontSize = 2*self.fontSize
            self.df_settings.at['pxMode', 'value'] = 1
            self.df_settings.to_csv(settings_csv_path)
        self.fontSizeSpinBox.setValue(self.fontSize)
        self.fontSizeSpinBox.editingFinished.connect(self.changeFontSize) 
        self.fontSizeSpinBox.sigUpClicked.connect(self.changeFontSize)
        self.fontSizeSpinBox.sigDownClicked.connect(self.changeFontSize)

        self.quickSettingsGroupbox.setLayout(layout)
        self.quickSettingsLayout.addWidget(self.quickSettingsGroupbox)
        self.quickSettingsLayout.addStretch(1)
