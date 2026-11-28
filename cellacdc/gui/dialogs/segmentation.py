"""Main-GUI-only segmentation dialogs (from apps.py).

Promptable model selection and loading (SelectPromptableModelDialog,
LoadSavedPromptableModelDialog, addCustomPromptModelMessages), the model
info viewer ViewTextDialog, startStopFramesDialog, the post-processing
dialog PostProcessSegmDialog, the combine-channels setup
CombineChannelsSetupDialogGUI and the unused randomWalkerDialog, plus
the widgets only they use.
"""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QSpinBox, QGridLayout, QGroupBox
from cellacdc import myutils
from cellacdc.shared.widgets import item_views as widgets_item_views


class selectStartStopFrames(QGroupBox):
    def __init__(self, SizeT, currentFrameNum=0, parent=None):
        super().__init__(parent)
        selectFramesLayout = QGridLayout()

        self.startFrame_SB = QSpinBox()
        self.startFrame_SB.setAlignment(Qt.AlignCenter)
        self.startFrame_SB.setMinimum(1)
        self.startFrame_SB.setMaximum(SizeT-1)
        self.startFrame_SB.setValue(currentFrameNum)

        self.stopFrame_SB = QSpinBox()
        self.stopFrame_SB.setAlignment(Qt.AlignCenter)
        self.stopFrame_SB.setMinimum(1)
        self.stopFrame_SB.setMaximum(SizeT)
        self.stopFrame_SB.setValue(SizeT)

        selectFramesLayout.addWidget(QLabel('Start frame n.'), 0, 0)
        selectFramesLayout.addWidget(self.startFrame_SB, 1, 0)

        selectFramesLayout.addWidget(QLabel('Stop frame n.'), 0, 1)
        selectFramesLayout.addWidget(self.stopFrame_SB, 1, 1)

        self.warningLabel = QLabel()
        palette = self.warningLabel.palette()
        palette.setColor(self.warningLabel.backgroundRole(), Qt.red)
        palette.setColor(self.warningLabel.foregroundRole(), Qt.red)
        self.warningLabel.setPalette(palette)
        selectFramesLayout.addWidget(
            self.warningLabel, 2, 0, 1, 2, alignment=Qt.AlignCenter
        )

        self.setLayout(selectFramesLayout)

        self.stopFrame_SB.valueChanged.connect(self._checkRange)

    def _checkRange(self):
        start = self.startFrame_SB.value()
        stop = self.stopFrame_SB.value()
        if stop <= start:
            self.warningLabel.setText(
                'stop frame smaller than start frame'
            )
        else:
            self.warningLabel.setText('')


class selectTrackerGUI(widgets_item_views.QDialogListbox):
    def __init__(
            self, SizeT, currentFrameNo=1, parent=None
        ):
        trackers = myutils.get_list_of_trackers()
        super().__init__(
            'Select tracker', 'Select one of the following trackers',
            trackers, multiSelection=False, parent=parent,
            allowEmptySelection=False
        )
        self.setWindowTitle('Select tracker')

        self.selectFramesGroupbox = selectStartStopFrames(
            SizeT, currentFrameNum=currentFrameNo, parent=parent
        )

        self.mainLayout.insertWidget(1, self.selectFramesGroupbox)

    def ok_cb(self, checked=False, button=None):
        if self.selectFramesGroupbox.warningLabel.text():
            return
        else:
            self.startFrame = self.selectFramesGroupbox.startFrame_SB.value()
            self.stopFrame = self.selectFramesGroupbox.stopFrame_SB.value()
            super().ok_cb(checked, button=button)
