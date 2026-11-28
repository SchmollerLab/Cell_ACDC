"""Main-GUI-only dialogs to edit objects and annotations (from apps.py).

EditIDDialog, the manual separation window manualSeparateGui, the cell
cycle table editor editCcaTableWidget, the future-frames prompt
FutureFramesAction_QDialog and the unused wandToleranceWidget, plus the
helpers only they use. Sets pyqtgraph row-major image axis order at
import, as apps.py did.
"""
import re
import numpy as np
import scipy.interpolate
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.exposure
import skimage.draw
import skimage.registration
import skimage.color
import skimage.segmentation
import pandas as pd
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from qtpy.QtGui import QIcon, QFont, QFont
from qtpy.QtCore import Qt, QSize, QEvent, Signal, QEventLoop
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QMenu, QLabel, QToolBar, QScrollBar, QWidget,
    QVBoxLayout, QLineEdit, QPushButton, QHBoxLayout, QDialog, QFormLayout,
    QButtonGroup, QCheckBox, QComboBox, QGridLayout, QToolButton, QScrollArea,
    QFrame, QGroupBox, QRadioButton, QMessageBox, QAction, QActionGroup,
    QAbstractSlider
)
from cellacdc import _window_stacking
from cellacdc import html_utils
from cellacdc import printl
from cellacdc import colors
from cellacdc import myutils
from cellacdc import base_cca_dict
from cellacdc import widgets
from cellacdc import core_split_IDs
from cellacdc.shared.dialogs import generic as dialogs_generic


class wandToleranceWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.slider = widgets.sliderWithSpinBox(title='Tolerance')
        self.slider.setMaximum(255)
        self.slider._layout.setColumnStretch(2, 21)

        self.setLayout(self.slider.layout)


class FutureFramesAction_QDialog(QDialog):
    def __init__(
            self, frame_i, last_tracked_i, change_txt,
            applyTrackingB=False, parent=None, 
            addApplyAllButton=False
        ):
        self.decision = None
        self.last_tracked_i = last_tracked_i
        super().__init__(parent)
        self.setWindowTitle('Future frames action?')

        mainLayout = QVBoxLayout()
        txtLayout = QVBoxLayout()
        doNotShowLayout = QVBoxLayout()
        buttonsLayout = QVBoxLayout()

        txt = html_utils.paragraph(
            'You already visited/checked future frames '
            f'{frame_i+1}-{last_tracked_i+1}.<br><br>'
            f'The requested <b>"{change_txt}"</b> change might result in<br>'
            '<b>NON-correct segmentation/tracking</b> for those frames.<br>'
        )

        txtLabel = QLabel(txt)
        txtLabel.setAlignment(Qt.AlignCenter)
        txtLayout.addWidget(txtLabel, alignment=Qt.AlignCenter)

        options = [
            f'Apply the "{change_txt}" <b>only to current frame and re-initialize</b><br>'
            'the future frames to the segmentation file present<br>'
            'on the hard drive.',
            'Apply <b>only to this frame and keep the future frames</b> as they are.',
            'Apply the change to <b>ALL visited/checked future frames</b>.'
        ]
        if addApplyAllButton:
            options.append('Apply to <b>ALL future frames including unvisited ones</b>.')
        if applyTrackingB:
            options.append('Repeat ONLY tracking for all future frames (RECOMMENDED)')

        infoTxt = html_utils.paragraph(
           f'Choose <b>one of the following options:</b>'
           f'{html_utils.to_list(options, ordered=True)}'
        )

        infotxtLabel = QLabel(infoTxt)
        txtLayout.addWidget(infotxtLabel, alignment=Qt.AlignCenter)

        noteLayout = QHBoxLayout()
        noteTxt = html_utils.paragraph(
            'Only changes applied to current frame can be undone.<br>'
            'Changes applied to <b>future frames CANNOT be UNDONE</b><br>'
        )
        noteLayout.addWidget(
            QLabel(html_utils.paragraph('NOTE:')), alignment=Qt.AlignTop
        )
        noteTxtLabel = QLabel(noteTxt)
        noteLayout.addWidget(noteTxtLabel)
        noteLayout.addStretch(1)
        txtLayout.addSpacing(10)
        txtLayout.addLayout(noteLayout)

        # Do not show this message again checkbox
        doNotShowCheckbox = QCheckBox(
            'Remember my choice and do not show this message again')
        doNotShowLayout.addWidget(doNotShowCheckbox)
        doNotShowLayout.setContentsMargins(50, 0, 0, 10)
        self.doNotShowCheckbox = doNotShowCheckbox

        apply_and_reinit_b = widgets.reloadPushButton(
            ' 1. Apply only to this frame and re-initialize future frames'
        )

        self.apply_and_reinit_b = apply_and_reinit_b
        buttonsLayout.addWidget(apply_and_reinit_b)

        apply_and_NOTreinit_b = widgets.currentPushButton(
            ' 2. Apply only to this frame and keep future frames as they are'
        )
        self.apply_and_NOTreinit_b = apply_and_NOTreinit_b
        buttonsLayout.addWidget(apply_and_NOTreinit_b)

        apply_to_all_visited_b = widgets.futurePushButton(
            ' 3. Apply to all future VISITED frames'
        )
        self.apply_to_all_visited_b = apply_to_all_visited_b
        buttonsLayout.addWidget(apply_to_all_visited_b)

        
        if addApplyAllButton:
            apply_to_all_b = QPushButton(
                ' 4. Apply to ALL future frames (including unvisted)'
            )
            apply_to_all_b.setIcon(QIcon(':arrow_future_all.svg'))
            self.apply_to_all_b = apply_to_all_b
            buttonsLayout.addWidget(apply_to_all_b)

        self.applyTrackingButton = None
        if applyTrackingB:
            n = '5' if addApplyAllButton else '4'
            applyTrackingButton = QPushButton(
                f' {n}. Repeat ONLY tracking for all future frames'
            )
            applyTrackingButton.setIcon(QIcon(':repeat-tracking.svg'))
            self.applyTrackingButton = applyTrackingButton
            buttonsLayout.addWidget(applyTrackingButton)

        buttonsLayout.setContentsMargins(20, 0, 20, 0)

        self.formLayout = QFormLayout()

        ButtonsGroup = QButtonGroup(self)
        ButtonsGroup.addButton(apply_and_reinit_b)
        ButtonsGroup.addButton(apply_and_NOTreinit_b)
        ButtonsGroup.addButton(apply_to_all_visited_b)
        if addApplyAllButton:
            ButtonsGroup.addButton(apply_to_all_b)
        if applyTrackingB:
            ButtonsGroup.addButton(applyTrackingButton)
        
        mainLayout.addLayout(txtLayout)
        mainLayout.addLayout(doNotShowLayout)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addLayout(self.formLayout)
        mainLayout.addStretch(1)
        self.mainLayout = mainLayout
        self.setLayout(mainLayout)

        # Connect events
        ButtonsGroup.buttonClicked.connect(self.buttonClicked)
        self.ButtonsGroup = ButtonsGroup

        # self.setModal(True)

    def buttonClicked(self, button):
        if button == self.apply_and_reinit_b:
            self.decision = 'apply_and_reinit'
            self.endFrame_i = None
        elif button == self.apply_and_NOTreinit_b:
            self.decision = 'apply_and_NOTreinit'
            self.endFrame_i = None
        elif button == self.apply_to_all_visited_b:
            self.decision = 'apply_to_all_visited'
            self.endFrame_i = self.last_tracked_i
        elif button == self.applyTrackingButton:
            self.decision = 'only_tracking'
            self.endFrame_i = self.last_tracked_i
        elif button == self.apply_to_all_b:
            self.decision = 'apply_to_all'
            self.endFrame_i = self.last_tracked_i
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        super().show()
        for button in self.ButtonsGroup.buttons():
            button.setMinimumHeight(int(button.height()*1.2))
        if hasattr(self, 'apply_to_all_b'):
            iconHeight = self.apply_to_all_b.iconSize().height()
            self.apply_to_all_b.setIconSize(QSize(iconHeight*2, iconHeight))
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()


class editCcaTableWidget(QDialog):
    sigApplyChangesFutureFrames = Signal(object, int)
    
    def __init__(
            self, cca_df, SizeT, title='Edit cell cycle annotations', 
            parent=None, current_frame_i=0
        ):
        self.inputCca_df = cca_df
        self.cancel = True
        self.SizeT = SizeT
        self.cca_df = None
        self.current_frame_i = current_frame_i

        super().__init__(parent)
        self.setWindowTitle(title)

        _window_stacking.set_window_flags_on_top(self, Qt.Window)

        # Layouts
        mainLayout = QVBoxLayout()
        headerLayout = QGridLayout()
        tableLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()
        self.scrollArea = QScrollArea()
        self.viewBox = QWidget()

        # Header labels
        col = 0
        row = 0
        IDsLabel = QLabel('Cell ID')
        AC = Qt.AlignCenter
        IDsLabel.setAlignment(AC)
        headerLayout.addWidget(IDsLabel, 0, col, alignment=AC)

        col += 1
        ccsLabel = QLabel('Cell cycle stage')
        ccsLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(ccsLabel, 0, col, alignment=AC)

        col += 1
        relIDLabel = QLabel('Relative ID')
        relIDLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(relIDLabel, 0, col, alignment=AC)

        col += 1
        genNumLabel = QLabel('Generation number')
        genNumLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(genNumLabel, 0, col, alignment=AC)
        genNumColWidth = genNumLabel.sizeHint().width()

        col += 1
        relationshipLabel = QLabel('Relationship')
        relationshipLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(relationshipLabel, 0, col, alignment=AC)

        col += 1
        emergFrameLabel = QLabel('Emerging frame num.')
        emergFrameLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(emergFrameLabel, 0, col, alignment=AC)

        col += 1
        divitionFrameLabel = QLabel('Division frame num.')
        divitionFrameLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(divitionFrameLabel, 0, col, alignment=AC)

        col += 1
        historyKnownLabel = QLabel('Is history known?')
        historyKnownLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(historyKnownLabel, 0, col, alignment=AC)
        
        self.headerLayout = headerLayout

        tableLayout.setHorizontalSpacing(20)
        self.tableLayout = tableLayout

        # Add buttons
        cancelButton = widgets.cancelPushButton('Cancel')
        moreInfoButton = widgets.helpPushButton('More info...')
        moreInfoButton.setIcon(QIcon(':info.svg'))
        applyToFutureFramesbutton = widgets.futurePushButton(
            'Apply changes to future frames...'
        )
        okButton = widgets.okPushButton('Ok')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(moreInfoButton)
        buttonsLayout.addWidget(applyToFutureFramesbutton)
        buttonsLayout.addWidget(okButton)

        # Scroll area properties
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setFrameStyle(QFrame.Shape.NoFrame)
        self.scrollArea.setWidgetResizable(True)

        # Add layouts
        self.viewBox.setLayout(tableLayout)
        self.scrollArea.setWidget(self.viewBox)
        mainLayout.addLayout(headerLayout)
        mainLayout.addWidget(self.scrollArea)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        # Populate table Layout
        IDs = cca_df.index
        self.IDs = IDs.to_list()
        relIDsOptions = [str(ID) for ID in IDs]
        relIDsOptions.insert(0, '-1')
        self.IDlabels = []
        self.ccsComboBoxes = []
        self.genNumSpinBoxes = []
        self.relIDComboBoxes = []
        self.relationshipComboBoxes = []
        self.emergFrameSpinBoxes = []
        self.divisFrameSpinBoxes = []
        self.emergFrameSpinPrevValues = []
        self.divisFrameSpinPrevValues = []
        self.historyKnownCheckBoxes = []
        for row, ID in enumerate(IDs):
            col = 0
            IDlabel = QLabel(f'{ID}')
            IDlabel.setAlignment(Qt.AlignCenter)
            tableLayout.addWidget(IDlabel, row+1, col, alignment=AC)
            self.IDlabels.append(IDlabel)

            col += 1
            ccsComboBox = QComboBox()
            ccsComboBox.setFocusPolicy(Qt.StrongFocus)
            ccsComboBox.installEventFilter(self)
            ccsComboBox.addItems(['G1', 'S/G2/M'])
            ccsValue = cca_df.at[ID, 'cell_cycle_stage']
            if ccsValue == 'S':
                ccsValue = 'S/G2/M'
            
            try:
                ccsComboBox.setCurrentText(ccsValue)
            except Exception as err:
                printl(ccsValue)
                printl(cca_df)
                raise err
            tableLayout.addWidget(ccsComboBox, row+1, col, alignment=AC)
            self.ccsComboBoxes.append(ccsComboBox)
            ccsComboBox.activated.connect(self.clearComboboxFocus)

            col += 1
            relIDComboBox = QComboBox()
            relIDComboBox.setFocusPolicy(Qt.StrongFocus)
            relIDComboBox.installEventFilter(self)
            relIDComboBox.addItems(relIDsOptions)
            relIDComboBox.setCurrentText(str(cca_df.at[ID, 'relative_ID']))
            tableLayout.addWidget(relIDComboBox, row+1, col)
            self.relIDComboBoxes.append(relIDComboBox)
            relIDComboBox.currentIndexChanged.connect(self.setRelID)
            relIDComboBox.activated.connect(self.clearComboboxFocus)

            col += 1
            genNumSpinBox = widgets.SpinBox()
            genNumSpinBox.setFocusPolicy(Qt.StrongFocus)
            genNumSpinBox.installEventFilter(self)
            genNumSpinBox.setValue(2)
            genNumSpinBox.setMaximum(2147483647)
            genNumSpinBox.setAlignment(Qt.AlignCenter)
            genNumSpinBox.setFixedWidth(int(genNumColWidth*2/3))
            genNumSpinBox.setValue(int(cca_df.at[ID, 'generation_num']))
            tableLayout.addWidget(genNumSpinBox, row+1, col, alignment=AC)
            self.genNumSpinBoxes.append(genNumSpinBox)

            col += 1
            relationshipComboBox = QComboBox()
            relationshipComboBox.setFocusPolicy(Qt.StrongFocus)
            relationshipComboBox.installEventFilter(self)
            relationshipComboBox.addItems(['mother', 'bud'])
            relationshipComboBox.setCurrentText(
                str(cca_df.at[ID, 'relationship'])
            )
            tableLayout.addWidget(relationshipComboBox, row+1, col)
            self.relationshipComboBoxes.append(relationshipComboBox)
            relationshipComboBox.currentIndexChanged.connect(
                                                self.relationshipChanged_cb)
            relationshipComboBox.activated.connect(self.clearComboboxFocus)

            col += 1
            emergFrameSpinBox = widgets.SpinBox()
            emergFrameSpinBox.setFocusPolicy(Qt.StrongFocus)
            emergFrameSpinBox.installEventFilter(self)
            emergFrameSpinBox.setMaximum(SizeT)
            emergFrameSpinBox.setMinimum(-1)
            emergFrameSpinBox.setValue(-1)
            emergFrameSpinBox.setAlignment(Qt.AlignCenter)
            emergFrameSpinBox.setFixedWidth(int(genNumColWidth*2/3))
            emergFrame_i = cca_df.at[ID, 'emerg_frame_i']
            val = emergFrame_i+1 if emergFrame_i>=0 else -1
            emergFrameSpinBox.setValue(val)
            tableLayout.addWidget(emergFrameSpinBox, row+1, col, alignment=AC)
            self.emergFrameSpinBoxes.append(emergFrameSpinBox)
            self.emergFrameSpinPrevValues.append(emergFrameSpinBox.value())
            emergFrameSpinBox.valueChanged.connect(self.skip0emergFrame)


            col += 1
            divisFrameSpinBox = widgets.SpinBox()
            divisFrameSpinBox.setFocusPolicy(Qt.StrongFocus)
            divisFrameSpinBox.installEventFilter(self)
            divisFrameSpinBox.setMinimum(-1)
            divisFrameSpinBox.setMaximum(SizeT)
            divisFrameSpinBox.setValue(-1)
            divisFrameSpinBox.setAlignment(Qt.AlignCenter)
            divisFrameSpinBox.setFixedWidth(int(genNumColWidth*2/3))
            divisFrame_i = int(cca_df.at[ID, 'division_frame_i'])
            val = divisFrame_i+1 if divisFrame_i>=0 else -1
            divisFrameSpinBox.setValue(val)
            tableLayout.addWidget(divisFrameSpinBox, row+1, col, alignment=AC)
            self.divisFrameSpinBoxes.append(divisFrameSpinBox)
            self.divisFrameSpinPrevValues.append(divisFrameSpinBox.value())
            divisFrameSpinBox.valueChanged.connect(self.skip0divisFrame)

            col += 1
            HistoryCheckBox = QCheckBox()
            HistoryCheckBox.setChecked(bool(cca_df.at[ID, 'is_history_known']))
            tableLayout.addWidget(HistoryCheckBox, row+1, col, alignment=AC)
            self.historyKnownCheckBoxes.append(HistoryCheckBox)

        self.setLayout(mainLayout)

        # Connect to events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)
        moreInfoButton.clicked.connect(self.moreInfo)
        applyToFutureFramesbutton.clicked.connect(self.applyToFutureFrames)

        # self.setModal(True)
    
    def getChanges(self):
        newCcaDf = self.getCca_df()
        changes = {}
        for row in newCcaDf.itertuples():
            ID = row.Index
            for col in newCcaDf.columns:
                inputValue = self.inputCca_df.at[ID, col]
                newValue = getattr(row, col)
                if newValue == inputValue:
                    continue
                
                if ID not in changes:
                    changes[ID] = {col: (inputValue, newValue)}
                else:
                    changes[ID][col] = (inputValue, newValue)
        return changes

    def applyToFutureFrames(self):        
        txt = 'Enter <b>up to which frame number</b> you want to apply the changes<br>'
        win = dialogs_generic.NumericEntryDialog(
            title='Stop frame', 
            instructions=txt, 
            parent=self, 
            minValue=1, 
            maxValue=self.SizeT, 
            currentValue=self.current_frame_i+1
        )
        win.exec_()
        if win.cancel:
            return
        
        stop_frame_n = win.value
        changes = self.getChanges()
        changes_format = myutils.format_cca_manual_changes(changes)
        detailsText = (
            f'Changes that will be applied from frame n. {self.current_frame_i+1}'
            f' to frame n. {stop_frame_n}:\n\n{changes_format}'
        )
        txt = html_utils.paragraph("""
Use this feature with <b>caution</b>!<br><br>
Before propagating to future frames <b>carefully inspect what changes</b> 
will be applied (see below).<br><br>
""")
        msg = widgets.myMessageBox(wrapText=False)
        msg.setDetailedText(detailsText, visible=True)
        msg.warning(
            self, 'Caution!', txt, buttonsTexts=('Yes, I am sure', 'Cancel')
        )
        if msg.cancel:
            return
        
        self.sigApplyChangesFutureFrames.emit(changes, stop_frame_n)     
    
    def moreInfo(self, checked=True):
        desc = myutils.get_cca_colname_desc()
        msg = widgets.myMessageBox(parent=self)
        msg.setWindowTitle('Cell cycle annotations info')
        msg.setWidth(400)
        msg.setIcon()
        for col, txt in desc.items():
            msg.addText(html_utils.paragraph(f'<b>{col}</b>: {txt}'))
        msg.addButton('  Ok  ')
        msg.exec_()

    def setRelID(self, itemIndex):
        idx = self.relIDComboBoxes.index(self.sender())
        relID = self.sender().currentText()
        IDofRelID = self.IDs[idx]
        relIDidx = self.IDs.index(int(relID))
        relIDComboBox = self.relIDComboBoxes[relIDidx]
        relIDComboBox.setCurrentText(str(IDofRelID))

    def skip0emergFrame(self, value):
        idx = self.emergFrameSpinBoxes.index(self.sender())
        prevVal = self.emergFrameSpinPrevValues[idx]
        if value == 0 and value > prevVal:
            self.sender().setValue(1)
            self.emergFrameSpinPrevValues[idx] = 1
        elif value == 0 and value < prevVal:
            self.sender().setValue(-1)
            self.emergFrameSpinPrevValues[idx] = -1

    def skip0divisFrame(self, value):
        idx = self.divisFrameSpinBoxes.index(self.sender())
        prevVal = self.divisFrameSpinPrevValues[idx]
        if value == 0 and value > prevVal:
            self.sender().setValue(1)
            self.divisFrameSpinPrevValues[idx] = 1
        elif value == 0 and value < prevVal:
            self.sender().setValue(-1)
            self.divisFrameSpinPrevValues[idx] = -1

    def relationshipChanged_cb(self, itemIndex):
        idx = self.relationshipComboBoxes.index(self.sender())
        ccs = self.sender().currentText()
        if ccs == 'bud':
            self.ccsComboBoxes[idx].setCurrentText('S/G2/M')
            self.genNumSpinBoxes[idx].setValue(0)

    def getCca_df(self):
        ccsValues = [var.currentText() for var in self.ccsComboBoxes]
        ccsValues = [val if val=='G1' else 'S' for val in ccsValues]
        genNumValues = [var.value() for var in self.genNumSpinBoxes]
        relIDValues = [int(var.currentText()) for var in self.relIDComboBoxes]
        relatValues = [var.currentText() for var in self.relationshipComboBoxes]
        emergFrameValues = [
            var.value()-1 if var.value()>0 else -1
            for var in self.emergFrameSpinBoxes
        ]
        divisFrameValues = [
            var.value()-1 if var.value()>0 else -1
            for var in self.divisFrameSpinBoxes
        ]
        historyValues = [
            var.isChecked() for var in self.historyKnownCheckBoxes
        ]
        check_rel = [ID == relID for ID, relID in zip(self.IDs, relIDValues)]
        
        # Buds in S phase must have 0 as number of cycles
        check_buds_S = [
            ccs=='S' and rel_ship=='bud' and not numc==0
            for ccs, rel_ship, numc
            in zip(ccsValues, relatValues, genNumValues)
        ]
        
        # Mother cells must have at least 1 as number of cycles if history known
        check_mothers = [
            rel_ship=='mother' and not numc>=1
            if is_history_known else False
            for rel_ship, numc, is_history_known
            in zip(relatValues, genNumValues, historyValues)
        ]
        
        # Buds cannot be in G1
        check_buds_G1 = [
            ccs=='G1' and rel_ship=='bud' for ccs, rel_ship
            in zip(ccsValues, relatValues)
        ]
        
        # The number of cells in S phase must be half mothers and half buds
        num_moth_S = len([
            0 for ccs, rel_ship in zip(ccsValues, relatValues)
            if ccs=='S' and rel_ship=='mother'
        ])
        num_bud_S = len([
            0 for ccs, rel_ship in zip(ccsValues, relatValues)
            if ccs=='S' and rel_ship=='bud'
        ])
        
        # Cells in S phase cannot have -1 as relative's ID
        check_relID_S = [
            ccs=='S' and relID==-1
            for ccs, relID in zip(ccsValues, relIDValues)
        ]
        
        # Mother cells with unknown history at emergence is recommended to have
        # generation number = 2 (easier downstream analysis)
        check_unknown_mothers = [
            rel_ship=='mother' and not is_history_known and gen_num!=2
            and (emerg_frame_i == self.current_frame_i or self.current_frame_i==0)
            for rel_ship, is_history_known, gen_num, emerg_frame_i
            in zip(relatValues, historyValues, genNumValues, emergFrameValues)
        ]
        
        if any(check_rel):
            msg = widgets.myMessageBox(wrapText=False)
            txt = html_utils.paragraph(""" 
                Some cells are mother or bud of itself!<br><br>
                Make sure that the relative ID is different from the Cell ID.
            """)
            msg.critical(self, 'Some IDs are equal to relative ID', txt)
            return None
        elif any(check_unknown_mothers):
            txt = html_utils.paragraph("""
                We recommend to set <b>generation number to 2 for mother cells 
                with unknown history<br>
                that just appeared</b> (i.e., first cell cycle in the video).<br><br>
                While it is allowed to insert any number, knowing that these 
                cells start at generation number 2<br>
                <b>makes downstream analysis easier</b>.<br><br>
                What do you want to do?
            """)
            correctButton = widgets.editPushButton(' Fine, let me correct. ')
            keepButton = widgets.WarningButton(
                ' Keep the generation number that I chose. '
            )
            buttonsTexts = (correctButton, keepButton)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            msg.warning(self, 'Recommendation', txt, buttonsTexts=buttonsTexts)
            if msg.cancel or msg.clickedButton == correctButton:
                return None
        elif any(check_buds_S):
            msg = widgets.myMessageBox(wrapText=False)
            title = (
                'Bud in S/G2/M not in 0 Generation number'
            )
            txt = html_utils.paragraph(
                'Some buds '
                'in S phase do not have 0 as Generation number!<br>'
                'Buds in S phase must have 0 as "Generation number"'
            )
            msg.critical(self, title, txt)
            return None
        elif any(check_mothers):
            msg = widgets.myMessageBox(wrapText=False)
            title = (
                'Mother not in >=1 Generation number'
            )
            txt = html_utils.paragraph(
                'Some mother cells do not have >=1 as "Generation number"!<br>'
                'Mothers MUST have >1 "Generation number"'
            )
            msg.critical(self, title, txt)
            return None
        elif any(check_buds_G1):
            msg = widgets.myMessageBox(wrapText=False)
            title = (
                'Buds in G1!'
            )
            txt = html_utils.paragraph(
                'Some buds are in G1 phase!<br><br>'
                'Buds MUST be in S/G2/M phase'
            )
            msg.critical(self, title, txt)
            return None
        elif num_moth_S != num_bud_S:
            msg = widgets.myMessageBox(wrapText=False)
            title = (
                'Number of mothers-buds mismatch!'
            )
            txt = html_utils.paragraph(
                f'There are {num_moth_S} mother cells in "S/G2/M" phase,'
                f'but there are {num_bud_S} bud cells.<br><br>'
                'The number of mothers and buds in "S/G2/M" '
                'phase must be equal!'
            )
            msg.critical(self, title, txt)
            return None
        elif any(check_relID_S):
            msg = widgets.myMessageBox(wrapText=False)
            title = (
                'Relative\'s ID of cells in S/G2/M = -1'
            )
            txt = html_utils.paragraph(
                'Some cells are in "S/G2/M" phase but have -1 as Relative\'s ID!<br>'
                'Cells in "S/G2/M" phase must have an existing '
                'ID as Relative\'s ID!'
            )
            msg.critical(self, title, txt)
            return None
        
        corrected_on_frame_i = self.inputCca_df['corrected_on_frame_i']
        cca_df = pd.DataFrame({
            'cell_cycle_stage': ccsValues,
            'generation_num': genNumValues,
            'relative_ID': relIDValues,
            'relationship': relatValues,
            'emerg_frame_i': emergFrameValues,
            'division_frame_i': divisFrameValues,
            'is_history_known': historyValues,
            'corrected_on_frame_i': corrected_on_frame_i,
            'will_divide': self.inputCca_df['will_divide'],
            }, index=self.IDs
        )
        cca_df.index.name = 'Cell_ID'
        
        # Add missing columns
        for column, default in base_cca_dict.items():
            if column in cca_df.columns:
                continue
            
            value = self.inputCca_df.get(column, default=default)
            cca_df[column] = value
            
        # Check that every pair of cells in S are relative of each other
        proceed = self.check_ID_rel_ID_mismatches(cca_df)
        if not proceed:
            return None
            
        d = dict.fromkeys(cca_df.select_dtypes(np.int64).columns, np.int32)
        cca_df = cca_df.astype(d)
        return cca_df

    def check_ID_rel_ID_mismatches(self, cca_df):
        ID_rel_ID_mismatches = []
        for row in cca_df.itertuples():
            if row.cell_cycle_stage == 'G1':
                continue
            
            ID = row.Index
            relID = row.relative_ID
            relID_of_relID = cca_df.at[relID, 'relative_ID']
            
            if relID_of_relID != ID:
                ID_rel_ID_mismatches.append((ID, relID, relID_of_relID))
        
        if not ID_rel_ID_mismatches:
            return True
        
        items = [
            f'Cell ID {ID} has relative ID = {relID}, '
            f'while cell ID {relID} has relative ID = {relID_of_relID}'
            for ID, relID, relID_of_relID in ID_rel_ID_mismatches
        ]
        title = '`ID-relative_ID` mismatches'
        txt = html_utils.paragraph(
            f'`ID-relative_ID` mismatches:'
            f'{html_utils.to_list(items)}'
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.critical(self, title, txt)
        return False
    
    def ok_cb(self, checked):
        cca_df = self.getCca_df()
        if cca_df is None:
            return
        self.cca_df = cca_df
        self.cancel = False
        self.close()

    def cancel_cb(self, checked):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        super().show()
        ncols = self.tableLayout.columnCount()
        maxLabelWidth = max([
            self.headerLayout.itemAt(j).widget().sizeHint().width()
            for j in range(ncols)
        ])
        minWidth = (maxLabelWidth+5)*ncols
        self.setMinimumWidth(minWidth)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def eventFilter(self, object, event):
        # Disable wheel scroll on widgets to allow scroll only on scrollarea
        if event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return False

    def clearComboboxFocus(self):
        self.sender().clearFocus()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()


class EditIDDialog(QDialog):
    def __init__(
            self, clickedID, IDs, 
            entryID=None, 
            doNotShowAgain=False, 
            parent=None, 
            nextUniqueID=1, 
            allIDs=None,
            addPropagateCheckbox=False,
            isSegm3D=False,
            currentZslice=1,
            SizeZ=None,
            zSliceStartValue=None
        ):
        self.assignNewID = False
        self.IDs = IDs
        self.clickedID = clickedID
        self.cancel = True
        self.how = None
        self.mergeWithExistingID = False
        self.doNotAskAgainExistingID = doNotShowAgain
        self.doPropagateFutureFrames = False
        self.howToPropagateZslices = None
        self.allIDs = allIDs
        self.isSegm3D = isSegm3D
        if allIDs is None:
            self.allIDs = set(self.IDs)
        self.nextUniqueID = nextUniqueID

        super().__init__(parent)
        self.setWindowTitle("Edit ID")
        mainLayout = QVBoxLayout()

        VBoxLayout = QVBoxLayout()
        msg = QLabel(f'Replace ID {clickedID} with:')
        _font = QFont()
        _font.setPixelSize(12)
        msg.setFont(_font)
        # padding: top, left, bottom, right
        msg.setStyleSheet("padding:0px 0px 3px 0px;")
        VBoxLayout.addWidget(msg, alignment=Qt.AlignCenter)

        entryWidget = QLineEdit()
        entryWidget.setFont(_font)
        entryWidget.setAlignment(Qt.AlignCenter)
        self.entryWidget = entryWidget
        VBoxLayout.addWidget(entryWidget)
        if entryID is not None:
            entryWidget.setText(str(entryID))
            entryWidget.selectAll()

        VBoxLayout.addWidget(
            QLabel(f'Next unique ID = {nextUniqueID}'), alignment=Qt.AlignCenter
        )
        
        VBoxLayout.addWidget(widgets.QHLine())
        
        self.warnExistingIDLabel = QLabel()
        self.warnExistingIDLabel.setStyleSheet('color: red')
        VBoxLayout.addWidget(
            self.warnExistingIDLabel, alignment=Qt.AlignCenter
        )
        
        note = QLabel(
            'NOTE: To replace multiple IDs at once\n'
            'write "(old ID, new ID), (old ID, new ID)" etc.'
        )
        note.setFont(_font)
        note.setAlignment(Qt.AlignCenter)
        # padding: top, left, bottom, right
        note.setStyleSheet("padding:12px 0px 0px 0px;")
        VBoxLayout.addWidget(note, alignment=Qt.AlignCenter)
        mainLayout.addLayout(VBoxLayout)

        if isSegm3D:
            howToPropagateGroupbox = self.addHowToPropagateZslicesOptions(
                currentZslice, SizeZ, zSliceStartValue
            )
            mainLayout.addWidget(widgets.QHLine())
            mainLayout.addSpacing(10)
            mainLayout.addWidget(howToPropagateGroupbox)
        
        self.propagateCheckbox = None
        if addPropagateCheckbox:
            mainLayout.addSpacing(10)
            self.propagateCheckbox = QCheckBox('Apply to future frames')
            mainLayout.addWidget(self.propagateCheckbox)

        buttonsLayout = QHBoxLayout()
        okButton = widgets.okPushButton('Ok')
        cancelButton = widgets.cancelPushButton('Cancel')
        applyNewIDButton = widgets.AssignNewIDButton('Assign new, unique ID')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(applyNewIDButton)
        buttonsLayout.addWidget(okButton)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        # Connect events
        self.prevText = ''
        entryWidget.textChanged[str].connect(self.onTextChanged)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)
        applyNewIDButton.clicked.connect(self.assignNewIDclicked)
        
        # self.setModal(True)
    
    def addHowToPropagateZslicesOptions(
            self, currentZslice, SizeZ, zSliceStartValue
        ):
        howToPropagateGroupbox = QGroupBox('Apply ID change to')
        howToPropagateLayout = QGridLayout()
        self.howToPropagateButtonsGroup = QButtonGroup()
        howToPropagateGroupbox.setLayout(howToPropagateLayout)
        self.howToPropagateRadioCurrent = QRadioButton(
            'Current z-slice'
        )
        self.howToPropagateRadioCurrent.setChecked(True)
        self.howToPropagateRadioCurrent.setToolTip(
            'Change the object ID only in the current z-slice'
        )
        self.howToPropagateRadioEntireObj = QRadioButton(
            'Entire object'
        )
        self.howToPropagateRadioEntireObj.setToolTip(
            'Change the object ID in the entire object'
        )
        self.howToPropagateRadioToEnd = QRadioButton(
            'From current to last z-slice')
        self.howToPropagateRadioToEnd.setToolTip(
            'Change the object ID from the current Z-slice to the last '
            "Z-slice of the object's bounding box (inclusive)."
        )
        self.howToPropagateRadioToStart = QRadioButton(
            'From current to first z-slice'
        )
        self.howToPropagateRadioToStart.setToolTip(
            'Change the object ID from the first Z-slice of the '
            "object's bounding box to the current Z-slice (inclusive)."
        )

        self.howToPropagateRadioRange = QRadioButton('Custom range')
        self.howToPropagateRadioRange.setToolTip(
            'Change the object ID within a defined range of Z-slices (last included).'
        )

        self.howToPropagateRangeStartZSpinbox = widgets.SpinBox()
        self.howToPropagateRangeStartZSpinbox.setMinimum(0)
        self.howToPropagateRangeStartZSpinbox.setStartValue(1)
        self.howToPropagateRangeStartZSpinbox.setValue(currentZslice)
        self.howToPropagateRangeStartZSpinbox.setDisabled(True)
        if SizeZ is not None:
            self.howToPropagateRangeStartZSpinbox.setMaximum(SizeZ)

        self.howToPropagateRangeEndZSpinbox = widgets.SpinBox()
        self.howToPropagateRangeEndZSpinbox.setMinimum(0)
        self.howToPropagateRangeEndZSpinbox.setStartValue(1)
        self.howToPropagateRangeEndZSpinbox.setValue(currentZslice)
        if SizeZ is not None:
            self.howToPropagateRangeEndZSpinbox.setMaximum(SizeZ-1)
        self.howToPropagateRangeEndZSpinbox.setDisabled(True)

        howToPropagateLayout.addWidget(self.howToPropagateRadioEntireObj, 0, 0)
        howToPropagateLayout.addWidget(self.howToPropagateRadioCurrent, 1, 0)
        howToPropagateLayout.addWidget(self.howToPropagateRadioToEnd, 2, 0)
        howToPropagateLayout.addWidget(self.howToPropagateRadioToStart, 3, 0)
        howToPropagateLayout.addWidget(self.howToPropagateRadioRange, 4, 0)

        self.howToPropagateButtonsGroup.addButton(
            self.howToPropagateRadioCurrent
        )
        self.howToPropagateButtonsGroup.addButton(
            self.howToPropagateRadioEntireObj
        )
        self.howToPropagateButtonsGroup.addButton(
            self.howToPropagateRadioToEnd
        )
        self.howToPropagateButtonsGroup.addButton(
            self.howToPropagateRadioToStart
        )
        self.howToPropagateButtonsGroup.addButton(
            self.howToPropagateRadioRange
        )

        self.howToPropagateRangeStartZSpinbox.setStartValue(zSliceStartValue)
        self.howToPropagateRangeEndZSpinbox.setStartValue(zSliceStartValue)

        howToPropagateLayout.addWidget(QLabel('From z-slice'), 4, 1)
        howToPropagateLayout.addWidget(QLabel('To z-slice'), 4, 2)
        howToPropagateLayout.addWidget(
            self.howToPropagateRangeStartZSpinbox, 5, 1
        )
        howToPropagateLayout.addWidget(
            self.howToPropagateRangeEndZSpinbox, 5, 2
        )
        howToPropagateLayout.addWidget(self.howToPropagateRadioRange, 5, 0)

        self.howToPropagateRadioRange.toggled.connect(
            self.howToPropagateRadioRangeToggled
        )
        self.howToPropagateRangeStartZSpinbox.sigValueChanged.connect(
            self.howToPropagateRangeStartZSpinboxValueChanged
        )

        return howToPropagateGroupbox

    def howToPropagateRangeStartZSpinboxValueChanged(self, value):
        self.howToPropagateRangeEndZSpinbox.setMinimum(value)
        if value > self.howToPropagateRangeEndZSpinbox.value():
            self.howToPropagateRangeEndZSpinbox.setValue(value)

    def howToPropagateRadioRangeToggled(self, checked: bool):
        self.howToPropagateRangeStartZSpinbox.setDisabled(not checked)
        self.howToPropagateRangeEndZSpinbox.setDisabled(not checked)
        start_z = self.howToPropagateRangeStartZSpinbox.value()
        end_z = self.howToPropagateRangeEndZSpinbox.value()
        if start_z > end_z:
            self.howToPropagateRangeEndZSpinbox.setValue(start_z)

    def onTextChanged(self, text):
        self.warnExistingIDLabel.setText('')
        try:
            ID = int(text)
            if ID in self.allIDs:
                self.warnExistingIDLabel.setText(
                    f'WARNING: ID {ID} was already used'
                )
        except Exception as err:
            pass
        
        # Get inserted char
        idx = self.entryWidget.cursorPosition()
        if idx == 0:
            return

        newChar = text[idx-1]

        # Do nothing if user is deleting text
        if idx == 0 or len(text)<len(self.prevText):
            self.prevText = text
            return

        # Do not allow chars except for "(", ")", "int", ","
        m = re.search(r'\(|\)|\d|,', newChar)
        if m is None:
            self.prevText = text
            text = text.replace(newChar, '')
            self.entryWidget.setText(text)
            return

        # Cast integers greater than uint32 machine limit
        m_iter = re.finditer(r'\d+', self.entryWidget.text())
        for m in m_iter:
            val = int(m.group())
            uint32_max = np.iinfo(np.uint32).max
            if val > uint32_max:
                text = self.entryWidget.text()
                text = f'{text[:m.start()]}{uint32_max}{text[m.end():]}'
                self.entryWidget.setText(text)

        # Automatically close ( bracket
        if newChar == '(':
            text += ')'
            self.entryWidget.setText(text)
        self.prevText = text
    
    def _warnExistingID(self, existingID, newID):
        warn_msg = html_utils.paragraph(f"""
            ID {existingID} is <b>already existing</b>.<br><br>
            How do you want to proceed?<br>
        """)
        msg = widgets.myMessageBox()
        doNotAskAgainCheckbox = QCheckBox('Remember my choice and do not ask again')
        swapButton = widgets.reloadPushButton(f'Swap {newID} with {existingID}')
        mergeButton = widgets.mergePushButton(f'Merge {newID} with {existingID}')
        msg.warning(
            self, 'Existing ID', warn_msg, 
            buttonsTexts=('Cancel', mergeButton, swapButton),
            widgets=doNotAskAgainCheckbox
        )
        if msg.cancel:
            return False
        self.doNotAskAgainExistingID = doNotAskAgainCheckbox.isChecked()
        if msg.clickedButton == swapButton:
            self.mergeWithExistingID = False
        else:
            self.mergeWithExistingID = True
        return True

    def assignNewIDclicked(self):
        self.cancel = False
        self.how = None
        self.assignNewID = True
        self.howToPropagateZslices = self.getHowToPropagateZslices()
        self.close()
    
    def ok_cb(self, event):
        txt = self.entryWidget.text()
        valid = False

        # Check validity of inserted text
        try:
            ID = int(txt)
            how = [(self.clickedID, ID)]
            if ID in self.IDs and not self.doNotAskAgainExistingID:
                proceed = self._warnExistingID(self.clickedID, ID)
                if not proceed:
                    return
                if not self.mergeWithExistingID:
                    how.append((ID, self.clickedID))
                valid = True
            else:
                valid = True
        except ValueError:
            pattern = r'\((\d+),\s*(\d+)\)'
            fa = re.findall(pattern, txt)
            if fa:
                how = [(int(g[0]), int(g[1])) for g in fa]
                valid = True
            else:
                valid = False

        if not valid:
            err_msg = html_utils.paragraph(
                'You entered invalid text. Valid text is either a single integer'
                f' ID that will be used to replace ID {self.clickedID} '
                'or a list of elements enclosed in parenthesis separated by a comma<br>'
                'such as (5, 10), (8, 27) to replace ID 5 with ID 10 and ID 8 with ID 27'
            )
            msg = widgets.myMessageBox()
            msg.warning(
                self, 'Invalid entry', err_msg
            )
            return
        
        self.cancel = False
        self.how = how
        self.doPropagateFutureFrames = False
        if self.propagateCheckbox is not None:
            self.doPropagateFutureFrames = self.propagateCheckbox.isChecked()
        
        self.howToPropagateZslices = self.getHowToPropagateZslices()
        self.close()
    
    def setHowToPropagateZslices(self, how: str, z_range):
        z_start, z_end = z_range
        self.howToPropagateRangeStartZSpinbox.setValue(z_start)
        self.howToPropagateRangeEndZSpinbox.setValue(z_end)
        for button in self.howToPropagateButtonsGroup.buttons():
            if button.text() != how:
                continue
            
            button.setChecked(True)
            break

    def getHowToPropagateZslices(self):
        if not self.isSegm3D:
            return
        
        how = self.howToPropagateButtonsGroup.checkedButton().text()
        z_range = (
            self.howToPropagateRangeStartZSpinbox.value(), 
            self.howToPropagateRangeEndZSpinbox.value()
        )
        return how, z_range

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()


class manualSeparateGui(QMainWindow):
    def __init__(
            self, lab, ID, img, fontSize='12pt', IDcolor=[255, 255, 0],
            parent=None, loop=None, drawMode='threepoints_arc', start_slice=None,
            mouseBindings=None, labelsLut=None, labelsAlpha=0.3, rp=None,
            img_lut_df=None,
        ):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.loop = loop
        self._parent = parent
        self.setSessionData(
            lab, ID, img, fontSize, IDcolor, drawMode, start_slice,
            mouseBindings, labelsLut, labelsAlpha, rp=rp, img_lut_df=img_lut_df
        )
        self.setWindowTitle("Split object")
        # self.setGeometry(Left, Top, 850, 800)

        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()

        self.gui_createStatusBar()

        self.img_lut_df = img_lut_df
        self.gui_createGraphics()
        self.gui_connectImgActions()

        self.gui_createImgWidgets()
        self.gui_connectActions()

        self.updateImg()
        self.zoomToObj()

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.graphLayout, 0, 0, 1, 1)
        mainLayout.addLayout(self.img_Widglayout, 1, 0)

        mainContainer.setLayout(mainLayout)

        self.setWindowModality(Qt.WindowModal)

        if self.img_lut_df is not None:
            self.imgGrad.restoreState(self.img_lut_df)
            
    def setSessionData(
            self, lab, ID, img, fontSize='12pt', IDcolor=[255, 255, 0],
            drawMode='threepoints_arc', start_slice=None, mouseBindings=None,
            labelsLut=None, labelsAlpha=0.3, rp=None, img_lut_df=None
        ):

        self.is_3D_mode = lab.ndim == 3
        if rp is None:
            from cellacdc import regionprops
            rp = regionprops.acdcRegionprops(self.lab, precache_centroids=False)
        obj = rp.get_obj_from_ID(ID)

        self.lab = np.zeros_like(lab)
        self.lab[obj.slice][obj.image] = ID

        if self.is_3D_mode:
            # skimage regionprops bbox for 3D object is (min_z, min_row, min_col, max_z, max_row, max_col)
            min_row, min_col, max_row, max_col = obj.bbox[1:3] + obj.bbox[4:6]
        else:
            min_row, min_col, max_row, max_col = obj.bbox

        self.objectBBox = (min_row, min_col, max_row, max_col)

        self.ID = ID
        self.img = img
        self.IDcolor = IDcolor
        self.fontSize = fontSize
        self.drawMode = drawMode
        self.mouseBindings = mouseBindings or {}
        self.labelsLut = labelsLut
        self.labelsAlpha = labelsAlpha
        self.start_slice = start_slice if start_slice is not None else 0
        self.current_slice = self.start_slice
        self.cancel = True
        self.countClicks = 0
        self.prevLabs = []
        self.prevAllCutsCoords = []
        self.undoIdx = 0
        self.sliceCutsCoords = {}
        self.AllCutsCoords = self.sliceCutsCoords.setdefault(
            self.current_slice, []
        )

        if not hasattr(self, 'imgItem'):
            self.labelItemsIDs = []
            return
        
        self.img_lut_df = img_lut_df
        if self.img_lut_df is not None:
            self.imgGrad.restoreState(self.img_lut_df)

        self.undoAction.setEnabled(False)
        self.threePointsArcAction.setChecked(
            drawMode == 'threepoints_arc'
        )
        self.freeHandAction.setChecked(drawMode == 'freehand')
        self.use2DsepAction.setEnabled(self.is_3D_mode)
        self.use2DsepAction.setVisible(self.is_3D_mode)
        self.warnLabel.setText(
            html_utils.paragraph(
                'Hold "Ctrl" to apply separation on all slices',
            ) if self.is_3D_mode else ''
        )
        self.labelsLayer.setOpacity(labelsAlpha)
        self.alphaScrollBar.setValue(round(labelsAlpha*40))
        self.alphaScrollBar.setVisible(self.overlayButton.isChecked())
        self.alphaScrollBar_label.setVisible(self.overlayButton.isChecked())
        self.zSliceScrollBar.blockSignals(True)
        self.zSliceScrollBar.setMaximum(
            self.lab.shape[0] - 1 if self.is_3D_mode else 0
        )
        self.zSliceScrollBar.setValue(self.current_slice)
        self.zSliceScrollBar.blockSignals(False)
        self.zSliceScrollBar.setVisible(self.is_3D_mode)
        self.zSliceLabel.setVisible(self.is_3D_mode)
        self._clearDrawing()
        self.updateImg()
        self.zoomToObj()

    def _clearDrawing(self):
        self.curvHoverPlotItem.setData([], [])
        self.lineHoverPlotItem.setData([], [])
        self.curvAnchors.setData([], [])
        self.freeHandItem.setData([], [])

    def centerWindow(self):
        parent = self._parent
        if parent is not None:
            # Center the window on main window
            mainWinGeometry = parent.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinCenterX = int(mainWinLeft + mainWinWidth/2)
            mainWinCenterY = int(mainWinTop + mainWinHeight/2)
            winGeometry = self.geometry()
            winWidth = winGeometry.width()
            winHeight = winGeometry.height()
            winLeft = int(mainWinCenterX - winWidth/2)
            winRight = int(mainWinCenterY - winHeight/2)
            self.move(winLeft, winRight)

    def gui_createActions(self):
        # File actions
        self.exitAction = QAction("&Exit", self)
        self.helpAction = QAction('Help', self)
        self.undoAction = QAction(QIcon(":undo.svg"), "Undo (Ctrl+Z)", self)
        self.undoAction.setEnabled(False)
        self.undoAction.setShortcut("Ctrl+Z")

        self.okAction = QAction(QIcon(":applyCrop.svg"), "Happy with that", self)
        self.cancelAction = QAction(QIcon(":cancel.svg"), "Cancel", self)

        self.drawModesActionGroup = QActionGroup(self)

        self.threePointsArcAction = QAction(
            QIcon(":threepoints_arc.svg"), 'Separate with three-points arc', 
            self
        )
        self.threePointsArcAction.setCheckable(True)
        self.threePointsArcAction.drawMode = 'threepoints_arc'
        self.drawModesActionGroup.addAction(self.threePointsArcAction)

        self.freeHandAction = QAction(
            QIcon(":freehand.svg"), 'Separate with freehand line', self
        )
        self.freeHandAction.setCheckable(True)
        self.freeHandAction.drawMode = 'freehand'
        self.drawModesActionGroup.addAction(self.freeHandAction)

        if self.drawMode == 'threepoints_arc':
            self.threePointsArcAction.setChecked(True)
        elif self.drawMode == 'freehand':
            self.freeHandAction.setChecked(True)
            
        self.swapIDsAction = QAction(
            QIcon(":reload.svg"), "Swap IDs", self
        )
        self.swapIDsAction.setToolTip(
            'Swap IDs of the two largest objects\n\n'
            'Shortcut: "S"'
        )
        self.swapIDsAction.setShortcut('S')

        self.nextAction = QAction('Next z-slice', self)
        self.nextAction.setShortcut(Qt.Key_Right)
        self.prevAction = QAction('Previous z-slice', self)
        self.prevAction.setShortcut(Qt.Key_Left)
        self.addAction(self.nextAction)
        self.addAction(self.prevAction)
        self.widgetsWithShortcut = {
            'Next': self.nextAction,
            'Previous': self.prevAction
        }
    
    def state(self):
        return {
            'is_overlay_active': self.overlayButton.isChecked(),
            'is_three_points_active': self.threePointsArcAction.isChecked(),
            'is_free_hand_active': self.freeHandAction.isChecked()
        }
    
    def show(self, block=False):
        super().show()
        if not block:
            return
        self.loop = QEventLoop(self)
        self.loop.exec_()
    
    def setState(self, state):
        if state is None:
            return
        self.overlayButton.setChecked(state.get('is_overlay_active', False))
        self.threePointsArcAction.setChecked(
            state.get('is_three_points_active', True)
        )
        self.freeHandAction.setChecked(state.get('is_free_hand_active', False))
        
    def gui_storeDrawMode(self):
        self.drawMode = self.sender().drawMode
        
    def gui_createMenuBar(self):
        menuBar = self.menuBar()
        # style = "QMenuBar::item:selected { background: white; }"
        # menuBar.setStyleSheet(style)
        # File menu
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)

        menuBar.addAction(self.helpAction)
        fileMenu.addAction(self.exitAction)

    def gui_createToolBars(self):
        toolbarSize = 30

        editToolBar = QToolBar("Edit", self)
        editToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(editToolBar)

        editToolBar.addAction(self.okAction)
        editToolBar.addAction(self.cancelAction)

        editToolBar.addAction(self.undoAction)

        self.overlayButton = QToolButton(self)
        self.overlayButton.setIcon(QIcon(":overlay.svg"))
        self.overlayButton.setCheckable(True)
        self.overlayButton.setToolTip(
            'Overlay channel\'s image'
        )
        editToolBar.addWidget(self.overlayButton)

        editToolBar.addAction(self.threePointsArcAction)
        editToolBar.addAction(self.freeHandAction)
        
        editToolBar.addAction(self.swapIDsAction)
        
        self.use2DsepAction = QToolButton(self)
        self.use2DsepAction.setIcon(QIcon(":separate-bud-2D.svg"))
        self.use2DsepAction.setToolTip(
            'Use 2D automatic separation on the current slice\n\n'
            'Shortcut: A'
        )
        editToolBar.addWidget(self.use2DsepAction)
        
        if not self.is_3D_mode:
            self.use2DsepAction.setEnabled(False)
            self.use2DsepAction.hide()
            
        self.use2DsepAction.setShortcut('A')
        
        self.warnLabel = QLabel()
        editToolBar.addWidget(self.warnLabel)
        
        if self.is_3D_mode:
            self.warnLabel.setText(
                html_utils.paragraph(
                    'Hold "Ctrl" to apply separation on all slices',
                )
            )

    def gui_connectActions(self):
        self.exitAction.triggered.connect(self.close)
        self.helpAction.triggered.connect(self.help)
        self.okAction.triggered.connect(self.ok_cb)
        self.cancelAction.triggered.connect(self.close)
        self.undoAction.triggered.connect(self.undo)
        self.overlayButton.toggled.connect(self.toggleOverlay)
        self.imgGrad.sigLookupTableChanged.connect(self.histLUT_cb)
        self.swapIDsAction.triggered.connect(self.swapIDs)
        self.nextAction.triggered.connect(self.nextSlice)
        self.prevAction.triggered.connect(self.previousSlice)
        self.use2DsepAction.clicked.connect(self.use2Dsep)

    def use2Dsep(self):
        lab = self.currentLab()
        if not np.any(lab == self.ID):
            self.warnLabel.setText(
                html_utils.paragraph(
                    'The selected object is not present on this z-slice',
                    font_color='red'
                )
            )
            return

        max_ID = int(self.lab.max())
        if self._parent is not None:
            posData = self._parent.data[self._parent.pos_i]
            max_ID = max(max_ID, max(posData.IDs, default=1))

        result, success, _ = core_split_IDs.split_along_convexity_defects(
            self.ID, lab, max_ID
        )
        if not success:
            self.warnLabel.setText(
                html_utils.paragraph(
                    'Automatic separation was not successful',
                    font_color='red'
                )
            )
            return

        self.storeUndoState()
        lab[:] = result
        self.warnLabel.setText('')
        self.updateImg()

    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        # Temporary message
        self.statusbar.showMessage("Ready", 3000)
        # Permanent widget
        self.wcLabel = QLabel(f"")
        self.statusbar.addPermanentWidget(self.wcLabel)

    def gui_createGraphics(self):
        self.graphLayout = pg.GraphicsLayoutWidget()

        # Plot Item container for image
        self.ax = pg.PlotItem()
        self.ax.invertY(True)
        self.ax.setAspectLocked(True)
        self.ax.hideAxis('bottom')
        self.ax.hideAxis('left')
        self.graphLayout.addItem(self.ax, row=1, col=1)

        # Image Item
        self.imgItem = pg.ImageItem(np.zeros((512,512)))
        self.ax.addItem(self.imgItem)

        self.labelsLayer = widgets.BaseLabelsImageItem()
        self.labelsLayer.setOpacity(self.labelsAlpha)
        self.ax.addItem(self.labelsLayer)

        #Image histogram
        self.imgGrad = widgets.myHistogramLUTitem()
        self.imgGrad.setImageItem(self.imgItem)
        
        if self.img_lut_df is not None:
            self.imgGrad.restoreState(self.img_lut_df)
            
        # Curvature items
        self.hoverLinSpace = np.linspace(0, 1, 1000)
        self.hoverLinePen = pg.mkPen(color=(200, 0, 0, 255*0.5),
                                     width=2, style=Qt.DashLine)
        self.hoverCurvePen = pg.mkPen(color=(200, 0, 0, 255*0.5), width=3)
        self.lineHoverPlotItem = pg.PlotDataItem(pen=self.hoverLinePen)
        self.curvHoverPlotItem = pg.PlotDataItem(pen=self.hoverCurvePen)
        self.curvAnchors = pg.ScatterPlotItem(
            symbol='o', size=9,
            brush=pg.mkBrush((255,0,0,50)),
            pen=pg.mkPen((255,0,0), width=2),
            hoverable=True, hoverPen=pg.mkPen((255,0,0), width=3),
            hoverBrush=pg.mkBrush((255,0,0))
        )
        self.ax.addItem(self.curvAnchors)
        self.ax.addItem(self.curvHoverPlotItem)
        self.ax.addItem(self.lineHoverPlotItem)

        self.freeHandItem = widgets.PlotCurveItem(
            pen=pg.mkPen(color='r', width=2)
        )
        self.ax.addItem(self.freeHandItem)

    def gui_createImgWidgets(self):
        self.img_Widglayout = QGridLayout()
        self.img_Widglayout.setContentsMargins(50, 0, 50, 0)

        alphaScrollBar_label = QLabel('Overlay alpha  ')
        alphaScrollBar = QScrollBar(Qt.Horizontal)
        alphaScrollBar.setFixedHeight(20)
        alphaScrollBar.setMinimum(0)
        alphaScrollBar.setMaximum(40)
        alphaScrollBar.setValue(round(self.labelsAlpha*40))
        alphaScrollBar.setToolTip(
            'Control the alpha value of the overlay.\n'
            'alpha=0 results in NO overlay,\n'
            'alpha=1 results in only labels visible'
        )
        alphaScrollBar.sliderMoved.connect(self.alphaScrollBarMoved)
        self.alphaScrollBar = alphaScrollBar
        self.alphaScrollBar_label = alphaScrollBar_label
        self.img_Widglayout.addWidget(
            alphaScrollBar_label, 0, 0, alignment=Qt.AlignCenter
        )
        self.img_Widglayout.addWidget(alphaScrollBar, 0, 1, 1, 20)
        self.alphaScrollBar.hide()
        self.alphaScrollBar_label.hide()

        self.zSliceScrollBar = QScrollBar(Qt.Horizontal)
        self.zSliceScrollBar.setMinimum(0)
        self.zSliceScrollBar.setMaximum(self.lab.shape[0]-1)
        self.zSliceScrollBar.setFixedHeight(20)
        self.zSliceScrollBar.setValue(self.current_slice)
        self.zSliceScrollBar.setToolTip('Navigate z-slices')
        self.zSliceScrollBar.valueChanged.connect(self.update_z_slice)
        self.zSliceLabel = QLabel()
        self.img_Widglayout.addWidget(
            self.zSliceLabel, 1, 0, alignment=Qt.AlignCenter
        )
        self.img_Widglayout.addWidget(self.zSliceScrollBar, 1, 1, 1, 20)
        if not self.is_3D_mode:
            self.zSliceLabel.hide()
            self.zSliceScrollBar.hide()

    def currentLab(self):
        if self.is_3D_mode:
            return self.lab[self.current_slice]
        return self.lab

    def currentImg(self):
        if self.is_3D_mode and self.img.ndim == 3:
            return self.img[self.current_slice]
        return self.img

    def update_z_slice(self, z):
        self.current_slice = z
        self.AllCutsCoords = self.sliceCutsCoords.setdefault(z, [])
        self.countClicks = 0
        self.curvHoverPlotItem.setData([], [])
        self.lineHoverPlotItem.setData([], [])
        self.curvAnchors.setData([], [])
        self.freeHandItem.setData([], [])
        self.zSliceLabel.setText(f'z-slice {z+1}/{self.lab.shape[0]}')
        self.updateImg()

    def nextSlice(self):
        if self.is_3D_mode:
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )

    def previousSlice(self):
        if self.is_3D_mode:
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepSub
            )

    def mousePressEvent(self, event):
        for name, button in self.mouseBindings.items():
            if button != event.button() or name not in self.widgetsWithShortcut:
                continue
            self.widgetsWithShortcut[name].trigger()
            event.accept()
            return True
        super().mousePressEvent(event)

    def gui_connectImgActions(self):
        self.imgItem.hoverEvent = self.gui_hoverEventImg
        self.imgItem.mousePressEvent = self.gui_mousePressEventImg
        self.imgItem.mouseMoveEvent = self.gui_mouseDragEventImg
        self.imgItem.mouseReleaseEvent = self.gui_mouseReleaseEventImg

    def gui_hoverEventImg(self, event):
        # Update x, y, value label bottom right
        try:
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            _img = self.currentLab()
            Y, X = _img.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                self.wcLabel.setText(f'(x={x:.2f}, y={y:.2f}, ID={val:.0f})')
            else:
                self.wcLabel.setText(f'')
        except Exception as e:
            self.wcLabel.setText(f'')

        if event.isExit():
            return

        self.drawHoverEvent(*event.pos())

    def gui_mousePressEventImg(self, event):
        right_click = event.button() == Qt.MouseButton.RightButton
        left_click = event.button() == Qt.MouseButton.LeftButton

        if not right_click and not left_click:
            self.mousePressEvent(event)
            return

        dragImg = (left_click)

        if dragImg:
            pg.ImageItem.mousePressEvent(self.imgItem, event)
        
        if not right_click:
            return

        self.drawPressEvent(event)
    
    def gui_mouseDragEventImg(self, event):
        pass

    def gui_mouseReleaseEventImg(self, event):
        if self.countClicks == 0:
            return
        if self.freeHandAction.isChecked():
            self.countClicks = 0
            xx, yy = self.freeHandItem.getData()
            self.applySplitCurve(xx, yy)
            self.freeHandItem.setData([], [])
            self.curvAnchors.setData([], [])
    
    def getSpline(self, xx, yy):
        tck, u = scipy.interpolate.splprep([xx, yy], s=0, k=2)
        xi, yi = scipy.interpolate.splev(self.hoverLinSpace, tck)
        return xi, yi

    def drawPressEvent(self, event):
        if self.freeHandAction.isChecked():
            self.countClicks = 1
            x, y = event.pos().x(), event.pos().y()
            self.curvAnchors.addPoints([x], [y])
        elif self.threePointsArcAction.isChecked():
            self.threePointsArcPressEvent(event)
    
    def drawHoverEvent(self, x, y):
        if self.freeHandAction.isChecked():
            self.freeHandHoverEvent(x, y)
        elif self.threePointsArcAction.isChecked():
            self.threePointsArcHoverEvent(x, y)
        
    def freeHandHoverEvent(self, x, y):
        if self.countClicks == 0:
            return
        self.freeHandItem.addPoint(int(x), int(y))
        _xx, _yy = self.freeHandItem.getData()
        xx = [_xx[0], x]
        yy = [_yy[0], y]
        self.curvAnchors.setData(xx, yy)

    def threePointsArcHoverEvent(self, x, y):
        if self.countClicks == 1:
            self.lineHoverPlotItem.setData([self.x0, x], [self.y0, y])
        elif self.countClicks == 2:
            xx = [self.x0, x, self.x1]
            yy = [self.y0, y, self.y1]
            xi, yi = self.getSpline(xx, yy)
            self.curvHoverPlotItem.setData(xi, yi)
        elif self.countClicks == 0:
            self.curvHoverPlotItem.setData([], [])
            self.lineHoverPlotItem.setData([], [])
            self.curvAnchors.setData([], [])
    
    def threePointsArcPressEvent(self, event):
        if self.countClicks == 0:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.x0, self.y0 = xdata, ydata
            self.curvAnchors.addPoints([xdata], [ydata])
            self.countClicks = 1
        elif self.countClicks == 1:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.x1, self.y1 = xdata, ydata
            self.curvAnchors.addPoints([xdata], [ydata])
            self.countClicks = 2
        elif self.countClicks == 2:
            self.countClicks = 0
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            xx = [self.x0, xdata, self.x1]
            yy = [self.y0, ydata, self.y1]
            xi, yi = self.getSpline(xx, yy)
            yy, xx = np.round(yi).astype(int), np.round(xi).astype(int)
            self.applySplitCurve(xx, yy)

    def applySplitCurve(self, xx, yy):
        cut_all_slices = (
            self.is_3D_mode
            and QApplication.keyboardModifiers()
            & Qt.ControlModifier
        )
        self.storeUndoState()
        if not cut_all_slices:
            previous_lab = self.currentLab().copy()
            cut_coords = self.applyCutCurve(xx, yy)
            if cut_coords is not None:
                self.splitObjectAlongCurve(
                    previous_lab, cut_coords, xx, yy
                )
            return

        current_slice = self.current_slice
        split_IDs = []
        self.AllCutsCoords = self.sliceCutsCoords.setdefault(current_slice, [])
        previous_lab = self.currentLab().copy()
        cut_coords = self.applyCutCurve(xx, yy)
        if cut_coords is not None:
            split_IDs = self.splitObjectAlongCurve(
                previous_lab, cut_coords, xx, yy
            )

        directions = (
            range(current_slice + 1, self.lab.shape[0]),
            range(current_slice - 1, -1, -1),
        )
        for slices in directions:
            for z in slices:
                self.current_slice = z
                self.AllCutsCoords = self.sliceCutsCoords.setdefault(z, [])
                previous_lab = self.currentLab().copy()
                cut_coords = self.applyCutCurve(xx, yy)
                if cut_coords is None:
                    continue
                new_split_IDs = self.splitObjectAlongCurve(
                    previous_lab, cut_coords, xx, yy,
                    split_IDs=split_IDs,
                )
                split_IDs.extend(
                    ID for ID in new_split_IDs if ID not in split_IDs
                )
        self.current_slice = current_slice
        self.AllCutsCoords = self.sliceCutsCoords.setdefault(current_slice, [])
        self.updateImg()
    
    def applyCutCurve(self, xx, yy):
        lab = self.currentLab()
        xxCurve, yyCurve = [], []
        for i, (r0, c0) in enumerate(zip(yy, xx)):
            if i == len(yy)-1:
                break
            r1 = yy[i+1]
            c1 = xx[i+1]
            rr, cc, _ = skimage.draw.line_aa(r0, c0, r1, c1)
            # rr, cc = skimage.draw.line(r0, c0, r1, c1)
            valid = (
                (rr > 0) & (rr < lab.shape[0] - 1)
                & (cc > 0) & (cc < lab.shape[1] - 1)
            )
            rr, cc = rr[valid], cc[valid]
            nonzeroMask = lab[rr, cc]>0
            xxCurve.extend(cc[nonzeroMask])
            yyCurve.extend(rr[nonzeroMask])
        if not xxCurve:
            return None
        self.AllCutsCoords.append((yyCurve, xxCurve))
        return yyCurve, xxCurve

    def histLUT_cb(self, LUTitem):
        if self.overlayButton.isChecked():
            self.imgItem.setImage(self.currentImg())
        elif hasattr(self, 'lut'):
            self.imgItem.setLookupTable(self.lut)

    def swapIDs(self, checked=False):
        IDs, areas = np.unique(self.lab[self.lab > 0], return_counts=True)
        if len(IDs) < 2:
            self.warnLabel.setText(
                html_utils.paragraph(
                    'WARNING: At least two objects must be present to swap IDs',
                    font_color='red'
                )
            )
            return
        
        self.warnLabel.setText('')

        largest = np.argsort(areas, kind='stable')[-2:]
        obj1_ID, obj2_ID = IDs[largest]
        obj1_mask = self.lab == obj1_ID
        obj2_mask = self.lab == obj2_ID
        
        self.lab[obj1_mask] = obj2_ID
        self.lab[obj2_mask] = obj1_ID
        
        self.updateImg()
    
    def updateImg(self):
        from cellacdc import regionprops as acdc_regionprops

        self.updateLookuptable()
        lab = self.currentLab()
        rp = acdc_regionprops.acdcRegionprops(lab)
        self.rp = rp

        if self.overlayButton.isChecked():
            self.applyImgGradLut()
            self.imgItem.setImage(self.currentImg())
            self.labelsLayer.show()
        else:
            self.imgItem.setLookupTable(self.lut)
            self.imgItem.setImage(lab)
            self.labelsLayer.hide()
        self.labelsLayer.setImage(lab)

        # Draw ID on centroid of each label
        for labelItemID in self.labelItemsIDs:
            self.ax.removeItem(labelItemID)
        self.labelItemsIDs = []
        for obj in rp:
            labelItemID = widgets.myLabelItem()
            labelItemID.setText(
                f'{obj.label}', color='r', size=f'{self.fontSize}px'
            )
            y, x = obj.centroid
            w, h = labelItemID.rect().right(), labelItemID.rect().bottom()
            labelItemID.setPos(x-w/2, y-h/2)
            self.labelItemsIDs.append(labelItemID)
            self.ax.addItem(labelItemID)

    def zoomToObj(self):
        # Zoom to object
        (min_row, min_col, max_row, max_col) = self.objectBBox

        xRange = min_col-10, max_col+10
        yRange = max_row+10, min_row-10
        self.ax.setRange(xRange=xRange, yRange=yRange)

    def storeUndoState(self):
        self.prevLabs.append(self.lab.copy())
        self.prevAllCutsCoords.append({
            z: cuts.copy() for z, cuts in self.sliceCutsCoords.items()
        })
        self.undoIdx += 1
        self.undoAction.setEnabled(True)

    def undo(self):
        self.undoIdx -= 1
        if self.undoIdx < 0:
            self.undoIdx = 0
            return
        self.lab = self.prevLabs[self.undoIdx]
        self.sliceCutsCoords = self.prevAllCutsCoords[self.undoIdx]
        self.AllCutsCoords = self.sliceCutsCoords.setdefault(
            self.current_slice, []
        )
        self.updateImg()
        if self.undoIdx == 0:
            self.undoAction.setEnabled(False)
            self.prevLabs = []
            self.prevAllCutsCoords = []

    def splitObjectAlongCurve(
            self, previous_lab, cut_coords, curve_x, curve_y,
            split_IDs=None
        ):
        lab = self.currentLab()

        if self._parent is not None:
            self._parent.setBrushID()
            posData = self._parent.data[self._parent.pos_i]
            max_ID = max(int(posData.brushID) - 1, int(self.lab.max()))
        else:
            max_ID = int(self.lab.max())

        lab, max_ID, new_IDs = core_split_IDs.split_cut_components(
            previous_lab, lab, cut_coords, curve_y, curve_x, max_ID,
            split_IDs=split_IDs,
            bbox=self.objectBBox
        )
        if self._parent is not None:
            posData.brushID = max(posData.brushID, max_ID)

        if not self.is_3D_mode:
            self.cutLab = lab
            self.updateImg()
            return new_IDs

        neighboring_labs = [
            self.lab[z]
            for z in (self.current_slice - 1, self.current_slice + 1)
            if 0 <= z < self.lab.shape[0]
        ]
        tracked_lab, _ = core_split_IDs.track_split_slice(
            lab,
            neighboring_labs,
            unique_ID=max_ID + 1,
        )
        lab[:] = tracked_lab
        if self._parent is not None:
            posData.brushID = max(posData.brushID, int(lab.max()))

        self.cutLab = lab
        self.updateImg()
        return [
            int(ID) for ID in np.unique(lab)
            if ID not in (0, self.ID)
        ]

    def updateLookuptable(self):
        # Lookup table
        lab = self.currentLab()
        self.cmap = colors.getFromMatplotlib('viridis')
        self.lut = self.cmap.getLookupTable(0,1,self.lab.max()+1)
        self.lut[0] = [25,25,25]
        self.lut[self.ID] = self.IDcolor
        if self.labelsLut is None or len(self.labelsLut) <= lab.max():
            labelsLut = np.zeros((len(self.lut), 4), dtype=np.uint8)
            labelsLut[:, :3] = self.lut
            labelsLut[:, 3] = 255
            labelsLut[0] = [0, 0, 0, 0]
        else:
            labelsLut = self.labelsLut
        self.labelsLayer.setLevels([0, len(labelsLut)])
        self.labelsLayer.setLookupTable(labelsLut)
        if self.overlayButton.isChecked():
            self.applyImgGradLut()
        else:
            self.imgItem.setLookupTable(self.lut)

    def applyImgGradLut(self):
        if self.imgGrad.gradient.isLookupTrivial():
            self.imgItem.setLookupTable(None)
        else:
            self.imgItem.setLookupTable(self.imgGrad.getLookupTable)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.countClicks = 0
            self.curvHoverPlotItem.setData([], [])
            self.lineHoverPlotItem.setData([], [])
            self.curvAnchors.setData([], [])
            self.freeHandItem.setData([], [])
        elif ev.key() == Qt.Key_Enter or ev.key() == Qt.Key_Return:
            self.ok_cb(True)

    def alphaScrollBarMoved(self, alpha_int):
        self.labelsLayer.setOpacity(
            alpha_int/self.alphaScrollBar.maximum()
        )

    def toggleOverlay(self, checked):
        if checked:
            self.graphLayout.addItem(self.imgGrad, row=1, col=0)
            self.alphaScrollBar.show()
            self.alphaScrollBar_label.show()
        else:
            self.graphLayout.removeItem(self.imgGrad)
            self.alphaScrollBar.hide()
            self.alphaScrollBar_label.hide()
        self.updateImg()

    def help(self):
        msg = QMessageBox()
        msg.information(self, 'Help',
            'Separate object along a curved line.\n\n'
            'To draw a curved line you will need 3 right-clicks:\n\n'
            '1. Right-click outside of the object --> a line appears.\n'
            '2. Right-click to end the line and a curve going through the '
            'mouse cursor will appear.\n'
            '3. Once you are happy with the cutting curve right-click again '
            'and the object will be separated along the curve.\n\n'
            'Note that you can separate as many times as you want.\n\n'
            'Once happy click on the green tick on top-right or '
            'cancel the process with the "X" button')

    def ok_cb(self, checked):
        self.cancel = False
        self.close()

    def closeEvent(self, event):
        if self.loop is not None:
            self.loop.exit()


def get_cca_colname_desc():
    desc = {
        'Cell ID': (
            'ID of the segmented cell. All of the other columns '
            'are properties of this ID.'
        ),
        'Cell cycle stage': (
            'G1 if the cell does NOT have a bud. S/G2/M if it does.'
        ),
        'Relative ID': (
            'ID of the bud related to the Cell ID (row). For cells in G1 write the '
            'bud ID it had in the previous cycle.'
        ),
        'Generation number': (
            'Number of times the cell divided from a bud. For cells in the first '
            'frame write any number greater than 1.'
        ),
        'Relationship': (
            'Relationship of the current Cell ID (row). '
            'Either <b>mother</b> or <b>bud</b>. An object is a bud if '
            'it didn\'t divide from the mother yet. All other instances '
            '(e.g., cell in G1) are still labelled as mother.'
        ),
        'Emerging frame num.': (
            'Frame number at which the object emerged/appeared in the scene.'
        ),
        'Division frame num.': (
            'Frame number at which the bud separated from the mother.'
        ),
        'Is history known?': (
            'Cells that are already present in the first frame or appears '
            'from outside of the field of view, have some information missing. '
            'For example, for cells in the first frame we do not know how many '
            'times it budded and divided in the past. '
            'In these cases Is history known? is True.'
        )
    }
    return desc


def format_cca_manual_changes(changes: dict):
    txt = ''
    for ID, changes_ID in changes.items():
        txt = f'{txt}* ID {ID}:\n'
        for col, (old_val, new_val) in changes_ID.items():
            txt = f'{txt}    - {col}: {old_val} --> {new_val}\n'
        txt = f'{txt}--------------------------------\n\n'
    return txt
