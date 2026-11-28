"""File, folder and Position selection dialogs used by several program
parts and batch utilities (from apps.py).

filenameDialog with FilenameEntryFormatter, SelectSegmFileDialog,
SelectFoldersToAnalyse (MODULES_TYPES is defined above it: its signature
annotation is evaluated at import), get_existing_directory, the Position
selection that load.py asks for (SelectPositionFoldersDialogue) and
selectPositionsMultiExp (spotMAX).
"""
import os
import sys
import re
from typing import Literal, Iterable
from collections import defaultdict
from natsort import natsorted
from functools import partial
import html
from qtpy.QtGui import QIcon, QKeySequence
from qtpy.QtCore import Qt, QEventLoop, QTimer
from qtpy.QtWidgets import (
    QLabel, QVBoxLayout, QHBoxLayout, QDialog, QAbstractItemView, QCheckBox,
    QGridLayout, QStyle, QTreeWidget, QTreeWidgetItem
)
import qtpy.compat
from cellacdc import _window_stacking
from cellacdc import load, html_utils
from cellacdc import is_mac
from cellacdc.myutils import filesystem
from ..widgets import (
    item_views as widgets_item_views,
    line_edits as widgets_line_edits,
    message_box as widgets_message_box,
    push_buttons as widgets_push_buttons,
)
from cellacdc.acdc_regex import is_alphanumeric_filename, to_alphanumeric
from cellacdc import path
from cellacdc import fonts
from . import base as dialogs_base
from ..widgets import style as widgets_style
from . import generic as dialogs_generic


class FilenameEntryFormatter:
    @staticmethod
    def format_entry_text(text: str) -> str:
        text = to_alphanumeric(text or '')
        text = text.replace(' ', '_')
        return text.replace('.', '_')

    @staticmethod
    def build_filename_preview(basename: str, text: str, ext: str) -> str:
        if not text:
            return f'{basename}{ext}'

        text = text.replace(' ', '_')
        if basename:
            if basename.endswith('_'):
                return f'{basename}{text}{ext}'
            return f'{basename}_{text}{ext}'

        return f'{text}{ext}'

    @staticmethod
    def invalid_chars_warning_text(characters: set[str]) -> str:
        statement = 'is <b>not a valid</b> character'
        if len(characters) > 1:
            statement = 'are <b>not valid</b> characters'

        characters_str = ''.join(characters)
        characters_str = html.escape(characters_str)
        warning_text = html_utils.span(f"""
            WARNING: "<code>{characters_str}</code>" {statement}.<br>
        """)
        return (
            f'{warning_text}'
            '<i>Valid characters are letters, numbers, underscore, and dash.</i>'
        )

    @staticmethod
    def is_valid_name(name: str) -> bool:
        return bool(name) and is_alphanumeric_filename(name)

class filenameDialog(QDialog):
    def __init__(
            self, ext='.npz', basename='', title='Insert file name',
            hintText='', existingNames='', parent=None, allowEmpty=True,
            helpText='', defaultEntry='', resizeOnShow=True,
            additionalButtons=None, addDoNotSaveButton=False,
            caseInsensitiveNames=False
        ):
        self.cancel = True
        super().__init__(parent)

        self.resizeOnShow = resizeOnShow
        # If True, `existingNames` are matched ignoring the case (e.g., names
        # of files on case-insensitive file systems)
        self.caseInsensitiveNames = caseInsensitiveNames

        if hintText.find('segmentation') != -1:
            if helpText:
                helpText = (f'{helpText}')
            helpText_loc = ("""
                With Cell-ACDC you can create as many segmentation files 
                <b>as you want</b>.<br><br>
                If you plan to create <b>only one file</b> then you can leave the 
                text entry <b>empty</b>.<br>
                Cell-ACDC will save the segmentation file with the filename 
                ending with <code>_segm.npz</code>.<br><br>
                However, <b>we recommend to insert some text</b> that will easily 
                allow you <b>to identify</b> what is the segmentation file about.<br><br>
                For example, if you are about to segment the channel 
                <code>phase_contr</code>, you could write 
                <code>phase_contr</code>.<br>
                Cell-ACDC will then save the file with the
                filename ending with <code>_segm_phase_contr.npz</code>.<br><br>
                This way you can create <b>multiple segmentation files</b>, 
                for example one for each channel or one for each segmentation model.<br><br>
                Note that the <b>numerical features and annotations</b> will be saved 
                in a CSV file ending with the same text as the segmentation file,<br> 
                e.g., ending with <code>_acdc_output_phase_contr.csv</code>.
            """)
            helpText = (f'{helpText}{html_utils.paragraph(helpText_loc)}')

        self.isSegmFile = basename.endswith('_segm')
        self.allowEmpty = allowEmpty
        self.basename = basename
        if ext and not ext.startswith('.'):
            ext = f'.{ext}'
        self.ext = ext

        self.setWindowTitle(title)
        _window_stacking.set_window_flags_on_top(self, Qt.Window)

        layout = QVBoxLayout()
        entryLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        hintLabel = QLabel(hintText)

        basenameLabel = QLabel(basename)

        self.lineEdit = widgets_line_edits.alphaNumericLineEdit(onlyWarn=True)
        self.lineEdit.setAlignment(Qt.AlignCenter)
        defaultEntry = FilenameEntryFormatter.format_entry_text(defaultEntry)
        self.lineEdit.setText(defaultEntry)

        extLabel = QLabel(ext)

        self.filenameLabel = QLabel()
        self.filenameLabel.setText(f'{basename}{ext}')

        entryLayout.addWidget(basenameLabel, 0, 1)
        entryLayout.addWidget(self.lineEdit, 0, 2)
        entryLayout.addWidget(extLabel, 0, 3)
        entryLayout.addWidget(
            self.filenameLabel, 1, 1, 1, 3, alignment=Qt.AlignCenter
        )
        # entryLayout.setColumnStretch(0, 1)
        entryLayout.setColumnStretch(2, 1)
        
        self.warningInvalidCharLabel = QLabel()

        okButton = widgets_push_buttons.okPushButton('Ok')
        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        self.okButton = okButton

        buttonsLayout.addStretch()
        buttonsLayout.addWidget(cancelButton)

        if addDoNotSaveButton:
            doNotSaveButton = widgets_push_buttons.noPushButton('Do not save')
            doNotSaveButton.clicked.connect(self.doNotSave_cb)
            buttonsLayout.addWidget(doNotSaveButton)
            self.doNotSave = False

        buttonsLayout.addSpacing(20)
        if helpText:
            helpButton = widgets_push_buttons.helpPushButton('Help...')
            helpButton.clicked.connect(partial(self.showHelp, helpText))
            buttonsLayout.addWidget(helpButton)
        if additionalButtons is not None:
            for button in additionalButtons:
                buttonsLayout.addWidget(button)
        buttonsLayout.addWidget(okButton)

        cancelButton.clicked.connect(self.close)
        okButton.clicked.connect(self.ok_cb)
        self.lineEdit.textChanged.connect(self.updateFilename)
        self.lineEdit.sigInvalidCharactersEntered.connect(
            self.warnInvalidCharactersEntered
        )
        
        self.existingNames = []
        if existingNames:
            self.existingNames = existingNames
            # self.lineEdit.editingFinished.connect(self.checkExistingNames)

        layout.addWidget(hintLabel)
        layout.addSpacing(20)
        layout.addLayout(entryLayout)
        layout.addSpacing(10)
        layout.addWidget(self.warningInvalidCharLabel)
        layout.addStretch(1)
        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)

        self.setLayout(layout)
        self.setFont(fonts.font)

        if defaultEntry:
            self.updateFilename(defaultEntry)
    
    def doNotSave_cb(self):
        msg = widgets_message_box.myMessageBox()
        txt = html_utils.paragraph(
            'Are you sure you do <b>not</b> want to save the file?'
        )
        noButton, yesButton = msg.warning(
            self, 'Do not save?', txt, buttonsTexts=('No', 'Yes')
        )
        if msg.clickedButton == noButton:
            return
        
        self.doNotSave = True
        self.cancel = False
        self.close()
        
    def showHelp(self, text):
        text = html_utils.paragraph(text)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.information(self, 'Filename help', text)

    def _text(self):
        return self.lineEdit.text()
    
    def warnInvalidCharactersEntered(self, characters: set[str]):
        warning_text = FilenameEntryFormatter.invalid_chars_warning_text(
            characters
        )
        self.warningInvalidCharLabel.setText(warning_text)

    def checkExistingNames(self):
        candidates = (self._text(), self.filenameLabel.text())
        existingNames = self.existingNames
        if self.caseInsensitiveNames:
            candidates = [name.casefold() for name in candidates]
            existingNames = [name.casefold() for name in existingNames]
        is_existing = any(name in existingNames for name in candidates)
        if not is_existing:
            return True

        filename = self.filenameLabel.text()
        msg = widgets_message_box.myMessageBox()
        txt = html_utils.paragraph(
            'The following file<br><br>'
            f'<code>{filename}</code><br><br>'
            'is <b>already existing</b>.<br><br>'
            'Do you want to <b>overwrite</b> the existing file?'
        )
        noButton, yesButton = msg.warning(
            self, 'File name existing', txt, buttonsTexts=('No', 'Yes')
        )
        return msg.clickedButton == yesButton

    def updateFilename(self, text):
        if self.lineEdit.invalidCharacters():
            return

        self.filenameLabel.setText(
            FilenameEntryFormatter.build_filename_preview(
                self.basename, text, self.ext
            )
        )
        
        self.warningInvalidCharLabel.setText('')

    def checkEmptyText(self):
        if self.allowEmpty:
            return True
        
        if self._text():
            return True
       
        msg = widgets_message_box.myMessageBox()
        msg.critical(
            self, 'Empty text', 
            html_utils.paragraph('Text entry field <b>cannot be empty</b>')
        )
        return False
    
    def checkSegmFilename(self):
        if not self.isSegmFile:
            return True
        
        if 'segm' not in self._text():
            return True
       
        msg = widgets_message_box.myMessageBox()
        txt = html_utils.paragraph(
            'The text appended to the filename cannot contain the text '
            '"segm".<br><br>'
            'Sorry, that would confuse me. Thank you for your patience!'
        )
        msg.critical(
            self, 'Cannot use "segm" in filename', txt
        )
        return False
    
    def ok_cb(self, checked=True):
        if self.warningInvalidCharLabel.text():
            return
        
        valid = self.checkExistingNames()
        if not valid:
            return
        
        valid = self.checkEmptyText()
        if not valid:
            return
        
        valid = self.checkSegmFilename()
        if not valid:
            return
        
        self.filename = self.filenameLabel.text()
        self.entryText = self._text()
        self.cancel = False
        self.close()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        if self.resizeOnShow:
            self.lineEdit.setMinimumWidth(self.lineEdit.width()*2)
        self.okButton.setDefault(True)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()


class selectPositionsMultiExp(dialogs_base.QBaseDialog):
    def __init__(self, expPaths: dict, infoPaths: dict=None, parent=None):
        super().__init__(parent=parent)

        self.expPaths = expPaths
        self.cancel = True

        mainLayout = QVBoxLayout()

        self.setWindowTitle('Select Positions to process')

        infoTxt = html_utils.paragraph(
            'Select one or more Positions to process<br><br>'
            '<code>Click</code> on experiment path <i>to select all positions</i><br>'
            '<code>Ctrl+Click</code> <i>to select multiple items</i><br>'
            '<code>Shift+Click</code> <i>to select a range of items</i><br>',
            center=True
        )
        infoLabel = QLabel(infoTxt)

        self.treeWidget = QTreeWidget()
        self.treeWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.treeWidget.setHeaderHidden(True)
        self.treeWidget.setFont(fonts.font)
        for exp_path, positions in expPaths.items():
            pathLevels = exp_path.split(os.sep)
            posFoldersInfo = None
            if infoPaths is not None:
                posFoldersInfo = infoPaths.get(exp_path)
            if len(pathLevels) > 4:
                itemText = os.path.join(*pathLevels[-4:])
                itemText = f'...{itemText}'
            else:
                itemText = exp_path
            exp_path_item = QTreeWidgetItem([itemText])
            exp_path_item.setToolTip(0, exp_path)
            exp_path_item.full_path = exp_path
            self.treeWidget.addTopLevelItem(exp_path_item)
            postions_items = []
            for pos in positions:
                if posFoldersInfo is not None:
                    status = posFoldersInfo.get(pos, '')
                else:
                    status = ''
                pos_item_text = f'{pos}{status}'
                pos_item = QTreeWidgetItem(exp_path_item, [pos_item_text])
                pos_item.posFoldername = pos
                postions_items.append(pos_item)
            exp_path_item.addChildren(postions_items)
            exp_path_item.setExpanded(True)

        self.treeWidget.itemClicked.connect(self.selectAllChildren)

        buttonsLayout = QHBoxLayout()
        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        okButton = widgets_push_buttons.okPushButton(' Ok ')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        mainLayout.addWidget(self.treeWidget)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        self.setStyleSheet(widgets_style.TREEWIDGET_STYLESHEET)

    def selectAllChildren(self, item, col):
        if item.parent() is not None:
            return

        for i in range(item.childCount()):
            item.child(i).setSelected(True)

    def ok_cb(self):
        if not self.treeWidget.selectedItems():
            msg = widgets_message_box.myMessageBox(wrapText=False)
            txt = 'You did not select any experiment/Position folder!'
            msg.warning(self, 'Empty selection!', html_utils.paragraph(txt))
            return

        self.cancel = False
        self.selectedPaths = {}
        for item in self.treeWidget.selectedItems():
            if item.parent() is None:
                continue
            parent = item.parent()
            exp_path = parent.full_path
            pos_folder = item.posFoldername
            if exp_path not in self.selectedPaths:
                self.selectedPaths[exp_path] = []
            self.selectedPaths[exp_path].append(pos_folder)

        self.close()

    def showEvent(self, event):
        self.resize(int(self.width()*2), self.height())


class SelectPositionFoldersDialogue(QDialog):
    def __init__(
            self, title, items, informativeText,
            CbLabel='Select value:  ', parent=None,
            showInFileManagerPath=None,
            addSelectByInfoCheckboxes=False, 
            allowMultiSelection=True
        ):
        self.cancel = True
        self.selectedItemsText = ''
        self.selectedItemsIdx = None
        self.showInFileManagerPath = showInFileManagerPath
        self.items = items
        self.allowMultiSelection = allowMultiSelection
        super().__init__(parent)
        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()
        topLayout = QHBoxLayout()
        self.topLayout = topLayout
        bottomLayout = QHBoxLayout()

        if informativeText:
            infoLabel = QLabel(informativeText)
            mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        label = QLabel(CbLabel)
        topLayout.addWidget(label, alignment=Qt.AlignRight)

        okButton = widgets_push_buttons.okPushButton('Ok')
        self.okButton = okButton
        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        if showInFileManagerPath is not None:
            txt = filesystem.get_open_filemaneger_os_string()
            showInFileManagerButton = widgets_push_buttons.showInFileManagerButton(txt)

        bottomLayout.addStretch(1)
        bottomLayout.addWidget(cancelButton)
        bottomLayout.addSpacing(20)
        if showInFileManagerPath is not None:
            bottomLayout.addWidget(showInFileManagerButton)
        bottomLayout.addWidget(okButton)
        
        self.buttonsLayout = bottomLayout

        self.selectByInfoCheckboxes = []
        selectByInfoLayout = None
        if addSelectByInfoCheckboxes and allowMultiSelection:
            selectByInfoLayout = self._createSelectByInfoCheckboxes(items)

        listBox = widgets_item_views.listWidget()
        listBox.addItems(items)
        if allowMultiSelection:
            selectionMode = QAbstractItemView.SelectionMode.ExtendedSelection
        else:
            selectionMode = QAbstractItemView.SelectionMode.SingleSelection
        listBox.setSelectionMode(selectionMode)
        listBox.setFont(fonts.font)
        listBox.setFocusPolicy(Qt.StrongFocus)
        topLayout.addWidget(listBox)
        self.ListBox = listBox
        self._setListBoxMinimumHeight()

        self.numSelectedLabel = None
        if addSelectByInfoCheckboxes:
            self.numSelectedLabel = QLabel()

        mainLayout.addLayout(topLayout)
        if selectByInfoLayout is not None:
            mainLayout.addSpacing(10)
            mainLayout.addLayout(selectByInfoLayout)
        if self.numSelectedLabel is not None:
            mainLayout.addSpacing(10)
            mainLayout.addWidget(
                self.numSelectedLabel, alignment=Qt.AlignRight
            )
        mainLayout.addSpacing(20)
        mainLayout.addLayout(bottomLayout)
        mainLayout.setStretchFactor(topLayout, 2)

        self.setLayout(mainLayout)
        self.mainLayout = mainLayout
        self.topLayout = topLayout

        # self.setModal(True)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        listBox.itemSelectionChanged.connect(self.onSelectionChanged)
        if showInFileManagerPath is not None:
            showInFileManagerButton.clicked.connect(self.showInFileManager)

        self.setFont(fonts.font)
        
        # Select the first item by default
        listBox.setCurrentRow(0)
        self.onSelectionChanged()
    
    def _setListBoxMinimumHeight(self):
        # Show up to 10 items without scrolling
        n = min(self.ListBox.count(), 10)
        h = sum([self.ListBox.sizeHintForRow(i) for i in range(n)])
        self.ListBox.setMinimumHeight(h+5)
    
    def onSelectionChanged(self):
        numSelected = len(self.ListBox.selectedItems())
        self.okButton.setDisabled(numSelected == 0)
        self.updateNumSelectedLabel()

    def updateNumSelectedLabel(self, *args):
        if self.numSelectedLabel is None:
            return

        numSelected = len(self.ListBox.selectedItems())
        numTotal = self.ListBox.count()
        self.numSelectedLabel.setText(
            f'<i>Number of selected Positions</i>: <b>{numSelected}/{numTotal}</b>'
        )

    # (regex pattern searched in the item text, checkbox text)
    # The info text is added to the Position folder names by methods like
    # `load.select_exp_folder.get_values_acdc_and_spotmax`,
    # `load.select_exp_folder.get_values_segmGUI`, and
    # `load.select_exp_folder.get_values_dataprep`
    SELECT_BY_INFO_RULES = (
        (
            r'last tracked frame',
            'Select all Positions with at least one tracked frame'
        ),
        (
            r'last (cc )?annotated frame',
            'Select all Positions with cell cycle annotations'
        ),
        (
            r'SpotMAX output exists',
            'Select all Positions with SpotMAX output'
        ),
        (
            r'z-slices selected',
            'Select all Positions with z-slices selected'
        ),
        (
            r'\baligned\b',
            'Select all aligned Positions'
        ),
        (
            r'(?<!bkgr )ROI info present',
            'Select all Positions with ROI info'
        ),
        (
            r'bkgr ROI info present',
            'Select all Positions with background ROI info'
        ),
        (
            r'\bcropped\b',
            'Select all cropped Positions'
        ),
    )

    def _createSelectByInfoCheckboxes(self, items):
        """Add one checkbox for each type of info found in the items text
        (e.g., 'last tracked frame' or 'SpotMAX output exists'). Every time
        a checkbox is toggled, the selection is set to the items that match
        ALL the checked checkboxes (AND). If no checkbox is checked, all
        the items are deselected.
        """
        if len(items) < 2:
            return

        layout = QGridLayout()
        n_cols = 2
        for pattern, checkboxText in self.SELECT_BY_INFO_RULES:
            regex = re.compile(pattern, flags=re.IGNORECASE)
            isMatch = [
                regex.search(self._getItemInfoText(item)) is not None
                for item in items
            ]
            if not any(isMatch):
                continue

            checkbox = QCheckBox(checkboxText)
            checkbox.isMatch = isMatch
            checkbox.toggled.connect(
                partial(self.selectItemsByInfoToggled, checkbox)
            )
            idx = len(self.selectByInfoCheckboxes)
            row, col = divmod(idx, n_cols)
            layout.addWidget(checkbox, row, col)
            self.selectByInfoCheckboxes.append(checkbox)

        if not self.selectByInfoCheckboxes:
            return

        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addLayout(layout)

        return hLayout

    def _getItemInfoText(self, item):
        # Search only in the info part (after the Position folder name)
        idx = item.find('(')
        if idx == -1:
            return ''
        return item[idx:]

    def selectItemsByInfoToggled(self, checkbox, checked):
        # Select only the items that match ALL the checked checkboxes
        checkedCheckboxes = [
            cb for cb in self.selectByInfoCheckboxes if cb.isChecked()
        ]
        firstSelectedItem = None
        for i in range(self.ListBox.count()):
            item = self.ListBox.item(i)
            isSelected = (
                len(checkedCheckboxes) > 0
                and all(cb.isMatch[i] for cb in checkedCheckboxes)
            )
            item.setSelected(isSelected)
            if isSelected and firstSelectedItem is None:
                firstSelectedItem = item

        if firstSelectedItem is not None:
            self.ListBox.scrollToItem(firstSelectedItem)

    def setAllSelected(self, selected: bool):
        for i in range(self.ListBox.count()):
            item = self.ListBox.item(i)
            item.setSelected(selected)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.SelectAll):
            self.selectAllItems()
            event.accept()
            return
        super().keyPressEvent(event)

    def selectAllItems(self):
        if not self.allowMultiSelection:
            return
        self.setAllSelected(True)

    def setSelectedItems(self, selectedItemsText):
        self.ListBox.clearSelection()
        firstSelectedItem = None
        for i in range(self.ListBox.count()):
            item = self.ListBox.item(i)
            if item.text() not in selectedItemsText:
                continue
            item.setSelected(True)
            if firstSelectedItem is None:
                firstSelectedItem = item
            if not self.allowMultiSelection:
                break

        if firstSelectedItem is not None:
            self.ListBox.scrollToItem(firstSelectedItem)

    def showInFileManager(self):
        selectedTexts, _ = self.getSelectedItems()
        showPath = self.showInFileManagerPath
        if selectedTexts:
            folder = selectedTexts[0].split('(')[0].strip()
            path = os.path.join(self.showInFileManagerPath, folder)
            if os.path.exists(path) and os.path.isdir(path):
                showPath = path
        filesystem.showInExplorer(showPath)

    def getSelectedItems(self):
        selectedItems = self.ListBox.selectedItems()
        selectedItemsText = [item.text() for item in selectedItems]
        selectedItemsText = natsorted(selectedItemsText)
        selectedItemsIdx = [
            self.items.index(txt) for txt in selectedItemsText
        ]
        return selectedItemsText, selectedItemsIdx

    def ok_cb(self, event):
        self.cancel = False
        self.selectedItemsText, self.selectedItemsIdx = self.getSelectedItems()
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        super().show()
        self.ListBox.setFocus()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()


class SelectSegmFileDialog(QDialog):
    def __init__(
            self, images_ls, parent_path, parent=None, 
            addNewFileButton=False, basename='', infoText=None, 
            fileType='segmentation', allowMultipleSelection=False,
            custom_first=None
        ):
        self.cancel = True
        self.selectedItemText = ''
        self.selectedItemIdx = None
        self.removeOthers = False
        self.okAllPos = False
        self.newSegmEndName = None
        self.allowMultipleSelection = allowMultipleSelection
        self.basename = basename
        images_ls = sorted(images_ls, key=len)
        if custom_first is not None:
            images_ls.remove(custom_first)
            images_ls.insert(0, custom_first)

        # Remove the 'segm_' part to allow filenameDialog to check if
        # a new file is existing (since we only ask for the part after
        # 'segm_')
        self.existingEndNames = [
            n.replace('segm', '', 1).replace('_', '', 1) for n in images_ls
        ]

        self.images_ls = images_ls
        self.parent_path = parent_path
        super().__init__(parent)

        informativeText = html_utils.paragraph(f"""
            The loaded Position folders already contains
            <b>{len(self.existingEndNames)} {fileType} masks</b><br>
        """)

        self.setWindowTitle(f'{fileType.capitalize()} files detected')
        is_win = sys.platform.startswith("win")

        mainLayout = QVBoxLayout()
        infoLayout = QHBoxLayout()
        selectionLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        # Standard Qt Question icon
        label = QLabel()
        standardIcon = getattr(QStyle, 'SP_MessageBoxQuestion')
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        label.setPixmap(pixmap)
        infoLayout.addWidget(label)

        infoLabel = QLabel(informativeText)
        infoLayout.addWidget(infoLabel)
        infoLayout.addStretch(1)
        mainLayout.addLayout(infoLayout)

        if infoText is None:
            infoText = f'Select which {fileType} file to load:'

        questionText = html_utils.paragraph(infoText)
        label = QLabel(questionText)
        listWidget = widgets_item_views.listWidget()
        listWidget.addItems(images_ls)
        listWidget.setCurrentRow(0)
        listWidget.itemDoubleClicked.connect(self.listDoubleClicked)
        if allowMultipleSelection:
            listWidget.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection
            )
        self.items = list(images_ls)
        self.listWidget = listWidget

        okButton = widgets_push_buttons.okPushButton(' Load selected ')
        txt = 'Reveal in Finder...' if is_mac else 'Show in Explorer...'
        showInFileManagerButton = widgets_push_buttons.showInFileManagerButton(txt)
        cancelButton = widgets_push_buttons.cancelPushButton(' Cancel ')

        if addNewFileButton:
            newFileButton = widgets_push_buttons.newFilePushButton('New file...')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addWidget(showInFileManagerButton)
        buttonsLayout.addSpacing(20)
        if addNewFileButton:
            buttonsLayout.addWidget(newFileButton)
        buttonsLayout.addWidget(okButton)

        buttonsLayout.setContentsMargins(0, 10, 0, 10)

        selectionLayout.addWidget(label, 0, 1, alignment=Qt.AlignLeft)
        selectionLayout.addWidget(listWidget, 1, 1)
        selectionLayout.setColumnStretch(0, 0)
        selectionLayout.setColumnStretch(1, 1)
        selectionLayout.setColumnStretch(2, 0)
        selectionLayout.addLayout(buttonsLayout, 2, 1)

        mainLayout.addLayout(selectionLayout)
        self.setLayout(mainLayout)

        self.okButton = okButton

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        if addNewFileButton:
            newFileButton.clicked.connect(self.newFile_cb)
        cancelButton.clicked.connect(self.close)
        showInFileManagerButton.clicked.connect(self.showInFileManager)
    
    def listDoubleClicked(self, item):
        self.ok_cb()

    def showInFileManager(self, checked=True):
        filesystem.showInExplorer(self.parent_path)
    
    def newFile_cb(self):
        win = filenameDialog(
            basename=f'{self.basename}segm',
            hintText='Insert a <b>filename</b> for the segmentation file:',
            existingNames=self.existingEndNames
        )
        win.exec_()
        if win.cancel:
            return
        self.cancel = False
        self.newSegmEndName = win.entryText
        self.close()
    
    def setSelectedItemFromText(self, itemText):
        for i in range(self.listWidget.count()):
            if self.listWidget.item(i).text() == itemText:
                self.listWidget.setCurrentRow(i)
                break

    def ok_cb(self, event=None):
        self.cancel = False
        try:
            self.selectedItemText = self.listWidget.selectedItems()[0].text()
        except IndexError:
            self.cancel = True
            self.close()
            return
        self.selectedItemIdx = self.items.index(self.selectedItemText)
        self.selectedItemTexts = [
            selectedItem.text() 
            for selectedItem in self.listWidget.selectedItems()
        ]
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


def get_existing_directory(allow_images_path=True, **kwargs):
    while True:
        folder_path = qtpy.compat.getexistingdirectory(**kwargs)
        if not folder_path:
            return
        
        if allow_images_path:
            return folder_path
        
        pos_folderpath = os.path.dirname(folder_path)
        is_images_folder = (
            folder_path.endswith('Images') 
            and os.path.basename(pos_folderpath).startswith('Position_')
            and os.path.isdir(folder_path)
        )
        if not is_images_folder:
            return folder_path
        
        txt = html_utils.paragraph(
            'You <b>cannot save</b> to the <code>Images</code> folder '
            'because it is reserved to files that start with the same '
            'basename.<br><br>Thank you for your patience!'
        )
        msg = widgets_message_box.myMessageBox()
        msg.warning(kwargs['parent'], 'Cannot save here', txt)


MODULES_TYPES = Literal['Dataprep', 'Segm&Track', 'GUI', 'Utility']

class SelectFoldersToAnalyse(dialogs_base.QBaseDialog):
    def __init__(
            self, parent=None, 
            preSelectedPaths=None, 
            onlyExpPaths=False, 
            scanFolderTree=True,
            instructionsText='Select experiment folders to analyse',
            askSelectPosFolders=False,
            title='Select experiments to analyse',
            callingModule: MODULES_TYPES='Segm&Track',
        ):
        super().__init__(parent)
        
        self.cancel = True
        self.onlyExpPaths = onlyExpPaths
        self.setWindowTitle(title)
        self.scanTree = scanFolderTree
        self.askSelectPosFolders = askSelectPosFolders
        self.callingModule = callingModule
        
        mainLayout = QVBoxLayout()
        
        instructionsText = html_utils.paragraph(
            f'{instructionsText}<br><br>'
            'Drag and drop folders or click on <code>Add folder</code> button to '
            '<b>add</b> as many <b>folders</b> '
            'as needed.<br>', font_size='14px'
        )
        instructionsLabel = QLabel(instructionsText)
        instructionsLabel.setAlignment(Qt.AlignCenter)
        
        infoText = html_utils.paragraph(            
            'A <b>valid folder</b> is either a <b>Position</b> folder, '
            'or an <b>experiment folder</b> (containing Position_n folders),<br>'
            'or any folder that contains <b>multiple experiment folders</b>.<br><br>'
            
            'In the last case, Cell-ACDC will automatically scan the entire tree of '
            'sub-directories<br>'
            'and will add all experiments having the right folder structure.<br>',
            font_size='12px'
        )
        infoLabel = QLabel(infoText)
        infoLabel.setAlignment(Qt.AlignCenter)
        
        self.listWidget = widgets_item_views.listWidget()
        self.listWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        if preSelectedPaths is not None:
            self.listWidget.addItems(preSelectedPaths)
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()

        delButton = widgets_push_buttons.delPushButton('Remove selected path(s)')
        browseButton = widgets_push_buttons.browseFileButton(
            'Add folder...', openFolder=True, 
            start_dir=filesystem.getMostRecentPath()
        )
        recentPathsButton = widgets_push_buttons.PushButton('Add from recent paths...')
        recentPathsButton.setIcon(QIcon(':open_file.svg'))
        recentPathsButton.setToolTip(
            'Select one or more existing folders from recent paths.'
        )
        buttonsLayout.insertWidget(3, recentPathsButton)
        buttonsLayout.insertWidget(4, delButton)
        buttonsLayout.insertWidget(5, browseButton)
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        browseButton.sigPathSelected.connect(self.addFolderPath)
        recentPathsButton.clicked.connect(self.addFromRecentPaths)
        delButton.clicked.connect(self.removePaths)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addWidget(instructionsLabel)
        mainLayout.addWidget(infoLabel)
        mainLayout.addWidget(self.listWidget)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.setStretch(0, 0)
        mainLayout.setStretch(1, 0)
        mainLayout.setStretch(2, 1)
        mainLayout.setStretch(3, 0)
        
        self.setLayout(mainLayout)
        
        self.setAcceptDrops(True)
        
        self.setFont(fonts.font)
    
    def dragEnterEvent(self, event):
        event.acceptProposedAction()
    
    def dropEvent(self, event):
        event.setDropAction(Qt.CopyAction)
        for url in event.mimeData().urls():
            dropped_path = url.toLocalFile()
            if os.path.isfile(dropped_path):
                dropped_path = os.path.dirname(dropped_path)
            
            QTimer.singleShot(50, partial(self.addFolderPath, dropped_path))
    
    def pathsList(self):
        return [
            self.listWidget.item(i).text().replace('\\', '/') 
            for i in range(self.listWidget.count())
        ]
    
    def expFolderToPosFoldernamesMapper(self):
        expPathsPosFoldernamesMapper = defaultdict(set)
        for selectedPath in self.pathsList():
            pos_foldernames = filesystem.get_pos_foldernames(
                selectedPath, check_if_is_sub_folder=True
            )
            if not pos_foldernames: 
                images_path = filesystem.get_images_folderpath(selectedPath)
                expPathsPosFoldernamesMapper[selectedPath].add('')
            else:
                expPath = load.get_exp_path(selectedPath)
                expPathsPosFoldernamesMapper[expPath].update(pos_foldernames)
        
        expPathsPosFoldernamesMapper = {
            expPath: natsorted(pos_foldernames) 
            for expPath, pos_foldernames in expPathsPosFoldernamesMapper.items()
        }
        return expPathsPosFoldernamesMapper
    
    def ok_cb(self):
        #verify all selected folders have Images folder:
        faultyFolders = []
        for path, selected_pos in self.expFolderToPosFoldernamesMapper().items():
            if selected_pos == ['']:
                images_path = filesystem.get_images_folderpath(path)
                if images_path is None or not os.path.exists(images_path):
                    faultyFolders.append(path)
                    
            else:
                for pos in selected_pos:
                    pos_path = os.path.join(path, pos)
                    images_path = filesystem.get_images_folderpath(pos_path)
                    if images_path is None or not os.path.exists(images_path):
                        faultyFolders.append(pos_path)
        
        if faultyFolders:
            self.warnNoAllValid(faultyFolders)
            return
            
        self.paths = self.pathsList()
        self.selectedExpFolderToPosFoldernamesMapper = (
            self.expFolderToPosFoldernamesMapper()
        )
        if not self.selectedExpFolderToPosFoldernamesMapper:
            self.warnEmptySelection()
            return
        self.cancel = False
        self.close()
        
    def warnNoAllValid(self, faultyFolders=None):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            Some of the selected folders (see below) do not contain an Images folder.<br><br>
            Please, make sure to select Position folders, the Images folder inside Position folders, or any folder containing Position folders as sub-directories.<br><br>
            Thank you for your patience!<br><br>
            Selected folders:
            <ul>
                {''.join(f'<li>{folder}</li>' for folder in faultyFolders)}
            </ul>
        """)
        msg.warning(
            self, 'Some folders are not valid', txt
        )
    
    def warnEmptySelection(self):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            No folder was selected.<br><br>
            """)
        msg.warning(
            self, 'No folder selected', txt
        )
    
    def warnNoValidPathsFound(self, selected_path):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            The selected path (see below) <b>does not contain any valid folder.</b><br><br>
            Please, make sure to select a Position folder, the Images folder 
            inside a Position folder, or any folder containing a Position folder 
            as a sub-directory.<br><br>
            Thank you for your patience!<br><br>
            Selected path:
        """)
        msg.warning(
            self, 'Training workflow generated', txt, 
            commands=(f'{selected_path}',),
            path_to_browse=selected_path
        )
    
    def warnNoValidExpPaths(self, selected_path):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            The selected folder does 
            <b>not contain any valid experiment folders</b>.
        """)
        command = selected_path.replace('\\', os.sep)
        command = selected_path.replace('/', os.sep)
        msg.warning(
            self, 'No valid folders found', txt,
            commands=(command,), 
            path_to_browse=selected_path
        )
    
    def parse_select_from_exp_paths(
            self, exp_paths: dict[os.PathLike, Iterable[str]]
        ):
        if not self.askSelectPosFolders:
            return list(exp_paths.keys())
        
        paths = []
        doNotAskAgain = False
        for exp_path, pos_foldernames in exp_paths.items():
            if len(pos_foldernames) == 1:
                paths.append(exp_path)
                continue
            
            if doNotAskAgain:
                for pos in pos_foldernames:
                    paths.append(os.path.join(exp_path, pos))
                continue
            
            informativeText = html_utils.paragraph(
                'The following experiment folder<br><br>'
                f'<code>{exp_path}</code><br><br>'
                'contains multiple Position folders.<br><br>'
                'Please, select which Position folder(s) you want to analyse:<br>'
            )
            select_folder = load.select_exp_folder()
            if self.callingModule == 'Utility':
                values = select_folder.get_values_acdc_and_spotmax(exp_path)
            else:
                values = select_folder.get_values_dataprep(exp_path)
            select_folder.QtPrompt(
                self, values, toggleMulti=True, 
                informativeText=informativeText,
                selectedValues=values, 
                addDoNotAskAgain=True
            )
            if select_folder.cancel:
                return
            
            if select_folder.doNotAskAgain:
                doNotAskAgain = True
                for pos in pos_foldernames:
                    paths.append(os.path.join(exp_path, pos))
                continue
            
            for pos in select_folder.selected_pos:
                paths.append(os.path.join(exp_path, pos))
                
        return paths
    
    def addFolderPath(self, selected_path):
        filesystem.addToRecentPaths(selected_path)
        
        folder_type = filesystem.determine_folder_type(selected_path)     
        is_pos_folder, is_images_folder, folder_path = folder_type
        if is_pos_folder:
            paths = [selected_path]
        elif is_images_folder:
            paths = [os.path.dirname(selected_path) if selected_path.endswith('Images') else selected_path]
        elif self.scanTree:
            print(f'Scanning selected folder "{selected_path}"...')
            exp_paths = path.get_posfolderpaths_walk(selected_path)
            if not exp_paths:
                self.warnNoValidExpPaths(selected_path)
                return
            
            paths = self.parse_select_from_exp_paths(exp_paths)
            if paths is None:
                return
        else:
            paths = [selected_path]
        
        if not paths:
            self.warnNoValidPathsFound(selected_path)
        
        for selectedPath in paths:
            if self.onlyExpPaths:
                selectedPath = load.get_exp_path(selectedPath)
            
            selectedPath = selectedPath.replace('\\', '/')
            if selectedPath in self.pathsList():
                print(
                    f'[WARNING]: The following path was already selected: '
                    f'"{selectedPath}"'
                )
                continue
                
            self.listWidget.addItem(selectedPath)

    def addFromRecentPaths(self):
        """Open a multi-select dialog containing existing recent folders."""
        recent_paths, dates = filesystem.get_recent_paths()
        if not recent_paths:
            return
        
        self.recentPathsDialog = dialogs_generic.SearchableListboxDialog(
            recent_paths,
            title='Add from recent paths',
            searchLineEditText='Search recent paths (Ctrl+click to select multiple)...',
            parent=self,
            multiSelection=True,
            dates=dates,
        )
        self.recentPathsDialog.sigOk.connect(
            self.addRecentFolderPaths
        )
        self.recentPathsDialog.show()

    def addRecentFolderPaths(self, selected_paths):
        """Add selected recent folders through the standard validation path."""
        for selected_path in selected_paths:
            self.addFolderPath(selected_path)
    
    def removePaths(self):
        for item in self.listWidget.selectedItems():
            row = self.listWidget.row(item)
            self.listWidget.takeItem(row)
