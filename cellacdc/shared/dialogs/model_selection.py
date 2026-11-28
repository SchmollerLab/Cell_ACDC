"""Segmentation model selection and installation dialogs (from apps.py).

QDialogSelectModel (main GUI, segmentation module) with the custom model
instructions addCustomModelMessages, the weights download window
downloadModel (also myutils/segm_models.py, test_segm_model.py) and the
PyTorch install prompt InstallPyTorchDialog (myutils/torch_utils.py,
service/prompts.py), plus the model picker widgets only they use.
"""
from collections import defaultdict
from qtpy.QtCore import (
    Signal, Qt, QSize, QItemSelectionModel, QAbstractListModel, QModelIndex,
    QByteArray, QDataStream, QMimeData, QAbstractItemModel, QIODevice,
    QItemSelection
)
from qtpy.QtWidgets import (
    QLabel, QHBoxLayout, QApplication, QWidget, QVBoxLayout, QPushButton,
    QSpacerItem, QListView, QAbstractItemView, QListWidgetItem
)
from cellacdc.myutils import downloads, filesystem, installs, segm_models
from cellacdc import fonts
from ..widgets import item_views as widgets_item_views
from ..widgets import style as widgets_style
import os
import sys
import datetime
import pathlib
from functools import partial
from qtpy.QtGui import QKeyEvent
from qtpy.QtCore import QEventLoop
from qtpy.QtWidgets import (
    QFileDialog, QDialog, QSizePolicy, QComboBox, QGridLayout, QStyle
)
from cellacdc import _window_stacking
from cellacdc import html_utils
from cellacdc import is_mac, is_win
from cellacdc import is_conda_env
from cellacdc.myutils import torch_utils
from cellacdc.myutils.torch_utils import torch_requirement
from ..widgets import (
    line_edits as widgets_line_edits,
    message_box as widgets_message_box,
    push_buttons as widgets_push_buttons,
)
from cellacdc import models_path
from cellacdc import urls
from . import generic as dialogs_generic
from . import file_selection as dialogs_file_selection
from . import base as dialogs_base


class _ReorderableListModel(QAbstractListModel):
    '''
    ReorderableListModel is a list model which implements reordering of its
    items via drag-n-drop
    '''
    dragDropFinished = Signal()

    def __init__(self, items, parent=None):
        QAbstractItemModel.__init__(self, parent)
        self.nodes = list(items)
        self.lastDroppedItems = []
        self.pendingRemoveRowsAfterDrop = False

    def rowForItem(self, text):
        '''
        rowForItem method returns the row corresponding to the passed in item
        or None if no such item exists in the model
        '''
        try:
            row = self.nodes.index(text)
        except ValueError:
            return None
        return row

    def index(self, row, column, parent):
        if row < 0 or row >= len(self.nodes):
            return QModelIndex()
        return self.createIndex(row, column)

    def parent(self, index):
        return QModelIndex()

    def rowCount(self, index):
        if index.isValid():
            return 0
        return len(self.nodes)

    def data(self, index, role):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            row = index.row()
            if row < 0 or row >= len(self.nodes):
                return None
            return self.nodes[row]
        elif role == Qt.SizeHintRole:
            return QSize(48, 32)
        else:
            return None

    def supportedDropActions(self):
        return Qt.MoveAction

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | \
               Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled

    def insertRows(self, row, count, index):
        if index.isValid():
            return False
        if count <= 0:
            return False
        # inserting 'count' empty rows starting at 'row'
        self.beginInsertRows(QModelIndex(), row, row + count - 1)
        for i in range(0, count):
            self.nodes.insert(row + i, '')
        self.endInsertRows()
        return True

    def removeRows(self, row, count, index):
        if index.isValid():
            return False
        if count <= 0:
            return False
        num_rows = self.rowCount(QModelIndex())
        self.beginRemoveRows(QModelIndex(), row, row + count - 1)
        for i in range(count, 0, -1):
            self.nodes.pop(row - i + 1)
        self.endRemoveRows()

        if self.pendingRemoveRowsAfterDrop:
            '''
            If we got here, it means this call to removeRows is the automatic
            'cleanup' action after drag-n-drop performed by Qt
            '''
            self.pendingRemoveRowsAfterDrop = False
            self.dragDropFinished.emit()

        return True

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if index.row() < 0 or index.row() > len(self.nodes):
            return False
        self.nodes[index.row()] = str(value)
        self.dataChanged.emit(index, index)
        return True

    def mimeTypes(self):
        return ['application/vnd.treeviewdragdrop.list']

    def mimeData(self, indexes):
        mimedata = QMimeData()
        encoded_data = QByteArray()
        stream = QDataStream(encoded_data, QIODevice.WriteOnly)
        for index in indexes:
            if index.isValid():
                text = self.data(index, 0)
        stream << QByteArray(text.encode('utf-8'))
        mimedata.setData('application/vnd.treeviewdragdrop.list', encoded_data)
        return mimedata

    def dropMimeData(self, data, action, row, column, parent):
        if action == Qt.IgnoreAction:
            return True
        if not data.hasFormat('application/vnd.treeviewdragdrop.list'):
            return False
        if column > 0:
            return False

        num_rows = self.rowCount(QModelIndex())
        if num_rows <= 0:
            return False

        if row < 0:
            if parent.isValid():
                row = parent.row()
            else:
                return False

        encoded_data = data.data('application/vnd.treeviewdragdrop.list')
        stream = QDataStream(encoded_data, QIODevice.ReadOnly)

        new_items = []
        rows = 0
        while not stream.atEnd():
            text = QByteArray()
            stream >> text
            text = bytes(text).decode('utf-8')
            index = self.nodes.index(text)
            new_items.append((text, index))
            rows += 1

        self.lastDroppedItems = []
        for (text, index) in new_items:
            target_row = row
            if index < row:
                target_row += 1
            self.beginInsertRows(QModelIndex(), target_row, target_row)
            self.nodes.insert(target_row, self.nodes[index])
            self.endInsertRows()
            self.lastDroppedItems.append(text)
            row += 1

        self.pendingRemoveRowsAfterDrop = True
        return True

class _SelectionModel(QItemSelectionModel):
    def __init__(self, parent=None, isSingleSelection=False):
        QItemSelectionModel.__init__(self, parent)
        self.isSingleSelection = isSingleSelection

    def onModelItemsReordered(self):
        new_selection = QItemSelection()
        new_index = QModelIndex()
        for item in self.model().lastDroppedItems:
            row = self.model().rowForItem(item)
            if row is None:
                continue
            new_index = self.model().index(row, 0, QModelIndex())
            new_selection.select(new_index, new_index)

        self.clearSelection()
        flags = (
            QItemSelectionModel.SelectionFlag.ClearAndSelect 
            | QItemSelectionModel.SelectionFlag.Rows 
            | QItemSelectionModel.SelectionFlag.Current
        )
        self.select(new_selection, flags)
        self.setCurrentIndex(new_index, flags)
        if not self.isSingleSelection:
            self.reset()

class ReorderableListView(QListView):
    def __init__(
            self, items=None, parent=None, isSingleSelection=False
        ) -> None:
        super().__init__(parent)
        if items is None:
            items = []

        self.isSingleSelection = isSingleSelection
        self._model = _ReorderableListModel(items)
        self._selectionModel = _SelectionModel(self._model)
        self._model.dragDropFinished.connect(
            self._selectionModel.onModelItemsReordered
        )
        self.setModel(self._model)
        self.setSelectionModel(self._selectionModel)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDragDropOverwriteMode(False)
        styleSheet = (f"""
            QListView {{
                selection-background-color: rgba(200, 200, 200, 0.30);
                selection-color: black;
                show-decoration-selected: 1;
            }}
            QListView::item {{
                border-bottom: 1px solid rgba(180, 180, 180, 0.5);
            }}
            QListView::item:hover {{
                background-color: rgba(200, 200, 200, 0.30);
            }}
        """)
        self.setStyleSheet(styleSheet)
    
    def setItems(self, items):
        self._model.nodes = list(items)
    
    def items(self):
        return self._model.nodes
    
    # def mouseReleaseEvent(self, e: QMouseEvent) -> None:
    #     super().mouseReleaseEvent(e)
    #     self._selectionModel.reset()


class MultiPickListWidget(QWidget):
    """Generic list widget with multi-pick (repeated-selection) support.

    Each pickable row shows ``-  count  +`` controls.  Left-clicking adds
    one instance; right-clicking or Ctrl+left-click removes one.  The same
    item can appear multiple times in :attr:`selectionSequence`.

    Parameters
    ----------
    items:
        Initial list of item labels.
    excludedItems:
        Labels that are shown in the list but *not* given +/- controls
        (e.g. placeholder entries like "Add custom model…").  Click events
        on these are silently ignored.
    parent:
        Optional parent widget.
    """

    sigSelectionChanged = Signal(list)  # emits selectionSequence on every change

    def __init__(self, items=None, excludedItems=None, parent=None):
        super().__init__(parent)

        self._excludedItems = set(excludedItems or [])
        self._itemsMap = {}          # label → QListWidgetItem
        self._countMap = defaultdict(int)
        self._countLabelMap = {}
        self.selectionSequence = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.listBox = widgets_item_views.listWidget(isMultipleSelection=False)
        self.listBox.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        for label in (items or []):
            self._addListItem(label)

        if self._itemsMap:
            self.listBox.setCurrentRow(0)

        self.listBox.itemClicked.connect(self._onItemClicked)
        self.listBox.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.listBox.customContextMenuRequested.connect(self._onRightClick)

        # self.listBox.setStyleSheet(LISTWIDGET_STYLESHEET)
        # self.setStyleSheet(LISTWIDGET_STYLESHEET)
        layout.addWidget(self.listBox)
        

    @property
    def itemsMap(self):
        """Dict mapping label → QListWidgetItem for all pickable items."""
        return dict(self._itemsMap)

    def currentItemName(self):
        """Return the label of the currently highlighted item, or ``None``."""
        item = self.listBox.currentItem()
        return item.text() if item is not None else None

    def addSelection(self, label):
        """Add one instance of *label* to the selection."""
        if label not in self._itemsMap:
            return
        self.selectionSequence.append(label)
        self._countMap[label] += 1
        self._updateCountLabel(label)
        self.listBox.setCurrentItem(self._itemsMap[label])
        self.sigSelectionChanged.emit(list(self.selectionSequence))

    def removeSelection(self, label):
        """Remove the last instance of *label* from the selection."""
        if self._countMap.get(label, 0) <= 0:
            return
        for i in range(len(self.selectionSequence) - 1, -1, -1):
            if self.selectionSequence[i] == label:
                self.selectionSequence.pop(i)
                break
        self._countMap[label] = max(0, self._countMap[label] - 1)
        self._updateCountLabel(label)
        self.sigSelectionChanged.emit(list(self.selectionSequence))

    def resetSelection(self):
        """Clear all selections and reset all counters to zero."""
        self.selectionSequence = []
        self._countMap = defaultdict(int)
        for label in self._countLabelMap:
            self._updateCountLabel(label)
        self.sigSelectionChanged.emit([])

    def setSelectionFromList(self, labels):
        """Set the selection to *labels* (duplicates supported)."""
        self.resetSelection()
        for label in labels:
            self.addSelection(label)

    def registerItem(self, label, insertBeforeLabel=None):
        """Dynamically add a new pickable item.

        Parameters
        ----------
        label:
            Text for the new item.
        insertBeforeLabel:
            If given, insert the new item immediately before this label.
            If not found or not given, the item is appended.

        Returns the created ``QListWidgetItem``.
        """
        if label in self._itemsMap:
            return self._itemsMap[label]

        item = QListWidgetItem(label)

        if insertBeforeLabel is not None:
            target = self._itemsMap.get(insertBeforeLabel)
            if target is None:
                for row in range(self.listBox.count()):
                    row_item = self.listBox.item(row)
                    if row_item is not None and row_item.text() == insertBeforeLabel:
                        target = row_item
                        break
            if target is not None:
                row = self.listBox.row(target)
                self.listBox.insertItem(row, item)
            else:
                self.listBox.addItem(item)
        else:
            self.listBox.addItem(item)

        self._itemsMap[label] = item
        self._addCounterWidget(label, item)
        return item

    def _addListItem(self, label):
        """Create a QListWidgetItem and, if pickable, attach a counter widget."""
        item = QListWidgetItem(label)
        self.listBox.addItem(item)
        if label not in self._excludedItems:
            self._itemsMap[label] = item
            self._addCounterWidget(label, item)

    def _addCounterWidget(self, label, item):
        rowWidget = QWidget()
        rowLayout = QHBoxLayout(rowWidget)
        rowLayout.setContentsMargins(4, 0, 4, 0)
        rowLayout.setSpacing(6)

        nameLabelPlaceholder = QSpacerItem(2, 0)
        minusBtn = QPushButton('-')
        plusBtn = QPushButton('+')
        countLabel = QLabel(str(self._countMap.get(label, 0)))

        minusBtn.setFixedWidth(24)
        plusBtn.setFixedWidth(24)
        countLabel.setMinimumWidth(20)
        countLabel.setAlignment(Qt.AlignCenter)

        minusBtn.clicked.connect(lambda _, lbl=label: self.removeSelection(lbl))
        plusBtn.clicked.connect(lambda _, lbl=label: self.addSelection(lbl))

        rowLayout.addItem(nameLabelPlaceholder)
        rowLayout.addStretch(1)
        rowLayout.addWidget(minusBtn)
        rowLayout.addWidget(countLabel)
        rowLayout.addWidget(plusBtn)

        self._countLabelMap[label] = countLabel
        self._updateCountLabel(label)
        self.listBox.setItemWidget(item, rowWidget)

    def _updateCountLabel(self, label):
        lbl = self._countLabelMap.get(label)
        if lbl is not None:
            count = self._countMap.get(label, 0)
            lbl.setText(str(count))
            if count <= 0:
                lbl.setStyleSheet('color: gray;')
            else:
                lbl.setStyleSheet('')

    def _onItemClicked(self, item):
        label = item.text()
        if label in self._excludedItems:
            return
        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.ControlModifier:
            self.removeSelection(label)
        else:
            self.addSelection(label)

    def _onRightClick(self, pos):
        item = self.listBox.itemAt(pos)
        if item is None:
            return
        label = item.text()
        if label in self._excludedItems:
            return
        self.removeSelection(label)


class ModelSelectionWidget(QWidget):
    """List widget for selecting segmentation models.

    Thin wrapper around :class:`MultiPickListWidget` that populates the list
    with the installed models and adds a special "Add custom model…" entry.

    ``sigSelectionChanged`` and ``selectionSequence`` are proxied from the
    underlying :class:`MultiPickListWidget`.
    """

    _ADD_CUSTOM = 'Add custom model...'

    sigSelectionChanged = Signal(list)

    def __init__(self, parent=None, customFirst='', allowMultiSelection=False):
        super().__init__(parent)

        self.allowMultiSelection = allowMultiSelection

        models = segm_models.get_list_of_models()
        if customFirst:
            try:
                models.insert(0, models.pop(models.index(customFirst)))
            except ValueError:
                pass

        items = models + [self._ADD_CUSTOM]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if allowMultiSelection:
            items = models
            self._picker = MultiPickListWidget(
                items=items,
                excludedItems=[self._ADD_CUSTOM],
                parent=self,
            )
            self._picker.listBox.setFont(widgets_style.font)
            self._picker.sigSelectionChanged.connect(self.sigSelectionChanged)
            self.listBox = self._picker.listBox
            layout.addWidget(self._picker)
        else:
            self.listBox = widgets_item_views.listWidget(isMultipleSelection=False)
            self.listBox.setFont(widgets_style.font)
            self.listBox.addItems(models)
            add_item = QListWidgetItem(self._ADD_CUSTOM)
            add_item.setFont(fonts.italicFont)
            self.listBox.addItem(add_item)
            self.listBox.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.listBox.setCurrentRow(0)
            self._picker = None
            layout.addWidget(self.listBox)

    # ------------------------------------------------------------------
    # Proxy helpers (multi-selection mode only)
    # ------------------------------------------------------------------

    @property
    def selectionSequence(self):
        return self._picker.selectionSequence if self._picker is not None else []

    @property
    def modelItemsMap(self):
        return self._picker.itemsMap if self._picker is not None else {}

    def currentModelName(self):
        if self._picker is not None:
            return self._picker.currentItemName()
        item = self.listBox.currentItem()
        return item.text() if item is not None else None

    def addModelSelection(self, name):
        if self._picker is not None:
            self._picker.addSelection(name)

    def removeModelSelection(self, name):
        if self._picker is not None:
            self._picker.removeSelection(name)

    def resetSelectionSequence(self):
        if self._picker is not None:
            self._picker.resetSelection()

    def setSelectionFromList(self, models):
        if self._picker is not None:
            self._picker.setSelectionFromList(models)

    def registerCustomModel(self, model_name):
        """Add a newly registered custom model and return its item."""
        if self._picker is not None:
            return self._picker.registerItem(
                model_name, insertBeforeLabel=self._ADD_CUSTOM
            )
        item = QListWidgetItem(model_name)
        self.listBox.insertItem(self.listBox.count() - 1, item)
        return item


def addCustomModelMessages(QParent=None):
    modelFilePath = None
    msg = widgets_message_box.myMessageBox(showCentered=False, wrapText=False)
    txt = html_utils.paragraph("""
    Do you <b>already have</b> the <code>acdcSegment.py</code> file for your code 
    or do you <b>need instructions</b> on how to set-up your custom model?<br>
    """)
    infoButton = widgets_push_buttons.infoPushButton(' I need instructions')
    browseButton = widgets_push_buttons.browseFileButton(' I have the model, let me select it')
    msg.information(
        QParent, 'Add custom model', txt, 
        buttonsTexts=('Cancel', infoButton, browseButton),
        showDialog=False
    )
    browseButton.clicked.disconnect()
    browseButton.clicked.connect(msg.buttonCallBack)
    msg.exec_()
    if msg.cancel:
        return
    if msg.clickedButton == infoButton:           
        txt = segm_models.get_add_custom_model_instructions()
        msg = widgets_message_box.myMessageBox(showCentered=False, wrapText=False)
        msg.information(
            QParent, 'Custom model instructions', txt, buttonsTexts=('Ok',),
            path_to_browse=models_path,
            browse_button_text='Open models folder...'
        )
    else:
        homePath = pathlib.Path.home()
        modelFilePath = QFileDialog.getOpenFileName(
            QParent, 'Select the acdcSegment.py file of your model',
            str(homePath), 'acdcSegment.py file (*.py);;All files (*)'
        )[0]
        if not modelFilePath:
            return
    
    return modelFilePath


class QDialogSelectModel(QDialog):
    def __init__(
            self, parent=None, addSkipSegmButton=False, customFirst='',
            allowMultiSelection=False, lastSelection=None,
            addSelectLastSelectionButton=False,
            addSelectLastRecipeButton=False,
            addSelectSavedRecipeButton=False,
            add_save_func=False,
            recipes_path='',
            recipe_prefix='',
            recipe_ext_label=None,
            recipe_ext=None,
            custom_title=None,
            info_label='',
        ):
        self.cancel = True
        self.loadLastRecipe = False
        self.loadSavedRecipe = False
        self.selectedRecipeFilepath = None
        self.selectionSaveName = ''
        self.selectionSaveNameFormatted = ''
        super().__init__(parent)
        self.setWindowTitle('Select model')
        self.info_label = info_label
        self.recipes_path = recipes_path
        self.recipe_prefix = recipe_prefix
        self.recipe_ext_label = recipe_ext_label
        self.recipe_ext = recipe_ext

        self.allowMultiSelection = allowMultiSelection
        self.lastSelection = []
        for m in (lastSelection or []):
            if not isinstance(m, str):
                continue
            if m == 'thresholding':
                m = 'Automatic thresholding'
            self.lastSelection.append(m)

        mainLayout = QVBoxLayout()
        self.mainLayout = mainLayout

        title = custom_title or 'Select model to use for segmentation: '

        titleContainer = QWidget(self)
        titleLayout = QGridLayout(titleContainer)
        titleLayout.setContentsMargins(0, 0, 0, 0)
        titleLayout.setSpacing(0)
        label = QLabel(html_utils.paragraph(
            title
        ))
        label.setStyleSheet("padding:0px 0px 3px 0px;")
        titleLayout.addWidget(label, 0, 0, Qt.AlignCenter)
        if info_label:
            moreInfoButton = widgets_push_buttons.infoPushButton()
            moreInfoButton.clicked.connect(self.showInfoLabel)
            moreInfoButton.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
            )
            titleLayout.addWidget(moreInfoButton, 0, 0, Qt.AlignTop | Qt.AlignRight)
        mainLayout.addWidget(titleContainer)

        self.modelSelector = ModelSelectionWidget(
            parent=self,
            customFirst=customFirst,
            allowMultiSelection=allowMultiSelection,
        )
        # Convenience aliases kept for backward compatibility
        self.listBox = self.modelSelector.listBox
        mainLayout.addWidget(self.modelSelector)

        self.add_save_func = add_save_func
        if add_save_func:
            saveLayout = QGridLayout()

            saveNameLabel = QLabel('Save recipe as (optional):')
            self.saveSelectionNameLineEdit = widgets_line_edits.alphaNumericLineEdit(
                onlyWarn=True, formatter=self.formattedSaveSelectionName,
                ignore_file_ext=f".{recipe_ext}"
            )
            self.saveSelectionNameLineEdit.setAlignment(Qt.AlignCenter)
            self.saveSelectionNameLineEdit.textChanged.connect(
                self.updateSaveSelectionPreview
            )
            self.saveSelectionNameLineEdit.sigInvalidCharactersEntered.connect(
                self.warnInvalidSaveNameChars
            )

            self.saveSelectionPreviewLabel = QLabel('')
            self.saveSelectionPreviewLabel.setTextFormat(Qt.RichText)
            self.saveSelectionPreviewLabel.setStyleSheet('font-size: 11px;')

            saveLayout.addWidget(saveNameLabel, 0, 0)
            saveLayout.addWidget(self.saveSelectionNameLineEdit, 0, 1)
            saveLayout.addWidget(self.saveSelectionPreviewLabel, 1, 0, 1, 2)
            saveLayout.setColumnStretch(1, 1)

            mainLayout.addSpacing(6)
            mainLayout.addLayout(saveLayout)
            self.updateSaveSelectionPreview('')

        if not allowMultiSelection:
            self.listBox.itemDoubleClicked.connect(self.ok_cb)

        bottomLayout = QHBoxLayout()
        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        okButton = widgets_push_buttons.okPushButton(' Ok ')
        okButton.setShortcut(Qt.Key_Enter)

        bottomLayout.addStretch(1)
        bottomLayout.addWidget(cancelButton)
        bottomLayout.addSpacing(20)
        if addSkipSegmButton:
            skipSegmButton = widgets_push_buttons.SkipPushButton('Skip segmentation')
            bottomLayout.addWidget(skipSegmButton)
            skipSegmButton.clicked.connect(self.skipSegm)
        if addSelectLastSelectionButton:
            selectLastSelButton = widgets_push_buttons.reloadPushButton('Load last selection...')
            selectLastSelButton.clicked.connect(self.selectLastSelection)
            selectLastSelButton.setEnabled(bool(self.lastSelection))
            bottomLayout.addWidget(selectLastSelButton)
        if addSelectLastRecipeButton:
            selectLastRecipeButton = widgets_push_buttons.reloadPushButton('Load last recipe...')
            selectLastRecipeButton.clicked.connect(self.selectLastRecipe)
            selectLastRecipeButton.setEnabled(bool(self.lastSelection))
            bottomLayout.addWidget(selectLastRecipeButton)
        if addSelectSavedRecipeButton:
            selectSavedRecipeButton = widgets_push_buttons.OpenFilePushButton(
                'Choose from saved recipes...'
            )
            selectSavedRecipeButton.clicked.connect(self.selectSavedRecipe)
            bottomLayout.addWidget(selectSavedRecipeButton)
        if allowMultiSelection:
            addCustomModelButton = widgets_push_buttons.addPushButton('Add custom model...')
            addCustomModelButton.clicked.connect(self.addCustomModel)
            bottomLayout.addWidget(addCustomModelButton)
        bottomLayout.addWidget(okButton)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)


    @property
    def selectionSequence(self):
        return self.modelSelector.selectionSequence

    @property
    def modelItemsMap(self):
        return self.modelSelector.modelItemsMap

    def skipSegm(self):
        self.cancel = False
        self.selectedModel = 'skip_segmentation'
        self.close()

    def selectLastSelection(self):
        if not self.lastSelection:
            return
        self.modelSelector.setSelectionFromList(self.lastSelection)

    def selectLastRecipe(self):
        if not self.lastSelection:
            return
        self.selectLastSelection()
        self.cancel = False
        self.loadLastRecipe = True
        self.selectedModel = self.lastSelection.copy()
        self.close()

    def recipeIniFileSelected(
            self, ini_filepath, selectRecipeWin=None, sender=None
        ):
        selectRecipeWin.clickedButton = sender
        selectRecipeWin.selectedIniFilepath = ini_filepath
        selectRecipeWin.cancel = False
        selectRecipeWin.close()

    def selectRecipeFilepath(self, recipes_path, recipe_prefix, ext_label, ext):
        availableRecipes = []
        displayNameToFilenameMapper = {}
        ext = ext.lstrip('.')
        if os.path.exists(recipes_path):
            for file in filesystem.listdir(recipes_path):
                if not file.startswith(recipe_prefix):
                    continue

                if not file.lower().endswith(f'.{ext.lower()}'):
                    continue

                parts = file.split(f'{recipe_prefix}_', 1)
                if len(parts) != 2:
                    continue

                endname = parts[1]
                filepath = os.path.join(recipes_path, file)
                modified_on = datetime.datetime.fromtimestamp(
                    os.path.getmtime(filepath)
                ).strftime(r'%Y/%m/%d %H:%M')
                availableRecipes.append((endname, modified_on))
                displayNameToFilenameMapper[endname] = file

        if not availableRecipes:
            import qtpy.compat
            filepath = qtpy.compat.getopenfilename(
                parent=self,
                caption=f'Select {ext_label} file to load recipe',
                filters=f'{ext_label} (*.{ext});;All Files (*)'
            )[0]
            return filepath or None

        browseButton = widgets_push_buttons.browseFileButton(
            f'Select {ext_label} file...',
            title=f'Select {ext_label} file to load recipe',
            openFolder=False,
            start_dir=filesystem.getMostRecentPath(),
            ext={ext_label: f'.{ext}'}
        )
        selectRecipeWin = dialogs_generic.QTreeDialog(
            availableRecipes,
            headerLabels=['Name', 'Date Modified'],
            title='Select recipe',
            infoText='Select recipe to load:<br>',
            parent=self,
            path_to_browse=recipes_path,
            additional_buttons=(browseButton,)
        )
        browseButton.sigPathSelected.connect(
            partial(
                self.recipeIniFileSelected,
                selectRecipeWin=selectRecipeWin,
                sender=browseButton
            )
        )
        selectRecipeWin.exec_()
        if selectRecipeWin.cancel:
            return None

        if selectRecipeWin.clickedButton == browseButton:
            return selectRecipeWin.selectedIniFilepath

        selected_endname = selectRecipeWin.selectedText
        filename = displayNameToFilenameMapper.get(selected_endname)
        if filename is None:
            return None

        return os.path.join(recipes_path, filename)

    def selectSavedRecipe(self):
        if not self.recipes_path:
            return

        filepath = self.selectRecipeFilepath(
            self.recipes_path,
            self.recipe_prefix,
            self.recipe_ext_label,
            self.recipe_ext,
        )
        if filepath is None:
            return

        self.cancel = False
        self.loadSavedRecipe = True
        self.selectedRecipeFilepath = filepath
        self.close()

    def _runAddCustomModelWorkflow(self):
        modelFilePath = addCustomModelMessages(self)
        if modelFilePath is None:
            return None

        segm_models.store_custom_model_path(modelFilePath)
        modelName = os.path.basename(os.path.dirname(modelFilePath))
        self.modelSelector.registerCustomModel(modelName)
        return modelName

    def addCustomModel(self):
        modelName = self._runAddCustomModelWorkflow()
        if modelName is None:
            return

        if self.allowMultiSelection:
            self.modelSelector.addModelSelection(modelName)
        else:
            item = QListWidgetItem(modelName)
            self.listBox.addItem(item)
            self.listBox.setCurrentItem(item)

    def showInfoLabel(self):
        if not self.info_label:
            return
        msg = widgets_message_box.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph(self.info_label)
        msg.information(self, 'More info', txt)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            event.ignore()
            return
        super().keyPressEvent(event)

    def askSelectedModelsOrder(self, selected_models):
        dialog = QDialog(self)
        dialog.setWindowTitle('Order selected models')
        _window_stacking.set_window_flags_on_top(dialog, Qt.Window)

        layout = QVBoxLayout(dialog)
        infoTxt = html_utils.paragraph(
            'Drag and drop to change the order of the selected models.<br>'
            'The top model will run first.'
        )
        layout.addWidget(QLabel(infoTxt))

        modelOrderView = ReorderableListView(
            selected_models, parent=dialog, isSingleSelection=True
        )
        layout.addWidget(modelOrderView)

        buttonsLayout = QHBoxLayout()
        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        okButton = widgets_push_buttons.okPushButton('Ok')
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)
        layout.addLayout(buttonsLayout)

        cancelButton.clicked.connect(dialog.reject)
        okButton.clicked.connect(dialog.accept)

        if dialog.exec_() != QDialog.Accepted:
            return None

        return modelOrderView.items()

    def saveSelectionName(self):
        if not self.add_save_func:
            return ''

        return self.saveSelectionNameLineEdit.text().strip()

    def formattedSaveSelectionName(self, raw_name=None):
        if raw_name is None:
            raw_name = self.saveSelectionName()
        if not raw_name:
            return ''

        formatted = dialogs_file_selection.FilenameEntryFormatter.format_entry_text(raw_name)
        formatted = self.recipe_prefix + "_" + formatted + f".{self.recipe_ext}"
        return formatted

    def warnInvalidSaveNameChars(self, characters: set[str]):
        if not self.add_save_func:
            return

        warning_text = dialogs_file_selection.FilenameEntryFormatter.invalid_chars_warning_text(
            characters
        )
        self.saveSelectionPreviewLabel.setText(warning_text)
        self.saveSelectionPreviewLabel.setStyleSheet('font-size: 11px; color: red;')
        self.saveSelectionNameLineEdit.setStyleSheet('border: 2px solid red;')
        
    def warnIfInvalidSaveName(self):
        if not self.add_save_func:
            return
        
        if not self.formattedSaveSelectionName():
            self.saveSelectionPreviewLabel.setStyleSheet('font-size: 11px;')
            self.saveSelectionNameLineEdit.setStyleSheet('')
            return
        
        path = os.path.join(self.recipes_path, self.formattedSaveSelectionName())
        if not os.path.exists(path):
            self.saveSelectionPreviewLabel.setStyleSheet('font-size: 11px;')
            self.saveSelectionNameLineEdit.setStyleSheet('')
            return
        
        warning_text = html_utils.paragraph(
            '<span style="color: yellow;">Save name already exists!</span>'
        )
        self.saveSelectionPreviewLabel.setText(warning_text)
        self.saveSelectionPreviewLabel.setStyleSheet('font-size: 11px; color: yellow;')
        self.saveSelectionNameLineEdit.setStyleSheet('border: 2px solid yellow;')

    def updateSaveSelectionPreview(self, text):
        if not self.add_save_func:
            return

        if self.saveSelectionNameLineEdit.invalidCharacters():
            self.saveSelectionNameLineEdit.setStyleSheet('border: 2px solid red;')
            return

        self.saveSelectionNameLineEdit.setStyleSheet('')
        self.saveSelectionPreviewLabel.setStyleSheet('font-size: 11px;')

        formatted = self.formattedSaveSelectionName()
        if not text.strip():
            preview_text = html_utils.paragraph(
                '<i>Leave empty to skip saving this selection.</i>'
            )
        elif not dialogs_file_selection.FilenameEntryFormatter.is_valid_name(formatted):
            preview_text = html_utils.paragraph(
                '<span style="color: red;">Invalid save name.</span>'
            )
            self.saveSelectionNameLineEdit.setStyleSheet('border: 2px solid red;')
        else:
            preview_text = html_utils.paragraph(
                f'Formatted save name preview: <code>{formatted}</code>'
            )

        self.saveSelectionPreviewLabel.setText(preview_text)
        
        self.warnIfInvalidSaveName()

    def checkSaveSelectionName(self):
        if not self.add_save_func:
            self.selectionSaveName = ''
            self.selectionSaveNameFormatted = ''
            return True

        if self.saveSelectionNameLineEdit.invalidCharacters():
            return False

        save_name = self.saveSelectionName()
        if not save_name:
            self.selectionSaveName = ''
            self.selectionSaveNameFormatted = ''
            return True

        formatted = self.formattedSaveSelectionName()
        if not dialogs_file_selection.FilenameEntryFormatter.is_valid_name(formatted):
            self.saveSelectionNameLineEdit.setStyleSheet('border: 2px solid red;')
            return False

        self.selectionSaveName = save_name
        self.selectionSaveNameFormatted = formatted
        return True

    def ok_cb(self, event=None):
        self.clickedButton = self.sender()

        valid_save_name = self.checkSaveSelectionName()
        if not valid_save_name:
            return

        if self.allowMultiSelection:
            if not self.selectionSequence:
                return

            selected_models = list(self.selectionSequence)
            if len(selected_models) > 1:
                ordered_models = self.askSelectedModelsOrder(selected_models)
                if ordered_models is None:
                    return
                selected_models = ordered_models

            self.selectedModel = selected_models
            self.cancel = False
            self.close()
            return

        selected_items = self.listBox.selectedItems()
        if not selected_items:
            return

        selected_models = [item.text() for item in selected_items]
        if len(selected_models) > 1:
            ordered_models = self.askSelectedModelsOrder(selected_models)
            if ordered_models is None:
                return
            self.selectedModel = ordered_models
            self.cancel = False
            self.close()
            return

        item = selected_items[0]
        model = item.text()
        if model == 'Add custom model...':
            modelName = self._runAddCustomModelWorkflow()
            if modelName is None:
                return
            item = QListWidgetItem(modelName)
            self.listBox.addItem(item)
            self.listBox.setCurrentItem(item)
        elif model == 'Automatic thresholding':
            self.selectedModel = model
            self.cancel = False
            self.close()
        else:
            self.selectedModel = model
            self.cancel = False
            self.close()

    def cancel_cb(self, event=None):
        self.cancel = True
        self.selectedModel = None
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        super().show()

        horizontal_sb = self.listBox.horizontalScrollBar()
        while horizontal_sb.isVisible():
            self.resize(self.height(), self.width() + 10)

        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()


class downloadModel:
    def __init__(self, model_name, parent=None):
        self.loop = None
        self.model_name = model_name
        self._parent = parent

    def download(self):
        model_url = downloads._model_url(self.model_name)
        if model_url is None:
            return
        
        _, model_path = downloads.get_model_path(
            self.model_name, create_temp_dir=False
        )
        model_name = self.model_name
        model_exists = downloads.check_model_exists(model_path, model_name)
        if not model_exists:
            self.warnDownloadModel(model_path, self.model_name)
        try:
            self._parent.logger.info(
                f'Downloading {self.model_name} model(s) to "{model_path}"'
            )
        except Exception as err:
            pass
        
        success = downloads.download_model(self.model_name)
        if not success:
            self.criticalDowloadFailed()
        
    def warnDownloadModel(self, model_path, model_name):
        txt = html_utils.paragraph(
            'Cell-ACDC needs to <b>download the model</b> '
            f'<code>{model_name}</code>.<br><br>'
            'The files will be dowloaded into the following folder:<br><br>'
            f'<code>{model_path}</code><br><br>'
            '<b>Progress</b> will be displayed in the <b>terminal</b>.<br>'
        )
        msg = widgets_message_box.myMessageBox()
        msg.information(self._parent, 'Download model', txt)

    def criticalDowloadFailed(self):
        import cellacdc
        model_name = self.model_name
        m = model_name.lower()
        weights_filenames = getattr(cellacdc, f'{m}_weights_filenames')
        url, alternative_url = downloads._model_url(
            model_name, return_alternative=True
        )
        url_href = f'<a href="{url}">this link</a>'
        alternative_url_href = f'<a href="{alternative_url}">this link</a>'
        _, model_path = downloads.get_model_path(model_name, create_temp_dir=False)
        txt = html_utils.paragraph(f"""
            Automatic download of {model_name} failed.<br><br>
            Please, <b>manually download</b> the model weights from {url_href} or
            {alternative_url_href}.<br><br>
            Next, unzip the content (or move the files if not a zip archive) 
            of the downloaded file into the following folder:<br><br>
            <code>{model_path}</code><br><br>
            <i>NOTE: if clicking on the link above does not work
            copy one of the links below and paste it into the browser</i><br><br>
            <code>{url}</code>
            <br><br>
            <code>{alternative_url}</code>
        """)
        weights_paths = [os.path.join(model_path, f) for f in weights_filenames]
        weights = '\n\n'.join(weights_paths)
        detailsText = (
            f'Files that {model_name} requires:\n\n'
            f'{weights}'
        )
        msg = widgets_message_box.myMessageBox()
        msg.critical(
            self._parent, f'Download of {model_name} failed', txt, 
            detailsText=detailsText
        )
        self.close_()

    def close_(self):
        return
        # self.hide()
        # self.close()
        # if self.loop is not None:
        #     self.loop.exit()


class InstallPyTorchDialog(dialogs_base.QBaseDialog):
    def __init__(
            self, parent=None, caller_name='Cell-ACDC',
            target_python=None, platform=None, found_cuda_version=None,
            possible_links=None, torch_version=None,
        ):
        """Dialog to choose how to install PyTorch.

        Parameters
        ----------
        possible_links : tuple, optional
            Precomputed result of `torch_utils.get_torch_options`, i.e.
            (options, preferred_index). If None, it is fetched here
            (needs network access).
        torch_version : str, optional
            PyTorch release or PEP 440 constraint (e.g. '<2') required by the
            model. Pip resolves a compatible torchvision release.
        """
        super().__init__(parent=parent)

        self.cancel = True
        self.platform = None
        self.indexUrl = None
        self._targetPython = target_python
        self._torchVersion = torch_version

        mainLayout = QVBoxLayout()

        innerLayout = QGridLayout()

        iconLabel = QLabel(self)
        standardIcon = getattr(QStyle, 'SP_MessageBoxInformation')
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        iconLabel.setPixmap(pixmap)
        innerLayout.addWidget(iconLabel, 0, 0, alignment=Qt.AlignTop)

        if found_cuda_version is None:
            found_cuda_version = torch_utils.get_driver_cuda_version()

        if possible_links is None:
            options_kwargs = {'driver_cuda_version': found_cuda_version}
            if target_python is not None:
                options_kwargs['exec_path'] = target_python
            if torch_version:
                options_kwargs['torch_version'] = torch_version
            possible_links = torch_utils.get_torch_options_safe(**options_kwargs)
        self._options, self._preferred = possible_links
        self._detectedOnMac = sys.platform == 'darwin'

        href = html_utils.href_tag('How to install PyTorch', urls.install_pytorch)
        important = html_utils.to_admonition("""
            Should you choose to install PyTorch yourself, <b>make sure to 
            activate<br>
            the correct <code>acdc</code> environment first</b>.
        """, admonition_type='important')

        infoText = html_utils.paragraph(f"""
            {caller_name} needs to <b>install the package</b> <code>PyTorch</code>.<br><br>
            Select your preferences and click ok to install it now.<br>
            You can check the currently running `CUDA UMD Version` by running:""")
        if found_cuda_version is not False and found_cuda_version is not None:
            infoText += html_utils.paragraph(
                f'Current CUDA UMD Version: {found_cuda_version[0]}.{found_cuda_version[1]}'
            )

        self.commandWidgetSMI = widgets_message_box.CopiableCommandWidget(
            command='nvidia-smi'
        )
        infoText2 = html_utils.paragraph(f"""
            You will have to <b>confirm the installation in the terminal</b>.<br><br>
            Alternatively, you can close {caller_name} and run the command 
            yourself.<br><br>
            For more details see this guide: {href}<br>
            {important}
        """)
        innerLayout.addWidget(QLabel(infoText), 0, 1)
        innerLayout.addWidget(self.commandWidgetSMI, 1, 1)
        innerLayout.addWidget(QLabel(infoText2), 2, 1)
        innerLayout.addItem(QSpacerItem(10, 10), 3, 1)

        preferencesLayout = QGridLayout()

        row = 0
        if torch_version:
            preferencesLayout.addWidget(QLabel('PyTorch version'), row, 0)
            preferencesLayout.addWidget(QLabel(torch_version), row, 1)
            row += 1
        self.osCombobox = QComboBox()
        self.osCombobox.addItems(['Linux', 'Mac', 'Windows'])
        osLabel = QLabel('Your OS')
        preferencesLayout.addWidget(osLabel, row, 0)
        preferencesLayout.addWidget(self.osCombobox, row, 1)

        if not platform:
            if is_mac:
                self.osCombobox.setCurrentText('Mac')
            elif is_win:
                self.osCombobox.setCurrentText('Windows')
        else:
            if platform.startswith('linux'):
                platform = 'linux'
            if platform.startswith('darwin'):
                platform = 'mac'
            if platform.startswith("win"):
                platform = 'win'
            if platform == 'mac':
                self.osCombobox.setCurrentText('Mac')
            elif platform == 'win':
                self.osCombobox.setCurrentText('Windows')
            elif platform == 'linux':
                self.osCombobox.setCurrentText('Linux')

        row += 1
        self.pkgManagerCombobox = QComboBox()
        self.pkgManagerCombobox.addItems(['Pip'])
        if not is_conda_env():
            self.pkgManagerCombobox.setCurrentText('Pip')
            self.pkgManagerCombobox.setDisabled(True)

        pkgManagerLabel = QLabel('Package manager')
        preferencesLayout.addWidget(pkgManagerLabel, row, 0)
        preferencesLayout.addWidget(self.pkgManagerCombobox, row, 1)

        row += 1
        # Items are filled in populatePlatforms() below
        self.cmptPlatformCombobox = QComboBox()
        preferencesLayout.addWidget(QLabel('Compute Platform'), row, 0)
        preferencesLayout.addWidget(self.cmptPlatformCombobox, row, 1)

        row += 1
        _, pip_prefix = installs.get_pip_conda_prefix(exec_path=target_python)
        self.commandWidget = widgets_message_box.CopiableCommandWidget(
            command=f'{pip_prefix} torch'
        )
        preferencesLayout.addWidget(QLabel('Run this command: '), row, 0)
        preferencesLayout.addWidget(self.commandWidget, row, 1, 1, 2)
        preferencesLayout.setColumnStretch(0, 0)
        preferencesLayout.setColumnStretch(1, 0)
        preferencesLayout.setColumnStretch(2, 1)

        innerLayout.addLayout(preferencesLayout, 4, 1)

        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addLayout(innerLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        # Changing the OS changes the available platforms -> repopulate
        self.osCombobox.currentTextChanged.connect(self.populatePlatforms)
        self.pkgManagerCombobox.currentTextChanged.connect(self.updateCommand)
        self.cmptPlatformCombobox.currentTextChanged.connect(self.updateCommand)

        self.populatePlatforms()

    def _platformOptions(self, osText):
        """(options, preferred_index) for the selected OS."""
        if osText == 'Mac':
            return [('Default (CPU / Apple GPU)', None)], 0
        if self._detectedOnMac:
            # Options were detected on a Mac, so there is no CUDA info
            return [('CPU', f'{torch_utils.BASE_PYTORCH_LINK}cpu')], 0
        return self._options, self._preferred

    def populatePlatforms(self, *args, **kwargs):
        options, preferred = self._platformOptions(
            self.osCombobox.currentText()
        )
        self._urls = dict(options)

        self.cmptPlatformCombobox.blockSignals(True)
        self.cmptPlatformCombobox.clear()
        self.cmptPlatformCombobox.addItems([label for label, _ in options])
        self.cmptPlatformCombobox.setCurrentIndex(preferred)
        self.cmptPlatformCombobox.blockSignals(False)

        self.updateCommand()

    def updateCommand(self, *args, **kwargs):
        indexUrl = self._urls.get(self.cmptPlatformCombobox.currentText())
        _, pip_prefix = installs.get_pip_conda_prefix(
            exec_path=self._targetPython
            )
        torch_spec = torch_requirement(self._torchVersion, quoted=True)
        command = f'{pip_prefix} {torch_spec} torchvision'
        if indexUrl is not None:    # None -> default PyPI install (macOS)
            command += f' --index-url {indexUrl}'
        self.commandWidget.setCommand(command)

    def ok_cb(self):
        label = self.cmptPlatformCombobox.currentText()
        self.platform = label
        self.indexUrl = self._urls.get(label)
        self.command = self.commandWidget.command()
        self.cancel = False
        self.close()
