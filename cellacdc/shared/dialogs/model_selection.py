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
from cellacdc import myutils
from cellacdc import fonts
from ..widgets import item_views as widgets_item_views
from ..widgets import style as widgets_style


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

        models = myutils.get_list_of_models()
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
