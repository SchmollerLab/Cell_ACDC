from qtpy.QtCore import (
    QAbstractItemModel,
    QAbstractListModel,
    QDataStream,
    QIODevice,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    Qt,
    Signal,
    QSize,
    QByteArray,
    QObject,
    QMimeData,
)
from qtpy.QtGui import QBrush
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QTextEdit,
    QWidget,
)

from .. import html_utils
from .palette import LISTWIDGET_STYLESHEET, TREEWIDGET_STYLESHEET, font

class _ReorderableListModel(QAbstractListModel):
    """
    ReorderableListModel is a list model which implements reordering of its
    items via drag-n-drop
    """

    dragDropFinished = Signal()

    def __init__(self, items, parent=None):
        QAbstractItemModel.__init__(self, parent)
        self.nodes = items
        self.lastDroppedItems = []
        self.pendingRemoveRowsAfterDrop = False

    def rowForItem(self, text):
        """
        rowForItem method returns the row corresponding to the passed in item
        or None if no such item exists in the model
        """
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
        return (
            Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsDropEnabled
        )

    def insertRows(self, row, count, index):
        if index.isValid():
            return False
        if count <= 0:
            return False
        # inserting 'count' empty rows starting at 'row'
        self.beginInsertRows(QModelIndex(), row, row + count - 1)
        for i in range(0, count):
            self.nodes.insert(row + i, "")
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
            """
            If we got here, it means this call to removeRows is the automatic
            'cleanup' action after drag-n-drop performed by Qt
            """
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
        return ["application/vnd.treeviewdragdrop.list"]

    def mimeData(self, indexes):
        mimedata = QMimeData()
        encoded_data = QByteArray()
        stream = QDataStream(encoded_data, QIODevice.WriteOnly)
        for index in indexes:
            if index.isValid():
                text = self.data(index, 0)
        stream << QByteArray(text.encode("utf-8"))
        mimedata.setData("application/vnd.treeviewdragdrop.list", encoded_data)
        return mimedata

    def dropMimeData(self, data, action, row, column, parent):
        if action == Qt.IgnoreAction:
            return True
        if not data.hasFormat("application/vnd.treeviewdragdrop.list"):
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

        encoded_data = data.data("application/vnd.treeviewdragdrop.list")
        stream = QDataStream(encoded_data, QIODevice.ReadOnly)

        new_items = []
        rows = 0
        while not stream.atEnd():
            text = QByteArray()
            stream >> text
            text = bytes(text).decode("utf-8")
            index = self.nodes.index(text)
            new_items.append((text, index))
            rows += 1

        self.lastDroppedItems = []
        for text, index in new_items:
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
    def __init__(self, items=None, parent=None, isSingleSelection=False) -> None:
        super().__init__(parent)
        if items is None:
            items = []

        self.isSingleSelection = isSingleSelection
        self._model = _ReorderableListModel(items)
        self._selectionModel = _SelectionModel(self._model)
        self._model.dragDropFinished.connect(self._selectionModel.onModelItemsReordered)
        self.setModel(self._model)
        self.setSelectionModel(self._selectionModel)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDragDropOverwriteMode(False)
        styleSheet = f"""
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
        """
        self.setStyleSheet(styleSheet)

    def setItems(self, items):
        self._model.nodes = items

    def items(self):
        return self._model.nodes

    # def mouseReleaseEvent(self, e: QMouseEvent) -> None:
    #     super().mouseReleaseEvent(e)
    #     self._selectionModel.reset()


class listWidget(QListWidget):
    def __init__(
        self, *args, isMultipleSelection=False, minimizeHeight=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.itemHeight = None
        self.setStyleSheet(LISTWIDGET_STYLESHEET)
        self.setFont(font)
        if isMultipleSelection:
            self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self.minimizeHeight = minimizeHeight

    def setSelectedAll(self, selected):
        for i in range(self.count()):
            self.item(i).setSelected(selected)

    def setSelectedItems(self, itemsText):
        for i in range(self.count()):
            item = self.item(i)
            item.setSelected(item.text() in itemsText)

    def addItems(self, labels) -> None:
        super().addItems(labels)
        if self.itemHeight is not None:
            self.setItemHeight()

        if self.minimizeHeight:
            itemHeight = self.sizeHintForRow(0)
            self.setMaximumHeight(itemHeight * self.count() + itemHeight * 2)

    def addItem(self, text):
        super().addItem(text)
        if self.itemHeight is None:
            return
        self.setItemHeight()

    def setItemHeight(self, height=40):
        self.itemHeight = height
        for i in range(self.count()):
            item = self.item(i)
            item.setSizeHint(QSize(0, height))

    def selectedItemsText(self):
        return [item.text() for item in self.selectedItems()]


class OrderableListWidget(QWidget):
    sigEnterEvent = Signal(object)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._labels = []

    def setParentItem(self, item):
        self._item = item

    def setLabelsColor(self, selected):
        if selected:
            stylesheet = "color : black"
        else:
            stylesheet = ""

        for label in self._labels:
            label.setStyleSheet(stylesheet)

    def enterEvent(self, event):
        super().enterEvent(event)
        self.setLabelsColor(True)
        self.sigEnterEvent.emit(self._item)

    # def leaveEvent(self, event):
    #     super().leaveEvent(event)
    #     self.setLabelsColor(self._item.isSelected())
    #     printl('leave', self._item.isSelected())

    def addLabel(self, label):
        self._labels.append(label)


class OrderableList(listWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self.itemEntered.connect(self.onItemEntered)

    def onItemEntered(self, enteredItem):
        enteredRow = self.row(enteredItem)
        for i in range(self.count()):
            item = self.item(i)
            item._container.setLabelsColor(i == enteredRow or item.isSelected())

    def leaveEvent(self, event):
        super().leaveEvent(event)
        for i in range(self.count()):
            item = self.item(i)
            item._container.setLabelsColor(item.isSelected())

    def addItems(self, items):
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        nr_items = len(items)
        nn = [str(n) for n in range(1, nr_items + 1)]
        for i, item in enumerate(items):
            itemW = QListWidgetItem()
            itemContainer = OrderableListWidget()
            itemContainer.setParentItem(itemW)
            itemText = QLabel(item)
            tableNrLabel = QLabel("| Table nr.")
            itemContainer.addLabel(tableNrLabel)
            itemContainer.addLabel(itemText)
            itemLayout = QHBoxLayout()
            itemNumberWidget = QComboBox()
            itemNumberWidget.addItems(nn)
            itemLayout.addWidget(itemText)
            itemLayout.addWidget(tableNrLabel)
            itemLayout.addWidget(itemNumberWidget)
            itemContainer.setLayout(itemLayout)
            itemLayout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
            itemW.setSizeHint(itemContainer.sizeHint())
            self.addItem(itemW)
            self.setItemWidget(itemW, itemContainer)
            itemW._text = item
            itemW._nrWidget = itemNumberWidget
            itemW._container = itemContainer
            itemNumberWidget.setDisabled(True)
            itemNumberWidget.textActivated.connect(self.onTextActivated)
            itemNumberWidget._currentNr = 1
            itemNumberWidget.row = i
            itemContainer.sigEnterEvent.connect(self.onItemEntered)

        self.itemSelectionChanged.connect(self.onItemSelectionChanged)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.clearSelection()
            event.ignore()
            return
        super().keyPressEvent(event)

    def updateNr(self):
        for i in range(self.count()):
            item = self.item(i)
            item._currentNr = int(item._nrWidget.currentText())

    def onItemSelectionChanged(self):
        for i in range(self.count()):
            item = self.item(i)
            item._container.setLabelsColor(item.isSelected())
            item._nrWidget.setDisabled(not item.isSelected())
            if item._nrWidget.currentText() != "1":
                item._nrWidget.setCurrentText("1")
                item._currentNr = 1

        for i, item in enumerate(self.selectedItems()):
            item._nrWidget.setCurrentText(f"{i + 1}")
            item._currentNr = i + 1

    def onTextActivated(self, text):
        changedNr = self.sender()._currentNr
        for item in self.selectedItems():
            row = self.row(item)
            if self.sender().row == row:
                changedNr = item._currentNr
                continue

        for item in self.selectedItems():
            row = self.row(item)
            if self.sender().row == row:
                continue
            nr = int(item._nrWidget.currentText())
            if nr == int(text):
                item._nrWidget.setCurrentText(str(changedNr))
                break

        self.updateNr()


class TreeWidget(QTreeWidget):
    def __init__(self, *args, multiSelection=False):
        super().__init__(*args)
        self.setStyleSheet(TREEWIDGET_STYLESHEET)
        self.setFont(font)
        if multiSelection:
            self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.itemClicked.connect(self.selectAllChildren)

        self.isCtrlDown = False
        self.isShiftDown = False

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.clearSelection()
        elif ev.key() == Qt.Key_Control:
            self.isCtrlDown = True
        elif ev.key() == Qt.Key_Shift:
            self.isShiftDown = True

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key_Control:
            self.isCtrlDown = False
        elif ev.key() == Qt.Key_Shift:
            self.isShiftDown = False

    def onFocusChanged(self):
        self.isCtrlDown = False
        self.isShiftDown = False

    def selectAllChildren(self, item_or_label):
        label = None
        if isinstance(item_or_label, QLabel):
            label = item_or_label
        else:
            item = item_or_label
            if item.childCount() == 0:
                return

        if label is not None:
            if not self.isCtrlDown and not self.isShiftDown:
                self.clearSelection()
            label.item.setSelected(True)
            if self.isShiftDown:
                selectionStarted = False
                it = QTreeWidgetItemIterator(self)
                while it:
                    item = it.value()
                    if item is None:
                        break
                    if item.isSelected():
                        selectionStarted = not selectionStarted
                    if selectionStarted:
                        item.setSelected(True)
                    it += 1

        for item in self.selectedItems():
            if item.parent() is None:
                for i in range(item.childCount()):
                    item.child(i).setSelected(True)



class TreeWidgetItem(QTreeWidgetItem):
    def __init__(self, *args, columnColors=None):
        super().__init__(*args)

        if columnColors is not None:
            for c, color in enumerate(columnColors):
                if color is None:
                    continue
                self.setBackground(c, QBrush(color))


class FilterObject(QObject):
    sigFilteredEvent = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()

    def eventFilter(self, object, event):
        self.sigFilteredEvent.emit(object, event)
        return super().eventFilter(object, event)


class readOnlyQList(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.items = []

    def addItems(self, items):
        self.items.extend(items)
        items = [str(item) for item in self.items]
        columnList = html_utils.paragraph("<br>".join(items))
        self.setText(columnList)

