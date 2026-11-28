"""Lists and trees: QDialogListbox (the list selection dialog; pass
allowEmptySelection=False wherever an answer is required), listWidget
with rich-text items (RichTextListWidgetItem, _RichTextItemEventFilter),
TreeWidget, TreeWidgetItem and readOnlyQList.
"""
from functools import partial
from qtpy.QtCore import Signal, Qt, QSize, QEvent, QEventLoop, QObject
from qtpy.QtGui import QFont, QBrush, QGuiApplication, QTextDocument
from qtpy.QtWidgets import (
    QTextEdit, QLabel, QHBoxLayout, QVBoxLayout, QTreeWidgetItemIterator,
    QPushButton, QDialog, QListWidget, QAbstractItemView, QTreeWidget,
    QTreeWidgetItem, QListWidgetItem
)
from cellacdc import _window_stacking
from cellacdc import html_utils
from . import push_buttons as widgets_push_buttons
from . import style as widgets_style
from . import message_box as widgets_message_box


class QDialogListbox(QDialog):
    sigSelectionConfirmed = Signal(list)

    def __init__(
            self, title, text, items, cancelText='Cancel',
            multiSelection=True, parent=None,
            additionalButtons=(), includeSelectionHelp=False,
            allowSingleSelection=True, preSelectedItems=None, 
            allowEmptySelection=True
        ):
        self.cancel = True
        # Also when closed with the title bar button (no ok_cb/cancel_cb)
        self.selectedItemsText = None
        items = list(items)
        
        super().__init__(parent)
        self.setWindowTitle(title)
        
        if preSelectedItems is None:
            if items:
                preSelectedItems = (items[0],)
            else:
                preSelectedItems = set()

        self.allowSingleSelection = allowSingleSelection
        self.allowEmptySelection = allowEmptySelection
        self.multiSelection = multiSelection

        mainLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        bottomLayout = QHBoxLayout()

        self.mainLayout = mainLayout

        label = QLabel(text)
        _font = QFont()
        _font.setPixelSize(13)
        label.setFont(_font)
        # padding: top, left, bottom, right
        label.setStyleSheet("padding:0px 0px 3px 0px;")
        topLayout.addWidget(label, alignment=Qt.AlignCenter)

        if includeSelectionHelp:
            selectionHelpLabel = QLabel()
            txt = html_utils.paragraph("""<br>
                <code>Ctrl+Click</code> <i>to select multiple items</i><br>
                <code>Shift+Click</code> <i>to select a range of items</i><br>
            """)
            selectionHelpLabel.setText(txt)
            topLayout.addWidget(label, alignment=Qt.AlignCenter)

        listBox = listWidget()
        listBox.setFont(_font)
        listBox.addItems(items)            
        if multiSelection:
            listBox.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            listBox.setSelectionMode(
                QAbstractItemView.SelectionMode.SingleSelection)
        listBox.setCurrentRow(0)
        for i in range(listBox.count()):
            item = listBox.item(i)
            item.setSelected(item.text() in preSelectedItems)
            
        self.listBox = listBox
        if not multiSelection:
            listBox.itemDoubleClicked.connect(self.ok_cb)
        topLayout.addWidget(listBox)

        if cancelText.lower().find('cancel') != -1:
            cancelButton = widgets_push_buttons.cancelPushButton(cancelText)
        else:
            cancelButton = QPushButton(cancelText)
        okButton = widgets_push_buttons.okPushButton(' Ok ')
        self.okButton = okButton

        bottomLayout.addStretch(1)
        bottomLayout.addWidget(cancelButton)
        bottomLayout.addSpacing(20)

        if additionalButtons:
            self._additionalButtons = []
            for button in additionalButtons:
                if isinstance(button, str):
                    _button, isCancelButton = widgets_push_buttons.getPushButton(button)
                    self._additionalButtons.append(_button)
                    bottomLayout.addWidget(_button)
                    # The button is passed explicitly: ok_cb never reads
                    # `sender()` (see `ok_cb`)
                    _button.clicked.connect(
                        partial(self.ok_cb, button=_button)
                    )
                else:
                    bottomLayout.addWidget(button)

        bottomLayout.addWidget(okButton)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        if multiSelection:
            listBox.itemClicked.connect(self.onItemClicked)
            listBox.itemSelectionChanged.connect(self.onItemSelectionChanged)

        self.setStyleSheet(widgets_style.LISTWIDGET_STYLESHEET)
        self.areItemsSelected = [
            listBox.item(i).isSelected() for i in range(listBox.count())
        ]
        self.setFont(widgets_style.font)
    
    def keyPressEvent(self, event) -> None:
        mod = event.modifiers()
        isModifier = mod == Qt.ShiftModifier or mod == Qt.ControlModifier
        if isModifier and self.multiSelection:
            self.listBox.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        elif event.key() == Qt.Key_Escape:
            if not self.multiSelection:
                # Single choice: Escape cancels (as the Cancel button)
                # instead of emptying the choice. Not `reject()`: only
                # `closeEvent` exits the loop of `exec_`
                self.cancel_cb(None)
                return
            self.listBox.clearSelection()
            event.ignore()
            return
        super().keyPressEvent(event)
    
    def onItemSelectionChanged(self):
        if not self.listBox.selectedItems():
            self.areItemsSelected = [
                False for i in range(self.listBox.count())
            ]
    
    def onItemClicked(self, item):
        mod = QGuiApplication.keyboardModifiers()
        if mod == Qt.ShiftModifier or mod == Qt.ControlModifier:
            self.listBox.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            return
        
        self.listBox.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        itemIdx = self.listBox.row(item)
        wasSelected = self.areItemsSelected[itemIdx]
        if wasSelected:
            item.setSelected(False)
        
        self.areItemsSelected = [
            self.listBox.item(i).isSelected() 
            for i in range(self.listBox.count())
        ]
        # self.listBox.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # else:
        #     selectedItems.append(item)
        
        # self.listBox.clearSelection()
        # for i in range(self.listBox.count()):
        #     item = self.listBox.item(i).setSelected(True)
        
        # print(self.listBox.selectedItems())
    
    def setSelectedItems(self, itemsTexts):
        for i in range(self.listBox.count()):
            item = self.listBox.item(i)
            if item.text() in itemsTexts:
                item.setSelected(True)
        self.listBox.update()

    def warnSelectionEmpty(self):
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
        if self.multiSelection:
            txt = html_utils.paragraph(
                'You need to <b>select at least one item!</b>.<br><br>'
                'Use <code>Ctrl+Click</code> to select multiple items<br>'
                'or <code>Shift+Click</code> to select a range of items'
            )
        else:
            # E.g., after Ctrl+Click on the selected item
            txt = html_utils.paragraph(
                'You need to <b>select an item!</b><br><br>'
                'Click an item of the list to select it.'
            )
        msg.warning(self, 'Selection cannot be empty!', txt)
    
    def ok_cb(self, checked=False, button=None):
        """Confirm the selection and close.

        `button` is the clicked button: None means the OK button (also for a
        double-click on an item and for direct calls by code or tests), else
        an additional button (connected with `partial`; buttons passed as
        widgets must be connected the same way to be reported). The clicked
        button is stored as `clickedButton`. Never `self.sender()` here:
        outside a slot of this dialog, PyQt returns its last proxy sender,
        possibly a deleted object, and touching it segfaults.

        `allowEmptySelection=False` refuses an empty selection only when
        confirming with OK: additional buttons (e.g. "Do not track") answer
        without needing a selection.
        """
        if button is None:
            button = self.okButton
        self.clickedButton = button
        selectedItems = self.listBox.selectedItems()
        selectedItemsText = [item.text() for item in selectedItems]
        if not self.allowSingleSelection and len(selectedItemsText) < 2:
            msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
            txt = html_utils.paragraph(
                'You need to <b>select two or more items</b>.<br><br>'
                'Use <code>Ctrl+Click</code> to select multiple items<br>, or<br>'
                '<code>Shift+Click</code> to select a range of items'
            )
            msg.warning(self, 'Select two or more items', txt)
            return

        isOkButton = button is self.okButton
        isEmpty = not selectedItemsText
        if isOkButton and isEmpty and not self.allowEmptySelection:
            self.warnSelectionEmpty()
            return

        self.cancel = False
        self.selectedItemsText = selectedItemsText
        
        self.sigSelectionConfirmed.emit(self.selectedItemsText)
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.selectedItemsText = None
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


class _RichTextItemEventFilter(QObject):
    def __init__(self, item):
        super().__init__()
        self.item = item

    def eventFilter(self, obj, event):
        listWidget = self.item._listWidget

        if listWidget is None:
            return super().eventFilter(obj, event)

        if event.type() == QEvent.MouseMove:
            pos = (
                event.position().toPoint()
                if hasattr(event, 'position')
                else event.pos()
            )            
            hoveredItem = listWidget.itemAt(pos)
            self.item.setHovered(hoveredItem is self.item)

        elif event.type() == QEvent.Leave:
            self.item.setHovered(False)

        return super().eventFilter(obj, event)


class RichTextListWidgetItem(QListWidgetItem):
    def __init__(self, text='', parent=None):
        super().__init__(parent)

        self._plainText = ''
        self._listWidget = None
        self._hovered = False

        self.label = QLabel()
        self.label.setTextFormat(Qt.RichText)
        self.label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.label.setAttribute(
            Qt.WA_TransparentForMouseEvents
        )

        self._eventFilter = _RichTextItemEventFilter(self)

        self.setText(text)

    def setText(self, text):
        """Set rich text while keeping the plain text separately."""
        doc = QTextDocument()
        doc.setHtml(text)

        self._plainText = doc.toPlainText()

        # Do NOT let QListWidgetItem paint its own text.
        super().setText('')

        # Rich representation is painted by the QLabel.
        self.label.setText(text)

    def text(self):
        return self._plainText

    def setListWidget(self, listWidget):
        self._listWidget = listWidget

        listWidget.viewport().installEventFilter(
            self._eventFilter
        )

        self._updateLabelStyle()

    def setHovered(self, hovered):
        if self._hovered == hovered:
            return

        self._hovered = hovered
        self._updateLabelStyle()

    def _updateLabelStyle(self):
        if self._listWidget is None:
            return

        if self._hovered or self.isSelected():
            color = 'black'
        else:
            color = self._listWidget.palette().color(
                self._listWidget.foregroundRole()
            ).name()

        self.label.setStyleSheet(
            f'background: transparent; color: {color};'
        )

class listWidget(QListWidget):
    def __init__(
            self, 
            *args, 
            isMultipleSelection=False, 
            minimizeHeight=False, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.itemHeight = None
        self.setStyleSheet(widgets_style.LISTWIDGET_STYLESHEET)
        self.setFont(widgets_style.font)
        if isMultipleSelection:
            self.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection
            )
        
        self.minimizeHeight = minimizeHeight
    
    def addRichTextItem(self, text: str):
        item = RichTextListWidgetItem(text)

        self.addItem(item)
        self.setItemWidget(item, item.label)

        item.setListWidget(self)

        return item

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
            self.setMaximumHeight(itemHeight * self.count() + itemHeight*2)
    
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


class TreeWidget(QTreeWidget):
    def __init__(self, *args, multiSelection=False):
        super().__init__(*args)    
        self.setStyleSheet(widgets_style.TREEWIDGET_STYLESHEET)
        self.setFont(widgets_style.font)
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


class readOnlyQList(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.items = []

    def addItems(self, items):
        self.items.extend(items)
        items = [str(item) for item in self.items]
        columnList = html_utils.paragraph('<br>'.join(items))
        self.setText(columnList)
        
    def clear(self):
        self.items = []
        self.setText('')
