"""Generic input, list, tree and progress dialogs used by several program
parts, library modules (prompts.py, service/prompts.py) and shared
widgets (from apps.py).

QDialogCombobox, QLineEditDialog (with POSITIVE_FLOAT_REGEX),
NumericEntryDialog, QInput, QDialogEntriesWidget, the tree selectors
QTreeDialog, TreeSelectorDialog (spotMAX subclasses it) and
TreesSelectorDialog, SearchableListboxDialog (recent paths) and
QDialogWorkerProgress, plus the widgets that only these dialogs use.
"""
import os
from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItem
from qtpy.QtWidgets import QLineEdit, QTreeWidgetItem
from cellacdc import path as acdc_path
from ..widgets import search_widgets as widgets_search_widgets
import re
from typing import List, Tuple
import datetime
import numpy as np
import pandas as pd
import math
from qtpy.QtCore import Signal, QEventLoop, QTimer
from qtpy.QtWidgets import (
    QLabel, QVBoxLayout, QHBoxLayout, QDialog, QFormLayout, QAbstractItemView,
    QCheckBox, QSizePolicy, QComboBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTreeWidget, QHeaderView
)
from cellacdc import _window_stacking
from cellacdc import html_utils
from cellacdc import is_mac
from cellacdc.myutils import filesystem
from cellacdc import qutils
from qtpy.QtGui import QStandardItemModel
from ..widgets import (
    basic_widgets as widgets_basic_widgets,
    combo_boxes as widgets_combo_boxes,
    item_views as widgets_item_views,
    message_box as widgets_message_box,
    numeric_inputs as widgets_numeric_inputs,
    push_buttons as widgets_push_buttons,
)
from cellacdc.acdc_regex import float_regex
from cellacdc import fonts
from . import base as dialogs_base


class SearchLineEdit(QLineEdit):
    def __init__(self, text='Search...', parent=None):
        super().__init__(parent)
        
        self._text = text
        self.initSearch()
        self.setFocusPolicy(Qt.ClickFocus)
        
    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        if super().text() == self._text:
            self.setText('')
        self.setStyleSheet('')
    
    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        if not super().text():
            self.initSearch()
    
    def initSearch(self):
        self.setText(self._text)
        self.setStyleSheet('color: rgb(150, 150, 150)')
        self.clearFocus()
    
    def text(self):
        if super().text() == self._text:
            return ''
        return super().text()


class _RecentItemsSearchCompleter(widgets_search_widgets.ButtonSearchCompleter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alwaysRecency = False

    def setAlwaysRecency(self, alwaysRecency):
        self.alwaysRecency = alwaysRecency
        self.invalidate()
        self.sort(0, Qt.AscendingOrder)

    def lessThan(self, left, right):
        if self.alwaysRecency:
            return left.row() < right.row()

        left_score = self._getScore(left)
        right_score = self._getScore(right)

        if left_score[0] != right_score[0]:
            return left_score[0] < right_score[0]
        
        if left_score[1] != right_score[1]:
            return left_score[1] < right_score[1]

        return left.row() < right.row()


class StandardItem(QStandardItem):
    def __init__(self, other):
        super().__init__(other)
    
    def setData(self, value, role = ...):
        return super().setData(value, role)

class PathTreeWidgetItem(QTreeWidgetItem):
    def __init__(self, path, strings):
        path = acdc_path.unexpanduser(path)
        strings = [path, *strings]
        super().__init__(strings)
    
    def text(self, column):
        if column == 0:
            path = super().text(column)
            path = os.path.expanduser(path)
            return path
            
        return super().text(column)


POSITIVE_FLOAT_REGEX = float_regex(allow_negative=False)


class QDialogWorkerProgress(QDialog):
    sigClosed = Signal(bool)

    def __init__(
            self, title='Progress', infoTxt='',
            showInnerPbar=False, pbarDesc='',
            parent=None
        ):
        self.workerFinished = False
        self.aborted = False
        self.clickCount = 0
        super().__init__(parent)

        abort_text = 'Option+Command+C' if is_mac else 'Ctrl+Alt+C'
        self.abort_text = abort_text

        self.setWindowTitle(f'{title} ({abort_text} to cancel process)')
        self.setWindowFlags(Qt.Window)

        mainLayout = QVBoxLayout()
        pBarLayout = QGridLayout()

        if infoTxt:
            infoLabel = QLabel(infoTxt)
            mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        self.progressLabel = QLabel(pbarDesc)

        self.mainPbar = widgets_basic_widgets.ProgressBarWithETA(self)
        self.mainPbar.setValue(0)
        pBarLayout.addWidget(self.mainPbar, 0, 0)
        pBarLayout.addWidget(self.mainPbar.ETA_label, 0, 1)

        self.innerPbar = widgets_basic_widgets.ProgressBarWithETA(self)
        self.innerPbar.setValue(0)
        pBarLayout.addWidget(self.innerPbar, 1, 0)
        pBarLayout.addWidget(self.innerPbar.ETA_label, 1, 1)
        if showInnerPbar:
            self.innerPbar.show()
        else:
            self.innerPbar.hide()

        self.logConsole = widgets_basic_widgets.QLogConsole()

        mainLayout.addWidget(self.progressLabel)
        mainLayout.addLayout(pBarLayout)
        mainLayout.addWidget(self.logConsole)

        self.setLayout(mainLayout)
        # self.setModal(True)

    def keyPressEvent(self, event):
        isCtrlAlt = event.modifiers() == (Qt.ControlModifier | Qt.AltModifier)
        if isCtrlAlt and event.key() == Qt.Key_C:
            doAbort = self.askAbort()
            if doAbort:
                self.aborted = True
                self.workerFinished = True
                self.close()

    def askAbort(self):
        msg = widgets_message_box.myMessageBox()
        txt = html_utils.paragraph(f"""
            Aborting with <code>{self.abort_text}</code> to abort is 
            <b>not safe</b>.<br><br>
            The system status cannot be predicted and
            it will <b>require a restart</b>.<br><br>
            Are you sure you want to abort?
        """)
        yesButton, noButton = msg.critical(
            self, 'Are you sure you want to abort?', txt,
            buttonsTexts=('Yes', 'No')
        )
        return msg.clickedButton == yesButton

    def closeEvent(self, event):
        if not self.workerFinished:
            event.ignore()
            return

        self.sigClosed.emit(self.aborted)
    
    def log(self, text):
        self.logConsole.append(text)

    def show(self, app):
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        QDialog.show(self)
        screen = app.primaryScreen()
        screenWidth = screen.size().width()
        screenHeight = screen.size().height()
        parentGeometry = self.parent().geometry()
        mainWinLeft, mainWinWidth = parentGeometry.left(), parentGeometry.width()
        mainWinTop, mainWinHeight = parentGeometry.top(), parentGeometry.height()
        mainWinCenterX = int(mainWinLeft+mainWinWidth/2)
        mainWinCenterY = int(mainWinTop+mainWinHeight/2)

        width = int(screenWidth/3)
        width = width if self.width() < width else self.width()
        height = int(screenHeight/3)
        left = int(mainWinCenterX - width/2)
        left = left if left >= 0 else 0
        top = int(mainWinCenterY - height/2)

        self.setGeometry(left, top, width, height)

class QDialogCombobox(QDialog):
    def __init__(
            self, title, ComboBoxItems, informativeText,
            CbLabel='Select value:  ', parent=None,
            defaultChannelName=None, iconPixmap=None, centeredCombobox=False
        ):
        self.cancel = True
        self.selectedItemText = ''
        self.selectedItemIdx = None
        super().__init__(parent=parent)
        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()
        infoLayout = QHBoxLayout()
        topLayout = QHBoxLayout()
        bottomLayout = QHBoxLayout()

        self.mainLayout = mainLayout

        if iconPixmap is not None:
            label = QLabel()
            # padding: top, left, bottom, right
            # label.setStyleSheet("padding:5px 0px 12px 0px;")
            label.setPixmap(iconPixmap)
            infoLayout.addWidget(label)

        if informativeText:
            infoLabel = QLabel(informativeText)
            infoLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        if CbLabel:
            label = QLabel(CbLabel)
            topLayout.addWidget(label, alignment=Qt.AlignRight)

        if centeredCombobox:
            combobox = widgets_combo_boxes.QCenteredComboBox()
        else:
            combobox = QComboBox()
        combobox.addItems(ComboBoxItems)
        if defaultChannelName is not None and defaultChannelName in ComboBoxItems:
            combobox.setCurrentText(defaultChannelName)
        self.ComboBox = combobox
        topLayout.addWidget(combobox)
        topLayout.setContentsMargins(0, 10, 0, 0)

        okButton = widgets_push_buttons.okPushButton('Ok')

        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')

        bottomLayout.addStretch(1)
        bottomLayout.addWidget(cancelButton)
        bottomLayout.addSpacing(20)
        bottomLayout.addWidget(okButton)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(infoLayout)
        mainLayout.addLayout(topLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        # self.setModal(True)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        self.loop = None

        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        self.setFont(fonts.font)

    def ok_cb(self, checked=False):
        self.cancel = False
        self.selectedItemText = self.ComboBox.currentText()
        self.selectedItemIdx = self.ComboBox.currentIndex()
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        QDialog.show(self)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()


class QDialogEntriesWidget(QDialog):
    def __init__(self, entriesLabels, defaultTxts, winTitle='Input',
                 parent=None, font=None):
        self.cancel = True
        self.entriesTxt = []
        self.entriesLabels = entriesLabels
        self.QLEs = []
        super().__init__(parent)
        self.setWindowTitle(winTitle)

        mainLayout = QVBoxLayout()
        formLayout = QFormLayout()
        buttonsLayout = QHBoxLayout()

        if font is not None:
            self.setFont(font)

        for label, txt in zip(entriesLabels, defaultTxts):
            LE = QLineEdit()
            LE.setAlignment(Qt.AlignCenter)
            LE.setText(txt)
            formLayout.addRow(label, LE)
            self.QLEs.append(LE)

        okButton = widgets_push_buttons.okPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(formLayout)
        mainLayout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        self.formLayout = formLayout

        self.setLayout(mainLayout)
        # self.setModal(True)

    def ok_cb(self, event):
        self.cancel = False
        self.entriesTxt = [self.formLayout.itemAt(i, 1).widget().text()
                           for i in range(len(self.entriesLabels))]
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


class TreeSelectorDialog(dialogs_base.QBaseDialog):
    sigItemDoubleClicked = Signal(object)

    def __init__(
            self, title='Tree selector', infoTxt='', parent=None,
            multiSelection=True, widthFactor=None, heightFactor=None,
            expandOnDoubleClick=False, isTopLevelSelectable=True,
            allItemsExpanded=True, allowNoSelection=True
        ):
        super().__init__(parent)

        self.setWindowTitle(title)
        
        self.cancel = True
        self.widthFactor = widthFactor
        self.heightFactor = heightFactor
        self.allItemsExpanded = allItemsExpanded
        self.mainLayout = QVBoxLayout()
        self._isTopLevelSelectable = isTopLevelSelectable
        self.allowNoSelection = allowNoSelection

        if infoTxt:
            self.mainLayout.addWidget(QLabel(html_utils.paragraph(infoTxt)))
        
        self.treeWidget = widgets_item_views.TreeWidget(multiSelection=multiSelection)
        self.treeWidget.setExpandsOnDoubleClick(expandOnDoubleClick)
        self.treeWidget.setHeaderHidden(True)
        self.mainLayout.addWidget(self.treeWidget)

        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        self.mainLayout.addSpacing(20)
        self.mainLayout.addLayout(buttonsLayout)

        self.buttonsLayout = buttonsLayout

        self.setLayout(self.mainLayout)

        self.treeWidget.itemClicked.connect(self.onItemClicked)
        self.treeWidget.itemDoubleClicked.connect(self.onItemDoubleClicked)
    
    def onItemDoubleClicked(self, item):
        self.sigItemDoubleClicked.emit(item)
    
    def onItemClicked(self, item):
        if self._isTopLevelSelectable:
            return
        if item.parent() is None:
            item.setSelected(False)
        
    def addTree(self, tree: dict):
        for topLevel, children in tree.items():
            topLevelItem = widgets_item_views.TreeWidgetItem(self.treeWidget)
            topLevelItem.setText(0, topLevel)
            self.treeWidget.addTopLevelItem(topLevelItem)
            childrenItems = [widgets_item_views.TreeWidgetItem([c]) for c in children]
            topLevelItem.addChildren(childrenItems)
            if not self.allItemsExpanded:
                continue
            topLevelItem.setExpanded(True)
    
    def resizeVertical(self):
        if not self.isVisible():
            self.show()

        currentTreeWidgetHeight = self.treeWidget.height()
        treeWidgetHeight = 0
        for i in range(self.treeWidget.topLevelItemCount()):
            topLevelItem = self.treeWidget.topLevelItem(i)
            rect = self.treeWidget.visualItemRect(topLevelItem)
            treeWidgetHeight += rect.height()
            for j in range(topLevelItem.childCount()):
                childItem = topLevelItem.child(j)
                rect = self.treeWidget.visualItemRect(childItem)
                treeWidgetHeight += rect.height()
        
        deltaHeight = treeWidgetHeight - currentTreeWidgetHeight + 10
        self.resize(self.width(), self.height() + deltaHeight)
        self.move(self.x(), 20)
    
    def setCurrentItem(self, itemText: dict):
        if not itemText:
            return
        for i in range(self.treeWidget.topLevelItemCount()):
            topLevelItem = self.treeWidget.topLevelItem(i)
            topLevelName = topLevelItem.text(0)
            childText = itemText.get(topLevelName)
            if childText is None:
                continue
            for j in range(topLevelItem.childCount()):
                childItem = topLevelItem.child(j)
                childItemText = childItem.text(0)
                if childItemText == childText:
                    childItem.setSelected(True)
                    topLevelItem.setExpanded(True)
                    self.treeWidget.scrollToItem(topLevelItem)
                    break
    
    def selectedItems(self):
        self._selectedItems = {}
        for i in range(self.treeWidget.topLevelItemCount()):
            topLevelItem = self.treeWidget.topLevelItem(i)
            topLevelName = topLevelItem.text(0)
            for j in range(topLevelItem.childCount()):
                childItem = topLevelItem.child(j)
                if not childItem.isSelected():
                    continue
                if topLevelName not in self._selectedItems:
                    self._selectedItems[topLevelName] = [childItem.text(0)]
                else:
                    self._selectedItems[topLevelName].append(childItem.text(0))
        return self._selectedItems

    def warnSelectionIsEmpty(self):
        txt = html_utils.paragraph("""
            You did not select anything :(.<br><br>
            Please press <code>Cancel</code> to exit without selecting items. 
            Thanks! 
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(self, 'Selection is empty', txt)
    
    def ok_cb(self):
        if not self.allowNoSelection and not self.selectedItems():
            self.warnSelectionIsEmpty()
            return
        self.cancel = False
        self.close()
    
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.widthFactor is not None:
            self.resize(int(self.width()*self.widthFactor), self.height())
        if self.heightFactor is not None:
            self.resize(self.width(), int(self.height()*self.heightFactor))

class TreesSelectorDialog(dialogs_base.QBaseDialog):
    def __init__(
            self, trees, groupsDescr=None, title='Trees selector', 
            infoTxt='', parent=None
        ):
        super().__init__(parent)

        self.setWindowTitle(title)
        
        self.cancel = True
        self.mainLayout = QVBoxLayout()

        if infoTxt:
            self.mainLayout.addWidget(QLabel(html_utils.paragraph(infoTxt)))

        self.treeWidgets = {}
        self.setLayout(self.mainLayout)
    
        createdGroupLayouts = {}
        for treeName, tree in trees.items():
            if groupsDescr is None:
                groupName = ''
            else:
                groupName = groupsDescr.get(treeName, 'Group info missing')
            groupLayout = createdGroupLayouts.get(groupName, None)
            if groupLayout is None:
                self.mainLayout.addWidget(QLabel(html_utils.paragraph(groupName)))
                groupBox = QGroupBox()
                self.mainLayout.addWidget(groupBox)
                groupLayout = QVBoxLayout()
                groupBox.setLayout(groupLayout)
                createdGroupLayouts[groupName] = groupLayout
            else:
                groupLayout.addSpacing(10)
            groupLayout.addWidget(QLabel(html_utils.paragraph(treeName)))
            treeWidget = widgets_item_views.TreeWidget(multiSelection=True)
            treeWidget.setHeaderHidden(True)
            for topLevel, children in tree.items():
                topLevelItem = widgets_item_views.TreeWidgetItem(treeWidget)
                topLevelItem.setText(0, topLevel)
                treeWidget.addTopLevelItem(topLevelItem)
                childrenItems = [widgets_item_views.TreeWidgetItem([c]) for c in children]
                topLevelItem.addChildren(childrenItems)
                topLevelItem.setExpanded(True)
            self.treeWidgets[treeName] = treeWidget
            groupLayout.addWidget(treeWidget)
            self.mainLayout.addSpacing(20)
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        self.mainLayout.addSpacing(10)
        self.mainLayout.addLayout(buttonsLayout)
    
    def ok_cb(self):
        self.cancel = False
        self.selectedItems = {}
        for treeName, treeWidget in self.treeWidgets.items():
            for i in range(treeWidget.topLevelItemCount()):
                topLevelItem = treeWidget.topLevelItem(i)
                for j in range(topLevelItem.childCount()):
                    childItem = topLevelItem.child(j)
                    if not childItem.isSelected():
                        continue
                    if treeName not in self.selectedItems:
                        self.selectedItems[treeName] = [childItem.text(0)]
                    else:
                        self.selectedItems[treeName].append(childItem.text(0))
        self.close()


class QLineEditDialog(QDialog):
    def __init__(
            self, title='Entry messagebox', msg='Entry value',
            defaultTxt='', parent=None, allowedValues=None,
            warnLastFrame=False, isInteger=False, isFloat=False,
            stretchEntry=True, allowEmpty=True, allowedTextEntries=None, 
            allowText=False, lastVisitedFrame=None, allowList=False
        ):
        QDialog.__init__(self, parent)

        self.loop = None
        self.cancel = True
        self.assignNewID = False
        self.allowedValues = allowedValues
        self.warnLastFrame = warnLastFrame
        self.isFloat = isFloat
        self.allowEmpty = allowEmpty
        self.isInteger = isInteger
        self.allowedTextEntries = allowedTextEntries
        self.allowText = allowText
        self.lastVisitedFrame = lastVisitedFrame
        if allowedValues and warnLastFrame:
            self.maxValue = max(allowedValues)

        self.setWindowTitle(title)

        # Layouts
        mainLayout = QVBoxLayout()
        LineEditLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        # Widgets
        if not msg.startswith('<p'):
            msg = html_utils.paragraph(msg, center=True)
        msg = QLabel(msg)
        msg.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        msg.setOpenExternalLinks(True)
        msg.setStyleSheet("padding:0px 0px 3px 0px;")
        
        if isFloat:
            self._type = float
        elif isInteger:
            self._type = int
        else:
            self._type = str
        
        self.allowList = allowList

        if isFloat and not allowList:
            entryWidget = QDoubleSpinBox()
            if allowedValues is not None:
                _min, _max = min(allowedValues), max(allowedValues)
                entryWidget.setMinimum(_min)
                entryWidget.setMaximum(_max)
            else:
                entryWidget.setMaximum(2147483647)
            if defaultTxt:
                entryWidget.setValue(float(defaultTxt))

        elif isInteger and not allowList:
            entryWidget = QSpinBox()
            if allowedValues is not None:
                _min, _max = min(allowedValues), max(allowedValues)
                entryWidget.setMinimum(_min)
                entryWidget.setMaximum(_max)
            else:
                entryWidget.setMaximum(2147483647)
            if defaultTxt:
                entryWidget.setValue(int(defaultTxt))
        else:
            entryWidget = QLineEdit()
            entryWidget.setText(defaultTxt)
            if not self.allowText:
                entryWidget.textChanged[str].connect(self.onTextChanged)
        entryWidget.setFont(fonts.font)
        entryWidget.setAlignment(Qt.AlignCenter)

        self.entryWidget = entryWidget

        if allowedValues is not None:
            notValidLabel = QLabel()
            notValidLabel.setStyleSheet('color: red')
            notValidLabel.setFont(fonts.font)
            notValidLabel.setAlignment(Qt.AlignCenter)
            self.notValidLabel = notValidLabel

        okButton = widgets_push_buttons.okPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')

        # Events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        # Contents margins
        buttonsLayout.setContentsMargins(0,10,0,0)

        # Add widgets to layouts
        LineEditLayout.addWidget(msg, alignment=Qt.AlignCenter)
        if stretchEntry:
            LineEditLayout.addWidget(entryWidget)
        else:
            entryLayout = QHBoxLayout()
            entryLayout.addStretch(1)
            entryLayout.addWidget(entryWidget)
            entryLayout.addStretch(1)
            entryLayout.setStretch(1,1)
            LineEditLayout.addLayout(entryLayout)
        if allowedValues is not None:
            LineEditLayout.addWidget(notValidLabel, alignment=Qt.AlignCenter)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)
        
        self.okButton = okButton
        self.buttonsLayout = buttonsLayout

        # Add layouts
        mainLayout.addLayout(LineEditLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        # self.setModal(True)
    
    def value(self):
        if self._type == str:
            return self.entryWidget.text()
        
        if (self.isFloat or self.isInteger) and not self.allowList:
            val = self.entryWidget.value()
        elif not self.allowList:
            val = int(self.entryWidget.text())
        elif self.allowList:
            caster = int if self.isInteger else float
            text = self.entryWidget.text()
            m = re.findall(POSITIVE_FLOAT_REGEX, text)
            val = [caster(val) for val in m]
        return val

    def onTextChanged(self, text):
        # Get inserted char
        idx = self.entryWidget.cursorPosition()
        if idx == 0:
            return

        newChar = text[idx-1]
        if self.allowList and (newChar==',' or newChar == ' '):
            return

        if self._type == str:
            self.entryWidget.setText(text)
            return
        
        # Allow only integers
        try:
            val = int(newChar)
            if val > np.iinfo(np.uint32).max:
                self.entryWidget.setText(str(np.iinfo(np.uint32).max))
        except Exception as e:
            text = text.replace(newChar, '')
            self.entryWidget.setText(text)
            return
        
        if self.allowedValues is not None:
            currentVal = self.value()
            if self.allowList:
                currentVal = currentVal[-1]
            if currentVal not in self.allowedValues:
                self.notValidLabel.setText(f'{currentVal} not existing!')
            else:
                self.notValidLabel.setText('')

    def warnValLessLastFrame(self, val):
        msg = widgets_message_box.myMessageBox()
        warn_txt = html_utils.paragraph(f"""
            WARNING: saving until a frame number below the last visited
            frame ({self.lastVisitedFrame}) will result in <b>LOSS of information</b>
            about any <b>edit or annotation</b> you did <b>on frames
            {val+1}-{self.lastVisitedFrame}.</b><br><br>
            Are you sure you want to proceed?
        """)
        msg.warning(
           self, 'WARNING: Potential loss of information', warn_txt, 
           buttonsTexts=('Cancel', 'Yes, I am sure.')
        )
        return msg.cancel

    def warnValMoreLastVisitedFrame(self, val):
        msg = widgets_message_box.myMessageBox()
        warn_txt = html_utils.paragraph(f"""
            The <b>last visited/validated frame is {self.lastVisitedFrame}</b>
            .<br><br>
            Are you sure you want to save until frame n. {val}?<br>
        """)
        msg.warning(
           self, 'Saving past last visited frame', warn_txt, 
           buttonsTexts=('Cancel', 'Yes, I am sure.')
        )
        return msg.cancel
    
    def ok_cb(self, event):
        if not self.allowEmpty and not self.entryWidget.text():
            msg = widgets_message_box.myMessageBox(showCentered=False, wrapText=False)
            msg.critical(
                self, 'Empty text', 
                html_utils.paragraph('Text entry field <b>cannot be empty</b>')
            )
            return
        if self.allowedTextEntries is not None:
            if self.entryWidget.text() not in self.allowedTextEntries:
                msg = widgets_message_box.myMessageBox(showCentered=False, wrapText=False)
                txt = html_utils.paragraph(
                    f'"{self.entryWidget.text()}" is not a valid entry.<br><br>'
                    'Valid entries are:<br>'
                    f'{html_utils.to_list(self.allowedTextEntries)}'
                )
                msg.critical(self, 'Not a valid entry', txt)
                return
        
        if self.allowedValues:
            if self.notValidLabel.text():
                return

        val = self.value()
        
        if self.warnLastFrame and self.lastVisitedFrame is not None:
            if val < self.lastVisitedFrame:
                cancel = self.warnValLessLastFrame(val)
                if cancel:
                    return

        if self.lastVisitedFrame is not None:
            if val > self.lastVisitedFrame:
                cancel = self.warnValMoreLastVisitedFrame(val)
                if cancel:
                    return

        self.cancel = False
        try:
            self.EntryID = int(val)
        except Exception as err:
            self.EntryID = val
            
        self.enteredValue = val
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        super().show()
        QTimer.singleShot(10, self.entryWidget.selectAll)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()


class NumericEntryDialog(dialogs_base.QBaseDialog):
    def __init__(
            self, title='Entry a value', currentValue=0,
            instructions='Entry value', parent=None, 
            maxValue=None, minValue=None, stretch=False
        ):
        super().__init__(parent=parent)
        self.setWindowTitle(title)
        self.cancel = True
        mainLayout = QVBoxLayout()
        entryLayout = QHBoxLayout()
        cancelOkLayout = widgets_push_buttons.CancelOkButtonsLayout()
        cancelOkLayout.okButton.clicked.connect(self.ok_cb)
        cancelOkLayout.cancelButton.clicked.connect(self.close)
        
        instructionsLabel = QLabel(html_utils.paragraph(instructions))
        mainLayout.addWidget(instructionsLabel)
        
        if type(currentValue) == int:
            self.entryWidget = widgets_numeric_inputs.SpinBox()
            self.entryWidget.setValue(currentValue)
            self.valueGetter = 'value'
            if maxValue is not None:
                self.entryWidget.setMaximum(maxValue)
            if minValue is not None:
                self.entryWidget.setMinimum(minValue)
        
        if stretch:
            entryLayout.addWidget(self.entryWidget)
        else:
            entryLayout.addStretch(1)
            entryLayout.addWidget(self.entryWidget)
            entryLayout.addStretch(1)
            entryLayout.setStretch(0, 1)
            entryLayout.setStretch(1, 1)
            entryLayout.setStretch(2, 1)
        
        mainLayout.addLayout(entryLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(cancelOkLayout)
        
        self.setLayout(mainLayout)
    
    def ok_cb(self):
        self.cancel = False
        self.value = getattr(self.entryWidget, self.valueGetter)()
        self.close()


class QInput(dialogs_base.QBaseDialog):
    def __init__(self, parent=None, title='Input'):
        self.cancel = True
        self.allowEmpty = True

        super().__init__(parent)

        self.setWindowTitle(title)

        self.mainLayout = QVBoxLayout()

        self.infoLabel = QLabel()
        self.mainLayout.addWidget(self.infoLabel)

        promptLayout = QHBoxLayout()
        self.promptLabel = QLabel()
        promptLayout.addWidget(self.promptLabel)
        self.lineEdit = QLineEdit()
        promptLayout.addWidget(self.lineEdit)

        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        self.mainLayout.addLayout(promptLayout)
        self.mainLayout.addSpacing(20)
        self.mainLayout.addLayout(buttonsLayout)

        self.buttonsLayout = buttonsLayout

        self.setFont(fonts.font)
        self.setLayout(self.mainLayout)
    
    def askText(self, prompt, infoText='', allowEmpty=False):
        self.allowEmpty = allowEmpty
        if infoText:
            infoText = f'{infoText}<br>'
            self.infoLabel.setText(html_utils.paragraph(infoText))
        self.promptLabel.setText(prompt)
        self.exec_(resizeWidthFactor=1.5)

    def ok_cb(self):
        self.answer = self.lineEdit.text()
        if not self.allowEmpty and not self.answer:
            msg = widgets_message_box.myMessageBox(showCentered=False)
            msg.critical(self, 'Empty', 'Entry cannot be empty.')
            return
        self.cancel = False
        self.close()


class QTreeDialog(dialogs_base.QBaseDialog):
    def __init__(
            self, 
            items: List[Tuple[str]], 
            headerLabels: List[str]=None,
            parent=None, 
            infoText='Select item',
            title='Select item',
            path_to_browse=None,
            additional_buttons=None,
            cancelButtonText=None,
        ):
        self.cancel = True
        super().__init__(parent)
        
        self.setWindowTitle(title)
        
        mainLayout = QVBoxLayout()
        
        infoLabel = QLabel(html_utils.paragraph(infoText))
        
        self.treeWidget = widgets_item_views.TreeWidget()
        if headerLabels is not None:
            self.treeWidget.setHeaderLabels(headerLabels)
        else:
            self.treeWidget.setHeaderHidden(True)
        
        for row, texts in enumerate(items):
            item = widgets_item_views.TreeWidgetItem(self.treeWidget)
            for i, text in enumerate(texts):
                item.setText(i, text)
            self.treeWidget.addTopLevelItem(item)
        
        self.treeWidget.resizeColumnToContents(0)
        self.treeWidget.resizeColumnToContents(1)
        
        # self.treeWidget.header().setStretchLastSection(False)
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()
        if cancelButtonText is not None:
            buttonsLayout.cancelButton.setText(cancelButtonText)
        buttonsLayout.okButton.setEnabled(
            self.treeWidget.currentItem() is not None
        )
        self.treeWidget.currentItemChanged.connect(
            lambda current, previous: buttonsLayout.okButton.setEnabled(
                current is not None
            )
        )
        
        if path_to_browse is not None:
            browseButton = widgets_push_buttons.showInFileManagerButton(
                setDefaultText=True
            )
            browseButton.setPathToBrowse(path_to_browse)
            buttonsLayout.insertWidget(3, browseButton)

        if additional_buttons is not None:
            for btn in additional_buttons:
                buttonsLayout.insertWidget(3, btn)

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addWidget(infoLabel)
        mainLayout.addWidget(self.treeWidget)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def show(self, block=False):
        w = self.sizeHint().width()
        h = self.sizeHint().height()
        self.resize(int(w*1.3), h)
        super().show(block=block)
    
    def ok_cb(self):
        self.clickedButton = self.sender()
        self.cancel = False
        self.selectedItem = self.treeWidget.currentItem()
        self.selectedText = self.selectedItem.text(0)
        self.close()


class SearchableListboxDialog(dialogs_base.QBaseDialog):
    """Search and select one or more items from a recency-ordered list.

    Search ignores punctuation and separators, so paths can be searched with
    spaces in place of slashes or underscores. Results are ranked by match
    quality when the match-sorting checkbox is checked, with the original
    item order used to break ties. Otherwise, results stay in their original
    order.

    The ``fuzzyThreshold`` argument sets the minimum similarity ratio for
    fuzzy matches (0.0-1.0, default 0.5). Existing filesystem paths also show
    their modification date in a separate column.

    Args:
        items: Display strings in their initial, usually most-recent-first,
            order. Selected values are emitted unchanged by :attr:`sigOk`.
        title: Dialog window title.
        searchLineEditText: Initial prompt shown by the search field.
        parent: Optional parent widget.
        fuzzyThreshold: Initial minimum ratio accepted for fuzzy matches,
            from 0.0 (least strict) to 1.0 (most strict).
        multiSelection: Allow selecting multiple rows and emit their values
            as a list. When false, preserve the single-string result.
    """

    sigOk = Signal(object)

    def __init__(
            self, 
            items: list[str],
            dates: list[str] | None = None,
            title='Search items',
            searchLineEditText='Search...',
            parent=None,
            fuzzyThreshold: float = 0.5,
            multiSelection: bool = False,
        ):
        super().__init__(parent=parent)

        self.fuzzyThreshold = float(fuzzyThreshold)
        if not math.isfinite(self.fuzzyThreshold) or not (
                0 <= self.fuzzyThreshold <= 1
            ):
            raise ValueError('fuzzyThreshold must be between 0 and 1.')

        self.cancel = True
        self.allItems = items
        self.dates = dates
        self.multiSelection = multiSelection

        self.searchModel = QStandardItemModel()
        searchRoles = (widgets_search_widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE,)
        self.searchProxyModel = _RecentItemsSearchCompleter(
            searchRoles=searchRoles,
            tooltipRole=None,
        )
        self.searchProxyModel.FUZZY_THRESHOLD = self.fuzzyThreshold
        self.searchProxyModel.setSourceModel(self.searchModel)
        self.searchProxyModel.setDynamicSortFilter(True)
        self.searchProxyModel.sort(0, Qt.AscendingOrder)
        self.searchProxyModel.setAlwaysRecency(True)

        # add all items to the search model
        for item_text in self.allItems:
            model_item = StandardItem(item_text)
            model_item.setData(
                self._normalizeSearchText(item_text),
                widgets_search_widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE,
            )
            last_edited_item = QStandardItem('')
            last_edited, last_edited_tooltip = self._lastEditedDate(
                item_text
            )
            last_edited_item.setText(last_edited)
            if last_edited_tooltip:
                last_edited_item.setToolTip(last_edited_tooltip)
            self.searchModel.appendRow([model_item, last_edited_item])

        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()

        self.searchLineEdit = SearchLineEdit(
            text=searchLineEditText
        )

        self.searchLineEdit.textEdited.connect(self.searchItems)

        self.sortByGoodnessCheckBox = QCheckBox(
            'Sort by goodness of match'
        )
        self.sortByGoodnessCheckBox.toggled.connect(
            self._onSortByGoodnessChanged
        )

        self.sortByGoodnessCheckBox.setToolTip(
            'When enabled, better matches appear '
            'first, with recency used to break ties.\n'
            'When disabled, matching items stay in the original recent-first order.'
        )

        self.listWidget = QTreeWidget()
        self.listWidget.setHeaderLabels(['Item', 'Last edited'])
        self.listWidget.setRootIsDecorated(False)
        self.listWidget.setItemsExpandable(False)
        self.listWidget.setSortingEnabled(False)
        self.listWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
            if self.multiSelection
            else QAbstractItemView.SelectionMode.SingleSelection
        )
        self.listWidget.header().setStretchLastSection(False)
        self.listWidget.header().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.listWidget.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.listWidget.setColumnWidth(1, 135)
        self.listWidget.headerItem().setToolTip(
            1, 'Filesystem modification date and time.'
        )
        self.searchItems(self.searchLineEdit.text())

        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()

        is_any_item_path = any([os.path.exists(item) for item in items])
        if is_any_item_path:
            showInFileManagerButton = widgets_push_buttons.showInFileManagerButton(
                filesystem.get_open_filemaneger_os_string()
            )
            buttonsLayout.insertWidget(3, showInFileManagerButton)
            showInFileManagerButton.clicked.connect(self.showInFileManager)

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        if not self.multiSelection:
            self.listWidget.itemDoubleClicked.connect(
                lambda *_: self.ok_cb()
            )

        mainLayout.addWidget(self.searchLineEdit)
        mainLayout.addWidget(self.sortByGoodnessCheckBox)
        mainLayout.addWidget(self.listWidget)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        self.setFont(fonts.font)
    
    def showInFileManager(self):
        selected_items = self.listWidget.selectedItems()
        selected_texts = [
            item.data(0, Qt.UserRole) or item.text(0)
            for item in selected_items
        ]
        valid_paths = [
            text for text in selected_texts if os.path.exists(text)
        ]
        if not valid_paths:
            print('WARNING: None of the selected items is a valid path')
            msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
            txt = html_utils.paragraph(
                'None of the selected items is a valid path :(.'
            )
            msg.warning(self, 'Not a valid path', txt)
            return
        
        first_valid_path = valid_paths[0]
        filesystem.showInExplorer(first_valid_path)

    def _lastEditedDate(self, item_text):
        """Return a display date and tooltip for an item's modification time.

        Nonexistent paths have no date; filesystem errors are reported in the
        returned tooltip instead of being silently discarded.
        """
        if self.dates is not None:
            index = self.allItems.index(item_text) if item_text in self.allItems else None
            if index is not None:
                date = pd.to_datetime(self.dates[index], errors='coerce')
                # format the date for display
                if not pd.isna(date):
                    date = date.strftime('%d.%m.%Y %H:%M')
                    return date, ''
        
        if not os.path.exists(item_text):
            return '', ''

        try:
            modified_time = os.path.getmtime(item_text)
        except OSError as err:
            return (
                'Unavailable',
                qutils.wrap_tooltip(
                    f'Could not read filesystem modification time: {err}'
                ),
            )

        return (
            datetime.datetime.fromtimestamp(
                modified_time
            ).strftime('%d.%m.%Y %H:%M'),
            '',
        )

    def _populateListWidget(self):
        """Rebuild visible rows in the order supplied by the search proxy."""
        self.listWidget.clear()
        for row in range(self.searchProxyModel.rowCount()):
            proxy_index = self.searchProxyModel.index(row, 0)
            source_index = self.searchProxyModel.mapToSource(proxy_index)
            source_item = self.searchModel.item(source_index.row(), 0)
            item_text = source_item.text()
            last_edited = self.searchModel.item(
                source_index.row(), 1
            ).text()
            if os.path.exists(item_text):
                tree_item = PathTreeWidgetItem(item_text, [last_edited])
            else:
                tree_item = QTreeWidgetItem([item_text, last_edited])
            tree_item.setToolTip(
                1,
                self.searchModel.item(
                    source_index.row(), 1
                ).toolTip(),
            )
            self.listWidget.addTopLevelItem(tree_item)

    def _onSortByGoodnessChanged(self, sortByGoodness):
        """Switch between match-quality and original-order result ranking."""
        self.searchProxyModel.setAlwaysRecency(not sortByGoodness)
        self.searchItems(self.searchLineEdit.text())
    
    def searchItems(self, text):
        """Filter, rank, and highlight results for the current query."""
        from cellacdc._palettes import get_color_scheme
        scheme = get_color_scheme()
        if scheme == 'dark':
            highlight_color = '#4DA3FF'
        else:
            highlight_color = '#0067C5'
        query = text.strip()
        # Match against separator-free text so spaces, underscores, and path
        # separators are interchangeable without changing displayed names.
        normalized_query = self._normalizeSearchText(query)
        self.searchProxyModel.setFilterText(normalized_query)
        self._populateListWidget()
        for row in range(self.listWidget.topLevelItemCount()):
            tree_item = self.listWidget.topLevelItem(row)
            item = tree_item.text(0)
            # The proxy provides the ranking; highlight ranges are mapped back
            # onto the original display string.
            ranges = widgets_search_widgets._searchHighlightRanges(
                normalized_query,
                item,
                self.searchProxyModel.FUZZY_THRESHOLD,
            )

            if ranges:
                highlighted_text = widgets_search_widgets._highlightSearchText(
                    item, ranges, highlight_color
                )
                label = QLabel()
                label.setTextFormat(Qt.RichText)
                label.setAttribute(Qt.WA_TransparentForMouseEvents)
                label.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Preferred
                )
                label.setStyleSheet('background: transparent;')
                label.setToolTip(item)
                label.setText(highlighted_text)
                tree_item.setData(0, Qt.UserRole, item)
                tree_item.setText(0, '')
                self.listWidget.setItemWidget(tree_item, 0, label)

    @staticmethod
    def _normalizeSearchText(text):
        """Case-fold text and remove non-alphanumeric search separators."""
        return widgets_search_widgets._normalizeSearchText(text)

    def warnSelectionEmpty(self):
        """Explain that a row must be selected before confirming."""
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
        txt = html_utils.paragraph(
            'You need to <b>select at least one item</b> before pressing "Ok".<br><br>'
            'Otherwise, you can cancel any time with the Cancel button.<br><br>'
            'Thank you for your patience!'
        )
        msg.warning(self, 'Selection cannot be empty', txt)

    def ok_cb(self, *args, **kwargs):
        """Emit selected original strings and close the dialog.

        Single-selection mode emits a string for backwards compatibility;
        multi-selection mode emits a list of strings in displayed order.
        """
        selected_items = self.listWidget.selectedItems()
        if not selected_items:
            self.warnSelectionEmpty()
            return

        self.cancel = False
        selected_texts = [
            item.data(0, Qt.UserRole) or item.text(0)
            for item in selected_items
        ]
        if self.multiSelection:
            self.selectedItemTexts = selected_texts
            result = selected_texts
        else:
            self.selectedItemText = selected_texts[0]
            result = self.selectedItemText
        self.close()
        self.sigOk.emit(result)
    
    def show(self, block=False):
        super().show(block=False)
        desiredWidthFractionOfScreen = 1/2
        screenGeometry = self.screen().geometry()
        x0, y0 = screenGeometry.left(), screenGeometry.top()
        screenWidth = screenGeometry.width()
        screenHeight = screenGeometry.height()
        screenXCenter = x0 + screenWidth/2
        screenYCenter = y0 + screenHeight/2
        windowLeft = round(
            screenXCenter 
            - screenWidth*desiredWidthFractionOfScreen/2
        )
        windowWidth = round(screenWidth*desiredWidthFractionOfScreen)
        windowHeight = round(screenHeight/2)
        windowTop = y0
        self.resize(windowWidth, windowHeight)
        self.move(windowLeft, windowTop)
        self.raise_()
        self.activateWindow()
        super().show(block=block)
