"""myMessageBox, the standard Cell-ACDC message box, with its content
helpers (LabelsWidget, LatexLabel, CopiableCommandWidget,
addWidgetToScrollArea) and the Visual C++ prerequisite warning
(warnVisualCppRequired, view_visualcpp_screenshot, used by the YeastMate
model).

Tests patch myMessageBox in this module, so callers look it up here at
call time (widgets_message_box.myMessageBox), never through a
from-import. Button texts get their icon from getPushButton in
push_buttons.py.
"""
from typing import Sequence
import re
import numpy as np
import textwrap
from functools import partial
import html
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from qtpy.QtCore import QTimer, Qt, QEventLoop
from qtpy.QtGui import QIcon, QPixmap, QImage
from qtpy.QtWidgets import (
    QTextEdit, QLabel, QHBoxLayout, QCheckBox, QApplication, QWidget,
    QVBoxLayout, QGridLayout, QScrollArea, QPushButton, QGroupBox, QStyle,
    QDialog, QSpacerItem, QFrame
)
from cellacdc import _window_stacking
from cellacdc import myutils, is_mac, html_utils
from cellacdc import _base_widgets
from . import style as widgets_style
from . import push_buttons as widgets_push_buttons
from . import basic_widgets as widgets_basic_widgets


class myMessageBox(_base_widgets.QBaseDialog):
    def __init__(
            self, parent=None, showCentered=True, wrapText=True,
            scrollableText=False, enlargeWidthFactor=0,
            resizeButtons=True, allowClose=True
        ):
        super().__init__(parent)

        self.wrapText = wrapText
        self.enlargeWidthFactor = enlargeWidthFactor
        self.resizeButtons = resizeButtons

        self.cancel = True
        self.cancelButton = None
        self.okButton = None
        self.clickedButton = None
        self.alreadyShown = False
        self.allowClose = allowClose

        self.showCentered = showCentered

        self.scrollableText = scrollableText

        self._layout = QGridLayout()
        self.commandsLayout = None
        self._layout.setHorizontalSpacing(20)
        self.buttonsLayout = QHBoxLayout()
        self.buttonsLayout.setSpacing(2)
        self.buttons = []
        self.widgets = []
        self.layouts = []
        self.labels = []
        self.labelsWidgets = []
        self._pixmapLabels = []
        self.detailsTextWidget = None
        self.showInFileManagButton = None
        self.visibleDetails = False
        self.doNotShowAgainCheckbox = None

        self.currentRow = 0
        self.textWidget = None
        self._w = None

        self.textLayout = QVBoxLayout()
        
        self._layout.setColumnStretch(1, 1)
        self.setLayout(self._layout)
        
        self.setFont(widgets_style.font)

    def mousePressEvent(self, event):
        for label in self.labels:
            label.setTextInteractionFlags(
                Qt.TextBrowserInteraction | Qt.TextSelectableByKeyboard
            )

    def setIcon(self, iconName='SP_MessageBoxInformation'):
        label = QLabel(self)

        standardIcon = getattr(QStyle, iconName)
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        label.setPixmap(pixmap)

        self._layout.addWidget(label, 0, 0, alignment=Qt.AlignTop)

    def addImage(self, image_path):
        pixmap = QPixmap(image_path)
        label = QLabel()
        label.setPixmap(pixmap)
        self._layout.addWidget(label, self.currentRow, 1)
        self.currentRow += 1
    
    def addShowInFileManagerButton(self, path, txt=None):
        if txt is None:
            txt = 'Reveal in Finder...' if is_mac else 'Show in Explorer...'
        self.showInFileManagButton = widgets_push_buttons.showInFileManagerButton(txt)
        self.buttonsLayout.addWidget(self.showInFileManagButton)
        func = partial(myutils.showInExplorer, path)
        self.showInFileManagButton.clicked.connect(func)
    
    def addBrowseUrlButton(self, url, button_text=''):
        self.openUrlButton = widgets_push_buttons.OpenUrlButton(url, button_text)
        self.buttonsLayout.addWidget(self.openUrlButton)

    def addCancelButton(self, button=None, connect=False):
        if button is None:
            self.cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        else:
            self.cancelButton = button
            self.cancelButton.setIcon(QIcon(':cancelButton.svg'))

        # NOTE: final position of the cancel button is set in
        # `_arrangeButtonsLayout` when the dialog is shown
        self.buttonsLayout.insertWidget(0, self.cancelButton)
        if connect:
            self.cancelButton.clicked.connect(self.buttonCallBack)

    def splitLatexBlocks(self, text):
        texts = re.split(r"(<latex.*?>.+?)</latex>", text)
        return texts        
    
    def splitCopiableBlocks(self, texts: Sequence[str] | str):
        if isinstance(texts, str):
            texts = (texts,)
        
        texts_out = []
        for text in texts:
            texts_out.extend(re.split(r"(<copiable>.+?)</copiable>", text))
        return texts_out  

    def addText(self, text):
        texts = self.splitLatexBlocks(text)
        texts = self.splitCopiableBlocks(texts)
        
        labelsWidget = LabelsWidget(texts, wrapText=self.wrapText)
        self.labelsWidgets.append(labelsWidget)
        self.labels.extend(labelsWidget.labels)
        if self.scrollableText:
            textWidget = QScrollArea()
            textWidget.setFrameStyle(QFrame.Shape.NoFrame)
            textWidget.setWidget(labelsWidget)
        else:
            textWidget = labelsWidget

        self.textLayout.addWidget(textWidget)
        
        if self.textWidget is None:
            self.textWidget = QWidget()
            self.textWidget.setLayout(self.textLayout)
            self._layout.addWidget(self.textWidget, self.currentRow, 1)
            self.textRow = self.currentRow
            self.currentRow += 1
            
        return labelsWidget
    
    def addCopiableCommand(self, command):
        copiableCommandWidget = CopiableCommandWidget(command)
        screenWidth = self.screen().size().width()
        maxWidth = int(0.75*screenWidth)
        sizeHint = copiableCommandWidget.sizeHint()
        width = sizeHint.width()
        if width > maxWidth:
            copiableCommandWidget = addWidgetToScrollArea(
                copiableCommandWidget, 
                resizeMinHeightNoVerticalScrollbar=True
            )
        self._layout.addWidget(copiableCommandWidget, self.currentRow, 1)
        self.currentRow += 1
    
    def copyToClipboard(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        plain_text = html.unescape(self.sender()._command).replace("\xa0", " ")
        cb.setText(plain_text, mode=cb.Clipboard)
        print('Command copied!')

    def addButton(self, buttonText):
        if not isinstance(buttonText, str):
            # Passing button directly
            button = buttonText
            self.buttonsLayout.addWidget(button)
            button.clicked.connect(self.buttonCallBack)
            self.buttons.append(button)
            return button
        
        button, isCancelButton = widgets_push_buttons.getPushButton(buttonText, qparent=self)
        if not isCancelButton:
            self.buttonsLayout.addWidget(button)

        button.clicked.connect(self.buttonCallBack)
        self.buttons.append(button)
        return button

    def addDoNotShowAgainCheckbox(self, text='Do not show again'):
        self.doNotShowAgainCheckbox = QCheckBox(text)

    def addWidget(self, widget):
        self._layout.addWidget(widget, self.currentRow, 1)
        self.widgets.append(widget)
        self.currentRow += 1

    def addLayout(self, layout):
        self._layout.addLayout(layout, self.currentRow, 1)
        self.layouts.append(layout)
        self.currentRow += 1

    def setWidth(self, w):
        self._w = w

    def show(self, block=False):
        self.endOfScrollableRow = self.currentRow
        
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        # spacer
        spacer = QSpacerItem(10, 10)
        self._layout.addItem(spacer, self.currentRow, 1)
        self._layout.setRowStretch(self.currentRow, 0)

        # buttons
        self.currentRow += 1

        self._arrangeButtonsLayout()

        # Do not show again checkbox
        if self.doNotShowAgainCheckbox is not None:
            self._layout.addWidget(
                self.doNotShowAgainCheckbox, self.currentRow, 1, 1, 2
            )
            self.currentRow += 1
        
        # spacer
        self._layout.addItem(QSpacerItem(10, 10), self.currentRow, 1)
        self.currentRow += 1
        
        # buttons
        self._layout.addLayout(
            self.buttonsLayout, self.currentRow, 0, 1, 2,
            alignment=Qt.AlignRight
        )

        # Details
        if self.detailsTextWidget is not None:
            # spacer
            self.currentRow += 1
            self._layout.addItem(QSpacerItem(20, 20), self.currentRow, 1)
            
            # detailsTextWidget
            self.currentRow += 1
            self._layout.addWidget(
                self.detailsTextWidget, self.currentRow, 0, 1, 2
            )

        # spacer
        self.currentRow += 1
        spacer = QSpacerItem(10, 10)
        self._layout.addItem(spacer, self.currentRow, 1)
        self._layout.setRowStretch(self.currentRow, 0)
        
        screenHeight = self.screen().size().height()
        dialogHeight = self.sizeHint().height()
        dialogWidth = self.sizeHint().width()
        screenWidth = self.screen().size().width()
        
        # Check if scrollbar is needed
        if dialogHeight > screenHeight and self.textWidget is not None:
            textScrollArea = widgets_basic_widgets.ScrollArea()
            textScrollArea.setWidget(self.textWidget)
            scrollAreaWidthNoSB = textScrollArea.minimumWidthNoScrollbar()
            scrollAreaWidth = textScrollArea.sizeHint().width()
            desiredDeltaWidth = scrollAreaWidthNoSB - scrollAreaWidth
            if desiredDeltaWidth > 0:
                desiredWidth = dialogWidth + desiredDeltaWidth
                if desiredWidth < screenWidth:
                    self._w = desiredWidth
                    
            self._layout.removeWidget(self.textWidget)
            self._layout.addWidget(textScrollArea, self.textRow, 1)
            
        super().show()
        QTimer.singleShot(5, self._resize)
        
        self.alreadyShown = True

        if block:
            self._block()

    def _arrangeButtonsLayout(self):
        """Arrange buttons as: stretch, cancel button (if any), 20 px
        spacing, show details button (if any), and then all the other
        buttons in the order they were added.

        The stretch on the left also prevents the show details button from
        expanding (its internal layout has a stretch that makes it
        horizontally expanding).
        """
        detailsButton = None
        if self.detailsTextWidget is not None:
            detailsButton = self.detailsButton

        otherItems = []
        while self.buttonsLayout.count() > 0:
            item = self.buttonsLayout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                if widget is self.cancelButton or widget is detailsButton:
                    continue
                otherItems.append(widget)
            elif item.layout() is not None:
                otherItems.append(item.layout())
            # Spacers and stretches are dropped and re-added below

        self.buttonsLayout.addStretch(1)

        if self.cancelButton is not None:
            self.buttonsLayout.addWidget(self.cancelButton)
            if detailsButton is not None or otherItems:
                self.buttonsLayout.addSpacing(20)

        if detailsButton is not None:
            self.buttonsLayout.addWidget(detailsButton)

        for item in otherItems:
            if isinstance(item, QWidget):
                self.buttonsLayout.addWidget(item)
            else:
                self.buttonsLayout.addLayout(item)

    def setDetailedText(self, text, visible=False, wrap=True):
        text = text.replace('\n', '<br>')
        self.detailsTextWidget = QTextEdit(text)
        self.detailsTextWidget.setReadOnly(True)
        if not wrap:
            self.detailsTextWidget.setLineWrapMode(QTextEdit.NoWrap)
        self.detailsButton = widgets_push_buttons.showDetailsButton()
        self.detailsButton.setCheckable(True)
        self.detailsButton.clicked.connect(self._showDetails)
        self.detailsTextWidget.hide()
        self.visibleDetails = visible

    def _showDetails(self, checked):
        if checked:
            self.origHeight = self.height()
            self.resize(self.width(), self.height()+300)
            self.detailsTextWidget.show()
        else:
            self.detailsTextWidget.hide()
            func = partial(self.resize, self.width(), self.origHeight)
            QTimer.singleShot(10, func)

    def _resize(self):
        if self.resizeButtons:
            widths = [button.width() for button in self.buttons]
            if widths:
                max_width = max(widths)
                for button in self.buttons:
                    if button == self.cancelButton:
                        continue
                    button.setMinimumWidth(max_width)

        heights = [button.height() for button in self.buttons]
        if heights:
            max_h = max(heights)
            for button in self.buttons:
                button.setMinimumHeight(max_h)
            if self.detailsTextWidget is not None:
                self.detailsButton.setMinimumHeight(max_h)
            if self.showInFileManagButton is not None:
                self.showInFileManagButton.setMinimumHeight(max_h)

        if self._w is not None and self.width() < self._w:
            self.resize(self._w, self.height())

        if self.width() < 350:
            self.resize(350, self.height())

        if self.enlargeWidthFactor > 0:
            self.resize(int(self.width()*self.enlargeWidthFactor), self.height())

        if self.visibleDetails:
            self.detailsButton.click()

        if self.showCentered:
            screen = self.screen()
            screenWidth = screen.size().width()
            screenHeight = screen.size().height()
            screenLeft = screen.geometry().x()
            screenTop = screen.geometry().y()
            w, h = self.width(), self.height()
            left = int(screenLeft + screenWidth/2 - w/2)
            top = int(screenTop + screenHeight/2 - h/2)
            if top < screenTop:
                top = screenTop
            if left < screenLeft:
                left = screenLeft
            self.move(left, top)

        self._h = self.height()

        if self.okButton is not None:
            self.okButton.setFocus()

        screen = self.screen()
        screenWidth = screen.size().width()
        screenHeight = screen.size().height()
        
        # Check Force wrap Text
        for labelWidget in self.labelsWidgets:
            textWidth = labelWidget.width()
            if not textWidth > screenWidth-10:
                continue
            factor = np.ceil(textWidth/screenWidth)
            lineLength = int(labelWidget.nCharsLongestLine/factor)
            for label in labelWidget.labels:
                if isinstance(label, CopiableCommandWidget):
                    continue
                
                text = label.text()
                chunks = textwrap.wrap(text, lineLength)
                text = '<br>'.join(chunks)
                label.setText(text)
            
            QTimer.singleShot(100, self._resizeWrappedText)
        
        if self.widgets:
            return

        if self.layouts:
            return

        # # Start resizing height every 1 ms
        # self.resizeCallsCount = 0
        # self.timer = QTimer()
        # from config import warningHandler
        # warningHandler.sigGeometryWarning.connect(self.timer.stop)
        # self.timer.timeout.connect(self._resizeHeight)
        # self.timer.start(1)

    def _resizeWrappedText(self):
        screenWidth = self.screen().size().width() - 5
        self.resize(screenWidth, self.height())
        screenLeft = self.screen().geometry().left()
        self.move(screenLeft, self.geometry().top())
    
    def _resizeHeight(self):
        try:
            # Resize until a "Unable to set geometry" warning is captured
            # by copnfig.warningHandler._resizeWarningHandler or #
            # height doesn't change anymore
            self.resize(self.width(), self.height()-1)
            if self.height() == self._h or self.resizeCallsCount > 100:
                self.timer.stop()
                return

            self.resizeCallsCount += 1
            self._h = self.height()
        except Exception as e:
            # traceback.format_exc()
            self.timer.stop()

    def _template(
            self, parent, title, message, detailsText=None,
            buttonsTexts=None, layouts=None, widgets=None,
            commands=None, path_to_browse=None, browse_button_text=None,
            url_to_open=None, open_url_button_text='Open url', 
            image_paths=None, wrapDetails=True,
            add_do_not_show_again_checkbox=False,
            details_expanded=True
        ):
        if parent is not None:
            self.setParent(parent)
        self.setWindowTitle(title)
        self.addText(message)
        if commands is not None:
            if isinstance(commands, str):
                commands = (commands,)
            for command in commands:
                self.addCopiableCommand(command)
        
        if image_paths is not None:
            if isinstance(image_paths, str):
                image_paths = (image_paths,)
            for image_path in image_paths:
                self.addImage(image_path)
        
        if layouts is not None:
            if myutils.is_iterable(layouts):
                for layout in layouts:
                    self.addLayout(layout)
            else:
                self.addLayout(layout)

        if widgets is not None:
            self._layout.addItem(QSpacerItem(20, 20), self.currentRow, 1)
            self.currentRow += 1
            if myutils.is_iterable(widgets):
                for widget in widgets:
                    self.addWidget(widget)
            else:
                self.addWidget(widgets)

        if path_to_browse is not None:
            self.addShowInFileManagerButton(
                path_to_browse, txt=browse_button_text
            )
        
        if url_to_open is not None:
            self.addBrowseUrlButton(
                url_to_open, button_text=open_url_button_text
            )
        
        buttons = []
        if buttonsTexts is None:
            okButton = self.addButton('  Ok  ')
            buttons.append(okButton)
        elif isinstance(buttonsTexts, str):
            button = self.addButton(buttonsTexts)
            buttons.append(button)
        else:
            for buttonText in buttonsTexts:
                button = self.addButton(buttonText)
                buttons.append(button)
        
        if detailsText is not None:
            self.setDetailedText(
                detailsText, visible=details_expanded, wrap=wrapDetails
            )
        
        if add_do_not_show_again_checkbox:
            self.addDoNotShowAgainCheckbox()
        
        return buttons

    def critical(self, *args, showDialog=True, **kwargs):
        self.setIcon(iconName='SP_MessageBoxCritical')
        buttons = self._template(*args, **kwargs)
        if showDialog:
            self.exec_()
        return buttons

    def information(self, *args, showDialog=True, **kwargs):
        self.setIcon(iconName='SP_MessageBoxInformation')
        buttons = self._template(*args, **kwargs)
        if showDialog:
            self.exec_()
        return buttons

    def warning(self, *args, showDialog=True, **kwargs):
        self.setIcon(iconName='SP_MessageBoxWarning')
        buttons = self._template(*args, **kwargs)
        if showDialog:
            self.exec_()
        return buttons

    def question(self, *args, showDialog=True, **kwargs):
        self.setIcon(iconName='SP_MessageBoxQuestion')
        buttons = self._template(*args, **kwargs)
        if showDialog:
            self.exec_()
        return buttons

    def _block(self):
        self.loop = QEventLoop()
        self.loop.exec_()

    def exec_(self):
        self.show(block=True)
    
    def clickButtonFromText(self, buttonText):
        for button in self.buttons:
            if button.text() == buttonText:
                button.click()
                return

    def buttonCallBack(self, checked=True):
        self.clickedButton = self.sender()
        if self.clickedButton != self.cancelButton:
            self.cancel = False
        self.allowClose = True
        self.close()

    def closeEvent(self, event):
        if not self.allowClose:
            event.ignore()
            return
        super().closeEvent(event)


class view_visualcpp_screenshot(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()

        self.setWindowTitle('Visual Studio Builld Tools installation')

        pixmap = QPixmap(':visualcpp.png')
        label = QLabel()
        label.setPixmap(pixmap)

        layout.addWidget(label)
        self.setLayout(layout)


class CopiableCommandWidget(QGroupBox):
    def __init__(self, command='', parent=None, font_size='13px'):
        super().__init__(parent)
        
        layout = QHBoxLayout()
        
        label = QLabel(self)
        self.label = label
        self._font_size = font_size
        self.setCommand(command, font_size=font_size)
        label.setTextInteractionFlags(
            Qt.TextBrowserInteraction | Qt.TextSelectableByKeyboard
        )
        layout.addWidget(label)
        layout.addWidget(widgets_basic_widgets.QVLine(shadow='Plain', color='#4d4d4d'))
        copyButton = widgets_push_buttons.copyPushButton('Copy', flat=True, hoverable=True)
        copyButton.clicked.connect(self.copyToClipboard)
        layout.addWidget(copyButton)
        layout.addStretch(1)
        
        self.setLayout(layout)        
    
    def setWordWrap(self, wordWrap):
        self.label.setWordWrap(wordWrap)
    
    def copyToClipboard(self):        
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        plain_command = html.unescape(self._command).replace("\xa0", " ")
        cb.setText(plain_command, mode=cb.Clipboard)
        print('Command copied!')
    
    def setCommand(self, command, font_size=None, escape_html=True):
        if font_size is None:
            font_size = self._font_size
        
        self._command = command
        
        command = html.escape(command)
        txt = html_utils.paragraph(
            f'<code>{command}</code>', font_size=font_size
        )
        self.label.setText(txt)
    
    def command(self):
        return self._command
    
    def text(self):
        return self.label.text()
    
    def setTextInteractionFlags(self, flags):
        self.label.setTextInteractionFlags(flags)


class LatexLabel(QLabel):
    def __init__(self, latexText, parent=None):
        super().__init__(parent)
        
        latexText = latexText.replace('<latex>', '$')
        if not latexText.startswith('$'):
            latexText = f'${latexText}'
        
        if not latexText.endswith('$'):
            latexText = f'{latexText}$'
        
        latexText = latexText.replace('<br>', '\n')
        
        pixmap = self.mathTex_to_QPixmap(latexText)
        self.setPixmap(pixmap)
    
    def mathTex_to_QPixmap(self, mathTex):
        #---- set up a mpl figure instance ----

        fig = matplotlib.figure.Figure()
        fig.patch.set_facecolor('none')
        fig.set_canvas(FigureCanvasAgg(fig))
        renderer = fig.canvas.get_renderer()

        #---- plot the mathTex expression ----

        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.patch.set_facecolor('none')
        t = ax.text(
            0, 0, mathTex, 
            ha='left', va='bottom', 
            fontsize=13, 
            color=widgets_style.TEXT_COLOR
        )

        #---- fit figure size to text artist ----

        fwidth, fheight = fig.get_size_inches()
        fig_bbox = fig.get_window_extent(renderer)

        text_bbox = t.get_window_extent(renderer)

        tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
        tight_fheight = text_bbox.height * fheight / fig_bbox.height

        fig.set_size_inches(tight_fwidth, tight_fheight)

        #---- convert mpl figure to QPixmap ----

        buf, size = fig.canvas.print_to_buffer()
        qimage = QImage.rgbSwapped(QImage(
            buf, size[0], size[1], QImage.Format_ARGB32)
        )
        qpixmap = QPixmap(qimage)

        return qpixmap


class LabelsWidget(QWidget):
    def __init__(self, texts, wrapText=False, parent=None):
        super().__init__(parent=parent)
        
        layout = QVBoxLayout()
        
        texts = self.fixParagraphTags(texts)
        
        self.textLengths = []
        self.labels = []
        for t, text in enumerate(texts):
            if not text:
                continue

            if text.startswith('<latex>'):
                layout.addSpacing(10)
                label = LatexLabel(text)
                layout.addWidget(label, alignment=Qt.AlignCenter)
                try:
                    # Add spacing only if next text is not a formula
                    nextText = texts[t+1]
                    if not nextText.startswith('<latex>'):
                        layout.addSpacing(10)
                except IndexError:
                    layout.addSpacing(10)
            elif text.startswith('<copiable>'):
                text = (
                    text.removeprefix('<copiable>')
                    .removeprefix('</copiable>')
                )
                label = CopiableCommandWidget(command=text, parent=self)
                layout.addWidget(label)
            else:
                label = QLabel(text)
                label.setWordWrap(wrapText)
                label.setOpenExternalLinks(True)
                layout.addWidget(label)
                if wrapText:
                    self.textLengths.append(1)
                self.textLengths.extend(
                    [len(line) for line in text.split('<br>')]
                )
            
            self.labels.append(label)
        
        self.nCharsLongestLine = max(self.textLengths, default=1)
        
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
    
    def setWordWrap(self, wordWrap):
        for label in self.labels:
            label.setWordWrap(wordWrap)
    
    def fixParagraphTags(self, texts):
        firstText = texts[0]
        if firstText.find('<p style=') == -1:
            return texts
        
        searched = re.search(r'<p style="[\w\-\:\;]+">', firstText)
        if searched is None:
            openTag = '<p style="font-size:13px;">'
        else:
            openTag = searched.group()
        
        not_allowed = {' ', '\n'}
        
        fixedTexts = []
        for text in texts:
            if text.startswith('<latex>') or text.startswith('<copiable>'):
                fixedTexts.append(text)
                continue
            
            if set(text) <= not_allowed:
                # Ignore texts that are made of only \n and spaces
                continue
            
            if text.find('</p>') == -1:
                text = rf'{text}<\p>'
            
            if text.find(openTag) == -1:
                text = f'{openTag}{text}'
            
            text = text.replace('\n', '')
            
            fixedTexts.append(text)
        return fixedTexts


def addWidgetToScrollArea(
        widget, 
        resizeMinWidthNoHorizontalScrollbar=False, 
        resizeMinHeightNoVerticalScrollbar=False
    ):
    container = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(widget)
    layout.addStretch(1)
    container.setLayout(layout)
    scrollArea = QScrollArea()
    scrollArea.setWidgetResizable(True)
    scrollArea.setWidget(container)
    
    if resizeMinWidthNoHorizontalScrollbar:
        scrollArea.setMinimumWidth(
            container.sizeHint().width()
            + scrollArea.verticalScrollBar().sizeHint().width()
        )
    
    if resizeMinHeightNoVerticalScrollbar:
        scrollArea.setMinimumHeight(
            container.sizeHint().height()
            + scrollArea.horizontalScrollBar().sizeHint().height()
        )
    
    return scrollArea


class warnVisualCppRequired(myMessageBox):
    def __init__(self, pkg_name='javabridge', parent=None):
        super().__init__(parent)
        self.screenShotWin = None

        self.setIcon(iconName='SP_MessageBoxWarning')
        self.setWindowTitle(f'Installation of {pkg_name} info')
        txt = html_utils.paragraph(f"""
            Installation of {pkg_name} on Windows requires
            Microsoft Visual C++ 14.0 or higher.<br><br>
            Cell-ACDC will anyway try to install {pkg_name} now.<br><br>
            If the installation fails, please <b>close Cell-ACDC</b>,
            then download and install <b>"Microsoft C++ Build Tools"</b>
            from the link below
            before trying this module again.<br><br>
            <a href='https://visualstudio.microsoft.com/visual-cpp-build-tools/'>
                https://visualstudio.microsoft.com/visual-cpp-build-tools/
            </a><br><br>
            <b>IMPORTANT</b>: when installing "Microsoft C++ Build Tools"
            make sure to select <b>"Desktop development with C++"</b>.
            Click "See the screenshot" for more details.
        """)
        seeScreenshotButton = QPushButton('See screenshot...')
        okButton = widgets_push_buttons.okPushButton('Ok')
        okButton = self.addButton('Ok')
        okButton.disconnect()
        okButton.clicked.connect(self.ok_cb)
        self.addButton(seeScreenshotButton)
        seeScreenshotButton.disconnect()
        seeScreenshotButton.clicked.connect(
            self.viewScreenshot
        )
        self.addCancelButton(connect=True)
        self.addText(txt)

    def ok_cb(self):
        self.cancel = False
        self.close()
    
    def viewScreenshot(self, checked=False):
        self.screenShotWin = view_visualcpp_screenshot(self)
        self.screenShotWin.show()

    def closeEvent(self, event):
        if self.screenShotWin is not None:
            self.screenShotWin.close()
            
        return super().closeEvent(event)
