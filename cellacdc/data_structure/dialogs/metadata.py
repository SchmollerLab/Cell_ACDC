"""Data structure module dialogs for raw-file metadata and conversion (from
apps.py): the OME-XML metadata dialog QDialogMetadataXML and the Fiji
macro setup InitFijiMacroDialog, plus the widgets only they use.
"""
import traceback
from functools import partial
import html
from qtpy.QtCore import QTimer, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QLabel, QHBoxLayout, QToolButton, QApplication, QWidget, QVBoxLayout,
    QScrollArea, QComboBox
)
from cellacdc import myutils, is_mac, is_win, html_utils, is_linux
from cellacdc.shared.widgets import item_views as widgets_item_views
from cellacdc.shared.widgets import message_box as widgets_message_box


class ExpandableListBox(QComboBox):
    def __init__(self, parent=None, centered=True) -> None:
        super().__init__(parent)

        self.setEditable(True)
        self.lineEdit().setReadOnly(True)

        infoTxt = html_utils.paragraph(
            'Select <b>Positions to save</b><br><br>'
            '<code>Ctrl+Click</code> <i>to select multiple items</i><br>'
            '<code>Shift+Click</code> <i>to select a range of items</i><br>',
            center=True
        )

        self.listW = widgets_item_views.QDialogListbox(
            'Select Positions to save', infoTxt,
            [], multiSelection=True, parent=self
        )

        self.listW.listBox.itemClicked.connect(self.listItemClicked)
        self.listW.sigSelectionConfirmed.connect(self.updateCombobox)

        self.centered = centered 

    def listItemClicked(self, item):
        if item.text().find('All') == -1:
            return
        
        for i in range(self.listW.listBox.count()):
            _item = self.listW.listBox.item(i)
            _item.setSelected(True)

    def clear(self) -> None:
        self.listW.listBox.clear()
        return super().clear()
    
    def setItems(self, items):
        self.clear()
        self.addItems(items)
    
    def addItems(self, items):
        super().addItems(items)
        self.listW.listBox.addItems(items)
        self.listW.listBox.setCurrentRow(self.currentIndex())
        self.listItemClicked(self.listW.listBox.currentItem())
        if self.centered:
            self.centerItems()
    
    def updateCombobox(self, selectedItemsText):
        isAllItem = [
            i for i, t in enumerate(selectedItemsText) if t.find('All') != -1
        ]
        if len(selectedItemsText) == 1:
            self.setCurrentText(selectedItemsText[0])
        elif isAllItem:
            idx = isAllItem[0]
            self.setCurrentText(selectedItemsText[idx])
        else:
            super().clear()
            super().addItems(['Custom selection'])
    
    def centerItems(self, idx=None):
        self.lineEdit().setAlignment(Qt.AlignCenter)
    
    def selectedItems(self):
        return self.listW.listBox.selectedItems()
    
    def selectedItemsText(self):
        return [item.text() for item in self.selectedItems()]
    
    def showPopup(self) -> None:
        self.listW.show()


class installJavaDialog(widgets_message_box.myMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Install Java')
        self.setIcon('SP_MessageBoxWarning')

        txt_macOS = html_utils.paragraph("""
            Your system doesn't have the <code>Java Development Kit</code>
            installed<br> and/or a C++ compiler which is required for the installation of
            <code>javabridge</code><br><br>
            <b>Cell-ACDC is now going to install Java for you</b>.<br><br>
            <i><b>NOTE: After clicking on "Install", follow the instructions<br>
            on the terminal</b>. You will be asked to confirm steps and insert<br>
            your password to allow the installation.</i><br><br>
            If you prefer to do it manually, cancel the process<br>
            and follow the instructions below.
        """)

        txt_windows = html_utils.paragraph("""
            Unfortunately, installing pre-compiled version of
            <code>javabridge</code> <b>failed</b>.<br><br>
            Cell-ACDC is going to <b>try to compile it now</b>.<br><br>
            However, <b>before proceeding</b>, you need to install
            <code>Java Development Kit</code><br> and a <b>C++ compiler</b>.<br><br>
            <b>See instructions below on how to install it.</b>
        """)

        if not is_win:
            self.instructionsButton = self.addButton('Show intructions...')
            self.instructionsButton.setCheckable(True)
            self.instructionsButton.disconnect()
            self.instructionsButton.clicked.connect(self.showInstructions)
            installButton = self.addButton('Install')
            installButton.disconnect()
            installButton.clicked.connect(self.installJava)
            txt = txt_macOS
        else:
            okButton = self.addButton('Ok')
            txt = txt_windows

        self.cancelButton = self.addButton('Cancel')

        label = self.addText(txt)
        label.setWordWrap(False)

        self.resizeCount = 0

    def addInstructionsWindows(self):
        self.scrollArea = QScrollArea()
        _container = QWidget()
        _layout = QVBoxLayout()
        for t, text in enumerate(myutils.install_javabridge_instructions_text()):
            label = QLabel()
            label.setText(text)
            if (t == 1 or t == 2):
                label.setOpenExternalLinks(True)
                label.setTextInteractionFlags(Qt.TextBrowserInteraction)
                code_layout = QHBoxLayout()
                code_layout.addWidget(label)
                copyButton = QToolButton()
                copyButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                copyButton.setIcon(QIcon(':edit-copy.svg'))
                copyButton.setText('Copy link')
                if t==1:
                    copyButton.textToCopy = myutils.jdk_windows_url()
                    code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                else:
                    copyButton.textToCopy = myutils.cpp_windows_url()
                    screenshotButton = QToolButton()
                    screenshotButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                    screenshotButton.setIcon(QIcon(':cog.svg'))
                    screenshotButton.setText('See screenshot')
                    code_layout.addWidget(screenshotButton, alignment=Qt.AlignLeft)
                    code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                    screenshotButton.clicked.connect(self.viewScreenshot)
                copyButton.clicked.connect(self.copyToClipboard)
                code_layout.setStretch(0, 2)
                code_layout.setStretch(1, 0)
                _layout.addLayout(code_layout)
            else:
                _layout.addWidget(label)


        _container.setLayout(_layout)
        self.scrollArea.setWidget(_container)
        self.currentRow += 1
        self._layout.addWidget(
            self.scrollArea, self.currentRow, 1, alignment=Qt.AlignTop
        )

        # Stretch last row
        self.currentRow += 1
        self._layout.setRowStretch(self.currentRow, 1)

    def viewScreenshot(self, checked=False):
        self.screenShotWin = widgets_message_box.view_visualcpp_screenshot(parent=self)
        self.screenShotWin.show()

    def addInstructionsMacOS(self):
        self.scrollArea = QScrollArea()
        _container = QWidget()
        _layout = QVBoxLayout()
        for t, text in enumerate(myutils.install_javabridge_instructions_text()):
            label = QLabel()
            label.setText(text)
            # label.setWordWrap(True)
            if (t == 1 or t == 2):
                label.setWordWrap(True)
                code_layout = QHBoxLayout()
                code_layout.addWidget(label)
                copyButton = QToolButton()
                copyButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                copyButton.setIcon(QIcon(':edit-copy.svg'))
                copyButton.setText('Copy')
                if t==1:
                    copyButton.textToCopy = myutils._install_homebrew_command()
                else:
                    copyButton.textToCopy = myutils._brew_install_java_command()
                copyButton.clicked.connect(self.copyToClipboard)
                code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                # code_layout.addStretch(1)
                code_layout.setStretch(0, 2)
                code_layout.setStretch(1, 0)
                _layout.addLayout(code_layout)
            else:
                _layout.addWidget(label)
        _container.setLayout(_layout)
        self.scrollArea.setWidget(_container)
        self.currentRow += 1
        self._layout.addWidget(
            self.scrollArea, self.currentRow, 1, alignment=Qt.AlignTop
        )

        # Stretch last row
        self.currentRow += 1
        self._layout.setRowStretch(self.currentRow, 1)
        self.scrollArea.hide()

    def addInstructionsLinux(self):
        self.scrollArea = QScrollArea()
        _container = QWidget()
        _layout = QVBoxLayout()
        for t, text in enumerate(myutils.install_javabridge_instructions_text()):
            label = QLabel()
            label.setText(text)
            # label.setWordWrap(True)
            if (t == 1 or t == 2 or t==3):
                label.setWordWrap(True)
                code_layout = QHBoxLayout()
                code_layout.addWidget(label)
                copyButton = QToolButton()
                copyButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                copyButton.setIcon(QIcon(':edit-copy.svg'))
                copyButton.setText('Copy')
                if t==1:
                    copyButton.textToCopy = myutils._apt_update_command()
                elif t==2:
                    copyButton.textToCopy = myutils._apt_install_java_command()
                elif t==3:
                    copyButton.textToCopy = myutils._apt_gcc_command()
                copyButton.clicked.connect(self.copyToClipboard)
                code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                # code_layout.addStretch(1)
                code_layout.setStretch(0, 2)
                code_layout.setStretch(1, 0)
                _layout.addLayout(code_layout)
            else:
                _layout.addWidget(label)
        _container.setLayout(_layout)
        self.scrollArea.setWidget(_container)
        self.currentRow += 1
        self._layout.addWidget(
            self.scrollArea, self.currentRow, 1, alignment=Qt.AlignTop
        )

        # Stretch last row
        self.currentRow += 1
        self._layout.setRowStretch(self.currentRow, 1)
        self.scrollArea.hide()

    def copyToClipboard(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        plain_text = html.unescape(self.sender().textToCopy).replace("\xa0", " ")
        cb.setText(plain_text, mode=cb.Clipboard)
        print('Command copied!')

    def showInstructions(self, checked):
        if checked:
            self.instructionsButton.setText('Hide instructions')
            self.origHeight = self.height()
            self.resize(self.width(), self.height()+300)
            self.scrollArea.show()
        else:
            self.instructionsButton.setText('Show instructions...')
            self.scrollArea.hide()
            func = partial(self.resize, self.width(), self.origHeight)
            QTimer.singleShot(50, func)

    def installJava(self):
        import subprocess
        try:
            if is_mac:
                try:
                    subprocess.check_call(['brew', 'update'])
                except Exception as e:
                    subprocess.run(
                        myutils._install_homebrew_command(),
                        check=True, text=True, shell=True
                    )
                subprocess.run(
                    myutils._brew_install_java_command(),
                    check=True, text=True, shell=True
                )
            elif is_linux:
                subprocess.run(
                    myutils._apt_gcc_command()(),
                    check=True, text=True, shell=True
                )
                subprocess.run(
                    myutils._apt_update_command()(),
                    check=True, text=True, shell=True
                )
                subprocess.run(
                    myutils._apt_install_java_command()(),
                    check=True, text=True, shell=True
                )
            self.close()
        except Exception as e:
            print('=======================')
            traceback.print_exc()
            print('=======================')
            msg = widgets_message_box.myMessageBox(wrapText=False)
            err_msg = html_utils.paragraph("""
                Automatic installation of Java failed.<br><br>
                Please, try manually by following the instructions provided
                below (click on "Show instructions..." button). Thanks
            """)
            msg.critical(
               self, 'Java installation failed', err_msg
            )

    def show(self, block=False):
        super().show(block=False)
        print(is_linux)
        if is_win:
            self.addInstructionsWindows()
        elif is_mac:
            self.addInstructionsMacOS()
        elif is_linux:
            self.addInstructionsLinux()
        self.move(self.pos().x(), 20)
        if is_win:
            self.resize(self.width(), self.height()+200)
        if block:
            self._block()

    def exec_(self):
        self.show(block=True)
