import os, sys

import pandas as pd
import numpy as np

from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QTextEdit, QPushButton,
    QListWidget, QListWidgetItem, QCheckBox, QFrame, QStyleFactory,
    QLabel, QPushButton
)

script_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(os.path.dirname(script_path))
sys.path.append(src_path)

# NOTE: Enable icons
import qrc_resources

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.yeastacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

class welcomeWin(QWidget):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent)
        self.setWindowTitle('Welcome')
        self.setWindowIcon(QIcon(":assign-motherbud.svg"))
        self.loadSettings()

        self.itemsDict = {}

        # Initialize Permanent items
        self.mainLayout = QGridLayout()
        self.addListBox()
        self.addShowGuideCheckbox()

        # Create all pages of the guide as one frame for each page
        self.addWelcomePage()

        self.setStyleSheet(
            "QListWidget::item:hover {background-color:#E6E6E6;}"
            "QListWidget::item:selected {background-color:#CFEB9B;}"
            "QListWidget::item:selected {color:black;}"
            # "QTextEdit {background-color:#e8e8e8;}"
        )

        self.setLayout(self.mainLayout)

    def loadSettings(self):
        temp_path = os.path.join(src_path, 'temp')
        csv_path = os.path.join(temp_path, 'settings.csv')
        if os.path.exists(csv_path):
            self.df_settings = pd.read_csv(csv_path, index_col='setting')
            if 'showWelcomeGuide' not in self.df_settings.index:
                self.df_settings.at['showWelcomeGuide', 'value'] = 'True'
            else:
                idx = ['showWelcomeGuide']
                values = ['True']
                self.df_settings = pd.DataFrame({'setting': idx,
                                                 'value': values}
                                               ).set_index('setting')
                self.df_settings.to_csv(csv_path)

    def addListBox(self):
        listBox = QListWidget()

        listBox.setFrameStyle(QFrame.NoFrame)

        self.welcomeItem = QListWidgetItem(QIcon(':home.svg'), 'Welcome')
        listBox.addItem(self.welcomeItem)

        self.quickStartItem = QListWidgetItem(QIcon(':quickStart.svg'), 'Quick Start')
        listBox.addItem(self.quickStartItem)

        self.settingsItem = QListWidgetItem(QIcon(':cog.svg'), 'Settings')
        listBox.addItem(self.settingsItem)

        self.manualItem = QListWidgetItem(QIcon(':book.svg'), 'User Manual')
        listBox.addItem(self.manualItem)

        self.contributeItem = QListWidgetItem(QIcon(':contribute.svg'), 'Contribute')
        listBox.addItem(self.contributeItem)
        listBox.setSpacing(3)

        listBox.setCurrentItem(self.welcomeItem)

        self.listBox = listBox
        self.mainLayout.addWidget(listBox, 0, 0)

        listBox.currentTextChanged.connect(self.listItemChanged)

    def listItemChanged(self, itemText):
        for key, value in self.itemsDict.items():
            if key == itemText:
                value.show()
            else:
                value.hide()


    def addWelcomePage(self):
        self.welcomeFrame = QFrame(self)
        welcomeLayout = QGridLayout()

        welcomeTextWidget = QLabel()

        # welcomeTextWidget = QTextEdit()
        # welcomeTextWidget.setReadOnly(True)
        # welcomeTextWidget.setFrameStyle(QFrame.NoFrame)
        # welcomeTextWidget.viewport().setAutoFillBackground(False)

        htmlTxt = (
        """
        <html>
        <head>
        <title></title>
        <style type="text/css">
        blockquote {
         margin: 5;
         padding: 0;
        }
        </style>
        </head>
        <body>
        <blockquote>
        <p style="font-size:18pt; font-family:ubuntu">
            <b>Welcome to Yeast-ACDC</b>
        </p>
        <p style="font-size:12pt; font-family:ubuntu">
            Welcome to your new image analysis tool!
        </p>
        <p style="font-size:12pt; font-family:ubuntu">
            Yeast-ACDC is open-source software for
            <b>segmentation</b>, <b>tracking,</b> and<br>
            <b>cell cycle annotation</b> of microscopy imaging data.
        </p>
        <p style="font-size:12pt; font-family:ubuntu">
            For more info see our paper here or navigate the menus on the left.
        </p>
        <p style="font-size:12pt; font-family:ubuntu">
            What would you like to do, now?<br>
        </p>
        </blockquote>
        </body>
        </html>
        """
        )

        # welcomeTextWidget.setHtml(htmlTxt)
        welcomeTextWidget.setText(htmlTxt)

        welcomeLayout.addWidget(welcomeTextWidget, 0, 0, 1, 4,
                                alignment=Qt.AlignTop)

        startWizardButton = QPushButton(' Launch Wizard')
        startWizardButton.setIcon(QIcon(':wizard.svg'))

        welcomeLayout.addWidget(startWizardButton, 1, 0)

        startWizardButton = QPushButton(' Test segmentation with single image')
        startWizardButton.setIcon(QIcon(':image.svg'))
        startWizardButton.clicked.connect(self.openGUIsingleImage)

        welcomeLayout.addWidget(startWizardButton, 1, 1)

        self.welcomeFrame.setLayout(welcomeLayout)
        self.mainLayout.addWidget(self.welcomeFrame, 0, 1)
        self.itemsDict[self.welcomeItem.text()] = self.welcomeFrame

    def addShowGuideCheckbox(self):
        checkBox = QCheckBox('Show Welcome Guide when opening Yeast-ACDC')
        checked = self.df_settings.at['showWelcomeGuide', 'value'] == 'True'
        checkBox.setChecked(checked)
        self.mainLayout.addWidget(checkBox, 1, 1, alignment=Qt.AlignRight)

    def openGUIsingleImage(self):
        pass

    def showAndSetSize(self):
        self.show()
        font = QFont()
        font.setPointSize(10)
        self.listBox.setFont(font)
        self.listBox.setFixedWidth(self.listBox.sizeHintForColumn(0)+20)

        # h = int(self.welcomeTextWidget.document().size().height())
        # self.welcomeTextWidget.setMinimumHeight(h)

if __name__ == '__main__':
    app = QApplication([])
    win = welcomeWin()
    win.showAndSetSize()
    app.setStyle(QStyleFactory.create('Fusion'))
    sys.exit(app.exec_())
