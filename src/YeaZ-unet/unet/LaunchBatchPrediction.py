# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:38:58 2019
"""

from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QLineEdit, QFormLayout, 
                             QLabel, QListWidget, QAbstractItemView, QCheckBox,
                             QButtonGroup, QRadioButton)
from PyQt5 import QtGui


class CustomDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(CustomDialog, self).__init__(*args, **kwargs)
        
        app, = args
        maxtimeindex = app.reader.sizet
        
        self.setWindowTitle("Launch NN")
        self.setGeometry(100,100, 500,200)
        
        self.entry1 = QLineEdit()
        self.entry1.setValidator(QtGui.QIntValidator(0,int(maxtimeindex-1)))
        self.entry2 = QLineEdit()
        self.entry2.setValidator(QtGui.QIntValidator(0,int(maxtimeindex-1)))
        
        # FOV dialog
        self.listfov = QListWidget()
        self.listfov.setSelectionMode(QAbstractItemView.MultiSelection)
        for f in range(0, app.reader.Npos):
            self.listfov.addItem('Field of View {}'.format(f+1))

        self.labeltime = QLabel("Enter range of frames ({}-{}) to segment".format(0, app.reader.sizet-1))
        
        self.entry_threshold = QLineEdit()
        self.entry_threshold.setValidator(QtGui.QDoubleValidator())
        self.entry_threshold.setText('0.5')
        
        self.entry_segmentation = QLineEdit()
        self.entry_segmentation.setValidator(QtGui.QIntValidator())
        self.entry_segmentation.setText('5')
                
        flo = QFormLayout()
        flo.addWidget(self.labeltime)
        flo.addRow('Start from frame:', self.entry1)
        flo.addRow('End at frame:', self.entry2)        
        flo.addRow('Select field(s) of view:', self.listfov)
        flo.addRow('Threshold value:', self.entry_threshold)
        flo.addRow('Min. distance between seeds:', self.entry_segmentation)
        
        self.radiobuttons = QButtonGroup()
        self.buttonBF = QRadioButton('Images are bright-field')
        self.buttonPC = QRadioButton('Images are phase contrast')
        self.buttonPC.setChecked(True)
        self.radiobuttons.addButton(self.buttonBF, id=0)
        self.radiobuttons.addButton(self.buttonPC, id=1)
        flo.addWidget(self.buttonBF)
        flo.addWidget(self.buttonPC)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        flo.addWidget(self.buttonBox)
        self.setLayout(flo)
        

        
