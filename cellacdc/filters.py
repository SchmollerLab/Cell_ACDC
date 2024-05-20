import numpy as np

import skimage.filters

from cellacdc import html_utils

from . import GUI_INSTALLED, core, myutils
from . import preprocess

if GUI_INSTALLED:
    from . import widgets
    from qtpy import QtGui
    from qtpy.QtCore import Qt, Signal
    from qtpy.QtWidgets import (
        QDialog, QVBoxLayout, QFormLayout, QHBoxLayout, QComboBox, QDoubleSpinBox,
        QSlider, QCheckBox, QPushButton, QLabel, QGroupBox, QGridLayout,
        QWidget
    )
    font = QtGui.QFont()
    font.setPixelSize(13)

class FilterBaseDialog(QDialog):
    sigClose = Signal(object)
    sigApplyFilter = Signal(str)
    sigPreviewToggled = Signal(bool, object, str)
    
    def __init__(
            self, layersChannelNames, winTitle, parent=None, 
            currentChannel=None
        ):
        super().__init__(parent)

        self.cancel = True
        self.setFont(font)

        self.setWindowTitle(winTitle)
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        self.channelsWidget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(QLabel('Channel to filter:'))

        self.channelsComboBox = QComboBox()
        self.channelsComboBox.addItems(layersChannelNames)
        if currentChannel is not None:
            self.channelsComboBox.setCurrentText(currentChannel)
        layout.addWidget(self.channelsComboBox)
        layout.setStretch(1,1)

        self.channelsWidget.setLayout(layout)
    
    def addChannelName(self, newChannelName):
        self.channelsComboBox.addItem(newChannelName)

    def preview_cb(self, checked):
        channelName = self.channelsComboBox.currentText()
        self.sigPreviewToggled.emit(checked, self, channelName)

    def apply(self):
        channelName = self.channelsComboBox.currentText()
        self.sigApplyFilter.emit(channelName)
    
    def closeEvent(self, event):
        self.sigClose.emit(self)

class gaussBlurDialog(FilterBaseDialog):
    def __init__(self, layersChannelNames, parent=None, **kwargs):
        currentChannel = kwargs['currentChannel']

        super().__init__(
            layersChannelNames, 'Gaussian blur sigma', parent=parent,
            currentChannel=currentChannel        
        )
        
        mainLayout = QVBoxLayout()
        gridLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        mainLayout.addWidget(self.channelsWidget)

        self.sigmaWidget = widgets.VectorLineEdit()
        self.sigma = 1.0
        self.sigmaWidget.setValue(self.sigma)
        self.sigmaWidget.setMinimum(0)
        label = QLabel('Gaussian filter sigma:  ')
        gridLayout.addWidget(label, 0, 0)
        gridLayout.addWidget(self.sigmaWidget, 0, 1)
        stepUpButton = widgets.addPushButton()
        stepDownButton = widgets.subtractPushButton()
        gridLayout.addWidget(stepDownButton, 0, 2)
        gridLayout.addWidget(stepUpButton, 0, 3)
        for i in range(4):
            gridLayout.setColumnStretch(i, int(i==1))
        
        self.PreviewCheckBox = QCheckBox("Preview")
        self.PreviewCheckBox.setChecked(True)
        
        self.warnLabel = QLabel()

        mainLayout.addLayout(gridLayout)
        mainLayout.addWidget(self.warnLabel)
        mainLayout.addWidget(self.PreviewCheckBox)

        closeButton = widgets.cancelPushButton('Close')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(closeButton)

        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)

        stepUpButton.clicked.connect(self.increaseSigmaValue)
        stepDownButton.clicked.connect(self.decreaseSigmaValue)
        self.PreviewCheckBox.toggled.connect(self.preview_cb)
        self.sigmaWidget.valueChangeFinished.connect(self.sigmaValueChanged)
        self.channelsComboBox.currentTextChanged.connect(self.apply)
        closeButton.clicked.connect(self.close)

        self.setLayout(mainLayout)
    
    def increaseSigmaValue(self):
        self.sigmaWidget.increaseValue(0.5)
    
    def decreaseSigmaValue(self):
        self.sigmaWidget.decreaseValue(0.5)
    
    def filter(self, image):
        if not isinstance(self.sigma, float):
            if image.ndim != len(self.sigma):
                self.warnLenSigmaNotEqualImageNumDim(image.ndim)
                return image
        self.warnLabel.setText('')
        return skimage.filters.gaussian(image, self.sigma)

    def warnLenSigmaNotEqualImageNumDim(self, ndim):
        self.warnLabel.setText(html_utils.span(
            'Number of multiple sigmas must be equal to image number '
            f'of dimensions (={ndim})', font_size='9px'
        ))
    
    def sigmaValueChanged(self, val):
        self.sigma = val
        self.apply()
    
    def show(self):
        super().show()
        self.resize(int(self.width()*1.3), self.height())

class diffGaussFilterDialog(FilterBaseDialog):
    def __init__(self, layersChannelNames, parent=None, **kwargs):
        currentChannel = kwargs['currentChannel']

        super().__init__(
            layersChannelNames, 'Gaussian blur sigma', parent=parent,
            currentChannel=currentChannel        
        )

        is3D = kwargs['is3D']

        mainLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        mainLayout.addWidget(self.channelsWidget)

        firstGroupbox = QGroupBox('First gaussian filter')
        firstLayout = QVBoxLayout()
        self.firstSigmaSliderYX = widgets.sliderWithSpinBox(
            isFloat=True, title='Sigma YX-direction:',
            title_loc='in_line'
        )
        self.firstSigmaSliderYX.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.firstSigmaSliderYX.setSingleStep(0.5)
        self.firstSigmaSliderYX.setTickInterval(10)
        self.firstSigmaSliderZ = widgets.sliderWithSpinBox(
            isFloat=True, title='Sigma Z-direction:  ',
            title_loc='in_line'
        )
        self.firstSigmaSliderZ.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.firstSigmaSliderZ.setSingleStep(0.5)
        self.firstSigmaSliderZ.setTickInterval(10)
        firstLayout.addWidget(self.firstSigmaSliderYX)
        firstLayout.addWidget(self.firstSigmaSliderZ)
        firstGroupbox.setLayout(firstLayout)

        secondGroupbox = QGroupBox('Second gaussian filter')
        secondLayout = QVBoxLayout()
        self.secondSigmaSliderYX = widgets.sliderWithSpinBox(
            isFloat=True, title='Sigma YX-direction:',
            title_loc='in_line'
        )
        self.secondSigmaSliderYX.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.secondSigmaSliderYX.setSingleStep(0.5)
        self.secondSigmaSliderYX.setTickInterval(10)
        self.secondSigmaSliderYX.setValue(1)

        self.secondSigmaSliderZ = widgets.sliderWithSpinBox(
            isFloat=True, title='Sigma Z-direction:  ',
            title_loc='in_line'
        )
        self.secondSigmaSliderZ.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.secondSigmaSliderZ.setSingleStep(0.5)
        self.secondSigmaSliderZ.setTickInterval(10)

        secondLayout.addWidget(self.secondSigmaSliderYX)
        secondLayout.addWidget(self.secondSigmaSliderZ)
        secondGroupbox.setLayout(secondLayout)

        if not is3D:
            self.firstSigmaSliderZ.hide()
            self.secondSigmaSliderZ.hide()

        self.PreviewCheckBox = QCheckBox('Preview filter')
        self.PreviewCheckBox.setChecked(True)

        cancelButton = widgets.cancelPushButton('Cancel')
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)

        mainLayout.addWidget(firstGroupbox)
        mainLayout.addSpacing(20)
        mainLayout.addWidget(secondGroupbox)
        mainLayout.addSpacing(20)
        mainLayout.addWidget(self.PreviewCheckBox)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch(1)

        self.setLayout(mainLayout)
        self.setFont(font)

        self.firstSigmaSliderYX.sigValueChange.connect(self.apply)
        self.secondSigmaSliderYX.sigValueChange.connect(self.apply)
        if not is3D:
            self.firstSigmaSliderZ.sigValueChange.connect(self.apply)
            self.secondSigmaSliderZ.sigValueChange.connect(self.apply)

        cancelButton.clicked.connect(self.close)
        self.PreviewCheckBox.toggled.connect(self.preview_cb)
    
    def filter(self, img):
        sigmas1, sigmas2 = self.getSigmas()
        sigma1_yx = sigmas1 if isinstance(sigmas1, float) else sigmas1[1]
        sigma2_yx = sigmas2 if isinstance(sigmas2, float) else sigmas2[1]
        if sigma1_yx>0:
            filtered1 = skimage.filters.gaussian(img, sigma=sigmas1)
        else:
            filtered1 = myutils.img_to_float(img)

        if sigma2_yx>0:
            filtered2 = skimage.filters.gaussian(img, sigma=sigmas2)
        else:
            filtered2 = myutils.img_to_float(img)

        resultFiltered = filtered1 - filtered2
        return resultFiltered

    def initSpotmaxValues(self, posData):
        self.firstSigmaSliderYX.setValue(0)
        self.firstSigmaSliderZ.setValue(0)
        PhysicalSizeY = posData.PhysicalSizeY
        PhysicalSizeX = posData.PhysicalSizeX
        PhysicalSizeZ = posData.PhysicalSizeZ
        zyx_vox_dim = [PhysicalSizeZ, PhysicalSizeY, PhysicalSizeX]
        wavelen = 510
        NA = 1.4
        yx_resolution_multi = 1
        z_resolution_limit = 1
        _, zyx_resolution_pxl, _ = core.calc_resolution_limited_vol(
            wavelen, NA, yx_resolution_multi, zyx_vox_dim, z_resolution_limit
        )
        self.secondSigmaSliderYX.setValue(zyx_resolution_pxl[1])
        self.secondSigmaSliderZ.setValue(zyx_resolution_pxl[0])

    def getSigmas(self):
        sigma1_yx = self.firstSigmaSliderYX.value()
        sigma1_z = self.firstSigmaSliderZ.value()
        sigma2_yx = self.secondSigmaSliderYX.value()
        sigma2_z = self.secondSigmaSliderZ.value()
        sigmas1 = (sigma1_z, sigma1_yx, sigma1_yx) if sigma1_z>0 else sigma1_yx
        sigmas2 = (sigma2_z, sigma2_yx, sigma2_yx) if sigma2_z>0 else sigma2_yx
        return sigmas1, sigmas2

    def showEvent(self, event):
        self.resize(int(self.width()*1.5), self.height())
        self.firstSigmaSliderYX.setFocus()

class edgeDetectionDialog(FilterBaseDialog):
    def __init__(self, layersChannelNames, parent=None, **kwargs):
        currentChannel = kwargs['currentChannel']

        super().__init__(
            layersChannelNames, 'Gaussian blur sigma', parent=parent,
            currentChannel=currentChannel        
        )

        self.keys = layersChannelNames

        mainLayout = QVBoxLayout()
        paramsLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        mainLayout.addWidget(self.channelsWidget)

        row = 0
        sigmaQSLabel = QLabel('Blur:')
        paramsLayout.addWidget(sigmaQSLabel, row, 0)
        row += 1
        self.sigmaValLabel = QLabel('1.00')
        paramsLayout.addWidget(self.sigmaValLabel, row, 1)
        self.sigmaSlider = QSlider(Qt.Horizontal)
        self.sigmaSlider.setMinimum(1)
        self.sigmaSlider.setMaximum(100)
        self.sigmaSlider.setValue(20)
        self.sigma = 1.0
        self.sigmaSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sigmaSlider.setTickInterval(10)
        paramsLayout.addWidget(self.sigmaSlider, row, 0)

        row += 1
        sharpQSLabel = QLabel('Sharpen:')
        # padding: top, left, bottom, right
        sharpQSLabel.setStyleSheet("font-size:13px; padding:5px 0px 0px 0px;")
        paramsLayout.addWidget(sharpQSLabel, row, 0)
        row += 1
        self.sharpValLabel = QLabel('5.00')
        paramsLayout.addWidget(self.sharpValLabel, row, 1)
        self.sharpSlider = QSlider(Qt.Horizontal)
        self.sharpSlider.setMinimum(1)
        self.sharpSlider.setMaximum(100)
        self.sharpSlider.setValue(50)
        self.radius = 5.0
        self.sharpSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sharpSlider.setTickInterval(10)
        paramsLayout.addWidget(self.sharpSlider, row, 0)

        row += 1
        self.PreviewCheckBox = QCheckBox("Preview")
        self.PreviewCheckBox.setChecked(True)
        paramsLayout.addWidget(self.PreviewCheckBox, row, 0, 1, 2,
                               alignment=Qt.AlignCenter)


        closeButton = widgets.cancelPushButton('Close')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(closeButton)

        paramsLayout.setContentsMargins(0, 10, 0, 0)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(paramsLayout)
        mainLayout.addLayout(buttonsLayout)

        self.PreviewCheckBox.clicked.connect(self.preview_cb)
        self.sigmaSlider.sliderMoved.connect(self.sigmaSliderMoved)
        self.sharpSlider.sliderMoved.connect(self.sharpSliderMoved)
        self.channelsComboBox.currentTextChanged.connect(self.apply)
        closeButton.clicked.connect(self.close)

        self.setLayout(mainLayout)

    def filter(self, img):
        edge = skimage.filters.sobel(img)
        img = skimage.filters.gaussian(edge, sigma=self.sigma)
        img = img - skimage.filters.gaussian(img, sigma=self.radius)
        return img

    def sigmaSliderMoved(self, intVal):
        self.sigma = intVal/20
        self.sigmaValLabel.setText(f'{self.sigma:.2f}')
        self.apply()

    def sharpSliderMoved(self, intVal):
        self.radius = 10 - intVal/10
        if self.radius < 0.15:
            self.radius = 0.15
        self.sharpValLabel.setText(f'{intVal/10:.2f}')
        self.apply()
    
    def show(self):
        super().show()
        self.resize(int(self.width()*1.3), self.height())

class entropyFilterDialog(FilterBaseDialog):
    def __init__(self, layersChannelNames, parent=None,  **kwargs):
        currentChannel = kwargs['currentChannel']

        super().__init__(
            layersChannelNames, 'Gaussian blur sigma', parent=parent,
            currentChannel=currentChannel        
        )

        mainLayout = QVBoxLayout()
        paramsLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        mainLayout.addWidget(self.channelsWidget)

        row = 0
        sigmaQSLabel = QLabel('Radius: ')
        paramsLayout.addWidget(sigmaQSLabel, row, 0)
        row += 1
        self.radiusValLabel = QLabel('10')
        paramsLayout.addWidget(self.radiusValLabel, row, 1)
        self.radiusSlider = QSlider(Qt.Horizontal)
        self.radiusSlider.setMinimum(1)
        self.radiusSlider.setMaximum(100)
        self.radiusSlider.setValue(10)
        self.radiusSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.radiusSlider.setTickInterval(10)
        paramsLayout.addWidget(self.radiusSlider, row, 0)

        row += 1
        self.PreviewCheckBox = QCheckBox("Preview")
        self.PreviewCheckBox.setChecked(True)
        paramsLayout.addWidget(self.PreviewCheckBox, row, 0, 1, 2,
                               alignment=Qt.AlignCenter)

        closeButton = widgets.cancelPushButton('Close')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(closeButton)

        paramsLayout.setContentsMargins(0, 10, 0, 0)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(paramsLayout)
        mainLayout.addLayout(buttonsLayout)

        self.PreviewCheckBox.clicked.connect(self.preview_cb)
        self.radiusSlider.sliderMoved.connect(self.radiusSliderMoved)
        self.channelsComboBox.currentTextChanged.connect(self.apply)
        closeButton.clicked.connect(self.close)

        self.setLayout(mainLayout)

    def filter(self, img):
        radius = self.radiusSlider.sliderPosition()
        selem = skimage.morphology.disk(radius)
        if img.ndim == 3:
            filtered = np.zeros(img.shape)
            for z, _img in enumerate(img):
                filtered[z] = skimage.filters.rank.entropy(_img, selem)
        else:
            filtered = skimage.filters.rank.entropy(img, selem)
        return filtered

    def radiusSliderMoved(self, intVal):
        self.radiusValLabel.setText(f'{intVal}')
        self.apply()
    
    def show(self):
        super().show()
        self.resize(int(self.width()*1.3), self.height())

class RidgeFilterDialog(FilterBaseDialog):
    def __init__(self, layersChannelNames, parent=None, **kwargs):
        currentChannel = kwargs['currentChannel']

        super().__init__(
            layersChannelNames, 'Ridge filter', parent=parent,
            currentChannel=currentChannel        
        )
        
        mainLayout = QVBoxLayout()
        formLayout = QFormLayout()
        buttonsLayout = QHBoxLayout()

        mainLayout.addWidget(self.channelsWidget)
        
        self.sigmasWidget = widgets.VectorLineEdit()
        self.sigmasWidget.setValue(1.0)
        self.sigmas = (1.0,)
        formLayout.addRow('Sigmas (comma separated):  ', self.sigmasWidget)
        
        self.PreviewCheckBox = QCheckBox("Preview")
        self.PreviewCheckBox.setChecked(True)

        mainLayout.addLayout(formLayout)
        mainLayout.addWidget(self.PreviewCheckBox)
        
        closeButton = widgets.cancelPushButton('Close')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(closeButton)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(buttonsLayout)
        
        self.PreviewCheckBox.toggled.connect(self.preview_cb)
        self.channelsComboBox.currentTextChanged.connect(self.apply)
        self.sigmasWidget.valueChangeFinished.connect(self.sigmasChanged)
        closeButton.clicked.connect(self.close)

        self.setLayout(mainLayout)
    
    def sigmasChanged(self, value):
        if isinstance(value, float):
            self.sigmas = (value,)
        else:
            self.sigmas = value
        self.apply()
    
    def filter(self, image):
        return preprocess.ridge_filter(image, self.sigmas)
    
    # def show(self):
    #     super().show()
    #     self.resize(int(self.width()*1.3), self.height())