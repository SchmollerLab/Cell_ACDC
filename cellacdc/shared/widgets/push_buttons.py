"""PushButton and its whole family of icon buttons (okPushButton,
cancelPushButton, savePushButton, infoPushButton, ...).

load.py's get_all_buttons_names finds the button classes with a regex on
this file (every `class X(PushButton)`), so keep all of them here, also
those with a single user. Also here: getPushButton (picks the icon
button for a button text, used by myMessageBox and QDialogListbox: word
button texts by its keywords), browseFileButton, showDetailsButton,
CancelOkButtonsLayout, the Toggle switch, ToggleTerminalButton,
expandCollapseButton, ToggleVisibilityButton, ToggleVisibilityCheckBox,
myColorButton (pyqtgraph color button with sigColorRejected) and
FeatureSelectorButton. Several buttons are subclassed by spotMAX.
"""
import os
from functools import partial
from qtpy.QtCore import (
    Signal, QTimer, Qt, QUrl, Property, QPropertyAnimation, QEasingCurve,
    QSize, QRect, QRect, QEasingCurve, QPropertyAnimation, QAbstractAnimation,
    Slot
)
from qtpy.QtGui import QColor, QPen, QBrush, QPainter, QIcon, QDesktopServices
from qtpy.QtWidgets import (
    QColorDialog, QLabel, QHBoxLayout, QCheckBox, QApplication, QGridLayout,
    QPushButton, QFileDialog, QStylePainter, QAbstractButton
)
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import html_utils
from cellacdc.myutils import filesystem
from cellacdc import colors
from cellacdc import QtScoped
from . import message_box as widgets_message_box
from . import basic_widgets as widgets_basic_widgets


class PushButton(QPushButton):
    def __init__(
            self, *args, icon=None, alignIconLeft=False, 
            flat=False, hoverable=False
        ):
        super().__init__(*args)
        if icon is not None:
            self.setIcon(icon)
        self.alignIconLeft = alignIconLeft
        self._text = None
        if flat:
            self.setFlat(True)
        if hoverable:
            self.installEventFilter(self)
    
    def setRetainSizeWhenHidden(self, retainSize):
        sp = self.sizePolicy()
        sp.setRetainSizeWhenHidden(retainSize)
        self.setSizePolicy(sp)
    
    def eventFilter(self, object, event):
        if event.type() == QtScoped.QEventTypeAttribute('HoverEnter'):
            self.setFlat(False)
        elif event.type() == QtScoped.QEventTypeAttribute('HoverLeave'):
            self.setFlat(True)
        return False
    
    def show(self):
        text = self.text()
        if not self.alignIconLeft:
            super().show()
            return 

        self._text = text
        self.setStyleSheet('text-align:left;')
        self.setLayout(QGridLayout())

        self.textLabel = QLabel(self._text)
        self.textLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.textLabel.setAttribute(
            Qt.WA_TransparentForMouseEvents,
            True
        )

        self._layout().addWidget(self.textLabel)

        super().show()
    
    def confirmAction(self):
        self.baseIcon = self.icon()
        self.setIcon(QIcon(':greenTick.svg'))
        QTimer.singleShot(2000, self.resetButton)
    
    def resetButton(self):
        self.setIcon(self.baseIcon)
    
    def setText(self, text):
        if self._text is None:
            super().setText(text)
        else:
            super().setText(self._text)

class LoadPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':fork_lift.svg'))

class mergePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':merge-IDs.svg'))

class okPushButton(PushButton):
    def __init__(self, *args, isDefault=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':yesGray.svg'))
        if isDefault:
            self.setDefault(True)

class MagnifyingGlassPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':magnGlass.svg'))

class MagnifyingGlassAllPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':magnGlass_all.svg'))

class AssignNewIDButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':assign_new_id.svg'))

class dataPrepButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':prep.svg'))

class LockPushButton(PushButton):
    sigToggled = Signal(bool)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':lock.svg'))
        self.toggled.connect(self.onToggled)
    
    def onToggled(self, checked):
        if not self.isCheckable():
            return
        
        if checked:
            self.setIcon(QIcon(':lock_closed.svg'))
        else:
            self.setIcon(QIcon(':lock_open.svg'))
        
        self.sigToggled.emit(checked)
    
    def setCheckable(self, checkable: bool):
        super().setCheckable(checkable)
        if checkable:
            self.setIcon(QIcon(':lock_open.svg'))
        else:
            self.setIcon(QIcon(':lock.svg'))


class SkipPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':skip_arrow.svg'))

class BedPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':bed.svg'))

class BedPlusLabelPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':bed_plus_label.svg'))
        iconH = self.iconSize().height()
        iconW = int(iconH*2.5)
        self.setIconSize(QSize(iconW, iconH))

class NoBedPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':no_bed.svg'))

class NavigatePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':navigate.svg'))

class SwitchPlaneButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':switch_2d_plane.svg'))
        self._planes = ('xy', 'zy', 'zx')
        self._idx = 0
    
    def switchPlane(self):
        self._idx += 1
    
    def setPlane(self, plane):
        self._idx = self._planes.index(plane)
    
    def plane(self):
        return self._planes[self._idx % 3]

    def depthAxes(self):
        plane = self.plane()
        for axes in 'xyz':
            if axes not in plane:
                return axes

class zoomPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':zoom_out.svg'))
    
    def setIconZoomOut(self):
        self.setIcon(QIcon(':zoom_out.svg'))
    
    def setIconZoomIn(self):
        self.setIcon(QIcon(':zoom_in.svg'))

class WarningButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':warning.svg'))

class reloadPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':reload.svg'))

class SegmentPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':segment.svg'))


class savePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':file-save.svg'))

class autoPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':cog_play.svg'))

class newFilePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':file-new.svg'))

class helpPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':help.svg'))

class viewPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':eye.svg'))

class infoPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        """
        Button for displaying info. Provide `info_text` in kwargs to display a 
        message box with the info text when clicked.
        Provide `info_title` in kwargs to set the title of the message box. If 
        `info_text` is not provided, the button will not display any message 
        box.
        """
        if 'info_text' in kwargs:
            self.info_text = kwargs.pop('info_text')
        else:
            self.info_text = None
        if 'info_title' in kwargs:
            self.info_title = kwargs.pop('info_title')
        else:
            self.info_title = 'Information'

        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':info.svg'))
        if self.info_text is not None:
            self.clicked.connect(self.show_info)

        
    def show_info(self):
        if self.info_text is None:
            return
        msg = widgets_message_box.myMessageBox(parent=self.parent())
        txt = html_utils.paragraph(self.info_text)
        msg.information(
            self, self.info_title, txt
        )

class threeDPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':3d.svg'))

class twoDPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':2d.svg'))

class addPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':add.svg'))

class futurePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':arrow_future.svg'))

class FutureAllPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':arrow_future_all.svg'))

class currentPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':arrow_current.svg'))

class arrowUpPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        alignIconLeft = kwargs.get('alignIconLeft', False)
        super().__init__(
            *args, icon=QIcon(':arrow-up.svg'), alignIconLeft=alignIconLeft
        )

class arrowDownPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':arrow-down.svg'))

class selectAllPushButton(PushButton):
    sigClicked = Signal(object, bool)
    
    def __init__(
            self,
            *args,
            suffix_text_select='all', 
            suffix_text_deselect='all',
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self._status = 'deselect'
        self._suffix_text_select = suffix_text_select
        self._suffix_text_deselect = suffix_text_deselect
        self.setIcon(QIcon(':deselect_all.svg'))
        self.setText(f'Deselect {suffix_text_deselect}')
        self.clicked.connect(self.onClicked)
        self.setMinimumWidth(self.sizeHint().width())
    
    def isChecked(self):
        return self._status == 'deselect'

    def setChecked(self, checked):
        if self.isChecked() == checked:
            self.sigClicked.emit(self, checked)
            return

        if checked:
            self._status == 'deselect'
        else:
            self._status == 'select'
        self.click()
    
    def onClicked(self):
        if self._status == 'select':
            icon_fn = ':deselect_all.svg'
            self._status = 'deselect'
            checked = True
            text = f'Deselect {self._suffix_text_deselect}'
        else:
            icon_fn = ':select_all.svg'
            text = f'Select {self._suffix_text_select}'
            self._status = 'select'
            checked = False
        self.setIcon(QIcon(icon_fn))
        self.setText(text)
        self.sigClicked.emit(self, checked)

class subtractPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':subtract.svg'))

class continuePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':continue.svg'))

class calcPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':calc.svg'))

class playPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':play.svg'))

class stopPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':stop.svg'))

class copyPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':edit-copy.svg'))
        self.clicked.connect(self.onClicked)
        self._text_to_copy = None
    
    def setTextToCopy(self, text):
        self._text_to_copy = text
    
    def onClicked(self):
        self._original_text = self.text()
        if self._text_to_copy is not None:
            cb = QApplication.clipboard()
            cb.clear(mode=cb.Clipboard)
            cb.setText(self._text_to_copy, mode=cb.Clipboard)
            
        super().setText('Copied!')
        self.setIcon(QIcon(':greenTick.svg'))
        QTimer.singleShot(2000, self.resetButton)
    
    def resetButton(self):
        self.setText(self._original_text)
        self.setIcon(QIcon(':edit-copy.svg'))

class OpenFilePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':folder-open.svg'))

class movePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':folder-move.svg'))

class DownloadPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':download.svg'))

class showInFileManagerButton(PushButton):
    def __init__(self, *args, setDefaultText=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':drawer.svg'))
        self._path_to_browse = None
        if setDefaultText:
            self.setDefaultText()
    
    def setDefaultText(self):
        self._text = filesystem.get_show_in_file_manager_text()
        self.setText(self._text)

    def setPathToBrowse(self, path: os.PathLike):
        self._path_to_browse = path
        self.clicked.connect(partial(filesystem.showInExplorer, path))



class OpenUrlButton(PushButton):
    def __init__(self, url, *args, **kwargs):
        self._url = url
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':browser.svg'))
        self.clicked.connect(self.openUrl)
    
    def openUrl(self):
        QDesktopServices.openUrl(QUrl(self._url))

class LessThanPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':less_than.svg'))
        flat = kwargs.get('flat')
        if flat is not None:
            self.setFlat(True)

class showDetailsButton(PushButton):
    sigToggled = Signal(bool)

    def __init__(
            self, *args, txt='Show details...', parent=None, underline=False
        ):
        super().__init__(parent)

        self.txt = txt
        self.checkedText = txt.replace('Show', 'Hide')

        self.checkedIcon = QIcon(':hideUp.svg')
        self.uncheckedIcon = QIcon(':showDown.svg')

        self.setCheckable(True)

        # Keep the button from using QPushButton's icon/text layout
        self.setIcon(QIcon())
        self.setText('')

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # Fixed-size icon area
        self.iconLabel = QLabel()
        self.iconLabel.setFixedSize(12, 8)
        self.iconLabel.setAlignment(Qt.AlignCenter)
        self.iconLabel.setAttribute(Qt.WA_TransparentForMouseEvents)

        # Text
        self.textLabel = QLabel(self.txt)
        self.textLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.textLabel.setAttribute(Qt.WA_TransparentForMouseEvents)

        if underline:
            font = self.textLabel.font()
            font.setUnderline(True)
            self.textLabel.setFont(font)

        layout.addWidget(self.iconLabel)
        layout.addWidget(self.textLabel)
        layout.addStretch()

        self.setIconSize(QSize(12, 8))

        self._setIcon(self.uncheckedIcon)

        self.toggled.connect(self.onClicked)

    def _setIcon(self, icon):
        pixmap = icon.pixmap(QSize(12, 8))
        self.iconLabel.setPixmap(pixmap)

    def _widestTextExtraWidth(self):
        # Extra width needed to fit the widest of the "Show" and "Hide"
        # texts, so that the button does not change width when toggled
        fm = self.textLabel.fontMetrics()
        currentWidth = fm.horizontalAdvance(self.textLabel.text())
        widestWidth = max(
            fm.horizontalAdvance(self.txt),
            fm.horizontalAdvance(self.checkedText)
        )
        return max(0, widestWidth - currentWidth)

    def sizeHint(self):
        # QPushButton.sizeHint ignores the internal layout (text is set on
        # the child label) --> include the layout's size hint
        layoutHint = self.layout().sizeHint()
        layoutHint.setWidth(
            layoutHint.width() + self._widestTextExtraWidth()
        )
        return super().sizeHint().expandedTo(layoutHint)

    def minimumSizeHint(self):
        layoutMinSize = self.layout().minimumSize()
        layoutMinSize.setWidth(
            layoutMinSize.width() + self._widestTextExtraWidth()
        )
        return super().minimumSizeHint().expandedTo(layoutMinSize)

    def onClicked(self, checked):
        if checked:
            self.textLabel.setText(self.checkedText)
            self._setIcon(self.checkedIcon)
        else:
            self.textLabel.setText(self.txt)
            self._setIcon(self.uncheckedIcon)

        self.updateGeometry()
        self.sigToggled.emit(checked)

class cancelPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':cancelButton.svg'))

class setPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':cog.svg'))

class TrainPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':train.svg'))

class noPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':no.svg'))

class editPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':edit-id.svg'))

class delPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':bin.svg'))

class eraserPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':eraser.svg'))

class CrossCursorPointButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':cross_cursor.svg'))

class TestPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':test.svg'))

class browseFileButton(PushButton):
    sigPathSelected = Signal(str)

    def __init__(
            self, *args, ext=None, title='Select file', start_dir='', 
            openFolder=False, get_start_dir_func=None, **kwargs
        ):
        """PushButton with sigPathSelected Signal to select file or folder

        Parameters
        ----------
        ext : dict or None, optional
            If not None, this is a dictionary of 
            {'FILE NAME': ['.ext1', '.ext2', ...]}. 
            For example, to allow only selection of CSV files, 
            pass {'CSV': ['.csv']}. 
            
            Note that the 'FILE NAME' is arbitrary. Default is None
        title : str, optional
            Title of the File Manager window. Default is 'Select file'
        start_dir : str, optional
            Directory where the File Manager window will initially be open. 
            Default is ''
        openFolder : bool, optional
            If True, allows for selection of folders instead of files. 
            Default is False
        get_start_dir_func : callable, optional
            If a callable function without arguments, it will be used to retrieve the directory where the file manager window will initially open. 
            If the retrieved directory is not valid or not existing, `start_dir` will be used instead.
        """        
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':folder-open.svg'))
        self.clicked.connect(self.browse)
        
        self._title = title
        self._start_dir = start_dir
        self._openFolder = openFolder
        self._get_start_dir_func = get_start_dir_func
        self._file_types = 'All Files (*)'
        if ext is not None:
            s_li = []
            for name, extensions in ext.items():
                _s = ''
                if isinstance(extensions, str):
                    extensions = [extensions]
                for ext in extensions:
                    _s = f'{_s}*{ext} '
                s_li.append(f'{name} {_s.strip()}')

            self._file_types = ';;'.join(s_li)
            self._file_types = f'{self._file_types};;All Files (*)'

    def setStartPath(self, start_path):
        self._start_dir = start_path
    
    def getStartPath(self):
        if self._get_start_dir_func is None:
            return self._start_dir
        
        try:
            start_dir = str(self._get_start_dir_func())
            if os.path.exists(start_dir):
                return start_dir
        except Exception as err:
            return self._start_dir
    
    def browse(self):
        if self._openFolder:
            fileDialog = QFileDialog.getExistingDirectory
            args = (self, self._title, self.getStartPath())
        else:
            fileDialog = QFileDialog.getOpenFileName
            args = (self, self._title, self.getStartPath(), self._file_types)
        file_path = fileDialog(*args)
        if not isinstance(file_path, str):
            file_path = file_path[0]
        if file_path:
            self.sigPathSelected.emit(file_path)

def getPushButton(buttonText, qparent=None):
    isCancelButton = (
        buttonText.lower().find('cancel') != -1
        or buttonText.lower().find('abort') != -1
    )
    isYesButton = (
        buttonText.lower().find('yes') != -1
        or buttonText.lower().find('ok') != -1
        or buttonText.lower().find('continue') != -1
        or buttonText.lower().find('recommended') != -1
    )
    isSettingsButton = buttonText.lower().find('set') != -1
    isNoButton = (
        buttonText.replace(' ', '').lower() == 'no'
        or buttonText.lower().find('do not ') != -1
        or buttonText.lower().find("don't ") != -1
        or buttonText.lower().find('no, ') != -1
    )
    isDelButton = buttonText.lower().find('delete') != -1
    isAddButton = buttonText.lower().find('add ') != -1
    is3Dbutton = buttonText.find(' 3D ') != -1
    is2Dbutton = buttonText.find(' 2D ') != -1
    isSaveButton = (
        buttonText.lower().find('overwrite') != -1
        or buttonText.lower().lstrip().startswith('save')
    )
    isNewFileButton = buttonText.lower().find('rename') != -1
    isTryAgainButton = buttonText.lower().find('try again') != -1

    if isCancelButton:
        button = cancelPushButton(buttonText, qparent)
        if qparent is not None:
            qparent.addCancelButton(button=button)
    elif isYesButton:
        button = okPushButton(buttonText, qparent)
        if qparent is not None:
            qparent.okButton = button
    elif isSettingsButton:
        button = setPushButton(buttonText, qparent)
    elif isNoButton:
        button = noPushButton(buttonText, qparent)
    elif isDelButton:
        button = delPushButton(buttonText, qparent)
    elif isAddButton:
        button = addPushButton(buttonText, qparent)
    elif is3Dbutton:
        button = threeDPushButton(buttonText, qparent)
    elif is2Dbutton:
        button = twoDPushButton(buttonText, qparent)
    elif isSaveButton:
        button = savePushButton(buttonText, qparent)
    elif isNewFileButton:
        button = newFilePushButton(buttonText, qparent)
    elif isTryAgainButton:
        button = reloadPushButton(buttonText, qparent)
    else:
        button = QPushButton(buttonText, qparent)
    
    return button, isCancelButton


class CancelOkButtonsLayout(QHBoxLayout):
    def __init__(self, *args, additionalButtons=None):
        super().__init__(*args)

        self.cancelButton = cancelPushButton('Cancel')
        self.okButton = okPushButton(' Ok ')

        self.addStretch(1)
        self.addWidget(self.cancelButton)
        self.addSpacing(20)
        
        if additionalButtons is not None:
            for button in additionalButtons:
                self.addWidget(button)
        
        self.addWidget(self.okButton)


class Toggle(QAbstractButton):
    toggled = Signal(bool)
    clicked = Signal(bool)

    def __init__(
            self,
            label_text='',
            initial=False,
            width=36,
            height=18,
            bg_color='#b3b3b3',
            circle_color='#ffffff',
            active_color='#26dd66',
            animation_curve=QEasingCurve.Type.InOutQuad,
            parent=None,
            label_spacing=6,
        ):
        super().__init__(parent)

        self._label_text = label_text
        self._label = None
        self._label_spacing = label_spacing

        self._toggle_width = width
        self._toggle_height = height

        self._bg_color = bg_color
        self._circle_color = circle_color
        self._active_color = active_color

        self._disabled_active_color = (
            colors.lighten_color(active_color)
        )
        self._disabled_circle_color = (
            colors.lighten_color(circle_color)
        )
        self._disabled_bg_color = (
            colors.lighten_color(
                bg_color,
                amount=0.5
            )
        )

        self._circle_margin = 4

        self._checked = bool(initial)

        # Will be initialized when the widget gets its
        # actual geometry.
        self._circle_position = 0

        self.animation = QPropertyAnimation(
            self,
            b'circle_position',
            self
        )
        self.animation.setEasingCurve(animation_curve)
        self.animation.setDuration(200)

        # --------------------------------------------------
        # Label
        # --------------------------------------------------
        if label_text:
            self._label = widgets_basic_widgets.QClickableLabel(self._label_text)
            self._label.setAlignment(
                Qt.AlignLeft | Qt.AlignVCenter
            )
            self._label.setAttribute(
                Qt.WA_TransparentForMouseEvents
            )
            self._label.clicked.connect(self._on_label_clicked)

        # --------------------------------------------------
        # Layout
        # --------------------------------------------------

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if label_text:
            layout.setSpacing(self._label_spacing)
            layout.addWidget(self._label)
            layout.addStretch()

        self.setCursor(Qt.PointingHandCursor)

    def _on_label_clicked(self):
        self.clicked.emit(self.isChecked())

    # ------------------------------------------------------
    # Text
    # ------------------------------------------------------

    def text(self):
        return self._label_text

    def setText(self, text):
        if self._label is None:
            return

        self._label_text = text
        self._label.setText(text)

    # ------------------------------------------------------
    # State
    # ------------------------------------------------------

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        checked = bool(checked)

        if self._checked == checked:
            return

        self._checked = checked

        # If we already have valid geometry, animate.
        if self.width() > 0 and self.height() > 0:
            self._startTransition()

        self.toggled.emit(checked)

        self.update()

    def toggle(self):
        self.setChecked(not self._checked)
    
    def click(self):
        if not self.isEnabled():
            return
        self.toggle()
        self.clicked.emit(self.isChecked())

    # ------------------------------------------------------
    # Geometry
    # ------------------------------------------------------

    def sizeHint(self):
        if self._label is None:
            return QSize(
                self._toggle_width,
                self._toggle_height,
            )

        label_size = self._label.sizeHint()

        return QSize(
            label_size.width() + self._label_spacing + self._toggle_width,
            max(
                label_size.height(),
                self._toggle_height,
            )
        )

    def toggleRect(self):
        x = self.width() - self._toggle_width

        y = int(
            (self.height() - self._toggle_height) / 2
        )

        return QRect(
            x,
            y,
            self._toggle_width,
            self._toggle_height,
        )

    def circleDiameter(self):
        return (
            self._toggle_height
            - self._circle_margin
        )

    def circlePos(self, checked):
        rect = self.toggleRect()
        diameter = rect.height() - self._circle_margin
        margin = self._circle_margin / 2

        if checked:
            pos = rect.x() + rect.width() - margin - diameter
        else:
            pos = rect.x() + margin

        return pos

    def _updateCirclePosition(self):
        self._circle_position = self.circlePos(self._checked)
        self.update()

    # ------------------------------------------------------
    # Show / resize
    # ------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)

        # At this point Qt has assigned the real geometry.
        self.animation.stop()
        self._updateCirclePosition()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Keep the circle correctly positioned if the widget
        # is resized.
        if self.animation.state() != QAbstractAnimation.State.Running:
            self._updateCirclePosition()

    # ------------------------------------------------------
    # Animation
    # ------------------------------------------------------

    @Property(float)
    def circle_position(self):
        return self._circle_position

    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update()

    def _startTransition(self):
        self.animation.stop()

        self.animation.setStartValue(
            self._circle_position
        )
        self.animation.setEndValue(
            self.circlePos(self._checked)
        )

        self.animation.start()

    # ------------------------------------------------------
    # Interaction
    # ------------------------------------------------------

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.LeftButton
            and self.isEnabled()
        ):
            self.toggle()
            self.clicked.emit(self.isChecked())
            event.accept()
            return

        super().mousePressEvent(event)

    # ------------------------------------------------------
    # Enabled state
    # ------------------------------------------------------

    def setDisabled(self, disabled):
        super().setDisabled(disabled)
        if self._label is not None:
            self._label.setDisabled(disabled)
        self.update()

    # ------------------------------------------------------
    # Painting
    # ------------------------------------------------------

    def paintEvent(self, event):
        if self.isEnabled():
            circle_color = self._circle_color
            active_color = self._active_color
            unchecked_color = self._bg_color
        else:
            circle_color = self._disabled_circle_color
            active_color = self._disabled_active_color
            unchecked_color = self._disabled_bg_color

        rect = self.toggleRect()

        p = QPainter(self)
        p.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )
        p.setPen(Qt.NoPen)

        width = self.sizeHint().width()
        height = self.sizeHint().height()

        # --------------------------------------------------
        # Background
        # --------------------------------------------------

        background_color = (
            active_color
            if self._checked
            else unchecked_color
        )

        p.setBrush(QColor(background_color))

        p.drawRoundedRect(
            rect,
            rect.height() / 2,
            rect.height() / 2,
        )

        # --------------------------------------------------
        # Circle
        # --------------------------------------------------

        p.setBrush(QColor(circle_color))

        diameter = self.circleDiameter()

        p.drawEllipse(
            int(self._circle_position),
            int(
                rect.y()
                + self._circle_margin / 2
            ),
            diameter,
            diameter,
        )

        p.end()


class ToggleTerminalButton(PushButton):
    sigClicked = Signal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':terminal_up.svg'))
        self.setFixedSize(34,18)
        self.setIconSize(QSize(30, 14))
        self.setFlat(True)
        self.terminalVisible = False
        self.clicked.connect(self.mouseClick)
    
    def mouseClick(self):
        if self.terminalVisible:
            self.setIcon(QIcon(':terminal_up.svg'))
            self.terminalVisible = False
        else:
            self.setIcon(QIcon(':terminal_down.svg'))
            self.terminalVisible = True
        self.sigClicked.emit(self.terminalVisible)
    
    def showEvent(self, a0) -> None:
        self.idlePalette = self.palette()
        return super().showEvent(a0)
    
    def enterEvent(self, event) -> None:
        self.setFlat(False)
        # pal = self.palette()
        # pal.setColor(QPalette.ColorRole.Button, QColor(200, 200, 200))
        # self.setAutoFillBackground(True)
        # self.setPalette(pal)
        self.update()
        return super().enterEvent(event)
    
    def leaveEvent(self, event) -> None:
        self.setFlat(True)
        # self.setPalette(self.idlePalette)
        self.update()
        return super().leaveEvent(event)


class expandCollapseButton(PushButton):
    sigClicked = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setIcon(QIcon(":expand.svg"))
        self.setFlat(True)
        self.installEventFilter(self)
        self.isExpand = True
        self.clicked.connect(self.buttonClicked)

    def buttonClicked(self, checked=False):
        if self.isExpand:
            self.setIcon(QIcon(":collapse.svg"))
            self.isExpand = False
            if self.text():
                self.setText(self.text().replace('Hide', 'Show'))
        else:
            self.setIcon(QIcon(":expand.svg"))
            self.isExpand = True
            if self.text():
                self.setText(self.text().replace('Show', 'Hide'))
        self.sigClicked.emit()

    def eventFilter(self, object, event):
        if event.type() == QtScoped.QEventTypeAttribute('HoverEnter'):
            self.setFlat(False)
        elif event.type() == QtScoped.QEventTypeAttribute('HoverLeave'):
            self.setFlat(True)
        return False


class ToggleVisibilityButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFlat(True)
        # self.setCheckable(True)
        self._state = False
        self.setIcon(QIcon(':unchecked.svg'))
        self.clicked.connect(self.onClicked)
        self.setStyleSheet("""
            QPushButton::pressed {
                background-color: none;
                border-style: none;
            }
        """)
        
    def onClicked(self):
        self._state = not self._state
        if self._state:
            self.setIcon(QIcon(':eye-checked.svg'))
        else:
            self.setIcon(QIcon(':unchecked.svg'))

class ToggleVisibilityCheckBox(QCheckBox):
    def __init__(self, *args, pixelSize=24):
        super().__init__(*args)
        self._pixelSize = pixelSize
        self.onToggled(False)
        self.toggled.connect(self.onToggled)
        
    def setPixelSize(self, pixelSize):
        self._pixelSize = pixelSize
        
    def onToggled(self, checked):
        if checked:
            self.setStyleSheet(f"""
                QCheckBox::indicator {{
                    width: {self._pixelSize}px;
                    height: {self._pixelSize}px;
                }}

                QCheckBox::indicator:checked
                {{
                    image: url(:eye-checked.svg);
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QCheckBox::indicator {{
                    width: {self._pixelSize}px;
                    height: {self._pixelSize}px;
                }}
                
                QCheckBox::indicator:unchecked
                {{
                    image: url(:unchecked.svg);
                }}
            """)


class myColorButton(pg.ColorButton):
    # Cancel in the color dialog, after the preview (`sigColorChanging`) of
    # the color it started from. Connect here, not to `colorDialog.rejected`:
    # `selectColor` may replace the dialog (`_make_dialog`)
    sigColorRejected = Signal(object)

    def __init__(self, parent=None, color=(128,128,128), padding=5):
        super().__init__(parent=parent, color=color)
        if isinstance(padding, (int, float)):
            self.padding = (padding, padding, -padding, -padding)  
        else:
            self.padding = padding
        self._c = 225
        self._hoverDeltaC = 30
        self._alpha = 100
        self._bkgrColor = QColor(self._c, self._c, self._c, self._alpha) 
        self._borderColor = QColor(171, 171, 171)      
        self._rectBorderPen = QPen(QBrush(QColor(0,0,0)), 0.3)
   
    def paintEvent(self, event):
        # QPushButton.paintEvent(self, ev)
        p = QStylePainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        p.setBrush(QBrush(self._bkgrColor))
        p.setPen(QPen(self._borderColor))
        p.drawRoundedRect(rect, 5, 5)
        # p.fillRect(self.rect(), self._bkgrColor)
        rect = self.rect().adjusted(*self.padding)
        ## draw white base, then texture for indicating transparency, then actual color
        p.setBrush(pg.mkBrush('w'))
        p.drawRect(rect)
        p.setBrush(QBrush(Qt.BrushStyle.DiagCrossPattern))
        p.drawRect(rect)
        p.setPen(self._rectBorderPen)
        p.setBrush(pg.mkBrush(self._color))
        p.drawRect(rect)
        p.end()
    
    def enterEvent(self, event):
        QApplication.instance().setOverrideCursor(Qt.PointingHandCursor)
        c = self._c + self._hoverDeltaC
        self._bkgrColor = QColor(c, c, c, self._alpha) 
        self.update()
    
    def leaveEvent(self, event):
        QApplication.instance().restoreOverrideCursor()
        c = self._c
        self._bkgrColor = QColor(c, c, c, self._alpha) 
        self.update()
        
    def _make_dialog(self):
        # Replaces pyqtgraph's parentless dialog so that it opens window-modal
        # on top of the enclosing window (e.g. a modal dialog). It must keep
        # the same connections as pg.ColorButton.__init__: live preview
        # (sigColorChanging), restore on reject and commit on accept
        # (colorSelected -> sigColorChanged).
        dlg = QColorDialog(self.window())  # parent = enclosing dialog
        dlg.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
        dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        dlg.currentColorChanged.connect(self.dialogColorChanged)
        dlg.rejected.connect(self.colorRejected)
        dlg.colorSelected.connect(self.colorSelected)
        return dlg
        
    @Slot()
    def colorRejected(self):
        # After the preview of the color the dialog started from
        super().colorRejected()
        self.sigColorRejected.emit(self)

    # edit show color dialog on click
    @Slot()
    def selectColor(self):
        # Re-parent every time, in case the button moved to another window
        if self.colorDialog is None or self.colorDialog.parent() is not self.window():
            if self.colorDialog is not None:
                self.colorDialog.deleteLater()
            self.colorDialog = self._make_dialog()

        self.origColor = self.color()
        self.colorDialog.setCurrentColor(self.color())
        self.colorDialog.open()
        self.colorDialog.raise_()
        self.colorDialog.activateWindow()


class FeatureSelectorButton(QPushButton):
    def __init__(self, text, parent=None, alignment=''):
        super().__init__(text, parent=parent)
        self._isFeatureSet = False
        self._alignment = alignment
        self.setCursor(Qt.PointingHandCursor)
    
    def setFeatureText(self, text):
        self.setText(text)
        self.setFlat(True)
        self._isFeatureSet = True
        if self._alignment:
            self.setStyleSheet(f'text-align:{self._alignment};')
    
    def enterEvent(self, event) -> None:
        if self._isFeatureSet:
            self.setFlat(False)
        return super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._isFeatureSet:
            self.setFlat(True)
        self.update()
        return super().leaveEvent(event)

    def setSizeLongestText(self, longestText):
        currentText = self.text()
        self.setText(longestText)
        w, h = self.sizeHint().width(), self.sizeHint().height()
        self.setMinimumWidth(w+10)
        # self.setMinimumHeight(h+5)
        self.setText(currentText)
