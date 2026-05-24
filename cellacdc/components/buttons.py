import os
from functools import partial

from qtpy.QtCore import (
    QEvent,
    QTimer,
    Qt,
    QUrl,
    QSize,
)
from qtpy.QtGui import (
    QBrush,
    QIcon,
    QLinearGradient,
    QPainter,
    QPixmap,
)
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QWidgetAction,
)

from .. import myutils

class PushButton(QPushButton):
    def __init__(
        self, *args, icon=None, alignIconLeft=False, flat=False, hoverable=False
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
        if event.type() == QEvent.Type.HoverEnter:
            self.setFlat(False)
        elif event.type() == QEvent.Type.HoverLeave:
            self.setFlat(True)
        return False

    def show(self):
        text = self.text()
        if not self.alignIconLeft:
            super().show()
            return

        self._text = text
        self.setStyleSheet("text-align:left;")
        self.setLayout(QGridLayout())
        textLabel = QLabel(self._text)
        textLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        textLabel.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._layout().addWidget(textLabel)
        super().show()

    def confirmAction(self):
        self.baseIcon = self.icon()
        self.setIcon(QIcon(":greenTick.svg"))
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
        self.setIcon(QIcon(":fork_lift.svg"))


class mergePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":merge-IDs.svg"))


class okPushButton(PushButton):
    def __init__(self, *args, isDefault=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":yesGray.svg"))
        if isDefault:
            self.setDefault(True)
        # QShortcut(Qt.Key_Return, self, self.click)
        # QShortcut(Qt.Key_Enter, self, self.click)


class MagnifyingGlassPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":magnGlass.svg"))


class MagnifyingGlassAllPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":magnGlass_all.svg"))


class AssignNewIDButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":assign_new_id.svg"))


class LockPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":lock.svg"))
        self.toggled.connect(self.onToggled)

    def onToggled(self, checked):
        if not self.isCheckable():
            return

        if checked:
            self.setIcon(QIcon(":lock_closed.svg"))
        else:
            self.setIcon(QIcon(":lock_open.svg"))

    def setCheckable(self, checkable: bool):
        super().setCheckable(checkable)
        if checkable:
            self.setIcon(QIcon(":lock_open.svg"))
        else:
            self.setIcon(QIcon(":lock.svg"))


class SkipPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":skip_arrow.svg"))


class BedPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":bed.svg"))


class BedPlusLabelPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":bed_plus_label.svg"))
        iconH = self.iconSize().height()
        iconW = int(iconH * 2.5)
        self.setIconSize(QSize(iconW, iconH))


class NoBedPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":no_bed.svg"))


class NavigatePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":navigate.svg"))


class SwitchPlaneButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":switch_2d_plane.svg"))
        self._planes = ("xy", "zy", "zx")
        self._idx = 0

    def switchPlane(self):
        self._idx += 1

    def setPlane(self, plane):
        self._idx = self._planes.index(plane)

    def plane(self):
        return self._planes[self._idx % 3]

    def depthAxes(self):
        plane = self.plane()
        for axes in "xyz":
            if axes not in plane:
                return axes


class zoomPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":zoom_out.svg"))

    def setIconZoomOut(self):
        self.setIcon(QIcon(":zoom_out.svg"))

    def setIconZoomIn(self):
        self.setIcon(QIcon(":zoom_in.svg"))


class WarningButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":warning.svg"))


class reloadPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":reload.svg"))


class savePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":file-save.svg"))


class autoPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":cog_play.svg"))


class newFilePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":file-new.svg"))


class helpPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":help.svg"))


class viewPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":eye.svg"))


class infoPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":info.svg"))


class threeDPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":3d.svg"))


class twoDPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":2d.svg"))


class addPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":add.svg"))


class futurePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":arrow_future.svg"))


class FutureAllPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":arrow_future_all.svg"))


class currentPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":arrow_current.svg"))


class arrowUpPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        alignIconLeft = kwargs.get("alignIconLeft", False)
        super().__init__(
            *args, icon=QIcon(":arrow-up.svg"), alignIconLeft=alignIconLeft
        )


class arrowDownPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":arrow-down.svg"))


class selectAllPushButton(PushButton):
    sigClicked = Signal(object, bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._status = "deselect"
        self.setIcon(QIcon(":deselect_all.svg"))
        self.setText("Deselect all")
        self.clicked.connect(self.onClicked)
        self.setMinimumWidth(self.sizeHint().width())

    def setChecked(self, checked):
        if checked:
            self._status == "deselect"
        else:
            self._status == "select"
        self.click()

    def onClicked(self):
        if self._status == "select":
            icon_fn = ":deselect_all.svg"
            self._status = "deselect"
            checked = True
            text = "Deselect all"
        else:
            icon_fn = ":select_all.svg"
            text = "Select all"
            self._status = "select"
            checked = False
        self.setIcon(QIcon(icon_fn))
        self.setText(text)
        self.sigClicked.emit(self, checked)


class subtractPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":subtract.svg"))


class continuePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":continue.svg"))


class calcPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":calc.svg"))


class playPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":play.svg"))


class stopPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":stop.svg"))


class copyPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":edit-copy.svg"))
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

        super().setText("Copied!")
        self.setIcon(QIcon(":greenTick.svg"))
        QTimer.singleShot(2000, self.resetButton)

    def resetButton(self):
        self.setText(self._original_text)
        self.setIcon(QIcon(":edit-copy.svg"))


class OpenFilePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":folder-open.svg"))


class movePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":folder-move.svg"))


class DownloadPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":download.svg"))


class showInFileManagerButton(PushButton):
    def __init__(self, *args, setDefaultText=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":drawer.svg"))
        self._path_to_browse = None
        if setDefaultText:
            self.setDefaultText()

    def setDefaultText(self):
        self._text = myutils.get_show_in_file_manager_text()
        self.setText(self._text)

    def setPathToBrowse(self, path: os.PathLike):
        self._path_to_browse = path
        self.clicked.connect(partial(myutils.showInExplorer, path))


class OpenUrlButton(PushButton):
    def __init__(self, url, *args, **kwargs):
        self._url = url
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":browser.svg"))
        self.clicked.connect(self.openUrl)

    def openUrl(self):
        QDesktopServices.openUrl(QUrl(self._url))


class LessThanPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":less_than.svg"))
        flat = kwargs.get("flat")
        if flat is not None:
            self.setFlat(True)


class showDetailsButton(PushButton):
    sigToggled = Signal(bool)

    def __init__(self, *args, txt="Show details...", parent=None):
        super().__init__(txt, parent)
        # self.setText(txt)
        self.txt = txt
        self.checkedIcon = QIcon(":hideUp.svg")
        self.uncheckedIcon = QIcon(":showDown.svg")
        self.setIcon(self.uncheckedIcon)
        self.toggled.connect(self.onClicked)
        self.setCheckable(True)
        w = self.sizeHint().width() + 10
        self.setFixedWidth(w)

    def onClicked(self, checked):
        if checked:
            self.setText(self.txt.replace("Show", "Hide"))
            self.setIcon(self.checkedIcon)
        else:
            self.setText(self.txt)
            self.setIcon(self.uncheckedIcon)

        self.sigToggled.emit(checked)


class cancelPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":cancelButton.svg"))


class setPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":cog.svg"))


class TrainPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":train.svg"))


class noPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":no.svg"))


class editPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":edit-id.svg"))


class delPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":bin.svg"))


class eraserPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":eraser.svg"))


class CrossCursorPointButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":cross_cursor.svg"))


class TestPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":test.svg"))


class browseFileButton(PushButton):
    sigPathSelected = Signal(str)

    def __init__(
        self,
        *args,
        ext=None,
        title="Select file",
        start_dir="",
        openFolder=False,
        **kwargs,
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
        """
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":folder-open.svg"))
        self.clicked.connect(self.browse)

        self._title = title
        self._start_dir = start_dir
        self._openFolder = openFolder
        self._file_types = "All Files (*)"
        if ext is not None:
            s_li = []
            for name, extensions in ext.items():
                _s = ""
                if isinstance(extensions, str):
                    extensions = [extensions]
                for ext in extensions:
                    _s = f"{_s}*{ext} "
                s_li.append(f"{name} {_s.strip()}")

            self._file_types = ";;".join(s_li)
            self._file_types = f"{self._file_types};;All Files (*)"

    def setStartPath(self, start_path):
        self._start_dir = start_path

    def browse(self):
        if self._openFolder:
            fileDialog = QFileDialog.getExistingDirectory
            args = (self, self._title, self._start_dir)
        else:
            fileDialog = QFileDialog.getOpenFileName
            args = (self, self._title, self._start_dir, self._file_types)
        file_path = fileDialog(*args)
        if not isinstance(file_path, str):
            file_path = file_path[0]
        if file_path:
            self.sigPathSelected.emit(file_path)


def getPushButton(buttonText, qparent=None):
    isCancelButton = (
        buttonText.lower().find("cancel") != -1
        or buttonText.lower().find("abort") != -1
    )
    isYesButton = (
        buttonText.lower().find("yes") != -1
        or buttonText.lower().find("ok") != -1
        or buttonText.lower().find("continue") != -1
        or buttonText.lower().find("recommended") != -1
    )
    isSettingsButton = buttonText.lower().find("set") != -1
    isNoButton = (
        buttonText.replace(" ", "").lower() == "no"
        or buttonText.lower().find("Do not ") != -1
        or buttonText.lower().find("no, ") != -1
    )
    isDelButton = buttonText.lower().find("delete") != -1
    isAddButton = buttonText.lower().find("add ") != -1
    is3Dbutton = buttonText.find(" 3D ") != -1
    is2Dbutton = buttonText.find(" 2D ") != -1
    isSaveButton = buttonText.lower().find("overwrite") != -1
    isNewFileButton = buttonText.lower().find("rename") != -1
    isTryAgainButton = buttonText.lower().find("try again") != -1

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


def CustomGradientMenuAction(gradient: QLinearGradient, name: str, parent):
    pixmap = QPixmap(100, 15)
    painter = QPainter(pixmap)
    brush = QBrush(gradient)
    painter.fillRect(QRect(0, 0, 100, 15), brush)
    painter.end()
    label = QLabel()
    label.setPixmap(pixmap)
    label.setContentsMargins(1, 1, 1, 1)
    labelName = QLabel(name)
    hbox = QHBoxLayout()
    delButton = delPushButton()
    hbox.addWidget(labelName)
    hbox.addStretch(1)
    hbox.addWidget(label)
    hbox.addWidget(delButton)
    widget = QWidget()
    widget.setLayout(hbox)
    action = QWidgetAction(parent)
    action.name = name
    action.setDefaultWidget(widget)
    action.delButton = delButton
    delButton.action = action
    return action
