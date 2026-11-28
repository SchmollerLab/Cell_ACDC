"""Data structure setup dialogs (from apps.py): DataStructureSetupDialogue
(data_structure/data_struct.py) and
ReStructTiffDataStructureSetupDialogue
(data_structure/data_restruct.py), plus the widgets only they use.
"""
from qtpy.QtCore import QEvent, QObject
from qtpy.QtWidgets import QLineEdit
from cellacdc import QtScoped
import os
import re
import ast
import datetime
from collections import Counter
from natsort import natsorted
import pandas as pd
import json
import tifffile
from qtpy.QtGui import QRegularExpressionValidator, QPixmap
from qtpy.QtCore import Qt, QRegularExpression
from qtpy.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QButtonGroup,
    QComboBox, QGridLayout, QSpinBox, QDoubleSpinBox, QGroupBox, QRadioButton,
    QSpacerItem
)
from cellacdc import exception_handler
from cellacdc import html_utils
from cellacdc import myutils
from cellacdc import qutils
from cellacdc import widgets
from cellacdc import urls
from cellacdc.acdc_regex import is_alphanumeric_filename
from cellacdc import io
from cellacdc import path
from cellacdc import restruct_files_recipes_path
from cellacdc.shared.dialogs import base as dialogs_base
from cellacdc.shared.dialogs import generic as dialogs_generic
from cellacdc.shared.dialogs import file_selection as dialogs_file_selection


class ReadOnlyLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setReadOnly(True)
        # self.setStyleSheet(
        #     'background-color: rgba(240, 240, 240, 200);'
        # )
        self.installEventFilter(self)
    
    def eventFilter(self, a0: 'QObject', a1: 'QEvent') -> bool:
        if a1.type() == QtScoped.QEventTypeAttribute('FocusIn'):
            return True
        return super().eventFilter(a0, a1)

    def setValue(self, value):
        self.setText(str(value))
    
    def value(self, casting_func: callable = None):
        text = self.text()
        if casting_func is not None:
            return casting_func(text)
        return text


class DataStructureSetupDialogue(dialogs_base.QBaseDialog):
    def __init__(self, logger_func=print, parent=None):
        super().__init__(parent)
        
        self.logger_func = logger_func
        self.cancel = True
        self.selectedOptions = None
        self.actionsPosFoldersExisting = {
            'overwrite': False,
            'add_files': False,
            'create_new': False,
            'start_pos_n': 1
        }
        self.actionsFilesSourceFolder = {
            'files': [],
            'moveOtherFiles': False,
            'copyOtherFiles': False
        }
        
        mainLayout = QVBoxLayout()
        entriesLayout = widgets.FormLayout()

        headerText = ("""
<b>Data conversion setup</b><br><br>
<i>Choose your data structure, input and output folders, and conversion settings.</i>                     
""")
        headerLabel = QLabel(html_utils.paragraph(headerText))

        gh_href = html_utils.href_tag('GitHub page', urls.issues_url)

        row = 0
        self.howRawDataStructCombobox = widgets.ComboBox()
        options = [
            'Single microscopy file with multiple positions',
            'One or more microscopy files, one for each position',
            'One or more microscopy files, one for each channel',
            'NONE of the above'
        ]
        self.howRawDataStructCombobox.addItems(options)
        infoOptions = [
            'one file contains all positions/fields of view',
            'each file contains a separate position/field of view',
            'each file contains a separate imaging channel',
            'Your data layout is not currently supported. '
            f'Please report your setup on our {gh_href} '
            'so we can add support for it',
        ]
        listTexts = [f'<b>{o}</b>: {i}.' for o, i in zip(options, infoOptions)]
        infoText = html_utils.paragraph(f"""
Select how your raw microscopy data is organized.<br><br>
Typically, some microscopes (for example, Zeiss and Nikon systems) save an automated acquisition of multiple fields of view<br>
as a single file containing multiple positions. If fields of view were acquired manually, each position is typically saved as a separate file.<br>
{html_utils.to_list(listTexts)}
""")
        
        self.howRawDataStructFormWidget = widgets.formWidget(
            self.howRawDataStructCombobox, 
            labelTextLeft='Raw data structure',
            addInfoButton=True,
            infoTxt=infoText,
            wrapInfoTxt=False,
            labelLeftSuffix=': '
        )
        entriesLayout.addFormWidget(self.howRawDataStructFormWidget, row=row)
        self.howRawDataStructCombobox.currentTextChanged.connect(
            self.howRawDataStructChanged
        )

        row += 1
        self.sourceFolderPathControl = widgets.FolderPathControl(
            add_browse_recent=True
        )

        noteText = ("""
Microscopy files are those files that are typically generated 
by the microscope,<br>for example '.czi' (Zeiss), '.nd2' (Nikon), 
'.lif' (Leica), etc.
""")
        infoText = html_utils.paragraph(f"""
Select the folder containing the raw microscopy files to convert.<br><br>
Please make sure this folder contains <b>only microscopy files</b> and no other files or folders.<br><br>{html_utils.to_admonition(noteText)}                            
""")
        
        self.sourceFolderPathFormWidget = widgets.formWidget(
            self.sourceFolderPathControl, 
            labelTextLeft='Source folder with raw files',
            addInfoButton=True,
            addCogButton=True,
            infoTxt=infoText,
            wrapInfoTxt=False,
            labelLeftSuffix=': '
        )
        entriesLayout.addFormWidget(self.sourceFolderPathFormWidget, row=row)
        myutils.setRetainSizePolicy(self.sourceFolderPathFormWidget.cogButton)
        self.sourceFolderPathFormWidget.cogButton.hide()
        self.sourceFolderPathFormWidget.sigCogButtonClicked.connect(
            self.setupActionFilesSourceFolder
        )
        self.sourceFolderPathControl.sigValueChanged.connect(
            self.srcFolderPathSelected
        )

        row += 1
        self.dstFolderPathControl = widgets.FolderPathControl(
            add_browse_recent=True
        )

        data_structure_url_href = html_utils.href_tag(
            'documentation page', urls.data_structure_docs_url)

        infoText = html_utils.paragraph(f"""
Select the destination folder where Cell-ACDC will create the converted dataset.<br><br>
The folder will contain one folder per position (called `Position_n`, where `n` is the position number), TIFF files, and the folder/file structure required by Cell-ACDC.<br><br>
You can find more information about Cell-ACDC folder structure on our {data_structure_url_href}.                            
""")
        
        self.dstFolderPathFormWidget = widgets.formWidget(
            self.dstFolderPathControl, 
            labelTextLeft='Destination folder',
            addInfoButton=True,
            addCogButton=True,
            infoTxt=infoText,
            wrapInfoTxt=False,
            labelLeftSuffix=': '
        )
        entriesLayout.addFormWidget(self.dstFolderPathFormWidget, row=row)
        myutils.setRetainSizePolicy(self.dstFolderPathFormWidget.cogButton)
        self.dstFolderPathFormWidget.cogButton.hide()
        self.dstFolderPathFormWidget.sigCogButtonClicked.connect(
            self.setupActionPosFoldersExisting
        )
        self.dstFolderPathControl.sigValueChanged.connect(
            self.dstFolderPathSelected
        )
        
        row += 1

        infoText = html_utils.paragraph(f"""
Choose whether to load the entire position/field of view into RAM at once.<br><br>
Loading the entire position into RAM is much faster, but it requires more memory.<br><br>
Keep an eye on the ram usage and, if Cell-ACDC crashes or RAM is 
full, you can re-start<br>
the process and deactivate this option.
""")
        
        self.loadEntirePosInRamFormWidget = widgets.formWidget(
            widgets.Toggle(), 
            labelTextLeft='Load entire field of view in RAM at once',
            stretchWidget=False,
            valueGetterName='isChecked',
            addInfoButton=True,
            infoTxt=infoText,
            labelLeftSuffix=': '
        )
        entriesLayout.addFormWidget(self.loadEntirePosInRamFormWidget, row=row)

        row += 1

        important_text = ("""
            If you choose to use symbolic links, the source data file(s) cannot 
            be moved from their current location, otherwise the link will 
            be broken.
        """)
        important_admon = html_utils.to_admonition(
            important_text, admonition_type='important'
        )
        infoText = html_utils.paragraph(f"""
Choose whether to use symbolic links or not.<br><br>
Cell-ACDC can either copy the image data to TIFF or H5 files, or use 
symbolic links to the source image data.<br><br>
A symbolic link is a special type of file that acts as a pointer or alias, 
referring to another file by its path rather than its content.<br><br>
With symbolic links, <b>no image data will be copied</b> and only files 
you will generate later (e.g., segmentation data)<br>
will be created in the respective Position folders.<br>
{important_admon}<br>
""")
        
        self.useSymbolicLinksFormWidget = widgets.formWidget(
            widgets.Toggle(), 
            labelTextLeft='Use symbolic links',
            stretchWidget=False,
            valueGetterName='isChecked',
            addInfoButton=True,
            infoTxt=infoText,
            labelLeftSuffix=': '
        )
        entriesLayout.addFormWidget(self.useSymbolicLinksFormWidget, row=row)

        row += 1

        infoText = html_utils.paragraph(f"""
Choose whether to move raw microscopy files to a <code>raw_microscopy_files</code> sub-folder or not.<br><br>
If you activate this option, before the conversion process, Cell-ACDC will automatically move the raw microscopy files<br>
into a folder called <code>raw_microscopy_files</code> inside the destination folder.
""")
        
        self.moveRawMicroscopyFilesFormWidget = widgets.formWidget(
            widgets.Toggle(), 
            labelTextLeft='Move raw microscopy files',
            stretchWidget=False,
            valueGetterName='isChecked',
            addInfoButton=True,
            infoTxt=infoText,
            labelLeftSuffix=': '
        )
        entriesLayout.addFormWidget(self.moveRawMicroscopyFilesFormWidget, row=row)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(headerLabel)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(entriesLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def howRawDataStructChanged(self, *args):
        srcFolderPath = self.sourceFolderPathControl.path()
        if not srcFolderPath:
            return
            
        self.srcFolderPathSelected(srcFolderPath)

    @exception_handler
    def srcFolderPathSelected(self, srcFolderPath):
        self.actionsFilesSourceFolder = {
            'files': [],
            'moveOtherFiles': False,
            'copyOtherFiles': False,
        }
        if not os.path.isdir(srcFolderPath):
            self.sourceFolderPathControl.setInvalid(True)
            self.warnSourceFolderNotSetupCorrectly()
            return
        
        ls = natsorted(myutils.listdir(srcFolderPath))
        files = [
            filename for filename in ls
            if os.path.isfile(os.path.join(srcFolderPath, filename))
        ]
        if not files:
            self.sourceFolderPathControl.setInvalid(True)
            self.warnSelectedPathEmpty(srcFolderPath)
            return
        
        self.sourceFolderPathControl.setInvalid(False)
        myutils.addToRecentPaths(srcFolderPath)

        all_ext = [
            os.path.splitext(filename)[1] for filename in ls
            if os.path.isfile(os.path.join(srcFolderPath, filename))
        ]
        counter = Counter(all_ext)
        unique_ext = list(counter.keys())
        is_ext_unique = len(unique_ext) == 1
        most_common_ext, _ = counter.most_common(1)[0]

        if is_ext_unique:
            rawDataStruct = self.howRawDataStructCombobox.currentIndex()
            if rawDataStruct == 0 and len(files) > 1:
                files = self.warnMultipleFiles(files)
                if not files: 
                    return
            
            files = self.checkFileNames(files, srcFolderPath)
            if not files:
                return
            
            if not self.checkFilesExtensions(files):
                return
            
            self.actionsFilesSourceFolder['files'] = files
            return
        
        files = self.warnMultipleFileExtensions(
            srcFolderPath, 
            files,
            most_common_ext, 
            unique_ext
        )
        if not files: 
            return

        if self.howRawDataStructCombobox.currentIndex() == 0 and len(files) > 1:
            files = self.warnMultipleFiles(files)
            if not files:
                return
        
        files = self.checkFileNames(files, srcFolderPath)
        if not files: 
            return
        
        proceed = self.checkFilesExtensions(files)
        if not proceed:
            return

        self.actionsFilesSourceFolder['files'] = files

    def checkFileNames(self, raw_filenames, raw_src_path):
        allowed = (
            '.ome.tif',
        )
        for file in raw_filenames:
            if is_alphanumeric_filename(file, allowed=allowed):
                continue

            msg = widgets.myMessageBox(wrapText=False)
            txt = html_utils.paragraph(
                f"""
                The filename <code>{file}</code> contains <b>invalid 
                characters</b>.<br><br>
                Valid characters are letters, numbers, spaces, underscores 
                and dashes.<br><br>
                Please stop the process, <b>rename the file</b>, 
                and try again, or choose one of the options below.<br><br>
                Thank you for your patience!
                """
            )
            renameWithUnderscoresButton = widgets.editPushButton(
                'Rename file (replace invalid characters with "_")'
            )
            renameWithDashesButton = widgets.editPushButton(
                'Rename file (replace invalid characters with "-")'
            )
            msg.warning(
                self, 'Invalid filename', txt, 
                path_to_browse=raw_src_path,
                buttonsTexts=(
                    'Let me rename files myself', 
                    renameWithUnderscoresButton, 
                    renameWithDashesButton
                )
            )
            if msg.cancel:
                return []
            
            if msg.clickedButton == renameWithUnderscoresButton:
                self.logger_func(
                    'Renaming files to replace invalid characters with "_"...'
                )
                renamed_filenames = io.rename_files_replace_invalid_chars(
                    raw_filenames, raw_src_path, replacement_char='_'
                )
                return renamed_filenames
            elif msg.clickedButton == renameWithDashesButton:
                self.logger_func(
                    'Renaming files to replace invalid characters with "-"...'
                )
                renamed_filenames = io.rename_files_replace_invalid_chars(
                    raw_filenames, raw_src_path, replacement_char='-'
                )
                return renamed_filenames
            else:
                return []
            
        return raw_filenames

    def checkFilesExtensions(self, raw_filenames):
        from cellacdc.acdc_bioio_bioformats._utils import (
            get_supported_image_extensions
        )
        extensions = get_supported_image_extensions()
        not_supported_files = []
        for filename in raw_filenames:
            _, ext = os.path.splitext(filename)

            if ext not in extensions:
                not_supported_files.append(filename)
        
        if not_supported_files:
            self.warnExtensionNotSupported(filename)
            return False
        
        return True

    def warnExtensionNotSupported(self, not_supported_files):
        txt = html_utils.paragraph(f"""
            The selected source folder contains files not supported by 
            <code>bioio</code>.<br><br>
            Please, select another folder or remove these files (see below).<br><br>
            Thank you for your patience!<br><br>
            Not supported files:
        """)
        detailsText = '<br>'.join(not_supported_files)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'File not supported', txt,
            detailsText=detailsText
        )

    def warnMultipleFileExtensions(
            self, 
            srcFolderPath,
            files,
            most_common_ext, 
            unique_ext,
        ):
        if not most_common_ext:
            most_common_ext_msg = '<empty>'
        else:
            most_common_ext_msg = most_common_ext
        
        msg = widgets.myMessageBox(showCentered=False)
        txt = html_utils.paragraph(f"""
            The following folder<br>

            <copiable>{srcFolderPath}</copiable>

            contains files with different file extensions 
            (extensions detected: {unique_ext})<br><br>
            However, the most common extension is 
            <b>{most_common_ext_msg}</b>,
            do you want to proceed with
            loading only files with extension <b>{most_common_ext_msg}</b>?
            <br>
        """)
        _, yesButton, noButton = msg.warning(
            self, 'Multiple extensions detected', txt, 
            buttonsTexts=(
                'Cancel', 'Yes, load only most common', 
                'No, load all files'
            )
        )
        if msg.cancel:
            return []
        
        if msg.clickedButton == yesButton:
            files = [
                filename for filename in files
                if os.path.splitext(filename)[1] == most_common_ext
            ]
            otherExt = [
                ext for ext in unique_ext if ext != most_common_ext]
            files = self.askActionWithOtherFiles(files, otherExt)

        return files

    def askActionWithOtherFiles(self, files, otherExt):
        txt = html_utils.paragraph(f"""
            What should I do with the other files (ext: {otherExt})
            in the folder?<br><br>
            <i>NOTE: Only the files with the same basename and position number
            as the raw files will be moved or copied.</i>
        """)
        msg = widgets.myMessageBox(wrapText=False)
        leaveButton = widgets.currentPushButton('Leave them where they are')
        moveButton = widgets.movePushButton(
            'Attempt MOVING to their Position folder'
        )
        copyButton = widgets.copyPushButton(
            'Attempt COPYING to their Position folder'
        )
        msg.question(
            self, 'Action with the other files?', txt,
            buttonsTexts=(
                'Cancel', leaveButton, moveButton, copyButton
            )
        )
        if msg.cancel:
            return []
        
        self.actionsFilesSourceFolder['moveOtherFiles'] = (
            msg.clickedButton == moveButton
        )
        self.actionsFilesSourceFolder['copyOtherFiles'] = (
            msg.clickedButton == copyButton
        )

        return files

    def warnMultipleFiles(self, files):
        infoText = html_utils.paragraph(
            'You selected "Single microscopy file", '
            'but the <b>folder contains multiple files</b>.<br>'
        )
        win = dialogs_generic.QDialogCombobox(
            'Multiple microscopy files detected!', files, infoText,
            CbLabel='Select which file to load: ', parent=self,
            iconPixmap=QPixmap(':warning.svg')
        )
        win.exec_()
        if win.cancel:
            return []
        else:
            files = [win.selectedItemText]
            return files

    def warnSelectedPathEmpty(self, srcFolderPath):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            f"""
            The selected folder (see below) is either <b>empty</b> 
            or does not contain any files.<br><br>
            Please select a folder that contains raw microscopy files.<br><br>
            Thank you for your patience!
            """
        )
        self.sourceFolderPathControl.le.setToolTip(
            'Selected folder does not contain any files.\n\n'
            'Please select a folder that contains raw microscopy files.'
        )
        msg.warning(
            self, 'Empty folder', txt, 
            commands=(srcFolderPath, ),
            path_to_browse=srcFolderPath
        )

    def setupActionFilesSourceFolder(self):
        # Not needed for now since the user can simply select the 
        # folder again to trigger validation steps in srcFolderPathSelected
        pass

    def dstFolderPathSelected(self, dstFolderPath):
        pos_foldernames = myutils.get_pos_foldernames(dstFolderPath)
        if not pos_foldernames:
            self.actionsPosFoldersExisting = {
                'overwrite': False,
                'add_files': False,
                'create_new': False,
                'start_pos_n': 1,
            }
            self.dstFolderPathFormWidget.cogButton.hide()
            return
        
        self.dstFolderPathFormWidget.cogButton.show()
        self.setupActionPosFoldersExisting()

        self.blinker = qutils.QControlBlink(
            self.dstFolderPathFormWidget.cogButton, qparent=self
        )
        self.blinker.start()


    def setupActionPosFoldersExisting(self, *args):
        dstFolderPath = self.dstFolderPathControl.path()
        pos_foldernames = myutils.get_pos_foldernames(dstFolderPath)
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'The selected destination folder <b>already contains Position folders</b>.<br><br>'
            'Do you want to <b>overwrite</b> all of its content, '
            '<b>add files</b> to the existing Position folders,<br>'
            'or <b>create new</b> Position folders?'
        )
        _, overwriteButton, addFilesButton, createNewButton = msg.warning(
           self, 'Warning: existing Position folders detected!', txt,
           buttonsTexts=(
               'Cancel', 
               'Overwrite', 
               'Add image files to existing Positions', 
               widgets.newFilePushButton('Create new Position folders'),
            ),
           path_to_browse=dstFolderPath
        )
        self.actionsPosFoldersExisting = {
            'overwrite': False,
            'add_files': False,
            'create_new': False,
            'start_pos_n': 1
        }
        if msg.cancel:
            return 
        
        overwrite = overwriteButton == msg.clickedButton
        add_files = addFilesButton == msg.clickedButton
        create_new = createNewButton == msg.clickedButton
        
        start_pos_n = 1
        if create_new:
            pos_ns = [int(pos.split('_')[-1]) for pos in pos_foldernames]
            start_pos_n = max(pos_ns) + 1
        
        self.actionsPosFoldersExisting['overwrite'] = overwrite
        self.actionsPosFoldersExisting['add_files'] = add_files
        self.actionsPosFoldersExisting['create_new'] = create_new
        self.actionsPosFoldersExisting['start_pos_n'] = start_pos_n

    def warnSourceFolderNotSetupCorrectly(self):
        txt = html_utils.paragraph(f"""
            The source folder path is not valid.<br><br>
            Please select a source folder to go through the setup process.<br><br>
            Thank you for your patience!
        """)

        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Invalid source folder', txt)
    
    def warnDstFolderPathNotSelected(self):
        txt = html_utils.paragraph(f"""
            The destination folder path is not valid.<br><br>
            Please select a destination folder to go through the setup process.<br><br>
            Thank you for your patience!
        """)

        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Invalid destination folder', txt)

    def ok_cb(self):
        if self.howRawDataStructFormWidget.widget.currentIndex() != 3:
            if not self.actionsFilesSourceFolder['files']:
                self.warnSourceFolderNotSetupCorrectly()
                return
            
            if not self.dstFolderPathFormWidget.widget.path():
                self.warnDstFolderPathNotSelected()
                return
        
        self.cancel = False
        self.selectedOptions = {
            self.howRawDataStructFormWidget.labelTextLeft:
                self.howRawDataStructFormWidget.widget.currentIndex(),
            self.sourceFolderPathFormWidget.labelTextLeft:
                self.sourceFolderPathFormWidget.widget.path(),
            self.dstFolderPathFormWidget.labelTextLeft:
                self.dstFolderPathFormWidget.widget.path(),
            self.loadEntirePosInRamFormWidget.labelTextLeft:
                self.loadEntirePosInRamFormWidget.widget.isChecked(),
            self.useSymbolicLinksFormWidget.labelTextLeft:
                self.useSymbolicLinksFormWidget.widget.isChecked(),
            self.moveRawMicroscopyFilesFormWidget.labelTextLeft:
                self.moveRawMicroscopyFilesFormWidget.widget.isChecked(),
        }
        
        self.close()


class ReStructTiffDataStructureSetupDialogue(dialogs_base.QBaseDialog):
    DIMENSIONS = (
        ('channels', 'Channels', 'One file per channel', 'e.g. C or S'),
        ('timepoints', 'Timepoints', 'One file per timepoint', 'e.g. T'),
        ('z_slices', 'Z-slices', 'One file per z-slice', 'e.g. Z'),
    )

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Loose TIFF data structure setup')
        self.cancel = True
        self.selectedOptions = {}
        self.dimensionControls = {}
        self.channelEntries = []

        mainLayout = QVBoxLayout()

        infoText = html_utils.paragraph("""
            Select the source and destination folders, then specify how each
            dimension is represented in your TIFF files.<br><br>
            
            Choose <b>One file per ...</b> when that dimension is encoded in
            filenames, and enter the filename marker that comes before its
            index
            <br>(for example, <code>_z</code> in 
            <code>tile_x001_y001_z001.tif</code>).<br><br>
            
            Choose <b>In file</b> when
            the dimension is stored in the TIFF and enter its axis index<br>
            (for example, if the shape of the image is 
            <code>(T, Z, Y, X)</code>, the 
            axis index for <code>T</code> is 0, and for <code>Z</code> 
            it's 1).
            <br>
        """)
        mainLayout.addWidget(QLabel(infoText))

        pathsForm = QFormLayout()

        self.sourceFolderPathControl = widgets.FolderPathControl(
            add_browse_recent=True,
            add_open_in_file_manager=True
        )
        pathsForm.addRow('Source folder:', self.sourceFolderPathControl)

        self.dstFolderPathControl = widgets.FolderPathControl(
            add_browse_recent=True,
            add_open_in_file_manager=True
        )
        pathsForm.addRow('Destination folder:', self.dstFolderPathControl)

        self.segmFolderPathControl = widgets.FolderPathControl(
            add_browse_recent=True,
            add_open_in_file_manager=True
        )
        pathsForm.addRow('Segmentation mask folder (optional):',
                         self.segmFolderPathControl)

        self.baseNameLE = widgets.alphaNumericLineEdit()
        self.baseNameLE.setPlaceholderText('Optional')
        pathsForm.addRow('Basename (optional):', self.baseNameLE)

        pathsForm.addItem(QSpacerItem(20, 20))

        self.positionNameLE = widgets.ReadOnlyLineEdit()
        self.positionNameLE.setPlaceholderText(
            'After set up, here you can preview first position identifier'
        )
        pathsForm.addRow('Position identifier preview:', self.positionNameLE)

        self.shapeLE = widgets.ReadOnlyLineEdit()
        self.shapeLE.setPlaceholderText(
            'After set up, here you can see the shape of the first TIFF file'
        )
        pathsForm.addRow('Shape of first TIFF file:', self.shapeLE)

        mainLayout.addLayout(pathsForm)
        mainLayout.addSpacing(20)

        self.numberChannelsSpinBox = QSpinBox()
        self.numberChannelsSpinBox.setRange(1, 64)
        self.numberChannelsSpinBox.setValue(1)
        self.numberChannelsSpinBox.valueChanged.connect(
            self._rebuild_channel_rows
        )
        self.channelRowsLayout = QVBoxLayout()

        for key, title, perFileText, axisPlaceholder in self.DIMENSIONS:
            group = self._make_dimension_group(
                key, title, perFileText, axisPlaceholder
            )
            mainLayout.addWidget(group)

        mainLayout.addWidget(self._make_metadata_group())

        mainLayout.addStretch()

        self.sourceFolderPathControl.sigValueChanged.connect(
            self.srcFolderPathSelected
        )
        self.dstFolderPathControl.sigValueChanged.connect(
            self.dstFolderPathSelected
        )

        buttonsLayout = widgets.CancelOkButtonsLayout()

        loadRecipeButton = widgets.OpenFilePushButton('Load saved recipe...')
        self.loadRecipeButton = loadRecipeButton
        buttonsLayout.insertWidget(3, loadRecipeButton)
        
        saveRecipeButton = widgets.savePushButton('Save current recipe...')
        self.saveRecipeButton = saveRecipeButton
        buttonsLayout.insertWidget(3, saveRecipeButton)
        
        loadLastParametersButton = widgets.reloadPushButton(
            'Load last parameters'
        )
        self.loadLastParametersButton = loadLastParametersButton
        buttonsLayout.insertWidget(3, loadLastParametersButton)

        viewFirstImageButton = widgets.viewPushButton('View first image...')
        buttonsLayout.insertWidget(3, viewFirstImageButton)

        loadLastParametersButton.clicked.connect(self.loadLastParameters)
        loadRecipeButton.clicked.connect(self.loadRecipe)
        saveRecipeButton.clicked.connect(self.saveRecipe)
        viewFirstImageButton.clicked.connect(self.viewFirstImage)

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def _make_dimension_group(
            self, key, title, perFileText, axisPlaceholder
        ):
        groupBox = QGroupBox(title)
        groupLayout = QVBoxLayout(groupBox)

        radioLayout = QHBoxLayout()
        buttonGroup = QButtonGroup(groupBox)
        buttons = {}

        options = (
            ('not_present', 'Not present'),
            ('per_file', perFileText),
            ('in_file', 'In file'),
        )

        for mode, text in options:
            button = QRadioButton(text)
            buttonGroup.addButton(button)
            radioLayout.addWidget(button)
            buttons[mode] = button
            button.setChecked(mode == 'not_present')

        groupLayout.addLayout(radioLayout)

        detailRow = QWidget(groupBox)
        detailLayout = QHBoxLayout(detailRow)
        detailLayout.setContentsMargins(0, 0, 0, 0)

        detailLabel = QLabel()
        detailLE = QLineEdit()
        detailLayout.addWidget(detailLabel)
        detailLayout.addWidget(detailLE, stretch=1)
        groupLayout.addWidget(detailRow)

        controls = {
            'buttons': buttons,
            'detailRow': detailRow,
            'detailLabel': detailLabel,
            'detailLE': detailLE,
        }
        self.dimensionControls[key] = controls

        if key == 'channels':
            countLayout = QHBoxLayout()
            countLayout.addWidget(QLabel('Number of channels:'))
            countLayout.addWidget(self.numberChannelsSpinBox)
            countLayout.addStretch()
            groupLayout.addSpacing(10)
            groupLayout.addLayout(countLayout)
            groupLayout.addSpacing(10)

            groupLayout.addWidget(
                QLabel('Channel names (and filename identifiers, if split):')
            )
            groupLayout.addLayout(self.channelRowsLayout)

        for mode, button in buttons.items():
            button.toggled.connect(
                lambda checked, k=key, m=mode:
                    self._update_dimension_control(k, m) if checked else None
            )

        detailLE.textChanged.connect(self.updatePositionNamePreview)
        self._update_dimension_control(key, 'not_present')
        return groupBox

    def _make_metadata_group(self):
        groupBox = QGroupBox('Metadata')
        groupLayout = QGridLayout(groupBox)

        row = 0
        self.PhysicalSizeX_DSB = QDoubleSpinBox()
        self.PhysicalSizeX_DSB.setAlignment(Qt.AlignCenter)
        self.PhysicalSizeX_DSB.setMaximum(2147483647.0)
        self.PhysicalSizeX_DSB.setSingleStep(0.001)
        self.PhysicalSizeX_DSB.setDecimals(7)
        self.PhysicalSizeX_DSB.setValue(1.0)
        groupLayout.addWidget(
            QLabel('Pixel width (X):  '), row, 0, alignment=Qt.AlignRight
        )
        groupLayout.addWidget(self.PhysicalSizeX_DSB, row, 1)
        groupLayout.addWidget(QLabel('μm'), row, 2)

        row += 1
        self.PhysicalSizeY_DSB = QDoubleSpinBox()
        self.PhysicalSizeY_DSB.setAlignment(Qt.AlignCenter)
        self.PhysicalSizeY_DSB.setMaximum(2147483647.0)
        self.PhysicalSizeY_DSB.setSingleStep(0.001)
        self.PhysicalSizeY_DSB.setDecimals(7)
        self.PhysicalSizeY_DSB.setValue(1.0)
        groupLayout.addWidget(
            QLabel('Pixel height (Y):  '), row, 0, alignment=Qt.AlignRight
        )
        groupLayout.addWidget(self.PhysicalSizeY_DSB, row, 1)
        groupLayout.addWidget(QLabel('μm'), row, 2)

        row += 1
        self.PhysicalSizeZ_DSB = QDoubleSpinBox()
        self.PhysicalSizeZ_DSB.setAlignment(Qt.AlignCenter)
        self.PhysicalSizeZ_DSB.setMaximum(2147483647.0)
        self.PhysicalSizeZ_DSB.setSingleStep(0.001)
        self.PhysicalSizeZ_DSB.setDecimals(7)
        self.PhysicalSizeZ_DSB.setValue(1.0)
        groupLayout.addWidget(
            QLabel('Voxel depth (Z):  '), row, 0, alignment=Qt.AlignRight
        )
        groupLayout.addWidget(self.PhysicalSizeZ_DSB, row, 1)
        groupLayout.addWidget(QLabel('μm'), row, 2)

        row += 1
        self.TimeIncrement_DSB = widgets.FloatLineEdit(
            allowNegative=False, warningValues={1.0}
        )
        self.TimeIncrement_DSB.setValue(1.0)
        self.TimeIncrement_DSB.setMinimum(0.0)
        groupLayout.addWidget(
            QLabel('Frame interval:  '), row, 0, alignment=Qt.AlignRight
        )
        groupLayout.addWidget(self.TimeIncrement_DSB, row, 1)

        self.TimeIncrementUnit_CB = QComboBox()
        self.TimeIncrementUnit_CB.addItems(
            ['ms', 'seconds', 'minutes', 'hours']
        )
        self.TimeIncrementUnit_CB.setCurrentText('seconds')
        groupLayout.addWidget(
            self.TimeIncrementUnit_CB, row, 2, alignment=Qt.AlignLeft
        )

        groupLayout.setColumnStretch(1, 1)
        return groupBox

    def metadata(self):
        """Return the metadata entered by the user.

        Physical sizes are in μm, while the time increment is in the unit
        returned in `TimeIncrementUnit`.
        """
        return {
            'PhysicalSizeX': self.PhysicalSizeX_DSB.value(),
            'PhysicalSizeY': self.PhysicalSizeY_DSB.value(),
            'PhysicalSizeZ': self.PhysicalSizeZ_DSB.value(),
            'TimeIncrement': self.TimeIncrement_DSB.value(),
            'TimeIncrementUnit': self.TimeIncrementUnit_CB.currentText(),
        }

    def setMetadata(self, metadata):
        if not metadata:
            return

        for key in ('PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ'):
            value = metadata.get(key)
            if value is not None:
                getattr(self, f'{key}_DSB').setValue(float(value))

        TimeIncrement = metadata.get('TimeIncrement')
        if TimeIncrement is not None:
            self.TimeIncrement_DSB.setValue(float(TimeIncrement))

        TimeIncrementUnit = metadata.get('TimeIncrementUnit')
        if TimeIncrementUnit is not None:
            self.TimeIncrementUnit_CB.setCurrentText(TimeIncrementUnit)

    def metadataDataFrame(self):
        """Return the metadata as a Cell-ACDC metadata DataFrame.

        The DataFrame has index named 'Description' and a single column
        'values', like the `metadata.csv` files saved by Cell-ACDC.
        `TimeIncrement` is converted to seconds (Cell-ACDC convention).
        """
        metadata = self.metadata()
        TimeIncrementUnit = metadata['TimeIncrementUnit']
        toSecondsFactor = {
            'ms': 1e-3, 'seconds': 1.0, 'minutes': 60.0, 'hours': 3600.0
        }[TimeIncrementUnit]
        TimeIncrement = metadata['TimeIncrement']*toSecondsFactor

        values = {
            'TimeIncrement': TimeIncrement,
            'PhysicalSizeZ': metadata['PhysicalSizeZ'],
            'PhysicalSizeY': metadata['PhysicalSizeY'],
            'PhysicalSizeX': metadata['PhysicalSizeX'],
        }
        for c, (nameLE, _) in enumerate(self.channelEntries):
            values[f'channel_{c}_name'] = nameLE.text().strip()

        df = pd.DataFrame({
            'Description': list(values.keys()),
            'values': list(values.values())
        }).set_index('Description')

        return df

    def _get_dimension_mode(self, key):
        controls = self.dimensionControls[key]
        for mode, button in controls['buttons'].items():
            if button.isChecked():
                return mode
        return 'not_present'

    def _update_dimension_control(self, key, mode):
        controls = self.dimensionControls[key]
        detailRow = controls['detailRow']
        detailLabel = controls['detailLabel']
        detailLE = controls['detailLE']

        showDetail = (
            mode != 'not_present'
            and not (key == 'channels' and mode == 'per_file')
        )
        detailRow.setVisible(showDetail)
        detailLE.setEnabled(showDetail)
        detailLabel.setEnabled(showDetail)

        if mode == 'per_file':
            detailLabel.setText('Filename marker:')
            detailLE.setPlaceholderText(
                'e.g. _z; the slice/frame index follows the marker'
            )
            detailLE.setValidator(None)
        elif mode == 'in_file':
            detailLabel.setText('Axis index in TIFF:')
            validPattern = '[0-4]'
            regExpr = QRegularExpression(validPattern)
            detailLE.setValidator(QRegularExpressionValidator(regExpr))
            if key == 'channels':
                shape_eg = '(C, T, Y, X)'
                eg_text = 'C is 0, and for T it is 1'
            elif key == 'timepoints':
                shape_eg = '(C, T, Y, X)'
                eg_text = 'T is 1, and for C it is 0'
            else:
                shape_eg = '(T, Z, Y, X)'
                eg_text = 'T is 0, and for Z it is 1'
            detailLE.setPlaceholderText(
                f'e.g., if the shape is {shape_eg}, the axis index for '
                f'{eg_text}'
            )
            if key == 'channels':
                detailLE.textChanged.connect(self.updateNumberOfChannels)

        if key == 'channels':
            self.numberChannelsSpinBox.setReadOnly(mode == 'in_file')
            channelsPresent = mode != 'not_present'
            self.numberChannelsSpinBox.setEnabled(channelsPresent)
            if not channelsPresent:
                self.numberChannelsSpinBox.setValue(1)
            self._rebuild_channel_rows()
        
        self.updatePositionNamePreview()

    def _rebuild_channel_rows(self, *args):
        previousValues = [
            (nameLE.text(), tokenLE.text())
            for nameLE, tokenLE in self.channelEntries
        ]

        while self.channelRowsLayout.count():
            item = self.channelRowsLayout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.channelEntries = []
        mode = self._get_dimension_mode('channels')
        count = self.numberChannelsSpinBox.value()

        for index in range(count):
            rowWidget = QWidget()
            rowLayout = QHBoxLayout(rowWidget)
            rowLayout.setContentsMargins(0, 0, 0, 0)

            rowLayout.addWidget(QLabel(f'{index + 1}.'))

            nameLE = widgets.alphaNumericLineEdit()
            nameLE.setPlaceholderText(f'Channel {index + 1} name')
            if index < len(previousValues) and previousValues[index][0]:
                nameLE.setText(previousValues[index][0])
            else:
                nameLE.setText(f'Channel_{index + 1}')
            rowLayout.addWidget(nameLE, stretch=2)

            tokenLE = QLineEdit()
            tokenLE.setPlaceholderText(
                'Filename identifier, e.g. GFP'
            )
            if index < len(previousValues):
                tokenLE.setText(previousValues[index][1])
            tokenLE.setVisible(mode == 'per_file')
            rowLayout.addWidget(tokenLE, stretch=1)

            self.channelRowsLayout.addWidget(rowWidget)
            self.channelEntries.append((nameLE, tokenLE))

            tokenLE.textChanged.connect(self.updatePositionNamePreview)

    def updatePositionNamePreview(self, *args):
        folder = self.sourceFolderPathControl.path()
        if not folder or not os.path.isdir(folder):
            self.positionNameLE.clear()
            return

        tiffFiles = [
            name for name in myutils.listdir(folder)
            if os.path.isfile(os.path.join(folder, name))
            and name.lower().endswith(('.tif', '.tiff'))
        ]
        if not tiffFiles:
            self.positionNameLE.clear()
            return

        # Use the first TIFF in natural filename order as the preview example.
        filename = natsorted(tiffFiles)[0]
        stem = filename
        for suffix in ('.ome.tiff', '.ome.tif', '.tiff', '.tif'):
            if stem.lower().endswith(suffix):
                stem = stem[:-len(suffix)]
                break

        def removeMarker(text, marker):
            if not marker:
                return None

            matches = list(re.finditer(
                re.escape(marker) + r'\d+',
                text,
                flags=re.IGNORECASE,
            ))
            if len(matches) != 1:
                return None

            match = matches[0]
            return text[:match.start()] + text[match.end():]

        # Remove timepoint and z identifiers when those dimensions are per-file.
        for key in ('timepoints', 'z_slices'):
            if self._get_dimension_mode(key) != 'per_file':
                continue

            marker = self.dimensionControls[key]['detailLE'].text().strip()
            stem = removeMarker(stem, marker)
            if stem is None:
                self.positionNameLE.clear()
                self.positionNameLE.setToolTip(
                    f'Could not find exactly one numeric {key} marker '
                    f'in {filename!r}.'
                )
                return

        # For per-file channels, remove the matching channel filename identifier.
        if self._get_dimension_mode('channels') == 'per_file':
            matches = []
            for _, tokenLE in self.channelEntries:
                token = tokenLE.text().strip()
                if not token:
                    continue
                found = list(re.finditer(
                    re.escape(token), stem, flags=re.IGNORECASE
                ))
                if len(found) == 1:
                    matches.append(found[0])

            if len(matches) != 1:
                self.positionNameLE.clear()
                self.positionNameLE.setToolTip(
                    f'Could not identify exactly one channel token in '
                    f'{filename!r}.'
                )
                return

            match = matches[0]
            stem = stem[:match.start()] + stem[match.end():]

        positionName = re.sub(r'[_\-. ]{2,}', '_', stem).strip('_-. ')
        self.positionNameLE.setText(positionName or 'Position')
        self.positionNameLE.setToolTip(f'Preview from {filename}')

    def _warn(self, message):
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self,
            'Incomplete setup',
            html_utils.paragraph(message),
        )
    
    def warnSourceFolderNotSetupCorrectly(self):
        txt = html_utils.paragraph(f"""
            The source folder path is not valid.<br><br>
            Please select a source folder to go through the setup process.<br><br>
            Thank you for your patience!
        """)

        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Invalid source folder', txt)

    def ok_cb(self):
        sourceFolder = self.sourceFolderPathControl.path()
        destinationFolder = self.dstFolderPathControl.path()

        if not sourceFolder:
            self._warn('Select a source folder.')
            return
        if not destinationFolder:
            self._warn('Select a destination folder.')
            return
        
        if self.selectedOptions.get('files') is None:
            self.warnSourceFolderNotSetupCorrectly()
            return

        dimensionOptions = self.dimensionOptions()
        if dimensionOptions is None:
            return
        self.saveLastParameters(dimensionOptions)

        self.selectedOptions['source_folder'] = sourceFolder
        self.selectedOptions['destination_folder'] = destinationFolder
        self.selectedOptions['segmentation_mask_folder'] = (
            self.segmFolderPathControl.path()
        )
        self.selectedOptions['basename'] = (
            self.baseNameLE.text().strip() or None
        )
        self.selectedOptions['dimensions'] = dimensionOptions
        self.selectedOptions['metadata'] = self.metadata()
        self.selectedOptions['metadata_df'] = self.metadataDataFrame()

        self.cancel = False
        self.close()
    
    @exception_handler
    def srcFolderPathSelected(self, srcFolderPath):
        self.selectedOptions['files'] = None
        self.shapeLE.setText('')

        ls = natsorted(myutils.listdir(srcFolderPath))
        files = [
            filename for filename in ls
            if os.path.isfile(os.path.join(srcFolderPath, filename))
        ]
        if not files:
            msg = widgets.myMessageBox(wrapText=False)
            txt = html_utils.paragraph(
                'The following folder<br><br>'
                f'<code>{srcFolderPath}</code><br><br>'
                '<b>does not contain any file!</b><br>'
            )
            msg.addShowInFileManagerButton(srcFolderPath)
            msg.critical(
                self, 'Multiple extensions detected', txt
            )
            return []
        all_ext = [
            os.path.splitext(filename)[1] for filename in ls
            if os.path.isfile(os.path.join(srcFolderPath, filename))
        ]
        counter = Counter(all_ext)
        unique_ext = list(counter.keys())
        is_ext_unique = len(unique_ext) == 1
        most_common_ext, _ = counter.most_common(1)[0]
        if not is_ext_unique:
            msg = widgets.myMessageBox(wrapText=False)
            txt = html_utils.paragraph(
                'The following folder<br><br>'
                f'<code>{srcFolderPath}</code><br><br>'
                'contains <b>files with different file extensions</b> '
                f'(extensions detected: {unique_ext})<br><br>'
                f'However, the most common extension is <b>{most_common_ext}</b>, '
                'do you want to proceed with<br>'
                f'loading only files with extension <b>{most_common_ext}</b>?'
            )
            _, proceedWithMostCommon = msg.warning(
                self, 'Multiple extensions detected', txt,
                buttonsTexts=('Cancel', 'Yes')
            )
            if proceedWithMostCommon == msg.clickedButton:
                files = [
                    filename for filename in files
                    if os.path.splitext(filename)[1] == most_common_ext
                ]
                otherExt = [ext for ext in unique_ext if ext != most_common_ext]
            else:
                return []

        self.selectedOptions['files'] = files

        myutils.addToRecentPaths(srcFolderPath)

        self.updatePositionNamePreview()
        self.updateShapePreview()
    
    def updateShapePreview(self):
        first_file = self.selectedOptions['files'][0]
        first_tiff_filepath = os.path.join(
            self.sourceFolderPathControl.path(), first_file
        )

        with tifffile.TiffFile(first_tiff_filepath) as tif:
            if not tif.series:
                raise ValueError(f'No image series found in {path.name}.')
            series = tif.series[0]
            shape = tuple(series.shape)
        
        self.shapeLE.setText(str(shape))

        if self._get_dimension_mode('channels') == 'in_file':
            axis_text = self.dimensionControls['channels']['detailLE'].text()
            try:
                channel_axis = int(axis_text)
            except ValueError:
                return
            if 0 <= channel_axis < len(shape):
                self.updateNumberOfChannels(axis_text)
    
    def dstFolderPathSelected(self, dstFolderPath):
        self.selectedOptions['overwrite_destination'] = False
        self.selectedOptions['create_new_position_destination'] = False

        pos_foldernames = myutils.get_pos_foldernames(dstFolderPath)

        if not pos_foldernames:
            return
        
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'The selected destination folder <b>already contains Position folders</b>.<br><br>'
            'Do you want to <b>overwrite</b> all of its content, '
            'or <b>create new</b> Position folders?'
        )
        _, overwriteButton, createNewButton = msg.warning(
           self, 'Warning: existing Position folders detected!', txt,
           buttonsTexts=(
               'Cancel', 
               'Overwrite', 
               widgets.newFilePushButton('Create new Position folders'),
            ),
           path_to_browse=dstFolderPath
        )

        overwrite = overwriteButton == msg.clickedButton
        create_new = createNewButton == msg.clickedButton

        self.selectedOptions['overwrite_destination'] = overwrite
        self.selectedOptions['create_new_position_destination'] = create_new
    
    def viewFirstImage(self):
        if self.selectedOptions.get('files') is None:
            self.warnSourceFolderNotSetupCorrectly()
            return
        
        first_file = self.selectedOptions['files'][0]
        first_tiff_filepath = os.path.join(
            self.sourceFolderPathControl.path(), first_file
        )

        with tifffile.TiffFile(first_tiff_filepath) as tif:
            image = tif.series[0].asarray()
        
        from cellacdc import plot
        plot.imshow(image, axis_titles=[first_file])
    
    def updateNumberOfChannels(self, axis_index_str):
        if not axis_index_str:
            return
            
        axis_index = int(axis_index_str)
        shape_str = self.shapeLE.text()
        try:
            shape = ast.literal_eval(shape_str)
        except Exception as err:
            return 

        if not 0 <= axis_index < len(shape):
            return

        num_channels = shape[axis_index]
        self.numberChannelsSpinBox.setValue(num_channels)
    
    def saveRecipe(self):
        os.makedirs(restruct_files_recipes_path, exist_ok=True)
        folder_content = myutils.listdir(restruct_files_recipes_path)
        num_recipes = len(folder_content)
        default_text = f'{num_recipes + 1}'

        win = dialogs_file_selection.filenameDialog(
            title='Save recipe',
            basename='datastruct_restructure_recipe',
            ext='.json',
            allowEmpty=False,
            defaultEntry=default_text,
            hintText='Insert a <b>filename</b> for the current recipe:',
            parent=self,
        )
        win.exec_()
        if win.cancel:
            return 

        json_filepath = os.path.join(restruct_files_recipes_path, win.filename)
        dimensionOptions = self.dimensionOptions()
        if dimensionOptions is None:
            return

        recipe = {**dimensionOptions, 'metadata': self.metadata()}
        with open(json_filepath, 'w') as f:
            json.dump(recipe, f, indent=2)
        
        self.saveRecipeButton.confirmAction()
    
    def warnNoRecipesAvailable(self):
        message = (
            'There are no recipes available.<br><br>'
            'You can save recipes by clicking on the '
            '<code>Save current recipe...</code> button.'
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self,
            'No recipes available',
            html_utils.paragraph(message),
        )

    def selectRecipeFilepath(self):
        if not os.path.exists(restruct_files_recipes_path):
            self.warnNoRecipesAvailable()
            return

        availableRecipes = []
        displayNameToFilenameMapper = {}
        for file in myutils.listdir(restruct_files_recipes_path):
            if not file.startswith('datastruct_restructure_recipe'):
                continue

            if not file.lower().endswith('.json'):
                continue

            parts = file.split(f'datastruct_restructure_recipe_', 1)
            if len(parts) != 2:
                continue

            endname = parts[1]
            filepath = os.path.join(restruct_files_recipes_path, file)
            modified_on = datetime.datetime.fromtimestamp(
                os.path.getmtime(filepath)
            ).strftime(r'%Y/%m/%d %H:%M')
            availableRecipes.append((endname, modified_on))
            displayNameToFilenameMapper[endname] = file
        
        if not availableRecipes:
            self.warnNoRecipesAvailable()
            return

        selectRecipeWin = dialogs_generic.QTreeDialog(
            availableRecipes,
            headerLabels=['Name', 'Date Modified'],
            title='Select recipe',
            infoText='Select recipe to load:<br>',
            parent=self,
            path_to_browse=restruct_files_recipes_path,
        )
        selectRecipeWin.exec_()
        if selectRecipeWin.cancel:
            return
        
        selected_endname = selectRecipeWin.selectedText
        filename = displayNameToFilenameMapper.get(selected_endname)

        return os.path.join(restruct_files_recipes_path, filename)

    def loadRecipe(self):
        filepath = self.selectRecipeFilepath()
        if filepath is None:
            return

        with open(filepath, 'r') as f:
            savedOptions  = json.load(f)
        
        self.setDimensionOptions(savedOptions)

        self.loadRecipeButton.confirmAction()
    
    def loadLastParameters(self):
        if not os.path.exists(restruct_files_recipes_path):
            self.warnNoRecipesAvailable()
            return

        if not myutils.listdir(restruct_files_recipes_path):
            self.warnNoRecipesAvailable()
            return
        
        for file in myutils.listdir(restruct_files_recipes_path):
            if file.startswith('_last_setup'):
                break
        else:
            self.warnNoRecipesAvailable()
            return
        
        filepath = os.path.join(restruct_files_recipes_path, file)
        with open(filepath, 'r') as f:
            savedOptions  = json.load(f)
        
        self.setDimensionOptions(savedOptions)

        self.loadLastParametersButton.confirmAction()
    
    def saveLastParameters(self, dimensionOptions):
        os.makedirs(restruct_files_recipes_path, exist_ok=True)
        filepath = os.path.join(restruct_files_recipes_path, '_last_setup.json')
        recipe = {**dimensionOptions, 'metadata': self.metadata()}
        with open(filepath, 'w') as f:
            json.dump(recipe, f, indent=2)

    def dimensionOptions(self):
        dimensionOptions = {}
        for key, _, _, _ in self.DIMENSIONS:
            mode = self._get_dimension_mode(key)
            controls = self.dimensionControls[key]
            detail = controls['detailLE'].text().strip()

            if mode == 'in_file' and not detail:
                self._warn(f'Enter the TIFF axis label for {key}.')
                return
            if mode == 'per_file' and key != 'channels' and not detail:
                self._warn(f'Enter the filename marker for {key}.')
                return

            dimensionOptions[key] = {'mode': mode}
            if mode == 'in_file':
                dimensionOptions[key]['axis'] = detail
            elif mode == 'per_file' and key != 'channels':
                dimensionOptions[key]['filename_marker'] = detail

        channelNames = [
            nameLE.text().strip()
            for nameLE, _ in self.channelEntries
        ]
        if any(not name for name in channelNames):
            self._warn('Enter a name for every channel.')
            return
        if len({name.casefold() for name in channelNames}) != len(channelNames):
            self._warn('Channel names must be unique.')
            return

        channelsMode = dimensionOptions['channels']['mode']
        channelOptions = []
        for nameLE, tokenLE in self.channelEntries:
            channel = {'name': nameLE.text().strip()}
            if channelsMode == 'per_file':
                token = tokenLE.text().strip()
                if not token:
                    self._warn(
                        'Enter a filename identifier for every channel.'
                    )
                    return
                channel['filename_identifier'] = token
            channelOptions.append(channel)

        dimensionOptions['channels']['count'] = len(channelOptions)
        dimensionOptions['channels']['names'] = channelOptions

        return dimensionOptions
    
    def setDimensionOptions(self, options):
        for key, _, _, _ in self.DIMENSIONS:
            dimension = options.get(key, {'mode': 'not_present'})
            mode = dimension.get('mode', 'not_present')

            controls = self.dimensionControls[key]
            button = controls['buttons'].get(mode)
            if button is None:
                raise ValueError(f'Unknown mode {mode!r} for {key}')

            button.setChecked(True)
            self._update_dimension_control(key, mode)

            if mode == 'in_file':
                value = dimension.get(
                    'axis_index',
                    dimension.get('axis', ''),
                )
            elif mode == 'per_file' and key != 'channels':
                value = dimension.get('filename_marker', '')
            else:
                value = ''

            controls['detailLE'].setText(
                '' if value is None else str(value)
            )

        channels = options.get('channels', {})
        channelOptions = channels.get('names', [])
        channelCount = channels.get('count', len(channelOptions) or 1)

        if (
            self._get_dimension_mode('channels') == 'not_present'
            and channelCount != 1
        ):
            raise ValueError(
                'A channel configuration marked "Not present" must have '
                'exactly one channel.'
            )

        self.numberChannelsSpinBox.setValue(channelCount)
        self._rebuild_channel_rows()

        for (nameLE, tokenLE), channel in zip(
                self.channelEntries, channelOptions
            ):
            nameLE.setText(channel.get('name', ''))
            tokenLE.setText(channel.get('filename_identifier', ''))

        self.setMetadata(options.get('metadata'))

        self.updatePositionNamePreview()
