"""Qt view adapter for data loading and recovery workflows."""

from __future__ import annotations

import os
import shutil
import zipfile
from functools import partial

import numpy as np
import pandas as pd
import psutil
import skimage
from datetime import datetime
import cv2
import skimage.color
import skimage.io
from natsort import natsorted
from qtpy.QtCore import QEventLoop, QMutex, Qt, QThread, QTimer, QWaitCondition
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QFileDialog, QPushButton

from cellacdc import (
    _palettes,
    apps,
    autopilot,
    dataPrep,
    data_structure_docs_url,
    exception_handler,
    html_utils,
    load,
    myutils,
    prompts,
    user_manual_url,
    widgets,
    workers,
)

GREEN_HEX = _palettes.green()


class DataLoadingMixin:
    """Qt-facing adapter for data loading and recovery workflows."""

    """Headless data-loading rules and path plans."""

    @exception_handler
    def _createEmptyData(self):
        self.MostRecentPath = self.getMostRecentPath()
        exp_path = QFileDialog.getExistingDirectory(
            self,
            "Select experiment folder where to create empty data",
            self.MostRecentPath,
        )
        if not exp_path:
            return

        pos_path = os.path.join(exp_path, "Position_1")
        images_path = os.path.join(pos_path, "Images")
        if os.path.exists(images_path):
            raise FileExistsError(f'The following path already exists "{images_path}"')

        os.makedirs(images_path, exist_ok=True)

        basename = "test_empty_"
        tif_filename = f"{basename}channel_1.tif"
        tif_filepath = os.path.join(images_path, tif_filename)
        empty_img = np.zeros((256, 256), dtype=np.uint8)
        empty_img[0, 0] = 255
        skimage.io.imsave(tif_filepath, empty_img)

        metadata_filename = f"{basename}metadata.csv"
        metadata_filepath = os.path.join(images_path, metadata_filename)
        df_metadata = pd.DataFrame({"Description": ["basename"], "values": [basename]})
        df_metadata.to_csv(metadata_filepath, index=False)

        self.isNewFile = True
        self._openFolder(exp_path=images_path)

    def _loadFromExperimentFolder(self, exp_path):
        select_folder = load.select_exp_folder()
        values = select_folder.get_values_segmGUI(exp_path)
        if not values:
            self.criticalInvalidPosFolder(exp_path)
            self.openFolderAction.setEnabled(True)
            return []

        if len(values) > 1:
            select_folder.QtPrompt(self, values, allow_cancel=False)
            if select_folder.cancel:
                return []
        else:
            select_folder.cancel = False
            select_folder.selected_pos = select_folder.pos_foldernames

        images_paths = []
        for pos in select_folder.selected_pos:
            images_paths.append(os.path.join(exp_path, pos, "Images"))
        return images_paths

    @exception_handler
    def _openFile(self, file_path=None):
        """
        Function used for loading an image file directly.
        """
        if file_path is None:
            self.MostRecentPath = self.getMostRecentPath()
            file_path = QFileDialog.getOpenFileName(
                self,
                "Select image file",
                self.MostRecentPath,
                "Image/Video Files (*.png *.tif *.tiff *.jpg *.jpeg *.mov *.avi *.mp4)"
                ";;All Files (*)",
            )[0]
            if not file_path:
                return

        filename, ext = os.path.splitext(os.path.basename(file_path))
        ext = ext.lower()
        dirpath = os.path.dirname(file_path)
        dirname = os.path.basename(dirpath)
        filename = filename.rstrip("_")
        channel_name = None
        do_copy = True
        if dirname != "Images":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            acdc_folder = f"{timestamp}_acdc"
            exp_path = os.path.join(dirpath, acdc_folder, "Images")
            proceed, do_copy = self.warnUserCreationImagesFolder(exp_path, ext)
            if not proceed:
                self.logger.info("Loading image file cancelled.")
                return

            proceed, channel_name = self.askUserChannelName(filename, ".tif")
            if not proceed:
                self.logger.info("Loading image file cancelled.")
                return

            os.makedirs(exp_path, exist_ok=True)
        else:
            exp_path = dirpath

        if channel_name is not None:
            # Check if user wants to use the existing channel name
            underscore_splits = filename.split("_")
            if len(underscore_splits) > 1:
                default_ch_name = underscore_splits[-1]
                if channel_name == default_ch_name:
                    filename = "_".join(underscore_splits[:-1])

            basename = f"{filename}_"
            new_filename = f"{filename}_{channel_name}{ext}"
            df_metadata = pd.DataFrame(
                {"Description": ["basename"], "values": [basename]}
            )
            metadata_csv_filename = f"{basename}metadata.csv"
            metadata_csv_filepath = os.path.join(exp_path, metadata_csv_filename)
            df_metadata.to_csv(metadata_csv_filepath, index=False)
        else:
            new_filename = f"{filename}{ext}"

        if do_copy:
            action_text = "Copying"
        else:
            action_text = "Moving"

        if ext == ".tif" or ext == ".npz":
            new_filepath = os.path.join(exp_path, new_filename)
            if not os.path.exists(new_filepath):
                self.logger.info(f"{action_text} file to Images folder...")
                if do_copy:
                    shutil.copy2(file_path, new_filepath)
                else:
                    shutil.move(file_path, new_filepath)
            self._openFolder(exp_path=exp_path, imageFilePath=new_filepath)
        else:
            self.logger.info(f"{action_text} file to .tif format...")
            data = load.loadData(file_path, "", log_func=self.logger.info)
            data.loadImgData()
            img = data.img_data
            if img.ndim == 3 and (img.shape[-1] == 3 or img.shape[-1] == 4):
                self.logger.info("Converting RGB image to grayscale...")
                if img.shape[-1] == 3:
                    data.img_data = skimage.color.rgb2gray(data.img_data)
                else:
                    data.img_data = cv2.cvtColor(data.img_data, cv2.COLOR_RGBA2GRAY)
                data.img_data = skimage.img_as_ubyte(data.img_data)
            new_filename_no_ext, ext = os.path.splitext(new_filename)
            tif_filename = f"{new_filename_no_ext}.tif"
            tif_path = os.path.join(exp_path, tif_filename)
            if data.img_data.ndim == 3:
                data.img_data.shape[0]
            elif data.img_data.ndim == 4:
                data.img_data.shape[0]
                data.img_data.shape[1]
            else:
                pass
            is_imageJ_dtype = (
                data.img_data.dtype == np.uint8
                or data.img_data.dtype == np.uint32
                or data.img_data.dtype == np.uint32
                or data.img_data.dtype == np.float32
            )
            if not is_imageJ_dtype:
                data.img_data = skimage.img_as_ubyte(data.img_data)

            myutils.to_tiff(tif_path, data.img_data)
            self._openFolder(exp_path=exp_path, imageFilePath=tif_path)

    @exception_handler
    def _openFolder(self, checked=False, exp_path=None, imageFilePath=""):
        """Main function to load data.

        Parameters
        ----------
        checked : bool
            kwarg needed because openFolder can be called by openFolderAction.
        exp_path : string or None
            Path selected by the user either directly, through openFile,
            or drag and drop image file.
        imageFilePath : string
            Path of the image file that was either drag and dropped or opened
            from File --> Open image/video file (openFileAction).

        Returns
        -------
            None
        """

        if exp_path is None:
            self.MostRecentPath = self.getMostRecentPath()
            exp_path = QFileDialog.getExistingDirectory(
                self,
                "Select experiment folder containing Position_n folders "
                "or specific Position_n folder",
                self.MostRecentPath,
            )

        if not exp_path:
            self.openFolderAction.setEnabled(True)
            return

        proceed = self.reInitGui()
        if not proceed:
            self.openFolderAction.setEnabled(True)
            return

        self.openFolderAction.setEnabled(False)

        if self.slideshowWin is not None:
            self.slideshowWin.close()

        if self.ccaTableWin is not None:
            self.ccaTableWin.close()

        self.exp_path = exp_path
        self.logger.info(f"Loading from {self.exp_path}")
        self.addToRecentPaths(exp_path, logger=self.logger)
        self.addPathToOpenRecentMenu(exp_path)

        folder_type = myutils.determine_folder_type(exp_path)
        is_pos_folder, is_images_folder, exp_path = folder_type

        self.titleLabel.setText("Loading data...", color=self.titleColor)

        skip_channels = []
        ch_name_selector = prompts.select_channel_name(
            which_channel="segm", allow_abort=False
        )
        user_ch_name = None
        if not is_pos_folder and not is_images_folder and not imageFilePath:
            images_paths = self._loadFromExperimentFolder(exp_path)
            if not images_paths:
                self.loadingDataAborted()
                return

        elif is_pos_folder and not imageFilePath:
            pos_foldername = os.path.basename(exp_path)
            exp_path = os.path.dirname(exp_path)
            images_paths = [os.path.join(exp_path, pos_foldername, "Images")]

        elif is_images_folder and not imageFilePath:
            images_paths = [exp_path]
            pos_path = os.path.dirname(exp_path)
            exp_path = os.path.dirname(pos_path)

        elif imageFilePath:
            # images_path = exp_path because called by openFile func
            filenames = myutils.listdir(exp_path)
            ch_names, basenameNotFound = ch_name_selector.get_available_channels(
                filenames, exp_path
            )
            filename = os.path.basename(imageFilePath)
            self.ch_names = ch_names
            user_ch_name = [
                chName for chName in ch_names if filename.find(chName) != -1
            ][0]
            images_paths = [exp_path]
            pos_path = os.path.dirname(exp_path)
            exp_path = os.path.dirname(pos_path)

        self.images_paths = images_paths

        # Get info from first position selected
        images_path = self.images_paths[0]
        filenames = myutils.listdir(images_path)
        if ch_name_selector.is_first_call and user_ch_name is None:
            ch_names, _ = ch_name_selector.get_available_channels(
                filenames, images_path
            )
            self.ch_names = ch_names
            if not ch_names:
                self.openFolderAction.setEnabled(True)
                self.criticalNoTifFound(images_path)
                return
            if len(ch_names) > 1:
                CbLabel = "Select channel name to load: "
                ch_name_selector.QtPrompt(self, ch_names, CbLabel=CbLabel)
                if ch_name_selector.was_aborted:
                    self.openFolderAction.setEnabled(True)
                    return
                skip_channels.extend(
                    [ch for ch in ch_names if ch != ch_name_selector.channel_name]
                )
            else:
                ch_name_selector.channel_name = ch_names[0]
            ch_name_selector.setUserChannelName()
            user_ch_name = ch_name_selector.user_ch_name
        else:
            # File opened directly with self.openFile
            ch_name_selector.channel_name = user_ch_name

        user_ch_file_paths = []
        not_allowed_ends = ["btrack_tracks.h5"]
        for images_path in self.images_paths:
            channel_file_path = load.get_filename_from_channel(
                images_path,
                user_ch_name,
                skip_channels=skip_channels,
                not_allowed_ends=not_allowed_ends,
                logger=self.logger.info,
            )
            if not channel_file_path:
                self.criticalImgPathNotFound(images_path)
                return
            user_ch_file_paths.append(channel_file_path)

        ch_name_selector.setUserChannelName()
        self.user_ch_name = user_ch_name
        self.img1.channelName = user_ch_name

        self.AutoPilotProfile.storeSelectedChannel(self.user_ch_name)

        self.initGlobalAttr()
        self.createOverlayContextMenu()
        self.createUserChannelNameAction()
        self.gui_createOverlayColors()
        self.gui_createOverlayItems()
        lastRow = self.bottomLeftLayout.rowCount()
        self.bottomLeftLayout.setRowStretch(lastRow + 1, 1)

        self.num_pos = len(user_ch_file_paths)
        proceed = self.loadSelectedData(user_ch_file_paths, user_ch_name)
        if not proceed:
            self.openFolderAction.setEnabled(True)
            return

    def _workerDebug(self, stuff_to_debug):
        pass

    def addToRecentPaths(self, path, logger=None):
        myutils.addToRecentPaths(path, logger=self.logger)

    def askMismatchSegmDataShape(self, posData):
        msg = widgets.myMessageBox(wrapText=False)
        title = "Segm. data shape mismatch"
        f = "3D" if self.isSegm3D else "2D"
        f = f"{f} over time" if posData.SizeT > 1 else f
        r = "2D" if self.isSegm3D else "3D"
        r = f"{r} over time" if posData.SizeT > 1 else r
        text = html_utils.paragraph(f"""
            The segmentation masks of the first Position that you loaded is 
            <b>{f}</b>,<br>
            while {posData.pos_foldername} is <b>{r}</b>.<br><br>
            The loaded segmentation masks <b>must be</b> either <b>all 3D</b> 
            or <b>all 2D</b>.<br><br>
            Do you want to skip loading this position or cancel the process?
        """)
        _, skipPosButton = msg.warning(
            self, title, text, buttonsTexts=("Cancel", "Skip this Position")
        )
        if skipPosButton == msg.clickedButton:
            self.loadDataWorker.skipPos = True
        self.loadDataWorker.waitCond.wakeAll()

    def askRecoverNotSavedData(self, posData):
        last_modified_time_unsaved = "NEVER"
        if os.path.exists(posData.segm_npz_temp_path):
            if os.path.exists(posData.segm_npz_path):
                last_modified_time_unsaved = datetime.fromtimestamp(
                    os.path.getmtime(posData.segm_npz_path)
                ).strftime("%a %d. %b. %y - %H:%M:%S")
        else:
            posData.setTempPaths()
            if os.path.exists(posData.unsaved_acdc_df_autosave_path):
                zip_path = posData.unsaved_acdc_df_autosave_path
                with zipfile.ZipFile(zip_path, mode="r") as zip:
                    csv_names = natsorted(set(zip.namelist()))
                iso_key = csv_names[-1][:-4]
                most_recent_unsaved_acdc_df_datetime = datetime.strptime(
                    iso_key, load.ISO_TIMESTAMP_FORMAT
                )
                last_modified_time_unsaved = (
                    most_recent_unsaved_acdc_df_datetime
                ).strftime("%a %d. %b. %y - %H:%M:%S")

        if os.path.exists(posData.acdc_output_csv_path):
            acdc_df_mtime = os.path.getmtime(posData.acdc_output_csv_path)
            timestamp = datetime.fromtimestamp(acdc_df_mtime)
            last_modified_time_saved = timestamp.strftime("%a %d. %b. %y - %H:%M:%S")
        else:
            last_modified_time_saved = "Null"

        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph("""
            Cell-ACDC detected <b>unsaved data</b>.<br><br>
            Do you want to <b>load and recover</b> the unsaved data or 
            load the data that was <b>last saved by the user</b>?
        """)
        details = f"""
            The unsaved data was created on {last_modified_time_unsaved}\n\n
            The user saved the data last time on {last_modified_time_saved}
        """
        msg.setDetailedText(details)
        loadUnsavedButton = widgets.reloadPushButton("Recover unsaved data")
        loadSavedButton = widgets.savePushButton("Load saved data")
        infoButton = widgets.infoPushButton("More info...")
        loadSafeNpzButton = ""
        if posData.isSafeNpzOverwritePresent():
            loadSafeNpzButton = widgets.reloadPushButton(
                "Load .safe.npz file from crash"
            )
            buttons = (
                loadSavedButton,
                loadUnsavedButton,
                loadSafeNpzButton,
                infoButton,
            )
        else:
            buttons = (loadSavedButton, loadUnsavedButton, infoButton)
        msg.question(
            self.progressWin,
            "Recover unsaved data?",
            txt,
            buttonsTexts=("Cancel", *buttons),
            showDialog=False,
        )
        infoButton.disconnect()
        infoButton.clicked.connect(partial(self.showInfoAutosave, posData))
        msg.exec_()
        if msg.cancel:
            self.loadDataWorker.abort = True
        elif msg.clickedButton == loadUnsavedButton:
            self.loadDataWorker.loadUnsaved = True
        elif msg.clickedButton == loadSafeNpzButton:
            self.loadDataWorker.loadSafeOverwriteNpz = True

        self.loadDataWorker.waitCond.wakeAll()

    def askUserChannelName(self, filename_no_ext, ext):
        help_txt = html_utils.paragraph("""
            Cell-ACDC requires that every image file has a basename and some 
            additional text, typically the channel name.<br><br>
            The basename will be common to all created files, while the additional text is used to identify the image files.
        """)

        basename = filename_no_ext
        underscore_splits = filename_no_ext.split("_")
        if len(underscore_splits) > 1:
            channel_name = underscore_splits[-1]
            basename = "_".join(underscore_splits[:-1])
        else:
            channel_name = "channel_1"

        txt = html_utils.paragraph("""
            Provide some text (e.g., the channel name) to append at the end of the image file.
        """)
        win = apps.filenameDialog(
            basename=basename,
            ext=ext,
            hintText=txt,
            defaultEntry=channel_name,
            helpText=help_txt,
            allowEmpty=False,
            parent=self,
            title="Provide channel name for image file",
        )
        win.exec_()
        if win.cancel:
            return False, ""

        return True, win.entryText

    def channel_name_suggestion(self, filename_no_ext: str) -> ChannelNameSuggestion:
        underscore_splits = filename_no_ext.split("_")
        if len(underscore_splits) > 1:
            return ChannelNameSuggestion(
                basename="_".join(underscore_splits[:-1]),
                channel_name=underscore_splits[-1],
            )

        return ChannelNameSuggestion(
            basename=filename_no_ext,
            channel_name="channel_1",
        )

    def checkManageVersions(self):
        posData = self.data[self.pos_i]
        posData.setTempPaths(createFolder=False)
        loaded_acdc_df_filename = os.path.basename(posData.acdc_output_csv_path)

        if os.path.exists(posData.recoveryFolderpath()):
            self.manageVersionsAction.setDisabled(False)
            self.manageVersionsAction.setToolTip(
                f"Load an older version of the `{loaded_acdc_df_filename}` file "
                "(table with annotations and measurements)."
            )
        else:
            self.manageVersionsAction.setDisabled(True)

    def checkMemoryRequirements(self, required_ram):
        memory = psutil.virtual_memory()
        total_ram = memory.total
        available_ram = memory.available
        if required_ram / available_ram > 0.3:
            proceed = self.warnMemoryNotSufficient(
                total_ram, available_ram, required_ram
            )
            return proceed
        else:
            return True

    def copy_action_text(self, do_copy: bool) -> str:
        return "Copying" if do_copy else "Moving"

    def copy_single_zslice_segm_info(
        self,
        existing_df: pd.DataFrame,
        default_dst_df: pd.DataFrame,
        *,
        src_filename: str,
        dst_filename: str,
    ) -> pd.DataFrame:
        dst_df = default_dst_df.copy()
        src_df = existing_df.loc[src_filename].copy()

        for z_info in src_df.itertuples():
            frame_i = z_info.Index
            if z_info.which_z_proj != "single z-slice":
                continue

            src_idx = (src_filename, frame_i)
            if existing_df.at[src_idx, "resegmented_in_gui"]:
                col = "z_slice_used_gui"
            else:
                col = "z_slice_used_dataPrep"

            z_slice = existing_df.at[src_idx, col]
            dst_idx = (dst_filename, frame_i)
            dst_df.at[dst_idx, "z_slice_used_dataPrep"] = z_slice
            dst_df.at[dst_idx, "z_slice_used_gui"] = z_slice

        return self.merge_default_segm_info(existing_df, dst_df)

    def criticalFluoChannelNotFound(self, fluo_ch, posData):
        msg = widgets.myMessageBox(showCentered=False)
        ls = "\n".join(myutils.listdir(posData.images_path))
        msg.setDetailedText(f"Files present in the {posData.relPath} folder:\n{ls}")
        title = "Requested channel data not found!"
        txt = html_utils.paragraph(
            f"The folder <code>{posData.pos_path}</code> "
            "<b>does not contain</b> "
            "either one of the following files:<br><br>"
            f"{posData.basename}{fluo_ch}.tif<br>"
            f"{posData.basename}{fluo_ch}_aligned.npz<br><br>"
            "Data loading aborted."
        )
        msg.addShowInFileManagerButton(posData.images_path)
        msg.warning(self, title, txt, buttonsTexts=("Ok"))

    def criticalImgPathNotFound(self, images_path):
        self.logger.info(
            "The following folder does not contain valid image files: "
            f'"{images_path}"\n\n'
            "Check that all the positions loaded contain the same channel name. "
            "Make sure to double check for spelling mistakes or types in the "
            "channel names."
        )
        msg = widgets.myMessageBox()
        msg.addShowInFileManagerButton(images_path)
        err_msg = html_utils.paragraph(f"""
            The folder<br><br>
            <code>{images_path}</code><br><br>
            <b>does not contain any valid image file!</b><br><br>
            Valid file formats are .h5, .tif, _aligned.h5, _aligned.npz.
        """)
        msg.critical(self, "No valid files found!", err_msg, buttonsTexts=("Ok",))

    def criticalInvalidPosFolder(self, exp_path):
        href = html_utils.href_tag("here", data_structure_docs_url)
        txt = html_utils.paragraph(f"""
            The selected folder:<br><br>
            
            <code>{exp_path}</code><br><br>
            
            is <b>not a valid folder</b>.<br><br>
            
            Select a folder that contains the Position_n folders, 
            or a specific Position.<br><br>
            
            If you are trying to load a single image file go to 
            <code>File --> Open image/video file...</code>.<br><br>
            
            To load a folder containing multiple .tif files the folder must 
            be called either <code>Position_n</code><br>
            (with <code>n</code> being an integer) or <code>Images</code>.<br><br>
            
            For more information about the correct folder structure see {href}.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        helpButton = widgets.helpPushButton("Help...")
        msg.addButton(helpButton)
        helpButton.clicked.disconnect()
        helpButton.clicked.connect(partial(myutils.browse_url, data_structure_docs_url))
        msg.addShowInFileManagerButton(exp_path)
        msg.critical(self, "Incompatible folder", txt)

    def criticalNoTifFound(self, images_path):
        err_title = "No .tif files found in folder."
        err_msg = html_utils.paragraph(
            "The following folder<br><br>"
            f"<code>{images_path}</code><br><br>"
            "<b>does not contain .tif or .h5 files</b>.<br><br>"
            'Only .tif or .h5 files can be loaded with "Open Folder" button.<br><br>'
            "Try with <code>File --> Open image/video file...</code> "
            "and directly select the file you want to load."
        )
        msg = widgets.myMessageBox()
        msg.addShowInFileManagerButton(images_path)
        msg.critical(self, err_title, err_msg)

    def empty_data_plan(self, exp_path: str) -> EmptyDataPlan:
        pos_path = os.path.join(exp_path, "Position_1")
        images_path = os.path.join(pos_path, "Images")
        basename = "test_empty_"
        tif_filename = f"{basename}channel_1.tif"
        metadata_filename = f"{basename}metadata.csv"

        return EmptyDataPlan(
            exp_path=exp_path,
            pos_path=pos_path,
            images_path=images_path,
            basename=basename,
            tif_filename=tif_filename,
            tif_filepath=os.path.join(images_path, tif_filename),
            metadata_filename=metadata_filename,
            metadata_filepath=os.path.join(images_path, metadata_filename),
        )

    def getFileExtensions(self, images_path):
        alignedFound = any(
            [f.find("_aligned.np") != -1 for f in myutils.listdir(images_path)]
        )
        if alignedFound:
            extensions = (
                "Aligned channels (*npz *npy);; Tif channels(*tiff *tif);;All Files (*)"
            )
        else:
            extensions = "Tif channels(*tiff *tif);; All Files (*)"
        return extensions

    def getMostRecentPath(self):
        return myutils.getMostRecentPath()

    def getPathFromChName(self, chName, posData):
        ls = myutils.listdir(posData.images_path)
        endnames = {f[len(posData.basename) :]: f for f in ls}
        validEnds = ["_aligned.npz", "_aligned.h5", ".h5", ".tif", ".npz"]
        for end in validEnds:
            files = [
                filename
                for endname, filename in endnames.items()
                if endname == f"{chName}{end}"
            ]
            if files:
                filename = files[0]
                break
        else:
            self.criticalFluoChannelNotFound(chName, posData)
            self.app.restoreOverrideCursor()
            return None, None

        fluo_path = os.path.join(posData.images_path, filename)
        filename, _ = os.path.splitext(filename)
        return fluo_path, filename

    def helpNewFile(self):
        msg = widgets.myMessageBox(showCentered=False)
        href = f'<a href="{user_manual_url}">user manual</a>'
        txt = html_utils.paragraph(f"""
            Cell-ACDC can open both a single image file or files structured 
            into Position folders.<br><br>
            If you are just testing out you can load a single image file, but 
            in general <b>we reccommend structuring your data into Position 
            folders.</b><br><br>
            More info about Position folders in the {href} at the section 
            called "Create required data structure from microscopy file(s)".
        """)
        msg.information(self, "Help on Position folders", txt)

    def initFluoData(self):
        if len(self.ch_names) <= 1:
            return

        if "ask_load_fluo_at_init" in self.df_settings.index:
            if self.df_settings.at["ask_load_fluo_at_init", "value"] == "No":
                return
        msg = widgets.myMessageBox(allowClose=False)
        txt = (
            "Do you also want to <b>load fluorescence images?</b><br>"
            "You can load <b>as many channels as you want</b>.<br><br>"
            "If you load fluorescence images then the software will "
            "<b>calculate metrics</b> for each loaded fluorescence channel "
            "such as min, max, mean, quantiles, etc. "
            "of each segmented object.<br><br>"
            "NOTE: You can always load them later from the menu "
            "<code>File --> Load fluorescence images...</code> or when you set "
            "measurements from the menu "
            "<code>Measurements --> Set measurements...</code>"
        )
        msg.addDoNotShowAgainCheckbox(text="Don't ask again")
        no, yes = msg.question(
            self,
            "Load fluorescence images?",
            html_utils.paragraph(txt),
            buttonsTexts=("No", "Yes"),
        )
        if msg.doNotShowAgainCheckbox.isChecked():
            self.df_settings.at["ask_load_fluo_at_init", "value"] = "No"
            self.df_settings.to_csv(self.settings_csv_path)
        if msg.clickedButton == yes:
            self.loadFluo_cb(None)
        self.AutoPilotProfile.storeClickMessageBox(
            "Load fluorescence images?", msg.clickedButton.text()
        )

    def is_imagej_dtype(self, dtype: np.dtype) -> bool:
        return dtype in (np.uint8, np.uint32, np.float32)

    def loadDataWorkerDataIntegrityCritical(self):
        errTitle = "All loaded positions contains frames over time!"
        self.titleLabel.setText(errTitle, color="r")

        msg = widgets.myMessageBox(parent=self)

        err_msg = html_utils.paragraph(f"""
            {errTitle}.<br><br>
            To load data that contains frames over time you have to select
            only ONE position.
        """)
        msg.setIcon(iconName="SP_MessageBoxCritical")
        msg.setWindowTitle("Loaded multiple positions with frames!")
        msg.addText(err_msg)
        msg.addButton("Ok")
        msg.show(block=True)

    @exception_handler
    def loadDataWorkerDataIntegrityWarning(self, pos_foldername):
        err_msg = (
            'WARNING: Segmentation mask file ("..._segm.npz") not found. '
            "You could run segmentation module first."
        )
        self.workerProgress(err_msg, "INFO")
        self.titleLabel.setText(err_msg, color="r")
        abort = False
        msg = widgets.myMessageBox(parent=self)
        warn_msg = html_utils.paragraph(f"""
            The folder {pos_foldername} <b>does not contain a
            pre-computed segmentation mask</b>.<br><br>
            You can continue with a blank mask or cancel and
            pre-compute the mask with the segmentation module.<br><br>
            Do you want to continue?
        """)
        msg.setIcon(iconName="SP_MessageBoxWarning")
        msg.setWindowTitle("Segmentation file not found")
        msg.addText(warn_msg)
        msg.addButton("Ok")
        continueWithBlankSegm = msg.addButton(" Cancel ")
        msg.show(block=True)
        if continueWithBlankSegm == msg.clickedButton:
            abort = True
        self.loadDataWorker.abort = abort
        self.loadDataWaitCond.wakeAll()

    @exception_handler
    def loadDataWorkerFinished(self, data):
        self.funcDescription = "loading data worker finished"
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None

        if data is None or data == "abort":
            self.loadingDataAborted()
            return

        if data[0].onlyEditMetadata:
            self.loadingDataAborted()
            return

        self.pos_i = 0
        self.data = data
        self.gui_createGraphicsItems()
        return True

    def loadFluo_cb(self, checked=True, fluo_channels=None):
        if fluo_channels is None:
            posData = self.data[self.pos_i]
            ch_names = [
                ch
                for ch in self.ch_names
                if ch != self.user_ch_name and ch not in posData.loadedFluoChannels
            ]
            if not ch_names:
                msg = widgets.myMessageBox()
                txt = html_utils.paragraph(
                    "You already <b>loaded ALL channels</b>.<br><br>"
                    "To <b>change the overlaid channel</b> "
                    "<b>right-click</b> on the overlay button."
                )
                msg.information(self, "All channels are loaded", txt)
                return False
            selectFluo = widgets.QDialogListbox(
                "Select channel to load",
                "Select channel names to load:\n",
                ch_names,
                multiSelection=True,
                parent=self,
            )
            selectFluo.exec_()

            if selectFluo.cancel:
                return False

            fluo_channels = selectFluo.selectedItemsText
            self.AutoPilotProfile.storeLoadedFluoChannels(fluo_channels)

        for p, posData in enumerate(self.data):
            # posData.ol_data = None
            for fluo_ch in fluo_channels:
                fluo_path, filename = self.getPathFromChName(fluo_ch, posData)
                if fluo_path is None:
                    self.criticalFluoChannelNotFound(fluo_ch, posData)
                    return False
                fluo_data, bkgrData = self.load_fluo_data(fluo_path)
                if fluo_data is None:
                    return False
                posData.loadedFluoChannels.add(fluo_ch)

                if posData.SizeT == 1:
                    fluo_data = fluo_data[np.newaxis]

                posData.fluo_data_dict[filename] = fluo_data
                posData.fluo_bkgrData_dict[filename] = bkgrData
                posData.ol_data_dict[filename] = fluo_data.copy()

        self.overlayButton.setStyleSheet(f"background-color: {GREEN_HEX}")
        self.guiTabControl.addChannels(
            [posData.user_ch_name, *posData.loadedFluoChannels]
        )
        return True

    def loadNonAlignedFluoChannel(self, fluo_path):
        posData = self.data[self.pos_i]
        if posData.filename.find("aligned") != -1:
            filename, _ = os.path.splitext(os.path.basename(fluo_path))
            path = f".../{posData.pos_foldername}/Images/{filename}_aligned.npz"
            msg = widgets.myMessageBox()
            msg.critical(
                self,
                "Aligned fluo channel not found!",
                "Aligned data for fluorescence channel not found!\n\n"
                f"You loaded aligned data for the cells channel, therefore "
                "loading NON-aligned fluorescence data is not allowed.\n\n"
                'Run the script "dataPrep.py" to create the following file:\n\n'
                f"{path}",
            )
            return None
        fluo_data = np.squeeze(skimage.io.imread(fluo_path))
        return fluo_data

    def loadPosTriggered(self):
        if not self.isDataLoaded:
            return

        self.startAutomaticLoadingPos()

    def loadSelectedData(self, user_ch_file_paths, user_ch_name):
        len(user_ch_file_paths)
        self.user_ch_file_paths = user_ch_file_paths

        self.logger.info(f"Reading {user_ch_name} channel metadata...")
        # Get information from first loaded position
        posData = load.loadData(
            user_ch_file_paths[0], user_ch_name, log_func=self.logger.info
        )
        posData.getBasenameAndChNames(qparent=self)
        posData.buildPaths()

        if posData.ext != ".h5":
            self.lazyLoader.salute = False
            self.lazyLoader.exit = True
            self.lazyLoaderWaitCond.wakeAll()
            self.waitReadH5cond.wakeAll()

        # Get end name of every existing segmentation file
        existingSegmEndNames = set()
        for filePath in user_ch_file_paths:
            _posData = load.loadData(filePath, user_ch_name, log_func=self.logger.info)
            _posData.getBasenameAndChNames(qparent=self)
            segm_files = load.get_segm_files(_posData.images_path)
            _existingEndnames = load.get_endnames(_posData.basename, segm_files)
            existingSegmEndNames.update(_existingEndnames)

        selectedSegmEndName = ""
        self.newSegmEndName = ""
        if self.isNewFile or not existingSegmEndNames:
            self.isNewFile = True
            # Remove the 'segm_' part to allow filenameDialog to check if
            # a new file is existing (since we only ask for the part after
            # 'segm_')
            existingEndNames = [
                n.replace("segm", "", 1).replace("_", "", 1)
                for n in existingSegmEndNames
            ]
            if posData.basename.endswith("_"):
                basename = f"{posData.basename}segm"
            else:
                basename = f"{posData.basename}_segm"
            win = apps.filenameDialog(
                basename=basename,
                hintText="Insert a <b>filename</b> for the segmentation file:",
                existingNames=existingEndNames,
            )
            win.exec_()
            if win.cancel:
                self.loadingDataAborted()
                return
            self.newSegmEndName = win.entryText
        else:
            if len(existingSegmEndNames) > 0:
                win = apps.SelectSegmFileDialog(
                    existingSegmEndNames,
                    self.exp_path,
                    parent=self,
                    addNewFileButton=True,
                    basename=posData.basename,
                )
                win.exec_()
                if win.cancel:
                    self.loadingDataAborted()
                    return
                if win.newSegmEndName is None:
                    selectedSegmEndName = win.selectedItemText
                    self.AutoPilotProfile.storeSelectedSegmFile(selectedSegmEndName)
                else:
                    self.newSegmEndName = win.newSegmEndName
                    self.isNewFile = True
            elif len(existingSegmEndNames) == 1:
                selectedSegmEndName = list(existingSegmEndNames)[0]

        posData.loadImgData()

        required_ram = posData.getBytesImageData()
        if required_ram >= 5e8:
            # Disable autosave for data > 500MB
            self.autoSaveToggle.setChecked(False)

        proceed = self.checkMemoryRequirements(required_ram)
        if not proceed:
            self.loadingDataAborted()
            return

        posData.loadOtherFiles(
            load_segm_data=True,
            load_metadata=True,
            create_new_segm=self.isNewFile,
            new_endname=self.newSegmEndName,
            end_filename_segm=selectedSegmEndName,
        )
        self.selectedSegmEndName = selectedSegmEndName
        self.labelBoolSegm = posData.labelBoolSegm
        posData.labelSegmData()

        print("")
        self.logger.info(f"Segmentation filename: {posData.segm_npz_path}")

        proceed = posData.askInputMetadata(
            self.num_pos,
            ask_SizeT=self.num_pos == 1,
            ask_TimeIncrement=True,
            ask_PhysicalSizes=True,
            singlePos=False,
            save=True,
            warnMultiPos=True,
        )
        if not proceed:
            self.loadingDataAborted()
            return

        self.AutoPilotProfile.storeOkAskInputMetadata()

        if posData.isSegm3D is None:
            self.isSegm3D = False
        else:
            self.isSegm3D = posData.isSegm3D
        self.SizeT = posData.SizeT
        self.SizeZ = posData.SizeZ
        self.TimeIncrement = posData.TimeIncrement
        self.PhysicalSizeZ = posData.PhysicalSizeZ
        self.PhysicalSizeY = posData.PhysicalSizeY
        self.PhysicalSizeX = posData.PhysicalSizeX
        self.loadSizeS = posData.loadSizeS
        self.loadSizeT = posData.loadSizeT
        self.loadSizeZ = posData.loadSizeZ

        self.overlayLabelsItems = {}
        self.drawModeOverlayLabelsChannels = {}

        self.existingSegmEndNames = existingSegmEndNames
        self.createOverlayLabelsContextMenu(existingSegmEndNames)
        self.overlayLabelsButtonAction.setVisible(True)
        self.createOverlayLabelsItems(existingSegmEndNames)
        self.disableNonFunctionalButtons()

        self.isH5chunk = posData.ext == ".h5" and (
            self.loadSizeT != self.SizeT or self.loadSizeZ != self.SizeZ
        )

        required_ram = posData.checkH5memoryFootprint() * self.loadSizeS
        if required_ram > 0:
            proceed = self.checkMemoryRequirements(required_ram)
            if not proceed:
                self.loadingDataAborted()
                return

        if posData.SizeT == 1:
            self.isSnapshot = True
        else:
            self.isSnapshot = False

        self.progressWin = apps.QDialogWorkerProgress(
            title="Loading data...",
            parent=self,
            pbarDesc=f'Loading "{user_ch_file_paths[0]}"...',
        )
        self.progressWin.show(self.app)

        func = partial(
            self.startLoadDataWorker, user_ch_file_paths, user_ch_name, posData
        )

        QTimer.singleShot(150, func)

    def load_fluo_data(self, fluo_path, isGuiThread=True):
        self.logger.info(f'Loading fluorescence image data from "{fluo_path}"...')
        bkgrData = None
        posData = self.data[self.pos_i]
        # Load overlay frames and align if needed
        filename = os.path.basename(fluo_path)
        filename_noEXT, ext = os.path.splitext(filename)
        if ext == ".npy" or ext == ".npz":
            fluo_data = np.load(fluo_path)
            try:
                fluo_data = np.squeeze(fluo_data["arr_0"])
            except Exception:
                fluo_data = np.squeeze(fluo_data)

            # Load background data
            bkgrData_path = os.path.join(
                posData.images_path, f"{filename_noEXT}_bkgrRoiData.npz"
            )
            if os.path.exists(bkgrData_path):
                bkgrData = np.load(bkgrData_path)
        elif ext == ".tif" or ext == ".tiff":
            aligned_filename = f"{filename_noEXT}_aligned.npz"
            aligned_path = os.path.join(posData.images_path, aligned_filename)
            if os.path.exists(aligned_path):
                fluo_data = np.load(aligned_path)["arr_0"]

                # Load background data
                bkgrData_path = os.path.join(
                    posData.images_path, f"{aligned_filename}_bkgrRoiData.npz"
                )
                if os.path.exists(bkgrData_path):
                    bkgrData = np.load(bkgrData_path)
            else:
                fluo_data = self.loadNonAlignedFluoChannel(fluo_path)
                if fluo_data is None:
                    return None, None

                # Load background data
                bkgrData_path = os.path.join(
                    posData.images_path, f"{filename_noEXT}_bkgrRoiData.npz"
                )
                if os.path.exists(bkgrData_path):
                    bkgrData = np.load(bkgrData_path)
        elif isGuiThread:
            txt = html_utils.paragraph(
                f"File format {ext} is not supported!\n"
                "Choose either .tif or .npz files."
            )
            msg = widgets.myMessageBox()
            msg.critical(self, "File not supported", txt)
            return None, None

        return fluo_data, bkgrData

    def loadingDataAborted(self):
        self.openFolderAction.setEnabled(True)
        self.titleLabel.setText("Loading data aborted.")

    @exception_handler
    def loadingDataCompleted(self):
        self.isDataLoading = True
        posData = self.data[self.pos_i]

        files_format = "\n".join(
            [f"  - {file}" for file in posData.images_folder_files]
        )
        sep = "-" * 100
        self.logger.info(
            f"{sep}\nFiles present in the first Position folder loaded:\n\n"
            f"{files_format}\n{sep}"
        )
        self.logger.info(f"Basename of the first Position: {posData.basename}")
        self.secondLevelToolbar.setVisible(True)
        self.updateImageValueFormatter()
        self.checkManageVersions()
        self.initManualBackgroundImage()
        self.initPixelSizePropsDockWidget()

        self.setWindowTitle(
            f'Cell-ACDC v{self._acdc_version} - GUI - "{posData.exp_path}"'
        )

        self.setupPreprocessing()
        self.setupCombiningChannels()

        if self.isSegm3D:
            self.segmNdimIndicator.setText("3D")
        else:
            self.segmNdimIndicator.setText("2D")

        self.segmNdimIndicatorAction.setVisible(True)

        self.guiTabControl.addChannels([posData.user_ch_name])
        self.showPropsDockButton.setDisabled(False)

        self.bottomScrollArea.show()
        self.gui_createStoreStateWorker()
        self.init_segmInfo_df()
        self.connectScrollbars()
        self.initPosAttr()

        self.logger.info("Pre-computing min and max values of the images...")
        self.img1.preComputedMinMaxValues(self.data)
        self.img2.minMaxValuesMapper = self.img1.minMaxValuesMapper

        self.initMetrics()
        self.initFluoData()
        self.createChannelNamesActions()
        self.addActionsLutItemContextMenu(self.imgGrad)

        # Scrollbar for opacity of img1 (when overlaying)
        self.img1.alphaScrollbar = self.addAlphaScrollbar(self.user_ch_name, self.img1)

        self.navigateScrollBar.setSliderPosition(posData.frame_i + 1)

        # Connect events at the end of loading data process
        self.gui_connectGraphicsEvents()
        if not self.isEditActionsConnected:
            self.gui_connectEditActions()
            self.normalizeToFloatAction.setChecked(True)

        self.navSpinBox.connectValueChanged(self.navigateSpinboxValueChanged)

        self.setFramesSnapshotMode()
        if self.isSnapshot:
            self.navSizeLabel.setText(f"/{len(self.data)}")
        else:
            self.navSizeLabel.setText(f"/{posData.SizeT}")

        self.enableZstackWidgets(posData.SizeZ > 1)
        # self.showHighlightZneighCheckbox()

        self.exportToVideoAction.setDisabled(posData.SizeZ == 1 and posData.SizeT == 1)

        self.img1BottomGroupbox.show()

        isLabVisible = self.df_settings.at["isLabelsVisible", "value"] == "Yes"
        isRightImgVisible = self.df_settings.at["isRightImageVisible", "value"] == "Yes"
        isNextFrameVisible = self.df_settings.at["isNextFrameVisible", "value"] == "Yes"
        isNextFrameActive = (
            isNextFrameVisible and self.labelsGrad.showNextFrameAction.isEnabled()
        )
        self.updateScrollbars()
        self.openFolderAction.setEnabled(True)
        self.editTextIDsColorAction.setDisabled(False)
        self.imgPropertiesAction.setEnabled(True)
        self.navigateToolBar.setVisible(True)
        self.labelsGrad.showLabelsImgAction.setChecked(isLabVisible)
        self.labelsGrad.showRightImgAction.setChecked(isRightImgVisible)
        self.labelsGrad.showNextFrameAction.setChecked(isNextFrameActive)
        if isRightImgVisible or isNextFrameActive:
            self.rightBottomGroupbox.setChecked(True)

        isTwoImagesLayout = isRightImgVisible or isLabVisible or isNextFrameActive
        self.setTwoImagesLayout(isTwoImagesLayout)

        self.setBottomLayoutStretch()

        if isNextFrameActive:
            self.rightBottomGroupbox.show()
            self.rightBottomGroupbox.setChecked(True)
            self.drawNothingCheckboxRight.click()

        self.readSavedCustomAnnot()
        self.addCustomAnnotButtonAllLoadedPos()
        self.setStatusBarLabel()

        self.initLookupTableLab()
        if self.invertBwAction.isChecked() and not self.invertBwAlreadyCalledOnce:
            self.invertBw(True)
        self.restoreSavedSettings()

        self.initContoursImage()
        self.initTextAnnot()
        self.initDelRoiLab()

        self.update_rp()
        self.updateAllImages()
        if posData.SizeT > 1:
            self.rightImageFramesScrollbar.setValueNoSignal(posData.frame_i + 2)
        self.setMetricsFunc()

        self.gui_createLabelRoiItem()
        self.gui_createZoomRectItem()

        self.titleLabel.setText("Data successfully loaded.", color=self.titleColor)

        self.disableNonFunctionalButtons()
        self.setVisible3DsegmWidgets()

        if len(self.data) == 1 and posData.SizeZ > 1 and posData.SizeT == 1:
            self.zSliceCheckbox.setChecked(True)
        else:
            self.zSliceCheckbox.setChecked(False)

        self.labelRoiCircItemLeft.setImageShape(self.currentLab2D.shape)
        self.labelRoiCircItemRight.setImageShape(self.currentLab2D.shape)

        self.retainSpaceSlidersToggled(self.retainSpaceSlidersAction.isChecked())

        self.stopAutomaticLoadingPos()
        self.viewAllCustomAnnotAction.setChecked(True)

        self.updateImageValueFormatter()

        posData.loadWhitelist()

        self.setFocusGraphics()
        self.setFocusMain()

        # Overwrite axes viewbox context menu
        self.ax1.vb.menu = self.imgGrad.gradient.menu
        self.ax2.vb.menu = self.labelsGrad.menu

        QTimer.singleShot(200, self.resizeGui)

        self.isDataLoaded = True
        self.isDataLoading = False

        self.initImgGradRescaleIntensitiesHowPreference()

        self.rescaleIntensitiesLut(setImage=False)

        self.gui_createAutoSaveWorker()

    def merge_default_segm_info(
        self,
        existing_df: pd.DataFrame,
        default_df: pd.DataFrame,
    ) -> pd.DataFrame:
        merged_df = pd.concat([default_df, existing_df])
        unique_idx = ~merged_df.index.duplicated()
        return merged_df[unique_idx]

    def newFile(self):
        self.newSegmEndName = ""
        self.isNewFile = True
        msg = widgets.myMessageBox(parent=self, showCentered=False)
        msg.setWindowTitle("File or folder?")
        msg.addText(
            html_utils.paragraph("""
            Do you want to load an <b>image file</b> or <b>Position 
            folder(s)</b>?
        """)
        )
        loadPosButton = QPushButton("Load Position folder", msg)
        loadPosButton.setIcon(QIcon(":folder-open.svg"))
        loadFileButton = QPushButton("Load image file", msg)
        loadFileButton.setIcon(QIcon(":image.svg"))
        helpButton = widgets.helpPushButton("Help...")
        msg.addButton(helpButton)
        helpButton.disconnect()
        helpButton.clicked.connect(self.helpNewFile)
        msg.addCancelButton(connect=True)
        msg.addButton(loadFileButton)
        msg.addButton(loadPosButton)
        loadPosButton.setDefault(True)
        msg.exec_()
        if msg.cancel:
            return

        if msg.clickedButton == loadPosButton:
            self._openFolder()
        else:
            self._openFile()

    def openFile(self, checked=False, file_path=None):
        self.logger.info(f'Opening FILE "{file_path}"')

        self.isNewFile = False
        self._openFile(file_path=file_path)

    def openFolder(self, checked=False, exp_path=None, imageFilePath=""):
        if exp_path is None:
            self.logger.info("Asking to select a folder path...")
        else:
            self.logger.info(f'Opening FOLDER "{exp_path}"...')

        self.isNewFile = False
        if hasattr(self, "data") and self.titleLabel.text != "Saved!":
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                "Do you want to <b>save</b> before loading another dataset?"
            )
            _, no, yes = msg.question(
                self, "Save?", txt, buttonsTexts=("Cancel", "No", "Yes")
            )
            if msg.clickedButton == yes:
                func = partial(self._openFolder, exp_path, imageFilePath)
                self.saveData(finishedCallback=func)
                return
            elif msg.cancel:
                self.store_data()
                return
            else:
                self.store_data(autosave=False)

        self._openFolder(exp_path=exp_path, imageFilePath=imageFilePath)

    def openRecentFile(self, path):
        self.logger.info(f"Opening recent folder: {path}")
        self.addToRecentPaths(path, logger=self.logger)
        self.openFolder(exp_path=path)

    def open_image_file_context(
        self, file_path: str, timestamp: str | None = None
    ) -> OpenImageFileContext:
        filename_no_ext, ext = os.path.splitext(os.path.basename(file_path))
        filename_no_ext = filename_no_ext.rstrip("_")
        ext = ext.lower()
        dirpath = os.path.dirname(file_path)
        dirname = os.path.basename(dirpath)
        requires_images_folder = dirname != "Images"
        acdc_folder = None

        if requires_images_folder:
            timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            acdc_folder = f"{timestamp}_acdc"
            exp_path = os.path.join(dirpath, acdc_folder, "Images")
        else:
            exp_path = dirpath

        return OpenImageFileContext(
            file_path=file_path,
            filename_no_ext=filename_no_ext,
            extension=ext,
            source_dirpath=dirpath,
            source_dirname=dirname,
            exp_path=exp_path,
            acdc_folder=acdc_folder,
            requires_images_folder=requires_images_folder,
        )

    def open_image_file_target(
        self,
        context: OpenImageFileContext,
        channel_name: str | None = None,
    ) -> OpenImageFileTarget:
        filename_no_ext = context.filename_no_ext
        basename = None
        metadata_csv_filename = None
        metadata_csv_filepath = None

        if channel_name is not None:
            underscore_splits = filename_no_ext.split("_")
            if len(underscore_splits) > 1:
                default_ch_name = underscore_splits[-1]
                if channel_name == default_ch_name:
                    filename_no_ext = "_".join(underscore_splits[:-1])

            basename = f"{filename_no_ext}_"
            metadata_csv_filename = f"{basename}metadata.csv"
            metadata_csv_filepath = os.path.join(
                context.exp_path, metadata_csv_filename
            )
            new_filename = f"{filename_no_ext}_{channel_name}{context.extension}"
        else:
            new_filename = f"{filename_no_ext}{context.extension}"

        new_filepath = os.path.join(context.exp_path, new_filename)
        tif_filename_no_ext = os.path.splitext(new_filename)[0]
        tif_filename = f"{tif_filename_no_ext}.tif"
        tif_path = os.path.join(context.exp_path, tif_filename)

        return OpenImageFileTarget(
            context=context,
            filename_no_ext=filename_no_ext,
            channel_name=channel_name,
            basename=basename,
            new_filename=new_filename,
            new_filepath=new_filepath,
            metadata_csv_filename=metadata_csv_filename,
            metadata_csv_filepath=metadata_csv_filepath,
            tif_filename=tif_filename,
            tif_path=tif_path,
            direct_copy_supported=context.extension in (".tif", ".npz"),
        )

    def prepare_tiff_image_data(self, image: np.ndarray) -> ImageDataPreparation:
        converted_rgb_to_gray = False
        converted_dtype = False
        prepared_image = image

        if prepared_image.ndim == 3 and (
            prepared_image.shape[-1] == 3 or prepared_image.shape[-1] == 4
        ):
            converted_rgb_to_gray = True
            if prepared_image.shape[-1] == 3:
                prepared_image = skimage.color.rgb2gray(prepared_image)
            else:
                prepared_image = cv2.cvtColor(prepared_image, cv2.COLOR_RGBA2GRAY)
            prepared_image = skimage.img_as_ubyte(prepared_image)

        if not self.is_imagej_dtype(prepared_image.dtype):
            converted_dtype = True
            prepared_image = skimage.img_as_ubyte(prepared_image)

        return ImageDataPreparation(
            image=prepared_image,
            converted_rgb_to_gray=converted_rgb_to_gray,
            converted_dtype=converted_dtype,
        )

    def reload_cb(self):
        posData = self.data[self.pos_i]
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)
        labData = np.load(posData.segm_npz_path)
        # Keep compatibility with .npy and .npz files
        try:
            lab = labData["arr_0"][posData.frame_i]
        except Exception:
            lab = labData[posData.frame_i]
        posData.segm_data[posData.frame_i] = lab.copy()
        self.get_data()
        self.tracking()
        self.updateAllImages()

    def showInfoAutosave(self, posData):
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = """
            Cell-ACDC either detected unsaved data in a previous session and it 
            stored it because the <b>Autosave</b><br>
            function was active, or it crashed during saving.<br><br>
            You can toggle Autosave ON and OFF from the menu on the top menubar 
            <code>File --> Autosave</code>.
        """
        txt = f"""
            {txt}<br><br>
            If Cell-ACDC crashed during saving, the segmentation file ending 
            with <code>.new.npz</code><br>
            is present and you might be able to recover the data from there. 
        """

        txt = f"""
            {txt}<br><br>
            You can find additional recovered data in the following folder:
        """
        txt = html_utils.paragraph(txt)
        msg.information(
            self,
            "Autosave info",
            txt,
            path_to_browse=posData.recoveryFolderPath,
            commands=(posData.recoveryFolderPath,),
        )

    def startAutomaticLoadingPos(self):
        self.AutoPilot = autopilot.AutoPilot(self)
        self.AutoPilot.execLoadPos()

    @exception_handler
    def startLoadDataWorker(self, user_ch_file_paths, user_ch_name, firstPosData):
        self.funcDescription = "loading data"

        self.guiTabControl.propsQGBox.idSB.setValue(0)

        self.thread = QThread()
        self.loadDataMutex = QMutex()
        self.loadDataWaitCond = QWaitCondition()

        self.loadDataWorker = workers.loadDataWorker(
            self, user_ch_file_paths, user_ch_name, firstPosData
        )

        self.loadDataWorker.moveToThread(self.thread)
        self.loadDataWorker.signals.finished.connect(self.thread.quit)
        self.loadDataWorker.signals.finished.connect(self.loadDataWorker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.loadDataWorker.signals.finished.connect(self.loadDataWorkerFinished)
        self.loadDataWorker.signals.progress.connect(self.workerProgress)
        self.loadDataWorker.signals.initProgressBar.connect(self.workerInitProgressbar)
        self.loadDataWorker.signals.progressBar.connect(self.workerUpdateProgressbar)
        self.loadDataWorker.signals.critical.connect(self.workerCritical)
        self.loadDataWorker.signals.dataIntegrityCritical.connect(
            self.loadDataWorkerDataIntegrityCritical
        )
        self.loadDataWorker.signals.dataIntegrityWarning.connect(
            self.loadDataWorkerDataIntegrityWarning
        )
        self.loadDataWorker.signals.sigPermissionError.connect(
            self.workerPermissionError
        )
        self.loadDataWorker.signals.sigWarnMismatchSegmDataShape.connect(
            self.askMismatchSegmDataShape
        )
        self.loadDataWorker.signals.sigRecovery.connect(self.askRecoverNotSavedData)

        self.thread.started.connect(self.loadDataWorker.run)
        self.thread.start()

    def stopAutomaticLoadingPos(self):
        if self.AutoPilot is None:
            return

        if self.AutoPilot.timer.isActive():
            self.AutoPilot.timer.stop()
        self.AutoPilot = None

    def warnMemoryNotSufficient(self, total_ram, available_ram, required_ram):
        total_ram = myutils._bytes_to_GB(total_ram)
        available_ram = myutils._bytes_to_GB(available_ram)
        required_ram = myutils._bytes_to_GB(required_ram)
        required_perc = round(100 * required_ram / available_ram)
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(f"""
            The total amount of data that you requested to load is about
            <b>{required_ram:.2f} GB</b> ({required_perc}% of the available memory)
            but there are only <b>{available_ram:.2f} GB</b> available.<br><br>
            For <b>optimal operation</b>, we recommend loading <b>maximum 30%</b>
            of the available memory. To do so, try to close open apps to
            free up some memory. Another option is to crop the images
            using the data prep module.<br><br>
            If you choose to continue, the <b>system might freeze</b>
            or your OS could simply kill the process.<br><br>
            What do you want to do?
        """)
        cancelButton, continueButton = msg.warning(
            self,
            "Memory not sufficient",
            txt,
            buttonsTexts=("Cancel", "Continue anyway"),
        )
        if msg.clickedButton == continueButton:
            # Disable autosaving since it would keep a copy of the data and
            # we cannot afford it with low memory
            self.autoSaveToggle.setChecked(False)
            return True
        else:
            return False

    def warnUserCreationImagesFolder(self, images_path, ext):
        msg = widgets.myMessageBox(wrapText=False)
        txt = f"""
            Cell-ACDC requires a specific folder structure to load the data.<br><br>
            Specifically, it requires the <b>image(s) to be located in a
            folder called <code>Images</code></b>.<br><br>
            The <b>file format</b> of the images must be <b>TIFF or NPZ</b> 
            (.tif or .npz extension).<br><br>
            You can choose to let Cell-ACDC create the required data structure 
            from your file,<br>
            or you can stop the 
            process and manually place the image(s) into a folder called 
            <code>Images</code>.<br><br>
            If you choose to proceed, Cell-ACDC will create the following 
            folder:
            <copiable>{images_path}</copiable>
            <br>
        """

        if ext == ".tif" or ext == ".npz":
            txt = f"{txt}How do you want to proceed?"
        else:
            txt = f"{txt}Do you want to proceed?"
        txt = html_utils.paragraph(txt)

        if ext == ".tif" or ext == ".npz":
            copyButton = widgets.copyPushButton("Copy the image into the new folder")
            moveButton = widgets.movePushButton("Move the image into the new folder")
            _, copyButton, moveButton = msg.information(
                self,
                "Creating Images folder",
                txt,
                buttonsTexts=("Cancel", copyButton, moveButton),
            )
            if msg.cancel:
                return False, None

            if msg.clickedButton == copyButton:
                return True, True
            elif msg.clickedButton == moveButton:
                return True, False

        else:
            msg.information(
                self,
                "Creating Images folder",
                txt,
                buttonsTexts=("Cancel", "Yes, proceed"),
            )
            if msg.cancel:
                return False, None

            return True, True

    def workerPermissionError(self, txt, waitCond):
        msg = widgets.myMessageBox(parent=self)
        msg.setIcon(iconName="SP_MessageBoxCritical")
        msg.setWindowTitle("Permission denied")
        msg.addText(txt)
        msg.addButton("  Ok  ")
        msg.exec_()
        waitCond.wakeAll()

    def zSliceAbsent(self, filename, posData):
        self.app.restoreOverrideCursor()
        SizeZ = posData.SizeZ
        chNames = posData.chNames
        filenamesPresent = posData.segmInfo_df.index.get_level_values(0).unique()
        chNamesPresent = [
            ch
            for ch in chNames
            for file in filenamesPresent
            if file.endswith(ch) or file.endswith(f"{ch}_aligned")
        ]
        win = apps.QDialogZsliceAbsent(filename, SizeZ, chNamesPresent)
        win.exec_()
        if win.cancel:
            self.worker.abort = True
            self.waitCond.wakeAll()
            return
        if win.useMiddleSlice:
            user_ch_name = filename[len(posData.basename) :]
            for _posData in self.data:
                if _posData is None:
                    continue
                _, filename = self.getPathFromChName(user_ch_name, _posData)
                df = myutils.getDefault_SegmInfo_df(_posData, filename)
                _posData.segmInfo_df = pd.concat([df, _posData.segmInfo_df])
                unique_idx = ~_posData.segmInfo_df.index.duplicated()
                _posData.segmInfo_df = _posData.segmInfo_df[unique_idx]
                _posData.segmInfo_df.to_csv(_posData.segmInfo_df_csv_path)
        elif win.useSameAsCh:
            user_ch_name = filename[len(posData.basename) :]
            for _posData in self.data:
                if _posData is None:
                    continue
                _, srcFilename = self.getPathFromChName(win.selectedChannel, _posData)
                cellacdc_df = _posData.segmInfo_df.loc[srcFilename].copy()
                _, dstFilename = self.getPathFromChName(user_ch_name, _posData)
                if dstFilename is None:
                    self.worker.abort = True
                    self.waitCond.wakeAll()
                    return
                dst_df = myutils.getDefault_SegmInfo_df(_posData, dstFilename)
                for z_info in cellacdc_df.itertuples():
                    frame_i = z_info.Index
                    zProjHow = z_info.which_z_proj
                    if zProjHow == "single z-slice":
                        src_idx = (srcFilename, frame_i)
                        if _posData.segmInfo_df.at[src_idx, "resegmented_in_gui"]:
                            col = "z_slice_used_gui"
                        else:
                            col = "z_slice_used_dataPrep"
                        z_slice = _posData.segmInfo_df.at[src_idx, col]
                        dst_idx = (dstFilename, frame_i)
                        dst_df.at[dst_idx, "z_slice_used_dataPrep"] = z_slice
                        dst_df.at[dst_idx, "z_slice_used_gui"] = z_slice
                _posData.segmInfo_df = pd.concat([dst_df, _posData.segmInfo_df])
                unique_idx = ~_posData.segmInfo_df.index.duplicated()
                _posData.segmInfo_df = _posData.segmInfo_df[unique_idx]
                _posData.segmInfo_df.to_csv(_posData.segmInfo_df_csv_path)
        elif win.runDataPrep:
            user_ch_file_paths = []
            user_ch_name = filename[len(self.data[self.pos_i].basename) :]
            for _posData in self.data:
                if _posData is None:
                    continue
                user_ch_path = load.get_filename_from_channel(
                    _posData.images_path, user_ch_name
                )
                if user_ch_path is None:
                    self.worker.abort = True
                    self.waitCond.wakeAll()
                    return
                user_ch_file_paths.append(user_ch_path)
                exp_path = os.path.dirname(_posData.pos_path)

            dataPrepWin = dataPrep.dataPrepWin()
            dataPrepWin.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
            dataPrepWin.titleText = """
            Select z-slice (or projection) for each frame/position.<br>
            Once happy, close the window.
            """
            dataPrepWin.show()
            dataPrepWin.initLoading()
            dataPrepWin.SizeT = self.data[0].SizeT
            dataPrepWin.SizeZ = self.data[0].SizeZ
            dataPrepWin.metadataAlreadyAsked = True
            self.logger.info(f"Loading channel {user_ch_name} data...")
            dataPrepWin.loadFiles(exp_path, user_ch_file_paths, user_ch_name)
            dataPrepWin.startAction.setDisabled(True)
            dataPrepWin.onlySelectingZslice = True

            loop = QEventLoop(self)
            dataPrepWin.loop = loop
            loop.exec_()

        self.waitCond.wakeAll()
