import os
from functools import partial
from natsort import natsorted

from .. import myutils, apps, load, printl, core, widgets
from .. import exception_handler
from ..utils import base

from qtpy.QtCore import QTimer, Signal

class NapariArboretumDialog(base.MainThreadSinglePosUtilBase):
    def __init__(
            self, posPath, app, title: str, infoText: str, parent=None
        ):
        
        module = myutils.get_module_name(__file__)
        super().__init__(
            app, title, module, infoText, parent
        )

        self.sigClose.connect(self.close)

        func = partial(self.launchNapariArboretum, posPath)
        QTimer.singleShot(200, func)

    @exception_handler
    def launchNapariArboretum(self, posPath):
        images_path = os.path.join(posPath, 'Images')
        ls = myutils.listdir(images_path)

        image_files = [
            file for file in ls 
            if file.endswith('.tif') 
            or file.endswith('aligned.npz') 
            or file.endswith('.h5')
        ]

        selectImageFile = widgets.QDialogListbox(
            'Select image file',
            'Select which image file to load\n',
            image_files, multiSelection=False, parent=self
        )
        selectImageFile.exec_()
        if selectImageFile.cancel:
            self.logger.info('napari-arboretum utility aborted.')
            return

        imageFile = selectImageFile.selectedItemsText[0]
        self.logger.info(f'Loading image file {imageFile}...')
        
        imagePath = os.path.join(images_path, imageFile)
        posData = load.loadData(imagePath, '')
        posData.getBasenameAndChNames()
        posData.loadImgData()

        segm_files = load.get_segm_files(posData.images_path)
        existingEndnames = load.get_endnames(
            posData.basename, segm_files
        )

        if len(existingEndnames) > 1:
            win = apps.SelectSegmFileDialog(
                existingEndnames, images_path, parent=self, 
                basename=posData.basename
            )
            win.exec_()
            if win.cancel:
                self.logger.info('napari-arboretum utility aborted.')
                return
            selectedSegmEndName = win.selectedItemText
        else:
            selectedSegmEndName = existingEndnames[0]

        self.logger.info(f'Loading segmentation file ending with {selectedSegmEndName}...')

        posData.loadOtherFiles(
            load_segm_data=True,
            load_acdc_df=True,
            end_filename_segm=selectedSegmEndName
        )

        self.logger.info('Importing napari...')
        import napari

        self.logger.info('Building arboretum lineage tree...')
        acdc_df = posData.acdc_df.reset_index()
        tree = core.LineageTree(acdc_df, logging_func=self.logger.info)
        tracks_data, graph, properties = tree.to_arboretum()

        props = natsorted(acdc_df.columns.to_list())
        selectProps = widgets.QDialogListbox(
            'Select measurements',
            'Select measurements to add as <b>properties</b> in napari viewer<br><br>'
            '<code>Ctrl+Click</code> <i>to select multiple items</i><br>'
            '<code>Shift+Click</code> <i>to select a range of items</i><br>',
            props, multiSelection=True, parent=self
        )
        selectProps.exec_()
        if selectProps.cancel:
            self.logger.info('napari-arboretum utility aborted.')
            return
        
        for col in selectProps.selectedItemsText:
            try:
                properties[col] = acdc_df[col]
            except Exception as e:
                pass

        self.logger.info('Launching napari viewer...')
        viewer = napari.Viewer()
        viewer.add_image(posData.img_data, name=imageFile)
        viewer.add_labels(posData.segm_data, name=selectedSegmEndName)
        acdc_df_endname = selectedSegmEndName.replace('segm', 'acdc_tracks')
        viewer.add_tracks(
            tracks_data, graph=graph, name=acdc_df_endname, 
            properties=properties
        )
        viewer.window.add_plugin_dock_widget(
            plugin_name="napari-arboretum", widget_name="Arboretum"
        )

        napari.run(max_loop_level=2)

        self.logger.info('napari viewer closed.')
        self.close()

        
