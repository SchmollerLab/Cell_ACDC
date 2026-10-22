"""Overlay channels and overlay labels mixins shared by guiWin and the signals viewer.

The methods were moved verbatim from `guiWin`. The host window is responsible 
for creating the widgets and attributes the methods use.
"""
import os
from functools import partial

import numpy as np
import cv2
import skimage.exposure
import skimage.io
from natsort import natsorted

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QCursor, QFont
from qtpy.QtWidgets import QAction, QActionGroup, QLabel, QMenu

import pyqtgraph as pg

from . import load, html_utils, myutils, widgets, colors, issues_url
from . import _palettes

GREEN_HEX = _palettes.green()

_font = QFont()
_font.setPixelSize(11)


class OverlayChannelsGuiElements:
    def gui_createOverlayColors(self):
        fluoChannels = [ch for ch in self.ch_names if ch != self.user_ch_name]
        self.logger.info(
            f'Number of TIFF files detected: {len(fluoChannels)}'
        )
        self.overlayColors = {}
        for c, ch in enumerate(fluoChannels):
            if f'{ch}_rgb' in self.df_settings.index:
                rgb_text = self.df_settings.at[f'{ch}_rgb', 'value']
                rgb = tuple([int(val) for val in rgb_text.split('_')])
                self.overlayColors[ch] = rgb
            else:
                if c >= len(self.overlayRGBs) -1:
                    i = c/len(fluoChannels)
                    additional_color_num = c - len(self.overlayRGBs) + 1
                    rgbs = [
                        tuple([round(c*255) for c in self.overlayCmap(i)][:3]) 
                        for _ in range(additional_color_num)
                    ]
                    self.overlayRGBs.extend(rgbs)
                rgb = colors.FLUO_CHANNELS_COLORS.get(ch, self.overlayRGBs[c])
                self.overlayColors[ch] = rgb

    def gui_createOverlayItems(self):
        self.imgGrad.setAxisLabel(self.user_ch_name)
        self.baseLayerToolbutton = widgets.OverlayChannelToolButton(
            self.user_ch_name, self.imgGrad
        )
        self.baseLayerToolbutton.setChecked(True)
        self.baseLayerToolbutton.clicked.connect(
            self.overlayChannelToolbuttonClicked
        )
        self.allOverlayToolbuttons = {
            self.user_ch_name: self.baseLayerToolbutton
        }
        self.allOverlayToolbuttonsByIdx = {
            0: self.baseLayerToolbutton
        }
        self.baseLayerToolbutton.action = (
            self.overlayToolbar.addWidget(self.baseLayerToolbutton)
        )
        self.overlayLayersItems = {}
        self.overlayToolbarAreChannelsChecked = {}
        fluoChannels = [ch for ch in self.ch_names if ch != self.user_ch_name]
        for c, ch in enumerate(fluoChannels):
            overlayItems = self.getOverlayItems(ch, c+1)                
            self.overlayLayersItems[ch] = overlayItems
            imageItem, lutItem = overlayItems[:2]
            self.ax1.addItem(imageItem)
            self.lutItemsLayout.addItem(lutItem, row=0, col=c+1)
            toolbutton = overlayItems[3]
            self.allOverlayToolbuttons[ch] = toolbutton
            self.allOverlayToolbuttonsByIdx[c+1] = toolbutton

        self.overlayToolbuttonsSep = self.overlayToolbar.addSeparator()
        self.plotsCol = len(self.ch_names)
        
        self.ax1.addImageItem(self.rgbaImg1)

    def setOverlayColors(self):
        self.overlayRGBs = colors.overlay_rgbs.copy()

    def loadOverlayData(self, ol_channels, addToExisting=False):
        posData = self.data[self.pos_i]
        for ol_ch in ol_channels:
            if ol_ch not in list(posData.loadedFluoChannels):
                # Requested channel was never loaded --> load it at first
                # iter i == 0
                success = self.loadFluo_cb(fluo_channels=[ol_ch])
                if not success:
                    return False

        lastChannelName = ol_channels[-1]
        for action in self.fluoDataChNameActions:
            if action.text() == lastChannelName:
                action.setChecked(True)

        for p, posData in enumerate(self.data):
            if addToExisting:
                ol_data = posData.ol_data
            else:
                ol_data = {}
            for i, ol_ch in enumerate(ol_channels):
                _, filename = self.getPathAndFilenameNoExtFromChName(ol_ch, posData)
                ol_data[filename] = (
                    posData.ol_data_dict[filename].copy()
                )                        
                self.addFluoChNameContextMenuAction(ol_ch)
            posData.ol_data = ol_data

        return True

    def askSelectOverlayChannel(self):
        ch_names = [ch for ch in self.ch_names if ch != self.user_ch_name]
        selectFluo = widgets.QDialogListbox(
            'Select channel',
            'Select channel names to overlay:\n',
            ch_names, multiSelection=True, parent=self
        )
        selectFluo.exec_()
        if selectFluo.cancel:
            return

        return selectFluo.selectedItemsText

    def loadNonAlignedFluoChannel(self, fluo_path):
        posData = self.data[self.pos_i]
        if posData.filename.find('aligned') != -1:
            filename, _ = os.path.splitext(os.path.basename(fluo_path))
            path = f'.../{posData.pos_foldername}/Images/{filename}_aligned.npz'
            msg = widgets.myMessageBox()
            msg.critical(
                self, 'Aligned fluo channel not found!',
                'Aligned data for fluorescence channel not found!\n\n'
                f'You loaded aligned data for the cells channel, therefore '
                'loading NON-aligned fluorescence data is not allowed.\n\n'
                'Run the script "dataPrep.py" to create the following file:\n\n'
                f'{path}'
            )
            return None
        fluo_data = np.squeeze(skimage.io.imread(fluo_path))
        return fluo_data

    def load_fluo_data(self, fluo_path, isGuiThread=True):
        self.logger.info(f'Loading fluorescence image data from "{fluo_path}"...')
        fluo_data, bkgrData = load.load_image_and_bkgr_data(fluo_path)
        return fluo_data, bkgrData

    def setOverlaySingleChannel(self, *args, **kwargs):
        if self.overlayToolbar.isSingleChannel():
            self.overlayToolbarAreChannelsChecked = {
                channel:toolbutton.isChecked() 
                for channel, toolbutton in self.allOverlayToolbuttons.items()
            }
            firstActiveToolbutton = [
                toolbutton for toolbutton in self.allOverlayToolbuttons.values()
                if toolbutton.isChecked()
            ][0]
            firstActiveToolbutton.click()
        else:
            for ch, checked in self.overlayToolbarAreChannelsChecked.items():
                toolbutton = self.allOverlayToolbuttons[ch]
                toolbutton.setChecked(checked)
            
            self.setOverlayImages()

    def updateTransparentOverlayRgba(self, *args, **kwargs):
        self.setOverlayImages()

    def setOverlayTransparency(self, transparent: bool):
        opacity = float(transparent)
        opacity = opacity if opacity < 1.0 else 0.999
        self.rgbaImg1.setOpacity(opacity)

        if transparent:
            self.rgbaImg1.setVisible(True)
            self.img1.setOpacity(0.001, applyToLinked=False)
            self.imgGrad.sigLookupTableChanged.connect(
                self.updateTransparentOverlayRgba
            )
            self.imgGrad.sigLevelsChanged.connect(
                self.updateTransparentOverlayRgba
            )
            
            for channel, items in self.overlayLayersItems.items():
                imageItem, lutItem, alphaSB = items[:3]
                if transparent:
                    alphaSB.valueChanged.connect(
                        self.updateTransparentOverlayRgba
                    )
                    lutItem.sigLookupTableChanged.connect(
                        self.updateTransparentOverlayRgba
                    )
                    lutItem.sigLevelsChanged.connect(
                        self.updateTransparentOverlayRgba
                    )
                    imageItem.setOpacity(0)
        else:
            self.rgbaImg1.clear()
            self.rgbaImg1.setVisible(False)
            self.img1.setOpacity(1.0, applyToLinked=False)
            for channel, items in self.overlayLayersItems.items():
                imageItem, lutItem, alphaSB = items[:3]
                imageItem.setOpacity(1.0)
                try:
                    lutItem.sigLookupTableChanged.disconnect(
                        self.updateTransparentOverlayRgba
                    )
                except:
                    pass
                try:
                    lutItem.sigLevelsChanged.disconnect(
                        self.updateTransparentOverlayRgba
                    )
                except:
                    pass
                try:
                    alphaSB.valueChanged.disconnect(
                        self.updateTransparentOverlayRgba
                    )
                except:
                    pass

            self.setOverlayItemsOpacities()

        self.setOverlayImages()

    def overlay_cb(self, checked):
        self.overlayToolbar.setVisible(checked)
        
        self.UserNormAction, _, _ = self.getCheckNormAction()
        posData = self.data[self.pos_i]
        if checked:
            if posData.ol_data is None:
                selectedChannels = self.askSelectOverlayChannel()
                if selectedChannels is None:
                    self.overlayButton.toggled.disconnect()
                    self.overlayButton.setChecked(False)
                    self.overlayButton.toggled.connect(self.overlay_cb)
                    return
                
                success = self.loadOverlayData(selectedChannels)         
                if not success:
                    return False
                lastChannel = selectedChannels[-1]
                self.setCheckedOverlayContextMenusActions(selectedChannels)
                imageItem = self.overlayLayersItems[lastChannel][0]
                self.setOpacityOverlayLayersItems(None, imageItem=imageItem)
                self.setOverlayChannelsToolbuttonsChecked()

            self.setRetainSizePolicyLutItems()
            self.normalizeRescale0to1Action.setChecked(True)

            self.updateAllImages()
            self.updateImageValueFormatter()
            self.enableOverlayWidgets(True)
        else:
            self.img1.setOpacity(1.0)
            self.updateAllImages()
            self.updateImageValueFormatter()
            self.enableOverlayWidgets(False)
            self.clearOverlayImageItems()
            
        
        self.setOverlayItemsVisible()

    def showOverlayContextMenu(self, event):
        if not self.overlayButton.isChecked():
            return

        self.overlayContextMenu.exec_(QCursor.pos())

    def setCheckedOverlayContextMenusActions(self, channelNames):
        for action in self.overlayContextMenu.actions():
            if action.text() in channelNames:
                action.setChecked(True)
                self.checkedOverlayChannels.add(action.text())

    def enableOverlayWidgets(self, enabled):
        posData = self.data[self.pos_i]   
        if enabled:
            self.overlayColorButton.setDisabled(False)
            self.editOverlayColorAction.setDisabled(False)

            if posData.SizeZ == 1:
                return

            self.zSliceOverlay_SB.setMaximum(posData.SizeZ-1)
            if self.zProjOverlay_CB.currentText().find('max') != -1:
                self.overlay_z_label.setDisabled(True)
                self.zSliceOverlay_SB.setDisabled(True)
            else:
                z = self.zSliceOverlay_SB.sliderPosition()
                self.overlay_z_label.setText(f'Overlay z-slice  {z+1:02}/{posData.SizeZ}')
                self.zSliceOverlay_SB.setDisabled(False)
                self.overlay_z_label.setDisabled(False)
            self.zSliceOverlay_SB.show()
            self.overlay_z_label.show()
            self.zProjOverlay_CB.show()
            self.zSliceOverlay_SB.valueChanged.connect(self.updateOverlayZslice)
            self.zProjOverlay_CB.currentTextChanged.connect(self.updateOverlayZproj)
            self.zProjOverlay_CB.activated.connect(self.clearComboBoxFocus)
        else:
            self.zSliceOverlay_SB.setDisabled(True)
            self.zSliceOverlay_SB.hide()
            self.overlay_z_label.hide()
            self.zProjOverlay_CB.hide()
            self.overlayColorButton.setDisabled(True)
            self.editOverlayColorAction.setDisabled(True)

            if posData.SizeZ == 1:
                return

            self.zSliceOverlay_SB.valueChanged.disconnect()
            self.zProjOverlay_CB.currentTextChanged.disconnect()
            self.zProjOverlay_CB.activated.disconnect()

    def criticalFluoChannelNotFound(self, fluo_ch, posData):
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        ls = "\n".join(myutils.listdir(posData.images_path))
        detailsText = (
            f'Files present in the {posData.relPath} folder:\n'
            f'{ls}'
        )
        title = 'Requested channel data not found!'
        href = f'<a href="{issues_url}">GitHub page</a>'
        txt = html_utils.paragraph(f"""
            The folder
            <copiable>{posData.images_path}</copiable>
            <b>does not contain</b> valid image data files.<br><br>
            Valid files are <code>.tif</code>, 
            <code>_aligned.npz</code>, <code>.h5</code>, 
            or <code>_symlink.ini</code>.<br><br>
            If you need help understanding this, feel free to contact us by 
            opening an issue on our {href}.<br><br>
            Thank you for your patience!
        """)
        msg.warning(
            self, title, txt, buttonsTexts=('Ok'), 
            path_to_browse=posData.images_path,
            detailsText=detailsText,
        )

    def getOlImg(self, key, frame_i=None):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i

        img = posData.ol_data[key][frame_i]
        if posData.SizeZ > 1:
            zProjHow = self.zProjOverlay_CB.currentText()
            z = self.zSliceOverlay_SB.sliderPosition()
            if zProjHow == 'same as above':
                zProjHow = self.zProjComboBox.currentText()
                z = self.zSliceScrollBar.sliderPosition()
                reconnect = False
                try:
                    self.zSliceOverlay_SB.valueChanged.disconnect()
                    reconnect = True
                except TypeError:
                    pass
                self.zSliceOverlay_SB.setSliderPosition(z)
                if reconnect:
                    self.zSliceOverlay_SB.valueChanged.connect(
                        self.updateOverlayZslice
                    )
            if zProjHow == 'single z-slice':
                self.overlay_z_label.setText(
                    f'Overlay z-slice  {z+1:02}/{posData.SizeZ}')
                ol_img = img[z].copy()
            elif zProjHow == 'max z-projection':
                ol_img = img.max(axis=0)
            elif zProjHow == 'mean z-projection':
                ol_img = img.mean(axis=0)
            elif zProjHow == 'median z-proj.':
                ol_img = np.median(img, axis=0)
        else:
            ol_img = img.copy()

        return ol_img

    def setOverlayImages(self, frame_i=None, mode=None):
        if not self.overlayButton.isChecked():
            return
        
        posData = self.data[self.pos_i]
        if posData.ol_data is None:
            return
        
        if mode is None:
            if self.overlayToolbar.isAlphaEncodedIntensity():
                mode = 'intensity_to_alpha'
            elif self.overlayToolbar.isTransparent():
                mode = 'transparent'
            else:
                mode = 'normal'
        
        rgba_imgs_info = {}
        for filename in posData.ol_data:
            chName = myutils.get_chname_from_basename(
                filename, posData.basename, remove_ext=False
            )
            if chName not in self.checkedOverlayChannels:
                continue
            
            items = self.overlayLayersItems[chName]
            imageItem, lutItem, alphaSB = items[:3]

            ol_img = self.getOlImg(filename, frame_i=frame_i)

            if mode == 'transparent' or mode == 'intensity_to_alpha':
                toolbutton = items[3]
                if not toolbutton.isChecked():
                    continue
                
                rescale_lut_action = (
                    imageItem.lutItem.rescaleActionGroup.checkedAction()
                )
                rescale_lut_how = rescale_lut_action.text()
                in_range = 'image'
                if rescale_lut_how == 'Choose custom levels...':
                    in_range = tuple(imageItem.getLevels())
                    
                alpha_val = alphaSB.value()/alphaSB.maximum()
                ol_img = skimage.exposure.rescale_intensity(
                    ol_img, in_range=in_range, out_range=(0.0, 1.0)
                )               
                rgba_imgs_info[chName] = (ol_img, alpha_val, lutItem, imageItem)
            else:
                self.rescaleIntensitiesLut(setImage=False, imageItem=imageItem)
                imageItem.setImage(ol_img)
        
        if not (mode == 'transparent' or mode == 'intensity_to_alpha'):
            self.rgbaImg1.clear()
            self.rgbaImg1.setVisible(False)
            return

        self.rgbaImg1.setVisible(True)
        
        alpha_values = []
        images = []
        luts = []
        for channel, info in rgba_imgs_info.items():
            ol_img, alpha_val, lutItem, imageItem = info
            alpha_values.append(alpha_val)
            images.append(ol_img)
            luts.append(lutItem.gradient.getLookupTable(256, alpha=255)/255)
        
        if self.baseLayerToolbutton.isChecked():
            image1 = self._getImageupdateAllImages()
            rescale_lut_action = (
                self.imgGrad.rescaleActionGroup.checkedAction()
            )
            rescale_lut_how = rescale_lut_action.text()
            in_range = 'image'
            if rescale_lut_how == 'Choose custom levels...':
                in_range = tuple(self.img1.getLevels())
                
            image1 = skimage.exposure.rescale_intensity(
                image1, in_range=in_range, out_range=(0.0, 1.0)
            )        
            images.append(image1)
            baseLut = (
                self.imgGrad.gradient.getLookupTable(256, alpha=255)/255
            )
            luts.append(baseLut)
            
        if mode == 'intensity_to_alpha':
            # get base image item lut
            if self.baseLayerToolbutton.isChecked():
                base_img = images.pop()
                base_lut = luts.pop()
            else:
                base_img = np.zeros_like(images[0])
                base_lut = None
                
            rgba_merge = colors.combine_grayscale_images_with_alpha(
                base_img=base_img,
                images=images,
                alphas=alpha_values,
                luts=luts,
                base_lut=base_lut
            )
            self.rgbaImg1.setImage(rgba_merge)
            return
        
        images_rgba = []
        for img, lut in zip(images, luts):
            rgba = colors.grayscale_apply_lut(img, lut)            
            images_rgba.append(rgba)
        
        weights = colors.hierarchical_weights(alpha_values)
        rgba_merge = colors.hierarchical_blend(images_rgba, weights)        
        self.rgbaImg1.setImage(rgba_merge)

    def getOpacitiesFromAlphaScrollbarValues(self):
        alpha_values = []
        activeOverlayImageItems = []
        for items in self.overlayLayersItems.values():
            imgItem, lutItem, alphaSB = items[:3]
            _toolbutton = alphaSB.toolbutton
            if not _toolbutton.isChecked() or not _toolbutton.isVisible():
                continue

            alpha_values.append(alphaSB.value()/alphaSB.maximum())
            activeOverlayImageItems.append(imgItem)
        
        opacities = colors.hierarchical_weights(alpha_values)[::-1]
        channel_opacity_mapper = {}
        for i, imgItem in enumerate(activeOverlayImageItems):
            channel_opacity_mapper[imgItem.channelName] = opacities[i+1]
        
        channel_opacity_mapper[self.user_ch_name] = opacities[0]
        
        return channel_opacity_mapper

    def toggleOverlayColorButton(self, checked=True):
        self.mousePressColorButton(None)

    def changeOverlayColor(self, button):
        rgb = button.color().getRgb()[:3]
        lutItem = self.overlayLayersItems[button.channel][1]
        self.initColormapOverlayLayerItem(rgb, lutItem)
        lutItem.overlayColorButton.setColor(rgb)

    def saveOverlayColor(self, button):
        rgb = button.color().getRgb()[:3]
        rgb_text = '_'.join([str(val) for val in rgb])
        self.df_settings.at[f'{button.channel}_rgb', 'value'] = rgb_text
        self.df_settings.to_csv(self.settings_csv_path)

    def removeOverlayItems(self):
        self.lutItemsLayout.clear()
        
        try:
            for toolbutton in self.allOverlayToolbuttonsByIdx.values():
                self.overlayToolbar.removeAction(toolbutton.action)
            
            self.overlayToolbuttonsSep.removeFromToolbar()
        except Exception as err:
            pass

    def clearOverlayImageItems(self):
        for items in self.overlayLayersItems.values():
            imageItem = items[0]
            imageItem.clear()
        
        self.rgbaImg1.clear()

    def createOverlayContextMenu(self):
        ch_names = [ch for ch in self.ch_names if ch != self.user_ch_name]
        self.overlayContextMenu = QMenu()
        self.overlayContextMenu.addSeparator()
        self.checkedOverlayChannels = set()
        for chName in ch_names:
            action = QAction(chName, self.overlayContextMenu)
            action.setCheckable(True)
            action.toggled.connect(self.overlayChannelToggled)
            self.overlayContextMenu.addAction(action)

    def overlayChannelToggled(self, checked):
        # Action toggled from overlayButton context menu
        channelName = self.sender().text()
        posData = self.data[self.pos_i]
        if checked:
            if channelName not in posData.loadedFluoChannels:
                self.loadOverlayData([channelName], addToExisting=True)
            else:
                _, filename = self.getPathAndFilenameNoExtFromChName(channelName, posData)
                posData.ol_data[filename] = (
                    posData.ol_data_dict[filename].copy()
                )
            
            self.checkedOverlayChannels.add(channelName)    
        else:
            self.checkedOverlayChannels.remove(channelName)
            imageItem = self.overlayLayersItems[channelName][0]
            imageItem.clear()
        
        self.setOverlayChannelsToolbuttonsChecked()
        self.setOverlayItemsVisible()
        self.setRetainSizePolicyLutItems()
        self.updateAllImages()

    def initFluoData(self):
        if len(self.ch_names) <= 1:
            return
        
        if 'ask_load_fluo_at_init' in self.df_settings.index:
            if self.df_settings.at['ask_load_fluo_at_init', 'value'] == 'No':
                return   
        msg = widgets.myMessageBox(allowClose=False)
        txt = (
            'Do you also want to <b>load fluorescence images?</b><br>'
            'You can load <b>as many channels as you want</b>.<br><br>'
            'If you load fluorescence images then the software will '
            '<b>calculate metrics</b> for each loaded fluorescence channel '
            'such as min, max, mean, quantiles, etc. '
            'of each segmented object.<br><br>'
            'NOTE: You can always load them later from the menu '
            '<code>File --> Load fluorescence images...</code> or when you set '
            'measurements from the menu '
            '<code>Measurements --> Set measurements...</code>'
        )
        msg.addDoNotShowAgainCheckbox(text="Don't ask again")
        no, yes = msg.question(
            self, 'Load fluorescence images?', html_utils.paragraph(txt),
            buttonsTexts=('No', 'Yes')
        )
        if msg.doNotShowAgainCheckbox.isChecked():
            self.df_settings.at['ask_load_fluo_at_init', 'value'] = 'No'
            self.df_settings.to_csv(self.settings_csv_path)
        if msg.clickedButton == yes:
            self.loadFluo_cb(None)
        self.AutoPilotProfile.storeClickMessageBox(
            'Load fluorescence images?', msg.clickedButton.text()
        )

    def loadFluo_cb(self, checked=True, fluo_channels=None):
        if fluo_channels is None:
            posData = self.data[self.pos_i]
            ch_names = [
                ch for ch in self.ch_names if ch != self.user_ch_name
                and ch not in posData.loadedFluoChannels
            ]
            if not ch_names:
                msg = widgets.myMessageBox()
                txt = html_utils.paragraph(
                    'You already <b>loaded ALL channels</b>.<br><br>'
                    'To <b>change the overlaid channel</b> '
                    '<b>right-click</b> on the overlay button.'
                )
                msg.information(self, 'All channels are loaded', txt)
                return False
            selectFluo = widgets.QDialogListbox(
                'Select channel to load',
                'Select channel names to load:\n',
                ch_names, multiSelection=True, parent=self
            )
            selectFluo.exec_()

            if selectFluo.cancel:
                return False

            fluo_channels = selectFluo.selectedItemsText
            self.AutoPilotProfile.storeLoadedFluoChannels(fluo_channels)

        for p, posData in enumerate(self.data):
            # posData.ol_data = None
            for fluo_ch in fluo_channels:
                fluo_path, filename = self.getPathAndFilenameNoExtFromChName(fluo_ch, posData)
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
                
        self.overlayButton.setStyleSheet(f'background-color: {GREEN_HEX}')
        self.guiTabControl.addChannels([
            posData.user_ch_name, *posData.loadedFluoChannels
        ])
        return True

    def getOverlayItems(self, channelName, index):
        imageItem = widgets.OverlayImageItem()
        imageItem.setOpacity(0.5)
        imageItem.channelName = channelName

        lutItem = widgets.myHistogramLUTitem(
            parent=self, name='image', axisLabel=channelName
        )
        imageItem.lutItem = lutItem
        for action in lutItem.rescaleActionGroup.actions():
            if action.text() == self.defaultRescaleIntensHow:
                action.setChecked(True)
            break
        
        lutItem.removeAddScaleBarAction()
        lutItem.removeAddTimestampAction()
        lutItem.restoreState(self.df_settings)
        lutItem.setImageItem(imageItem)
        lutItem.vb.raiseContextMenu = lambda x: None
        initColor = self.overlayColors[channelName]
        self.initColormapOverlayLayerItem(initColor, lutItem)
        lutItem.addOverlayColorButton(initColor, channelName)
        lutItem.initColor = initColor
        lutItem.hide()

        lutItem.overlayColorButton.sigColorChanging.connect(
            self.changeOverlayColor
        )
        lutItem.overlayColorButton.sigColorChanged.connect(
            self.saveOverlayColor
        )

        lutItem.invertBwAction.toggled.connect(self.setCheckedInvertBW)

        lutItem.contoursColorButton.disconnect() 
        lutItem.contoursColorButton.clicked.connect(
            self.imgGrad.contoursColorButton.click
        )
        for act in lutItem.contLineWightActionGroup.actions():
            act.toggled.connect(self.contLineWeightToggled)
        
        lutItem.mothBudLineColorButton.disconnect() 
        lutItem.mothBudLineColorButton.clicked.connect(
            self.imgGrad.mothBudLineColorButton.click
        )
        for act in lutItem.mothBudLineWightActionGroup.actions():
            act.toggled.connect(self.mothBudLineWeightToggled)

        lutItem.textColorButton.disconnect()
        lutItem.textColorButton.clicked.connect(
            self.editTextIDsColorAction.trigger
        )

        lutItem.defaultSettingsAction.triggered.connect(
            self.restoreDefaultSettings
        )
        lutItem.labelsAlphaSlider.valueChanged.connect(
            self.setValueLabelsAlphaSlider
        )
        lutItem.sigRescaleIntes.connect(
            partial(self.rescaleIntensitiesLut, imageItem=imageItem)
        )
        if f'how_rescale_intensities_{channelName}' in self.df_settings.index:
            how = self.df_settings.at[
                f'how_rescale_intensities_{channelName}', 'value'
            ]
            if not 'custom' in how:
                lutItem.setRescaleIntensitiesHow(how)
        
        self.rescaleIntensChannelHowMapper[channelName] = (
            'Rescale each 2D image'
        )

        self.addActionsLutItemContextMenu(lutItem)

        alphaScrollBar = self.addAlphaScrollbar(channelName, imageItem)
        
        toolbutton = widgets.OverlayChannelToolButton(
            channelName, lutItem, shortcut=str(index)
        )
        toolbutton.action = self.overlayToolbar.addWidget(toolbutton)
        toolbutton.setVisible(False)
        
        toolbutton.clicked.connect(self.overlayChannelToolbuttonClicked)
        
        alphaScrollBar.toolbutton = toolbutton
        
        return imageItem, lutItem, alphaScrollBar, toolbutton

    def addAlphaScrollbar(self, channelName, imageItem):
        alphaScrollBar = widgets.ScrollBar(Qt.Horizontal)
        imageItem.alphaScrollBar = alphaScrollBar
        alphaScrollBar.channelName = channelName
        
        label = QLabel(f'Alpha {channelName}')
        label.setFont(_font)
        label.hide()
        alphaScrollBar.imageItem = imageItem
        alphaScrollBar.label = label
        alphaScrollBar.setFixedHeight(self.h)
        alphaScrollBar.hide()
        alphaScrollBar.setMinimum(0)
        alphaScrollBar.setMaximum(40)
        alphaScrollBar.setValue(20)
        alphaScrollBar.setToolTip(
            f'Control the alpha value of the overlaid channel {channelName}.\n'
            'alpha=0 results in NO overlay,\n'
            'alpha=1 results in only fluorescence data visible'
        )
        self.bottomLeftLayout.addWidget(
            alphaScrollBar.label, self.alphaScrollbarRow, 0, 
            alignment=Qt.AlignRight
        )
        self.bottomLeftLayout.addWidget(
            alphaScrollBar, self.alphaScrollbarRow, 1, 1, 2
        )

        alphaScrollBar.valueChanged.connect(
            partial(self.setOpacityOverlayLayersItems, scrollbar=alphaScrollBar)
        )

        self.alphaScrollbarRow += 1
        return alphaScrollBar

    def setOverlayChannelsToolbuttonsChecked(self):
        for channel, items in self.overlayLayersItems.items():
            _, lutItem, alphaSB, toolbutton = items[:4]
            toolbutton.setChecked(
                not self.overlayToolbar.isSingleChannel()
                and channel in self.checkedOverlayChannels
            )

    def setOverlayItemsVisible(self):
        for channel, items in self.overlayLayersItems.items():
            _, lutItem, alphaSB, toolbutton = items[:4]
            lutItem.hide()
            alphaSB.hide()
            alphaSB.label.hide()
            toolbutton.setVisible(False)    
        
        if not self.overlayButton.isChecked():
            return
        
        for channel, items in self.overlayLayersItems.items():
            _, lutItem, alphaSB, toolbutton = items[:4]
            if channel in self.checkedOverlayChannels:
                lutItem.show()
                alphaSB.show()
                alphaSB.label.show()
                toolbutton.setVisible(True)

    def overlayChannelToolbuttonClicked(self, checked=False, toolbutton=None):
        if toolbutton is None:
            toolbutton = self.sender()
        
        n_checked_buttons = (
            sum([b.isChecked() for b in self.allOverlayToolbuttons.values()])
        )
        
        channelName = toolbutton.channelName()
        
        if n_checked_buttons == 0 or self.overlayToolbar.isSingleChannel():
            # At least one button must be checked
            toolbutton.setChecked(True)
        
        if self.overlayToolbar.isSingleChannel():
            # Exclusive buttons 
            for channel, otherToolbutton in self.allOverlayToolbuttons.items():
                if channel == channelName:
                    continue

                otherToolbutton.setChecked(False)
        
        if self.overlayToolbar.isTransparent():            
            self.setOverlayImages()
            return
        
        self.setOverlayItemsOpacities()

    def setOverlayItemsOpacities(self):
        n_checked_buttons = (
            sum([b.isChecked() for b in self.allOverlayToolbuttons.values()])
        )
        
        isSingleChannel = (
            self.overlayToolbar.isSingleChannel()
            or n_checked_buttons == 1
        )
        
        channel_opacity_mapper = self.getOpacitiesFromAlphaScrollbarValues()
        
        # Set opacity of every layer accordingly
        for channel, otherToolbutton in self.allOverlayToolbuttons.items():          
            if channel == self.user_ch_name:
                otherImageItem = self.img1
                alphaScrollbar = None
                # alpha_value = channel_opacity_mapper[channel]
            else:
                otherItems = self.overlayLayersItems[channel]
                otherImageItem = otherItems[0]
                alphaScrollbar = otherItems[2]
                # alpha_value = alphaScrollbar.value()/alphaScrollbar.maximum()
            
            if otherToolbutton.isChecked() and isSingleChannel:
                op_val = 1.0
            elif otherToolbutton.isChecked():
                op_val = channel_opacity_mapper.get(channel, 0.01)
            else:
                op_val = 0.0
            
            if op_val == 0:
                op_val = 0.01

            op_val = op_val if op_val < 1.0 else 0.999
            
            otherImageItem.setOpacity(op_val, applyToLinked=False)
            
            if alphaScrollbar is None:
                continue
            
            alphaScrollbar.setDisabled(bool(op_val == 0))

    def initColormapOverlayLayerItem(self, foregrColor, lutItem):
        if self.invertBwAction.isChecked():
            bkgrColor = (255,255,255,255)
        else:
            bkgrColor = (0,0,0,255)
        gradient = colors.get_pg_gradient((bkgrColor, foregrColor))
        lutItem.setGradient(gradient)

    def setOpacityOverlayLayersItems(self, value, imageItem=None, scrollbar=None):
        if self.overlayToolbar.isTransparent():
            return

        if scrollbar is None:
            scrollbar = imageItem.alphaScrollBar

        channel = scrollbar.channelName
        toolbutton = self.allOverlayToolbuttons[channel]
        if not toolbutton.isChecked() or not toolbutton.isVisible():
            return

        if value is None:
            value = scrollbar.value()

        if imageItem is None:
            imageItem = scrollbar.imageItem
            alpha = value/scrollbar.maximum()
        elif value > 1:
            alpha = value/scrollbar.maximum()
        else:
            alpha = value

        alpha_values = []
        activeOverlayImageItems = []
        for items in self.overlayLayersItems.values():
            imgItem, lutItem, alphaSB = items[:3]
            _toolbutton = alphaSB.toolbutton
            if alphaSB.channelName == channel:
                alpha_values.append(alpha)
            elif not _toolbutton.isChecked() or not _toolbutton.isVisible():
                continue
            else:
                alpha_values.append(alphaSB.value()/alphaSB.maximum())

            activeOverlayImageItems.append(imgItem)

        if not alpha_values:
            self.img1.setOpacity(1.0, applyToLinked=False)
            return

        opacities = colors.hierarchical_weights(alpha_values)[::-1]

        for i, imgItem in enumerate(activeOverlayImageItems):
            imgItem.setOpacity(opacities[i+1])

        self.img1.setOpacity(opacities[0], applyToLinked=False)

    def addFluoChNameContextMenuAction(self, ch_name):
        posData = self.data[self.pos_i]
        allTexts = [
            action.text() for action in self.chNamesQActionGroup.actions()
        ]
        if ch_name not in allTexts:
            action = QAction(self)
            action.setText(ch_name)
            action.setCheckable(True)
            self.chNamesQActionGroup.addAction(action)
            action.setChecked(True)
            self.fluoDataChNameActions.append(action)

    def updateOverlayZslice(self, z):
        self.setOverlayImages()

    def updateOverlayZproj(self, how):
        if how.find('max') != -1 or how == 'same as above':
            self.overlay_z_label.setDisabled(True)
            self.zSliceOverlay_SB.setDisabled(True)
        else:
            self.overlay_z_label.setDisabled(False)
            self.zSliceOverlay_SB.setDisabled(False)
        self.setOverlayImages()

    def checkOverlayToolbuttonClicked(self, event):
        success = False
        try:
            n = int(event.text())
            toolbutton = self.allOverlayToolbuttonsByIdx.get(n, None)
            toolbutton.click()
            success = True
        except Exception as e:
            # printl(traceback.format_exc())
            success = False        
        return success

    def setRetainSizePolicyLutItems(self):
        if not self.retainSizeLutItems:
            return
        for channel, items in self.overlayLayersItems.items():
            _, lutItem, alphaSB = items[:3]
            myutils.setRetainSizePolicy(lutItem, retain=True)
        QTimer.singleShot(300, self.autoRange)

    def mousePressColorButton(self, event):
        posData = self.data[self.pos_i]
        items = list(self.checkedOverlayChannels)
        if len(items)>1:
            selectFluo = widgets.QDialogListbox(
                'Select image',
                'Select which fluorescence image you want to update the color of\n',
                items, multiSelection=False, parent=self
            )
            selectFluo.exec_()
            keys = selectFluo.selectedItemsText
            if selectFluo.cancel or not keys:
                return
            else:
                self.overlayColorButton.channel = keys[0]
        else:
            self.overlayColorButton.channel = items[0]
        self.overlayColorButton.selectColor()

    def addActionsLutItemContextMenu(self, lutItem):      
        lutItem.gradient.menu.addSection('Visible channels: ')
        for action in self.overlayContextMenu.actions():
            if action.isSeparator():
                continue
            lutItem.gradient.menu.addAction(action)
        lutItem.gradient.menu.addSeparator()

    def setValueLabelsAlphaSlider(self, value):
        self.imgGrad.labelsAlphaSlider.setValue(value)
        self.updateLabelsAlpha(value)

class OverlayLabelsGuiElements:
    def gui_addOverlayLayerItems(self):
        for items in self.overlayLabelsItems.values():
            imageItem, contoursItem, gradItem = items
            self.ax1.addItem(imageItem)
            self.ax1.addItem(contoursItem)

    def clearOverlayLabelsItems(self):
        for segmEndname, drawMode in self.drawModeOverlayLabelsChannels.items():
            items = self.overlayLabelsItems[segmEndname]
            imageItem, contoursItem, gradItem = items
            imageItem.clear()
            contoursItem.clear()

    def overlayLabels_cb(self, checked, selectedLabelsEndnames=None):
        if checked:
            if not self.drawModeOverlayLabelsChannels:
                if selectedLabelsEndnames is None:
                    selectedLabelsEndnames = self.askLabelsToOverlay()
                if selectedLabelsEndnames is None:
                    self.logger.info('Overlay labels cancelled.')
                    self.overlayLabelsButton.setChecked(False)
                    return
                for selectedEndname in selectedLabelsEndnames:
                    self.loadOverlayLabelsData(selectedEndname)
                    for action in self.overlayLabelsContextMenu.actions():
                        if not action.isCheckable():
                            continue
                        if action.text() == selectedEndname:
                            action.setChecked(True)
                lastSelectedName = selectedLabelsEndnames[-1]
                for action in self.selectOverlayLabelsActionGroup.actions():
                    if action.text() == lastSelectedName:
                        action.setChecked(True)
        self.updateAllImages()

    def askLabelsToOverlay(self):
        selectOverlayLabels = widgets.QDialogListbox(
            'Select segmentation to overlay',
            'Select segmentation file to overlay:\n',
            natsorted(self.existingSegmEndNames), 
            multiSelection=True, 
            parent=self
        )
        selectOverlayLabels.exec_()
        if selectOverlayLabels.cancel:
            return

        return selectOverlayLabels.selectedItemsText

    def showOverlayLabelsContextMenu(self, event):
        if not self.overlayLabelsButton.isChecked():
            return

        self.overlayLabelsContextMenu.exec_(QCursor.pos())

    def initOverlayLabelsContoursImages(self):
        posData = self.data[self.pos_i]
        z_slice = self.z_lab()
        img = posData.img_data[posData.frame_i]
        Y, X = img[z_slice].shape[-2:]
        
        self.overlayLabelsContoursImages = {}
        for segmEndname in self.drawModeOverlayLabelsChannels.keys():
            overlayLabelsContoursImage = np.zeros((Y, X, 4), dtype=np.uint8)
            self.overlayLabelsContoursImages[segmEndname] = (
                overlayLabelsContoursImage
            )

    def hideOverlayLabelsItems(self, specific=None):
        if specific is None:
            specific = self.overlayLabelsItems.keys()
        for segmEndname in specific:
            imageItem, contoursItem, gradItem = self.overlayLabelsItems[segmEndname]
            imageItem.setVisible(False)
            contoursItem.setVisible(False)
            gradItem.setVisible(False)

    def showOverlayLabelsItems(self, specific=None):
        if specific is None:
            specific = self.overlayLabelsItems.keys()
        for segmEndname in specific:
            imageItem, contoursItem, gradItem = self.overlayLabelsItems[segmEndname]
            drawMode = self.drawModeOverlayLabelsChannels[segmEndname]
            if drawMode == 'Draw contours':
                contoursItem.setVisible(True)
            elif drawMode == 'Overlay labels':
                imageItem.setVisible(True)
                gradItem.setVisible(True)

    def setOverlayLabelsItems(self, specific=None):
        if not self.overlayLabelsButton.isChecked():
            self.hideOverlayLabelsItems(specific=specific)
            return

        if specific is None:
            specific = self.drawModeOverlayLabelsChannels.keys()
        
        if not hasattr(self, 'overlayLabelsContoursImages'):
            self.initOverlayLabelsContoursImages()

        for segmEndname in specific:
            drawMode = self.drawModeOverlayLabelsChannels[segmEndname]
            overlayLabelsContoursImage = (
                self.overlayLabelsContoursImages[segmEndname]
            )
            overlayLabelsContoursImage[:] = 0
            ol_lab = self.getOverlayLabelsData(segmEndname)
            items = self.overlayLabelsItems[segmEndname]
            imageItem, contoursItem, gradItem = items
            contours = []
            if drawMode == 'Draw contours':
                rp = self._acdcRegionProps(ol_lab, precache_centroids=False)
                for obj in rp:
                    contours_obj = self.getObjContours(
                        obj,
                        all_external=True,
                        include_internal=self.showAllContoursToggle.isChecked()
                    )
                    contours.extend(contours_obj)
                cv2.drawContours(
                    overlayLabelsContoursImage, 
                    contours, 
                    -1, 
                    self.contLineColor, 
                    self.contLineWeight
                )
                contoursItem.setImage(overlayLabelsContoursImage)
            elif drawMode == 'Overlay labels':
                imageItem.setImage(ol_lab, autoLevels=False)
        self.showOverlayLabelsItems(specific=specific)

    def getOverlayLabelsData(self, segmEndname):
        posData = self.data[self.pos_i]
        
        if posData.ol_labels_data is None:
            self.loadOverlayLabelsData(segmEndname)            
        elif segmEndname not in posData.ol_labels_data:
            self.loadOverlayLabelsData(segmEndname)
        
        comb_seg = False
        if 'combined segm.' == segmEndname:
            comb_seg = True
            if not self.isSegm3D:
                zStackImg = self.data[0].SizeZ > 1
                if zStackImg:
                    z_slice = self.zSliceScrollBar.sliderPosition()
                else:
                    z_slice = 0
                ol_labels_data = posData.ol_labels_data['combined segm.']
                out = ol_labels_data[posData.frame_i][z_slice]
                return out.astype(np.uint32)
        
        ol_labels_data = posData.ol_labels_data[segmEndname][posData.frame_i]
        if ol_labels_data.ndim == 3:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if isZslice:
                z = self.zSliceScrollBar.sliderPosition()
                ol_lab = ol_labels_data[z]
                if comb_seg:
                    ol_lab = ol_lab.astype(np.uint32)
                return ol_lab
            else:
                ol_lab = ol_labels_data.max(axis=0)
                if comb_seg:
                    ol_lab = ol_lab.astype(np.uint32)
                return ol_lab
        else:
            return ol_labels_data

    def loadOverlayLabelsData(self, segmEndname, pos_i=None):
        if pos_i is None:
            pos_i = self.pos_i
        posData = self.data[pos_i]

        if posData.ol_labels_data is None:
            posData.ol_labels_data = {}
        if segmEndname == 'combined segm.':
             posData.ol_labels_data['combined segm.'] = posData.combine_img_data
             return
        
        filePath, filename = load.get_path_from_endname(
            segmEndname, posData.images_path
        )
        self.logger.info(f'Loading "{segmEndname}.npz"...')
        labelsData = np.load(filePath)['arr_0']
        if posData.SizeT == 1:
            labelsData = labelsData[np.newaxis]

        if self.isSegm3D and labelsData.ndim == 3:
            # 2D segm --> stack to 3D
            T, Y, X = labelsData.shape
            repeat = [labelsData]*posData.SizeZ
            labelsData = np.stack(repeat, axis=1)

        posData.ol_labels_data[segmEndname] = labelsData

    def createOverlayLabelsContextMenu(self, segmEndnames):
        self.overlayLabelsContextMenu = QMenu()
        self.overlayLabelsContextMenu.addSeparator()
        self.drawModeOverlayLabelsChannels = {}
        segmEndnames_extended = list(segmEndnames.copy())
        segmEndnames_extended = ['combined segm.'] + segmEndnames_extended
        for segmEndname in segmEndnames_extended:
            action = QAction(segmEndname, self.overlayLabelsContextMenu)
            if segmEndname == 'combined segm.':
                action.setCheckable(False)
                self.combineSegmViewToggle = action
            else:
                action.setCheckable(True)
            action.toggled.connect(self.addOverlayLabelsToggled)
            self.overlayLabelsContextMenu.addAction(action)

    def createOverlayLabelsItems(self, segmEndnames):
        selectActionGroup = QActionGroup(self)
        segmEndnames_extended = list(segmEndnames.copy())
        segmEndnames_extended = ['combined segm.'] + segmEndnames_extended
        for segmEndname in segmEndnames_extended:
            action = QAction(segmEndname)
            if segmEndname == 'combined segm.':
                action.setCheckable(False)
            else:
                action.setCheckable(True)
            action.toggled.connect(self.setOverlayLabelsItemsVisible)
            selectActionGroup.addAction(action)
        self.selectOverlayLabelsActionGroup = selectActionGroup

        self.overlayLabelsItems = {}
        for segmEndname in segmEndnames_extended:
            imageItem = pg.ImageItem()

            gradItem = widgets.overlayLabelsGradientWidget(
                imageItem, selectActionGroup, segmEndname
            )
            gradItem.hide()
            gradItem.drawModeActionGroup.triggered.connect(
                self.overlayLabelsDrawModeToggled
            )
            self.mainLayout.addWidget(gradItem, 0, 0)

            contoursItem = pg.ImageItem()

            items = (imageItem, contoursItem, gradItem)
            self.overlayLabelsItems[segmEndname] = items

    def addOverlayLabelsToggled(self, checked, name=None):
        if name is None:
            name = self.sender().text()
        if checked:
            gradItem = self.overlayLabelsItems[name][-1]
            drawMode = gradItem.drawModeActionGroup.checkedAction().text()
            self.drawModeOverlayLabelsChannels[name] = drawMode
        else:
            self.drawModeOverlayLabelsChannels.pop(name)
            self.hideOverlayLabelsItems(specific=[name])
        self.setOverlayLabelsItems()

    def overlayLabelsDrawModeToggled(self, action):
        segmEndname = action.segmEndname
        drawMode = action.text()
        if segmEndname in self.drawModeOverlayLabelsChannels:
            self.drawModeOverlayLabelsChannels[segmEndname] = drawMode
            items = self.overlayLabelsItems[segmEndname]
            imageItem, contoursItem, gradItem = items
            if drawMode == 'Draw contours':
                imageItem.clear()
            else:
                contoursItem.clear()
            self.setOverlayLabelsItems()

    def setOverlayLabelsItemsVisible(self, checked):  
        for _segmEndname, drawMode in self.drawModeOverlayLabelsChannels.items():
            items = self.overlayLabelsItems[_segmEndname]
            gradItem = items[-1]
            gradItem.hide()
        
        if checked:
            segmEndname = self.sender().text()
            gradItem = self.overlayLabelsItems[segmEndname][-1]
            gradItem.show()

class ChannelImageHelpersGuiElements:
    def getCheckNormAction(self):
        normalize = False
        how = ''
        for action in self.normalizeQActionGroup.actions():
            if action.isChecked():
                how = action.text()
                normalize = True
                break
        return action, normalize, how

    def updateImageValueFormatter(self):
        if self.img1.image is not None:
            dtype = self.img1.image.dtype
            n_digits = len(str(int(self.img1.image.max())))
            self.imgValueFormatter = myutils.get_number_fstring_formatter(
                dtype, precision=abs(n_digits-5)
            )

        rawImgData = self.data[self.pos_i].img_data
        dtype = rawImgData.dtype
        n_digits = len(str(int(rawImgData.max())))
        self.rawValueFormatter = myutils.get_number_fstring_formatter(
            dtype, precision=abs(n_digits-5)
        )

    def _getImageupdateAllImages(self, image=None):
        if image is not None:
            return image
        
        img = self.getImage()
        return img

    def getPathAndFilenameNoExtFromChName(self, chName, posData):
        channel_file_path = load.get_filename_from_channel(
            posData.images_path, chName
        )
        if not channel_file_path:
            self.app.restoreOverrideCursor()
            return None, None
        
        # The function `get_filename_from_channel` will append ';;channel_name' 
        # when the imgPath is the symlink.ini file
        filename = os.path.basename(channel_file_path)
        filename_no_ext, ext = os.path.splitext(filename)
        if not ext.startswith('.ini;;'):
            filename = filename_no_ext
        else:
            filename_no_ext = filename.replace('.ini', '')
        
        return channel_file_path, filename_no_ext
