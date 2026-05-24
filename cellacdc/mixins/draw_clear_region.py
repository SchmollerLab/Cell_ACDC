"""View adapter for draw-clear-region workflows."""

from __future__ import annotations


class DrawClearRegion:
    """Extracted from guiWin."""

    def drawClearRegion_cb(self, checked):
        posData = self.data[self.pos_i]
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.drawClearRegionButton)
            self.connectLeftClickButtons()

        self.drawClearRegionToolbar.setVisible(checked)
        
        if not self.isSegm3D:
            self.drawClearRegionToolbar.setZslicesControlEnabled(False)
            return
        
        if not checked:
            return
        
        self.drawClearRegionToolbar.setZslicesControlEnabled(
            True, SizeZ=posData.SizeZ
        )

    def clearObjsFreehandRegion(self):
        self.logger.info('Clearing objects inside freehand region...')
        
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False, storeImage=False, storeOnlyZoom=True)
        
        posData = self.data[self.pos_i]
        zRange = None
        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if isZslice:
                z_slice = self.z_lab()
                zRange = self.drawClearRegionToolbar.zRange(
                    z_slice, posData.SizeZ
                )
            else:
                zRange = (0, posData.SizeZ)
            
        regionSlice = self.freeRoiItem.slice(zRange=zRange)
        mask = self.freeRoiItem.mask()
        
        regionLab = posData.lab[(...,) + regionSlice].copy()
        
        clearBorders = (
            self.drawClearRegionToolbar
            .clearOnlyEnclosedObjsRadioButton.isChecked()
        )
        if clearBorders:
            if regionLab.ndim == 2:
                regionLab = transformation.clear_objects_not_in_mask(
                    regionLab, mask
                )
                regionRp = skimage.measure.regionprops(regionLab)
                for obj in regionRp:
                    if np.all(mask[obj.slice][obj.image]):
                        continue
                    
                    regionLab[obj.slice][obj.image] = 0
            else:
                for z, regionLab_z in enumerate(regionLab):
                    regionLab[z] = transformation.clear_objects_not_in_mask(
                        regionLab_z, mask
                    )
        else:
            regionLab[..., ~mask] = 0
        
        regionRp = skimage.measure.regionprops(regionLab)
        clearIDs = [obj.label for obj in regionRp]
        
        if not clearIDs:
            if clearBorders:
                self.logger.warning(
                    'None of the objects in the freehand region are '
                    'fully enclosed'
                )
            else:
                self.logger.warning(
                    'None of the objects are touching the freehand region'
                )
            return
        
        self.deleteIDmiddleClick(clearIDs, False, False)
        self.update_cca_df_deletedIDs(posData, clearIDs)
        
        self.freeRoiItem.clear()
        
        self.updateAllImages()
