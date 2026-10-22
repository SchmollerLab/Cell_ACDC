from . import printl

class GUICoreElements():
    # settings helper
    def _get_setting_value(self, index_name, default, cast=int):
        if self.df_settings is None or index_name not in self.df_settings.index:
            return default
        value = self.df_settings.at[index_name, 'value']
        if cast is bool:
            return value in ('Yes', 'True', '1') if isinstance(value, str) else bool(int(value))
        try:
            return cast(value)
        except Exception:
            printl(f"Warning: Could not cast setting '{index_name}' value '{value}' to {cast}. Using default '{default}'.")
            return default

    def _set_setting_value(self, index_name, value):
        if self.df_settings is None:
            return
        self.df_settings.at[index_name, 'value'] = value
        self.df_settings.to_csv(self.settings_csv_path)
        
    def get2DRP(
            self, 
            pos_i=None, 
            frame_i=None, 
            slice_i=None, 
            zProjHow=None, 
            depthAxes=None,
        ):
        """Get 2D regionprops for the current position and frame.

        This function returns the 2D regionprops for the specified position, frame, and slice.
        If the segmentation is 3D and a projection is requested, it returns the projected 2D regionprops.

        Parameters
        ----------
        pos_i : int, optional
            The index of the position to get the 2D regionprops for, by default None
        frame_i : int, optional
            The index of the frame to get the 2D regionprops for, by default None
        slice_i : int, optional
            The index of the slice to get the 2D regionprops for, by default None
        zProjHow : str, optional
            The type of z-projection to use ('single z-slice' or other), by default None
        depthAxes : tuple, optional
            The axes to use for depth slicing, by default None

        Returns
        -------
        acdc.RegionProps2D
            The 2D regionprops object for the specified position, frame, and slice.
        """
        posData = self.data[self.pos_i if pos_i is None else pos_i]
        frame_i = posData.frame_i if frame_i is None else frame_i

        if frame_i == posData.frame_i:
            rp = posData.rp
        else:
            rp = posData.allData_li[frame_i]['regionprops']

        if rp is None:
            self.logger.warning(
                f"[WARNING] rp for displaying contours was requested before being set properly"
            )
            if frame_i == posData.frame_i:
                lab = posData.lab
            else:
                lab = posData.allData_li[frame_i]['labels']
            rp = self._acdcRegionProps(lab, precache_centroids=False)
            if frame_i == posData.frame_i:
                posData.rp = rp
            posData.allData_li[frame_i]['regionprops'] = rp
        if not self.isSegm3D:
            return rp

        if zProjHow is None:
            zProjHow = self.zProjComboBox.currentText()
        if depthAxes is None:
            depthAxes = self.switchPlaneCombobox.depthAxes()
        
        if zProjHow != 'single z-slice':
            rp_2D = rp.get_proj_rp('most_common', slicing=depthAxes)
            return rp_2D

        if slice_i is None:
            slice_i = self.zSliceScrollBar.sliderPosition()

        rp_2D = rp.get_slice_rp(slice_i, depthAxes)
        return rp_2D