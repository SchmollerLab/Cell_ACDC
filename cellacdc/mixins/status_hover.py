"""View adapter for hover and status-bar formatting."""

from __future__ import annotations


import math
import os
import re

from .image_display import ImageDisplay


class StatusHover(ImageDisplay):
    """Extracted from guiWin."""

    def _addOverlayHoverValuesFormatted(self, txt, xdata, ydata):
        posData = self.data[self.pos_i]
        if posData.ol_data is None:
            return txt

        for filename in posData.ol_data:
            chName = utils.get_chname_from_basename(
                filename, posData.basename, remove_ext=False
            )
            if chName not in self.checkedOverlayChannels:
                continue

            raw_overlay_img = self.getRawImage(filename=filename)
            raw_overlay_value = raw_overlay_img[ydata, xdata]
            # raw_overlay_max_value = raw_overlay_img.max()

            raw_txt = self._channelHoverValues("Raw", chName, raw_overlay_value)

            txt = f"{txt} | {raw_txt}"
        return txt

    def _addRulerMeasurementText(self, txt):
        posData = self.data[self.pos_i]
        xx, yy = self.ax1_rulerPlotItem.getData()
        if xx is None:
            return txt

        lenPxl = math.sqrt((xx[0] - xx[1]) ** 2 + (yy[0] - yy[1]) ** 2)
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if depthAxes != "z":
            pxlToUm = posData.PhysicalSizeZ
        else:
            pxlToUm = posData.PhysicalSizeX

        length_txt = f"length = {int(lenPxl)} pxl ({lenPxl * pxlToUm:.2f} μm)"
        txt = f"{txt} | <b>Measurement</b>: {length_txt}"
        return txt

    def _channelHoverValues(self, descr, channel, value, ff=None):
        if ff is None:
            n_digits = len(str(int(value)))
            ff = utils.get_number_fstring_formatter(
                type(value), precision=abs(n_digits - 5)
            )
        txt = f"<b>{descr} {channel}</b>: value={value:{ff}}"
        return txt

    def getActiveToolButton(self):
        for button in self.LeftClickButtons:
            if button.isChecked():
                return button

    def updateValuesStatusBar(self):
        (xl, xr), (yt, yb) = self.ax1ViewRange(integers=True)
        W = round(xr - xl)
        H = round(yb - yt)
        txt = self.wcLabel.text()
        pattern = (
            r"W=.*?, H=.*? \| "
            r"x_left=.*?, y_top=.*? \| "
            r"x_right=.*?, y_bottom=.*? \| "
        )
        replacing = (
            f"W={W:d}, H={H:d} | "
            f"x_left={xl:d}, y_top={yt:d} | "
            f"x_right={xr:d}, y_bottom={yb:d} | "
        )
        txt = re.sub(pattern, replacing, txt)
        self.wcLabel.setText(txt)

    def hoverValuesFormatted(self, xdata, ydata, activeToolButton, is_ax0):
        (xl, xr), (yt, yb) = self.ax1ViewRange(integers=True)
        W = round(xr - xl)
        H = round(yb - yt)
        ax_idx = 0 if is_ax0 else 1
        txt = (
            f"x={xdata:d}, y={ydata:d} | "
            f"W={W:d}, H={H:d} | "
            f"x_left={xl:d}, y_top={yt:d} | "
            f"x_right={xr:d}, y_bottom={yb:d} | "
            f"(ax{ax_idx})"
        )
        if activeToolButton == self.rulerButton:
            txt = self._addRulerMeasurementText(txt)
            return txt
        elif activeToolButton is not None:
            return txt

        posData = self.data[self.pos_i]

        raw_img = self.getRawImage()
        raw_value = raw_img[ydata, xdata]
        # raw_max_value = raw_img.max()

        ch = self.user_ch_name
        raw_txt = self._channelHoverValues("Raw", ch, raw_value)

        txt = f"{txt} | {raw_txt}"

        txt = self._addOverlayHoverValuesFormatted(txt, xdata, ydata)

        ID = self.currentLab2D[ydata, xdata]
        maxID = max(posData.IDs, default=0)

        num_obj = len(posData.IDs)
        lab_txt = (
            f"<b>Objects</b>: ID={ID}, <i>max ID={maxID}, num. of objects={num_obj}</i>"
        )
        txt = f"{txt} | {lab_txt}"

        txt = self._addRulerMeasurementText(txt)
        return txt

    def setStatusBarLabel(self, log=True):
        self.statusbar.clearMessage()
        posData = self.data[self.pos_i]
        segmentedChannelname = posData.filename[len(posData.basename) :]
        segmFilename = os.path.basename(posData.segm_npz_path)
        segmEndName = segmFilename[len(posData.basename) :]
        txt = (
            f"{posData.pos_foldername} || "
            f"Basename: {posData.basename} || "
            f"Segmented channel: {segmentedChannelname} || "
            f"Segmentation file name: {segmEndName}"
        )
        mode = str(self.modeComboBox.currentText())
        if log:
            self.logger.info(txt)
        self.statusBarLabel.setText(txt)

    def getRulerLengthText(self):
        text = self.wcLabel.text()
        lengthText = re.findall(r"length = (.*)\)", text)[0]
        lengthText = lengthText.replace("pxl", "pixels")
        return f"{lengthText})"
