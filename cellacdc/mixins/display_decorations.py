"""View adapter for timestamp, scale-bar, and view-range decorations."""

from __future__ import annotations

import numpy as np

from cellacdc import apps, widgets


class DisplayDecorations:
    """Extracted from guiWin."""

    def addScaleBar(self, checked):
        if checked:
            posData = self.data[self.pos_i]
            Y, X = self.img1.image.shape[:2]
            viewRange = self.ax1ViewRange()
            self.scaleBarDialog = apps.ScaleBarPropertiesDialog(
                X, Y, posData.PhysicalSizeX, parent=self
            )
            self.scaleBarDialog.show()
            self.scaleBar = widgets.ScaleBar((Y, X), viewRange, parent=self.ax1)
            self.scaleBar.sigEditProperties.connect(self.editScaleBarProperties)
            self.scaleBar.sigRemove.connect(self.editScaleBarRemove)
            self.scaleBar.addToAxis(self.ax1)
            self.scaleBar.draw(**self.scaleBarDialog.kwargs())
            self.scaleBarDialog.sigValueChanged.connect(self.updateScaleBar)
            self.scaleBarDialog.exec_()
            if self.scaleBarDialog.cancel:
                self.addScaleBarAction.setChecked(False)
                return
        else:
            self.scaleBar.removeFromAxis(self.ax1)

        self.scaleBarDialog = None
        self.imgGrad.addScaleBarAction.setChecked(checked)

    def addTimestamp(self, checked):
        if checked:
            posData = self.data[self.pos_i]
            Y, X = self.img1.image.shape[:2]
            viewRange = self.ax1ViewRange()
            self.timestampDialog = apps.TimestampPropertiesDialog(parent=self)
            self.timestampDialog.show()
            self.timestamp = widgets.TimestampItem(
                Y,
                X,
                viewRange,
                secondsPerFrame=posData.TimeIncrement,
                start_timedelta=self.timestampStartTimedelta,
            )
            self.timestamp.sigEditProperties.connect(self.editTimestampProperties)
            self.timestamp.sigRemove.connect(self.editTimestampRemove)
            self.timestamp.addToAxis(self.ax1)
            self.timestamp.draw(posData.frame_i, **self.timestampDialog.kwargs())
            self.timestampDialog.sigValueChanged.connect(self.updateTimestamp)
            self.timestampDialog.exec_()
        else:
            self.timestamp.removeFromAxis(self.ax1)

        self.timestampDialog = None
        self.imgGrad.addTimestampAction.setChecked(checked)

    def ax1ViewRange(self, integers=False):
        if self.exportToImageWindow is None:
            viewRange = self.ax1.viewRange()
        else:
            exportMask = np.all(self.exportMaskImage == [0, 0, 0, 0], axis=-1)
            if np.all(exportMask):
                viewRange = self.ax1.viewRange()
            else:
                viewRange = self.ax1.viewRange(exportMask)

        if not integers:
            return viewRange

        xRange, yRange = viewRange
        xmin = round(xRange[0])
        ymin = round(yRange[0])
        xmax = round(xRange[1])
        ymax = round(yRange[1])
        return [xmin, xmax], [ymin, ymax]

    def getViewRange(self):
        Y, X = self.img1.image.shape[:2]
        xRange, yRange = self.ax1.viewRange()
        xmin = 0 if xRange[0] < 0 else xRange[0]
        ymin = 0 if yRange[0] < 0 else yRange[0]

        xmax = X if xRange[1] >= X else xRange[1]
        ymax = Y if yRange[1] >= Y else yRange[1]
        return int(ymin), int(ymax), int(xmin), int(xmax)

    def editScaleBarProperties(self, properties):
        Y, X = self.img1.image.shape[:2]
        posData = self.data[self.pos_i]
        self.scaleBarDialog = apps.ScaleBarPropertiesDialog(
            X, Y, posData.PhysicalSizeX, parent=self, **properties
        )
        self.scaleBarDialog.sigValueChanged.connect(self.updateScaleBar)
        self.scaleBarDialog.exec_()

    def editScaleBarRemove(self, timestamp):
        self.addScaleBarAction.setChecked(False)

    def editTimestampProperties(self, properties):
        self.timestampDialog = apps.TimestampPropertiesDialog(parent=self, **properties)
        self.timestampDialog.sigValueChanged.connect(self.updateTimestamp)
        self.timestampDialog.show()

    def editTimestampRemove(self, timestamp):
        self.addTimestampAction.setChecked(False)

    def viewRangeChanged(self, viewBox, viewRange, updateExportImageMask=True):
        # self.updateViewRangeExportToImage(viewRange)
        self.updateValuesStatusBar()

        if hasattr(self, "scaleBar"):
            isScaleBarMoveWithZoom = self.scaleBar.properties()["move_with_zoom"]
        else:
            isScaleBarMoveWithZoom = False
        doMoveScaleBar = self.scaleBarDialog is not None or isScaleBarMoveWithZoom
        if doMoveScaleBar:
            self.scaleBar.updatePosViewRangeChanged(viewRange)

        if hasattr(self, "timestamp"):
            isTimestampMoveWithZoom = self.timestamp.properties()["move_with_zoom"]
        else:
            isTimestampMoveWithZoom = False

        doMoveTimestamp = self.timestampDialog is not None or isTimestampMoveWithZoom
        if doMoveTimestamp:
            self.timestamp.updatePosViewRangeChanged(viewRange)

        self._viewRange = viewRange

    def updateScaleBar(self, scaleBarKwargs):
        self.scaleBar.draw(**scaleBarKwargs)

    def updateTimestamp(self, timeStampKwargs):
        posData = self.data[self.pos_i]
        self.timestamp.draw(posData.frame_i, **timeStampKwargs)

    def updateTimestampFrame(self):
        if not hasattr(self, "timestamp"):
            return

        if not self.addTimestampAction.isChecked():
            return

        posData = self.data[self.pos_i]
        self.timestamp.setText(posData.frame_i)
