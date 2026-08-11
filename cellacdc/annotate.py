import os
import traceback

import numpy as np
import pandas as pd

from . import GUI_INSTALLED
from . import cellacdc_path, printl, ignore_exception, debugutils

import math

if GUI_INSTALLED:
    from PIL import Image, ImageFont, ImageDraw
    from qtpy.QtGui import QFont
    import pyqtgraph as pg
    pg.setConfigOption('imageAxisOrder', 'row-major')
    
    from . import plot
    
    from qtpy.QtGui import QFont, QPicture, QPainter, QColor, QPen, QImage, QTransform
    from qtpy.QtCore import QRectF, QPointF, Qt, QLineF

INVERTIBLE_COLOR_NAMES = [
    'label', 'S_phase_mother', 'G1_phase'
]
FONT_FAMILY = 'Helvetica'
font_path = os.path.join(
    cellacdc_path, 'resources', 'fonts', f'{FONT_FAMILY}-Regular.ttf')
font_bold_path = os.path.join(
    cellacdc_path, 'resources', 'fonts', f'{FONT_FAMILY}-Bold.ttf'
)

ZOOM_BUCKET_PERC_CHANGE_THRESHOLD = 0.1 # 0.5 would be the lowest 
# percentage where it would do something (every second step), but IDK if its 
# really worth it

def get_obj_text_label_annot(
        obj, add_num_zslices: bool
    ) -> str:
    annot_label = obj.label
    if not add_num_zslices:
        return str(annot_label)
    
    num_z_slices = np.sum(np.any(obj.image, axis=(1,2)))
    return f'{annot_label} ({num_z_slices})'

def get_obj_text_cca_annot(
        obj, acdc_df: pd.DataFrame,
        moth_bud_pairs_cca=None,
        is_tree_annot: bool=False,
    ) -> str:
    
    ID = obj.label
    if not is_tree_annot:
        if moth_bud_pairs_cca is not None and ID in moth_bud_pairs_cca.index:
            cca_df_obj = moth_bud_pairs_cca.loc[ID]
        elif ID in acdc_df.index:
            cca_df_obj = acdc_df.loc[ID]
        else:
            return str(ID), None
        
        if 'cell_cycle_stage' not in cca_df_obj:
            return str(ID), None
        ccs = cca_df_obj['cell_cycle_stage']

        if 'generation_num' not in cca_df_obj:
            return str(ID), None
        generation_num = int(cca_df_obj['generation_num'])
    
    else:
        if ID not in acdc_df.index:
            return str(ID), None
        cca_df_obj = acdc_df.loc[ID]
        
        if 'Cell_ID_tree' in cca_df_obj:
            displ_ID = cca_df_obj['Cell_ID_tree']
        else:
            displ_ID = ID
        
        if 'generation_num_tree' not in cca_df_obj:
            return str(displ_ID), None
        generation_num = cca_df_obj['generation_num_tree']
        if generation_num is None or pd.isna(generation_num):
            generation_num = -1
        generation_num = int(float(generation_num))
    
    generation_num = 'ND' if generation_num==-1 else generation_num

    if not is_tree_annot:
        txt = f'{ccs}-{generation_num}'
    else:
        txt = f'{displ_ID} ({generation_num})'

    is_history_known = cca_df_obj['is_history_known']
    if pd.notna(is_history_known) and not is_history_known:
        txt = f'{txt}?'

    return txt, cca_df_obj

def get_obj_color_cca_annot(cca_df_obj, frame_i):    
    ccs = cca_df_obj['cell_cycle_stage']
    relationship = cca_df_obj['relationship']
    is_bud = relationship == 'bud'
    emerg_frame_i = int(cca_df_obj['emerg_frame_i'])
    bud_emerged_now = (emerg_frame_i == frame_i) and is_bud

    bold = bud_emerged_now

    # Check if it will divide to use orange instead of red
    bud_will_divide = False
    if ccs == 'S' and is_bud:
        bud_will_divide = cca_df_obj['will_divide'] > 0

    if bud_will_divide:
        color_name = 'bud_will_divide'
    elif ccs == 'S':
        if relationship == 'mother':
            color_name = 'S_phase_mother'
        else:
            color_name = 'S_phase_bud'
    elif ccs == 'G1':
        color_name = 'G1_phase'
        
    return color_name, bold

def get_obj_text_annot_opts(
        obj, acdc_df: pd.DataFrame, is_cca_annot: bool, is_new_obj: bool, 
        add_num_zslices: bool, is_lineage_annot: bool, 
        frame_i: int, moth_bud_pairs_cca=None
    ) -> dict: 
    if acdc_df is not None and is_cca_annot:
        text, cca_df_obj = get_obj_text_cca_annot(
            obj, acdc_df,
            moth_bud_pairs_cca=moth_bud_pairs_cca,
            is_tree_annot=False
        )
        if cca_df_obj is None:
            if is_new_obj:
                color_name = 'new_object'
            else:
                color_name = 'label'
            opts = {'text': text, 'color_name': color_name, 'bold': False}
            return opts
        
        color_name, bold = get_obj_color_cca_annot(cca_df_obj, frame_i)
            
    elif acdc_df is not None and is_lineage_annot:
        text, cca_df_obj = get_obj_text_cca_annot(
            obj, acdc_df,
            is_tree_annot=True
        )
        if cca_df_obj is None:
            if is_new_obj:
                color_name = 'new_object'
            else:
                color_name = 'label'
            opts = {'text': text, 'color_name': color_name, 'bold': False}
            return opts
        
        bold = is_new_obj and (cca_df_obj['parent_ID_tree'] == -1)
        if is_new_obj:
            color_name = 'new_object'
        else:
            color_name = 'label'
        
    else:
        bold = False
        if is_new_obj:
            color_name = 'new_object'
        else:
            color_name = 'label'
        text = get_obj_text_label_annot(
            obj, add_num_zslices
        )
        
    opts = {'text': text, 'color_name': color_name, 'bold': bold}

    return opts

class TextAnnotationsImageItem(pg.ImageItem):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.texts = []
        self.annotData = []
        self.highlighterItem = None
    
    def initFonts(self, fontSize):
        self.fontSize = fontSize
        self.fontRegular = ImageFont.truetype(font_path, fontSize)
        self.fontBold = ImageFont.truetype(font_bold_path, fontSize)
        if self.highlighterItem is None:
            self.highlighterItem = TextAnnotationsScatterItem(
                size=self.fontSize, pxMode=False
            )
        self.highlighterItem.initFonts(fontSize)
        self.highlighterItem.initSymbols(range(10))
    
    def initSizes(self):
        pass
    
    def init(self, image_shape):
        shape = (*image_shape, 4)
        self.pilImage = Image.fromarray(np.zeros(shape, dtype=np.uint8))
        self.pilDraw = ImageDraw.Draw(self.pilImage)
    
    def clearImage(self):
        try:
            self.pilDraw.rectangle([(0,0), self.pilDraw.im.size], fill=(0,0,0,0))
        except Exception as e:
            pass
    
    def clearData(self):
        self.clearImage()
        self.setOpacity(1.0)
        if self.highlighterItem is not None:
            self.highlighterItem.clearData()
        self.texts = []
        self.annotData = []

    def clear(self):
        self.clearData()
        self.setVisible(False)
    
    def update(self):
        super().update()
    
    def appendData(self, data, text):
        self.annotData.append(data)
        self.texts.append(text)
    
    def highlightObject(self, obj, rp=None, getObjCentroidFunc=None):
        self.highlighterItem.texts = self.texts
        self.highlighterItem.highlightObject(
            obj, rp=rp, getObjCentroidFunc=getObjCentroidFunc)
    
    def grayOutAnnotations(self, IDsToSkip=None):
        self.setOpacity(0.3)
    
    def addObjAnnot(self, pos, draw=True, **objOpts):
        if objOpts['bold']:
            font = self.fontBold
        else:
            font = self.fontRegular
        
        text = objOpts['text']
        color = self._colors[objOpts['color_name']]
        self.pilDraw.text(pos, text, color, font=font, anchor='mm')
        return objOpts    
    
    def draw(self):
        super().setImage(np.array(self.pilImage))

    def setColors(self, colors):
        self._colors = colors.copy()
        if self.highlighterItem is not None:
            self.highlighterItem.setColors(colors)
    
    def initSymbols(self, allIDs):
        pass

    def colors(self):
        return self._colors

class TextAnnotationsScatterItem(pg.GraphicsObject):
    """
    Draws ID/annotation text directly with QPainter.drawText() into a
    single cached QPicture, instead of building one unique vector-path
    symbol per text label (as ScatterPlotItem's SymbolAtlas does).

    This keeps the same public API as the old TextAnnotationsScatterItem
    (addObjAnnot, appendData, draw, highlightObject, removeHighlightObject,
    grayOutAnnotations, setColors, etc.) so it's a drop-in swap.
    """
    def __init__(self, size=10, pxMode=False, anchor=(0.5, 0.5), scalingMode=False):
        super().__init__()
        self._pxMode = pxMode
        self._scaling = scalingMode
        self._exporting = False
        self._exportTargetZoom = None
        self._forcedExportZoom = None
        self._anchor = anchor
        self.picture = QPicture()
        self._boundingRect = QRectF()
        self.annotData = []  # list of dicts: pos, text, bold, color_name, data(ID)
        self.texts = []      # kept parallel to annotData for API compatibility
        self.initFonts(size)
        self.setPxMode(pxMode)
        self.zoom = None
        self._zoomBucket = None
        self._cachedScaleFactor = None
        self.cached_picture = None
        self._cached_view_rect = QRectF()
        self._curr_zoom_bucket = None
        self._rectsZoomBucket = None
        self._pictureTargetZoom = None
        
    # ---- setup / config, API-compatible no-ops where the atlas is gone ----

    def initFonts(self, fontSize):
        self.fontSize = fontSize
        
        self.fontBold = QFont(FONT_FAMILY)
        self.fontBold.setBold(True)
        self.fontBold.setPixelSize(fontSize)

        self.fontRegular = QFont(FONT_FAMILY)
        self.fontRegular.setPixelSize(fontSize)
        
        self._invalidateCache()

    def init(self, *args):
        pass

    def initSymbols(self, allIDs, onlyIDs=False):
        # No pre-built symbol atlas needed with drawText() rendering.
        pass

    def addSymbols(self, annotTexts, includeBold=True):
        pass

    def createSymbols(self, annotTexts, includeBold=True):
        pass

    def initSizes(self, includeBold=True):
        pass

    def setPxMode(self, mode):
        self._pxMode = mode
        # This item renders many labels in one shared picture in data coordinates.
        # Ignoring view transforms for the whole item breaks label positions.
        self.setFlag(
            pg.GraphicsObject.GraphicsItemFlag.ItemIgnoresTransformations, False
        )

    def setScaling(self, scaling):
        scaling_new = bool(scaling)
        if self._scaling == scaling_new:
            return
        self._scaling = scaling_new
        self._invalidateCache()

    def setExportZoom(self, zoom):
        if zoom is None:
            self._forcedExportZoom = None
            return
        try:
            zoom = float(zoom)
        except (TypeError, ValueError):
            self._forcedExportZoom = None
            return
        self._forcedExportZoom = zoom if zoom > 0 else None

    def setExportMode(self, export, opts=None):
        self._exporting = bool(export)
        if not export:
            self._exportTargetZoom = None
            return

        # Capture zoom before export rendering starts. During SVG item painting
        # the viewbox can be unavailable, so this keeps scaling behavior stable.
        self._exportTargetZoom = self._forcedExportZoom
        if self._exportTargetZoom is None:
            self._exportTargetZoom = self._currentViewZoom()

        # Exporters provide a painter with a different transform stack than the
        # interactive view. The cached subset/image path reconstructs that
        # transform approximately and can shift or mis-scale text, so export
        # should always use direct vector/text drawing instead.
        self._invalidateCache()

    def setColors(self, colors):
        self._colors = colors.copy()
        self._brushes = {}
        self._pens = {}
        for name, color in self._colors.items():
            self._brushes[name] = pg.mkBrush(color)
            self._pens[name] = pg.mkPen(color[:3], width=1)
        self._invalidateCache()

    def pens(self):
        return self._pens

    def brushes(self):
        return self._brushes

    def colors(self):
        return self._colors

    # ---- data management ----

    def clearData(self):
        self.annotData = []
        self.texts = []
        self._generatePicture()

    def appendData(self, data, text):
        self.annotData.append(data)
        self.texts.append(text)

    def addObjAnnot(self, pos, draw=False, anchor=None, **objOpts):
        objData = {
            'pos': tuple(pos),
            'text': objOpts['text'],
            'bold': objOpts.get('bold', False),
            'color_name': objOpts['color_name'],
        }
        if draw:
            self.annotData.append(objData)
            self.texts.append(objData['text'])
            self._generatePicture()
        return objData

    def draw(self):
        target_zoom = self._currentViewZoom() if self._scaling else self.zoom
        self._generatePicture(target_zoom=target_zoom)

    # ---- rendering ----

    def _invalidateCache(self):
        self.zoom = None
        self._zoomBucket = None
        self.cached_picture = None
        self._cached_view_rect = QRectF()

    def _effective_font_size(self, target_zoom):
        if target_zoom is None:
            return self.fontSize
        if self._scaling:
            return max(self.fontSize * target_zoom, 1.0)
        return self.fontSize

    def _font_for(self, objData, target_zoom=None):
        font_size = self._effective_font_size(target_zoom)
        base_font = self.fontBold if objData.get('bold') else self.fontRegular
        font = QFont(base_font)
        font.setPixelSize(max(int(round(font_size)), 1))
        return font

    def _currentViewZoom(self):
        viewbox = self.getViewBox()
        if viewbox is None:
            return None
        return viewbox.viewPixelSize()[0]

    def _draw_pos_for(self, painter, objData, text):
        x, y = objData['pos']
        fm = painter.fontMetrics()
        rect = fm.boundingRect(text)
        return QPointF(
            x - rect.width() / 2,
            y + rect.height() / 2 - fm.descent(),
        ), rect, fm


    def _zoomBucketFor(self, target_zoom):
        if target_zoom is None:
            return None
        
        # if zoom becomes excessive, we don't want to zoom in too deep
        effective_font_size = self._effective_font_size(target_zoom)
        effective_rendered_font_px = effective_font_size / target_zoom
        if effective_rendered_font_px > 2500:
            if self._curr_zoom_bucket is not None:
                return self._curr_zoom_bucket
            
        rendered_font_px = self.fontSize / target_zoom
        requested_bucket = max(int(round(rendered_font_px / 2.0) * 2), 1)
        if self._curr_zoom_bucket is None:
            self._curr_zoom_bucket = requested_bucket
            return self._curr_zoom_bucket
        
        perc_change = abs(requested_bucket - self._curr_zoom_bucket) / self._curr_zoom_bucket
        if perc_change > ZOOM_BUCKET_PERC_CHANGE_THRESHOLD:
            self._curr_zoom_bucket = requested_bucket
        return self._curr_zoom_bucket
    # zoom buckets are basically for 2 step font size changes so they are 
    # always even

    def _expandedCacheRect(self, exposed_rect):
        if exposed_rect.isNull():
            return QRectF()
        x_margin = max(exposed_rect.width() * 0.25, 1.0)
        y_margin = max(exposed_rect.height() * 0.25, 1.0)
        expanded_rect = exposed_rect.adjusted(
            -x_margin, -y_margin, x_margin, y_margin
        )
        return expanded_rect.intersected(self._boundingRect)

    def _drawSubset(self, painter, source_rect=None, target_zoom=None):
        if source_rect is not None and source_rect.isNull():
            return

        for objData in self.annotData:
            text = objData.get('text')
            if text is None:
                continue

            font = self._font_for(objData, target_zoom=target_zoom)
            painter.setFont(font)
            draw_pos, rect, fm = self._draw_pos_for(painter, objData, text)
            text_rect = QRectF(
                draw_pos.x(),
                draw_pos.y() - rect.height() + fm.descent(),
                rect.width(),
                rect.height(),
            )
            if source_rect is not None and not text_rect.intersects(source_rect):
                continue

            color = self._colors.get(objData.get('color_name'), (255, 255, 255, 255))
            painter.setPen(QColor(*color))
            painter.drawText(draw_pos, text)

    def _textRectForObj(self, painter, objData, text, target_zoom=None):
        painter.setFont(self._font_for(objData, target_zoom=target_zoom))
        draw_pos, rect, fm = self._draw_pos_for(painter, objData, text)
        return QRectF(
            draw_pos.x(),
            draw_pos.y() - rect.height() + fm.descent(),
            rect.width(),
            rect.height(),
        )

    def _doesAnyTextTouchCurrentBounds(self, painter, target_zoom=None):
        if self._boundingRect.isNull():
            return True

        left = self._boundingRect.left()
        right = self._boundingRect.right()
        top = self._boundingRect.top()
        bottom = self._boundingRect.bottom()
        eps = 0.5

        for idx, objData in enumerate(self.annotData):
            text = objData.get('text')
            if text is None and idx < len(self.texts):
                text = self.texts[idx]
            if not text:
                continue

            text_rect = self._textRectForObj(
                painter, objData, text, target_zoom=target_zoom
            )
            touches_outline = (
                text_rect.left() <= left + eps
                or text_rect.right() >= right - eps
                or text_rect.top() <= top + eps
                or text_rect.bottom() >= bottom - eps
            )
            if touches_outline:
                return True

        return False

    def _cachePaddingFor(self, painter, target_zoom=None):
        # Use measured text extents so long labels are not clipped when
        # the cache is generated for only a subset of the visible area.
        max_half_width = 0.0
        max_half_height = 0.0
        for objData in self.annotData:
            text = objData.get('text')
            if not text:
                continue

            painter.setFont(self._font_for(objData, target_zoom=target_zoom))
            rect = painter.fontMetrics().boundingRect(text)
            max_half_width = max(max_half_width, rect.width() / 2)
            max_half_height = max(max_half_height, rect.height() / 2)

        base_pad = max(self._effective_font_size(target_zoom), 1.0)
        x_pad = max(base_pad, max_half_width + 2.0)
        y_pad = max(base_pad, max_half_height + 2.0)
        return x_pad, y_pad

    def _refreshCachedPicture(self, painter, target_zoom, source_rect, zoom_bucket):
        if (
            self._boundingRect.isNull()
            or source_rect.isNull()
            or target_zoom is None
        ):
            self.cached_picture = None
            self._cached_view_rect = QRectF()
            self._zoomBucket = None
            return

        x_pad, y_pad = self._cachePaddingFor(painter, target_zoom=target_zoom)
        draw_rect = source_rect.adjusted(-x_pad, -y_pad, x_pad, y_pad)

        device_rect = painter.worldTransform().mapRect(draw_rect)
        logical_width = abs(device_rect.width())
        logical_height = abs(device_rect.height())
        if logical_width <= 0 or logical_height <= 0:
            self.cached_picture = None
            self._cached_view_rect = QRectF()
            self._zoomBucket = None
            return

        device = painter.device()
        if device is not None and hasattr(device, 'devicePixelRatioF'):
            dpr = float(device.devicePixelRatioF())
        else:
            dpr = 1.0
        dpr = max(dpr, 1.0)

        width = max(int(round(logical_width * dpr)), 1)
        height = max(int(round(logical_height * dpr)), 1)

        cached_picture = QImage(
            width, height, QImage.Format.Format_ARGB32_Premultiplied
        )
        cached_picture.fill(0)

        cache_painter = QPainter(cached_picture)
        if not cache_painter.isActive():
            self.cached_picture = None
            self._cached_view_rect = QRectF()
            self._zoomBucket = None
            return

        cache_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale_x = (logical_width * dpr) / draw_rect.width()
        scale_y = (logical_height * dpr) / draw_rect.height()
        transform = QTransform()
        transform.scale(scale_x, scale_y)
        transform.translate(-draw_rect.left(), -draw_rect.top())
        cache_painter.setWorldTransform(transform)
        self._drawSubset(cache_painter, draw_rect, target_zoom=target_zoom)
        cache_painter.end()

        self.cached_picture = cached_picture
        self._cached_view_rect = draw_rect
        self._zoomBucket = zoom_bucket
        self.zoom = target_zoom

    def _generatePicture(self, target_zoom=None):
        if target_zoom is None:
            if self._scaling:
                target_zoom = self._currentViewZoom()
            if target_zoom is None:
                target_zoom = self.zoom

        self.picture = QPicture()
        painter = QPainter(self.picture)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rects = []
        for idx, objData in enumerate(self.annotData):
            text = objData.get('text')
            if text is None and idx < len(self.texts):
                text = self.texts[idx]

            font = self._font_for(objData, target_zoom=target_zoom)
            painter.setFont(font)
            color = self._colors.get(objData.get('color_name'), (255, 255, 255, 255))
            painter.setPen(QColor(*color))

            # Center the text on (x, y), matching the 'mm' anchor used by
            # the low-res PIL-based item.
            draw_pos, rect, fm = self._draw_pos_for(painter, objData, text)
            painter.drawText(draw_pos, text)

            objData['_draw_pos'] = draw_pos
            objData['_rect'] = QRectF(
                draw_pos.x(),
                draw_pos.y() - rect.height() + fm.descent(),
                rect.width(),
                rect.height(),
            )
            rects.append(objData['_rect'])

        painter.end()

        if rects:
            pad = max(2, self.fontSize // 4)
            bounds = QRectF(rects[0])
            for rect in rects[1:]:
                bounds = bounds.united(rect)
            self._boundingRect = bounds.adjusted(-pad, -pad, pad, pad)
        else:
            self._boundingRect = QRectF()
        self._rectsZoomBucket = self._zoomBucketFor(target_zoom)
        self._pictureTargetZoom = target_zoom
        self._invalidateCache()
        self.prepareGeometryChange()
        self.update()

    def paint(self, painter, *args):
        if self._boundingRect.isNull() or not self.annotData:
            return

        if self._exporting:
            target_zoom = None
            if self._scaling:
                target_zoom = self._exportTargetZoom
                if target_zoom is None:
                    target_zoom = self._currentViewZoom()

            # Draw directly in export mode so font sizing is evaluated against
            # the export painter state instead of replayed picture commands.
            self._drawSubset(painter, source_rect=None, target_zoom=target_zoom)
            return

        if self.picture.isNull():
            self._generatePicture()
            if self.picture.isNull():
                return

        viewbox = self.getViewBox()
        target_zoom = viewbox.viewPixelSize()[0] if viewbox is not None else None
        zoom_bucket = self._zoomBucketFor(target_zoom)
        bucket_changed = (
            self._scaling
            and (
                (zoom_bucket is not None and self._rectsZoomBucket != zoom_bucket)
                or self._rectsZoomBucket is None
            )
        )
        if bucket_changed:
            prev_bucket = self._rectsZoomBucket
            is_zooming_out = (
                prev_bucket is None
                or zoom_bucket < prev_bucket
            )
            if is_zooming_out and self._doesAnyTextTouchCurrentBounds(
                painter, target_zoom=target_zoom
            ):
                # Rebuild only when zooming out pushes text against the
                # current annotation bounds.
                self._generatePicture(target_zoom=target_zoom)
            else:
                # No boundary risk at this bucket: avoid costly full redraw.
                self._rectsZoomBucket = zoom_bucket
                self._pictureTargetZoom = target_zoom

        render_zoom = target_zoom
        if self._scaling and self._pictureTargetZoom is not None:
            # If we skipped picture regeneration, keep rendering text at the
            # existing picture zoom to accept pixelation without geometry
            # mismatches that can hide labels.
            render_zoom = self._pictureTargetZoom

        visible_rect = (
            viewbox.viewRect().intersected(self._boundingRect)
            if viewbox is not None else self._boundingRect
        )
        if visible_rect.isNull():
            return

        if (
            self.cached_picture is None
            or zoom_bucket is None
            or self._zoomBucket != zoom_bucket
            or not self._cached_view_rect.contains(visible_rect)
        ):
            cache_rect = self._expandedCacheRect(visible_rect)
            self._refreshCachedPicture(painter, render_zoom, cache_rect, zoom_bucket)

        if self.cached_picture is None:
            painter.drawPicture(0, 0, self.picture)
            return

        cache_width = self.cached_picture.width()
        cache_height = self.cached_picture.height()
        source_rect = QRectF(
            ((visible_rect.left() - self._cached_view_rect.left())
             / self._cached_view_rect.width()) * cache_width,
            ((visible_rect.top() - self._cached_view_rect.top())
             / self._cached_view_rect.height()) * cache_height,
            (visible_rect.width() / self._cached_view_rect.width()) * cache_width,
            (visible_rect.height() / self._cached_view_rect.height()) * cache_height,
        )
        painter.drawImage(visible_rect, self.cached_picture, source_rect)

    def boundingRect(self):
        return self._boundingRect

    # ---- highlight / gray-out ----

    def highlightObject(self, obj, rp=None, getObjCentroidFunc=None):
        ID = obj.label
        objIdx = next(
            (i for i, d in enumerate(self.annotData) if d.get('data') == ID), None
        )
        if objIdx is None:
            centroid = rp.get_centroid(obj.label) if rp is not None else obj.centroid
            yc, xc = (
                getObjCentroidFunc(centroid) if getObjCentroidFunc is not None
                else centroid[-2:]
            )
            objData = self.addObjAnnot(
                (int(xc), int(yc)), draw=False,
                text=str(ID), bold=True, color_name='new_object'
            )
            objData['data'] = ID
            self.annotData.append(objData)
            self.texts.append(objData['text'])
            self._generatePicture()
            return

        self.annotData[objIdx]['color_name'] = 'new_object'
        self.annotData[objIdx]['bold'] = True
        self._generatePicture()

    def removeHighlightObject(self, obj):
        ID = obj.label
        objIdx = next(
            (i for i, d in enumerate(self.annotData) if d.get('data') == ID), None
        )
        if objIdx is None:
            return
        self.annotData[objIdx]['color_name'] = 'label'
        self.annotData[objIdx]['bold'] = False
        self._generatePicture()

    def grayOutAnnotations(self, IDsToSkip=None):
        for objData in self.annotData:
            doNotGray = (
                IDsToSkip.get(objData.get('data'), False)
                if IDsToSkip is not None else False
            )
            if not doNotGray:
                objData['color_name'] = 'grayed'
        self._generatePicture()
        
class TextAnnotations:
    def __init__(self):
        self._isLabelAnnot = False
        self._isCcaAnnot = False
        self._isAnnotateNumZslices = False
        self._isLineageAnnot = False
    
    def initFonts(self, fontSize):
        self.fontSize = fontSize
    
    def initItem(self, *args):
        self.item.init(*args)
    
    def clear(self):
        self.item.clear()
        if hasattr(self.item, 'highlighterItem') and self.item.highlighterItem is not None:
            self.item.highlighterItem.clearData()
    
    def invertBlackAndWhite(self):
        invertedColors = {
            name:color[:3] for name, color in self.item.colors().items()
        }
        for color_name in INVERTIBLE_COLOR_NAMES:
            color = self.item.colors()[color_name]
            invertedColors[color_name] = tuple([255-val for val in color[:3]])

        self.setColors(**invertedColors)

    def createItems(self, isHighResolution, allIDs, pxMode=False, scalingMode=False):
        self._pxMode = pxMode
        if isHighResolution:
            self._createHighResolutionItems(allIDs, pxMode=pxMode, scalingMode=scalingMode)
        else:
            self._createLowResolutionItem()        
        
    def _createLowResolutionItem(self):
        self.item = TextAnnotationsImageItem()
        self.setFontSize(self.fontSize, [])
    
    def _createHighResolutionItems(self, allIDs, pxMode=False, scalingMode=False):
        self.item = TextAnnotationsScatterItem(
            size=self.fontSize, pxMode=pxMode, scalingMode=scalingMode
        )
        self.setFontSize(self.fontSize, allIDs)
    
    def setFontSize(self, fontSize, allIDs):
        self.fontSize = fontSize
        self.item.initFonts(self.fontSize)
        self.item.initSymbols(allIDs)
    
    def changeFontSize(self, fontSize):
        self.fontSize = fontSize
        self.item.initFonts(fontSize)
        self.item.initSizes()
  
    def changeResolution(self, mode, allIDs, ax, img_shape, scalingMode=False):
        self.removeFromPlotItem(ax)
        highRes = True if mode == 'high' else False        
        self.createItems(highRes, allIDs, pxMode=self._pxMode, scalingMode=scalingMode)
        self.initItem(img_shape)
        self.item.setColors(self.colors())
        self.item.clearData()
        self.addToPlotItem(ax)
    
    def addToPlotItem(self, ax):
        ax.addItem(self.item)
        if hasattr(self.item, 'highlighterItem'):
            ax.addItem(self.item.highlighterItem)

    def removeFromPlotItem(self, ax):
        ax.removeItem(self.item)
        if hasattr(self.item, 'highlighterItem'):
            ax.removeItem(self.item.highlighterItem)
    
    def addObjAnnotation(self, obj, color_name, text, bold, rp=None, getObjCentroidFunc=None):
        objOpts = {
            'text': text,
            'bold': bold,
            'color_name': color_name,
        }
        if rp is not None:
            centroid = rp.get_centroid(obj.label)
        else:
            centroid = obj.centroid
        if getObjCentroidFunc is not None:
            yc, xc = getObjCentroidFunc(centroid)
        else:
            yc, xc = centroid[-2:]
        pos = (int(xc), int(yc))
        objData = self.item.addObjAnnot(pos, draw=True, **objOpts)
        self.item.appendData(objData, objOpts['text'])
    
    def setAnnotations(self, **kwargs):
        if self.isDisabled():
            self.item.setVisible(False)
            return

        if isinstance(self.item, TextAnnotationsImageItem):
            self.item.clearImage()
            self.item.setOpacity(1.0)
            if self.item.highlighterItem is not None:
                self.item.highlighterItem.clearData()

        annotData = []
        texts = []
        
        labelsToSkip = kwargs.get('labelsToSkip')
        posData = kwargs['posData']
        delROIsIDs = kwargs.get('delROIsIDs', [])
        isObjVisibleFunc = kwargs.get('isVisibleCheckFunc')
        highlightedID = kwargs.get('highlightedID')
        annotateLost = kwargs.get('annotateLost')
        getObjCentroidFunc = kwargs.get('getObjCentroidFunc')
        isCcaAnnot = self.isCcaAnnot()
        isAnnotateNumZslices = self.isAnnotateNumZslices()
        isLineageAnnot = self.isLineageAnnot()
        rp_func = kwargs.get('rp_func')
        rp3D = kwargs.get('rp3D')
        updateAllTextAnnotations = kwargs.get('updateAllTextAnnotations', True)
        
        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
        if posData.cca_df is not None and acdc_df is not None:
            cols = posData.cca_df.columns
            idx = posData.cca_df.index.intersection(acdc_df.index)
            acdc_df.loc[idx, cols] = posData.cca_df
        
        if acdc_df is None and posData.cca_df is not None:
            acdc_df = posData.cca_df
    
        moth_bud_pairs_cca = (
            posData.allData_li[posData.frame_i].get('moth_bud_pairs_cca')
        )

        rp = rp_func()
        for obj in rp:
            if labelsToSkip is not None:
                if labelsToSkip.get(obj.label, False):
                    continue
            
            obj3D = rp3D.get_obj_from_ID(obj.label)
            if obj3D is not None and not isObjVisibleFunc(obj3D.bbox):
                continue
            
            if obj.label in delROIsIDs:
                continue
                
            yc, xc = rp.get_centroid(obj.label)
                
            pos = (int(xc), int(yc))
            
            isNewObject = obj.label in posData.new_IDs
            if not isNewObject and not updateAllTextAnnotations:
                continue
            
            objOpts = get_obj_text_annot_opts(
                obj, acdc_df, isCcaAnnot, isNewObject,
                isAnnotateNumZslices, isLineageAnnot,
                posData.frame_i,
                moth_bud_pairs_cca=moth_bud_pairs_cca
            )
            
            objData = self.item.addObjAnnot(pos, draw=False, **objOpts)
            objData['data'] = obj.label
            annotData.append(objData)
            texts.append(objOpts['text'])

        if posData.trackedLostIDs and annotateLost:
            prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
            if prev_rp is None:
                prev_rp = ()
            
            for obj in prev_rp:
                if obj.label not in posData.trackedLostIDs:
                    continue

                if obj.label in delROIsIDs:
                    continue
                
                if not isObjVisibleFunc(obj.bbox):
                    continue

                objOpts = {
                    'text': f'{obj.label}',
                    'color_name': 'tracked_lost_object',
                    'bold': False,
                }
                centroid = prev_rp.get_centroid(obj.label)
                yc, xc = getObjCentroidFunc(centroid)
                pos = (int(xc), int(yc))
                objData = self.item.addObjAnnot(pos, draw=False, **objOpts)
                annotData.append(objData)
                texts.append(objOpts['text'])


        if posData.lost_IDs and annotateLost:
            prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
            if prev_rp is None:
                prev_rp = ()
            for obj in prev_rp:
                if obj.label not in posData.lost_IDs:
                    continue
                
                if obj.label in delROIsIDs:
                    continue
                
                if not isObjVisibleFunc(obj.bbox):
                    continue
                
                objOpts = {
                    'text': f'{obj.label}?',
                    'color_name': 'lost_object',
                    'bold': False,
                }
                centroid = prev_rp.get_centroid(obj.label)
                yc, xc = getObjCentroidFunc(centroid)
                try:
                    pos = (int(xc), int(yc))
                except Exception as err:
                    printl("""WARNING: Could not annotate lost object, failed 
                           to get position. Skipping annotation.""")
                    # Sometimes xc or yc can be nan, causing an error when 
                    # converting to int --> skip annotation in this case
                    continue
                objData = self.item.addObjAnnot(pos, draw=False, **objOpts)
                annotData.append(objData)
                texts.append(objOpts['text'])

        if not annotData:
            self.item.annotData = []
            self.item.texts = []
            self.item.setVisible(False)
            return

        self.item.annotData = annotData
        self.item.texts = texts
        self.item.setVisible(True)
        self.item.draw()
    
    def highlightObject(self, obj, rp=None, getObjCentroidFunc=None):
        self.item.highlightObject(obj, rp=rp, getObjCentroidFunc=getObjCentroidFunc)
    
    def removeHighlightObject(self, obj):
        self.item.removeHighlightObject(obj)
    
    def grayOutAnnotations(self, IDsToSkip=None):
        self.item.grayOutAnnotations(IDsToSkip=IDsToSkip)

    def isDisabled(self):
        _isEnabled = self._isLabelAnnot or self._isCcaAnnot or self._isAnnotateNumZslices or self._isLineageAnnot
        return (not _isEnabled)
    
    def setColors(
            self, label, bud_will_divide, S_phase_mother, G1_phase,
            lost_object, tracked_lost_object, **kwargs
        ):
        alpha = 200
        if len(G1_phase) == 3:
            G1_phase = (*G1_phase, 220)
        else:
            G1_phase = tuple(G1_phase)
        colors = {
            'label': (*label, alpha),
            'bud_will_divide': (*bud_will_divide, alpha),
            'S_phase_mother': (*S_phase_mother, alpha),
            'G1_phase': G1_phase,
            'new_object': (255,0,0,255),
            'lost_object': (*lost_object, alpha),
            'tracked_lost_object': (*tracked_lost_object, alpha),
            'grayed': (100,100,100,75),
            'highlight': (255,0,0,200),
            'S_phase_bud': (255,0,0,220),
            'green': (0,255,0,220)
        }
        self.item.setColors(colors)
        self._colors = colors
    
    def colors(self):
        return self._colors

    def setLabelAnnot(self, isLabelAnnot):
        self._isLabelAnnot = isLabelAnnot

    def setCcaAnnot(self, isCcaAnnot):
        self._isCcaAnnot = isCcaAnnot
    
    def isCcaAnnot(self):
        return self._isCcaAnnot

    def isLabelAnnot(self):
        return self._isLabelAnnot

    def setAnnotateNumZslices(self, isAnnotateNumZslices):
        self._isAnnotateNumZslices = isAnnotateNumZslices
    
    def isAnnotateNumZslices(self):
        return self._isAnnotateNumZslices
    
    def setLineageAnnot(self, isTreeAnnotations):
        self._isLineageAnnot = isTreeAnnotations
    
    def isLineageAnnot(self):
        return self._isLineageAnnot
    
    def setPxMode(self, mode):
        self.item.setPxMode(mode)

    def setScaling(self, scaling):
        self._scaling = bool(scaling)
        if hasattr(self.item, 'setScaling'):
            self.item.setScaling(scaling)
    
    def update(self):
        self.item.update()
        
    def clear(self):
        if hasattr(self.item, 'clearData'):
            self.item.clearData()
        elif hasattr(self.item, 'clear'):
            self.item.clear()
        self.item.setVisible(False)
        if hasattr(self.item, 'highlighterItem') and self.item.highlighterItem is not None:
            self.item.highlighterItem.clearData()
        
        
class FadingTracksItem(pg.GraphicsObject):
    def __init__(self, tracks=None, color=(255, 100, 0), n_fade=15,
                 max_width=4, min_width=1, max_alpha=255, min_alpha=20):
        super().__init__()
        self.color = tuple(int(c) for c in color[:3])
        self.n_fade = max(n_fade, 1)
        self.max_width, self.min_width = max_width, min_width
        self.max_alpha, self.min_alpha = int(max_alpha), int(min_alpha)
        self.tracks = []
        self._bounding_rect = QRectF()
        self._pen_cache = {}
        self._build_pen_cache()
        self.setTracks([] if tracks is None else tracks)

    def _build_pen_cache(self):
        self._pen_cache = {}
        max_dist = int(math.ceil(self.n_fade))
        for dist_from_head in range(max_dist + 1):
            frac = max(0.0, 1.0 - dist_from_head / self.n_fade)
            alpha = int(self.min_alpha + frac * (self.max_alpha - self.min_alpha))
            width = self.min_width + frac * (self.max_width - self.min_width)
            pen = QPen(QColor(*self.color, alpha))
            pen.setWidthF(width)
            pen.setCapStyle(Qt.RoundCap)
            self._pen_cache[dist_from_head] = pen

        min_pen = QPen(QColor(*self.color, self.min_alpha))
        min_pen.setWidthF(self.min_width)
        min_pen.setCapStyle(Qt.RoundCap)
        self._pen_cache['min'] = min_pen

    def setAppearance(
            self, color=None, n_fade=None, max_width=None, min_width=None,
            max_alpha=None, min_alpha=None
        ):
        changed = False

        if color is not None:
            new_color = tuple(int(c) for c in color[:3])
            if new_color != self.color:
                self.color = new_color
                changed = True

        if n_fade is not None:
            new_n_fade = max(n_fade, 1)
            if new_n_fade != self.n_fade:
                self.n_fade = new_n_fade
                changed = True

        if max_width is not None and max_width != self.max_width:
            self.max_width = max_width
            changed = True

        if min_width is not None and min_width != self.min_width:
            self.min_width = min_width
            changed = True

        if max_alpha is not None:
            new_max_alpha = int(max_alpha)
            if new_max_alpha != self.max_alpha:
                self.max_alpha = new_max_alpha
                changed = True

        if min_alpha is not None:
            new_min_alpha = int(min_alpha)
            if new_min_alpha != self.min_alpha:
                self.min_alpha = new_min_alpha
                changed = True

        if changed:
            self._build_pen_cache()

    def setTracks(self, tracks):
        tracks = [] if tracks is None else tracks
        self.prepareGeometryChange()
        self.tracks = tracks
        self._generate_picture()
        self.setVisible(bool(tracks))
        self.update()

    def clear(self):
        self.setTracks([])

    def _generate_picture(self):
        self.picture = QPicture()
        painter = QPainter(self.picture)
        bounding_left = bounding_top = None
        bounding_right = bounding_bottom = None

        n_buckets = int(math.ceil(self.n_fade)) + 1
        MIN_BUCKET = n_buckets  # extra slot for the 'min' fallback pen
        buckets = [[] for _ in range(n_buckets + 1)]

        for points, frames in self.tracks:
            n = len(points)
            if n < 2:
                continue

            head_frame = frames[-1]

            for i in range(n - 1):
                f0, f1 = frames[i], frames[i + 1]
                if f1 - f0 > 1:
                    continue

                dist_from_head = head_frame - f1
                bucket_idx = dist_from_head if dist_from_head < n_buckets else MIN_BUCKET

                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                bounding_left = x0 if bounding_left is None else min(bounding_left, x0, x1)
                bounding_top = y0 if bounding_top is None else min(bounding_top, y0, y1)
                bounding_right = x0 if bounding_right is None else max(bounding_right, x0, x1)
                bounding_bottom = y0 if bounding_bottom is None else max(bounding_bottom, y0, y1)

                buckets[bucket_idx].append(QLineF(x0, y0, x1, y1))

        # draw tail-to-head (largest dist_from_head first) so the freshest
        # segments paint on top
        for bucket_idx in range(n_buckets, -1, -1):
            lines = buckets[bucket_idx]
            if not lines:
                continue
            pen_key = 'min' if bucket_idx == MIN_BUCKET else bucket_idx
            painter.setPen(self._pen_cache[pen_key])
            painter.drawLines(lines)

        painter.end()
        if bounding_left is None:
            self._bounding_rect = QRectF()
        else:
            self._bounding_rect = QRectF(
                bounding_left, bounding_top,
                bounding_right - bounding_left, bounding_bottom - bounding_top,
            )

        # def pen_sort_key(key):
        #     return float('inf') if key == 'min' else key

        # for pen_key in sorted(lines_by_pen.keys(), key=pen_sort_key, reverse=True):
        #     painter.setPen(self._pen_cache[pen_key])
        #     painter.drawLines(lines_by_pen[pen_key])

        # painter.end()
        # if bounding_left is None:
        #     self._bounding_rect = QRectF()
        # else:
        #     self._bounding_rect = QRectF(
        #         bounding_left,
        #         bounding_top,
        #         bounding_right - bounding_left,
        #         bounding_bottom - bounding_top,
        #     )

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return self._bounding_rect
    
    def settingsDict(self):
        return {
            'color': self.color,
            'n_fade': self.n_fade,
            'max_width': self.max_width,
            'min_width': self.min_width,
            'max_alpha': self.max_alpha,
            'min_alpha': self.min_alpha
        }