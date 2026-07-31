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
    
    from qtpy.QtGui import QFont, QPicture, QPainter, QColor, QPen
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

def get_obj_text_label_annot(
        obj, acdc_df: pd.DataFrame, is_tree_annot: bool, add_num_zslices: bool
    ) -> str:
    if is_tree_annot and acdc_df is not None:
        try:
            annot_label = acdc_df.at[obj.label, 'Cell_ID_tree']
        except Exception as err:
            # print(traceback.format_exc())
            annot_label = obj.label
    else:
        annot_label = obj.label
    
    if not add_num_zslices:
        return str(annot_label)
    
    num_z_slices = np.sum(np.any(obj.image, axis=(1,2)))
    return f'{annot_label} ({num_z_slices})'

def get_obj_text_cca_annot(
        obj, acdc_df: pd.DataFrame, is_tree_annot: bool,     
        moth_bud_pairs_cca=None
    ) -> str:
    ID = obj.label
    try:
        cca_df_obj = moth_bud_pairs_cca.loc[ID].copy()
    except Exception as e:
        try:
            cca_df_obj = acdc_df.loc[ID]
        except Exception as e:
            return str(ID), None
    
    try:
        ccs = cca_df_obj['cell_cycle_stage']
    except Exception as err:
        return str(ID), None 

    try:
        generation_num = int(cca_df_obj['generation_num'])
    except Exception as e:
        return str(ID), None
    
    generation_num = 'ND' if generation_num==-1 else generation_num
    if is_tree_annot:
        try:
            generation_num = cca_df_obj['generation_num_tree']
        except Exception as e:
            generation_num = generation_num

    txt = f'{ccs}-{generation_num}'

    is_history_known = cca_df_obj['is_history_known']
    if not is_history_known:
        txt = f'{txt}?'

    return txt, cca_df_obj

def get_obj_text_annot_opts(
        obj, acdc_df: pd.DataFrame, is_cca_annot: bool, is_new_obj: bool, 
        add_num_zslices: bool, is_label_tree_annot: bool, 
        is_gen_num_tree_annot: bool, frame_i: int,
        moth_bud_pairs_cca=None
    ) -> dict: 
    if acdc_df is None or not is_cca_annot:
        bold = False
        if is_new_obj:
            color_name = 'new_object'
        else:
            color_name = 'label'
        text = get_obj_text_label_annot(
            obj, acdc_df, is_label_tree_annot, add_num_zslices
        )
    else:
        text, cca_df_obj = get_obj_text_cca_annot(
            obj, acdc_df, is_gen_num_tree_annot, 
            moth_bud_pairs_cca=moth_bud_pairs_cca
        )
        if cca_df_obj is None:
            if is_new_obj:
                color_name = 'new_object'
            else:
                color_name = 'label'
            opts = {'text': text, 'color_name': color_name, 'bold': False}
            return opts
        
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
        
    opts = {'text': text, 'color_name': color_name, 'bold': bold}

    return opts

class TextAnnotationsImageItem(pg.ImageItem):
    def __init__(self, **kargs):
        super().__init__(**kargs)
    
    def initFonts(self, fontSize):
        self.fontSize = fontSize
        self.fontRegular = ImageFont.truetype(font_path, fontSize)
        self.fontBold = ImageFont.truetype(font_bold_path, fontSize)
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
        self.pilDraw.rectangle([(0,0), self.pilDraw.im.size], fill=(0,0,0,0))
    
    def clearData(self):
        self.clearImage()
        self.setOpacity(1.0)
        self.highlighterItem.setData([], [])
        self.texts = []
        self.annotData = []
    
    def update(self):
        pass
    
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
    def __init__(self, size=10, pxMode=False, anchor=(0.5, 0.5)):
        super().__init__()
        self._pxMode = pxMode
        self._anchor = anchor
        self.picture = QPicture()
        self._boundingRect = QRectF()
        self.annotData = []  # list of dicts: pos, text, bold, color_name, data(ID)
        self.texts = []      # kept parallel to annotData for API compatibility
        self.initFonts(size)
        self.setPxMode(pxMode)

    # ---- setup / config, API-compatible no-ops where the atlas is gone ----

    def initFonts(self, fontSize):
        self.fontSize = fontSize
        self.fontBold = QFont(FONT_FAMILY)
        self.fontBold.setBold(True)
        self.fontBold.setPixelSize(fontSize)

        self.fontRegular = QFont(FONT_FAMILY)
        self.fontRegular.setPixelSize(fontSize)

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

    def setColors(self, colors):
        self._colors = colors.copy()
        self._brushes = {}
        self._pens = {}
        for name, color in self._colors.items():
            self._brushes[name] = pg.mkBrush(color)
            self._pens[name] = pg.mkPen(color[:3], width=1)

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
        self._generatePicture()

    # ---- rendering ----

    def _generatePicture(self):
        self.picture = QPicture()
        painter = QPainter(self.picture)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        xs, ys = [], []
        for idx, objData in enumerate(self.annotData):
            text = objData.get('text')
            if text is None and idx < len(self.texts):
                text = self.texts[idx]

            font = self.fontBold if objData.get('bold') else self.fontRegular
            painter.setFont(font)
            color = self._colors.get(objData.get('color_name'), (255, 255, 255, 255))
            painter.setPen(QColor(*color))

            x, y = objData['pos']
            fm = painter.fontMetrics()
            rect = fm.boundingRect(text)
            # Center the text on (x, y), matching the 'mm' anchor used by
            # the low-res PIL-based item.
            draw_x = x - rect.width() / 2
            draw_y = y + rect.height() / 2 - fm.descent()
            painter.drawText(QPointF(draw_x, draw_y), text)

            xs.extend([x - rect.width() / 2, x + rect.width() / 2])
            ys.extend([y - rect.height() / 2, y + rect.height() / 2])

        painter.end()

        self._boundingRect = (
            QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            if xs else QRectF()
        )
        self.prepareGeometryChange()
        self.update()

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self.picture)

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
        self._isLabelTreeAnnotation = False
        self._isGenNumTreeAnnotation = False
        self._isGenNumTreeAnnotation = False
    
    def initFonts(self, fontSize):
        self.fontSize = fontSize
    
    def initItem(self, *args):
        self.item.init(*args)
    
    def clear(self):
        self.item.clear()
        if hasattr(self.item, 'highlighterItem'):
            self.item.highlighterItem.setData([], [])
    
    def invertBlackAndWhite(self):
        invertedColors = {
            name:color[:3] for name, color in self.item.colors().items()
        }
        for color_name in INVERTIBLE_COLOR_NAMES:
            color = self.item.colors()[color_name]
            invertedColors[color_name] = tuple([255-val for val in color[:3]])

        self.setColors(**invertedColors)

    def createItems(self, isHighResolution, allIDs, pxMode=False):
        self._pxMode = pxMode
        if isHighResolution:
            self._createHighResolutionItems(allIDs, pxMode=pxMode)
        else:
            self._createLowResolutionItem()        
        
    def _createLowResolutionItem(self):
        self.item = TextAnnotationsImageItem()
        self.setFontSize(self.fontSize, [])
    
    def _createHighResolutionItems(self, allIDs, pxMode=False):
        self.item = TextAnnotationsScatterItem(
            size=self.fontSize, pxMode=pxMode
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
  
    def changeResolution(self, mode, allIDs, ax, img_shape):
        self.removeFromPlotItem(ax)
        highRes = True if mode == 'high' else False        
        self.createItems(highRes, allIDs, pxMode=self._pxMode)
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
            self.item.highlighterItem.setData([], [])

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
        isLabelTreeAnnotation = self.isLabelTreeAnnotation()
        isGenNumTreeAnnotation = self.isGenNumTreeAnnotation()
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
                isAnnotateNumZslices, isLabelTreeAnnotation, 
                isGenNumTreeAnnotation, posData.frame_i,
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
        _isEnabled = self._isLabelAnnot or self._isCcaAnnot
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
    
    def setLabelTreeAnnotationsEnabled(self, isTreeAnnotations):
        self._isLabelTreeAnnotation = isTreeAnnotations
    
    def setGenNumTreeAnnotationsEnabled(self, isTreeAnnotations):
        self._isGenNumTreeAnnotation = isTreeAnnotations
    
    def isLabelTreeAnnotation(self):
        return self._isLabelTreeAnnotation

    def isGenNumTreeAnnotation(self):
        return self._isGenNumTreeAnnotation
    
    def setPxMode(self, mode):
        self.item.setPxMode(mode)
    
    def update(self):
        self.item.update()
        
    def clear(self):
        self.item.setVisible(False)
        
        
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