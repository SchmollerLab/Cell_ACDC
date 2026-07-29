import os
import traceback

import numpy as np
import pandas as pd

from . import GUI_INSTALLED
from . import cellacdc_path, printl, ignore_exception, debugutils

if GUI_INSTALLED:
    from PIL import Image, ImageFont, ImageDraw
    from qtpy.QtGui import QFont
    import pyqtgraph as pg
    pg.setConfigOption('imageAxisOrder', 'row-major')
    
    from . import plot

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
        self.fontBold = ImageFont.truetype(font_path, fontSize)
        self.fontRegular = ImageFont.truetype(font_bold_path, fontSize)
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

    def beginFrameData(self):
        # For image-based annotations, frame preparation and clear are equivalent.
        self.clearData()
    
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

class TextAnnotationsScatterItem(pg.ScatterPlotItem):
    def __init__(self, *args, anchor=(0.5, 0.5), **kargs):
        super().__init__(*args, **kargs)
        self.initFonts(kargs.get('size', 10))
        self.texts = []
        self.annotData = []
        self._anchor = anchor
        self._poolCapacity = 0
        self._poolActiveCount = 0
        self._poolSignatures = []
        self._poolData = []
        self._hiddenBrush = pg.mkBrush((0, 0, 0, 0))
        self._hiddenPen = pg.mkPen((0, 0, 0, 0), width=0)

    def _initialPoolCapacity(self, numObjects):
        numObjects = int(numObjects)
        if numObjects <= 0:
            return 1
        extra = max(1, int(np.ceil(numObjects*0.1)))
        return numObjects + extra

    def _hiddenPointOpts(self):
        symbol = self.getObjTextAnnotSymbol('?', bold=False)
        size = self.sizesRegular.get('?', self.fontSize)
        return {
            'pos': (0, 0),
            'symbol': symbol,
            'size': size,
            'brush': self._hiddenBrush,
            'pen': self._hiddenPen,
            'data': None,
        }

    def _ensurePoolCapacity(self, required):
        required = int(required)
        if required <= self._poolCapacity:
            return

        addCount = required - self._poolCapacity
        newPoints = [self._hiddenPointOpts() for _ in range(addCount)]
        super().addPoints(newPoints)

        points = self.points()
        for idx in range(self._poolCapacity, required):
            points[idx].setVisible(False)

        self._poolSignatures.extend([None]*addCount)
        self._poolData.extend([None]*addCount)
        self._poolCapacity = required

    def _initPool(self, numObjects):
        targetCapacity = self._initialPoolCapacity(numObjects)
        if self._poolCapacity > 0:
            self._ensurePoolCapacity(targetCapacity)
            return

        super().setData([], [])
        self._poolCapacity = 0
        self._poolActiveCount = 0
        self._poolSignatures = []
        self._poolData = []
        self._ensurePoolCapacity(targetCapacity)

    def _rebuildSizes(self, bold=False):
        if bold:
            self.sizesBold = plot.get_symbol_sizes(
                self.scalesBold, self.symbolsBold, self.fontSize
            )
            self._maxScaleBold = max(self.scalesBold.values(), default=None)
        else:
            self.sizesRegular = plot.get_symbol_sizes(
                self.scalesRegular, self.symbolsRegular, self.fontSize
            )
            self._maxScaleRegular = max(self.scalesRegular.values(), default=None)

    def _updateSizesForTexts(self, texts, bold=False):
        if not texts:
            return

        if bold:
            scales = self.scalesBold
            sizes_attr = 'sizesBold'
            max_scale_attr = '_maxScaleBold'
        else:
            scales = self.scalesRegular
            sizes_attr = 'sizesRegular'
            max_scale_attr = '_maxScaleRegular'

        current_max_scale = getattr(self, max_scale_attr, None)
        if current_max_scale is None:
            self._rebuildSizes(bold=bold)
            return

        added_max_scale = max(scales[text] for text in texts)
        if added_max_scale > current_max_scale:
            self._rebuildSizes(bold=bold)
            return

        sizes = getattr(self, sizes_attr)
        for text in texts:
            sizes[text] = int(np.round(self.fontSize*current_max_scale/scales[text]))
    
    def clearData(self):
        active = self._poolActiveCount
        if active > 0:
            data = self.data
            data['visible'][:active] = False
            data['sourceRect'][:active] = 0
            self.updateSpots(data[:active])

            for idx in range(active):
                self._poolSignatures[idx] = None
                self._poolData[idx] = None

        self._poolActiveCount = 0
        self.setVisible(False)
        self.annotData = []
        self.texts = []

    def beginFrameData(self):
        # Prepare next frame payload without triggering render updates.
        self.annotData = []
        self.texts = []
    
    def appendData(self, data, text):
        self.annotData.append(data)
        self.texts.append(text)
    
    def draw(self):
        required = len(self.annotData)
        self._ensurePoolCapacity(required)
        data = self.data
        prevActive = self._poolActiveCount
        touched = max(required, prevActive)
        hasUpdates = False

        for idx, pointOpts in enumerate(self.annotData):
            signature = (
                pointOpts['pos'], pointOpts['symbol'], pointOpts['size'],
                pointOpts['brush'], pointOpts['pen'], pointOpts.get('data')
            )
            if signature != self._poolSignatures[idx]:
                x, y = pointOpts['pos']
                data['x'][idx] = x
                data['y'][idx] = y
                data['symbol'][idx] = pointOpts['symbol']
                data['brush'][idx] = pointOpts['brush']
                data['pen'][idx] = pointOpts['pen']
                data['size'][idx] = pointOpts['size']
                data['data'][idx] = pointOpts.get('data')
                data['sourceRect'][idx] = 0
                hasUpdates = True
                self._poolSignatures[idx] = signature
                self._poolData[idx] = pointOpts.get('data')
            if not data['visible'][idx]:
                data['visible'][idx] = True
                data['sourceRect'][idx] = 0
                hasUpdates = True

        for idx in range(required, prevActive):
            if data['visible'][idx]:
                data['visible'][idx] = False
                data['sourceRect'][idx] = 0
                hasUpdates = True
            self._poolSignatures[idx] = None
            self._poolData[idx] = None

        if hasUpdates and touched > 0:
            self.updateSpots(data[:touched])

        self._poolActiveCount = required
        self.setVisible(required > 0)
        
    def clear(self):
        self.setVisible(False)

    def initFonts(self, fontSize):
        self.fontSize = fontSize
        self.fontBold = QFont(FONT_FAMILY.lower())
        self.fontBold.setBold(True)
        self.fontBold.setPixelSize(fontSize)

        self.fontRegular = QFont(FONT_FAMILY.lower())
        self.fontRegular.setPixelSize(fontSize)
    
    def init(self, *args):
        pass

    def initSymbols(self, allIDs, onlyIDs=False):
        allIDs = list(allIDs)
        annotTexts = ['?']
        for ID in allIDs:
            annotTexts.append(str(ID))
            if not onlyIDs:
                annotTexts.append(f'{ID}?')
            
        if not onlyIDs:
            for gen_num in range(20):
                annotTexts.append(f'G1-{gen_num}')
                annotTexts.append(f'G1-{gen_num}?')
                annotTexts.append(f'S-{gen_num}')
                annotTexts.append(f'S-{gen_num}?')
        
        if hasattr(self, 'symbolsBold'):
            # Symbols already created in prev. session --> add missing ones
            self.addSymbols(annotTexts)
        else:
            # Symbols never created --> create now
            self.createSymbols(annotTexts)

        # Preallocate spots once with 10% headroom to avoid frequent reallocations.
        self._initPool(len(allIDs))
    
    def addSymbols(self, annotTexts, includeBold=True):
        if includeBold:
            missing_bold = [
                text for text in annotTexts if text not in self.symbolsBold
            ]
            if missing_bold:
                symbolsBold, scalesBold = plot.texts_to_pg_scatter_symbols(
                    missing_bold, font=self.fontBold, return_scales=True
                )
                self.symbolsBold.update(symbolsBold)
                self.scalesBold.update(scalesBold)
                self._updateSizesForTexts(missing_bold, bold=True)

        missing_regular = [
            text for text in annotTexts if text not in self.symbolsRegular
        ]
        if missing_regular:
            symbolsRegular, scalesRegular = plot.texts_to_pg_scatter_symbols(
                missing_regular, font=self.fontRegular, return_scales=True
            )
            self.symbolsRegular.update(symbolsRegular)
            self.scalesRegular.update(scalesRegular)
            self._updateSizesForTexts(missing_regular, bold=False)

    def createSymbols(self, annotTexts, includeBold=True):
        if includeBold:
            self.symbolsBold, self.scalesBold = plot.texts_to_pg_scatter_symbols(
                annotTexts, font=self.fontBold, return_scales=True
            )

        self.symbolsRegular, scalesRegular = plot.texts_to_pg_scatter_symbols(
            annotTexts, font=self.fontRegular, return_scales=True
        )
        self.scalesRegular = scalesRegular
        self.initSizes(includeBold=includeBold)
    
    def initSizes(self, includeBold=True):
        if not hasattr(self, 'scalesBold'):
            includeBold = False
            
        if includeBold:
            self._rebuildSizes(bold=True)
        self._rebuildSizes(bold=False)
    
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

    def getObjTextAnnotSymbol(self, text, bold=False, initSizes=True):
        if bold:
            symbols = self.symbolsBold
            font = self.fontBold
            scales = self.scalesBold
        else:
            symbols = self.symbolsRegular
            font = self.fontRegular
            scales = self.scalesRegular
        
        symbol = symbols.get(text)
        if symbol is not None:
            return symbol

        symbol, scale = plot.text_to_pg_scatter_symbol(
            text, font=font, return_scale=True
        )
        symbols[text] = symbol
        scales[text] = scale
        if initSizes:
            self._updateSizesForTexts([text], bold=bold)
        return symbol

    def grayOutAnnotations(self, IDsToSkip=None):
        data = self.data
        hasUpdates = False
        for idx in range(self._poolActiveCount):
            ID = self._poolData[idx]
            doNotGray = IDsToSkip is not None and IDsToSkip.get(ID, False)
            if doNotGray:
                pointOpts = self.annotData[idx]
                brush = pointOpts['brush']
                pen = pointOpts['pen']
            else:
                brush = self._brushes['grayed']
                pen = self._pens['grayed']

            if data['brush'][idx] is not brush or data['pen'][idx] is not pen:
                data['brush'][idx] = brush
                data['pen'][idx] = pen
                data['sourceRect'][idx] = 0
                hasUpdates = True

        if hasUpdates and self._poolActiveCount > 0:
            self.updateSpots(data[:self._poolActiveCount])

    def highlightObject(self, obj, rp=None, getObjCentroidFunc=None):
        ID = obj.label
        objIdx = None
        for idx in range(self._poolActiveCount):
            if ID == self._poolData[idx]:
                objIdx = idx
                break
        if objIdx is None:
            # Keep pool consistency: only highlight points currently present.
            return
        
        pointItem = self.points()[objIdx]
        symbol = self.getObjTextAnnotSymbol(str(ID), bold=True)
        pointItem.setSymbol(symbol)

        pointItem.setBrush(self._brushes['new_object'])
        pointItem.setPen(self._pens['new_object'])

    def removeHighlightObject(self, obj):
        ID = obj.label
        objIdx = None
        for idx in range(self._poolActiveCount):
            if ID == self._poolData[idx]:
                objIdx = idx
                break
        if objIdx is None:
            return

        pointItem = self.points()[objIdx]

        default_symbol = self.getObjTextAnnotSymbol(str(ID), bold=False)
        pointItem.setSymbol(default_symbol)

        pointItem.setBrush(self._brushes['label'])
        pointItem.setPen(self._pens['label'])
    
    def modifyPosAnchor(self, pointOpts, anchor, symbol):
        if anchor is None:
            return pointOpts
        
        xa, ya = anchor
        if (xa, ya) == (0.5, 0.5):
            return pointOpts
        
        br = symbol.boundingRect()
        xf = br.width()*(anchor[0]-0.5)
        yf = br.height()*(anchor[1]-0.5)
        x, y = pointOpts['pos']
        pointOpts['pos'] = (x-xf, y-yf)
        
        return pointOpts      
    
    def addObjAnnot(self, pos, draw=False, anchor=None, **objOpts):        
        text = objOpts['text']
        bold = objOpts['bold']
        symbol = self.getObjTextAnnotSymbol(text, bold)

        if bold:
            size = self.sizesBold[text]
        else:
            size = self.sizesRegular[text]

        color_name = objOpts['color_name']

        pointOpts = {}
        pointOpts['brush'] = self._brushes[color_name]
        pointOpts['pen'] = self._pens[color_name]
        pointOpts['symbol'] = symbol
        pointOpts['size'] = size
        pointOpts['pos'] = tuple(pos)
        pointOpts = self.modifyPosAnchor(pointOpts, anchor, symbol)

        if draw:
            self.addPoints([pointOpts])
        
        return pointOpts        

class TextAnnotations:
    def __init__(self):
        self._isLabelAnnot = False
        self._isCcaAnnot = False
        self._isAnnotateNumZslices = False
        self._isLabelTreeAnnotation = False
        self._isGenNumTreeAnnotation = False
        self._isGenNumTreeAnnotation = False
        self._symbolsPreinitPosDataId = None
        self._symbolsPreinitFrameCount = None

    def _iter_all_frames_symbol_texts(self, posData):
        texts = set()

        allIDs = getattr(posData, 'allIDs', None)
        if allIDs:
            for ID in allIDs:
                texts.add(str(ID))
                texts.add(f'{ID}?')

        for frameData in posData.allData_li:
            acdc_df = frameData.get('acdc_df')
            if acdc_df is None:
                continue

            for ID in acdc_df.index:
                texts.add(str(ID))
                texts.add(f'{ID}?')

            if 'Cell_ID_tree' in acdc_df.columns:
                for val in acdc_df['Cell_ID_tree'].dropna().unique():
                    texts.add(str(val))

            requiredCols = {
                'cell_cycle_stage', 'generation_num', 'is_history_known'
            }
            if not requiredCols.issubset(set(acdc_df.columns)):
                continue

            for _, row in acdc_df.iterrows():
                ccs = row.get('cell_cycle_stage')
                if pd.isna(ccs):
                    continue

                gen_num = row.get('generation_num')
                if pd.isna(gen_num):
                    continue

                try:
                    gen_num_int = int(gen_num)
                    gen_txt = 'ND' if gen_num_int == -1 else str(gen_num_int)
                except Exception:
                    gen_txt = str(gen_num)

                texts.add(f'{ccs}-{gen_txt}')

                gen_tree = row.get('generation_num_tree', None)
                if gen_tree is not None and not pd.isna(gen_tree):
                    texts.add(f'{ccs}-{gen_tree}')

                is_history_known = row.get('is_history_known', True)
                if not bool(is_history_known):
                    texts.add(f'{ccs}-{gen_txt}?')
                    if gen_tree is not None and not pd.isna(gen_tree):
                        texts.add(f'{ccs}-{gen_tree}?')

        return texts

    def _preinitSymbolsAllFrames(self, posData):
        if not isinstance(self.item, TextAnnotationsScatterItem):
            return

        posDataId = id(posData)
        frameCount = len(posData.allData_li)
        if (
            self._symbolsPreinitPosDataId == posDataId
            and self._symbolsPreinitFrameCount == frameCount
        ):
            return

        try:
            texts = self._iter_all_frames_symbol_texts(posData)
            if texts:
                self.item.addSymbols(list(texts), includeBold=True)
        except Exception:
            # Symbol warm-up is best-effort and must never interrupt rendering.
            pass

        self._symbolsPreinitPosDataId = posDataId
        self._symbolsPreinitFrameCount = frameCount
    
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
            return
        
        labelsToSkip = kwargs.get('labelsToSkip')
        posData = kwargs['posData']
        self._preinitSymbolsAllFrames(posData)
        self.item.beginFrameData()

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
            self.item.appendData(objData, objOpts['text'])

        if posData.trackedLostIDs and annotateLost:
            prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
            if prev_rp is None:
                self.item.draw()
                return
            
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
                self.item.appendData(objData, objOpts['text'])


        if posData.lost_IDs and annotateLost:
            prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
            if prev_rp is None:
                self.item.draw()
                return
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
                self.item.appendData(objData, objOpts['text'])

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
