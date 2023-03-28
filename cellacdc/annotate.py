import os
import traceback

import numpy as np
import pandas as pd

from PIL import Image, ImageFont, ImageDraw

from PyQt5.QtGui import QFont

import pyqtgraph as pg

from . import plot, cellacdc_path, printl, ignore_exception

INVERTIBLE_COLOR_NAMES = [
    'label', 'S_phase_mother', 'G1_phase'
]
FONT_FAMILY = 'Helvetica'
font_path = os.path.join(cellacdc_path, 'resources', 'fonts', f'{FONT_FAMILY}-Regular.ttf')
font_bold_path = os.path.join(cellacdc_path, 'resources', 'fonts', f'{FONT_FAMILY}-Bold.ttf')

def get_obj_text_label_annot(
        obj, cca_df: pd.DataFrame, is_tree_annot: bool, add_num_zslices: bool
    ) -> str:
    if is_tree_annot and cca_df is not None:
        try:
            annot_label = cca_df.at[obj.label, 'Cell_ID_tree']
        except Exception as e:
            annot_label = obj.label
    else:
        annot_label = obj.label
    
    if not add_num_zslices:
        return str(annot_label)
    
    num_z_slices = np.sum(np.any(obj.image, axis=(1,2)))
    return f'{annot_label} ({num_z_slices})'

def get_obj_text_cca_annot(
        obj, cca_df: pd.DataFrame, is_tree_annot: bool    
    ) -> str:
    ID = obj.label
    try:
        cca_df_obj = cca_df.loc[ID]
    except Exception as e:
        return f'{obj.label}', None
    
    ccs = cca_df_obj['cell_cycle_stage']

    generation_num = int(cca_df_obj['generation_num'])
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
        obj, cca_df: pd.DataFrame, is_cca_annot: bool, is_new_obj: bool, 
        add_num_zslices: bool, is_label_tree_annot: bool, 
        is_gen_num_tree_annot: bool, frame_i: int
    ) -> dict: 
    if cca_df is None or not is_cca_annot:
        bold = False
        if is_new_obj:
            color_name = 'new_object'
        else:
            color_name = 'label'
        text = get_obj_text_label_annot(
            obj, cca_df, is_label_tree_annot, add_num_zslices
        )
    else:
        text, cca_df_obj = get_obj_text_cca_annot(
            obj, cca_df, is_gen_num_tree_annot
        )
        if cca_df_obj is None:
            opts = {'text': text, 'color_name': 'label', 'bold': False}
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
        self.fontBold = ImageFont.truetype(font_path, fontSize)
        self.fontRegular = ImageFont.truetype(font_bold_path, fontSize)
    
    def init(self, image_shape):
        shape = (*image_shape, 4)
        self.pilImage = Image.fromarray(np.zeros(shape, dtype=np.uint8))
        self.pilDraw = ImageDraw.Draw(self.pilImage)
    
    def clearImage(self):
        self.pilDraw.rectangle([(0,0), self.pilDraw.im.size], fill=(0,0,0,0))
    
    def clearData(self):
        self.clearImage()
        self.texts = []
    
    def update(self):
        pass
    
    def appendData(self, data, text):
        self.annotData.append(data)
        self.texts.append(text)
    
    def addObjAnnot(self, pos, draw=True, **objOpts):
        if objOpts['bold']:
            font = self.fontBold
        else:
            font = self.fontRegular
        
        text = objOpts['text']
        color = self._colors[objOpts['color_name']]
        self.pilDraw.text(pos, text, color, font=font, anchor='mm')
    
    def draw(self):
        super().setImage(np.array(self.pilDraw))

    def setColors(self, colors):
        self._colors = colors
    
    def initSymbols(self, allIDs):
        pass

    def colors(self):
        return self._colors

class TextAnnotationsScatterItem(pg.ScatterPlotItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
    def clearData(self):
        self.setData([], [])
        self.annotData = []
        self.texts = []
    
    def appendData(self, data, text):
        self.annotData.append(data)
        self.texts.append(text)
    
    def draw(self):
        super().setData(self.annotData)

    def initFonts(self, fontSize):
        self.fontSize = fontSize
        self.fontBold = QFont(FONT_FAMILY.lower())
        self.fontBold.setBold(True)
        self.fontBold.setPixelSize(fontSize)

        self.fontRegular = QFont(FONT_FAMILY.lower())
        self.fontRegular.setPixelSize(fontSize)
    
    def init(self, *args):
        pass

    def initSymbols(self, allIDs):
        annotTexts = ['?']
        for ID in allIDs:
            annotTexts.append(str(ID))
            annotTexts.append(f'{ID}?')
        for gen_num in range(20):
            annotTexts.append(f'G1-{gen_num}')
            annotTexts.append(f'G1-{gen_num}?')
            annotTexts.append(f'S-{gen_num}')
            annotTexts.append(f'S-{gen_num}?')
        
        self.symbolsBold, self.scalesBold = plot.texts_to_pg_scatter_symbols(
            annotTexts, font=self.fontBold, return_scales=True
        )

        self.symbolsRegular, scalesRegular = plot.texts_to_pg_scatter_symbols(
            annotTexts, font=self.fontRegular, return_scales=True
        )
        self.scalesRegular = scalesRegular
        self.initSizes()
    
    def initSizes(self):
        self.sizesBold = plot.get_symbol_sizes(
            self.scalesBold, self.symbolsBold, self.fontSize
        )
        self.sizesRegular = plot.get_symbol_sizes(
            self.scalesRegular, self.symbolsRegular, self.fontSize
        )
    
    def setColors(self, colors):
        self._colors = colors
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

    def getObjTextAnnotSymbol(self, text, bold=False):
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
        self.initSizes()
        return symbol

    def addObjAnnot(self, pos, draw=False, **objOpts):        
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

        if draw:
            self.addPoints([pointOpts])
        
        return pointOpts
    
    @ignore_exception
    def highlightObject(self, obj_idx):
        for idx in range(len(self.data)):
            if idx == obj_idx:
                brush = self._brushes['highlight']
                pen = self._pens['highlight']
                text = self.texts[idx]
                symbol = self.getObjTextAnnotSymbol(text, True)
            else:
                brush = self._brushes['grayed']
                pen = self._pens['grayed']
                symbol = self.data[idx]['symbol']
            
            self.data[idx]['brush'] = brush
            self.data[idx]['pen'] = pen
            self.data[idx]['symbol'] = symbol
        self.updateSpots()
        

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
    
    def invertBlackAndWhite(self):
        lost_object_rgb = self.item.colors()['lost_object'][:3]
        invertedColors = {}
        for color_name in INVERTIBLE_COLOR_NAMES:
            color = self.item.colors()[color_name]
            invertedColors[color_name] = tuple([255-val for val in color[:3]])

        invertedColors['lost_object'] = lost_object_rgb
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
  
    def changeResolution(self, mode, allIDs, ax):
        ax.removeItem(self.item)
        highRes = True if mode == 'high' else False        
        self.createItems(highRes, allIDs, pxMode=self._pxMode)
    
    def addObjAnnotation(self, obj, color_name, text, bold):
        objOpts = {
            'text': text,
            'bold': bold,
            'color_name': color_name,
        }
        yc, xc = obj.centroid[-2:]
        pos = (int(xc), int(yc))
        objData = self.item.addObjAnnot(pos, draw=True, **objOpts)
        self.item.appendData(objData, objOpts['text'])
    
    def setAnnotations(self, **kwargs):
        if self.isDisabled():
            return
        
        self.item.clearData()
        
        labelsToSkip = kwargs.get('labelsToSkip')
        posData = kwargs['posData']
        isCcaAnnot = self.isCcaAnnot()
        isAnnotateNumZslices = self.isAnnotateNumZslices()
        isLabelTreeAnnotation = self.isLabelTreeAnnotation()
        isGenNumTreeAnnotation = self.isGenNumTreeAnnotation()
        for obj in posData.rp:
            if labelsToSkip is not None:
                if labelsToSkip.get(obj.label, False):
                    continue
            
            isNewObject = obj.label in posData.new_IDs
            objOpts = get_obj_text_annot_opts(
                obj, posData.cca_df, isCcaAnnot, isNewObject,
                isAnnotateNumZslices, isLabelTreeAnnotation, 
                isGenNumTreeAnnotation, posData.frame_i
            )
            yc, xc = obj.centroid[-2:]
            pos = (int(xc), int(yc))
            objData = self.item.addObjAnnot(pos, draw=False, **objOpts)
            self.item.appendData(objData, objOpts['text'])

        if posData.lost_IDs:
            prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
            for obj in prev_rp:
                if obj.label not in posData.lost_IDs:
                    continue
                objOpts = {
                    'text': f'{obj.label}?',
                    'color_name': 'lost_object',
                    'bold': False,
                }
                yc, xc = obj.centroid[-2:]
                pos = (int(xc), int(yc))
                objData = self.item.addObjAnnot(pos, draw=False, **objOpts)
                self.item.appendData(objData, objOpts['text'])

        self.item.draw()
    
    def isDisabled(self):
        _isEnabled = self._isLabelAnnot or self._isCcaAnnot
        return (not _isEnabled)
    
    def setColors(
            self, label, bud_will_divide, S_phase_mother, G1_phase,
            lost_object, **kwargs
        ):
        alpha = 200
        colors = {
            'label': (*label, alpha),
            'bud_will_divide': (*bud_will_divide, alpha),
            'S_phase_mother': (*S_phase_mother, alpha),
            'G1_phase': tuple(G1_phase),
            'new_object': (255,0,0,255),
            'lost_object': (*lost_object, alpha),
            'grayed': (100,100,100,75),
            'highlight': (255,0,0,200),
            'S_phase_bud': (255,0,0,220),
            'green': (0,255,0,220)
        }
        self.item.setColors(colors)

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
    
    def highlightObject(self, obj_idx):
        self.item.highlightObject(obj_idx)
    
    def setPxMode(self, mode):
        self.item.setPxMode(mode)
    
    def update(self):
        self.item.update()
