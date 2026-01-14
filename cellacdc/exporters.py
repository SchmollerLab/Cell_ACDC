import os
import tempfile
import shutil

import numpy as np
import cv2

import skimage.io
import skimage.color

from qtpy.QtSvg import QSvgRenderer
from qtpy.QtGui import QImage, QPainter

import pyqtgraph.exporters
import pyqtgraph as pg

from . import transformation, printl, myutils
from . import is_mac, is_win
from . import acdc_ffmpeg_path

class ImageExporter(pyqtgraph.exporters.ImageExporter):
    def __init__(self, item, background=(0, 0, 0, 0), dpi=100, save_pngs=True):
        super().__init__(item)
        self._save_pngs = save_pngs
        
        self._dpi = dpi
        
        # DPI using A4 width
        desired_width = 8.268 * dpi
        self.params['width'] = desired_width
    
        self.parameters()['background'] = (0, 0, 0, 0)
    
    def super_export(self, filepath):
        super().export(filepath)
    
    def svg_to_image(self, svg_filepath, image_filepath):
        width = self.params['width']
        height = self.params['height']
        
        renderer = QSvgRenderer(svg_filepath)
        img = QImage(width, height, QImage.Format_ARGB32)
        img.fill(0)

        p = QPainter(img)
        renderer.render(p)
        p.end()

        img.save(image_filepath)
    
    def crop_from_mask(self, img_rgba):
        if not hasattr(self.item, 'exportMaskImageItem'):
            return img_rgba
        
        crop_mask_rgba = self.item.exportMaskImageItem.image
        alpha = crop_mask_rgba[..., 3]
        rows, cols = np.where(alpha == 0)
        top, bottom = rows.min(), rows.max() + 1
        left, right = cols.min(), cols.max() + 1
        
        x0, y0 = self.item.exportMaskImageItem.pos()
        
        view_range = self.item.viewRange()
        (x_min, x_max), (y_min, y_max) = view_range
        H, W = img_rgba.shape[:2]
        
        # x mapping
        left_px_f = (left - x_min) / (x_max - x_min) * W
        right_px_f = (right - x_min) / (x_max - x_min) * W
        
        # y mapping (PNG origin top-left)
        top_px_f = (y_max - top) / (y_max - y_min) * H
        bottom_px_f = (y_max - bottom) / (y_max - y_min) * H
        
        left_px   = int(np.floor(left_px_f))
        right_px  = int(np.ceil(right_px_f))

        bottom_px = int(np.floor(bottom_px_f))
        top_px    = int(np.ceil(top_px_f))
        
        if left_px < 0:
            left_px = 0
        
        if right_px > W:
            right_px = W
        
        if bottom_px < 0:
            bottom_px = 0
        
        if top_px > H:
            top_px = H
        
        return img_rgba[bottom_px:top_px, left_px:right_px]

    def export(self, filepath):     
        no_ext_filepath, ext = os.path.splitext(filepath)
        svg_filepath = f'{no_ext_filepath}.svg'    
        svg_exporter = SVGExporter(self.item)                  
        svg_exporter.export(svg_filepath)
        self.svg_to_image(svg_filepath, filepath)
        
        try: 
            os.remove(svg_filepath)
        except Exception as err:
            pass
        
        # Remove padding
        img_rgba = skimage.io.imread(filepath)    
        img_rgba = self.crop_from_mask(img_rgba)
        
        img_rgba = transformation.crop_outer_padding(
            img_rgba, value=(0, 0, 0, 255)
        )
        img_rgba = transformation.crop_outer_padding(
            img_rgba, value=(255, 255, 255, 255)
        )

        if self._save_pngs:
            skimage.io.imsave(filepath, img_rgba, check_contrast=False)

        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        
        return img_bgr

class SVGExporter(pyqtgraph.exporters.SVGExporter):
    def __init__(self, item):
        super().__init__(item)
        self.parameters()['background'] = (0, 0, 0, 0)

class VideoExporter:
    def __init__(self, avi_filepath, fps):
        self.writer = None
        self._avi_filepath = avi_filepath
        self._fps = fps
        self._fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    def add_frame(self, img_bgr):
        if self.writer is None:
            height, width = img_bgr.shape[:-1]
            self.writer = cv2.VideoWriter(
                self._avi_filepath, self._fourcc, self._fps, (width, height)
            )
        self.writer.write(img_bgr)
    
    def release(self):
        self.writer.release()
    
    def avi_to_mp4(self):
        avi_to_mp4(self._avi_filepath)

def avi_to_mp4(in_filepath_avi, out_filepath_mp4=None):
    ffmep_exec_path = myutils.download_ffmpeg()
    
    if out_filepath_mp4 is None:
        out_filepath_mp4 = in_filepath_avi.replace('.avi', '.mp4')
    
    ffmep_exec_path = ffmep_exec_path.replace('\\', '/')
    out_filepath_mp4 = out_filepath_mp4.replace('\\', '/')
    in_filepath_avi = in_filepath_avi.replace('\\', '/')
    
    args = [
        '-i', f'{in_filepath_avi}', '-c:v', 'libx264', 
        '-crf', '18', '-an', f'{out_filepath_mp4}'
    ]
    
    _run_ffmpeg(ffmep_exec_path, args)

def _run_ffmpeg(ffmep_exec_path, command_args):
    import subprocess, os
    
    command_args_no_quotes = [
        arg.replace('"', '').replace("'", '') for arg in command_args
    ]
    full_command = ' '.join(command_args_no_quotes)
    full_command = f'{ffmep_exec_path} {full_command}'
    
    separator = '-'*100
    print(
        f'{separator}\n'
        f'Converting to MP4 with the following command:\n\n'
        f'`{full_command}`\n'
        f'{separator}'
    )
    if is_win:
        subprocess.check_call(full_command)
        return
    
    ffmpeg_exec_path = os.path.join(acdc_ffmpeg_path, 'ffmpeg')
    if is_mac:
        args_ffmpeg_executable = [f'chmod 755 {ffmpeg_exec_path}']
        subprocess.check_call(args_ffmpeg_executable, shell=True)

    try:
        ffmpeg_args = ['ffmpeg', *command_args]
        subprocess.check_call(ffmpeg_args, shell=True)
    except Exception as err:
        ffmpeg_args = ['ffmpeg', *command_args_no_quotes]
        args = ' '.join(ffmpeg_args)
        subprocess.check_call(args, shell=True)