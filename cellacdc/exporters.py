import numpy as np
import cv2

import skimage.io
import skimage.color

import pyqtgraph.exporters

from . import transformation, printl, myutils
from . import is_mac, is_win
from . import acdc_ffmpeg_path

class ImageExporter(pyqtgraph.exporters.ImageExporter):
    def __init__(self, item, background=(0, 0, 0, 0), save_pngs=True):
        super().__init__(item)
        self._save_pngs = save_pngs
    
        self.parameters()['background'] = (0, 0, 0, 0)
    
    def export(self, filepath):
        super().export(filepath)
        
        # Remove padding
        img = skimage.io.imread(filepath)
        img_gray = img.sum(axis=2)
        
        _, crop_slice = transformation.remove_padding_2D(
            img_gray, return_crop_slice=True
        )
        img_cropped = img[crop_slice]
        
        if self._save_pngs:
            skimage.io.imsave(filepath, img_cropped, check_contrast=False)

        img_bgr = cv2.cvtColor(img_cropped, cv2.COLOR_RGBA2BGR)
        
        return img_bgr

class SVGExporter(pyqtgraph.exporters.SVGExporter):
    def __init__(self, item):
        super().__init__(item)

class VideoExporter:
    def __init__(self, avi_filepath, fps):
        self.writer = None
        self._avi_filepath = avi_filepath
        self._fps = fps
    
    def add_frame(self, img_bgr):
        if self.writer is None:
            height, width = img_bgr.shape[:-1]
            self.writer = cv2.VideoWriter(
                self._avi_filepath, 0, self._fps, (width, height)
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
    
    full_command = ' '.join(command_args)
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
    
    ffmpeg_args = ['ffmpeg', *command_args]
    try:
        subprocess.check_call(ffmpeg_args, shell=True)
    except Exception as err:
        args = ' '.join(ffmpeg_args)
        subprocess.check_call(args, shell=True)