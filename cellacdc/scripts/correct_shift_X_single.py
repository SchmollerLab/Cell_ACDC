from calendar import c
from hmac import new
import os

import argparse

import re

import skimage.io

import numpy as np

from cellacdc.plot import imshow
from cellacdc import load

import concurrent.futures

import copy

from itertools import islice

import json


PREVIEW_Z = None
NEW_PATH_SUF = None
INCLUDE_PATTERN_TIF_SEARCH = None

def load_constants():

    print('Loading constants...')
    global PREVIEW_Z
    global INCLUDE_PATTERN_TIF_SEARCH
    with open('regex.txt', 'r') as input_file:
        regex_file = input_file.read()
    for line in regex_file.splitlines():
        if re.search('x_INCLUDE_PATTERN_TIF_SEARCH', line):
            line = line.split(':', 1)[1].strip().lstrip().rstrip(',')
            INCLUDE_PATTERN_TIF_SEARCH = line

    with open('config.json', 'r') as input_file:
        config = json.load(input_file)
    PREVIEW_Z = config['correct_shift_x_single']['PREVIEW_Z']
    NEW_PATH_SUF = config['correct_shift_x_single']['NEW_PATH_SUF']
    return NEW_PATH_SUF

def correct_constant_shift_X_img(img, shift):
    for i, row in enumerate(img[::2]):
        l = i*2
        img[l] = np.roll(row, shift)
    return img

def correct_constant_shift_X(z_stack, shift):
    for z, img in enumerate(z_stack):
        img = correct_constant_shift_X_img(img, shift)
        z_stack[z] = img
    return z_stack

def find_other_tif(file_path):
    folder_path = os.path.dirname(file_path)
    file_list = os.listdir(folder_path)
    tif_files = [filename for filename in file_list if filename.lower().endswith('.tif')]
    return tif_files

def finding_shift(tif_data, shift, start_frame, NEW_PATH_SUF):
    eval_img = (tif_data[start_frame][PREVIEW_Z]).copy()
    eval_img = correct_constant_shift_X_img(eval_img, shift) 
    imshow(tif_data[start_frame][PREVIEW_Z], eval_img)
    while True:
        answer = input('Do you want to proceed with the shift or change it ([y]/n/"number"/help)? ')
        if answer.lower() == 'n':
            exit()
        elif answer.isdigit():
            shift = int(answer)
            shift = finding_shift(tif_data, shift, start_frame, NEW_PATH_SUF)
            return shift
        elif answer.lstrip('-').isdigit():
            shift = int(answer)
            shift = finding_shift(tif_data, shift, start_frame, NEW_PATH_SUF)
            return shift
        elif answer.lower() == 'help':
            print('Change the shown image by changing PREVIEW_Z in the beginning of the code. \nChange the ending of the new file name by changing NEW_PATH_SUF in the code. \nCurrent z stack and z displayed: ' + str(PREVIEW_Z) + '\nCurrent ending: ' + NEW_PATH_SUF)
            finding_shift(tif_data, shift, start_frame, NEW_PATH_SUF)
            return shift
        elif not answer:
            return shift
        elif answer.lower() == 'y':
            return shift
        else:
            print('The input is not an integer')
        

def shiftingstuff_main(shift, tif_data, tif_path, start_frame, end_frame, NEW_PATH_SUF):
    corrected_data = tif_data.copy()
    for frame_i, img in islice(enumerate(tif_data), start_frame, end_frame):
        corrected_data[frame_i] = correct_constant_shift_X(img.copy(), shift)
    new_path = tif_path.replace('.tif', NEW_PATH_SUF + '.tif' )
    skimage.io.imsave(new_path, corrected_data, check_contrast=False)
    del corrected_data
    del tif_data
    return

def shiftingstuff_other(tif_name, shift, tif_path, scan_other, start_frame, end_frame, NEW_PATH_SUF):
    if scan_other == True:
        tif_path = os.path.join(os.path.dirname(tif_path), tif_name)
        tif_data = load.imread(tif_path)
        shiftingstuff_main(shift, tif_data, tif_path, start_frame, end_frame, NEW_PATH_SUF)
        del tif_data
    return

def sequential(NEW_PATH_SUF):
    parser = argparse.ArgumentParser()
    parser.add_argument('tif_path', help='Path to the tif-file')
    parser.add_argument('shift', help='Amount of shift')
    parser.add_argument('frame_start', help='Start of frames which should be shifted')
    parser.add_argument('frame_end', help='End of frames which should be shifted')
    args = parser.parse_args()
    tif_path = args.tif_path
    shift = int(args.shift)
    start_frame = int(args.frame_start)
    end_frame = int(args.frame_end)

    print('Path: \n' + tif_path)
    print('Original Shift: ' + str(shift))
    print('Start from frame: ' + str(start_frame))
    print('End on frame: ' + str(end_frame))

    tif_data = load.imread(tif_path)

    start_frame -= 1

    print('Please close the window after inspecting if the shift value is right in order to proceed.')
    shift = finding_shift(tif_data, shift, start_frame, NEW_PATH_SUF)
    print('Shift used: ' +str(shift))

    tif_files = find_other_tif(tif_path)    
    tif_names = [tif_file for tif_file in tif_files if re.match(INCLUDE_PATTERN_TIF_SEARCH, tif_file)]
    print('New tif file(s) found:\n' + "\n".join(tif_names))

    while True:
        answer = input('Do you want to shift the other .tif files in the folder too? ([y]/n/help)')
        if answer.lower() == 'n':
            scan_other = False
            break
        elif answer.lower() == 'help':
            print('You can change the regex pattern in the beginning of the code (INCLUDE_PATTERN_TIF_SEARCH). \nIf you dont know regex, ask Chat_GPT to generate one for you by giving it examples of file names and then asking it to generate a regex code which excludes the files you want to exclude. \nCurrent expression is: ' + INCLUDE_PATTERN_TIF_SEARCH)
            exit()
        else:
            scan_other = True
            break
    return shift, tif_data, tif_names, scan_other, tif_path, start_frame, end_frame


if __name__ == "__main__":
    NEW_PATH_SUF = load_constants()
    shift, tif_data, tif_names, scan_other, tif_path, start_frame, end_frame = sequential(NEW_PATH_SUF)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        futures = [executor.submit(shiftingstuff_other, tif_name, shift, tif_path, scan_other, start_frame, end_frame, NEW_PATH_SUF) for tif_name in tif_names]
        futures.append(executor.submit(shiftingstuff_main, shift, tif_data, tif_path, start_frame, end_frame, NEW_PATH_SUF))
        results = [future.result() for future in futures]
    print('Done!')
    exit()