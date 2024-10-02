from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
import scipy.ndimage as ndimage
import skimage
import numpy as np
import json
import itertools
import shutil
import pandas as pd

from ... import load, myutils

def process_frame(imgs, images_indx, factor, is_segm):
    T, Z = images_indx
    if not is_segm:
        img_resized = ndimage.zoom(imgs[T, Z], factor, order=3)
    else:
        img_resized = ndimage.zoom(imgs[T, Z], factor, order=0)
    return images_indx, img_resized

def process_frames(imgs, factor, is_segm=False):

    results = []

    T, Z = imgs.shape[0], imgs.shape[1]
    images_indxs = list(itertools.product(range(T), range(Z)))
    images = None
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, imgs, images_indx, factor, is_segm) for images_indx in images_indxs]
        for future in futures:
            results.append(future.result())
    

    if not results:
        raise TypeError("No images to process (or this funciton has a funky error)")
    
    images_indx, img_resized = results[0]
    Y, X = img_resized.shape
    images = np.zeros((T, Z, Y, X), dtype=img_resized.dtype)
    
    for result in results:
        images_indx, img_resized = result

        t, z = images_indx
        images[t, z] = img_resized

    return images

def load_images(images_path_in, file_path):
    path = os.path.join(images_path_in, file_path)

    if file_path.endswith(".tif"):
        imgs = load.imread(path)
    elif file_path.endswith(".npz"):
        imgs = np.load(path)["arr_0"]

    print(f"Image shape: {imgs.shape}")

    if imgs.ndim == 2:
        imgs = imgs[np.newaxis, np.newaxis, ...]
    elif imgs.ndim == 3:
        imgs = imgs[np.newaxis, ...]

    return imgs

def save_images(images, filename_in, images_path_out, text_to_append=''):
    if images is None:
        print("No images to save.")
        return
    
    images = np.squeeze(images)
    filename_in_noext, ext = os.path.splitext(filename_in)
    filename_out = f'{filename_in_noext}{text_to_append}{ext}'
    
    images_path_out_file = os.path.join(images_path_out, filename_out)

    if images_path_out_file.endswith(".tif"):
        skimage.io.imsave(images_path_out_file, images, check_contrast=False)
    elif images_path_out_file.endswith(".npz"):
        np.savez_compressed(images_path_out_file, images)

    print(f"Sampling completed. File saved in:")
    print(f"{images_path_out_file}\n")         

def resize_imgs(images_path_in, factor, images_path_out=None, text_to_append=''):
    if images_path_out is None:
        images_path_out = images_path_in
        
    list_dir = myutils.listdir(images_path_in)
    
    # Get a list of all PNG files in the input folder
    images_files = [
        file for file in list_dir if (
            file.endswith(".tif")
            or file.endswith('aligned.npz')
        )
    ]

    if not images_files:
        print("No image files found in the specified folder.")
        return
    
    for filename in images_files:
        print(f"Processing {filename}...")

        images = load_images(images_path_in, filename)

        images = process_frames(images, factor)

        save_images(
            images, filename, images_path_out=images_path_out,
            text_to_append=text_to_append)

def edit_subs_bkgrROIs(
        images_path_in, factor, images_path_out=None, text_to_append=''
    ):
    if images_path_out is None:
        images_path_out = images_path_in
        
    list_dir = myutils.listdir(images_path_in)

    bkgrROIs_jsons = [file for file in list_dir if file.endswith("bkgrROIs.json")]
    bkgrROIs_npzs = [file for file in list_dir if file.endswith("bkgrROIs.npz")]
    
    # Is this fine to interpolate bkgrROIs_npzs or do I get the same issues as
    # with the segmentaion masks?"


    if not bkgrROIs_jsons and not bkgrROIs_npzs:
        return

    for bkgrROIs_json_file in bkgrROIs_jsons:
        print(f"Processing {bkgrROIs_json_file}...")
        bkgrROIs_json_file_path = os.path.join(images_path_in, bkgrROIs_json_file)
        with open(bkgrROIs_json_file_path, 'r') as file:
            data = json.load(file)

        data_scaled = []

        for data_part in data:
            for key, value in data_part.items():
                if key == "angle":
                    continue

                value_scaled = []
                for val in value:
                    val_scaled = int(val * factor)
                    value_scaled.append(val_scaled)

                data_part[key] = value_scaled

            data_scaled.append(data_part)
        
        bkgrROIs_json_file_out = myutils.append_text_filename(
            bkgrROIs_json_file, text_to_append
        )
        images_path_out_file = os.path.join(
            images_path_out, bkgrROIs_json_file_out
        )

        with open(images_path_out_file, 'w') as file:
            json.dump(data_scaled, file)

        print(f'bkgrROIs.json files edited and saved in:')
        print(f'{images_path_out_file}\n')

    for bkgrROIs_npz_file in bkgrROIs_npzs:
        print('WARNING: Not tested yet')

        print(f"Processing {bkgrROIs_npz_file}...")
        
        images = load_images(images_path_in, bkgrROIs_npz_file)

        images = process_frames(images, factor)

        save_images(
            images, bkgrROIs_npz_file, 
            images_path_out=images_path_out, 
            text_to_append=text_to_append
        )

def edit_acdc_csvs(
        images_path_in, factor, images_path_out=None, text_to_append=''
    ):    
    if images_path_out is None:
        images_path_out = images_path_in
        
    columns_for_scaling = ["x_centroid", "y_centroid"]

    acdc_csvs = load.get_acdc_output_files(images_path_in)
    if not acdc_csvs:
        return
    
    for acdc_csv_file in acdc_csvs:
        print(f"Processing {acdc_csv_file}...")
        acdc_csv_file_path = os.path.join(images_path_in, acdc_csv_file)
        if not os.path.exists(acdc_csv_file_path):
            continue
        
        try:
            acdc_df = pd.read_csv(acdc_csv_file_path)
        except PermissionError as e:
            print(f"Implement PermissionError handling")
            raise e

        for column in columns_for_scaling:
            acdc_df[column] = (acdc_df[column] * factor).astype(int)

        acdc_csv_file_out = myutils.append_text_filename(
            acdc_csv_file, text_to_append
        )
        images_path_out_file = os.path.join(images_path_out, acdc_csv_file_out)
        acdc_df.to_csv(images_path_out_file, index=False)
        print(f"Modified CSV saved to:")
        print(f"{images_path_out_file}\n")

def edit_metadata(
        images_path_in, factor, images_path_out=None, text_to_append=''
    ):
    if images_path_out is None:
        images_path_out = images_path_in
        
    list_dir = myutils.listdir(images_path_in)
    data_to_scale_int = ["SizeX", "SizeY"]
    data_to_scale_float = ["PhysicalSizeY", "PhysicalSizeX"]
    metadata_files = [
        file for file in list_dir if file.endswith("metadata.csv")
    ]

    if not metadata_files:
        return

    for metadata_file in metadata_files:
        print(f"Processing {metadata_file}...")
        metadata_file_path = os.path.join(images_path_in, metadata_file)
        with open(metadata_file_path, 'r') as file:
            metadata = file.read()

        new_metadata = ""
        for line in metadata.split("\n"):
            entries = line.split(",")

            if entries[0] in data_to_scale_int:
                entries[1] = str(int(float(entries[1]) * factor))
            elif entries[0] in data_to_scale_float:
                entries[1] = str(float(entries[1]) / factor)

            new_metadata += ",".join(entries) + "\n"

        metadata_file_out = myutils.append_text_filename(
            metadata_file, text_to_append
        )
        images_path_out_file = os.path.join(images_path_out, metadata_file_out)
        with open(images_path_out_file, 'w') as file:
            file.write(new_metadata)

        print(f"Metadata edited and saved in:")
        print(f"{images_path_out_file}\n")

def edit_lost_centroids(
        images_path_in, factor, images_path_out=None, text_to_append=''
    ):
    if images_path_out is None:
        images_path_out = images_path_in
        
    list_dir = myutils.listdir(images_path_in)
    
    lost_centroids_jsons = [
        file for file in list_dir 
        if file.endswith("tracked_lost_centroids.json")
    ]

    if not lost_centroids_jsons:
        return

    for lost_centroids_json in lost_centroids_jsons:
        print(f"Processing {lost_centroids_json}...")
        
        lost_centroids_json_path = os.path.join(images_path_in, lost_centroids_json)

        with open(lost_centroids_json_path, 'r') as file:
            lost_centroids = json.load(file)

        for frame_i, frame in lost_centroids.items():
            frame_new = []
            for centroid in frame:
                new_centroid = []
                for value in centroid:
                    value = int(value * factor)
                    new_centroid.append(value)
                frame_new.append(new_centroid)
            lost_centroids[frame_i] = frame_new

        lost_centroids_json_out = myutils.append_text_filename(
            lost_centroids_json, text_to_append
        )
        images_path_out_file = os.path.join(
            images_path_out, lost_centroids_json_out
        )
        with open(images_path_out_file, 'w') as file:
            json.dump(lost_centroids, file, indent=4)
        
        print(f"Lost centroids edited and saved in:")
        print(f"{images_path_out_file}\n")

def resize_segms(
        images_path_in, factor, images_path_out=None, text_to_append=''
    ):
    if images_path_out is None:
        images_path_out = images_path_in
        
    segm_npzs = load.get_segm_files(images_path_in)

    if not segm_npzs:
        return

    for segm_npz_file in segm_npzs:
        print(f"Processing {segm_npz_file}...")
        
        images = load_images(images_path_in, segm_npz_file)

        images = process_frames(images, factor, is_segm=True)

        save_images(
            images, segm_npz_file, images_path_out=images_path_out, 
            text_to_append=text_to_append
        )

def copy_aux_files(images_path_in, images_path_out=None):
    if images_path_out is None:
        images_path_out = images_path_in

    list_dir = myutils.listdir(images_path_in)
    files_endings = [
        "_last_tracked_i.txt", "_combine_metrics.ini", "_segm_hyperparams.ini"
    ]
    aux_files = [
        file for file in list_dir 
        if any(file.endswith(ending) for ending in files_endings)
    ]
    for aux_file in aux_files:
        print(f"Copying {aux_file}...")
        shutil.copyfile(
            os.path.join(images_path_in, aux_file), 
            os.path.join(images_path_out, aux_file)
        )
        print(f"File {aux_file} copied to")
        print(f"{images_path_out}\n")

def run(
        images_path_in, factor, images_path_out=None, text_to_append=''
    ):
    resize_imgs(
        images_path_in, factor, text_to_append=text_to_append, 
        images_path_out=images_path_out
    )
    edit_subs_bkgrROIs(
        images_path_in, factor, text_to_append=text_to_append, 
        images_path_out=images_path_out
    )
    copy_aux_files(images_path_in, images_path_out=images_path_out)
    resize_segms(
        images_path_in, factor, text_to_append=text_to_append, 
        images_path_out=images_path_out
    )
    edit_acdc_csvs(
        images_path_in, factor, text_to_append=text_to_append, 
        images_path_out=images_path_out
    )
    edit_metadata(
        images_path_in, factor, text_to_append=text_to_append, 
        images_path_out=images_path_out
    )
    edit_lost_centroids(
        images_path_in, factor, text_to_append=text_to_append, 
        images_path_out=images_path_out
    )