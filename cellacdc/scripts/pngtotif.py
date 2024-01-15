from PIL import Image
import os

def convert_png_to_tif(input_folder, output_tif):
    images = []
    
    # Get a list of all PNG files in the input folder
    png_files = [file for file in os.listdir(input_folder) if file.endswith(".png")]

    if not png_files:
        print("No PNG files found in the specified folder.")
        return

    for png_file in png_files:
        png_path = os.path.join(input_folder, png_file)
        img = Image.open(png_path)
        images.append(img)

    # Save the images as a multi-page TIFF file
    images[0].save(output_tif, save_all=True, append_images=images[1:])

    print(f"Conversion completed. TIFF file saved at {output_tif}")

input_folder = r"C:\Users\SchmollerLab\Documents\Timon\DeepSea_data\test\set_22_MESC\images"
output_tif = r"C:\Users\SchmollerLab\Documents\Timon\DeepSea_data\test\set_22_MESC\images.tiff"

convert_png_to_tif(input_folder, output_tif)
