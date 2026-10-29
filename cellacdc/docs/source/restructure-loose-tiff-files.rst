Restructure loose TIFF files
============================

Use this workflow to organize TIFF image files into the folder structure
used by Cell-ACDC. The conversion creates one TIFF file per channel inside
each position folder.

Open the module
---------------

1. Click **0. Create data structure from microscopy/image file(s)…**.
2. Choose **Re-structure image files**.
3. In the setup dialog, select the source folder and destination folder,
   configure the image dimensions, and click **OK** to start the conversion.

The source folder should contain the TIFF files to restructure. The module
reads TIFF files directly inside that folder.

Configure the dialog
--------------------

Source and destination folders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Source folder**: folder containing the loose TIFF files.
* **Destination folder**: folder where the ``Position_n`` folders will be
  created.
* **Segmentation mask folder**: optionally select the folder containing
  segmentation masks associated with the images.
* **Basename**: optionally enter a shared filename prefix for the generated
  channel TIFFs.

Choose how each dimension is stored
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **Channels**, **Timepoints**, and **Z-slices**, choose one of these
options:

* **Not present**: the input files do not represent this dimension. The
  output has one channel, timepoint, or z-slice for that dimension.
* **One file per ...**: the dimension is represented by separate TIFF files.
  Enter the filename identifier for each channel, or the filename marker for
  timepoints and z-slices. A timepoint or z-slice marker is literal text
  immediately followed by a number. For example, ``_z`` identifies the
  z-slice in ``tile_x001_y001_z001.tif``.
* **In file**: the dimension is stored in the TIFF array. Enter its
  **zero-based axis index** in the array shape returned when the TIFF is
  read. For example, axis index ``0`` is the first array axis.

When you set a channel count, the dialog displays one channel-name field per
channel. If channels are stored in separate files, also provide the filename
identifier that distinguishes each channel.

Position identifier preview
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dialog shows a position identifier preview based on the first
naturally sorted TIFF in the source folder. It removes the filename markers
and channel identifiers for dimensions set to **One file per ...**. The
remaining filename stem is used to group files into positions.

For example, if the first file is ``tile_x001_y001_z001.tif`` and ``_z`` is
the marker for z-slices, the preview is ``tile_x001_y001``. During
conversion, the same identifiers are removed from each TIFF filename. Files
with the same remaining stem are grouped in the same position.

Conversion output
-----------------

For each position, the module creates an ``Images`` folder and writes one
OME-TIFF file per channel. Each output image is arranged in
``(T, Z, Y, X)`` order. Dimensions not present in the input have a size of
one. Missing timepoint or z-slice indices in separate-file data are filled
with zeros; the smallest index found for a position becomes the first output
frame or slice.

The resulting structure is similar to:

.. code-block:: text

   destination_folder/
   ├── Position_1/
   │   └── Images/
   │       ├── <basename>_<channel_1>.ome.tif
   │       └── <basename>_<channel_2>.ome.tif
   └── Position_2/
       └── Images/
           └── ...

If no basename is provided, the output filenames contain the channel name.

See also
--------

See the `Cell-ACDC folder structure documentation
<https://cell-acdc.readthedocs.io/en/latest/data-structure.html>`_ for more
information about the folders and image files used by Cell-ACDC.
