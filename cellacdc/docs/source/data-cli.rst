.. _data-cli:

Command-line data tools (acdc-data)
===================================

``acdc-data`` creates the Cell-ACDC data structure from the command line,
without the graphical user interface (GUI). It has three commands:

* ``acdc-data metadata``: read and print the metadata of one raw microscopy
  file.
* ``acdc-data convert``: convert raw microscopy files into ``Position_n``
  folders, like **0. Create data structure from microscopy/image file(s)...**
  in the main menu (see :ref:`creating-data-structure`).
* ``acdc-data restructure``: organize loose TIFF files into ``Position_n``
  folders, like **Re-structure image files** (see
  :doc:`restructure-loose-tiff-files`).

The commands run the same code as the GUI modules, so they write the same
files as the GUI does with the same settings. Use them on computers without
a display, in scripts, or to process many experiments with the same
settings.

The ``acdc-data`` command and the names of its commands were first proposed
by keejkrej in
`pull request #1109 <https://github.com/SchmollerLab/Cell_ACDC/pull/1109>`__.

.. contents::
   :local:

Installation
------------

``acdc-data`` is installed together with Cell-ACDC (see
:doc:`installation`). It does not need the GUI packages (Qt).

``acdc-data metadata`` and ``acdc-data convert`` read the raw files with
`BioIO <https://github.com/bioio-devs/bioio>`__. Install BioIO and the
reader plugin of your file format, for example for TIFF files:

.. code-block:: bash

    pip install bioio bioio-tifffile

Reader packages per file extension:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Extension
     - Package
   * - ``.tif``, ``.tiff``, ``.ome.tif``
     - ``bioio-tifffile``
   * - ``.ome.tiff``
     - ``bioio-ome-tiff``
   * - ``.czi``
     - ``bioio-czi``
   * - ``.nd2``
     - ``bioio-nd2``
   * - ``.lif``
     - ``bioio-lif``
   * - ``.dv``, ``.r3d``
     - ``bioio-dv``
   * - ``.png``, ``.jpg``, ``.jpeg``, ``.gif``
     - ``bioio-imageio``
   * - ``.zarr``
     - ``bioio-ome-zarr``
   * - ``.sldy``, ``.dir``
     - ``bioio-sldy``
   * - ``.ims``
     - ``imaris-ims-file-reader``
   * - any other extension
     - ``bioio-bioformats``, which uses Bio-Formats (Java) through
       ``scyjava``; ``scyjava`` must be installed with ``conda``.

Lazy loading (``convert --lazy-load``, the default) also needs ``zarr``.
``acdc-data restructure`` needs no BioIO package: it reads the TIFF files
with ``tifffile``, which is installed with Cell-ACDC.

Missing reader packages
~~~~~~~~~~~~~~~~~~~~~~~

Before reading any file, ``acdc-data metadata`` and ``acdc-data convert``
check that the packages needed for the selected files are installed:

* With ``--yes``, missing packages are installed without asking, with the
  same versions as in the GUI: ``scyjava`` 1.10.2 with ``conda`` and
  ``bioio-bioformats>=1.0,<2.0`` with ``pip``. ``scyjava`` can only be
  installed in a conda environment; in any other environment the command
  stops with exit code 1.
* Without ``--yes``, in an interactive terminal, ``acdc-data`` asks before
  installing a package (the question is printed on standard error).
* Without ``--yes`` and without an interactive terminal (scripts, jobs on a
  cluster), the command stops with exit code 1 and prints the commands to
  run: ``<python> -m pip install <packages>``, plus
  ``conda install -c conda-forge scyjava`` when ``scyjava`` is missing
  (without the version pins that ``--yes`` uses). Or re-run the command
  with ``--yes``.

The files are always read in a separate worker process, which never
installs packages and never asks questions.

Usage
-----

.. code-block:: bash

    acdc-data COMMAND [options]
    acdc-data COMMAND -h
    acdc-data -h
    acdc-data --version

``-h`` shows the help of ``acdc-data`` or of one command and ``--version``
prints the Cell-ACDC version. The options of a command, including the
common options below, go after the command name, for example
``acdc-data convert -v -i raw -o exp --layout one-per-pos``.

Relative paths are relative to the current folder and ``~`` is expanded:
``acdc-data`` turns every path into an absolute path before it starts.

Common options
~~~~~~~~~~~~~~

Every command accepts these options:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Option
     - Default
     - Description
   * - ``-v``, ``--verbose``
     - off
     - Show debug messages on standard error.
   * - ``-q``, ``--quiet``
     - off
     - Show only warnings and errors on standard error; no progress bar.
   * - ``--log-file PATH``
     - see `Output and log files`_
     - Write the log to PATH.
   * - ``--no-log-file``
     - off
     - Do not write a log file.
   * - ``--yes``
     - off
     - Install missing reader packages (BioIO plugins) without asking
       (``metadata`` and ``convert``).
   * - ``--debug``
     - off
     - Enable the Cell-ACDC debug mode.

``-v`` and ``-q`` cannot be combined, and neither can ``--log-file`` and
``--no-log-file`` (exit code 2).

Safe defaults
~~~~~~~~~~~~~

* Existing ``Position_n`` folders are never written into unless you choose
  so with ``--on-existing``: the default, ``error``, stops the command and
  lists the existing folders that would be written.
* The raw files stay where they are unless you pass ``convert --move-raw``.
* Every check that can be done without writing (options, file names, file
  extensions, metadata, selected positions and channels, the frames,
  z-slices and channels of every raw file and scene, existing
  ``Position_n`` folders) runs before the first file or folder is created,
  renamed, moved or deleted. When a check fails, the command exits with
  code 2 and the source and destination folders are unchanged.
* ``--dry-run`` (``convert`` and ``restructure``) shows what would be
  written, renamed, moved or deleted without changing anything.

Output and log files
~~~~~~~~~~~~~~~~~~~~

* Standard output contains only the result, in the format chosen with
  ``--format``, so it can be redirected to a file or passed to other
  programs.
* Log messages, warnings, the progress bar and the output of the reader
  libraries go to standard error. Warnings start with ``WARNING:`` and
  errors with ``ERROR:``. The progress bar is shown only when standard
  error is a terminal.
* The full log, including debug messages, is also written to a log file,
  by default
  ``<logs folder>/<YYYY-mm-dd_HH-MM-SS>_acdc-data-<command>_<id>.log``.
  The logs folder is the ``.acdc-logs`` folder of the Cell-ACDC user
  profile (``~/acdc-appdata/.acdc-logs`` by default), the same folder as
  for the GUI logs. Choose another file with ``--log-file PATH`` or skip
  it with ``--no-log-file``. When a command fails, the last line on
  standard error is ``See the log file: <path>``.
* ``Ctrl+C`` stops a command with exit code 130. A running worker process
  is stopped and its temporary files are removed; positions that were
  already written stay in the destination folder.

Exit codes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Code
     - Meaning
   * - 0
     - Success: ``metadata`` printed the metadata, ``convert`` wrote at
       least one position, ``restructure`` finished, or a ``--dry-run``
       succeeded.
   * - 1
     - The run failed: for example a file could not be read, the worker
       process failed, the raw files could not be moved, a reader package
       is missing, a file could not be written, or ``convert`` wrote no
       position.
   * - 2
     - Invalid command line, or options that do not fit the input files,
       for example existing ``Position_n`` folders with
       ``--on-existing error``, mixed file extensions without ``--ext``, or
       unknown positions or channels. The source and destination folders
       were not changed (a log file may have been written and, with
       ``--yes``, packages may have been installed).
   * - 130
     - Interrupted with ``Ctrl+C``.

acdc-data metadata
------------------

.. code-block:: bash

    acdc-data metadata -i INPUT [--format {text,json,csv}] [-o DIR] [--basename TEXT] [--include-xml] [common options]

Reads the metadata of one raw microscopy file in a worker process, applies
the same defaults as the GUI and prints the values that ``convert`` would
use.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Option
     - Default
     - Description
   * - ``-i``, ``--input``
     - required
     - Raw microscopy file, or a folder containing exactly one supported
       file (otherwise exit code 2, listing the files).
   * - ``--format {text,json,csv}``
     - ``text``
     - Output on standard output. ``text``: ``key: value`` lines.
       ``json``: resolved values, defaulted fields, warnings and raw reader
       values. ``csv``: the ``Description,values`` rows that ``convert``
       would write without overrides (usable with
       ``convert --metadata-csv``), always encoded as UTF-8 (the encoding
       that ``--metadata-csv`` reads).
   * - ``-o``, ``--output``
     - none
     - Also write ``<basename>metadata.csv`` and
       ``<basename>metadataXML.txt`` into this folder (any ``--format``).
   * - ``--basename``
     - ``<file stem>_``
     - Basename of the files written with ``--output`` and of the
       ``basename`` row of ``--format csv`` (dots in the file stem become
       ``_``). Needs ``--output`` or ``--format csv``.
   * - ``--include-xml``
     - off
     - ``json`` only: include the full metadataXML text (an error with the
       other formats).

Values that are not found in the file fall back to the same defaults as in
the GUI: LensNA 1.4, SizeS, SizeZ, SizeT and SizeC 1, TimeIncrement 1.0 s,
PhysicalSizeX, PhysicalSizeY and PhysicalSizeZ 1.0 um, no channel name
(``convert`` then uses ``channel_<c>`` with a warning) and an emission
wavelength of 0.0 per channel. These fields are listed in ``defaulted``
(``--format json``) and reported as warnings on standard error, together
with the ``convert`` option that sets the value when there is one.

.. note::

   For files read with BioIO (every format except ``.ome.tif``), a pixel
   size that the reader does not report is set to 1.0 without a warning and
   is not listed in ``defaulted``, exactly as in the GUI. When
   PhysicalSizeX, PhysicalSizeY or PhysicalSizeZ is 1.0, check the real
   pixel size of your images and set it with ``--physical-size-x``,
   ``--physical-size-y`` and ``--physical-size-z`` or with
   ``--metadata-csv``.

JSON output
~~~~~~~~~~~

``--format json`` prints one JSON object:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Content
   * - ``schema``
     - ``"acdc-data/metadata/1"``
   * - ``file``
     - The file that was read.
   * - ``reader``
     - The reader that read it.
   * - ``metadata``
     - Values with the defaults applied: ``LensNA``, ``SizeS``, ``SizeT``,
       ``SizeZ``, ``SizeC``, ``TimeIncrement``, ``TimeIncrementUnit``,
       ``TimeIncrement_s``, ``PhysicalSizeX``, ``PhysicalSizeY``,
       ``PhysicalSizeZ``, ``PhysicalSizeUnit``, ``PhysicalSizeX_um``,
       ``PhysicalSizeY_um``, ``PhysicalSizeZ_um``, ``ImageName``,
       ``channel_names``, ``channel_names_sanitized``, ``em_wavelens``.
       ``TimeIncrement_s`` and ``PhysicalSize*_um`` are the values
       converted to seconds and micrometers, or ``null`` when the unit is
       unknown.
   * - ``defaulted``
     - Keys of ``metadata`` whose value fell back to a default.
   * - ``warnings``
     - One object per value that was not found in the file, with
       ``field`` (the key of ``metadata``, e.g. ``"LensNA"`` or
       ``"ImageName"``), ``file`` (the file) and ``message`` (e.g.
       ``"WARNING: LensNA not found in metadata."``).
   * - ``raw``
     - The values as read from the file (``null`` when not found), without
       the metadataXML text.
   * - ``metadata_xml_length``
     - Number of characters of the metadataXML text.
   * - ``metadata_xml``
     - The metadataXML text (only with ``--include-xml``).

When ``metadata`` fails, nothing is printed on standard output (with any
``--format``).

acdc-data convert
-----------------

.. code-block:: bash

    acdc-data convert -i SRC -o DST --layout {single-multi-pos,one-per-pos,one-per-channel} [options] [common options]

Converts the raw microscopy files in SRC into ``Position_n/Images`` folders
in DST, like Module 0 of the GUI. One worker process reads the metadata of
all files, then every position is planned and checked, a second worker
process checks the planned frames, z-slices and channels against every raw
file and scene, and a third worker process writes all positions
(``--dry-run`` stops after the second one). ``convert`` never asks for
confirmation: it uses the metadata of the files, with the defaults
described in `acdc-data metadata`_, and the metadata options below change
single values.

Layouts
~~~~~~~

``--layout`` tells how the raw files are organized, like the
**Raw data structure** option of the GUI:

* ``single-multi-pos``: one file contains all positions (scenes). If SRC
  contains several files, select one with ``--files``.
* ``one-per-pos``: one file per position. The positions are numbered in
  the natural sort order of the file names (1 = first file).
* ``one-per-channel``: one file per channel and position, named
  ``<basename><N>_<channel>.<ext>``, for example ``ASY015_1_GFP.tif`` and
  ``ASY015_1_mCherry.tif``. ``N`` is the position number and the text after
  it is the channel name. The metadata is read from the first file.

Options
~~~~~~~

All values are checked before any file or folder is changed. An invalid
value or combination of options gives exit code 2, and so does a raw file
or scene that does not fit the plan: the plan comes from the metadata of
the first file (``one-per-channel``) or of the first scene
(``single-multi-pos``), so for example a channel file or a scene with
fewer frames needs a ``--time-range`` that fits it.

.. list-table:: Input
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``-i``, ``--input``
     - required
     - Folder with the raw microscopy files.
   * - ``-o``, ``--output``
     - required
     - Experiment folder that receives the ``Position_n`` folders (created
       when writing starts).
   * - ``--layout``
     - required
     - ``single-multi-pos``, ``one-per-pos`` or ``one-per-channel`` (see
       `Layouts`_).
   * - ``--files NAME [NAME ...]``
     - all files
     - Only use these file names (in natural sort order). Required for
       ``single-multi-pos`` when SRC has several files.
   * - ``--ext EXT``
     - none
     - Only use files with this extension (case-insensitive, e.g. ``.czi``
       or ``.ome.tif``). Required when SRC has mixed extensions and
       ``--files`` is not given.
   * - ``--rename-invalid {underscore,dash}``
     - off
     - Rename files whose names contain invalid characters (as the GUI
       offers), replacing them with ``_`` or ``-``. Without this option
       such files are an error. Renaming happens right before writing.
   * - ``--other-files {leave,move,copy}``
     - ``leave``
     - ``one-per-channel`` only: what to do with the other files of each
       position (files starting with ``<basename><N>_`` that are not
       channel images, e.g. notes). Error with the other layouts.

.. list-table:: Output
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``--on-existing {error,overwrite,add,create-new}``
     - ``error``
     - When DST already has ``Position_n`` folders. ``error``: stop and
       list the existing folders that would be written. ``overwrite``:
       DELETE ``Position_n/Images`` (including segmentation and
       ``acdc_output`` files) of every position written. ``add``: write
       into the existing folders and merge ``metadata.csv``.
       ``create-new``: number the new positions after the highest existing
       one.
   * - ``--move-raw``, ``--no-move-raw``
     - off
     - Move the raw files into ``DST/raw_microscopy_files`` right before
       writing (GUI behaviour; symbolic link files then point to the moved
       files).
   * - ``--use-symlinks``
     - off
     - Write ``<basename>symlink.ini`` files pointing to the raw files
       instead of image files. Not combinable with ``--image-format h5``.
   * - ``--image-format {tif,h5}``
     - ``tif``
     - Output image format.
   * - ``--lazy-load``, ``--no-lazy-load``
     - lazy
     - ``--no-lazy-load`` loads each position into RAM at once (GUI option
       **Load entire field of view in RAM at once**).

.. list-table:: Metadata
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``--lens-na F``
     - file, else 1.4
     - Objective numerical aperture (a number greater than 0).
   * - ``--time-increment F``
     - file, else 1.0 s
     - Frame interval, in ``--time-unit`` (a number greater than 0).
   * - ``--time-unit {ms,s,min,h}``
     - ``s``
     - Unit of ``--time-increment`` (only together with
       ``--time-increment``).
   * - ``--physical-size-x F``, ``--physical-size-y F``,
       ``--physical-size-z F``
     - file, else 1.0 um
     - Pixel size, in ``--physical-size-unit`` (a number greater than 0).
   * - ``--physical-size-unit {nm,um,mm,cm}``
     - ``um``
     - Unit of ``--physical-size-x``, ``--physical-size-y`` and
       ``--physical-size-z`` (only together with at least one of them).
   * - ``--channels A,B,...``
     - from the file (``one-per-channel``: from the file names)
     - Output channel names, one per channel. Characters other than
       letters, digits and ``_`` become ``_`` (trailing ``_`` are removed).
       The names must be non-empty and unique (ignoring upper and lower
       case) after that.
   * - ``--em-wavelens NM,NM,...``
     - file, else 0.0 (``one-per-channel``: 500.0)
     - Emission wavelengths, one per channel (numbers >= 0).
   * - ``--save-channels A,B,...``
     - all
     - Channel names (after ``--channels``) to save.
   * - ``--positions LIST``
     - all
     - Input positions to save: ``N``, ``Position_N`` or ``A-B``,
       comma-separated. ``single-multi-pos``: scene number;
       ``one-per-pos``: file order (1 = first file); ``one-per-channel``:
       number in the file name.
   * - ``--time-range START:END``
     - all
     - Frames to save, 1-based and inclusive like the GUI (e.g. ``5:20``,
       ``5:``, ``:20``, ``7``).
   * - ``--add-image-name``
     - off
     - Add the series image name (OME-TIFF) to every output file name.
       Error (exit code 2) when a selected file has no image name.
   * - ``--metadata-csv PATH``
     - none
     - ``Description,values`` overrides (as written by
       ``metadata --format csv``). Explicit options win over the CSV file,
       the CSV file wins over the raw file.

.. list-table:: Run
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``--dry-run``
     - off
     - Read the metadata, check the raw files and print the plan
       (positions, names, files to move, rename or delete); change
       nothing.
   * - ``--format {text,json}``
     - ``text``
     - Format of the plan or result summary on standard output.
   * - ``--keep-temp``
     - off
     - Keep the folders of the worker jobs (for debugging).

Metadata values
~~~~~~~~~~~~~~~

Each value comes, in this order of priority, from the metadata option
(e.g. ``--lens-na``), the ``--metadata-csv`` file, the raw file, or the
default. The time increment and the pixel sizes are converted to seconds
and micrometers before they are written.

The ``--metadata-csv`` file has the ``Description,values`` format written
by ``acdc-data metadata --format csv``. It may contain the rows
``LensNA``, ``TimeIncrement`` (in seconds), ``PhysicalSizeX``,
``PhysicalSizeY``, ``PhysicalSizeZ`` (in micrometers), ``channel_<c>_name``
and ``channel_<c>_emWavelen``. The rows ``basename``, ``SizeT`` and
``SizeZ`` are ignored with a warning; any other row is an error.

The sizes (SizeT, SizeZ, SizeC and SizeS) always come from the file. Select
the data to save with ``--time-range``, ``--positions`` and
``--save-channels``.

Position folders and file names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The output folder of an input position is ``Position_<N>``, with ``N``
  the number used by ``--positions`` (scene number, file order or number
  in the file name). For example ``--positions 2,3`` writes ``Position_2``
  and ``Position_3``.
* With ``--on-existing create-new``, the new positions are instead
  numbered after the highest existing ``Position_n`` folder.
* Every written position gets ``<basename>metadata.csv``,
  ``<basename>metadataXML.txt`` and one image file per saved channel
  (``<basename><channel>.tif``, ``.h5`` with ``--image-format h5``, or a
  single ``<basename>symlink.ini`` with ``--use-symlinks``), as in the GUI.
* ``<basename>`` is ``<file stem>_s<N>_``, or
  ``<file stem>_<image name>_s<N>_`` with ``--add-image-name``. Dots in the
  file stem become ``_`` and ``N`` is the Position number, zero-padded to
  the number of digits of the number of positions. For ``one-per-channel``
  files, the file stem is the ``<basename>`` part of the file names.

JSON result
~~~~~~~~~~~

``--format json`` prints one JSON object; ``status`` and ``error`` tell
whether the command succeeded:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Content
   * - ``schema``
     - ``"acdc-data/convert/1"``
   * - ``command``
     - ``"convert"``
   * - ``status``
     - ``"ok"`` or ``"error"``
   * - ``exit_code``
     - The exit code of the command.
   * - ``dry_run``
     - ``true`` with ``--dry-run``.
   * - ``source``, ``destination``, ``layout``
     - SRC, DST and ``--layout``.
   * - ``positions``
     - One object per written position (with ``--dry-run``: per planned
       position) with ``input_id``, ``output`` (e.g. ``"Position_3"``),
       ``source_file``, ``series``, ``images_path``, ``channels`` (names
       of the saved channels), ``time_range`` (first and last saved
       frame, 1-based), ``files`` (empty with ``--dry-run``) and
       ``other_files`` (``one-per-channel``: the other files moved or
       copied into the position, each with ``src``, ``dst`` and
       ``mode``). After an error, the positions written before the
       error.
   * - ``skipped``
     - Input positions that were not selected.
   * - ``moved``, ``renamed``
     - ``[old path, new path]`` pairs of the moved and renamed raw files
       (with ``--dry-run``: the planned ones).
   * - ``deleted_images_folders``
     - ``Images`` folders deleted by ``--on-existing overwrite`` (after an
       error: deleted or partly deleted, for example when a file in the
       folder was open in another program; with ``--dry-run``: the folders
       that would be deleted).
   * - ``warnings``
     - One object per value that was not found in a raw file and fell
       back to its default, with ``field`` (the key of the
       ``metadata`` object of ``acdc-data metadata``), ``file`` and
       ``message`` (as for ``metadata``). Other warnings, e.g. about a channel
       without a name or an ignored ``--metadata-csv`` row, are
       written only to standard error and the log file.
   * - ``error``
     - ``null``, or an object with ``type``, ``message`` and ``hint``.
   * - ``log_file``
     - Path of the log file.

.. _data-cli-restructure:

acdc-data restructure
---------------------

.. code-block:: bash

    acdc-data restructure -i SRC -o DST [--recipe SPEC] [dimension options] [options] [common options]

Organizes loose TIFF files into ``Position_n/Images`` folders, like
**Re-structure image files** in the GUI. See
:doc:`restructure-loose-tiff-files` for how the files are grouped into
positions and what is written. ``restructure`` reads only the TIFF files
directly inside SRC and runs in the ``acdc-data`` process.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``-i``, ``--input``
     - required
     - Folder with the TIFF files (not recursive).
   * - ``-o``, ``--output``
     - required
     - Destination experiment folder.
   * - ``--recipe SPEC``
     - none
     - Recipe JSON saved by the GUI **Re-structure image files** dialog: a
       file path, ``last`` (last GUI setup) or ``NAME`` (saved recipe).
       The options below replace the matching recipe entries.
   * - ``--channel NAME[=ID]``
     - ``Channel_1``
     - Channel name and, for one file per channel, the text identifying it
       in the file names. Repeat it for every channel. If one
       ``--channel`` has ``=ID``, all must.
   * - ``--channel-axis N``
     - none
     - Channels are axis ``N`` inside each TIFF (give one ``--channel`` per
       channel, without ``=ID``).
   * - ``--time-marker TEXT``, ``--time-axis N``
     - not present
     - Timepoints: one file per frame (``TEXT`` followed by the frame
       number, e.g. ``_t``) or axis ``N`` inside each TIFF. Mutually
       exclusive.
   * - ``--z-marker TEXT``, ``--z-axis N``
     - not present
     - Same for z-slices.
   * - ``--basename TEXT``
     - none
     - Prefix of the output files (letters, digits, ``_`` and ``-``
       only).
   * - ``--files NAME [NAME ...]``
     - all TIFFs
     - Only use these TIFF files: names of ``.tif`` or ``.tiff`` files
       directly inside SRC, spelled exactly as in the folder (also upper
       and lower case, on Windows and macOS too). Other names are an error.
   * - ``--on-existing {error,overwrite,create-new}``
     - ``error``
     - ``overwrite`` DELETES ``Position_n/Images`` of every position
       written; ``create-new`` numbers the new positions after the highest
       existing one.
   * - ``--physical-size-x UM``, ``--physical-size-y UM``,
       ``--physical-size-z UM``
     - recipe, else 1.0
     - Pixel size in micrometers (``metadata.csv`` rows; a number greater
       than 0).
   * - ``--time-increment V``, ``--time-unit {ms,s,min,h}``
     - recipe, else 1.0 s
     - Frame interval (``metadata.csv`` row, converted to seconds; a number
       greater than 0).
   * - ``--dry-run``
     - off
     - Print identifier -> ``Position_n``, the files of every channel, the
       T/Z sizes and the folders to delete; write nothing in SRC or DST
       (``--save-recipe`` still writes the recipe file).
   * - ``--save-recipe SPEC``
     - none
     - Write the effective recipe in the GUI format (a file path, or
       ``NAME`` in the GUI recipes folder).
   * - ``--format {text,json}``
     - ``text``
     - Format of standard output.

Dimensions
~~~~~~~~~~

The dimension options set the three dimensions of the GUI dialog:

.. list-table::
   :header-rows: 1
   :widths: 19 27 27 27

   * - GUI choice
     - Channels
     - Timepoints
     - Z-slices
   * - Not present
     - one ``--channel NAME`` (default ``Channel_1``)
     - default
     - default
   * - One file per ...
     - ``--channel NAME=ID`` for every channel
     - ``--time-marker TEXT``
     - ``--z-marker TEXT``
   * - In file
     - ``--channel-axis N`` and one ``--channel NAME`` per channel
     - ``--time-axis N``
     - ``--z-axis N``

Axis numbers are zero-based, as in the dialog. Files that only differ by
the channel identifiers and the time or z markers belong to the same
position.

Recipes
~~~~~~~

``--recipe`` loads a recipe saved with **Save current recipe...** in the
**Re-structure image files** dialog:

* a path to a recipe JSON file;
* ``last``: the setup of the last conversion started from the dialog;
* ``NAME``: the recipe ``datastruct_restructure_recipe_NAME.json`` in the
  recipes folder of the dialog (``restructure_files_recipes`` in the
  Cell-ACDC settings folder). An unknown name is an error that lists the
  available names.

Options given on the command line replace the matching recipe entries.
``--save-recipe`` writes the effective setup in the same format, so it can
be loaded in the dialog (**Load saved recipe...**) or with ``--recipe``.

JSON result and exit codes
~~~~~~~~~~~~~~~~~~~~~~~~~~

``--format json`` prints one JSON object with the ``schema``
``"acdc-data/restructure/1"``, the plan (what ``--dry-run`` prints:
``source_folder``, ``destination_folder``, ``dimensions``, ``prefix``,
``overwrite``, ``create_new``, ``positions`` with ``identifier``,
``number``, ``folder``, ``size_t``, ``size_z``, ``yx``, ``min_time``,
``min_z`` and ``channels`` (``name``, ``files``, ``image_file``) per
position, ``to_delete`` and ``metadata_rows``) and
the keys ``status``, ``exit_code``, ``position_folders`` (identifier ->
``Position_n`` folder), ``image_files`` (image files written),
``deleted_images_folders``, ``error`` and ``log_file`` (as for
``convert``). With ``--dry-run``, ``position_folders`` and ``image_files``
are empty and ``deleted_images_folders`` lists the folders that would be
deleted. When planning fails, the plan keys are missing and these three
keys are empty; when writing fails, they are ``null`` (what was written
before the error is not known).

A problem found while planning (missing channel files, different Y/X
sizes, overlapping files, existing ``Position_n`` folders, a pixel type
that OME-TIFF cannot store, invalid options) gives exit code 2 before
anything is written; an error while writing gives exit code 1.

Examples
--------

.. code-block:: bash

    # Metadata of one file as JSON
    acdc-data metadata -i raw/exp1.czi --format json > exp1.json

    # Edit the metadata in a CSV file, then use it for the conversion
    acdc-data metadata -i raw/pos1.nd2 --format csv > metadata.csv
    acdc-data convert -i raw -o exp --layout one-per-pos --metadata-csv metadata.csv

    # Check a conversion first (one file with several positions)
    acdc-data convert -i raw -o exp --layout single-multi-pos --channels phase,GFP --time-range 1:50 --dry-run

    # Add new positions, move the raw files and write symbolic links
    acdc-data convert -i raw -o exp --layout one-per-pos --on-existing create-new --move-raw --use-symlinks

    # Loose TIFF files with one file per channel and per timepoint
    acdc-data restructure -i tiffs -o exp --channel GFP=_GFP --channel mCherry=_mCh --time-marker _t

    # Use the last setup of the Re-structure dialog
    acdc-data restructure -i tiffs -o exp --recipe last --on-existing create-new

Not supported
-------------

* ``convert``: the python-bioformats backend of the GUI; changing SizeT,
  SizeZ, SizeC or SizeS; different metadata per scene in multi-scene files
  (the values of the first scene are used for every scene, as in the GUI).
* ``restructure``: moving the source files (they are only read); importing
  segmentation masks; Position numbers taken from the file names (the
  positions are numbered in the natural sort order of their identifiers);
  splitting the files into several experiment folders; input files other
  than TIFF;
  automatic detection of the time, z or channel patterns in the file
  names; a common time origin for all positions (the smallest frame index
  of each position becomes its first frame); skipping files that do not
  match the setup with a warning (select the files with ``--files``
  instead).

Differences from pull request #1109
-----------------------------------

The first version of ``acdc-data`` (pull request #1109, not released) had
some options that changed. Removed options are rejected with exit code 2
and a message that names the replacement.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Pull request #1109
     - Now
   * - ``convert --format tif|h5``
     - ``--image-format tif|h5``; ``--format`` now selects the format of
       standard output (``text|json``) of every command.
   * - ``convert --size-t``, ``--size-z``, ``--size-c``, ``--size-s``
     - Removed: sizes are read from the file and cannot be overridden; use
       ``--time-range``, ``--positions``, ``--save-channels`` to select
       data.
   * - ``convert --trust-metadata``
     - Removed: the file metadata is always used; override single values
       with the metadata options or ``--metadata-csv``.
   * - ``convert --basename``
     - Removed: output names follow the GUI (``<file stem>_s<NN>_``); the
       ``one-per-channel`` basename is detected from the file names.
   * - ``convert --on-existing`` (default ``overwrite``)
     - Default ``error``; ``overwrite`` deletes the ``Images`` folder of
       every position written.
   * - ``convert --move-raw`` (default on; after the conversion, into
       ``<input>/raw_microscopy_files``)
     - Off by default; ``--move-raw`` moves the raw files before writing,
       into ``<output>/raw_microscopy_files``, as the GUI does.
   * - ``convert --time-range`` (0-based)
     - 1-based and inclusive, as in the GUI.
   * - ``convert --save-channels true,false,...``
     - Channel names; unknown names are an error. ``--channels`` and
       ``--em-wavelens`` need one value per channel.
   * - ``convert --positions``
     - Selects input positions for every layout; unknown positions are an
       error.
   * - ``convert --metadata-csv`` (applied over the options)
     - The options win over the CSV file.
   * - Emission wavelength default 500.0; channel names used as given
     - 0.0 per channel without a value (500.0 for ``one-per-channel``);
       channel names are sanitized as in the GUI.
   * - ``metadata --format json``
     - New schema ``acdc-data/metadata/1``; ``-o`` writes the files for
       every ``--format``; a folder given with ``-i`` must contain exactly
       one supported file.
   * - ``restructure --layout multi-timepoint|multi-channel``
     - Dimension options (``--channel``, ``--channel-axis``,
       ``--time-marker``, ``--time-axis``, ``--z-marker``, ``--z-axis``)
       or ``--recipe``.
   * - ``restructure --channels A,B``
     - Repeat ``--channel NAME[=ID]``.
   * - ``restructure --action copy|move``
     - Not supported: the source TIFF files are only read and new TIFF
       files are written.
   * - ``restructure --segm-folder``
     - Not supported: segmentation masks are not imported.
   * - ``restructure --basename``
     - Output files are ``<basename>_<channel>.tif``, as in the GUI.
   * - Exit code 0 on errors; log messages on standard output;
       ``acdc-data -h`` showed the help of ``acdc``
     - Exit codes 0, 1, 2 and 130; standard output contains only the
       result; ``acdc-data -h`` shows its own help.

New ``convert`` options: ``--use-symlinks``, ``--other-files``,
``--rename-invalid``, ``--ext``, ``--files``, ``--dry-run`` and
``--keep-temp``; ``--yes`` is new for every command. Mixed file extensions
and invalid file names are now errors instead of being handled silently,
and ``--add-image-name`` fails when a file has no image name.
