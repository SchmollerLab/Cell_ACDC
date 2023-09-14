Getting started
===============
Running Cell-ACDC
-----------------

1. Open a terminal (on Windows use the Anaconda Prompt if you installed
   with ``conda``)
2. Activate the environment (conda: ``conda activate acdc``, pip on
   Windows: ``.\env\Scripts\activate``, pip on Unix:
   ``source env/bin/activate``)
3. Run the command ``acdc`` or ``cellacdc``

The Main Menu
-------------
The main menu is a hub through which you can access all relevant modules.

.. Image:: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/MainMenu.png?raw=true
    :alt: Overview of the main menu
    :target: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/MainMenu.png

Overview of the main menu

Module buttons
~~~~~~~~~~~~~~
Through the main menu, all modules of Cell-ACDC can be accessed:

0. Create data structure from microscopy/image file(s)...
    This module will allow you to create a suitable data structure from raw microscopy files.
1. Launch data prep module...
    With this module you can align time-lapse microscopy data and select a sharp image from a z-stack, as well as crop images
2. Launch segmentation module...
    Using this module, you can perform segmentation and tracking tasks using common methods.
3. Launch GUI...
    Lastly, using the GUI the resulting data can be inspected and corrected for mistakes. Also, cell cycle annotation can be carried out.

Top ribbon options include:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Recent paths
    * Here you can access a history of recent paths used in any of the modules.
* Utilities
    * Some extra utilities which might be useful, including:
        * Convert file formats
        * Segmentation: Creating 3D segmentation masks
        * Tracking: Tracking and counting sub-cellular objects as well as applying tracking information from tabular data
        * Measurements: Tools for computing measurements
        * Add lineage tree table to one or more experiments
        * Concatenate acdc output tables from multiple positions
        * Create required data structure from image files
        * Re-apply data prep steps to selected channels
        * Align or revert alignment
        * Rename files by appending additional text
* Settings
    * Allows manipulation of the user profile path.
* Napari
    * View the Napari lineage tree. 
* Help
    * Provides links to user manuals and start up guide, as well as a link to the relevant paper for citation and guides on how to contribute, viewing the log files and show additional info about Cell-ACDC.

Additional options include:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Restoring all open windows
        Restores all windows which were minimized 
* Closing the application and all active modules
* Toggle switch for dark mode

Creating Data Structures
------------------------
**0. Create data structure from microscopy/image file(s)...**

The first step in analysing data with Cell-ACDC is creating a suitable data structure from raw microscopy files. This can be done completely automated using module 0.

To start off, launch the module by pressing on the corresponding button in the main menu.

This will open a window in which you can choose how you want to proceed.

.. note::

    Cell-ACDC can use the Bio-Formats or the AICSlmagelO libraries to read microscopy files.

    Bio-Formats requires Java and a python package called javabridge, that will be automatically installed if missing. We recommend using Bio-Formats, since it can read the metadata of the file, such as pixel size, numerical aperture etc.

    If Bio-Formats fails, try using AICSlmagelO.

    Alternatively, if you already pre-processed your microscopy files into TIF files, you could choose to simply re-structure them into the Cell-ACDC compatible format.

After choosing an option, another window will open prompting you to select what kind of data you want to extract from the raw microscopy file:

* Single microscopy file with one or more positions

* Multiple microscopy files, one for each position

* Multiple microscopy files, one for each channel

* NONE of the above

Please select the appropriate option. Afterwards, you are prompted to create an empty folder in which only the microscopy file(s) are present. After doing so, select “Done”. Afterwards, you will be prompted to select this folder. After selecting the destination folder, which by default is the folder you selected in the step before, Cell-ACDC will attempt to load OEM metadata.

After a short wait, a window with the extracted metadata should appear. Make sure to double check all values and **change “Order of Dimensions”** to the appropriate value. To double check if the dimensions are in the correct order, press on the eye icon next to “Channel 0” and use the scrollbars to go through the z coordinate and time coordinate. Once all values are in order, press “Ok”. If the values are the same for all positions, feel free to click “Use the above metadata for all the next positions”. Note that if you have several files, and you press “Ok” and not one of the two other options, the process will stop after each file, and you need to confirm the metadata again.

Each position is saved in a separate folder. The metadata are stored both in a TXT and SCV file, while the channels are stored in separate TIF files.

.. note:: 
    A computer with sufficient RAM is needed in this step! The required amount is heavily reliant on the size of the project.

    It is good practice to keep the original files for future reference, even though they are not needed in the future steps.

.. image:: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc1.png?raw=true
    :target: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc1.png
    :alt: Menu for selecting original file structure

.. image:: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc2.png?raw=true
    :target: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc2.png
    :alt: Second menu for selecting original file structure

.. image:: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc3.png?raw=true
    :target: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc3.png
    :alt: Prompt for creating a empty folder and putting microscopy files inside

.. image:: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc4.png?raw=true
    :target: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc4.png
    :alt: Folder selection

.. image:: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc5.png?raw=true
    :target: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc5.png
    :alt: Metadata menu

.. image:: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc6.png?raw=true
    :target: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc6.png
    :alt: Window for checking order of dimensions

.. image:: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc7.png?raw=true
    :target: https://github.com/Teranis/Cell_ACDC/blob/UserManual/docs/source/images/DataStruc7.png
    :alt: Data structure