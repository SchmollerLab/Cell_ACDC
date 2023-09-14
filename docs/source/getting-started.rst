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
Through the main menu, all modules of Cell-ACDC can be accessed:

0. Create data structure from microscopy/image file(s)...
    This module will allow you to create a suitable data structure from raw microscopy files.
1. Launch data prep module...
    With this module you can align time-lapse microscopy data and select a sharp image from a z-stack 
2. Launch segmentation module...
    Using this module, you can segment cells.  
3. Launch GUI...
    Lastly, using the GUI the resulting data can be inspected and corrected for mistakes. Also, cell cycle annotation can be carried out.

Top ribbon options include:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Recent paths
    * Here you can access a history of recent paths used in any of the modules.
* Utilities
    * Some extra utilities which might be useful including:
        * Convert file formats
        * Segmentation: Creating 3D segmentation masks
        * Tracking: Tracking and counting sub-cellular objects as well as applying tracking information from tabular data
        * Measurements: Tools for computing measurements
        * Add lineage tree table to one or more experiments
        * Concatenate acdc output tables from multiple Positions
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
