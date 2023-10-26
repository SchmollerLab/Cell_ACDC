.. role:: raw-latex(raw)
   :format: latex

Scripts to correct shifts in bidirectional scanning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using a bidirectional microscope can cause shifts in every second
x-axis, resulting in extremely bad segmentation and tracking
performance. These scripts correct this. 

**Work-flow**

Run this **before** dataprep but after creating a structure out of microscopy file. 

*If you have multiple positions*

1. activate the ``acdc`` environment 
2. Navigate to ``cd \Cell_ACDC\cellacdc\scripts`` (if you have installed Cell_ACDC in a different directory navigate to that directory first) 
3. Use ``python correct_shift_X_multi.py [path to folder] [initial shift]`` to run the script. Path to folder would be for example ``C:\Users\SchmollerLab\Documents\MyData\TimelapseExp\2022-08-04`` when one of the .tif files is in ``C:\Users\SchmollerLab\Documents\MyData\TimelapseExp\2022-08-04\Position_8\Images``
4. The program will run, pay attention to the console for user input requests!
  
.. note::

   After the program is finished, you can run the data prep module. Check before running it that all frames are fine now, as sometimes the sign of the shift changes for a few frames. In this case, you can use the ``correct_shift_X_single.py`` script. Sometimes you need to use a negative shift! Most of the times (but not always) the shift is the same for all positions. 

   The expected structure of the folder is as the “Create data structure” module from Cell_ACDC creates.

*If you have a single position*

As above, except in step 3. You need to run ``python correct_shift_X.py [path to tif file] [initial shift]``

.. note::

  Notes As above

*If you need to change single frames*


As above, except in step 3. You need to run ``python correct_shift_X_single.py [path to tif file] [initial shift] [start frame] [end frame]`` 

.. note::
  
  Notes As above 

  Usually, the sign of the shift changes in these frames. Since you usually should have run a shift on the TIF file before, now you need to shift it with twice the amount back. If the original shift was 3, now you need to apply a shift of -6!

  The numbers for the frames match exactly with the numbers shown in the
  data prep module, so the first frame is the first and not the zeroth.

**Possible problems with solutions**

* I get errors from 'imshow': Change ``PREVIEW_Z_STACK`` and ``PREVIEW_Z``
  (see Configs) in the script to values that make sense for you. Worst
  case try 0 for both.
* TIF files are not found: Change ``INCLUDE_PATTERN_TIF_SEARCH`` for
  additional TIF files and ``INCLUDE_PATTERN_TIF_BASESEARCH`` if ``_multi``
  does not find any paths accordingly.
* Wrong TIF files found: Refer to TIF files are not found.
* The shift is different depending on where on one picture I'm currently at: No fix, usually this effect is low enough to not cause problems with segmentation and tracking. *pew*

**Configs**

There are quite a few things you can change in scripts. To change them, change them in configs.json in the ``scripts`` folder. For the regex expression please find ``regex.txt``
  
*Same in all scripts*

``NEW_PATH_SUF``: Changes the suffix of the new files. Leaving it empty
causes old files to be overwritten, which is recommended, as otherwise
the data prep process will also align the old files.

``PREVIEW_Z_STACK``: Changes the frame which is used in the preview.

``PREVIEW_Z``: Changes the z-slice which is used in preview.

``INCLUDE_PATTERN_TIF_SEARCH``: Regex expression which is used to filter the .tif files if you choose to search for other TIF files. If you don't know regex, ask Chat_GPT to generate one for you by giving it examples of file names and then asking it to generate a regex code which excludes the files you want to exclude. In the code it is used in a ``re.match`` function which iterates over all TIF files in the folder.


*\_multi*

``PRESET_SHIFT``: Allows you to set a standard shift.

``INCLUDE_PATTERN_TIF_BASESEARCH``: Same as in ``INCLUDE_PATTERN_TIF_SEARCH``, but this regex expression is used to match TIF files which are used to determine the shift.

``FOLDER_FILTER``: Filter which is applied in order to make sure only folders which contain tif files (and not for example the original microscopy file) are considered. The first function ``finding_base_tif_files_path()`` is used to find the base TIF files. If your basic directory structure is different, change this function accordingly. ``root_path`` is simply the argument parsed from user input. ``base_tif_files_paths`` is the path directly to your main TIF file which should be used to determine the shift.

``tif_files_paths`` is the path to the folder in which they are contained.
