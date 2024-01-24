Folder structure
================

To load data into Cell-ACDC you need to structure the files in a specific way. 

In the section :ref:`creating-data-structure` we explain how to achieve this in 
an automatic fashion. 

Here, instead we will focus on how the folder structure should look like. 

.. important:: 
    
    We do not recommend creating the folder structure manually. Here we only 
    want to provide more details about how the folder structure look like. 
    To create the folder structure automatically from a microscopy file use 
    either module 0 of Cell-ACDC (explained here :ref:`creating-data-structure`) 
    or Fiji macros that you can find `here <https://github.com/SchmollerLab/Cell_ACDC/tree/main/FijiMacros>_`.

In Cell-ACDC we refer to the folders with two names:

1. The experiment folder with an arbitrary name
2. The position folder with names like ``Position_1``, ``Position_2``, etc. 

The experiment folder is a folder that typically identifies a specific experiment 
and it contains one or more position folders. 

The position folder is a folder inside the experiment folder that must contain 
a folder called ``Images``. In Cell-ACDC we refer to "positions" with the same 
meaning of the term "series" in ImageJ/Fiji. These are also called "Tiles" by 
some microscopy manufacturers. You can have as many position folders as you 
like in each experiment folder. 

The ``Images`` folder contains the image files. You will need to create one TIFF 
file per channel. Each TIFF file can be 2D, 3D (z-stack or 2D over time), or 
4D (z-stack over time). You can have as many channels as you want. 

The filenames of each file must all start with the same ``<basename>``, which is 
a text that is common to all the files in the folder. 

This is probably more clear with an example. Let's say that you have an experiment 
called ``mitochondria_medium_switch`` with 5 positions and each position has 
three channels: ``phase_contrast``, ``GFP``, and ``mCitrine``. With this 
structure you create a folder called ``mitochondria_medium_switch`` and 
inside this folder you create 5 folders, one for each position, called 
``Position_1``, ``Position_2``, ..., and ``Position_5``. Inside each position 
folder you create a folder called ``Images``. Inside each ``Images`` folder 
you create the TIFF files, one for each channel, all starting with the same 
basename. The basename is a text that allows you to identify what the experiment 
is about, for this example we will use the basename 
``ASY015_mitochondria_medium_switch_``. 

In this hypothetical example, the folder structure wood look like the following::

    mitochondria_medium_switch
    ├── Position_26
    │    └── Images
    │       ├── ASY015_mitochondria_medium_switch_metadata.csv
    │       ├── ASY015_mitochondria_medium_switch_GFP.tif
    │       ├── ASY015_mitochondria_medium_switch_mCitrine.tif
    │       └── ASY015_mitochondria_medium_switch_phase_contrast.tif
    │ 
    ├── Position_2
    │    └── Images
    │        ...
    │
    ├── ...
    │
    └── Position_5
        └── Images
            ...

You probably have noticed an additional file that ends with ``_metadata.csv``. 
If this file is missing it will be created by Cell-ACDC. It contains useful 
metadata like the pixel size and other metadata, but it also contains an entry 
called ``basename``. If you don't create this file Cell-ACDC will try to guess 
the ``basename`` from the filenames. While this usually works fine, it is better 
to create this ``_metadata.csv`` file with the following content::

    Description,values
    basename,ASY015_mitochondria_medium_switch_

This way Cell-ACDC will not try to guess the basename and you will avoid weird 
naming of additional files due to the wrong basename being guessed. 