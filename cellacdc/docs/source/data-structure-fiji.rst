.. _Fiji Downloads: https://imagej.net/software/fiji/#downloads
.. _Cell-ACDC Fiji macros: https://github.com/SchmollerLab/Cell_ACDC/tree/main/FijiMacros
.. _GitHub page: https://github.com/SchmollerLab/Cell_ACDC/issues

.. _data-structure-fiji:

Create Data Structure with ImageJ/Fiji macros
=============================================

Follow this steps to create the required data structure from the raw microscopy 
files using the provided Fiji macros:

1. If you don't have it, **install ImageJ/Fiji** from here `Fiji Downloads`_.
2. **Download the Fiji macros** from here `Cell-ACDC Fiji macros`_.
3. **Open the macro** with Fiji:
   
    If you have one or more microscopy files one for each position use the macro 
    called ``multiple_files.ijm``, while if you have multiple positions within 
    a single file use the ``single_file.ijm`` macro.

4. **Modify the list of channels:**
   
    In the first lines of the macro you will find the variable ``channels`` 
    defined as follows:

    .. code-block:: javascript

        channels = newArray("phase_contr", "mCitrine");
    
    Modify that list with the correct order of channels as they are found in 
    your microscopy file(s). 

5. **Run the macro** with the ``Run --> Run`` menu
6. If you are using ``single_file.ijm`` **select the microscopy file**, if not 
   **select the folder containing the microscopy file(s)**. 

If successful, at the end of the process you should have all the Position folders 
correctly generated and ready to be opened in Cell-ACDC. 

If you are having issues, feel free to open an issue on our `GitHub page`_. 

Until next time! 
