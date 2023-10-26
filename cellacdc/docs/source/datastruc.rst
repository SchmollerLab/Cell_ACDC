Cell-ACDC output data
=====================

**Files** saved by Cell-ACDC for a **fully analysed experiment** (example with original raw microscopy file called ``Example1`` and first position ``_s01_``)

.. list-table::
    :header-rows: 1
    :class: tight-table

    * - File:
      - Description:
    * - .. code-block::

            Example1_s01_acdc_output.csv

      - Main table containing cell cycle annotations and additional metrics such as mean, median etc. for all the loaded channels plus all the region properties (as calculated by `skimage.measure.regionprops <https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops>`__) for each segmented object.
    * - .. code-block::

            Example1_s01_act1.tif

      - .tif file for the channel called ``act1``.
    * - .. code-block::

            Example1_s01_cdc10.tif

      - .tif file for the channel called ``cdc10``.
    * - .. code-block::

            Example1_s01_last_tracked_i.txt

      - **Last visited frame** in “Segmentation and Tracking mode” with the main GUI.
    * - .. code-block::

            Example1_s01_metadata.csv

      - Table containing the **metadata** such as number of frames, number of z-slices, pixel size etc.
    * - .. code-block::

            Example1_s01_phase_contr.tif

      - .tif file for the channel called ``phase_contr``.
    * - .. code-block::

            Example1_s01_segm.npz

      - **Segmentation labels**. This is a numpy array (compressed).
    * - .. code-block::

            Example1_s01_segmInfo.csv

      - Table containing information such as which z-slice or z-projection was used for segmentation or saving metrics of each channel.
    * - .. code-block::

            Example1_s01_align_shift.npy

      - Numpy array containing the **shifts** applied to each frame when **aligning**. Useful for reverting to non-aligned state.
    * - .. code-block::

            Example1_s01_dataPrepROIs_coords.csv

      - Table containing the **coordinates** of the **ROI** that was used to either **crop**, or **segment** only objects in the ROI. This is created during the data prep or segmentation stage.
    * - .. code-block::

    
            Example1_s01_phase_contr_aligned.npz

      - Aligned data for the channel called ``phase_contr``. This is a numpy array (compressed).
    * - .. code-block::

            Example1_s01_phase_contr
            _aligned_bkgrRoiData.npz

      - **Data** from the **background ROIs** generated at the data prep stage.
    * - .. code-block::

            Example1_s01_phase_contr
            _dataPrep_bkgrROIs.json

      - **Coordinates** of the **background ROIs** generated at the data prep stage.