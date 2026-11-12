from . import urls, html_utils

forum_href = html_utils.href_tag('forum page', urls.forum_url)

utilsInfo = {
    'Convert _segm.npz file(s) to ImageJ ROIs...': (f"""
        Not documented yet. You can ask help about utilities  on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Create connected 3D segmentation mask from z-slices segmentation...': (f"""
        This utility is used to create a 3D segmentation mask by projecting the 
        center z-slice of the 3D objects to their own z-boundaries.<br><br>
        
        For example, an object with three segmented z-slices will become 
        a "cylindrical" object,<br>
        where the largest z-slice is projected up and down to the max and 
        min z-slice.
    """),

    'Track sub-cellular objects (assign same ID as the cell they belong to)...': (f"""
        Not documented yet. You can ask help about utilities 
        on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Apply tracking info from tabular data...': (f"""
        This utility is used to <b>load the information of an external 
        tracker</b> into Cell-ACDC.<br><br>
        
        It creates a new (or overwrites existing) segmentation file where 
        the <b>IDs of the segmented objects are taken from a table</b> in 
        CSV format.<br><br>
        
        This table is a typical output of trackers. It <b>must contain a column 
        with the frame index</b> (starting from 0), a column with the<br>
        <b>tracked IDs</b> of the segmented objects</b>, plus either a column 
        with the corresponding IDs in the segmentation mask or the <br>
        two columns with the <code>(x, y)</code> coordinates of the objects' 
        centroids.<br><br>
        
        The <b>name of the columnns is not relevant</b>, you will be asked to 
        choose which column is what.<br><br>
        
        Note that to use this utility you <b>need to have a Cell-ACDC 
        compatible segmentation file</b>.
    """),

    'Create required data structure from image files...': (f"""
        Not documented yet. You can ask help about utilities 
        on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Re-apply data prep steps to selected channels...': (f"""
        Not documented yet. You can ask help about utilities 
        on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Concatenate acdc output tables from multiple Positions...': (f"""
        Not documented yet. You can ask help about utilities 
        on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Compute measurements for one or more experiments...': (f"""
        Not documented yet. You can ask help about utilities 
        on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Combine measurements from multiple segmentation files...': (f"""
        Not documented yet. You can ask help about utilities 
        on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Add lineage tree table to one or more experiments...': (f"""
        Not documented yet. You can ask help about utilities
        on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>!
    """),

    'Measurements viewer...': ("""
        Open the <b>measurements viewer</b>, a window to plot per-cell
        time courses of the measurements in the <code>acdc_output</code>
        tables next to the images, or measurements vs measurements in scatter
        plots.<br><br>

        It can also compare and combine measurements, apply measurement
        processing steps, identify time points, translate them into
        phases and annotate cells. Configurations can be saved and
        loaded again.
    """),

    'Edit metadata of one or more experiments...': (f"""
        This utility is used to <b>edit the <code>metadata.csv</code> file</b>
        of multiple Positions and experiments at once.<br><br>

        After selecting the experiment folders (and Positions), Cell-ACDC
        will scan the metadata files and <b>warn you if the metadata is not
        the same</b> in all of them.<br><br>

        You can then edit the number of frames (SizeT), number of z-slices
        (SizeZ), time interval, physical sizes, and custom fields.
        The <b>same values will be applied to all</b> the selected
        Positions.<br><br>

        Note that measurements already saved in the
        <code>acdc_output.csv</code> tables are not updated.
        Run the "Compute measurements" utility to update them.
    """),

    'Compute background intensity outside of segmented objects...': (f"""
        This utility computes, for every frame, the <b>background
        intensity</b> of the selected channels as the mean and median of the
        pixels <b>outside of all segmented objects</b> (label 0).<br><br>

        Select a segmentation file where (nearly) <b>all cells are
        segmented</b>, otherwise unsegmented cells raise the background.
        Frames without segmented objects get no background value. Pixels with
        intensity 0 (e.g., padding added by the alignment) can be
        ignored.<br><br>

        The background values of each Position are saved in the Images
        folder as <code>&lt;basename&gt;background_values.csv</code> (the
        name can be changed, but not to the name of another existing
        file).<br><br>

        Optionally, the utility computes <b>normalization constants</b> from
        the mean intensity of the cells in a selected <code>acdc_output</code>
        table (<code>&lt;channel&gt;_mean</code>; for z-stacks
        <code>&lt;channel&gt;_mean_meanProj</code> or, with 3D masks,
        <code>&lt;channel&gt;_mean_3D</code>; the table is not modified): a
        constant background (median of the per-frame backgrounds) and the low
        and high percentiles (default 5th and 95th) of the mean intensities
        minus that background, pooled over all cells, frames and Positions
        (excluded cells are ignored). Select only the
        experiments of the reference condition to compute the constants from
        them. The constants are the min/max values of the
        "Min-max normalization" measurement processing (e.g., in the premade
        FUCCI configuration of the Measurements viewer) and are saved in each
        experiment folder as <code>background_values_summary.csv</code>.
        <br><br>
        When setting up the premade FUCCI configuration of the Measurements
        viewer, this utility can be run (or its summary table loaded) from
        the setup dialog to fill in the normalization values.
    """)
}