Measurements viewer
===================

The **measurements viewer** shows your segmented and tracked time-lapse data
next to plots of the measurements of the ``acdc_output`` table. For the
selected cell, you can plot how its measurements change over time (e.g., cell
area or the mean intensity of a reporter), mark important time points (e.g.,
bud emergence or division), annotate cells, process and combine measurements,
and translate the time points into cell cycle phases (or any other phases).
You can also plot two measurements against each other for all cells of a
Position.

.. tip::

    You can open this page from the viewer: use the **Help** menu or the
    info buttons of the dialogs. Hover over any button of the viewer to
    read a short description of what it does.

.. contents:: On this page
    :local:
    :depth: 2


Opening the viewer and loading data
-----------------------------------

What you need
~~~~~~~~~~~~~

The viewer works with data in the Cell-ACDC folder structure (see
:doc:`data-structure`): each ``Position_<n>/Images`` folder must contain the
image files, a segmentation file (``segm.npz``) and, ideally, the
``acdc_output`` table with the measurements. The measurements are computed
when you save in the main GUI, or with the launcher utility
**Utilities > Measurements > Compute measurements for one or more
experiments...**. Tracking and cell cycle annotations are optional, but they
are needed for lineage-based inputs (mother, daughter, sister cell).

Starting the viewer
~~~~~~~~~~~~~~~~~~~

* From the launcher: **Utilities > Measurements > Measurements viewer...**
* From a terminal, in the environment where Cell-ACDC is installed::

      python -m cellacdc.measurements_viewer

  Add the path of a folder at the end of the command to load it directly.

When the viewer starts, a dialog lists your **saved** configurations and the
**premade** ones (for example ``[Premade] fucci_cell_cycle.json``).
A configuration stores everything you set up in the viewer (see
`Configurations`_). Select one and click **Ok**, browse to a configuration
file elsewhere with **Browse configuration...**, or click
**Skip/New configuration** to start from scratch.

Loading a folder
~~~~~~~~~~~~~~~~

1. Click **File > Load Folder...** (:kbd:`Ctrl+O`) and select an experiment
   folder, a ``Position_<n>`` folder or an ``Images`` folder. All Positions
   found in the folder are loaded.
2. Select the channel to display (skipped if there is only one).
3. Select the segmentation file to load (skipped if there is only one).

Alternatively, use **File > Open Image File...** to open one image file of an
``Images`` folder, or **File > Open Recent**. If the loaded data have unsaved
changes, you are first asked whether to save them (see `Configurations`_).

Loading large experiments can take a while. The window stays visible but does
not accept input until loading is complete; progress is shown in the status
bar and in the terminal.

Partially analyzed videos
~~~~~~~~~~~~~~~~~~~~~~~~~

The viewer checks how far each Position was analyzed in the main GUI, like
the main GUI's "Last tracked frame" information: the **last cell cycle
annotated frame** if the table has cell cycle annotations, otherwise the
**last tracked frame** (the last frame of the ``acdc_output`` table).

If a Position was not analyzed until the end of the video, you are asked
whether to **ignore the later frames** (default, recommended) or to **use all
frames**. When the later frames are ignored, all calculations of that
Position use only the analyzed frames:

* time point and custom annotation predictors (and the tests in their
  editors): later frames get no time points, and annotations are decided
  from the analyzed frames only;
* phase evaluation: later frames get no phase;
* measurement processing and combined measurements: later frames get empty
  values;
* conditional formatting.

In equations, ``SizeT`` is then the number of analyzed frames (e.g.,
``LAST(frame_i) == SizeT - 1`` means "present at the last analyzed frame").
Images, navigation and plots still show all frames: the ignored frames are
shaded gray in the time course and phase band, and the label next to the
frames slider shows the last analyzed frame.

Change the choice with **Annotations > Ignore frames after the last analyzed
frame**. It is saved in the configuration, so loading that configuration does
not ask again. Results calculated before changing it are not recalculated
automatically (except automatically evaluated phases): re-evaluate the
predictors and phases, or load the configuration again to recalculate the
measurement steps. After annotating more frames in the main GUI, click
**Reload table** (:kbd:`F5`): new calculations (and automatically evaluated
phases) then use the new last analyzed frame. The outputs of the measurement
steps stay empty after the frame they were calculated up to (the
configuration remembers it) until you load the configuration again and
choose **Calculate**.

The window
~~~~~~~~~~

* **Toolbars** (top): files and configurations, viewer tools (search, zoom,
  overlay of other channels, ruler, etc.), **Time points**,
  **Custom annotations** and **Phases**, with the button of the
  `Time point to phase translator`_ (icon of a colored band split by two
  time point markers; hover over it for a description).
* **Image** (center): the current frame with the segmentation contours and
  cell IDs. The selected cell is highlighted.
* **Navigate** box: Position, frame, z-slice and Cell ID controls (labels,
  value boxes and sliders in aligned columns).
* **Annotations** box: what to display on the image (contours, IDs, cell
  cycle info, etc.).
* **Measurements** box: the plot (left) and the list of columns to plot with
  the buttons to process, combine and export measurements (right).
* **Cell-ACDC objects** dock: click the small arrow button at the left edge
  of the window (**Show object properties**) to show the properties of the
  selected cell (ID, area, intensities, etc.).


Selecting cells and navigating
------------------------------

The time-course plot always shows the **selected cell**. You can select a cell
in several ways:

* **Cell ID spinbox and scrollbar** in the **Navigate** box (bottom). They
  blink briefly after loading to show you where they are.
* **Click on a cell** in the image.
* Press :kbd:`Up` / :kbd:`Down` to go to the next / previous Cell ID.
* Press :kbd:`Ctrl+F` (or click the magnifier button in the toolbar). The
  search bar at the top right of the window blinks; type the Cell ID and
  press :kbd:`Enter`.
* Change the ID in the **Cell-ACDC objects** dock. The dock and the viewer are
  synchronized: selecting a cell in the viewer shows its properties in the
  dock, and changing the ID in the dock selects that cell in the viewer.
* In the plot, click a point to go to its frame and zoom to its cell (useful
  in **Measurement vs Measurement** mode).

To move in time, use the frame scrollbar, the :kbd:`Left` / :kbd:`Right`
arrow keys, or drag the **red vertical line** (current frame) in the plot.
With z-stacks, the z-slice scrollbar selects the slice or the projection.
Use the **Position** drop-down menu to switch between Positions.

Options at the bottom of the **Navigate** box (below the Cell ID controls and
the overlay transparency sliders):

* **Follow cell**: keep the view zoomed on the selected cell when changing
  frame or z-slice.
* **Only frames with selected cell**: restrict the frame scrollbar to the
  frames where the selected cell exists.
* **Rescale plot**: rescale the plot to the data whenever you change cell.
* **Normalize measurements (0-1)**: display every plotted measurement scaled
  to the range 0-1, to compare measurements with different units. This only
  changes the display, not the table.
* **Show phases**: show the phase band below the plot (see
  `Time point to phase translator`_).

**Follow cell**, **Only frames with selected cell**, **Rescale plot** and
**Show phases** are remembered the next time you open the viewer.
**Normalize measurements (0-1)** is saved in the configuration (see
`Configurations`_).


Plotting measurements
---------------------

Time course
~~~~~~~~~~~

With **Time course** selected at the top of the column list, click a column
in **Columns to plot** to plot it for the selected cell; click it again to
remove it. Type in **Search column...** to find a column quickly. The columns
are organized in groups (e.g., intensity measurements of each channel, size
measurements, region properties); only numerical and yes/no columns can be
plotted.

* **Colors**: each plotted column gets a colorblind-friendly default color
  (Okabe-Ito palette) and keeps it when you plot or remove other columns.
  No default color is red: red is reserved for the current frame line (the
  cursor). Time point lines use the color chosen for each time point. With
  the light background (**Image > Invert black/white**), black replaces the
  Okabe-Ito yellow. To choose another color, **right-click** the column and select
  **Change color...** (**Reset to default color** restores the default).
  Custom colors are saved with the configuration (the premade FUCCI
  configuration plots the Geminin reporter in green and the Cdt1 reporter
  in red).
* **Zoom**: when you add a column, the y axis is fitted to the plotted data
  and the current x zoom is kept. Scroll or drag in the plot to zoom and pan.
  With the mouse over the plot, press :kbd:`H` to fit the plot to the data
  and press :kbd:`H` twice quickly to show all frames. After the fit, the y
  axis keeps following the data within the visible frames (e.g. when you
  change cell or edit time points) until you zoom or pan by hand; the x zoom
  is kept unless **Rescale plot** is checked.
* **Hover** over a point to see its frame, value and phase; **click** a point
  to go to that frame.

Measurement vs Measurement
~~~~~~~~~~~~~~~~~~~~~~~~~~

Select **Measurement vs Measurement** at the top of the column list and click
two columns: the first one is the X axis and the second one the Y axis (the
column list shows X and Y markers instead of colors; uncheck a column to
choose another one). The scatter plot shows
**all cells and all frames** of the current Position, so it does not change
when you select another cell. Hover for details and click a point to go to its
cell and frame. When phases are evaluated, each point is colored by its
phase (gray for frames without a phase). The selections of the two modes are
independent.

Conditional formatting
~~~~~~~~~~~~~~~~~~~~~~

Right-click a plotted column and select **Conditional formatting...** to color
parts of its time course. Add ordered rules; each rule has a condition and a
color. A condition is an equation of the table columns evaluated on the
selected cell, for example::

    cell_area_pxl > 1.5 * FIRST(cell_area_pxl)

The color is either a fixed color or a lighter/darker shade of the column
color. The first rule that is true colors the point and the line to the next
point; other frames keep the column color. Hover over a point to see which
condition matched. Conditional formatting applies only to the time course and
is saved in the configuration. **Clear conditional formatting** removes it.


Processing, combining and exporting measurements
------------------------------------------------

The buttons below the column list add new columns to the table of the
**current Position**. The original columns are never changed, and new
columns can be plotted and used by time point and annotation predictors like
any other column.

Process measurements
~~~~~~~~~~~~~~~~~~~~

**Process measurements...** applies a recipe of processing steps to one or
more measurements:

* **Smooth measurement**: centered moving average, computed separately for
  each cell.
* **Min-max normalization**: rescales the values with fixed minimum and
  maximum constants that are the same for all cells (so that cells remain
  comparable).
* **Ramp detection**: marks sustained rising (+1) and falling (-1) ramps of
  each cell's measurement (used, e.g., by the FUCCI premade configuration).

In the **Measurement-processing recipe** dialog, select the measurements,
add the steps and set their parameters (click the info buttons for an
explanation of each parameter), and enter the text appended to the output
column names. For each input, a new column ``<measurement>_<name>`` is added.
Processing runs in the background, so you can keep working (e.g., adding time
points) meanwhile. The recipe is recorded in the configuration and can be
recalculated on other data when the configuration is loaded.

Combine measurements
~~~~~~~~~~~~~~~~~~~~

**Combine measurements...** opens an equation editor to calculate new columns
from existing ones, for example the ratio of two reporters::

    GFP_mean / mCherry_mean

Equations can also use the time points as yes/no inputs
(``timepoint_<name>``), and ``any(condition)`` / ``all(condition)`` evaluate
a condition over all frames of each cell. The equations are recorded in the
configuration.

Export and save
~~~~~~~~~~~~~~~

* **Export measurements...** saves the table of the current Position to a
  separate CSV file, including processed and combined columns, time points,
  custom annotations, phases and the ``measurements_viewer_visited`` column
  (which cells you have viewed). If phase rules (or annotations that override
  phases) are configured, you are asked whether to evaluate the phases of all
  cells first.
* When closing the viewer or loading other data, you are asked whether to
  **save the tables**, which overwrites each Position's ``acdc_output`` file.
  The previous file is backed up in the Position's ``recovery`` folder.
* **Reload table** (or **File > Reload acdc_output Table**, :kbd:`F5`)
  reloads the table from the file, e.g. after saving in the main GUI. Your
  time points, custom annotations, visited cells and evaluated phases are
  kept, but processing results that were not saved are discarded.


Time point identifiers
----------------------

A **time point identifier** is a named event that happens at one (or more)
frames of a cell, such as "Bud emergence", "Division" or "Onset of reporter
expression". The time points are saved in the table as yes/no columns named
``timepoint_<name>`` (yes at the frames where the cell is marked).

Adding an identifier
~~~~~~~~~~~~~~~~~~~~

Click **Add time point identifier...** in the **Time points** toolbar and set:

* **Name**, for example ``Bud emergence``.
* **Color** of the vertical time point lines in the plot.
* **Shortcut**: the key used to mark the time point (defaults Q, W, E, R,
  ...). A key cannot be used if it, or :kbd:`Shift` + the key, is already
  taken by another action.
* **Predictor** (optional, see below) and **Auto apply on unvisited cells**.

Each identifier gets its own button in the toolbar.

Marking time points manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Hover** with the mouse over the time-course plot at the right frame and
  press the identifier's shortcut. The time point is added for the selected
  cell at the frame under the mouse cursor. Clicking the identifier button
  adds it at the current frame.
* **Drag** a time point line to move it.
* **Right-click** a time point line to remove it.
* **Right-click** an identifier button to edit it, re-evaluate its predictor
  (on all cells or on the displayed cell), clear its time points or remove it.

Predicting time points automatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A predictor finds the time point of each cell for you. Choose
**Equation** and click **Edit equation...** to write a condition on the table
columns, then choose under **Time point(s) of each cell** how the per-frame
results become time points:

* First frame where the equation is true
* Last frame where the equation is true
* Every frame where the equation becomes true
* Frame of the maximum value
* Frame of the minimum value

For example, the first frame where the cell is 1.5 times larger than at its
first frame::

    cell_area_pxl > 1.5 * FIRST(cell_area_pxl)

**Per-cell functions** compute one value over all frames of the cell:
``MIN``, ``MAX``, ``MEAN``, ``MEDIAN``, ``SUM``, ``STD``, ``FIRST``,
``LAST``, ``COUNT``, ``NUNIQUE``, and ``DIFF`` (change from the previous
frame) and ``SHIFT(x, n)`` (value ``n`` frames earlier) per frame.
Optional filters select which cells are predicted (**Main-cell filter**), and
**Other-cell measurements** can use the measurements of other cells, e.g. of
the mother or sister cell. Use **Test predictor** to check the result.

Advanced users can select **Select own function (Python file)**: the file
must define ``predictor(df)``, which receives the table of one cell and
returns a ``frame_i`` (0-based, i.e. the displayed frame number minus 1), a
list of them, or ``None``. **Save template...** writes an example file.

How and when predictors run:

* With **Auto apply on unvisited cells**, a cell is predicted the first time
  you display it. Cells that you have already viewed (also in an earlier
  session, as recorded in the ``measurements_viewer_visited`` column) are not
  predicted again, so your corrections are never overwritten automatically.
* :kbd:`Shift` + the identifier's shortcut re-evaluates the predictor on the
  selected cell.
* **Re-evaluate predictors on all cells** and **Re-evaluate predictors on
  displayed cell** (toolbar) re-run all predictors. If cells have manual
  edits, you are asked whether to overwrite or skip them.
* Predictors can use other identifiers (``timepoint_<name>``) and custom
  annotations (``<name>``) as inputs; these are evaluated first. Circular
  dependencies are refused.

An identifier cannot be renamed while other predictors, phase rules,
formatting conditions or recorded equations use its column (edit those
first). Removing it warns you about such dependents.


Custom annotations
------------------

A **custom annotation** is a yes/no label of a whole cell, for example
"Excluded", "Arrested" or "Dead". It is saved in the table as a 0/1 column
named like the annotation. Custom annotations are shared with the main GUI.

* Click **Add custom annotation** in the **Custom annotations** toolbar to
  define the name, symbol, color, shortcut (defaults A, D, F, ...) and an
  optional predictor. Other buttons load annotations used before (also those
  defined in the main GUI) and show all annotations.
* Select a cell and press the annotation's shortcut to annotate or
  un-annotate it **across its entire history**. Alternatively, activate the
  annotation button in the toolbar and **right-click** on a cell in the image.
* A predictor works like a time point predictor, but it returns yes/no
  values (one per frame, or one for the whole cell). These are combined into
  one value per cell with **All** (default: yes only if yes in every frame),
  **None** (yes only if no frame is yes) or **Most** (yes if more than half of
  the frames are yes; ties count as no).
* **Auto apply**, :kbd:`Shift` + shortcut and the toolbar buttons
  **Re-evaluate annotation predictors on all cells / on displayed cell** work
  as for time point identifiers. Right-click an annotation button to edit or
  re-evaluate it.
* **Overrides phases** (under **Additional options**, measurements viewer
  only): every cell with this annotation gets the annotation's name as its
  phase on all frames (e.g., ``Excluded``), instead of the phases of the
  `Time point to phase translator`_ rules. Use it for cells that do not go
  through the normal phases, e.g. excluded or arrested cells. If a cell has
  several such annotations, the one listed first in the
  **Custom annotations** toolbar wins (see `How it works`_).

Annotation definitions (including **Overrides phases**) are saved in the
configuration; the annotations of the cells are saved in the table (save the
tables or export the measurements).


Time point to phase translator
------------------------------

The **Time point to phase translator** turns the time points of each cell
into **phases**, i.e. a name for every frame, such as the cell cycle phase.
Open it with the phase translator button of the **Phases** toolbar (icon of
a colored band split by two time point markers) or with
**Annotations > Time point to phase translator...**.

How it works
~~~~~~~~~~~~

* The translator has an ordered list of **rules**. For each cell, the rules
  are tried from top to bottom; the **first rule that matches** gives the
  phases of the whole cell. A cell that matches no rule gets no phases.
* Each rule has **boundaries**: time point identifiers, in a fixed order.
  The boundaries split the cell's frames into intervals, and every interval
  gets a phase name: one name **before** the first boundary, one **between**
  each pair of boundaries and one **after** the last boundary (N boundaries,
  N + 1 names).
* An interval **includes** the frame of its starting time point and
  **excludes** the frame of its ending time point.
* A rule **matches** a cell only if:

  1. every boundary occurs **exactly once** in the cell, in the configured
     order;
  2. every **required custom annotation** is present in the cell; and
  3. the rule's **eligibility formula** (optional) is true for the cell.

* Leave a phase name blank to leave that interval without a phase. A blank
  interval is not filled by lower rules: the first matching rule owns the
  whole cell.
* A rule without boundaries has a single phase for the whole cell, which is
  useful as a fallback at the bottom of the list or, together with a
  required annotation, at the top.
* Custom annotations with the **Overrides phases** option (see
  `Custom annotations`_) take precedence over **all** rules: a cell with such
  an annotation gets the annotation's name as its phase on every frame, and
  no rule is checked for it. If a cell has several of these annotations, the
  one listed first in the **Custom annotations** toolbar wins. The rules give
  the phases of all other cells. Phases can also be evaluated with these
  annotations alone, without any rule.

Worked example: cell cycle phases of budding yeast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal: label every frame of each cell as **G1** or **S-G2-M**, using two time
points: **Bud emergence** (start of S-G2-M) and **Division** (start of the
next G1). Cells annotated as "Excluded" should get the phase "excluded".

**Step 1: create the time point identifiers.** In the **Time points**
toolbar, click **Add time point identifier...** twice and create
``Bud emergence`` (shortcut Q) and ``Division`` (shortcut W). Mark the time
points of a few cells (hover over the plot and press Q or W), or set up
predictors to find them automatically.

**Step 2 (optional): create the required annotation.** Click
**Add custom annotation** and create ``Excluded``. Annotate the cells that
should be excluded.

**Step 3: open the translator** with the phase translator button of the
**Phases** toolbar (or **Annotations > Time point to phase translator...**)
and click **Add...** to create the first rule.

**Step 4: rule "Excluded".** In the rule editor:

* **Rule name**: ``Excluded``.
* **Rule eligibility formula**: keep **Always eligible (no formula)**.
* Under **Custom annotations (requirements only)**, check ``Excluded``.
  Leave all time points unchecked.
* On the right, enter ``excluded`` in the only phase field
  (**Phase for entire cell**) and click **OK**.

**Step 5: rule "Full cycle".** Click **Add...** again:

* **Rule name**: ``Full cycle``.
* Under **Time points (interval boundaries)**, check ``Bud emergence`` and
  then ``Division``. The right-hand list now shows the boundaries in this
  order, with a phase field before, between and after them. Use the **Up** /
  **Down** buttons next to a boundary to change the order.
* Enter the phase names:

  * **Phase before first point**: ``G1``
  * **Phase between points** (between Bud emergence and Division):
    ``S-G2-M``
  * **Phase after last point**: ``G1``

* Click **OK**.

**Step 6: rule "Bud emergence only"** for cells that do not divide before
the end of the movie: check only ``Bud emergence`` and enter ``G1`` before
and ``S-G2-M`` after it.

**Step 7: check the rule order.** The list should read:

1. Excluded
2. Full cycle
3. Bud emergence only

Select a rule and use **Up** / **Down** to reorder it. Rules higher in the
list win, so put the most specific rules first (here, a cell annotated as
"Excluded" always gets "excluded", even if it has time points).

.. tip::

    Instead of the rule "Excluded", you can turn on **Overrides phases** for
    the ``Excluded`` annotation (right-click its toolbar button to edit it)
    and keep only the rules "Full cycle" and "Bud emergence only". The
    excluded cells then get the phase ``Excluded`` (the annotation's name) on
    every frame.

**Step 8: apply.** Click **Apply phase evaluation on displayed cell** to check
the result on the cell you are looking at, then
**Apply phase evaluation on all cells** for all cells of the current Position.
Click **OK** to close the dialog and keep the rules.

Afterwards, you do not need to open the dialog again to update a cell: click
**Evaluate phases on displayed cell** next to the translator button in the
**Phases** toolbar (or **Annotations > Evaluate phases on displayed cell**),
or press :kbd:`Alt+P`. It applies the saved rules (and the annotations that
override phases) to the displayed cell only, e.g. after correcting its time
points.

**Result.** A cell present from frame 1 to 40 with Bud emergence at frame 10
and Division at frame 25 gets:

* frames 1-9: ``G1`` (before Bud emergence);
* frames 10-24: ``S-G2-M`` (Bud emergence is included, Division is not);
* frames 25-40: ``G1`` (from Division onwards).

A cell with Bud emergence at frame 30 but no Division matches
"Bud emergence only": G1 until frame 29, S-G2-M from frame 30.

.. note::

    A rule only checks its own boundaries, and each boundary must occur
    exactly once. A cell with two Bud emergence time points (e.g., a mother
    cell tracked through two cycles) matches neither "Full cycle" nor "Bud
    emergence only" and gets no phases. Cells that match no rule because of
    repeated time points are listed in a warning.

Eligibility formula
~~~~~~~~~~~~~~~~~~~

The **Rule eligibility formula** box restricts a rule to some cells. It uses
the same editor as custom annotation predictors (equations, filters,
other-cell inputs), and the results are combined over the cell's frames with
**All**, **None** or **Most**. For example, the equation
``cell_area_pxl > 200`` with **All** makes the rule apply only to cells that
are always larger than 200 pixels. Keep **Always eligible (no formula)** to
skip this check.

Automatic evaluation
~~~~~~~~~~~~~~~~~~~~

Check **Automatically evaluate phases when time points or annotations change
or phases are missing** to keep the phases up to date: whenever you add,
move, remove, clear or predict time points, or add, remove, predict or clear
custom annotations, the phases of the affected cells are evaluated again, and
a displayed cell without phases is evaluated once. Turning **Overrides
phases** of an annotation on or off, and renaming or removing an annotation
that overrides phases, also evaluates the cells with this annotation again
(in all loaded Positions; Positions that are not displayed are updated when
you display them or save the tables). Messages then appear in the status
bar. This option is saved in the configuration.

If there are no phase rules and you turn off or remove the last annotation
that overrides phases, its name is removed from the phases of its cells
(also without automatic evaluation), because nothing else could assign them
a phase.

Phase colors and phase band
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* When the table has phases, the phases of the selected cell are shown as a
  colored **phase band** directly below the time-course plot (one bar per
  phase, labeled with its name; hover for the frame range). In
  **Measurement vs Measurement** mode, the points are colored by phase.
  Toggle both with **Show phases**.
* The **Phase colors** box of the translator lists every phase name: first
  the phases of the rules, then the names of the annotations that override
  phases (so adding such an annotation does not change the default colors of
  the rule phases). Click a color to change it; the change is applied with
  **Apply** or **OK** and saved in the configuration. Phases without a chosen
  color get default colors from the colorblind-friendly Okabe-Ito palette (no
  red, which is reserved for the current frame line drawn over the band).

Output columns
~~~~~~~~~~~~~~

Evaluating the phases adds these columns to the table of the current
Position:

* ``phase``: the phase name of each frame (empty if no phase).
* ``phase_<name>``: one yes/no column per phase name, e.g. ``phase_G1`` and
  ``phase_S-G2-M`` (a name used in several intervals or rules shares one
  column).
* ``phase_<annotation name>``: one yes/no column per annotation that
  overrides phases, e.g. ``phase_Excluded``, even if no cell has this
  annotation.

They are saved with **Export measurements...** (which offers to evaluate all
cells first) or when you save the tables. The rules and the
**Overrides phases** option of the annotations (not the evaluated values) are
saved in the configuration.


Configurations
--------------

A configuration stores your setup, so you can reuse it on other experiments:

* the inputs (Images folders, displayed and overlay channels, segmentation
  file);
* the plotted columns of both plot modes, **Normalize measurements (0-1)**,
  conditional formatting and custom column colors;
* measurement processing recipes and combine equations;
* time point identifiers, custom annotations (including their
  **Overrides phases** option) and their predictors;
* phase rules, phase colors and automatic phase evaluation;
* whether frames after the last analyzed frame are ignored, and up to which
  frame the measurement steps were calculated in each Position (see
  `Partially analyzed videos`_).

The time points, annotations and phases of the cells are **not** stored in the
configuration, but in the tables.

* **Save configuration** (:kbd:`Ctrl+S`), **Save configuration as...**
  (:kbd:`Ctrl+Shift+S`) and **Load configuration...** are in the **File** menu
  and toolbar. Configurations are JSON files saved in the
  ``measurements_viewer_configurations`` folder of the Cell-ACDC settings
  folder (launcher: **Settings > Open settings folder...**).
* When loading a configuration, select one of its saved inputs or a new
  experiment. For a new experiment, the configured channels and segmentation
  are reused when every Position has them; otherwise you are asked for
  replacements. You are then asked whether to **calculate the measurement
  steps** (processing recipes and combine equations) again on the loaded
  data, or to use the existing values.
* Before loading data or a configuration, and when closing, the
  **Save changes?** dialog lets you save the tables, the configuration, both,
  or neither (**Don't save**). **Cancel** keeps the loaded data. If an
  ``acdc_output`` file was changed on disk in the meantime (e.g., saved in
  the main GUI), you are asked whether to reload it before saving, to
  overwrite it, or to cancel.

Premade configurations
~~~~~~~~~~~~~~~~~~~~~~

Premade configurations are ready-made templates shipped with Cell-ACDC, for
example:

* ``fucci_cell_cycle.json``: cell cycle phases (G1, S Early, S, S Late, G2,
  M) from the two FUCCI reporters (Cdt1 and Geminin), with background
  correction, min-max normalization, smoothing, ramp detection, time point
  predictors and phase rules. Its "Excluded" and "Arrested" annotations
  override phases: annotated cells get the phase "Excluded" or "Arrested" on
  every frame ("Excluded" wins if a cell has both).
* ``example_two_channel_ratio.json``: a simple example that plots two
  channels and their ratio (a combined measurement).

When you load a premade configuration (listed as ``[Premade]``), select the
experiment folder; the **Set up premade configuration** dialog then asks:

1. which of your channels corresponds to each placeholder (e.g., the Geminin
   and the Cdt1 channel; likely channels are preselected based on their
   names);
2. the values of the inputs (e.g., background intensities and normalization
   constants). Click the info buttons for an explanation of each value. For
   FUCCI, **Calculate background values...** runs the launcher utility
   **Utilities > Measurements > Compute background intensity outside of
   segmented objects...** on the selected experiment and fills the values, while
   **Load calculated background values...** reads a summary table saved
   before (e.g., of another experiment);
3. which channel to display as base image (for FUCCI, typically the phase
   contrast channel, with the reporters as overlays).

The result is saved as a new configuration in the settings folder and loaded.
The template itself is never changed.


Keyboard shortcuts
------------------

Hover over a button to see its shortcut. Use
**Edit > Customize keyboard shortcuts...** (:kbd:`Ctrl+K`) to change the
shortcuts of the viewer tools.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Shortcut
      - Action
    * - :kbd:`Ctrl+O`
      - Load folder
    * - :kbd:`F5`
      - Reload the ``acdc_output`` table from the file
    * - :kbd:`Ctrl+S` / :kbd:`Ctrl+Shift+S`
      - Save configuration / save configuration as
    * - :kbd:`Ctrl+F`
      - Search for a Cell ID or a functionality (the search bar blinks)
    * - :kbd:`Up` / :kbd:`Down`
      - Next / previous Cell ID
    * - :kbd:`Left` / :kbd:`Right`
      - Previous / next frame (or z-slice, when z-slice navigation is
        active)
    * - :kbd:`Z`
      - Toggle between frame and z-slice navigation (z-stacks)
    * - :kbd:`H` (mouse over the image)
      - Zoom to the cells; press twice to zoom out
    * - :kbd:`H` (mouse over the plot)
      - Fit the plot to the data; press twice to show all frames
    * - Time point shortcut (defaults Q, W, E, R, ...)
      - With the mouse over the plot: add a time point at the hovered frame
    * - :kbd:`Shift` + time point shortcut
      - Re-evaluate the identifier's predictor on the selected cell
    * - Annotation shortcut (defaults A, D, F, ...)
      - Annotate / un-annotate the selected cell
    * - :kbd:`Shift` + annotation shortcut
      - Re-evaluate the annotation's predictor on the selected cell
    * - :kbd:`Shift+Z`
      - Zoom to a rectangular area
    * - :kbd:`Ctrl+W`
      - Open a window to browse frames or channels
    * - :kbd:`Ctrl+Shift+C`
      - Count objects in the current view
    * - :kbd:`Ctrl+P`
      - View cell cycle annotations
    * - :kbd:`Alt+X`
      - Toggle the maximum z-projection
    * - :kbd:`Alt+P`
      - Evaluate the phases of the displayed cell with the phase translator
        (rules and annotations that override phases)
    * - :kbd:`Shift+S` / :kbd:`Alt+Shift+S`
      - Shuffle the colormap of the segmentation masks
    * - :kbd:`Esc`
      - Cancel the active tool

Because :kbd:`Shift` + a time point or annotation shortcut re-evaluates the
predictor, a key cannot be used as such a shortcut when :kbd:`Shift` + the key
is already taken (e.g., :kbd:`S`, because :kbd:`Shift+S` shuffles the
colormap).


Troubleshooting
---------------

**The window does not react while loading.**
    Loading large experiments takes time. The window does not accept input
    until loading is complete; follow the progress in the status bar (bottom
    of the window) and in the terminal. Do not close the window, even if the
    operating system reports it as "Not Responding".

**Where can I see what the viewer did?**
    The viewer prints what you load and change (e.g., plotting or removing a
    column, changing a color) and any error to the terminal from which
    Cell-ACDC was started. When the viewer is opened from the launcher, these
    messages are also written to the Cell-ACDC log files (launcher:
    **Help > Show log files...**). Please include them when you report an
    issue on `GitHub <https://github.com/SchmollerLab/Cell_ACDC/issues>`__ or
    on the `image.sc forum <https://forum.image.sc/tag/Cell-ACDC>`__.

**Pressing a time point shortcut does nothing.**
    Keep the mouse over the time-course plot while pressing the key: the
    time point is added at the frame under the mouse cursor. If a shortcut is
    already used by another action, it is left unset and a warning is shown;
    choose another key in the identifier's **Edit...** dialog.

**A predictor did not run on a cell.**
    Automatic predictions run only on cells that you have never viewed, so
    that your corrections are kept. Use :kbd:`Shift` + shortcut, the
    identifier's right-click menu or the toolbar re-evaluation buttons to
    predict cells again. Failed predictions are not retried automatically.

**A cell has no phases.**
    Check that each boundary of the expected rule occurs exactly once and in
    the configured order, that required annotations are present and that the
    eligibility formula is true for the cell. Apply the evaluation again
    after correcting the time points (or enable automatic evaluation).

**A cell has the name of an annotation as phase on every frame.**
    The cell has a custom annotation with the **Overrides phases** option,
    which takes precedence over the phase rules (see `How it works`_).
    Remove the annotation from the cell, or turn the option off in the
    annotation's dialog (right-click its toolbar button), and evaluate the
    phases again.

**The table was changed in the main GUI.**
    Click **Reload table** (:kbd:`F5`). Your time points, annotations,
    visited cells and phases are kept.

**The last frames have no time points, phases or processed values.**
    These frames are after the last frame analyzed in the main GUI and are
    ignored by the calculations (shaded gray in the plot). Annotate the
    frames in the main GUI and reload the table, or uncheck **Annotations >
    Ignore frames after the last analyzed frame** (see
    `Partially analyzed videos`_).

**A column is missing from the list.**
    Only numerical and yes/no columns can be plotted; text columns are kept
    in the exported table. Use the search field and check that the
    measurement was computed (see `Opening the viewer and loading data`_).

**A premade configuration refuses a channel.**
    Channel names starting with a digit, or with a digit right after a dash
    (e.g., ``561`` or ``GFP-488``), cannot be used in equations. Rename the
    channel (e.g., ``ch561``) and try again.
