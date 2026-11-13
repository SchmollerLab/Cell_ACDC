.. _equation-editor:

Equation editor
===============

The :doc:`measurements viewer <measurements-viewer>` uses the same
**equation editor** wherever you write a formula over the columns of the
``acdc_output`` table:

.. list-table::
    :header-rows: 1
    :widths: 22 40 38

    * - Used for
      - How to open it
      - The equation must return
    * - `Combine measurements`_
      - **Combine measurements...** below the list of columns to plot
      - one value per frame (number, yes/no or text): the new column
    * - `Time point predictors`_
      - Time point identifier dialog: set **Predictor** to **Equation** and
        click **Edit equation...**
      - yes/no per frame, or a number for the maximum/minimum rules
    * - `Custom annotation predictors and eligibility formulas`_
      - Custom annotation dialog, or **Rule eligibility formula** of a phase
        rule: set **Predictor** to **Equation** and click
        **Edit equation...**
      - ``True`` / ``False``, per frame or once for the whole cell
    * - `Cell filters`_
      - **Edit filter...** in the **Main-cell filter** or
        **Other-cell measurements** box of a predictor equation
      - ``True`` / ``False`` or 0/1
    * - `Conditional formatting conditions`_
      - **Edit...** next to a rule of the **Conditional formatting** dialog
      - ``True`` / ``False`` or 0/1 (missing values count as ``False``)

All of them share the same layout, inputs and syntax, which are described
first. The later sections explain what is specific to each use.

.. admonition:: TL;DR

    1. Click in the **Equation** field where you want to insert something.
    2. Type in **Search measurement...** and **double-click** a measurement in
       the list to insert its name.
    3. Add operators and functions with the buttons, or type them, e.g.
       ``cell_area_pxl > 1.5 * FIRST(cell_area_pxl)``.
    4. Click **Test equation** and check the preview below the buttons.
    5. Click **Ok**. It is enabled only after a successful test, and every
       change disables it again.

.. contents:: On this page
    :local:
    :depth: 2


The window
----------

From top to bottom, the editor shows:

* **Instructions**: a short reminder of how to build the equation (the filter
  and condition editors also state what the result must be).
* **Search measurement...**: filters the lists of inputs (see `Inputs`_).
* **Input lists**: the columns of the table that you can use. The predictor
  editors have a second list, **Other-cell measurements** (see
  `Other-cell measurements`_); the other-cell filter editor has the lists
  **1. Candidate cell measurements** and **2. Enquiring cell measurements**
  (see `Cell filters`_).
* **Operator buttons**: ``+``, ``-``, ``*``, ``/``, ``(``, ``)``, square
  (``**2``), power (``**``), natural logarithm (``log(``) and base-10
  logarithm (``log10(``), followed by **Clear** (empties the equation) and
  **Backspace** (deletes the character before the cursor). Hover over a
  button to see what it inserts.
* **Comparisons / logic**: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``,
  **AND** (``&``), **OR** (``|``), **NOT** (``~``), ``True`` and ``False``. In
  the measurements viewer also **ANY** and **ALL** (``any(`` and ``all(``,
  see `Per-cell functions`_).
* Predictor editors only: the **Main-cell filter** box (see
  `Main-cell filter`_), the **Other-cell measurements** box, the
  **Per-cell functions** buttons and, for time points,
  **Time point(s) of each cell** (see `Time point predictors`_). The filter and
  condition editors also have the **Per-cell functions** buttons.
* **Name = Equation**: the left field is the name of the result. In
  **Combine measurements** it is the **New measurement name** that you enter;
  elsewhere it shows the time point identifier, the custom annotation or the
  kind of filter and cannot be edited. The right field is the equation.
* The **status line** and the buttons **Cancel**, **Help...** (a short
  summary of this page), **Test equation** and **Ok**.
* **Test output**: the result of the last test, with a green border if the
  test succeeded, or a red border and the error message if it failed.

Buttons and double-clicked inputs are inserted at the **cursor position** in
the equation field, so you can build the equation with the mouse, type it, or
mix both. You can also select text in the field and type over it.


Inputs
------

Input lists
~~~~~~~~~~~

The inputs are the columns of the table of the current Position (the
measurements viewer lists only columns of the loaded table). They are
grouped as:

* **Numerical inputs**: numbers, for arithmetic and comparisons.

  * **<channel> measurements**: **Cell measurements** (e.g. ``GFP_mean``),
    **Background values** and **Custom measurements** of each channel.
  * **All channels measurements**: used by the main GUI to define one
    measurement for every channel at once (empty in the measurements viewer).
  * **Size measurements** (e.g. ``cell_area_pxl``) and **Region properties**.
  * **Other**: all remaining numerical columns, e.g. processed and combined
    measurements, custom annotations (0/1) and lineage columns such as
    ``parent_ID_tree``.

* **Categorical inputs**: text and yes/no columns, for comparisons, e.g.
  ``cell_cycle_stage``, ``relationship`` and the time points
  ``timepoint_<name>``.

Groups without matching columns stay empty. Group headings cannot be
inserted: expand a group with its arrow and double-click a measurement.

**Double-click** a measurement to insert its name at the cursor. Names that
are not plain Python names (with spaces, dashes or other punctuation, or
starting with a digit) are inserted between backticks, e.g.
```GFP-x mean```. When you type such a name yourself, add the backticks too.

Searching
~~~~~~~~~

Type part of a name in **Search measurement...** to find an input. The search
ignores upper/lower case and also finds approximate matches. Matching names
are highlighted and their groups are expanded, in every list of the editor.
Clear the search field to restore the lists.

Inputs that are not listed
~~~~~~~~~~~~~~~~~~~~~~~~~~

These names can be used although they do not appear in the lists:

* ``frame_i``: the frame index, **starting from 0** (the frame displayed as
  number 1 has ``frame_i == 0``).
* ``Cell_ID``: the ID of the cell.
* ``SizeT`` (measurements viewer): the number of frames of the Position. If
  frames after the last analyzed frame are ignored, it is the number of
  analyzed frames (see "Partially analyzed videos" in
  :doc:`measurements-viewer`). For example, ``LAST(frame_i) == SizeT - 1``
  means "present at the last frame".

In **Combine measurements**, the equation must use at least one listed input:
``frame_i * 2`` alone is refused with "No measurement inputs used", while
``cell_area_pxl * frame_i`` is accepted.

Time points, annotations and phases as inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A **time point identifier** ``Division`` is the yes/no input
  ``timepoint_Division``: ``True`` at the frames where the cell has this time
  point, ``False`` at all other frames.
* A **custom annotation** ``Excluded`` is the input ``Excluded``: 1 on every
  frame of an annotated cell, 0 otherwise. Write ``Excluded == 1`` to get a
  yes/no value (required by custom annotation predictors).
* After the phases were evaluated, ``phase`` is the phase name of each frame
  (e.g. ``phase == "G1"``) and ``phase_<name>`` is a yes/no input per phase.

Names with spaces need backticks, e.g. ```timepoint_Bud emergence```.

A predictor can use the time points and annotations of other predictors:
these are evaluated first. Circular dependencies (A uses B and B uses A) are
refused when you click **Ok**.


Writing equations
-----------------

Equations are evaluated with the Python package
`pandas <https://pandas.pydata.org/docs/reference/api/pandas.eval.html>`__,
row by row (one row per frame and cell), plus the per-cell functions
described below. The following syntax is available:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Write
      - Meaning and example
    * - ``+``, ``-``, ``*``, ``/``
      - Arithmetic: ``GFP_mean / mCherry_mean``
    * - ``**``, ``//``, ``%``
      - Power, integer division, remainder: ``cell_area_pxl ** 0.5``
    * - ``(`` ``)``
      - Grouping: ``(GFP_mean + mCherry_mean) / 2``
    * - ``log(x)``, ``log10(x)``, ``sqrt(x)``, ``exp(x)``, ``abs(x)``
      - Math functions (``log`` is the natural logarithm):
        ``abs(DIFF(GFP_mean))``
    * - ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``
      - Comparisons, which give yes/no: ``cell_area_pxl > 500``
    * - ``"text"``
      - Text values must be in quotes: ``cell_cycle_stage == "G1"``
    * - ``x in [...]``
      - One of several values: ``cell_cycle_stage in ["G1", "S"]``
    * - ``&``, ``|``, ``~``
      - And, or, not:
        ``(cell_area_pxl > 500) & (cell_cycle_stage == "S")``,
        ``~(Excluded == 1)``
    * - ``True``, ``False``
      - Yes / no constants
    * - ``MAX(x)``, ``DIFF(x)``, ``any(c)``, ...
      - Per-cell functions (see `Per-cell functions`_)

Writing conditions ("if")
~~~~~~~~~~~~~~~~~~~~~~~~~

There is no ``if`` / ``else``: write a condition as a yes/no expression.

* Put **parentheses around each comparison** when you combine them, e.g.
  ``(cell_area_pxl > 500) & (GFP_mean > 10)``.
* Use ``&``, ``|`` and ``~`` to combine conditions.
* To use a value only where a condition holds, multiply it by the condition
  (``True`` counts as 1, ``False`` as 0): ``GFP_mean * (cell_cycle_stage ==
  "S")`` is ``GFP_mean`` in S phase and 0 in the other frames.
* Comparisons with a missing value (NaN) are ``False``; arithmetic with a
  missing value gives a missing value.

.. _equation-editor-per-cell-functions:

Per-cell functions
~~~~~~~~~~~~~~~~~~

In the measurements viewer, every equation is evaluated **separately for
each cell**, on the cell's rows sorted by frame. Per-cell functions use all
frames of that cell:

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Function
      - Result
    * - ``MIN(x)``, ``MAX(x)``
      - Minimum, maximum over all frames of the cell
    * - ``MEAN(x)`` (or ``AVERAGE(x)``), ``MEDIAN(x)``
      - Mean, median over all frames of the cell
    * - ``SUM(x)``, ``STD(x)``
      - Sum, standard deviation over all frames of the cell
    * - ``FIRST(x)``, ``LAST(x)``
      - Value at the first, last frame of the cell
    * - ``COUNT(x)``
      - Number of frames of the cell with a value
    * - ``NUNIQUE(x)``
      - Number of different values, e.g. ``NUNIQUE(frame_i)`` is the number of
        frames of the cell
    * - ``DIFF(x)``
      - Change from the previous frame of the cell (missing at its first
        frame)
    * - ``SHIFT(x, n)``
      - Value ``n`` frames of the cell earlier (``SHIFT(x)`` = ``SHIFT(x,
        1)``); a negative ``n`` gives a later value, e.g. ``SHIFT(x, -1)`` is
        the value at the next frame
    * - ``any(condition)``
      - ``True`` if the condition is true in at least one frame of the cell
    * - ``all(condition)``
      - ``True`` only if the condition is true in every frame of the cell

A function that gives one value (``MAX``, ``FIRST``, ``COUNT``, ``any``, ...)
repeats it on every frame of the cell, so you can compare it with the value of
each frame: ``cell_area_pxl > 0.5 * MAX(cell_area_pxl)`` is true at the frames
where the cell is larger than half of its maximum area.

* The aggregates ignore missing values, and ``FIRST`` / ``LAST`` give the
  first / last value that is not missing.
* ``DIFF`` and ``SHIFT`` move along the **frames where the cell exists**, not
  along the video: if the cell is missing at some frames (tracking gap),
  ``SHIFT(x, 1)`` is its value at the previous frame where it exists.
* The predictor, filter and condition editors have a button for each function
  (hover over it for a description). In **Combine measurements**, only
  **ANY** and **ALL** have buttons: type the other functions.
* Per-cell functions and ``SizeT`` are available only in the measurements
  viewer. In the main GUI, combined measurements are evaluated row by row.


Testing and confirming
----------------------

* **Live check** (Combine measurements only): while you type, the line below
  the equation shows the equation in green if it can be evaluated, or the
  error in red.
* **Test equation** evaluates the equation on the table of the current
  Position, without changing it, and shows:

  * Combine measurements: the first 10 rows with the inputs used and the
    result;
  * time point predictors: the predicted time points of up to 20 cells (frame
    numbers as displayed, starting from 1) and a status line such as
    "Predicted time points for 12 of 40 cells";
  * custom annotation predictors: the cells that would be annotated, with
    their frames, and a status line such as
    "Annotated 85 frame(s) of 3 of 40 cells";
  * filters and conditions: whether the equation is valid for all cells.

* If the test fails, the error is shown in the status line and in the test
  output (red border). Errors that occur for one cell name it, e.g.
  "Evaluating the equation failed for Cell_ID 12: ...": select that cell in the
  viewer to inspect its values.
* **Ok** is enabled only after a successful test. Any change of the equation,
  the name or the predictor settings disables it again. Clicking **Ok** while
  it is disabled runs the test: if the test succeeds, click **Ok** again (the
  filter and condition editors close directly after a successful test).
* **Cancel**, or closing the window, discards your changes.


Combine measurements
--------------------

**Combine measurements...** (below the list of columns to plot) calculates a
new column from existing ones.

1. Enter the **New measurement name** (letters, digits, ``_`` and ``-``;
   typed spaces become ``_``). It must not be the name of an existing column.
2. Write the equation and click **Test equation**.
3. Click **Ok**. The new column is added to the table of the
   **current Position** and appears in the list of columns to plot.

* **Channel prefix**: if all channel measurements of the equation belong to
  the same channel, the channel name is put in front of the name. For example,
  ``GFP_mean / cell_area_pxl`` named ``density`` creates ``GFP_density``. With
  measurements of several channels (or none), the name is used as entered.
* **Result**: one value per frame of each cell, or one value per cell (e.g.
  ``MAX(GFP_mean)``) repeated on its frames. Numerical and yes/no results can
  be plotted and used by predictors; text results are kept in the exported
  table.
* The equation is **recorded in the configuration** and calculated again when
  you load the configuration (choose **Calculate** when asked). Save the column
  in the ``acdc_output`` table with **Save tables** or
  **Export measurements...** (see :doc:`measurements-viewer`).
* If the frames after the last analyzed frame are ignored, these frames get
  empty values.

Examples:

* ``GFP_mean / mCherry_mean``: ratio of two reporters.
* ``GFP_mean / MAX(GFP_mean)``: reporter normalized to the maximum of each
  cell.
* ``cell_area_pxl / FIRST(cell_area_pxl)``: area relative to the first frame
  of the cell.
* ``any(timepoint_Division)``: yes on all frames of the cells that have a
  Division time point.
* ``(cell_cycle_stage == "S") & (GFP_mean > 10)``: yes at the frames in S
  phase with a high reporter level.


Time point predictors
---------------------

A time point predictor evaluates the equation on every frame of each cell and
turns the results into the cell's time points. Choose the rule under
**Time point(s) of each cell**:

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Rule
      - Time point(s) of each cell
    * - First frame where the equation is true
      - The first frame where the condition is true (one time point).
    * - Last frame where the equation is true
      - The last frame where the condition is true (one time point).
    * - Every frame where the equation becomes true
      - Every frame where the condition is true and was false at the previous
        frame of the cell (the first frame of the cell counts if the condition
        is true there). Use it for events that can happen several times.
    * - Frame of the maximum value
      - The frame where the (numerical) equation is largest. If several frames
        have the same value, the first one.
    * - Frame of the minimum value
      - The frame where the equation is smallest.

Missing results count as false; with the maximum and minimum rules they are
ignored. A cell without a true frame (or without any value) gets no time
point.

Examples:

* ``cell_area_pxl > 1.5 * FIRST(cell_area_pxl)`` with **First frame**: the
  first frame where the cell is 1.5 times larger than at its first frame.
* ``GFP_mean`` with **Frame of the maximum value**: the frame of the reporter
  peak.
* ``(GFP_mean > 10) & (SHIFT(GFP_mean, -1) > 10) & (SHIFT(GFP_mean, -2) >
  10)`` with **First frame**: the first frame from which the reporter stays
  above 10 for three frames (ignores single noisy frames).
* ``DIFF(cell_area_pxl) < -0.3 * SHIFT(cell_area_pxl)`` with
  **Every frame where the equation becomes true**: every sudden drop of the
  area by more than 30 % between two frames.

Optionally, restrict which cells are predicted with the `Main-cell filter`_
and use the measurements of other cells (e.g. the mother) with
`Other-cell measurements`_. Close the editor with **Ok**, then click
**Test predictor** in the identifier dialog to check the result again.


Custom annotation predictors and eligibility formulas
-----------------------------------------------------

The editor of a **custom annotation predictor** is the same as for time
points, without **Time point(s) of each cell**:

* The equation must give ``True`` / ``False``: one value per frame (e.g.
  ``cell_area_pxl > 500``) or one value for the whole cell (e.g.
  ``MAX(cell_area_pxl) > 500``). Numbers, also 0 and 1, are refused: compare
  them, e.g. ``Excluded == 1`` instead of ``Excluded``.
* The **Consolidation** of the annotation dialog (outside the editor) combines
  the results of the frames into one annotation for the whole cell: **All**
  (yes only if yes in every frame), **None** (yes only if no frame is yes) or
  **Most** (yes if more than half of the frames are yes). **Test equation**
  uses the consolidation selected when you opened the editor.
* With several values of an other-cell input, a frame is true if the equation
  is true for any of them, before the consolidation.

The **Rule eligibility formula** of a phase rule (see "Eligibility formula"
in :doc:`measurements-viewer`) is a custom annotation predictor: set it to
**Equation**, choose the **Consolidation**, click **Edit equation...** and
**Test predictor**. The rule then applies only to the cells for which the
formula is yes.


Main-cell filter
----------------

Check the **Main-cell filter** box of a predictor editor to evaluate the
predictor only on some cells; the other cells are not predicted. The cells
that pass the filter keep all their frames for the prediction.

Select a preset under **Select cells**:

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Preset
      - Equation
    * - All cells
      - (none)
    * - Present at the last frame
      - ``LAST(frame_i) == SizeT - 1``
    * - Not present at the last frame
      - ``LAST(frame_i) != SizeT - 1``
    * - Present at the first frame
      - ``FIRST(frame_i) == 0``
    * - Not present at the first frame
      - ``FIRST(frame_i) != 0``
    * - Present throughout the video
      - ``(FIRST(frame_i) == 0) & (LAST(frame_i) == SizeT - 1) &
        (NUNIQUE(frame_i) == SizeT)``
    * - Mother cells / Bud cells (budding)
      - ``relationship == "mother"`` / ``relationship == "bud"``
    * - Cells in G1 / Cells in S (budding)
      - ``cell_cycle_stage == "G1"`` / ``cell_cycle_stage == "S"``
    * - Cells with parent
      - ``parent_ID_tree != -1``

The budding presets are offered when the table has cell cycle annotations,
**Cells with parent** when it has a lineage tree. Presets are read-only:
select **Custom equation** to type your own filter or open the filter editor
with **Edit filter...** (see `Cell filters`_), and to choose
**Keep cells where true in**:

* **Any**: keep a cell if the filter is true in at least one of its frames.
* **All**: keep a cell only if the filter is true in every frame.

The filter can use the other-cell inputs configured in the
**Other-cell measurements** box. For example, select **Daughter cell(s)**
there and use ``COUNT(cell_area_pxl_other) > 0`` with **Any** to predict only
the cells that have daughters.


.. _equation-editor-other-cells:

Other-cell measurements
-----------------------

Check the **Other-cell measurements** box of a predictor editor to use the
measurements of other cells, e.g. of the mother or of all cells in the same
frame. The list **Other-cell measurements** then contains every column ``x``
as ``x_other`` (e.g. ``GFP_mean_other``).

How other cells are selected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each frame *t* of the cell being predicted (the **enquiring cell**):

1. The **Frame window** (from ... to ..., relative to *t*) selects the rows of
   all **other** cells in these frames. The enquiring cell itself is never
   included.
2. The filter **Equation** selects the candidate cells, e.g. the mother.
3. **Keep candidates** decides which rows of these cells are used:

   * **Only where true** (default): only the rows where the filter is true;
   * **Any**: all rows in the window of each cell for which the filter is
     true in at least one of these rows;
   * **All**: all rows in the window of each cell for which the filter is true
     in every one of these rows.

4. In the equation, a function of an ``_other`` input combines the values of
   the selected rows into one value for frame *t*, e.g.
   ``MEAN(GFP_mean_other)``. Available: ``MAX``, ``MIN``, ``MEAN``,
   ``AVERAGE``, ``MEDIAN``, ``SUM``, ``STD``, ``COUNT``, ``NUNIQUE``,
   ``FIRST`` and ``LAST``.

.. important::

    If no row is selected (e.g. the cell has no mother, or no other cell is in
    the window), the other-cell inputs and functions give **0**, also ``MEAN``.
    Use ``COUNT(x_other) > 0`` to check whether there is a selected cell.

An ``_other`` input without a function (e.g. ``GFP_mean_other``) uses the
value of the selected row. If the selected rows have different values, the
equation is evaluated once per value: a condition is true if it is true for
any of them, and the maximum/minimum rules use the most extreme result (a
warning is printed in the terminal). Use a function such as ``MEAN`` to
combine the values instead. Older configurations may contain
``MEAN_OTHER(GFP_mean)``, which is the same as ``MEAN(GFP_mean_other)``.

Presets
~~~~~~~

Select which cells to use under **Select cells**:

.. list-table::
    :header-rows: 1
    :widths: 22 50 28

    * - Preset
      - Filter equation
      - Frame window
    * - Other cells
      - (none: all other cells)
      - ``t`` to ``t``
    * - Mother cell (lineage tree)
      - ``Cell_ID == this_parent_ID_tree``
      - ``t - 1`` to ``t - 1``
    * - Sister cell(s) (lineage tree)
      - ``(parent_ID_tree > 0) & (parent_ID_tree == this_parent_ID_tree)``
      - ``t`` to ``t``
    * - Daughter cell(s) (lineage tree)
      - ``parent_ID_tree == this_Cell_ID``
      - ``t + 1`` to ``t + 1``
    * - Mother cell (budding)
      - ``(Cell_ID == this_relative_ID) & (this_relationship == "bud")``
      - ``t`` to ``t``
    * - Daughter cell(s) (budding)
      - ``(relative_ID == this_Cell_ID) & (relationship == "bud")``
      - ``t`` to ``t``

* The **budding** presets use the cell cycle annotations (``relative_ID`` and
  ``relationship``; the bud and its mother exist in the same frame). They are
  used when the table has buds or no lineage tree.
* The **lineage tree** presets use ``parent_ID_tree``. With symmetric
  division, the mother exists until the frame before its daughters appear,
  hence the windows ``t - 1`` (mother) and ``t + 1`` (daughters).
* Without these annotations, only **Other cells** and **Custom equation** are
  offered. A saved predictor whose source needs missing annotations shows it
  as "(annotations missing)" with a warning: it is kept unchanged but cannot
  be evaluated on these data. Select another source to replace it.

Presets are read-only. Select **Custom equation** to edit the filter equation,
**Keep candidates** and the frame window, e.g. starting from the equation of a
preset in the table above.

Frame window
~~~~~~~~~~~~

The bounds are expressions of ``t`` with whole numbers, ``+``, ``-``, ``*``,
``//``, ``%`` and parentheses, e.g. ``t`` to ``t`` (same frame), ``t - 1`` to
``t - 1`` (previous frame) or ``t - 5`` to ``t + 2``. Both bounds are
included and the start must not be after the end. Frames outside the video
are left out, and frames where a cell is missing are not filled in. Unlike
``SHIFT``, the window refers to video frames, not to the frames where a cell
exists.

Examples:

* **Other cells**, ``GFP_mean > 2 * MEAN(GFP_mean_other)``: the cell is
  twice as bright as the mean of the other cells in the same frame.
* **Mother cell**,
  ``(COUNT(GFP_mean_other) > 0) & (GFP_mean > MEAN(GFP_mean_other))``: the
  cell is brighter than its mother. The ``COUNT`` part leaves out the frames
  without a mother, where ``MEAN`` gives 0.
* **Daughter cell(s)**, ``COUNT(cell_area_pxl_other) > 0`` with
  **First frame**: the last frame of a mother before its daughters appear
  (lineage tree).


Cell filters
------------

**Edit filter...** (with **Custom equation** selected) opens the filter
editor of the **Main-cell filter** or of the **Other-cell measurements**.

* The filter must give ``True`` / ``False`` or 0/1, one value per frame or a
  single value; missing values are not allowed.
* An empty filter keeps all cells ("No filter equation; all cells are kept").
* **Test equation** checks the filter on the whole table, and **Ok** closes the
  editor as soon as the test succeeds.

The **main-cell filter editor** is evaluated on each cell across its frames,
like a predictor. It also offers the **Other-cell measurements** list, with
the source and frame window configured in the predictor editor.

The **other-cell filter editor** decides which other cells are selected. It
has two lists:

* **1. Candidate cell measurements**: plain names (e.g. ``parent_ID_tree``)
  are the values of the candidate cell (the other cell);
* **2. Enquiring cell measurements**: names with the ``this_`` prefix (e.g.
  ``this_Cell_ID``) are the values of the cell being predicted, at frame *t*.

For example, ``parent_ID_tree == this_Cell_ID`` selects the candidates whose
parent is the enquiring cell (its daughters), and
``cell_area_pxl > this_cell_area_pxl`` the cells larger than the enquiring
cell. Per-cell functions in this filter are computed over the candidate cell's
rows in the frame window.


.. _equation-editor-conditional-formatting:

Conditional formatting conditions
---------------------------------

In the **Conditional formatting** dialog (right-click a plotted column), click
**Edit...** next to a rule to write its condition in the editor (see
"Conditional formatting" in :doc:`measurements-viewer`).

* The condition is evaluated on the **selected cell** across its frames; the
  frames where it is true get the rule's color.
* It must give ``True`` / ``False`` or 0/1, one value per frame or one for the
  whole cell. Missing values count as ``False``.
* Per-cell functions and ``SizeT`` are available; other-cell inputs are not.

Examples: ``mCherry_ramp == 1``, ``cell_area_pxl > 1.5 * MEAN(cell_area_pxl)``,
``frame_i >= MAX(frame_i * timepoint_Division)`` (from the Division time point
onwards).


More examples
-------------

.. list-table::
    :header-rows: 1
    :widths: 45 55

    * - Goal
      - Equation
    * - Frame of the Division time point (repeated on all frames)
      - ``MAX(frame_i * timepoint_Division)``
    * - Frames from the Division time point onwards, only in cells that have
        it
      - ``(frame_i >= MAX(frame_i * timepoint_Division)) &
        any(timepoint_Division)``
    * - Present at the last (analyzed) frame
      - ``LAST(frame_i) == SizeT - 1``
    * - Cells tracked for at least 20 frames
      - ``COUNT(cell_area_pxl) >= 20``
    * - Not annotated as Excluded
      - ``Excluded == 0``
    * - Area growth since the previous frame, in percent
      - ``100 * DIFF(cell_area_pxl) / SHIFT(cell_area_pxl)``
    * - Reporter level above the cell's median for the first time
      - ``GFP_mean > MEDIAN(GFP_mean)`` (time point rule **First frame**)


Troubleshooting
---------------

**Ok stays disabled.**
    Click **Test equation**: every change of the equation, the name or the
    settings requires a new successful test.

**"No measurement inputs used".**
    The equation uses no input of the lists, e.g. only ``frame_i``,
    ``SizeT`` or numbers. Add a measurement.

**"name '...' is not defined".**
    The name is misspelled or the column is not in the table of the current
    Position. Insert names by double-clicking them in the lists; names with
    spaces or punctuation need backticks.

**A combined condition gives unexpected results.**
    Put parentheses around each comparison, e.g.
    ``(cell_area_pxl > 500) & (GFP_mean > 10)``.

**"Enable Other-cell measurements to use _other inputs."**
    Check the **Other-cell measurements** box, or remove the ``_other``
    inputs.

**"The result must be boolean (True/False); got dtype int64."**
    Custom annotation predictors need ``True`` / ``False``: compare numbers,
    e.g. ``Excluded == 1`` instead of ``Excluded``.

**"A cell filter must return only bool or 0/1 (no missing values)."**
    The filter gives other numbers or missing values. Use a comparison, e.g.
    ``GFP_mean > 0`` instead of ``GFP_mean``.

**"Evaluating the equation failed for Cell_ID ...".**
    The equation fails for this cell (e.g. a function that needs numbers
    gets text). Select the cell in the viewer to check its values.

**"Frame range start (...) exceeds end (...)".**
    The start of the frame window is after its end: swap the bounds.

**"Circular predictor dependency: ...".**
    Two or more predictors use each other's results. Remove one of the
    dependencies.

**"... is not a supported function".**
    Only the functions listed on this page are available (e.g. there is no
    ``where``). Multiply by a condition instead (see
    `Writing conditions ("if")`_).

**"Other-cell inputs ... are not available in conditional formatting."**
    Conditional formatting sees only the selected cell. Calculate the value
    with a predictor or a combined measurement first and use that column.
