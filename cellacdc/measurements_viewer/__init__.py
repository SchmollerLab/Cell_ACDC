"""Standalone window to navigate frames, z-slices and cell IDs of Cell-ACDC 
data while plotting the time-course of `acdc_output` table columns (measurements)
of the selected cell, or two measurements against each other for all cells and
frames of the current position. Hover over a point for measurement details;
click a point to navigate to its frame and zoom to its cell.
The plot options of the context menu disable the transforms FFT, dy/dx and
Y vs Y' in both plot modes (`disableUnsupportedPlotTransforms`), and
"Downsample" and "Clip to View" (`disableUnsupportedPlotDownsampling`):
every point is a cell at one frame with its own symbol brush and hover
data, while these options change the number of points or the x axis
(pyqtgraph raised).
The widget-free code (data loading and the window's data/logic methods,
`measurements_viewer_utils.MeasurementsViewerUtils`, inherited by
`MeasurementsViewerWin`) lives in `measurements_viewer_utils`; this
docstring specifies both modules.
In comparison mode, check two columns to select X and Y. X/Y markers replace
the color swatches; comparison and time-course selections are independent.
The comparison scatter does not depend on the selected cell: navigating to
another cell (or clicking a point) keeps the plot and its zoom, and redraws
it only if the cell's automatic predictions changed the X or Y values.
Column names are read-only, and the fixed-width selection indicator displays
the color or axis instead of a separate checkbox tick. Clicking anywhere on a
column (indicator or name) toggles its selection.
Plotted columns get default colors from the colorblind-friendly Okabe-Ito
palette (``SIGNAL_COLORS``: without black, and with vermillion last because
red is reserved for the current frame line; on the light background of
`Invert black/white` black replaces the yellow; further columns get tab10
colors and then evenly spaced hues, none of them red). The default colors are
stable: plotting another column never recolors
the plotted ones, a new column takes the first color no plotted column uses,
and a column plotted again gets its previous color back unless another
plotted column took it meanwhile. The comparison scatter uses the Y column's
color. Right-click a column to set a custom color or reset it to the default.
Custom colors are saved in the configuration's ``measurements_colors``
``{column: "#rrggbb"}`` (premade templates may preset them, e.g. green/red
for the FUCCI reporters).
Time course y axes: plotted columns share the default y axis (number 1)
unless the column context menu puts them on another numbered axis
("Create new y axis for this"; "Add to y-axis of..." lists the other
displayed axes by their plotted columns, never the axis of the clicked
column (its remembered axis if it is not plotted), so it stays empty and
disabled until a column is put on another axis; a plotted column alone on
its axis cannot get a new one). Moving the last plotted column of axis 1
to another axis merges that axis into axis 1 (`addToYAxisOf`), so that
columns plotted later, on axis 1 by default, join it. Nothing merges when
axis 1 has no plotted column (its columns were unplotted): another axis is
then the left one, and a column plotted later on axis 1 gets a new left
axis again. The axis numbers of
the columns not on axis 1 are kept in ``timeCourseYAxes``
``{column: number}``, also while a column is not plotted (plotting it
again puts it back on its axis); a new axis takes the number after every
number in use, also by columns that are not plotted, so it is the rightmost
axis. The displayed axes are those with plotted columns, sorted by number:
the first one is the left axis of the main view box (labeled "Value", or
"Normalized value", when it is the only axis, also if its number is not 1),
the others are extra axes on the right labeled with their columns, each
with its own view box (`_TimeCoursePlotItem`, which also documents mouse
handling: pan/zoom in the plot area moves every axis, on an axis only that
axis; "A", "View All", every automatic rescale and every change of the log
or Subtract Mean transform of the plot options fit each axis to its own
data; plot options and the plot exporters include every axis). The
hover tip of the points of every axis is shown on the main view box
(`measurementPointsHovered`). The unanalyzed frames are shaded under the
curves of every axis. The phase band gets an empty right axis as wide as
the extra axes so that frames stay aligned. Comparison mode has no extra
axes.
Renaming a column keeps its axis; removing a time point identifier or
custom annotation column drops it. The axes are saved in the
configuration's ``time_course_y_axes`` ``{column: number}``: every plotted
time course column (also those on axis 1), then the other columns not on
axis 1 (the FUCCI template puts ``area`` on axis 2, next to the 0-1
reporters). ``Normalize measurements (0-1)`` is independent of the axes:
it rescales every curve, whatever its axis (with several axes, each axis
label gets " (normalized)"). Conditional formatting segments of a column
are drawn on the axis of the column.
Measurement selectors use the equation editor's tree organization: numerical
and categorical inputs, channel-specific cell/background/custom measurements,
size measurements, region properties, and other numerical inputs. Plotting
lists only numerical and boolean columns. Search keeps matching leaves and
their ancestors visible; group headings cannot be selected for plotting or
processing.

The window loads data on its own (menu `File --> Load Folder...`, 
`Open Image File...` or `Open Recent`) and it shares only GUI elements with 
the main GUI (`gui.guiWin`) through the mixins in:

- `gui_navigation`: frames/z-slices navigation, zoom to object, recent paths
  and LUT. The arrow keys (Left/Right: frames or z-slices, Up/Down: Cell
  IDs) are handled by an application event filter (`eventFilter`) whatever
  widget of the window has the focus (buttons and check boxes would move
  the focus with them), except text inputs, spin boxes, drop-down menus and
  lists (`qutils.ARROW_KEYS_WIDGET_CLASSES`), the menu bar and open menus
  (keyboard navigation of the menus), and while the window is disabled
  (loading data, `runInThread`).
- `gui_annotations`: display of contours, segm. masks, IDs, cell cycle and 
  lineage info, mother-daughter lines and object tracks. As in the main
  GUI's 'Viewer' mode, mother-daughter lines come from the cell cycle
  annotations, or from the `parent_ID_tree` column in frames without cell
  cycle annotations (`isLineageTreeLinesModeActive`).
- `gui_viewer_tools`: the other 'Viewer' mode tools (zoom, ruler, scale bar, 
  timestamp, colormaps, search ID, hover info, shortcuts, slideshow, 
  3D viewer, objects properties, cell cycle table and objects count). The
  3D viewer follows the frame, z-slice/projection, plane, pan/zoom and the
  mouse cursor of the image (`update3DViewer`, `update3DViewerViewRegion`,
  `update3DViewerCursor`). The slideshow shows the viewer's z-slice,
  projection and plane (its own z-slider is hidden). Loading data turns
  the tools off (`uncheckViewerTools`, as the main GUI's `reInitGui`), also
  the scale bar and the timestamp, which keep the pixel size and time
  increment of the data they were added to. Adding or removing the scale
  bar keeps the zoom (the bar is placed in the current view).
- `gui_overlay`: overlay of other channels and of other segmentation files.
  As in the main GUI, the channels overlay is turned off and disabled on
  zy/zx planes and restored back on xy (`setOverlayEnabledForPlane`). The
  overlay modes (true transparency, alpha-encoded intensity) keep the check
  states the user clicked (channel display settings, see LUTs below; also
  after loading other data) and apply when the overlay is turned on.
- LUTs: as in the main GUI, the LUT (colormap) of the displayed channel and
  of each overlay channel is saved per channel name in the channel display
  settings (own file, shared with the main GUI and other open windows) and
  restored when a channel with that name is loaded
  (`gui_navigation.LutGuiElements`; `restoreImageChannelLutAfterLoading`
  runs in `initViewerToolsAfterLoading`).
- `gui_custom_annotations`: custom annotations of the objects.

Open it from the launcher (button "Launch measurements viewer..." in the
"Measurements visualisation" box, or Utilities > Measurements > Measurements
viewer..., both `_main.mainWin.launchMeasurementsViewer`) or run it with
`python -m cellacdc.measurements_viewer [folder_path]`.
After data is loaded (on the first event loop pass after `setData`, i.e.
after the rest of the load such as the overlay channels), the Cell ID
controls (label, spinbox and scrollbar at the bottom) blink and the status bar 
explains how to select the plotted cell. Ctrl+F focuses the search bar and
blinks it, like the toolbar search
button.
As in the main GUI (`gui.guiWin.setStatusBarLabel`), the left label of the
status bar shows what is displayed (`setStatusBarLabel`, text from
`measurements_viewer_utils.loaded_data_status`): experiment folder, Position
folder (with its index when several are loaded), basename, displayed and
overlay channels, segmentation file and configuration file name; its tool tip
has the full Images folder and configuration file paths (paths too long for a
tool tip line broken after separators on indented lines below their label,
`measurements_viewer_utils.path_tooltip_lines`). It is updated when
data, a position or a configuration is loaded, the overlay channels change
and a configuration is saved (logged once per load and position change),
and cleared without data. It is plain text and clipped (never widens the
window). Other messages are temporary (`QStatusBar.showMessage`, e.g.
"Measurements exported to ...", the color scheme restart notice of
`onToggleColorScheme`): they hide the label only while they are shown.
The Help menu opens the online user guide (`urls.measurements_viewer_docs_url`,
source ``docs/source/measurements-viewer.rst``) and the Cell-ACDC
documentation, shows a quick start (``MEASUREMENTS_VIEWER_QUICK_START``, also
shown by the info button next to the plot mode selector of the Measurements
panel, with a button opening the user guide) and has the Ctrl+F search action.

The "Cell-ACDC objects" dock (object properties, left side button) always
shows the selected cell: whenever the images update (frame, z-slice, cell,
position, dock shown) its values are recomputed for the current frame, or
it reports that the cell does not exist in this frame (or, with 3D
segmentation shown as a single z-slice, is not in this z-slice) with empty
value fields (`clearPropsDockValues`). Typing an Object ID
in the dock selects that cell (as clicking it) if it is in the table, while
hover updates ("Highlight objects on mouse hover") never change the
selection (`propsWidgetIDvalueChanged`, `syncPropsDockToSelectedID`).

At startup, load a saved configuration or skip to start normally (unless
started with a folder path, which is loaded directly). The File
menu and the file toolbar can save the tables (`saveTables`, Ctrl+S: only the
positions with unsaved changes, with the same backup and modified-on-disk
check as below) and load, save (Ctrl+Shift+S) or save a copy of a
configuration. On closing, if any
position's table or the configuration changed since it was loaded/saved, a
prompt offers to save the tables (overwriting each position's acdc_output
file), the configuration (or save it as a new file), both, or neither
(its buttons get the usual Cell-ACDC icons from `widgets.getPushButton`).
Before a table is overwritten, the previous file is backed up in the
Position's ``recovery`` folder (like in the main GUI). If the file was
modified on disk since the viewer loaded, reloaded or saved it (e.g., saved
in the main GUI), a warning offers to reload it and save (keeping the
viewer's time points, custom annotations, visited cells and the phases
evaluated in this session, like Reload table), to overwrite it, or to cancel.
The same prompt appears before loading a configuration or opening a folder,
file or recent folder replaces loaded data; Cancel aborts the load and keeps
the loaded data and their edits. The prompt does not block the window: if
tables or the configuration get changes it did not list while it is open,
any answer but Cancel shows it again with the new changes, and its saves use
the tables changed when it is answered (`askSaveChanges`).
Reading the positions (and the regionprops of the first displayed position,
`precompute_position_regionprops`, and the overlay channel files of a loaded
configuration, `preload_overlay_channels`, which `load_fluo_data` then
takes), and recalculating the measurement steps of a loaded configuration,
run in a worker thread (`runInThread`): the window keeps repainting with its
input (and that of the other application windows, so that no dialog's
event loop can nest in the wait) disabled and a wait cursor, while the
status bar and the terminal report the progress (e.g. "Loading Position_1
(1/3)...") and the elapsed time. Errors are reported as before. One load runs
at a time: loading other data or closing the window while a load (including
its dialogs) is in progress is refused with a status bar message, and so is
loading while the "Save changes before closing?" prompt is open. Loading and
measurement processing exclude each other (both directions, checked again
after the non-blocking dialogs), so that results are never merged into
replaced data.
Configurations are JSON files in the settings folder
and retain multiple experiment inputs (Images folders, image/overlay channels
and segmentation), both plot modes' selections, predictor/annotation definitions,
phase rules, and separate processing recipes and combine equations. Saving
updates the loaded input (same Images folders, channel and segmentation) in
place with the current overlay and visible channels (the loaded option if
several options share that experiment). Loading
prompts for a saved input or a new experiment (which reuses the configured
channel, segmentation and overlays when every position has them, asks for
replacements otherwise, and offers unused channels as extra overlays; the base
channel is shown when no configured visible channel remains), then asks whether to calculate
the measurement steps. Calculating replays them in their original order on all
positions (overwriting their previous outputs) before lazy per-cell
prediction; using existing values requires every output column to be present
in every position. Each output column is produced by exactly one recorded
step (validation rejects duplicate output names across steps, and steps that
read outputs of a step with a higher order): re-applying a
recipe or equations that write exactly the same outputs as a recorded step
replaces that step (keeping its order; the log and status bar say that the
configuration now records the new parameters), and outputs that only partly
overlap a recorded step's outputs are refused before running. A replacement
that reads outputs of later steps moves after all steps, or is refused
before running when other recorded steps read its outputs. Version 1 files remain
loadable; their viewed-cell lists are migrated to the table's visited column.

Partially analyzed videos: when a table is read (`load_position`, Reload
table), `last_analyzed_frame` finds the last frame analyzed in the main GUI
(``posData.lastAnalyzedFrame``): the last cell cycle annotated frame (rows
with all cell cycle columns filled) when there are cell cycle annotations,
else the last tracked frame (highest ``frame_i`` of the table). With the
option ``Annotations --> Ignore frames after the last analyzed frame``
(configuration key ``mask_unanalyzed_frames``, default true) every
calculation of a position whose last analyzed frame is before its last
frame (`analysisLastFrame`, `analysisDf`, `analysisSizeT`) uses only the
rows up to that frame, with ``SizeT`` = number of analyzed frames: time
point and custom annotation predictors (the table passed to ``predict``;
results are written into the full table, so later frames get no time
points, while annotations stay one value per cell computed from the
analyzed frames), the phase translator (``phase_translator.evaluate``
``last_frame``: no phase after that frame), measurement processing
(``apply_recipe`` ``max_frame_i``) and combined measurements, also when
recalculated at load (NaN on the later rows), conditional formatting, and
the editor dialogs' tests. Images, navigation and plots still show all
frames; the ignored frames are shaded gray in the time course and phase
band, and a label next to the frames scrollbar shows the last analyzed frame
(as the main GUI's last tracked frame). Loading data with partially analyzed
positions asks (after reading the positions, before the measurement steps
and predictions) whether to ignore the later frames (default) or use all
frames, unless the loaded configuration stores ``mask_unanalyzed_frames``;
an answer for a configuration that does not store it counts as an unsaved
configuration change. Masked boolean outputs are stored as 1.0/0.0 (NaN on
the ignored frames) so they stay plottable. The frame up to which each
position's step outputs were calculated is recorded
(``measurementStepsLastFrames``, configuration key
``measurement_steps_last_frames`` ``{Images folder: frame index}``; no
entry = all frames; a record that differs from the saved configuration
after its steps were calculated or their existing values blanked, also by
Open folder, counts as an unsaved configuration change): saved step
outputs used as existing values (also when opening data while steps are
recorded), or read
back on Reload table, are set to NaN after the smaller of that frame and the
current last analyzed frame (`blankSavedStepOutputs`), also in the "as on
disk" snapshot (NaN saved in
tables with cell cycle columns is read back as 0; restoring it is not an
edit). Toggling the option does not recalculate earlier results (except
phases evaluated automatically, which are re-evaluated on all cells):
re-evaluate predictors and phases, or reload the configuration to
recalculate the measurement steps. Toggling is refused while data load or
measurement processing runs, and Combine measurements refuses to apply
equations if the option changed while its dialog was open. A reload that
changes the last analyzed frame is logged (with a hint when the steps were
calculated on fewer frames) and handled like a toggle (also "Reload and
save" when saving a table changed on disk: the re-evaluated phases are
saved).

Premade configurations are templates shipped in
`cellacdc/measurements_viewer_premade_configurations/*.json` and listed as
"Premade" in the load dialog (any loaded file with `channel_placeholders` is
treated as premade). They use the normal configuration schema without
`images_paths`/`input_options` (`segm_endname` may be empty) plus
`"channel_placeholders": {"RED": "description", ...}`. Channels are written
as braced tokens such as `{RED}` anywhere in the file (channel fields, column
names like `{RED}_mean`, equations, predictors, phase rules; keys and values).
In equations, predictors and formatting conditions, column names built from
channel tokens are quoted in backticks (`` `{RED}_mean` / `{GREEN}_mean` ``):
the backticks are dropped when the resolved name is a plain identifier and
kept otherwise, so channels with dashes (`mKO2-hCdt1`) work. Quoted names
whose first character, or the first character after a dash or other
non-word character, is a digit (`561_mean`, `GFP-488_mean`) cannot be
parsed reliably (Python >= 3.12 fails, older versions split them), and
names with quotes, brackets, `#` or backslashes parse differently on
Python 3.10/3.11 and 3.12+, so these rules are checked explicitly on every
Python version (`isEquationName`) and such channels are rejected for
placeholders used inside backticks (`resolvePremadeTemplate`; the setup
dialog shows the message and keeps its selection).
Templates can also declare
`"input_placeholders": {"NAME": {"type": "float"|"int"|"str", "default": v,
"description": "...", "info": "..."}}` for values to enter during setup (e.g.
normalization constants); their `{NAME}` tokens are replaced like channels,
and a string that is exactly one token becomes the typed value (so recipe
kwargs stay numeric). Loading a premade configuration asks for an experiment
folder, then for one channel per placeholder (drop-down of the channels
common to all positions, each channel used at most once) and for the input
values (prefilled with the defaults, info button per input), replaces every
token, asks which channel
to display as base layer (template channel offered first; with an empty
template `channel`, e.g. the FUCCI template whose reporters are both overlays,
the unmapped channels such as phase contrast are offered first; a mapped
channel that is not the base becomes a visible overlay), resolves the
segmentation/overlays like a new experiment input, and writes the result
(without placeholder sections, with a `premade` source/mapping/input values
record) as a
new configuration in the settings folder, which is then loaded. The template
itself is never modified. Templates use the 2D mean intensity column
`{RED}_mean`; z-stack tables have none, so when the selected tables lack
`<channel>_mean`, the setup uses the column of the background utility
instead (`<channel>_mean_3D` with 3D segmentation, else
`<channel>_mean_meanProj`, `premadeMeanSuffixes`), replacing only whole
`{RED}_mean` names of the template (`substituteMeanColumns`); it stops with
an error if the tables have none of them.

Optional setup-only `"channel_hints": {"PLACEHOLDER": ["gfp", ...]}` lists
name parts (case-insensitive, in order of preference) used to preselect the
channel of each placeholder in the setup dialog when no channel contains the
placeholder name (FUCCI: GFP/mAG/Venus... for Geminin, mCherry/mKO2/RFP...
for Cdt1), so that "Calculate background values..." starts with the right
channels.

Templates can also link float inputs to the summary table of the "Compute
background intensity" utility (`background_values`, launcher Utilities >
Measurements) with a setup-only section `"background_values": {"statistic":
"median", "low_percentile": 5, "high_percentile": 95,
"exclude_zero_pixels": true, "inputs": {"INPUT": {"channel":
"CHANNEL_PLACEHOLDER", "value": "constant_min"}}}` (settings optional, values
from `background_values.SUMMARY_VALUE_COLUMNS`; validated by
`validateBackgroundValuesSpec`). The setup dialog then has two more buttons:
"Calculate background values..." (runs the utility on the experiment
selected for the configuration, `computeBackgroundSummary`: asks only its
Positions, preselecting the ones of the selected folder, and shows the
utility's parameters dialog prefilled with these settings and the channels
selected for the linked placeholders) and "Load calculated background
values..." (a `<name>_summary.csv` saved before, e.g. of another
experiment). Each linked
input gets the value of the table row of its placeholder's selected channel
(the user picks another row when the table has none); missing values (NaN)
are left unchanged and reported, as are table settings that differ from the
section (statistic, percentiles, ignored zero pixels). Changing the channel
of a linked placeholder afterwards refills its values from the same table.
The utility window closes (and the computation returns) only once its
worker has stopped, also after an error or a cancel. The FUCCI template links 
its background and min/max inputs to the `constant_*` values.

Any configuration may have an empty `channel` (no display channel). Loading
it asks for the channel to display (base layer), offering non-overlay channels
first, for both saved and new experiment inputs; an overlay chosen as base is
removed from the overlays. The choice becomes the loaded configuration's
`channel` and is kept when the configuration is saved.

Use ``Process measurements...`` (its tooltip summarizes what it does) to apply 
a recipe to selected measurements of the current position. Smooth measurement 
uses a centered moving average independently for each cell; Min-max 
normalization rescales with fixed constants shared by all cells; Ramp detection 
marks sustained rising (+1) and falling (-1) ramps of each cell's measurement. 
Accepting prompts for an output name and adds ``<measurement>_<name>`` 
columns without changing the originals. Results stay in memory across 
position switches; Reload table discards them. Processing runs in 
the background: on completion only the new output columns are merged into 
the position's current table (aligned on frame_i, Cell_ID), so time 
point/annotation edits made meanwhile are kept. The results are discarded 
with a warning when the position is no longer loaded, its table was 
reloaded from file, the processed input columns changed, or columns with the
output names were added meanwhile. A failed recipe leaves the table unchanged
and shows a warning with the traceback as details (also logged). One job runs
at a time: a recipe dialog accepted while a job runs (e.g. a second dialog),
or whose inputs are no longer in the table, is refused with a warning.
``Combine measurements...`` opens the combined-measurement equation editor.
Equations are evaluated on the current position's raw table as it is when OK
is clicked (edits made while the dialog is open included), including
categorical inputs, and add columns without overwriting existing measurements.
Combined columns stay in memory and are not saved to the global equation
configuration. Numerical and boolean results are available for plotting;
string/category results are retained in the exported table.
Combine-measurement equations can use each time point identifier as a
``timepoint_<name>`` boolean input. ``any(condition)`` and ``all(condition)``
reduce the condition across the cell's frames and repeat the result on each
of its rows.
``Export measurements...`` saves the table, including processed columns, to a
separate CSV. Processing is applied to raw measurements before plot normalization.

Right-click a plotted column and select ``Conditional formatting...`` to color
its time course with ordered rules ``{"condition": equation, "color": spec}``,
saved per column in the configuration's ``conditional_formatting``. The color
is ``"#rrggbb"`` or a shade of the column color: ``"lighter"`` /
``"darker"``, optionally with a QColor factor in percent (>= 100; defaults
150 and 200), e.g. ``"darker:250"``. Conditions are evaluated like predictor equations
on the selected cell's raw rows (per-cell functions and ``SizeT``; bool or
0/1, per frame or scalar; missing values are False). The first true rule
colors a point and the line segment to the next point; other frames keep the
column color. Hovering shows the matching condition. Evaluation errors are
shown in the status bar and the column is drawn unformatted. Formatting
applies to the time course only, not to the comparison mode. Editing time
points or custom annotations (manually or by a predictor) replots when an
edited column is plotted or read by a condition of a plotted column.

The ``Time points`` toolbar has ``Add time point identifier...`` to define a
named time point (e.g., "Bud emergence") with a color, a keyboard shortcut
(defaults Q, W, E, R, ...) and an optional per-cell predictor (an equation of
the table columns or ``predictor(df)`` from a Python file, see
``timepoint_predictors``). Each identifier gets its own toolbar action:
clicking it adds a time point for the selected cell at the current frame,
while hovering over the time-course plot and pressing its shortcut adds it
at the hovered frame (the shortcut does nothing, with a status bar hint,
when the plot is not hovered). Shortcut and click are told apart by the
action (`_ShortcutAwareAction`), never by the cursor position, so a click
works from anywhere (e.g., the toolbar extension menu); any other trigger
counts as a click. ``Shift+shortcut`` re-evaluates its predictor on the
selected cell. Shortcuts that already contain Shift get no such auto
shortcut, and an auto shortcut already used by another action (e.g., from an
older configuration) is left unset with a warning. A key is not available as
shortcut (in the dialogs and for the defaults) when it or its ``Shift+key``
is already used (e.g., S, because Shift+S shuffles the colormap); the same
rules apply to custom annotations. A time point identifier or custom
annotation whose own shortcut is already used by another action (possible
with 'Load annotation...', a configuration or a viewer tool key set in the
shortcuts editor or read from the shortcuts.ini shared with the main GUI,
which skip the dialog checks) gets neither shortcut, with a warning and a
tool tip note, because Qt triggers neither of two actions sharing a key. The
shortcuts editor lists their keys (not editable there,
`getHardcodedShortcuts`) to flag such collisions, and `setShortcuts` checks
them again (a key is set again once it is free). Time point lines can be
dragged like the
current frame line and are drawn above it (z-value 11 over 10, so a time
point added at the current frame is visible through the dashes; a drag at
that frame moves the time point, not the current frame);
right-click a line to remove it, or right-click an identifier action to edit,
re-evaluate its predictor, clear or remove it. Automatic predictions require
the identifier's ``autoApply`` flag and run only when a cell is first viewed.
Viewed Cell_IDs are tracked in the boolean ``measurements_viewer_visited``
column of each position's table and saved with Export measurements. The
visited, ``timepoint_*`` and custom annotation columns are parsed robustly
when a table is loaded or reloaded (`timepoint_predictors.parse_flags`): the
main GUI may save them as True/False/0 strings after adding cells, and only
true-like values ('True', non-zero numbers) count as marked. Revisiting
a cell does not retry failed predictions; use explicit re-evaluation to retry.
The toolbar's ``Re-evaluate predictors on all cells`` and ``... on displayed
cell`` re-run all predictors; cells with manual edits (added, moved or removed
time points) trigger a warning to overwrite or skip them. Identifier
definitions are stored in configurations. The time points themselves live in 
the position's acdc_df as ``timepoint_<name>`` boolean columns (True at the 
frames where the cell is marked), so they are part of any exported or processed 
table.

The ``Custom annotations`` toolbar shares the custom annotations of the main 
GUI (``gui_custom_annotations``): add, load, modify or remove annotations,
each saved as a 0/1 column ``<name>`` of the acdc_df (removing an annotation
together with its column warns about predictors, phase rules, formatting
conditions and recorded steps that read it, and drops the column's color
and conditional formatting). Each annotation has a
keyboard shortcut (defaults A, D, F, ...) that annotates/un-annotates the
selected cell across its entire video history, while ``Shift+shortcut``
re-evaluates its predictor on the selected cell. Alternatively, activate the
toolbar button and RIGHT-click on a cell. The symbols of all annotations are
drawn, also of inactive buttons: as in the main GUI, ``setData`` checks the
eye action ``Show all custom annotations`` (``viewAllCustomAnnotAction``);
unchecked, only the active annotation and those whose ``Hide annotations``
option (``isHideChecked``, kept per annotation name as the user setting
``<name>_annotation_hidden``, which wins over the configuration) is off are
drawn. Like time points, annotations can
have a predictor (equation or ``predictor(df)`` from a Python file, see 
``custom_annotation_predictors``) that must return booleans (one per row, 
or one for all rows of the cell). ``All`` (default), ``None`` or ``Most``
consolidates the results into one boolean for the entire Cell_ID history.
Most requires strictly more than half True; ties are False. Existing
frame-specific assignments are expanded to all frames if any frame is marked.
Adding or loading an annotation refuses names whose column is not a 0/1
flag column of a loaded table (e.g., a measurement) or is written by a time
point, another annotation, the phases, a recorded step or the visited
tracking (``customAnnotColumnConflict``), so no column is turned into 0/1.
The add/modify dialog checks it on OK and stays open with the entered values
(``columnConflict`` of `apps.customAnnotationDialog`; renaming checks only
the new name). For the same reason, loading a configuration or opening data (annotations
persist across loading) is refused, before any data is replaced, when one
of its annotations is named like a column of the new tables that is not a
flag column (``flag_column_conflicts``).
Definitions and predictor settings are saved in viewer configurations; the
main GUI's global saved definitions can be loaded, but are not overwritten
or deleted (the load dialogs have no delete button in the viewer).
Cell assignments are saved with Export measurements, not in configurations.
Automatic predictions (``autoApply``), the warning
before overwriting manually edited cells and the re-evaluation (right-click
on the annotation button) work as for the time points. In particular,
changing an annotation's predictor or enabling its auto apply, or
adding/loading an annotation whose
column already exists in a table, does not re-predict visited cells (their
saved or manual assignments are kept); only a brand-new column is also
predicted on visited cells. Use explicit re-evaluation to apply a changed
predictor to visited cells. The toolbar also has
actions to re-evaluate all annotation predictors on the displayed cell or on
all cells. Annotation definitions are stored in configurations.

Viewer-only option ``Overrides phases`` of a custom annotation (state and
configuration key ``overridesPhases``, strict bool, absent means false; the
main GUI's annotations never have it): a cell annotated with it (in at least
one analyzed row) gets the annotation's name, verbatim, as its phase on all
its analyzed frames, and the phase rules are not checked for it
(`phase_translator.evaluate` ``annotation_overrides``, from
`phaseOverrideAnnotations`). When several overriding annotations are set in
a cell, the first one in toolbar (configuration) order wins, e.g. the FUCCI
template lists "Excluded" before "Arrested". Each overriding annotation adds
its boolean ``phase_<name>`` output, and the annotation names take part in
the rules' input/output clash check (`validateConfiguration`). The button
tool tip says that the annotation overrides phases, and the removal prompt
notes that its cells get their phases from the rules again.

Predictor equations and their filters can use other identifiers' boolean
``timepoint_<name>`` columns and annotations' 0/1 ``<name>`` columns. Dependencies
are evaluated first, across both kinds of predictor. Explicit re-evaluation
also re-runs prerequisite predictors (with the same manual-edit protection);
automatic prediction still requires autoApply. Other-cell inputs evaluate
prerequisites on all cells. Automatic predictions never overwrite manually
edited cells (including those of prerequisites) and never show the
overwrite warning; cells already predicted are not recomputed. Cells visited
in an earlier session (visited column of a loaded table) count as predicted,
so a prerequisite predicted on all cells keeps their saved time points and
annotations; the same holds after changing a time point predictor or adding
an identifier whose column already exists in a table. Loading data or a
configuration resets this session state and rebuilds it from the visited
column of the loaded tables (e.g., reopening a position after 'Don't save'
predicts cells that are unvisited in the reloaded table). Circular
equation dependencies prevent accepting
the customization, with a warning showing the cycle. Python-file predictors
can read these columns, but their dependencies cannot be inferred.
Renaming a time point identifier or annotation is refused (with the list of
dependents) while other predictors, phase rule formulas, conditional
formatting conditions or recorded combine equations/processing steps read
its column, because references are not rewritten; edit those first. A
rename onto a name that is already a column of a loaded table (e.g., a
measurement, ``phase`` or a column kept by 'Remove only button') is refused,
because it would duplicate the column. On rename, the plot selection, color and
conditional formatting of the column follow the new name, and phase rule
boundaries/required annotations are renamed. Removing a time point
identifier or an annotation warns about such dependents and also about phase
rules that use its column as a boundary or required annotation (they are
skipped with 'missing inputs' afterwards); the column's own color and
conditional formatting are removed with it.

``Time point to phase translator...`` (icon-only button of the ``Phases``
toolbar, icon ``:phase_translator.svg``, tooltip explaining the steps; also in
the Annotations menu) opens an ordered list of phase rules.
Its ``Help...`` button explains the steps and links to the user guide section
(`urls.measurements_viewer_phase_translator_docs_url`).
Each rule reuses the custom annotation formula editor (including filters,
other-cell inputs and All/None/Most consolidation). A rule matches only if
its formula passes, every checked annotation occurs in the cell history,
and every checked time point occurs exactly once in the configured boundary
order. Otherwise the next rule is tried. Cells left unmatched because of
duplicate time points produce a warning. The first matching rule supplies
all phases for that cell; empty intervals do not fall through to later rules.
The right-hand list has phase names before, between and after its boundaries.
Intervals include their start and exclude their end; blank names leave frames
unassigned. Rules and boundaries can both be reordered.
Custom annotations with ``Overrides phases`` (see Custom annotations above)
are applied before the rules; the dialog receives their names
(``annotationOverrides``) and the rule list may be empty when at least one
annotation overrides phases (evaluation needs a rule or an overriding
annotation, `hasPhaseTranslation`).
Apply phase evaluation updates either the displayed cell or all cells of the
current position in memory. The ``Phases`` toolbar (and the Annotations menu)
also has ``Evaluate phases on displayed cell`` (icon
``:phase_translator_apply.svg``, shortcut ``Alt+P``, editable in the
shortcuts editor) to apply the configured
rules and overriding annotations to the displayed cell without opening the
dialog (status bar hint when neither is defined). With ``Automatically
evaluate phases when time points or annotations change or phases are
missing`` (saved in the configuration as ``auto_apply_phases``), every
change of time points on the current position (added, moved, removed,
cleared or predicted) re-evaluates the phases of the affected cells once the
change is complete, and so does every change of a custom annotation that
affects phases (`annotationAffectsPhases`: it overrides phases, is a rule's
required annotation or is read by a rule's eligibility formula) for the
cells whose value actually changed (toggled, cleared or predicted;
predictions of another position are evaluated when it is displayed).
Switching ``Overrides phases`` of an annotation, renaming or removing an
overriding annotation, or adding one whose column already has annotated
cells re-evaluates its annotated cells and the cells with its (old) name as
phase, on every loaded position; renaming also moves the custom color of
that phase to the new name (unless a rule assigns the old name). Loading a
configuration or replacing the annotations in bulk (`setCustomAnnotations`,
`reinitCustomAnnot`, `bulkSettingCustomAnnotations`) never re-evaluates, so
the loaded tables keep their saved phases. When neither phase rules nor
other overriding annotations are left after switching off or removing an
overriding annotation, nothing could evaluate its cells again: the phase
named after it is cleared instead, on every loaded position and also with
automatic evaluation off (`clearPendingPhases`, which evaluates the cells
without rules and overrides; the unused ``phase_<name>`` output is dropped,
and ``phase`` too once no cell keeps a phase, also when the table is
reloaded before saving: `restoreTimepointColumns`).
A displayed cell without
evaluated phases (no ``phase`` column or only empty phases) is evaluated once
per session (after its automatic predictors); errors and warnings then go to
the status bar. While measurement processing runs, these evaluations are
postponed until it finishes; cells of a position that is no longer displayed
by then are evaluated when that position is displayed again, or before its
table is saved (`flushPendingPhaseEvaluations` in `saveAcdcDfs`, also for
the other positions' cells of an annotation rename or removal). Save tables
and the ``Save changes?`` prompt first run the pending evaluations of every
loaded position, so that positions whose phases change are saved (or listed)
even if their tables had no other change. Export
measurements prompts to evaluate all cells
when rules or overriding annotations are configured, and exports ``phase`` (string, empty if unassigned)
and ``phase_<name>`` (boolean) columns. These output names are reserved by the
translator; existing unrelated measurements are not overwritten (a ``phase``
column without any phase name holds none and is overwritten). Rule
definitions, but not evaluated table values, are saved in viewer configurations.
Empty phases read back from a saved acdc_output file (which the loader fills
with 0 when the table has cell cycle columns) are restored to ``''``, and
``phase_<name>`` columns that the main GUI saved as True/False/0 strings after
adding cells are parsed back to bool, when the table is loaded or reloaded
(``phase_translator.normalize_loaded_outputs``).

When the table has a ``phase`` column, the selected cell's phases are drawn as
a solid colored band (one bar per run of consecutive frames, labeled with the
phase; hover for frame ranges) in a separate x-linked plot directly below the
time course, and Measurement vs Measurement colors each point by its phase
(gray for unassigned, with a legend). Hover tips show the phase. The
``Show phases`` checkbox toggles both; its state is stored in the settings CSV
(``measurements_viewer_show_phases``), like the navigation toggles ``Rescale
plot`` (``rescalePlot``) and ``Only frames with selected cell``
(``restrictFrames``). Every automatic rescaling of the measurements plot is
the same as clicking its "A" button (``autoscaleMeasurementsPlot``: x and y
auto range, y restricted to the visible x range with ``setAutoVisible``), and
the plot then keeps following the data (e.g. when changing cell or editing
the table) until the user zooms or pans by hand. The time course is
autoscaled whenever the set of plotted columns or their grouping on y axes
changes (column plotted, removed or moved to another y axis from the list
or its context menu, a configuration loaded), normalization is toggled, a
position is loaded or the view switches back from Measurement vs
Measurement. These autoscales wait until a plotted value is
finite; until then, without auto range, x shows frames 1..SizeT after loading
a position. While pyqtgraph's auto range is on (after an autoscale, or with
``Rescale plot`` checked) a cell whose plotted values are all NaN still fits
x to its frames, as "A" does, and y is unchanged.
``Rescale plot`` (checked) also autoscales when changing cell, even after a
manual zoom. Measurement vs Measurement autoscales the scatter plot when X,
Y or normalization change, a position is loaded or the view switches to it
(`_comparisonRangeKey`); other redraws (e.g. phases evaluated automatically
when selecting a cell) keep the zoom. Phase colors are edited in the
translator dialog's ``Phase colors`` box (committed on Apply or OK) and saved
in the configuration's ``phase_colors`` ``{name: "#rrggbb"}`` (only the colors
the user chose, `chosenPhaseColors`: a stored default could be repeated by the
default of a phase added later); phases without a
color get the colorblind-friendly, red-free defaults of
``phase_translator.phase_colors`` (red is reserved for the frame lines), in
the order of ``phase_translator.phase_names``: the rule phases first, then
the names of the annotations overriding phases (so these do not shift the
rule phase colors), then other phases found in the table (`phaseColorMap`).

User actions (loading/saving folders, files, configurations and tables, the
button chosen in the ``Save changes?`` prompt, plotting/removing columns, plot
mode, colors, conditional formatting, toggles, selected Cell ID, position,
time point and custom annotation edits, re-evaluations, measurement
processing/combining/export, phase translator) are logged to the terminal by
the ``cellacdc-measurements-viewer`` logger (`get_logger`), to help debugging
user issues. Frame scrolling and hovering are not logged.

The measurements viewer: a standalone window that plots per-cell time
courses of acdc_output columns, with time point identifiers, custom
annotations, phases and measurement processing.

- main_window/: the window class MeasurementsViewerWin (window.py) and
  its Viewer<Topic>GuiElements mixins.
- viewer_utils.py, position_data.py, annotation_data.py: the widget-free
  logic (MeasurementsViewerUtils and its bases) and the module functions
  (loading positions and tables, frame masks, logger).
- dialogs/, widgets.py, workers.py: dialogs, widgets and workers used
  only by the viewer.
- custom_annotation_predictors.py, phase_translator.py: the custom
  annotation predictors and the phase translation. The time point
  predictors stay in the top-level timepoint_predictors.py, which shared
  and library code also use.
- premade_configurations/: the premade configuration templates (JSON
  files).
- __main__.py: runs the viewer with python -m on this package,
  optionally loading a folder.

The window also composes the shared mixins of gui/mixins/. Nothing here
imports gui/main_window/, gui/dialogs/, gui/widgets/, gui/workers/ or
gui/helpers.py. This __init__ imports nothing eagerly: the names that
the former single-module viewer defined at this import path are served
lazily with a deprecation warning, for external code only (see
_compat.py).
"""

from cellacdc import _compat as _acdc_compat

__getattr__, __dir__ = _acdc_compat.package_compat(__name__)
