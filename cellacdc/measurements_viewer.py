"""Standalone window to navigate frames, z-slices and cell IDs of Cell-ACDC 
data while plotting the time-course of `acdc_output` table columns (measurements)
of the selected cell, or two measurements against each other for all cells and
frames of the current position. Hover over a point for measurement details;
click a point to navigate to its frame and zoom to its cell.
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
  and LUT.
- `gui_annotations`: display of contours, segm. masks, IDs, cell cycle and 
  lineage info, mother-daughter lines and object tracks.
- `gui_viewer_tools`: the other 'Viewer' mode tools (zoom, ruler, scale bar, 
  timestamp, colormaps, search ID, hover info, shortcuts, slideshow, 
  3D viewer, objects properties, cell cycle table and objects count).
- `gui_overlay`: overlay of other channels and of other segmentation files.
- `gui_custom_annotations`: custom annotations of the objects.

Open it from the launcher (Utilities > Measurements > Measurements viewer...)
or run it with `python -m cellacdc.measurements_viewer [folder_path]`.
After data is loaded (on the first event loop pass after `setData`, i.e.
after the rest of the load such as the overlay channels), the Cell ID
controls (label, spinbox and scrollbar at the bottom) blink and the status bar 
explains how to select the plotted cell. Ctrl+F focuses the search bar and 
blinks it, like the toolbar search
button.
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
the loaded data and their edits.
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
entry = all frames): saved step outputs used as existing values, or read
back on Reload table, are set to NaN after the smaller of that frame and the
current last analyzed frame, also in the "as on disk" snapshot (NaN saved in
tables with cell cycle columns is read back as 0; restoring it is not an
edit). Toggling the option does not recalculate earlier results (except
phases evaluated automatically, which are re-evaluated on all cells):
re-evaluate predictors and phases, or reload the configuration to
recalculate the measurement steps. Toggling is refused while data load or
measurement processing runs, and Combine measurements refuses to apply
equations if the option changed while its dialog was open. A reload that
changes the last analyzed frame is logged (with a hint when the steps were
calculated on fewer frames) and handled like a toggle.

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
parsed reliably (Python >= 3.12 fails, older versions split them), so the
rule is checked explicitly on every Python version (`isEquationName`) and
such channels are rejected for placeholders used inside backticks (`resolvePremadeTemplate`; the setup
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
itself is never modified.

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
and shows a warning with the traceback as details (also logged).
``Combine measurements...`` opens the combined-measurement equation editor.
Equations are evaluated on the current position's raw table, including
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
rules apply to custom annotations. A custom annotation whose own shortcut is
already used by another action (possible with 'Load annotation...' or a
configuration, which skip the dialog checks) gets neither shortcut, with a
warning and a tool tip note, because Qt triggers neither of two actions
sharing a key. Time point lines can be dragged like the
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
toolbar, icon drawn in code by `phase_translator_icon`, tooltip explaining the
steps; also in the Annotations menu) opens an ordered list of phase rules.
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
also has ``Evaluate phases on displayed cell`` (icon `phase_apply_icon`,
shortcut ``Alt+P``, editable in the shortcuts editor) to apply the configured
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
without rules and overrides; the unused ``phase_<name>`` output is dropped).
A displayed cell without
evaluated phases (no ``phase`` column or only empty phases) is evaluated once
per session (after its automatic predictors); errors and warnings then go to
the status bar. While measurement processing runs, these evaluations are
postponed until it finishes; cells of a position that is no longer displayed
by then are evaluated when that position is displayed again, or before its
table is saved (`flushPendingPhaseEvaluations` in `saveAcdcDfs`, also for
the other positions' cells of an annotation rename or removal). Export
measurements prompts to evaluate all cells
when rules or overriding annotations are configured, and exports ``phase`` (string, empty if unassigned)
and ``phase_<name>`` (boolean) columns. These output names are reserved by the
translator; existing unrelated measurements are not overwritten. Rule
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
autoscaled whenever the set of plotted columns changes (column plotted or
removed from the list or its context menu, a configuration loaded),
normalization is toggled, a position is loaded or the view switches back from
Measurement vs Measurement. These autoscales wait until a plotted value is
finite; until then, without auto range, x shows frames 1..SizeT after loading
a position. While pyqtgraph's auto range is on (after an autoscale, or with
``Rescale plot`` checked) a cell whose plotted values are all NaN still fits
x to its frames, as "A" does, and y is unchanged.
``Rescale plot`` (checked) also autoscales when changing cell, even after a
manual zoom. Measurement vs Measurement always autoscales the scatter plot
(e.g., when X or Y changes). Phase colors are edited in the
translator dialog's ``Phase colors`` box (committed on Apply or OK) and saved
in the configuration's ``phase_colors`` ``{name: "#rrggbb"}``; phases without a
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
"""
import os
import sys
import copy
import html
import json
import keyword
import re
import logging
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd

import skimage.measure

from qtpy.QtCore import (
    Qt, Signal, QTimer, QPoint, QEvent, QRect, QSaveFile, QIODevice,
    QThread, QMutex, QWaitCondition, QObject, QEventLoop, QCoreApplication,
    QPointF, QRectF
)
from qtpy.QtGui import (
    QFont, QIcon, QGuiApplication, QPixmap, QPainter, QCursor, QKeySequence,
    QColor, QMouseEvent, QPen, QPolygonF
)
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QLabel, QGridLayout, QVBoxLayout, QHBoxLayout,
    QGroupBox, QSplitter, QLineEdit,
    QCheckBox, QComboBox, QAbstractItemView, QFileDialog, QAction,
    QApplication, QToolButton, QPushButton, QActionGroup, QAbstractSlider,
    QWidgetAction, QMenu, QColorDialog, QStyleOptionViewItem, QTreeView,
    QAbstractButton
)

import pyqtgraph as pg

from . import widgets, core, load, myutils, colors, transformation, qutils
from . import urls
from . import (
    graphLayoutBkgrColor, darkBkgrColor, settings_csv_path, settings_folderpath
)
from . import cca_df_colnames, base_cca_dict
from . import regionprops as acdc_regionprops
from . import gui_navigation, gui_annotations, gui_viewer_tools, gui_overlay
from .config import STANDARD_MOUSE_BUTTONS
from . import gui_core
from . import apps
from . import html_utils
from . import workers
from . import timepoint_predictors
from . import custom_annotation_predictors
from . import apps_measurements_viewer
from .apps_measurements_viewer import (
    ChannelPlaceholderMappingDialog,
    ConditionalFormattingDialog,
    _FormattingConditionDialog,
    formatting_color,
    is_formatting_color,
)
from . import phase_translator, measurement_processing, background_values
from . import gui_custom_annotations
from .gui_custom_annotations import (
    custom_annot_state_to_save, custom_annot_symbol
)
from .custom_annotation_predictors import CustomAnnotationPredictor
from .timepoint_predictors import (
    TimepointIdentifier, TIMEPOINT_PREFIX, parse_flags
)
from . import disableWindow
from .measurements_viewer_utils import (
    MeasurementsViewerUtils, RESCALE_INTENS_HOW_TEXTS, VISITED_COLUMN,
    PREMADE_TEMPLATE_KEYS, SIGNAL_COLORS, LIGHT_BACKGROUND_SIGNAL_COLORS,
    EXTRA_SIGNAL_COLORS, _SIGNAL_FALLBACK_HUES, signal_color, LOGGER_NAME,
    _StdoutLogHandler, _LogFormatter, get_logger, get_images_paths,
    get_channels_and_segm_endnames, get_channel_from_filepath, load_position,
    normalize_loaded_acdc_df, LAST_ANALYZED_CCA, LAST_ANALYZED_TRACKED,
    last_analyzed_frame, analysis_last_frame, mask_frames, min_frame,
    float_if_bool, position_key, blank_unanalyzed_outputs,
    evaluate_masked_cell_equation, update_last_analyzed_frame,
    acdc_df_file_stamp, acdc_df_snapshot, position_name, load_positions,
    precompute_position_regionprops, preload_overlay_channels,
    load_display_positions,
)


pg.setConfigOption('imageAxisOrder', 'row-major')

_font = QFont()
_font.setPixelSize(11)

DEFAULT_MEASUREMENTS_COLUMNS = ('cell_area_pxl', 'cell_vol_vox', 'cell_area_um2')
INDEX_COLUMNS = ('frame_i', 'Cell_ID', 'time_seconds')
CCA_DF_INT_COLS = [
    col for col in cca_df_colnames if type(base_cca_dict[col]) == int
]
MEASUREMENTS_CONFIGURATIONS_PATH = os.path.join(
    settings_folderpath, 'measurements_viewer_configurations'
)
PREMADE_CONFIGURATIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'measurements_viewer_premade_configurations'
)
PREMADE_TAG = '[Premade] '
# Phase colors of the phase translator icon (Okabe-Ito sky blue, bluish
# green and orange: no red, readable on light and dark toolbars)
PHASE_TRANSLATOR_ICON_COLORS = ('#56B4E9', '#009E73', '#E69F00')


def phase_translator_icon(size=64):
    """Icon of the time point to phase translator drawn in code (no resource
    file): a phase band of three colored segments split by two time point
    ticks with a marker on top. The ticks are mid-gray, visible on both the
    light and the dark color scheme."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    margin = size / 16
    width = size - 2*margin
    bounds = (
        margin, margin + width*0.36, margin + width*0.66, size - margin
    )
    bandTop, bandHeight = size*0.4, size*0.32
    painter.setPen(QPen(Qt.PenStyle.NoPen))
    for x0, x1, color in zip(
            bounds[:-1], bounds[1:], PHASE_TRANSLATOR_ICON_COLORS
        ):
        painter.setBrush(QColor(color))
        painter.drawRect(QRectF(x0, bandTop, x1 - x0, bandHeight))
    tickColor = QColor(128, 128, 128)
    tickPen = QPen(tickColor)
    tickPen.setWidthF(size / 14)
    tickPen.setCapStyle(Qt.PenCapStyle.RoundCap)
    markerY, radius = size*0.2, size/9
    for x in bounds[1:-1]:
        painter.setPen(tickPen)
        painter.drawLine(QPointF(x, markerY), QPointF(x, size*0.86))
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(tickColor)
        painter.drawEllipse(QPointF(x, markerY), radius, radius)
    painter.end()
    return QIcon(pixmap)


def phase_apply_icon(size=64):
    """Icon of "Evaluate phases on displayed cell" drawn in code: the phase
    band of `phase_translator_icon` below a play triangle (mid-gray, visible
    on both color schemes)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    margin = size / 16
    width = size - 2*margin
    bounds = (
        margin, margin + width*0.36, margin + width*0.66, size - margin
    )
    bandTop, bandHeight = size*0.62, size*0.28
    painter.setPen(QPen(Qt.PenStyle.NoPen))
    for x0, x1, color in zip(
            bounds[:-1], bounds[1:], PHASE_TRANSLATOR_ICON_COLORS
        ):
        painter.setBrush(QColor(color))
        painter.drawRect(QRectF(x0, bandTop, x1 - x0, bandHeight))
    painter.setBrush(QColor(128, 128, 128))
    painter.drawPolygon(QPolygonF([
        QPointF(size*0.34, size*0.06), QPointF(size*0.34, size*0.52),
        QPointF(size*0.70, size*0.29)
    ]))
    painter.end()
    return QIcon(pixmap)


APPLY_PHASES_CELL_SHORTCUT = 'Alt+P'

MASK_UNANALYZED_TOOLTIP = (
    'Ignore frames after the last analyzed frame\n\n'
    'When checked, the calculations of each position use only the frames up to\n'
    'its last analyzed frame in the main GUI: the last cell cycle annotated\n'
    'frame, or the last tracked frame (last frame of the acdc_output table) when\n'
    'there are no cell cycle annotations. Time point and custom annotation\n'
    'predictors, phases, measurement processing, combined measurements and\n'
    'conditional formatting ignore the later frames (the "SizeT" of equations is\n'
    'the number of analyzed frames) and leave their outputs empty there. Images\n'
    'and plots still show all frames (ignored frames are shaded in the plot).\n\n'
    'Results calculated before changing this option are not recalculated\n'
    'automatically (except phases evaluated automatically): re-evaluate\n'
    'the predictors and the phases, or reload the configuration to\n'
    'recalculate the measurement steps. Saved in the configuration.'
)


PHASE_TRANSLATOR_TOOLTIP = (
    'Time point to phase translator\n\n'
    'Turn the time points of each cell into phases (e.g., "G1", "S/G2/M"):\n'
    'the frames between two consecutive time points get the same phase name.\n\n'
    'How to use it:\n'
    '1. Define time point identifiers ("Add time point identifier...")\n'
    '   and mark them on the cells (hover over the plot and press\n'
    '   their shortcut, or let their predictors find them).\n'
    '2. Open the translator and add a rule: check the time points that '
    'separate\n'
    '   the phases, put them in their order and name the phases before, '
    'between\n'
    '   and after them (optionally add a formula and required custom '
    'annotations).\n'
    '3. Click "Apply phase evaluation" on the displayed cell or on all cells\n'
    '   (or let the phases be evaluated automatically). Later, the "Evaluate\n'
    '   phases on displayed cell" button next to this one (shortcut "'
    + APPLY_PHASES_CELL_SHORTCUT + '")\n'
    '   re-evaluates the displayed cell without opening the translator.\n'
    '4. The phases are drawn as a colored band below the time course\n'
    '   plot and color the Measurement vs Measurement points. "Export\n'
    '   measurements..." saves the "phase" and "phase_<name>" columns.\n\n'
    'Rules are tried from top to bottom; the first one that matches a cell\n'
    'supplies its phases. Custom annotations with the "Overrides phases"\n'
    'option come first: an annotated cell gets the annotation name as\n'
    'phase on all its frames. Save the configuration to keep the rules.\n'
    'Details: Help --> Measurements viewer user guide.'
)

MEASUREMENTS_VIEWER_QUICK_START = (
    'The measurements viewer plots the measurements (columns of the '
    '<code>acdc_output</code> table) of the selected cell over time.'
    '<br><br><b>How to use it:</b>'
    + html_utils.to_list([
        'Load data with <code>File --> Load Folder...</code> (or load a '
        'saved configuration at startup).',
        'Select the cell with the <b>Cell ID</b> controls at the bottom or '
        'by clicking on it in the image; the frame scrollbar moves the '
        'current frame (red line on the plot).',
        'Click a column in <b>Columns to plot</b> to plot or remove its '
        'time course; right-click it to change its color or add '
        'conditional formatting. <b>Measurement vs Measurement</b> plots '
        'two columns against each other for all cells.',
        'Mark time points (<b>Time points</b> toolbar) and custom '
        'annotations, and turn time points into phases with the '
        '<b>Time point to phase translator</b>.',
        '<b>Process measurements...</b> and <b>Combine measurements...</b> '
        'add new columns; <b>Export measurements...</b> saves them. '
        '<code>File &gt; Save tables</code> (<code>Ctrl+S</code>) saves the '
        'acdc_output tables. Save the configuration (<code>File</code> menu, '
        '<code>Ctrl+Shift+S</code>) to reuse the setup on other data.',
    ], ordered=True)
    + 'Press <code>Ctrl+F</code> to search for a Cell ID or a '
    'functionality.'
)


def auto_shortcut_text(shortcut):
    """Key sequence (text) of the ``Shift+<shortcut>`` action that
    re-evaluates the predictor of a time point identifier or custom
    annotation with `shortcut`.

    Empty when `shortcut` is empty or already contains Shift (it would
    collide with the shortcut itself).
    """
    if isinstance(shortcut, QKeySequence):
        shortcut = shortcut.toString()
    text = QKeySequence(shortcut or '').toString()
    if not text or 'Shift' in text:
        return ''
    return QKeySequence(f'Shift+{text}').toString()


class _ShortcutAwareAction(QAction):
    """QAction that tells whether `triggered` is emitted by its shortcut.

    `triggeredByShortcut` is True only while the action handles the
    ``QEvent.Shortcut`` sent by Qt's shortcut map (i.e., during the
    `triggered` slots of a key press), and False for a click on its toolbar
    button (also from the toolbar extension) or `trigger()`. The time point
    actions use it to add a time point at the hovered plot frame (shortcut)
    or at the current frame (click), independently of the cursor position.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.triggeredByShortcut = False

    def event(self, event):
        if event.type() != QEvent.Shortcut:
            return super().event(event)
        self.triggeredByShortcut = True
        try:
            return super().event(event)
        finally:
            self.triggeredByShortcut = False


class _TimepointLine(pg.InfiniteLine):
    """Draggable line of a time point of the selected cell."""
    sigRemoveRequested = Signal(object)

    def __init__(self, identifier, ID, frame_i):
        color = identifier.color
        super().__init__(
            pos=frame_i+1, angle=90, movable=True,
            pen=pg.mkPen(color=color, width=2, style=Qt.DashLine),
            hoverPen=pg.mkPen(color=color, width=4),
            labelOpts={
                'position': 0.95, 'color': color, 'movable': True,
                'fill': (0, 0, 0, 120)
            }
        )
        self.uid = identifier.uid
        self.ID = ID
        self.frame_i = frame_i
        self.setToolTip(qutils.wrap_tooltip(
            f'{identifier.name} (Cell ID {ID}, frame n. {frame_i+1})\n\n'
            'Drag to move. Right-click to remove.'
        ))
        # Above the current frame line (10), else a time point added at the
        # current frame (button click) is hidden under it. The dash gaps keep
        # the red line visible; dragging there moves the time point (user
        # choice over an invisible new time point).
        self.setZValue(11)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton and not self.moving:
            ev.accept()
            self.sigRemoveRequested.emit(self)
            return
        super().mouseClickEvent(ev)

class _MeasurementColumnDelegate(widgets._ButtonSearchHighlightDelegate):
    indicatorWidth = 28

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if not index.flags() & Qt.ItemIsUserCheckable:
            return
        option.features &= ~(
            QStyleOptionViewItem.HasCheckIndicator
            | QStyleOptionViewItem.HasDecoration
        )
        option.icon = QIcon()
        option.rect.adjust(self.indicatorWidth, 0, 0, 0)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if not index.flags() & Qt.ItemIsUserCheckable:
            return
        rect = QRect(
            option.rect.left()+2, option.rect.center().y()-8, 24, 16
        )
        icon = index.data(Qt.DecorationRole)
        painter.save()
        if icon is not None and not icon.isNull():
            icon.paint(painter, rect, Qt.AlignCenter)
        else:
            painter.setPen(option.palette.mid().color())
            painter.drawRect(rect.adjusted(4, 2, -4, -2))
        painter.restore()

    def editorEvent(self, event, model, option, index):
        if not index.flags() & Qt.ItemIsUserCheckable:
            return False
        if event.type() in (
                QEvent.MouseButtonPress, QEvent.MouseButtonDblClick,
                QEvent.MouseButtonRelease):
            # The whole row (indicator and name) toggles the selection
            if event.button() != Qt.LeftButton:
                return False
            if not option.rect.contains(event.pos()):
                return False
            if event.type() != QEvent.MouseButtonRelease:
                return True
        elif event.type() == QEvent.KeyPress:
            if event.key() not in (Qt.Key_Space, Qt.Key_Select):
                return False
        else:
            return False
        state = (
            Qt.Unchecked
            if Qt.CheckState(index.data(Qt.CheckStateRole)) == Qt.Checked
            else Qt.Checked
        )
        return model.setData(index, state, Qt.CheckStateRole)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setWidth(size.width()+self.indicatorWidth)
        size.setHeight(max(size.height(), 20))
        return size

class _ThreadTaskWorker(QObject):
    """Run `func(*args, **kwargs)` in a worker `QThread` (see
    `MeasurementsViewerWin.runInThread`). The return value or the raised
    exception is stored in `result` / `error`, to be read once the thread
    finished. With `reportProgress`, `func` also gets the keyword argument
    ``progress``, a callable that emits `sigProgress` with a text."""
    sigProgress = Signal(str)

    def __init__(self, func, args, kwargs, reportProgress=False):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = dict(kwargs)
        if reportProgress:
            self.kwargs['progress'] = self.sigProgress.emit
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.func(*self.args, **self.kwargs)
        except BaseException as err:
            self.error = err
        finally:
            # Back to the GUI thread, so that `deleteLater` works after this
            # thread stopped
            app = QCoreApplication.instance()
            if app is not None:
                self.moveToThread(app.thread())
            QThread.currentThread().quit()

class _NoOpItem:
    """Placeholder for main GUI items that do not exist in the viewer 
    (e.g., the right image LUT). Every method call is a no-op."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class _NoAutoPilotProfile:
    """The viewer does not record auto-pilot profiles like the main GUI"""
    def storeSelectedChannel(self, *args, **kwargs):
        pass
    
    def storeClickMessageBox(self, *args, **kwargs):
        pass
    
    def storeLoadedFluoChannels(self, *args, **kwargs):
        pass

class MeasurementsViewerWin(
        QMainWindow,
        MeasurementsViewerUtils,
        gui_navigation.FramesNavigationGuiElements,
        gui_navigation.ZsliceNavigationGuiElements,
        gui_navigation.ZoomToObjGuiElements,
        gui_navigation.RecentPathsGuiElements,
        gui_navigation.LutGuiElements,
        gui_annotations.AnnotationsGuiElements,
        gui_custom_annotations.CustomAnnotationsGuiElements,
        gui_viewer_tools.ZoomGuiElements,
        gui_viewer_tools.RulerGuiElements,
        gui_viewer_tools.ScaleBarTimestampGuiElements,
        gui_viewer_tools.ColorSchemeGuiElements,
        gui_viewer_tools.LabelsColormapGuiElements,
        gui_viewer_tools.ImagePropertiesGuiElements,
        gui_viewer_tools.CcaTableGuiElements,
        gui_viewer_tools.CountObjectsGuiElements,
        gui_viewer_tools.WorkersGuiElements,
        gui_viewer_tools.SearchIDGuiElements,
        gui_viewer_tools.HoverInfoGuiElements,
        gui_viewer_tools.ShortcutsGuiElements,
        gui_viewer_tools.SlideshowGuiElements,
        gui_viewer_tools.Volume3DViewerGuiElements,
        gui_viewer_tools.RegionPropsGuiElements,
        gui_viewer_tools.BottomLayoutGuiElements,
        gui_viewer_tools.DefaultSettingsGuiElements,
        gui_overlay.OverlayChannelsGuiElements,
        gui_overlay.OverlayLabelsGuiElements,
        gui_overlay.ChannelImageHelpersGuiElements,
        gui_core.GUICoreElements,
    ):
    """Window to view images, navigate cells and plot their measurements.

    Parameters
    ----------
    data : list of `load.loadData`, optional
        Loaded positions (see `load_positions`). Each `posData` must have 
        `img_data` and `segm_data` with the time axis first. If None, the 
        window starts empty and data can be loaded from `File` menu.
    pos_i : int, optional
        Index of the position to display at startup. Default is 0.
    frame_i : int, optional
        Index of the frame to display at startup. Default is `posData.frame_i` 
        if present, otherwise 0.
    ID : int, optional
        Cell ID to select at startup. Default is the first ID.
    isSegm3D : bool, optional
        Whether the segmentation masks are 3D. If None, it is inferred from 
        the shape of `posData.segm_data`.
    invertBw : bool, optional
        If True, invert black/white (light background) as with the action 
        `Invert black/white` of the main GUI. Default is the value saved 
        in the settings.
    """
    sigClosed = Signal(object)
    # Columns of the Navigate grid (`gui_createNavigationWidgets`): label,
    # spin box, size label ('/N'), slider, then the extras (z-projection
    # combobox, ...) up to `NAV_NUM_COLS`
    NAV_LABEL_COL = 0
    NAV_SPINBOX_COL = 1
    NAV_SIZE_COL = 2
    NAV_SLIDER_COL = 3
    NAV_EXTRA_COL = 4
    NAV_NUM_COLS = 7
    # The bottom layout is different from guiWin's --> separate setting
    bottomLayoutHeightSettingKey = 'measurements_viewer_bottomLayoutHeight'
    # No delete-object tool --> hide its row in the shortcut editor
    showDeleteObjectShortcut = False

    def __init__(
            self, data=None, pos_i=0, frame_i=None, ID=None, isSegm3D=None,
            invertBw=False, parent=None
        ):
        super().__init__(parent)
        self.data = []
        self.pos_i = pos_i
        self.isSegm3D = isSegm3D
        self.isBwInverted = invertBw
        
        self.IDs = []
        self.selectedID = None
        self._IDframesCache = {}
        self.acdc_df = None
        self._lab = None
        self._labKey = None
        self.isDataLoaded = False
        self.user_ch_name = None
        self.configurationFilepath = None
        self.savedConfiguration = None
        self._startupConfigurationPrompted = False
        self._geometryRestored = False
        self.measurementProcessingThread = None
        self.measurementProcessingWorker = None
        # True while a load (`loadConfiguration`/`loadImagesPaths`, with
        # their dialogs) runs: other loads and closing are refused
        self._isLoadingData = False
        # True while the "Save changes before closing?" prompt is open:
        # loading is refused (`loadingInProgress`)
        self._isClosingPrompt = False
        self.rescaleIntensChannelHowMapper = {}
        self.timepointIdentifiers = []
        # Phase rules in descending priority; custom annotations with the
        # "Overrides phases" option are checked before them
        # (`phaseOverrideAnnotations`)
        self.phaseTranslatorRules = []
        # True while custom annotations are added/removed in bulk
        # (`bulkSettingCustomAnnotations`): no phase re-evaluation
        self._isBulkSettingCustomAnnotations = False
        # User-chosen phase colors {phase name: '#rrggbb'}
        self.phaseColors = {}
        self.timepointActions = {}
        self.timepointAutoActions = {}
        self.timepointLines = []
        self.timepointAutoPredictions = defaultdict(set)
        self.configurationInputs = []
        # inputOptionKey of the loaded (or last saved) input option, which
        # `configuration()` updates in place
        self.loadedInputKey = None
        self.measurementProcessingSteps = []
        self.combinedMeasurementSteps = []
        # Per-cell constants [{'name': str, 'equation': str}]
        # self.cellConstants = []
        self.logger = get_logger()
        self.loadSettings()
        # Re-evaluate phases of cells whose time points or phase-relevant
        # custom annotations change or whose phases are missing (saved in
        # the configuration)
        self.autoApplyPhases = False
        # Ignore frames after the last analyzed frame of each position in
        # all calculations (saved in the configuration; see
        # `analysisLastFrame`)
        self.maskUnanalyzedFrames = True
        # {`position_key`: last frame index} of the positions whose recorded
        # measurement step outputs were calculated only up to that frame
        # (NaN after it, read back as 0 from tables with cell cycle
        # columns); saved in the configuration
        self.measurementStepsLastFrames = {}
        # {id(posData): (posData, set of Cell_IDs)} awaiting phase auto-apply
        self._pendingPhaseIDs = {}
        self._phaseAutoApplyScheduled = False
        # {id(posData): (posData, set of phase names)} to clear because the
        # annotations overriding them are gone and nothing else can evaluate
        # phases (`schedulePhaseOverrideCells`)
        self._pendingPhaseClears = {}
        self._phaseClearScheduled = False
        self.initAnnotationsAttributes()
        self.initViewerToolsAttributes()
        self.initCustomAnnotationsAttributes()
        if sys.platform.startswith('win'):
            self.openFolderText = 'Show in Explorer...'
        else:
            self.openFolderText = 'Reveal in Finder...'
        
        self.setWindowTitle('Cell-ACDC - Measurements viewer')
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
        
        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()
        self.gui_connectActions()
        self.gui_createGraphicsPlots()
        self.gui_createNavigationWidgets()
        self.gui_createAnnotationsWidgets()
        self.gui_createMeasurementsPanel()
        self.gui_createLayout()
        self.gui_createStatusBar()
        self.gui_connectSignals()
        self.gui_connectViewerToolsSignals()
        self.initShortcuts()
        self.readRecentPaths()
        self.postInitLoadSetSettings()
        
        self.setData(data, pos_i=pos_i, frame_i=frame_i, ID=ID)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._startupConfigurationPrompted:
            self._startupConfigurationPrompted = True
            QTimer.singleShot(0, self.loadConfiguration)
        # Restore the geometry saved on close only once: Qt also sends
        # (spontaneous) show events when the window is restored from minimized
        if not self._geometryRestored:
            self._geometryRestored = True
            self.readSettings(setting_suffix='_measurements_viewer')


    def selectConfigurationFilepath(self):
        from .apps import QTreeDialog

        paths = {}
        types = {}
        for folder, prefix, kind in (
                (MEASUREMENTS_CONFIGURATIONS_PATH, '', 'Saved'),
                (PREMADE_CONFIGURATIONS_PATH, PREMADE_TAG, 'Premade')):
            if not os.path.isdir(folder):
                continue
            for filename in sorted(os.listdir(folder)):
                filepath = os.path.join(folder, filename)
                if filename.lower().endswith('.json') and os.path.isfile(filepath):
                    paths[f'{prefix}{filename}'] = filepath
                    types[f'{prefix}{filename}'] = kind

        browseButton = widgets.browseFileButton(
            'Browse configuration...', self,
            ext={'JSON': ['.json']},
            title='Load measurements viewer configuration',
            start_dir=MEASUREMENTS_CONFIGURATIONS_PATH,
        )
        skipButton = widgets.SkipPushButton('Skip/New configuration', self)
        dialog = QTreeDialog(
            [
                (name, datetime.fromtimestamp(os.path.getmtime(path)).strftime(
                    '%Y/%m/%d %H:%M'
                ), types[name])
                for name, path in paths.items()
            ],
            headerLabels=['Name', 'Date Modified', 'Type'],
            title='Load measurements viewer configuration',
            infoText='Select a configuration to load, or skip loading:',
            parent=self,
            path_to_browse=MEASUREMENTS_CONFIGURATIONS_PATH if paths else None,
            additional_buttons=(browseButton,skipButton),
        )
        dialog.treeWidget.setCurrentItem(dialog.treeWidget.topLevelItem(0))
        browsed = []

        def selected(path):
            if path:
                browsed.append(path)
                dialog.close()

        browseButton.sigPathSelected.connect(selected)
        skipButton.clicked.connect(lambda: setattr(dialog, 'cancel', True) or dialog.close())
        dialog.exec_()
        if browsed:
            return browsed[0]
        if dialog.cancel:
            return None
        return paths[dialog.selectedText]

    def configuration(self):
        posData = self.posData()
        segm_filename = os.path.splitext(
            os.path.basename(posData.segm_npz_path)
        )[0]
        segm_endname = segm_filename[len(posData.basename):].lstrip('_')
        overlay_channels = (
            sorted(self.checkedOverlayChannels)
            if self.overlayButton.isChecked() else []
        )
        visible_channels = [
            channel for channel, button in self.allOverlayToolbuttons.items()
            if button.isChecked()
            and (channel == self.user_ch_name or channel in overlay_channels)
        ] or [self.user_ch_name]
        currentInput = {
            'images_paths': [
                os.path.abspath(pos.images_path) for pos in self.data
            ],
            'channel': self.user_ch_name,
            'segm_endname': segm_endname,
            'overlay_channels': overlay_channels,
            'visible_channels': visible_channels,
        }
        inputs = copy.deepcopy(self.configurationInputs)
        # Same experiment (same Images folders, channel and segmentation),
        # possibly with another channel order or other overlay/visible
        # channels: update that option in place instead of adding a
        # duplicate. An option equal up to channel order is preferred, then
        # the loaded option (several options may share the experiment).
        currentKey = self.inputOptionKey(currentInput)
        currentIdentity = self.inputIdentityKey(currentInput)
        loadedKey = self.loadedInputKey
        matches = [
            i for i, option in enumerate(inputs)
            if self.inputOptionKey(option) == currentKey
        ] or [
            i for i, option in enumerate(inputs)
            if loadedKey is not None
            and self.inputOptionKey(option) == loadedKey
            and self.inputIdentityKey(option) == currentIdentity
        ] or [
            i for i, option in enumerate(inputs)
            if self.inputIdentityKey(option) == currentIdentity
        ]
        if matches:
            inputs[matches[0]] = copy.deepcopy(currentInput)
        else:
            inputs.append(currentInput)
        return {
            'version': 2,
            **currentInput,
            'input_options': inputs,
            'measurement_processing': copy.deepcopy(self.measurementProcessingSteps),
            'combined_measurements': copy.deepcopy(self.combinedMeasurementSteps),
            'measurements_mode': self.measurementsModeCombobox.currentIndex(),
            'time_course_columns': (
                self.checkedColumns() if self._measurementsMode == 0
                else sorted(self._timeCourseColumns)
            ),
            'measurements_axes': self.measurementsAxes.copy(),
            'normalize_measurements': self.normalizeCheckbox.isChecked(),
            'measurements_colors': {
                col: pg.mkColor(color).name()
                for col, color in self.measurementsColors.items()
            },
            'conditional_formatting': copy.deepcopy(self.conditionalFormatting),
            'timepoint_identifiers': [
                identifier.to_dict() for identifier in self.timepointIdentifiers
            ],
            'custom_annotations': self.customAnnotConfiguration(),
            'phase_translator_rules': copy.deepcopy(self.phaseTranslatorRules),
            'phase_colors': dict(self.phaseColors),
            'auto_apply_phases': self.autoApplyPhases,
            'mask_unanalyzed_frames': self.maskUnanalyzedFrames,
            'measurement_steps_last_frames': dict(
                self.measurementStepsLastFrames
            ),
            # 'cell_constants': [dict(constant) for constant in self.cellConstants],
        }

    def configurationError(self, operation, error):
        self.logger.error('Could not %s configuration: %s', operation, error)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Configuration failed',
            f'Could not {operation} configuration:\n\n{error}'
        )

    def saveConfiguration(self, checked=False, save_as=False):
        if self.measurementProcessingThread is not None:
            self.configurationError('save', 'Wait for measurement processing to finish.')
            return False
        if not self.isDataLoaded:
            self.configurationError('save', 'Load image data before saving.')
            return False
        filepath = self.configurationFilepath
        if save_as or filepath is None:
            try:
                os.makedirs(MEASUREMENTS_CONFIGURATIONS_PATH, exist_ok=True)
            except OSError as err:
                self.configurationError('save', err)
                return False
            current_files = sorted(os.listdir(MEASUREMENTS_CONFIGURATIONS_PATH))
            helpText = html_utils.paragraph(
                '''Select a filename for the measurements viewer configuration.
                Which allows you to later reload it, preserving displayed measurements.'''
                )
            date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            msg = apps.filenameDialog(
                ext='json', title='Save measurements viewer configuration as',
                parent=self, helpText=helpText,existingNames=current_files,
                allowEmpty=False, addDoNotSaveButton=True, defaultEntry=date
            )
            msg.exec_()
            if msg.cancel:
                return False
            if msg.doNotSave:
                return True
            filename = msg.filename
            if not filename:
                return False
            filepath = os.path.join(MEASUREMENTS_CONFIGURATIONS_PATH, filename)
            if not filepath.lower().endswith('.json'):
                filepath += '.json'
            # fileNameDialog handles overwrite confirmation internally
        try:
            configuration = self.configuration()
            self.validateConfiguration(configuration)
            self.writeConfigurationFile(filepath, configuration)
        except (OSError, ValueError, TypeError, AttributeError) as err:
            self.configurationError('save', err)
            return False
        self.configurationFilepath = filepath
        self.configurationInputs = copy.deepcopy(configuration['input_options'])
        self.loadedInputKey = self.inputOptionKey(configuration)
        self.savedConfiguration = configuration
        self.logger.info(f'Configuration saved to "{filepath}"')
        self.statusBar().showMessage(f'Configuration saved to "{filepath}".', 5000)
        return True

    def saveConfigurationAs(self, checked=False):
        return self.saveConfiguration(save_as=True)

    def selectChannelMapping(
            self, placeholders, channels, inputs=None, validate=None,
            backgroundValues=None, experimentPositions=None,
            channelHints=None
        ):
        """Return `(channel mapping, input values)` or None if cancelled.
        `validate(mapping, values)` may raise ValueError to refuse a
        selection (message shown, dialog stays open). With the template's
        validated `backgroundValues` section, the linked inputs can be
        filled by running the background utility on the Positions of
        `experimentPositions` (`{exp_path: [pos_foldername, ...]}`, see
        `computeBackgroundSummary`) or from a saved summary table."""
        experimentPositions = experimentPositions or {}
        compute = None
        if backgroundValues is not None:
            compute = partial(
                self.computeBackgroundSummary, experimentPositions
            )
        dialog = ChannelPlaceholderMappingDialog(
            placeholders, channels, parent=self, inputs=inputs,
            validate=validate, backgroundValues=backgroundValues,
            computeBackgroundValues=compute,
            tableStartDir=next(iter(experimentPositions), ''),
            channelHints=channelHints
        )
        dialog.exec_()
        if dialog.cancel:
            return None
        return dialog.mapping, dialog.values

    def computeBackgroundSummary(
            self, experimentPositions, channels, settings, parent=None
        ):
        """Run the "Compute background intensity" utility on the
        experiments of `experimentPositions` (`{exp_path: [preselected
        Positions]}`, the experiment selected for the premade configuration)
        with `channels` and `settings` (validated `background_values`
        section of a premade template) preset in its parameters dialog,
        which is still shown. Only the Positions are asked (not the
        experiments). Returns the summary table (empty if no Position was
        processed) or None if cancelled."""
        from .utils import computeBackgroundValues as utilsBackground
        parent = parent or self
        expPaths = utilsBackground.select_positions(
            parent, experimentPositions
        )
        if not expPaths:
            return None
        presetParams = {
            'channels': list(channels),
            'statistic': settings['statistic'],
            'exclude_zero_pixels': settings['exclude_zero_pixels'],
            'use_acdc_output': True,
            'low_percentile': settings['low_percentile'],
            'high_percentile': settings['high_percentile'],
        }
        win = utilsBackground.ComputeBackgroundValuesUtil(
            expPaths, QApplication.instance(), 'Compute background intensity',
            'Launching compute background intensity process...',
            'Computing background intensity', parent=parent,
            presetParams=presetParams,
            presetNote=(
                'The settings are preset by the configuration that is being '
                'set up. After the computation, its normalization values '
                'are filled in from the summary table.'
            )
        )
        # Keep a reference: the worker thread may still be finishing when
        # the dialog closes
        self.backgroundValuesUtilWin = win
        win.exec_()
        return win.summaryDf

    def selectPremadeBaseChannel(self, channels, default=None):
        """Ask which channel to display as base layer (default listed first)."""
        items = list(channels)
        if default in items:
            items.remove(default)
            items.insert(0, default)
        return self.selectItem(
            'Select channel',
            'Select channel to display (base layer).\n'
            'The mapped channels that are not displayed as base are '
            'added as overlays:', items
        )

    def askPremadeCopyFilename(self, defaultName):
        """Return the filepath for the copy of a premade configuration."""
        os.makedirs(MEASUREMENTS_CONFIGURATIONS_PATH, exist_ok=True)
        current_files = sorted(os.listdir(MEASUREMENTS_CONFIGURATIONS_PATH))
        helpText = html_utils.paragraph(
            'The premade configuration with your channels is saved as a new '
            'configuration, which you can later load and modify.'
        )
        msg = apps.filenameDialog(
            ext='json', title='Save configuration from premade template',
            parent=self, helpText=helpText, existingNames=current_files,
            allowEmpty=False, defaultEntry=defaultName
        )
        msg.exec_()
        if msg.cancel or not msg.filename:
            return None
        filepath = os.path.join(MEASUREMENTS_CONFIGURATIONS_PATH, msg.filename)
        if not filepath.lower().endswith('.json'):
            filepath += '.json'
        # fileNameDialog handles overwrite confirmation internally
        return filepath

    def instantiatePremadeConfiguration(self, configuration, source):
        """Create a configuration from a premade template.

        Asks for an experiment folder and a channel for each placeholder,
        replaces the `{NAME}` tokens, resolves the experiment input and writes
        the result to the configurations folder. Returns
        `(filepath, configuration, selectedInput)` or None if cancelled.
        """
        placeholders = configuration.get('channel_placeholders')
        self.validateChannelPlaceholders(placeholders)
        inputs = configuration.get('input_placeholders', {})
        self.validateInputPlaceholders(inputs, placeholders)
        backgroundSpec = self.validateBackgroundValuesSpec(
            configuration.get('background_values'), placeholders, inputs
        )
        channelHints = self.validateChannelHints(
            configuration.get('channel_hints'), placeholders
        )
        folder = QFileDialog.getExistingDirectory(
            self, 'Select experiment, Position or Images folder',
            self.getMostRecentPath()
        )
        if not folder:
            return None
        paths = get_images_paths(folder)
        if not paths:
            raise ValueError(f'No Images folders found in "{folder}".')
        available = [get_channels_and_segm_endnames(path) for path in paths]
        channels = set(available[0][0]).intersection(
            *(set(item[0]) for item in available)
        )
        if len(channels) < len(placeholders):
            raise ValueError(
                f'The premade configuration needs {len(placeholders)} '
                f'channels, but the positions have only {len(channels)} '
                'channels in common.'
            )
        # Experiments and Positions of the selected folder (background
        # values are computed from them)
        experimentPositions = {}
        for path in paths:
            pos_path = os.path.dirname(path)
            experimentPositions.setdefault(
                os.path.dirname(pos_path), []
            ).append(os.path.basename(pos_path))
        selection = self.selectChannelMapping(
            placeholders, sorted(channels), inputs=inputs,
            validate=partial(self.resolvePremadeTemplate, configuration),
            backgroundValues=backgroundSpec,
            experimentPositions=experimentPositions, channelHints=channelHints
        )
        if selection is None:
            return None
        mapping, values = selection
        result = self.resolvePremadeTemplate(configuration, mapping, values)
        result.setdefault('segm_endname', '')
        result.setdefault('overlay_channels', [])
        result.setdefault('visible_channels', [result.get('channel', '')])
        result.setdefault('version', 2)
        result.setdefault('time_course_columns', [])
        result.setdefault('measurements_axes', {'X': '', 'Y': ''})
        result.setdefault('measurements_mode', 0)
        result.setdefault('normalize_measurements', False)
        default = result.get('channel') or None
        offered = sorted(channels)
        if default is None:
            # No display channel in the template: offer the channels that
            # are not mapped (e.g. phase contrast) first
            mapped = set(mapping.values())
            offered.sort(key=lambda ch: ch in mapped)
        base = self.selectPremadeBaseChannel(offered, default)
        if base is None:
            return None
        self.setConfigurationBaseChannel(result, base)
        selectedInput = self.newExperimentInput(result, paths)
        if selectedInput is None:
            return None
        result.update(selectedInput)
        result['input_options'] = [copy.deepcopy(selectedInput)]
        result['premade'] = {
            'source': os.path.basename(source), 'channel_mapping': mapping
        }
        if inputs:
            result['premade']['input_values'] = dict(values or {})
        self.validateConfiguration(result)
        stem = os.path.splitext(os.path.basename(source))[0]
        defaultName = '_'.join([stem, *mapping.values()])
        filepath = self.askPremadeCopyFilename(defaultName)
        if filepath is None:
            return None
        self.writeConfigurationFile(filepath, result)
        return filepath, result, selectedInput

    def markConfigurationSaved(self):
        try:
            self.savedConfiguration = self.configuration()
        except Exception:
            self.logger.exception('Could not snapshot the configuration')
            self.savedConfiguration = None

    def hasUnsavedConfiguration(self):
        if not self.isDataLoaded:
            return False
        try:
            return self.configuration() != self.savedConfiguration
        except Exception:
            self.logger.exception('Could not compare the configuration')
            return True

    def askAcdcDfChangedOnDisk(self, posData):
        """Ask what to do with a table whose file changed on disk since it
        was loaded. Return 'reload', 'overwrite' or None (cancel)."""
        filepath = getattr(posData, 'acdc_output_csv_path', '')
        posName = getattr(posData, 'pos_foldername', filepath)
        txt = html_utils.paragraph(
            f'The acdc_output file of <b>{posName}</b> was modified on disk '
            '(e.g., saved in the main GUI) after the measurements viewer '
            'loaded it.<br><br>'
            '<b>Reload and save</b> reads the file again, keeps the time '
            'points, custom annotations, visited cells and evaluated phases '
            'of the viewer and saves them (other in-memory changes, such as '
            'processed or combined measurements, are discarded).<br><br>'
            '<b>Overwrite</b> replaces the file with the viewer\'s table '
            '(changes saved by other programs are lost).<br><br>'
            '<i>The previous file is backed up in the "recovery" folder in '
            'both cases.</i>'
        )
        msg = widgets.myMessageBox(wrapText=False)
        _, overwrite, reload = msg.warning(
            self, 'Table changed on disk', txt, detailsText=filepath or None,
            buttonsTexts=('Cancel', 'Overwrite', 'Reload and save')
        )
        if msg.cancel:
            return None
        if msg.clickedButton == reload:
            return 'reload'
        if msg.clickedButton == overwrite:
            return 'overwrite'
        return None

    def saveAcdcDfs(self, positions):
        # Pending phase evaluations of positions not displayed would
        # otherwise be saved stale (e.g. after renaming an annotation that
        # overrides phases)
        self.flushPendingPhaseEvaluations(positions)
        for posData in positions:
            filepath = getattr(posData, 'acdc_output_csv_path', None)
            if filepath and self.isAcdcDfChangedOnDisk(posData):
                answer = self.askAcdcDfChangedOnDisk(posData)
                if answer is None:
                    return False
                if answer == 'reload':
                    self.reloadAcdcDfFile(posData)
                    if self.data and posData is self.posData():
                        self.refreshAcdcDf()
            try:
                if not filepath:
                    raise OSError('No acdc_output file path for this position.')
                acdc_df = self._standardizeAcdcDf(posData.acdc_df).sort_index()
                acdc_df = acdc_df.copy()
                self.ensureTimepointColumns(acdc_df)
                self.ensureCustomAnnotColumns(acdc_df)
                if VISITED_COLUMN not in acdc_df:
                    acdc_df[VISITED_COLUMN] = False
                payload = acdc_df.to_csv().encode('utf-8')
                # Back up the previous file (restorable from the main GUI)
                load.store_copy_acdc_df(
                    posData, filepath, log_func=self.logger.warning
                )
                output = QSaveFile(filepath)
                if not output.open(QIODevice.WriteOnly):
                    raise OSError(output.errorString())
                if output.write(payload) != len(payload):
                    output.cancelWriting()
                    raise OSError(output.errorString())
                if not output.commit():
                    raise OSError(output.errorString())
            except (OSError, ValueError, TypeError) as err:
                self.logger.error('Could not save "%s": %s', filepath, err)
                self.measurementProcessingWarning(
                    'Saving tables failed',
                    f'Could not save the table of '
                    f'"{getattr(posData, "pos_foldername", filepath)}":\n\n{err}'
                )
                return False
            posData.savedAcdcDf = acdc_df.copy()
            posData.acdcDfFileStamp = acdc_df_file_stamp(filepath)
            self.logger.info(f'Table saved to "{filepath}"')
        if positions:
            self.statusBar().showMessage(
                f'Saved {len(positions)} acdc_output table(s).', 5000
            )
        return True

    def saveTables(self, checked=False):
        """Save the acdc_output tables of the positions with unsaved changes
        (File > Save tables, Ctrl+S). Tables without changes are not written,
        so that files saved by the main GUI in the meantime are kept. Return
        False if saving was refused, cancelled or failed."""
        if not self.isDataLoaded:
            # Qt5 emits `triggered` also for a disabled action
            self.statusBar().showMessage('Load data before saving tables.', 5000)
            return False
        if self._isLoadingData:
            text = 'Wait for the data being loaded before saving the tables.'
            self.logger.warning(text)
            self.statusBar().showMessage(text, 8000)
            return False
        positions = self.unsavedAcdcDfPositions()
        if not positions:
            text = 'Save tables: no unsaved changes to the tables.'
            self.logger.info(text)
            self.statusBar().showMessage(text, 5000)
            return True
        self.logger.info(f'Save tables: {len(positions)} position(s)')
        return self.saveAcdcDfs(positions)

    def askSaveBeforeClosing(self):
        """Prompt for unsaved tables/configuration. Return False to cancel."""
        return self.askSaveChanges('closing')

    def askSaveBeforeLoading(self):
        """Prompt for unsaved tables/configuration before other data (a
        configuration, folder or file) replaces the loaded data. Return False
        to cancel the load (no prompt and True when nothing is loaded)."""
        if not self.isDataLoaded:
            return True
        return self.askSaveChanges('loading other data')

    def askSaveChanges(self, action='closing'):
        """Prompt for unsaved tables/configuration before `action` (text of
        the prompt). Return False to cancel."""
        positions = self.unsavedAcdcDfPositions()
        configChanged = self.hasUnsavedConfiguration()
        if not positions and not configChanged:
            return True
        hasFile = self.configurationFilepath is not None
        saveConfigText = (
            'Save configuration' if hasFile else 'Save configuration...'
        )
        changes = []
        if positions:
            changes.append(
                f'the <b>measurement tables</b> of {len(positions)} '
                f'position(s)'
            )
        if configChanged:
            changes.append('the <b>configuration</b>')
        txt = html_utils.paragraph(
            f'There are unsaved changes to {" and ".join(changes)}.<br><br>'
            f'Do you want to save them before {action}?<br><br>'
            '<i>Saving the tables overwrites each position\'s acdc_output '
            'file (the previous file is backed up in the "recovery" '
            'folder; files modified on disk since loading are reported '
            'first).</i>'
        )
        details = '\n'.join(
            getattr(posData, 'acdc_output_csv_path', '') for posData in positions
        )
        options = [('Cancel', None), ("Don't save", (False, None))]
        if positions and configChanged:
            options += [
                ('Save tables only', (True, None)),
                (f'{saveConfigText} only', (False, 'save')),
            ]
            if hasFile:
                options.append(('Save both as...', (True, 'save_as')))
            options.append(('Save both', (True, 'save')))
        elif positions:
            options.append(('Save tables', (True, None)))
        else:
            if hasFile:
                options.append(('Save configuration as...', (False, 'save_as')))
            options.append((saveConfigText, (False, 'save')))
        msg = widgets.myMessageBox(wrapText=False)
        buttons = msg.warning(
            self, 'Save changes?', txt, detailsText=details or None,
            buttonsTexts=[text for text, _ in options]
        )
        if msg.cancel:
            self.logger.info(f'Save changes before {action}: Cancel')
            return False
        texts = dict(zip(buttons, (text for text, _ in options)))
        self.logger.info(
            f'Save changes before {action}: '
            f'"{texts.get(msg.clickedButton, "Cancel")}"'
        )
        actions = dict(zip(buttons, (action for _, action in options)))
        action = actions.get(msg.clickedButton)
        if action is None:
            return False
        saveTables, saveConfig = action
        if saveTables and not self.saveAcdcDfs(positions):
            return False
        if saveConfig is not None and not self.saveConfiguration(
                save_as=saveConfig == 'save_as'):
            return False
        return True

    def selectConfigurationInput(self, configuration):
        options = self.configurationInputOptions(configuration)
        def overlaysLabel(option):
            overlays = option.get('overlay_channels', [])
            return f'; overlays: {", ".join(overlays)}' if overlays else ''

        labels = [
            f'{i + 1}. {option["images_paths"][0]} '
            f'({len(option["images_paths"])} positions; '
            f'{option["channel"]}; {option["segm_endname"]}'
            f'{overlaysLabel(option)})'
            for i, option in enumerate(options)
        ]
        new = 'Load a new experiment...'
        selected = self.selectItem(
            'Select experiment', 'Load a saved experiment or a new one:',
            [*labels, new]
        )
        if selected is None:
            return None
        if selected != new:
            return options[labels.index(selected)]
        folder = QFileDialog.getExistingDirectory(
            self, 'Select experiment, Position or Images folder',
            self.getMostRecentPath()
        )
        if not folder:
            return None
        paths = get_images_paths(folder)
        if not paths:
            raise ValueError(f'No Images folders found in "{folder}".')
        return self.newExperimentInput(configuration, paths)

    def newExperimentInput(self, configuration, paths):
        """Map the configuration's input onto new Images folders.

        The configured channel, segmentation and overlays are reused when
        every position has them; otherwise the user picks a replacement
        (channel/segmentation) or confirms dropping them (overlays). Channels
        of the new experiment that the configuration does not use are then
        offered as optional extra overlays. The base channel becomes visible
        when none of the configured visible channels remains.
        """
        available = [get_channels_and_segm_endnames(path) for path in paths]
        channels = set(available[0][0]).intersection(*(set(item[0]) for item in available))
        segmentations = set(available[0][1]).intersection(*(set(item[1]) for item in available))
        if not channels or not segmentations:
            raise ValueError('The positions have no common channels or segmentations.')
        channel = configuration['channel']
        if not channel:
            channel = self.selectDisplayChannel(
                channels, configuration['overlay_channels']
            )
            if channel is None:
                return None
        elif channel not in channels:
            channel = self.selectItem(
                'Select channel',
                f'Channel "{channel}" is not in every position of the new '
                'experiment. Select channel to display:', sorted(channels)
            )
            if channel is None:
                return None
        segm = configuration['segm_endname']
        if segm not in segmentations:
            segm = self.selectItem(
                'Select segmentation',
                f'Segmentation "{segm}" is not in every position of the new '
                'experiment. Select segmentation to load:', sorted(segmentations)
            )
            if segm is None:
                return None
        configured = list(configuration['overlay_channels'])
        overlays = [c for c in configured if c in channels and c != channel]
        missing = [c for c in configured if c not in channels]
        if missing:
            msg = widgets.myMessageBox(wrapText=False)
            cancel, _ = msg.warning(
                self, 'Overlay channels missing',
                html_utils.paragraph(
                    'These overlay channels of the configuration are not in '
                    'every position of the new experiment and will be '
                    f'skipped:<br><br><b>{", ".join(missing)}</b>'
                ),
                buttonsTexts=('Cancel', 'Continue')
            )
            if msg.cancel or msg.clickedButton == cancel:
                return None
        used = {configuration['channel'], channel, *configured}
        added = self.selectConfigurationOverlayChannels(
            sorted(channels - used),
            text=(
                'The new experiment has channels that the configuration does '
                'not use.<br>Select the ones to add as overlay channels '
                '(or leave the selection empty):'
            )
        )
        if added is None:
            return None
        overlays += added
        visible = [
            c for c in configuration['visible_channels']
            if c == channel or c in overlays
        ]
        if channel != configuration['channel'] and channel not in visible:
            visible.insert(0, channel)
        visible += [c for c in added if c not in visible]
        if not visible:
            # E.g., base hidden and every visible overlay missing
            visible = [channel]
        return {
            'images_paths': [os.path.abspath(path) for path in paths],
            'channel': channel, 'segm_endname': segm,
            'overlay_channels': overlays,
            'visible_channels': visible,
        }

    def selectDisplayChannel(self, channels, overlays=()):
        """Ask for the display (base layer) channel of a configuration
        without one. Channels that are not overlays are offered first; an
        overlay chosen as base is removed from the overlays by the caller."""
        items = sorted(channels, key=lambda ch: (ch in overlays, ch))
        return self.selectItem(
            'Select channel',
            'The configuration does not specify a channel to display.\n'
            'Select channel to display (base layer):', items
        )

    def resolveInputChannel(self, selectedInput):
        """Return `selectedInput` with a display channel, asking for one if
        it is empty (None if cancelled). The chosen channel is removed from
        the overlays and shown first."""
        if selectedInput['channel']:
            return selectedInput
        available = [
            set(get_channels_and_segm_endnames(path)[0])
            for path in selectedInput['images_paths']
        ]
        channels = set.intersection(*available) if available else set()
        if not channels:
            raise ValueError('The positions have no common channels.')
        channel = self.selectDisplayChannel(
            channels, selectedInput['overlay_channels']
        )
        if channel is None:
            return None
        resolved = copy.deepcopy(selectedInput)
        self.setConfigurationBaseChannel(resolved, channel)
        return resolved

    def selectConfigurationOverlayChannels(self, channels, text=None):
        if not channels:
            return []
        dialog = widgets.QDialogListbox(
            'Select overlay channels',
            text or 'Select optional overlay channels (or leave the selection empty):',
            channels, multiSelection=True, parent=self, preSelectedItems=()
        )
        dialog.exec_()
        if dialog.cancel:
            return None
        return list(dialog.selectedItemsText)

    def askMeasurementReplay(self, outputs, data):
        """Return 'recalculate', 'existing', or None if the user cancels."""
        existing = {
            column for posData in data
            if getattr(posData, 'acdc_df', None) is not None
            for column in outputs if column in posData.acdc_df
        }
        incomplete = [
            os.path.basename(posData.pos_path) for posData in data
            if getattr(posData, 'acdc_df', None) is None
            or any(column not in posData.acdc_df for column in outputs)
        ]
        txt = (
            'This configuration contains <b>measurement processing</b> and/or '
            '<b>combined measurements</b> steps.<br><br>'
        )
        if incomplete:
            txt += (
                f'Their output columns are missing in {len(incomplete)} of '
                f'{len(data)} positions, so they must be calculated to use '
                'existing values.'
            )
        else:
            txt += 'Their output columns already exist in every position.'
        if existing:
            txt += (
                '<br><br>Calculating will <b>overwrite</b> the existing '
                'output columns.'
            )
        txt = html_utils.paragraph(
            f'{txt}<br><br>Do you want to calculate them?'
        )
        details = 'Output columns:\n' + '\n'.join(outputs)
        if incomplete:
            details += '\n\nPositions with missing columns:\n' + '\n'.join(incomplete)
        msg = widgets.myMessageBox(wrapText=False)
        cancel, existingButton, recalculate = msg.question(
            self, 'Calculate measurements?', txt, detailsText=details,
            buttonsTexts=('Cancel', 'Use existing values', 'Calculate')
        )
        if msg.cancel or msg.clickedButton == cancel:
            return None
        if msg.clickedButton == recalculate:
            return 'recalculate'
        if incomplete:
            raise ValueError(
                'Cannot use existing values: workflow columns are missing in '
                + ', '.join(incomplete) + '.'
            )
        return 'existing'

    def askMaskUnanalyzedFrames(self, partial, data):
        """Ask whether the calculations should ignore the frames after the
        last analyzed frame of the `partial` positions (of the loaded
        `data`). Return True (ignore them, default), False (use all frames)
        or None if the user cancels."""
        lines = '<br>'.join(
            f'&nbsp;&nbsp;- {html.escape(self.lastAnalyzedFrameText(posData))}'
            for posData in partial[:5]
        )
        if len(partial) > 5:
            lines += f'<br>&nbsp;&nbsp;- ... ({len(partial)-5} more)'
        txt = html_utils.paragraph(
            f'{len(partial)} of {len(data)} position(s) have <b>not been '
            'analyzed until the end of the video</b> in the main GUI '
            '(last cell cycle annotated frame, or last tracked frame '
            f'without cell cycle annotations):<br><br>{lines}<br><br>'
            'Do you want to <b>ignore the frames after the last analyzed '
            'frame</b> in all calculations (time point and custom '
            'annotation predictors, phases, measurement processing, '
            'combined measurements and conditional formatting)? Their '
            'outputs then stay empty, while images and plots still show '
            'all frames.<br><br>'
            'You can change this later with <code>Annotations --> Ignore '
            'frames after the last analyzed frame</code>; the choice is '
            'saved in the configuration.'
        )
        details = '\n'.join(
            self.lastAnalyzedFrameText(posData) for posData in partial
        )
        msg = widgets.myMessageBox(wrapText=False)
        cancel, useAll, ignore = msg.question(
            self, 'Video not analyzed until the end', txt,
            detailsText=details,
            buttonsTexts=(
                'Cancel', 'Use all frames', 'Ignore later frames (recommended)'
            )
        )
        if msg.cancel or msg.clickedButton == cancel:
            return None
        return msg.clickedButton != useAll

    def resolveMaskUnanalyzedFrames(self, data, stored=None, default=True):
        """Whether the calculations of the loaded `data` ignore the frames
        after the last analyzed frame: the `stored` choice of a loaded
        configuration (bool), else the answer to `askMaskUnanalyzedFrames`
        when some positions are partially analyzed, else `default`. Return
        ``(mask, asked)``; `mask` is None if the user cancels."""
        partial = self.partiallyAnalyzedPositions(data)
        if isinstance(stored, bool):
            mask, asked = stored, False
        elif partial:
            mask, asked = self.askMaskUnanalyzedFrames(partial, data), True
            if mask is None:
                self.logger.info('Loading cancelled (unanalyzed frames prompt)')
                return None, asked
        else:
            mask, asked = bool(default), False
        for posData in partial:
            self.logger.info(
                f'{self.lastAnalyzedFrameText(posData)}: later frames '
                f'{"ignored" if mask else "used"} by the calculations'
            )
        return mask, asked

    def loadingInProgress(self):
        """Return True, and tell the user (status bar and log), while other
        data is being loaded (`_isLoadingData`), to refuse a second load:
        the dialogs of a load do not block the window, so a load could
        otherwise start while another one is waiting for an answer.
        Loading is also refused while the "Save changes before closing?"
        prompt is open (`_isClosingPrompt`): it would save or close the
        replaced data."""
        if self._isClosingPrompt:
            text = 'Answer the "Save changes before closing?" prompt first.'
        elif self._isLoadingData:
            text = (
                'Wait for the data being loaded (or answer its dialog) first.'
            )
        else:
            return False
        self.logger.warning(f'Loading other data refused. {text}')
        self.statusBar().showMessage(text, 8000)
        return True

    def refuseLoadingWhileProcessing(self):
        """Return True, and tell the user, while measurement processing
        runs: its results would be merged into (and its step recorded for)
        the data being replaced."""
        if self.measurementProcessingThread is None:
            return False
        self.configurationError(
            'load', 'Wait for measurement processing to finish.'
        )
        return True

    def refuseProcessingWhileLoading(self):
        """Return True, and tell the user, while data is being loaded (see
        `refuseLoadingWhileProcessing`) or the "Save changes before
        closing?" prompt is open."""
        if self._isClosingPrompt:
            text = 'Answer the "Save changes before closing?" prompt first.'
        elif self._isLoadingData:
            text = (
                'Wait for the data being loaded (or answer its dialog) first.'
            )
        else:
            return False
        self.logger.warning(f'Process measurements refused. {text}')
        self.statusBar().showMessage(text, 8000)
        return True

    def loadConfiguration(self, checked=False, filepath=None):
        if self.loadingInProgress():
            return False
        self._isLoadingData = True
        try:
            return self._loadConfiguration(filepath=filepath)
        finally:
            self._isLoadingData = False

    def _loadConfiguration(self, filepath=None):
        if self.refuseLoadingWhileProcessing():
            return False
        if filepath is None:
            filepath = self.selectConfigurationFilepath()
        if not filepath:
            return False
        # Ask before anything is replaced (setTimepointIdentifiers removes the
        # old tables' time point columns) and before reading the file, so a
        # configuration saved over the same file is the one that is loaded.
        if not self.askSaveBeforeLoading():
            return False
        # Processing may have been started from the non-blocking dialogs
        if self.refuseLoadingWhileProcessing():
            return False
        self.logger.info(f'Loading configuration "{filepath}"...')
        try:
            with open(filepath, encoding='utf-8') as file:
                configuration = json.load(file)
            if (isinstance(configuration, dict)
                    and 'channel_placeholders' in configuration):
                premade = self.instantiatePremadeConfiguration(
                    configuration, filepath
                )
                if premade is None:
                    return False
                filepath, configuration, selectedInput = premade
            else:
                self.validateConfiguration(configuration)
                selectedInput = self.selectConfigurationInput(configuration)
                if selectedInput is not None:
                    selectedInput = self.resolveInputChannel(selectedInput)
                if selectedInput is not None:
                    # Report a bad input instead of applying it (it would
                    # also block saving the configuration later)
                    self.validateConfigurationInput(selectedInput)
            if selectedInput is None:
                return False
            inputs = self.configurationInputOptions(configuration)
            selectedKey = self.inputOptionKey(selectedInput)
            if all(self.inputOptionKey(option) != selectedKey
                   for option in inputs):
                inputs.append(copy.deepcopy(selectedInput))
            configuration.update(selectedInput)
            for path in configuration['images_paths']:
                if not os.path.isdir(path):
                    raise FileNotFoundError(f'Images folder not found: "{path}".')
                channels, segmentations = get_channels_and_segm_endnames(path)
                required = {
                    configuration['channel'], *configuration['overlay_channels']
                }
                missing = required.difference(channels)
                if missing:
                    raise ValueError(f'Missing channels in "{path}": {sorted(missing)}.')
                if configuration['segm_endname'] not in segmentations:
                    raise ValueError(f'Segmentation not found in "{path}".')
            legacyViewed = {
                os.path.normcase(os.path.abspath(path)): IDs
                for path, IDs in configuration.get('viewed_cells', {}).items()
            }
            self.logger.info(
                f'Loading channel "{configuration["channel"]}" of '
                f'{len(configuration["images_paths"])} position(s): '
                + ', '.join(
                    f'"{path}"' for path in configuration['images_paths']
                )
            )
            # In a worker thread (including the overlay channels, taken by
            # `load_fluo_data`): the window stays responsive. Errors are
            # re-raised here.
            data = self.runInThread(
                load_display_positions, configuration['images_paths'],
                configuration['channel'], configuration['segm_endname'],
                skip_errors=False, reportProgress=True,
                overlay_channels=configuration['overlay_channels'],
                text=(
                    f'Loading {len(configuration["images_paths"])} '
                    'position(s)...'
                )
            )
            # Before the measurement steps and the predictions (of
            # `setData`), which use only the analyzed frames when masking
            mask, maskAsked = self.resolveMaskUnanalyzedFrames(
                data, stored=configuration.get('mask_unanalyzed_frames')
            )
            if mask is None:
                return False
            replay = None
            outputs = self.measurementStepOutputs(configuration)
            if outputs:
                replay = self.askMeasurementReplay(outputs, data)
                if replay is None:
                    return False
            stepsLastFrames = dict(
                configuration.get('measurement_steps_last_frames', {})
            )
            if replay == 'recalculate':
                lastFrames = [
                    analysis_last_frame(posData, mask) for posData in data
                ]
                def replaySteps():
                    for posData, last in zip(data, lastFrames):
                        self.replayMeasurementSteps(
                            posData, configuration, overwrite=True,
                            last_frame=last
                        )
                self.runInThread(
                    replaySteps, text='Calculating the measurement steps...'
                )
                for posData, last in zip(data, lastFrames):
                    self.recordStepsLastFrame(
                        stepsLastFrames, posData, last, replace=True
                    )
                    # Saved NaN outputs are read back as 0 (tables with
                    # cell cycle columns): not an edit of the file
                    blank_unanalyzed_outputs(
                        getattr(posData, 'savedAcdcDf', None), outputs, last
                    )
            elif replay == 'existing':
                # Saved outputs are empty (read back as 0 from a table with
                # cell cycle columns) after the frame they were calculated
                # up to, and must be ignored after the current one
                for posData in data:
                    last = min_frame(
                        stepsLastFrames.get(position_key(posData)),
                        analysis_last_frame(posData, mask)
                    )
                    self.blankStepOutputs(posData, outputs, last)
                    self.recordStepsLastFrame(stepsLastFrames, posData, last)
            for posData in data:
                legacyIDs = legacyViewed.get(
                    os.path.normcase(os.path.abspath(posData.pos_path)), []
                ) if legacyViewed else []
                if legacyIDs and posData.acdc_df is not None:
                    if VISITED_COLUMN not in posData.acdc_df:
                        posData.acdc_df[VISITED_COLUMN] = False
                    rows = posData.acdc_df.index.get_level_values('Cell_ID').isin(legacyIDs)
                    posData.acdc_df.loc[rows, VISITED_COLUMN] = True
        except Exception as err:
            self.logger.exception('Loading measurements viewer workflow failed')
            self.configurationError('load', err)
            return False

        self.isSegm3D = None
        # Remove the previous custom annotations while their data are still
        # loaded: setData (getAcdcDf) adds a column to the new tables for
        # every registered annotation, and setTimepointIdentifiers checks
        # the new auto shortcuts against the registered annotation actions.
        self.setUncheckedAllCustomAnnotButtons()
        self.reinitCustomAnnot()
        # self.setCellConstants(configuration.get('cell_constants', []))
        self.setTimepointIdentifiers([
            TimepointIdentifier.from_dict(identifier)
            for identifier in configuration.get('timepoint_identifiers', [])
        ])
        self.configurationInputs = inputs
        self.loadedInputKey = selectedKey
        self.measurementProcessingSteps = copy.deepcopy(
            configuration.get('measurement_processing', [])
        )
        self.combinedMeasurementSteps = copy.deepcopy(
            configuration.get('combined_measurements', [])
        )
        # Before the deferred predictions of the displayed cell run (on exit
        # of the block below), so they auto-apply the new phase rules.
        self.phaseTranslatorRules = copy.deepcopy(
            configuration.get('phase_translator_rules', [])
        )
        self.phaseColors = dict(configuration.get('phase_colors', {}))
        self.autoApplyPhases = configuration.get('auto_apply_phases', False)
        self.setMaskUnanalyzedFrames(mask, update=False)
        self.measurementStepsLastFrames = stepsLastFrames
        with self.deferPredictorPredictions():
            # Session prediction state of the replaced data is rebuilt from
            # the visited column of the loaded tables (as in
            # `loadImagesPaths`).
            self.resetAutoPredictionTracking()
            self.setData(data)
            self.setCustomAnnotations(configuration.get('custom_annotations', []))
            # Exported assignments and time points of visited cells must
            # survive auto-prediction.
            self.seedVisitedCustomAnnotPredictions(
                self.customAnnotNames(), data
            )
            self.seedVisitedTimepointPredictions(data=data)
        self.configurationFilepath = None
        overlays = configuration['overlay_channels']
        if overlays:
            loaded = self.loadOverlayData(overlays)
            # Release the preloaded data that `load_fluo_data` did not take
            for posData in data:
                posData.__dict__.pop('_viewerPreloadedFluo', None)
            if not loaded:
                self.configurationError('load', 'Could not load overlay channels.')
                return False
            self.setCheckedOverlayContextMenusActions(overlays)
            self.overlayButton.setChecked(True)
        for channel, button in self.allOverlayToolbuttons.items():
            button.setChecked(channel in configuration['visible_channels'])
        self.setOverlayItemsOpacities()

        columns = {item.text() for item in self.columnsModel.measurementItems()}
        selected = set(configuration['time_course_columns'])
        axes = configuration['measurements_axes']
        missing = (selected | set(axes.values())).difference(columns, {''})
        self.measurementsModeCombobox.blockSignals(True)
        self.measurementsModeCombobox.setCurrentIndex(configuration['measurements_mode'])
        self.measurementsModeCombobox.blockSignals(False)
        # Do not replace the restored time-course selection with the old checks.
        self._measurementsMode = 1
        self._timeCourseColumns = selected & columns
        self.measurementsAxes = {
            axis: value if value in columns else '' for axis, value in axes.items()
        }
        self.conditionalFormatting = copy.deepcopy(
            configuration.get('conditional_formatting', {})
        )
        self.measurementsColors = {
            col: pg.mkColor(color) for col, color
            in configuration.get('measurements_colors', {}).items()
        }
        self.measurementsModeChanged()
        self.normalizeCheckbox.setChecked(configuration['normalize_measurements'])
        self.configurationFilepath = filepath
        self.markConfigurationSaved()
        if maskAsked and isinstance(self.savedConfiguration, dict):
            # The answer is not in the file yet: offer to save it on closing
            self.savedConfiguration.pop('mask_unanalyzed_frames', None)
        self.logger.info(f'Configuration loaded: "{filepath}"')
        if missing:
            self.configurationError(
                'restore all plot selections',
                f'Columns no longer available: {", ".join(sorted(missing))}.'
            )
        return True
    
    def postInitLoadSetSettings(self):
        followID = self._get_setting_value('followID', default=True, cast=bool)
        self.followIDcheckbox.setChecked(followID)
        
        restrictFrames = self._get_setting_value('restrictFrames', default=True, cast=bool)
        self.restrictFramesCheckbox.setChecked(restrictFrames)
        
        rescalePlot = self._get_setting_value('rescalePlot', default=True, cast=bool)
        self.rescalePlotCheckbox.setChecked(rescalePlot)


    def annotatedAxesIndices(self):
        # The viewer has only the left image (no right image like in guiWin)
        return (0,)
    
    def setDoNotAnnotate(self, checked):
        # Same as `gui.guiWin.setDoNotAnnotate` but only for the left image
        self.annotDoNotAnntoateCheckbox(0).setChecked(checked)
    
    def initViewerToolsAttributes(self):
        """Initialize the attributes required by the mixins in 
        `gui_viewer_tools` and `gui_overlay`"""
        self._appName = 'Cell-ACDC'
        try:
            self._version = myutils.read_version()
        except Exception as err:
            self._version = None
        self.app = QApplication.instance()
        self.mainWin = None
        self.isSnapshot = False
        self.isTwoImageLayout = False
        
        self.exportMaskImage = None
        self.exportToImageWindow = None
        self.timestampStartTimedelta = timedelta(seconds=0)
        
        self.scaleBarDialog = None
        self.timestampDialog = None
        self.slideshowWin = None
        self.ccaTableWin = None
        self.countObjsWindow = None
        self.progressWin = None
        
        self.tempSegmentON = False
        self.isMouseDragImg1 = False
        self.invertBwAlreadyCalledOnce = False
        self.currentPropsID = -1
        # True while the code (hover, sync with the selected cell) writes
        # the Object ID of the "Cell-ACDC objects" dock: only user edits
        # of the ID select a cell (see `propsWidgetIDvalueChanged`)
        self._isUpdatingPropsDock = False
        self.xHoverImg, self.yHoverImg = None, None
        self.countKeyPress = 0
        self.doubleKeyTimeElapsed = False
        self.isKeyDoublePress = False
        self.doubleSpaceBarState = False
        
        self.mouseBindings = {}
        self.defaultMouseShortcuts = {
            'Next': Qt.MouseButton.XButton2, 
            'Previous': Qt.MouseButton.XButton1
        }
        self.zoomOutKeyValue = Qt.Key_H
        self.delObjAction = None
        self.delObjToolAction = QAction(self)
        self.delObjToolAction.setCheckable(True)
        self.delObjToolAction.setVisible(False)
        self.widgetsWithShortcut = {}
        self.leftClickButtons = []
        
        self.fluoDataChNameActions = []
        self.retainSizeLutItems = False
        self.AutoPilotProfile = _NoAutoPilotProfile()
        self.checkedOverlayChannels = set()
        self.overlayLabelsItems = {}
        self.drawModeOverlayLabelsChannels = {}
        self.existingSegmEndNames = []
        self.ch_names = []
        
        # Main GUI widgets that do not exist in the viewer
        self.equalizeHistPushButton = QPushButton(self)
        self.equalizeHistPushButton.hide()
        self.imgGradRight = _NoOpItem()
    
    def initCustomAnnotationsAttributes(self):
        """Initialize the attributes required by 
        `gui_custom_annotations.CustomAnnotationsGuiElements`"""
        self.customAnnotDict = {}
        self.customAnnotButton = None
        self.savedCustomAnnot = {}
        # {button: (toggle action, Shift+shortcut re-evaluate action)}
        self.customAnnotShortcutActions = {}
        # Cells already predicted {(normcase pos_path, name): set of IDs}
        self.customAnnotAutoPredictions = defaultdict(set)
        # {name: positions whose table had the column before it was added}
        self.customAnnotPreexistingColumns = {}

    def initAnnotationsAttributes(self):
        """Initialize the attributes required by 
        `gui_annotations.AnnotationsGuiElements`"""
        self.annotOptionsCheckboxes = defaultdict(dict)
        self.annotOptionsToRestore = defaultdict(dict)
        self.textAnnot = {}
        self.tracksPlotItems = {}
        self.overlayLayersItems = {}
        self.highlightedID = 0
        self.isExportingVideo = False
        self.lineage_tree = None
        self.cca_df_colnames = cca_df_colnames
        self.cca_df_int_cols = CCA_DF_INT_COLS
        self._acdcRegionProps = acdc_regionprops.acdcRegionprops
        
        # Annotations in the viewer are always displayed as in 'Viewer' mode
        self.modeComboBox = QComboBox(self)
        self.modeComboBox.addItem('Viewer')
        self.modeComboBox.hide()
        
        savedFontSize = '12'
        if 'fontSize' in self.df_settings.index:
            savedFontSize = str(self.df_settings.at['fontSize', 'value'])
        if savedFontSize.find('pt') != -1:
            savedFontSize = savedFontSize[:-2]
        try:
            self.fontSize = int(float(savedFontSize))
        except Exception as err:
            self.fontSize = 12
    
    def gui_createActions(self):
        self.openFolderAction = QAction(
            QIcon(":folder-open.svg"), "&Load Folder...", self
        )
        self.openFolderAction.setShortcut('Ctrl+O')
        self.openFolderAction.setToolTip(
            'Load an experiment, Position or Images folder'
        )
        self.openFileAction = QAction(
            QIcon(":image.svg"), "&Open Image File...", self
        )
        self.openFileAction.setToolTip(
            'Open an image file from the Images folder of a Position'
        )
        self.showInExplorerAction = QAction(
            QIcon(":drawer.svg"), f"&{self.openFolderText}", self
        )
        self.reloadAction = QAction(
            QIcon(":reload.svg"), "Reload acdc_output Table", self
        )
        self.reloadAction.setShortcut('F5')
        self.reloadAction.setToolTip(
            'Reload the acdc_output table from the file '
            '(e.g., after saving in the main GUI)'
        )
        self.exitAction = QAction("&Exit", self)
        self.loadConfigurationAction = QAction(
            QIcon(":file-open.svg"), 'Load configuration...', self
        )
        self.loadConfigurationAction.setToolTip(
            'Load a measurements viewer configuration\n\n'
            'Restores inputs, measurement steps, plots, time point\n'
            'identifiers, custom annotations and phase rules.'
        )
        # Ctrl+S saves the tables (as in the main GUI), the configuration has
        # its own icon so that the two save buttons are told apart
        self.saveTablesAction = QAction(
            QIcon(":file-save.svg"), 'Save tables', self
        )
        self.saveTablesAction.setShortcut('Ctrl+S')
        self.saveTablesAction.setToolTip(
            'Save tables (Ctrl+S)\n\n'
            'Overwrite the acdc_output table of every Position with unsaved\n'
            'changes (time points, annotations, phases, measurement steps). The\n'
            'previous file is backed up in the Position\'s "recovery" folder.'
        )
        self.saveConfigurationAction = QAction(
            QIcon(":cog.svg"), 'Save configuration', self
        )
        self.saveConfigurationAction.setShortcut('Ctrl+Shift+S')
        self.saveConfigurationAction.setToolTip(
            'Save the current configuration (Ctrl+Shift+S)'
        )
        self.saveConfigurationAsAction = QAction(
            QIcon(":edit-copy.svg"), 'Save configuration as...', self
        )
        self.saveConfigurationAsAction.setToolTip(
            'Save the current configuration to a new file'
        )
        self.addTimepointIdentifierAction = QAction(
            QIcon(":add.svg"), 'Add time point identifier...', self
        )
        self.addTimepointIdentifierAction.setToolTip(
            'Add time point identifier\n\n'
            'Define a named time point (e.g., "Bud '
            'emergence") with a color, a keyboard\n'
            'shortcut and an optional per-cell predictor. Hover over the measurements\n'
            'plot and press the shortcut to add a time point at the hovered frame.'
        )
        self.reevaluateAllTimepointsAction = QAction(
            QIcon(":reload.svg"), 'Re-evaluate predictors on all cells', self
        )
        self.reevaluateAllTimepointsAction.setToolTip(
            'Re-evaluate predictors on all cells\n\n'
            'Run the predictors of all time point identifiers on every\n'
            'cell of the current position, replacing their time points.\n'
            'You will be warned if manual edits would be overwritten.'
        )
        self.reevaluateCellTimepointsAction = QAction(
            QIcon(":magic_wand.svg"), 
            'Re-evaluate predictors on displayed cell', self
        )
        self.reevaluateCellTimepointsAction.setToolTip(
            'Re-evaluate predictors on displayed cell\n\n'
            'Run the predictors of all time point identifiers only\n'
            'on the displayed cell, replacing its time points. You\n'
            'will be warned if manual edits would be overwritten.'
        )
        self.phaseTranslatorAction = QAction(
            phase_translator_icon(), 'Time point to phase translator...',
            self
        )
        self.phaseTranslatorAction.setToolTip(PHASE_TRANSLATOR_TOOLTIP)
        self.phaseTranslatorAction.setStatusTip(
            'Translate the time points of each cell into phases (colored '
            'band below the plot, phase columns in the exported table)'
        )
        self.applyPhasesCellAction = QAction(
            phase_apply_icon(), 'Evaluate phases on displayed cell', self
        )
        self.applyPhasesCellAction.setShortcut(APPLY_PHASES_CELL_SHORTCUT)
        self.applyPhasesCellAction.setShortcutContext(
            Qt.ShortcutContext.WindowShortcut
        )
        self.applyPhasesCellAction.setToolTip(
            'Evaluate phases on displayed cell\n\n'
            'Apply the rules of the time point to phase translator (and the '
            'custom\n'
            'annotations that override phases) to the displayed cell only, '
            'replacing its\n'
            'phases (like "Apply phase evaluation" on the displayed cell in '
            'the\n'
            'translator). Define the rules first with "Time point to phase '
            'translator...".\n\n'
            'Shortcut: "' + APPLY_PHASES_CELL_SHORTCUT + '"'
        )
        self.applyPhasesCellAction.setStatusTip(
            'Evaluate the phases of the displayed cell with the phase '
            'translator rules'
        )
        self.addAction(self.applyPhasesCellAction)
        # Editable in the shortcuts editor (Edit > Edit keyboard shortcuts)
        self.widgetsWithShortcut['Evaluate phases on displayed cell'] = (
            self.applyPhasesCellAction
        )
        self.maskUnanalyzedFramesAction = QAction(
            'Ignore frames after the last analyzed frame', self
        )
        self.maskUnanalyzedFramesAction.setCheckable(True)
        self.maskUnanalyzedFramesAction.setChecked(self.maskUnanalyzedFrames)
        self.maskUnanalyzedFramesAction.setToolTip(MASK_UNANALYZED_TOOLTIP)
        self.maskUnanalyzedFramesAction.setStatusTip(
            'Calculations use only the frames analyzed in the main GUI '
            '(last cell cycle annotated, else last tracked frame)'
        )

        # Help menu
        self.userGuideAction = QAction(
            QIcon(':browser.svg'), 'Measurements viewer user guide...', self
        )
        self.userGuideAction.setToolTip(
            'Open the measurements viewer documentation in the web browser'
        )
        self.quickStartAction = QAction(
            QIcon(':info.svg'), 'Quick start...', self
        )
        self.quickStartAction.setToolTip(
            'Show a short how-to of the measurements viewer'
        )
        self.acdcDocsAction = QAction('Cell-ACDC documentation...', self)
        # self.editCellConstantsAction = QAction(
        #     QIcon(":calc.svg"), 'Per-cell constants...', self
        # )
        # self.editCellConstantsAction.setToolTip(
        #     'Per-cell constants\n\n'
        #     'Define constants computed once per cell (e.g., "Maximum value" '
        #     'or "Was at beginning")\n'
        #     'that can be used like columns in the time point predictors.'
        # )
        
        self.gui_createCustomAnnotationsActions()
        self.reevaluateAllCustomAnnotationsAction = QAction(
            QIcon(":reload.svg"),
            'Re-evaluate annotation predictors on all cells', self
        )
        self.reevaluateAllCustomAnnotationsAction.setToolTip(
            'Re-evaluate annotation predictors on all cells\n\n'
            'Run the predictors of all custom annotations on every cell\n'
            'of the current position, replacing their annotations. You\n'
            'will be warned if manual edits would be overwritten.'
        )
        self.reevaluateCellCustomAnnotationsAction = QAction(
            QIcon(":magic_wand.svg"),
            'Re-evaluate annotation predictors on displayed cell', self
        )
        self.reevaluateCellCustomAnnotationsAction.setToolTip(
            'Re-evaluate annotation predictors on displayed cell\n\n'
            'Run the predictors of all custom annotations only on\n'
            'the displayed cell, replacing its annotations. You\n'
            'will be warned if manual edits would be overwritten.'
        )
        self.addCustomAnnotationAction.setToolTip(
            'Add custom annotation\n\n'
            'Define an annotation (saved as a 0/1 column) with a symbol, a keyboard\n'
            'shortcut and an optional predictor. Select a cell and press the\n'
            'shortcut to annotate/un-annotate it across its entire video history\n'
            '(Shift+shortcut re-evaluates its predictor on the selected cell).\n'
            'Alternatively, activate the toolbar button and right-click on a cell.'
        )
        
        # Annotations actions (same as in the main GUI)
        self.scalingLabelsAction = widgets.CheckableAction(
            'Fix text annotations to screen size'
        )
        self.scalingLabelsAction.setToolTip(
            'When the text annotations are fixed to screen size, they\n'
            'do not scale relative to the objects when zooming in/out.'
        )
        self.scalingLabelsAction.setChecked(
            self._get_setting_value('scalingLabels', False, cast=bool)
        )
        self.highLowResAction = widgets.CheckableAction(
            'High resolution text annotations'
        )
        self.highLowResAction.setToolTip(
            'Resolution of the text annotations. High '
            'resolution results in slower update\n'
            'of the annotations. Not recommended with '
            'a number of segmented objects > 500.'
        )
        self.highLowResAction.setChecked(
            self._get_setting_value('highLowRes', True, cast=bool)
        )
        self.editTextIDsColorAction = QAction(
            'Edit text annotations color...', self
        )
        self.annotObjTrackSettingsAction = QAction(
            'Object tracks settings...', self
        )
        self.resetAnnotOptionsAction = QAction(
            'Reset default annotation options', self
        )
        self.annotationsActions = (
            self.editTextIDsColorAction, self.annotObjTrackSettingsAction,
            self.resetAnnotOptionsAction
        )
        self.gui_createViewerToolsActions()
    
    def gui_createViewerToolsActions(self):
        """Create the actions of the 'Viewer' mode tools as in `gui.guiWin`"""
        if 'colorScheme' in self.df_settings.index:
            self._colorScheme = self.df_settings.at['colorScheme', 'value']
        else:
            self._colorScheme = 'light'
        self.toggleColorSchemeAction = QAction(
            'Switch to light theme', self
        )
        self.gui_updateSwitchColorSchemeActionText()
        
        self.editShortcutsAction = QAction(
            'Customize keyboard shortcuts...', self
        )
        self.editShortcutsAction.setShortcut('Ctrl+K')
        
        self.editOverlayColorAction = QAction('Overlay color...', self)
        self.editOverlayColorAction.setDisabled(True)
        
        self.viewCcaTableAction = QAction(
            'View cell cycle annotations...', self
        )
        self.viewCcaTableAction.setDisabled(True)
        self.viewCcaTableAction.setShortcut('Ctrl+P')
        
        self.addScaleBarAction = QAction('Add scale bar', self)
        self.addScaleBarAction.setCheckable(True)
        
        self.addTimestampAction = QAction('Add timestamp', self)
        self.addTimestampAction.setCheckable(True)
        
        self.invertBwAction = QAction('Invert black/white', self)
        self.invertBwAction.setCheckable(True)
        checked = (
            self._get_setting_value('is_bw_inverted', 'No', cast=str) == 'Yes'
            or self.isBwInverted
        )
        self.invertBwAction.setChecked(checked)
        
        self.shuffleCmapAction = QAction('Randomly shuffle colormap', self)
        self.shuffleCmapAction.setShortcut('Shift+S')
        
        self.greedyShuffleCmapAction = QAction(
            'Greedily shuffle colormap', self
        )
        self.greedyShuffleCmapAction.setShortcut('Alt+Shift+S')
        
        self.saveLabColormapAction = QAction(
            'Save labels colormap...', self
        )
        
        self.normalizeRawAction = QAction(
            'Do not normalize. Display raw image', self)
        self.normalizeToFloatAction = QAction(
            'Convert to floating point format with values [0, 1]', self)
        self.normalizeRescale0to1Action = QAction(
            'Rescale to [0, 1]', self)
        self.normalizeByMaxAction = QAction(
            'Normalize by max value', self)
        self.normalizeQActionGroup = QActionGroup(self)
        for action in (
                self.normalizeRawAction, self.normalizeToFloatAction,
                self.normalizeRescale0to1Action, self.normalizeByMaxAction
            ):
            action.setCheckable(True)
            self.normalizeQActionGroup.addAction(action)
        
        self.zoomToObjsAction = QAction(
            'Zoom to objects  (Shortcut: H key)', self
        )
        self.zoomOutAction = QAction(
            'Zoom out  (Shortcut: double press H key)', self
        )
        # Required by ZoomGuiElements.zoomToCells, but not shown in the
        # viewer (it never zooms automatically on Next/Previous)
        self.enableAutoZoomToCellsAction = QAction(
            'Automatic zoom to all cells when pressing "Next/Previous"', self
        )
        self.enableAutoZoomToCellsAction.setCheckable(True)
        
        self.imgPropertiesAction = QAction('Properties...', self)
        self.imgPropertiesAction.setDisabled(True)
        
        self.launch3dViewerAction = QAction(
            'Launch 3D viewer for the currently visualised volume', self
        )
        self.launch3dViewerAction.setIcon(QIcon(":3d.svg"))
        self.launch3dViewerAction.setCheckable(True)
        self.launch3dViewerAction.setDisabled(True)
        
        self.searchAction = QAction(self)
        self.searchAction.setIcon(QIcon(":find.svg"))
        self.searchAction.setToolTip('Search for ID or functionality')
        
        self.focusOnSearchAction = QAction(
            "Search for ID or functionality", self
        )
        self.focusOnSearchAction.setShortcut('Ctrl+F')
        
        # Frames (or z-slices if "z" is active) navigation as in guiWin
        self.nextAction = QAction('Next', self)
        self.widgetsWithShortcut['Next'] = self.nextAction
        self.prevAction = QAction('Previous', self)
        self.widgetsWithShortcut['Previous'] = self.prevAction
        self.addAction(self.nextAction)
        self.addAction(self.prevAction)
        
        self.MaxProjToggleAction = QAction(self)
        self.MaxProjToggleAction.triggered.connect(
            self.maxProjToggleActionTriggered
        )
        self.MaxProjToggleAction.setShortcutContext(
            Qt.ShortcutContext.WindowShortcut
        )
        self.addAction(self.MaxProjToggleAction)
        self.MaxProjToggleAction.setShortcut('Alt+X')
        self.widgetsWithShortcut['Toggle max z projection'] = (
            self.MaxProjToggleAction
        )
        
        self.viewerToolsActions = (
            self.viewCcaTableAction, self.imgPropertiesAction, 
            self.addScaleBarAction, self.addTimestampAction, 
            self.shuffleCmapAction, self.greedyShuffleCmapAction,
            self.saveLabColormapAction, self.zoomToObjsAction, 
            self.zoomOutAction, self.searchAction, self.focusOnSearchAction,
            self.launch3dViewerAction, self.MaxProjToggleAction,
            *self.normalizeQActionGroup.actions()
        )
    
    def gui_createMenuBar(self):
        fileMenu = self.menuBar().addMenu('&File')
        self.fileMenu = fileMenu
        fileMenu.addAction(self.openFolderAction)
        fileMenu.addAction(self.openFileAction)
        self.openRecentMenu = fileMenu.addMenu("Open Recent")
        fileMenu.addAction(self.saveTablesAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.showInExplorerAction)
        fileMenu.addAction(self.reloadAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.loadConfigurationAction)
        fileMenu.addAction(self.saveConfigurationAction)
        fileMenu.addAction(self.saveConfigurationAsAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
        
        editMenu = self.menuBar().addMenu('&Edit')
        editMenu.addAction(self.editShortcutsAction)
        editMenu.addAction(self.editOverlayColorAction)
        # enableAutoZoomToCellsAction is not in the menu: the viewer always
        # zooms with enforce=True (the action exists only for the mixin)

        viewMenu = self.menuBar().addMenu('&View')
        viewMenu.addAction(self.viewCcaTableAction)
        viewMenu.addSeparator()
        viewMenu.addAction(self.toggleColorSchemeAction)
        
        ImageMenu = self.menuBar().addMenu('&Image')
        ImageMenu.addAction(self.imgPropertiesAction)
        normalizeIntensitiesMenu = ImageMenu.addMenu('Normalize intensities')
        normalizeIntensitiesMenu.addAction(self.normalizeRawAction)
        normalizeIntensitiesMenu.addAction(self.normalizeToFloatAction)
        normalizeIntensitiesMenu.addAction(self.normalizeRescale0to1Action)
        normalizeIntensitiesMenu.addAction(self.normalizeByMaxAction)
        self.normalizeToFloatAction.setChecked(True)
        if 'how_normIntensities' in self.df_settings.index:
            how = self.df_settings.at['how_normIntensities', 'value']
            for action in self.normalizeQActionGroup.actions():
                if action.text() == how:
                    action.setChecked(True)
                    break
        self.rescaleIntensMenu = ImageMenu.addMenu('Rescale intensities (LUT)')
        ImageMenu.addAction(self.invertBwAction)
        ImageMenu.addAction(self.addScaleBarAction)
        ImageMenu.addAction(self.addTimestampAction)
        ImageMenu.addSeparator()
        ImageMenu.addAction(self.shuffleCmapAction)
        ImageMenu.addAction(self.greedyShuffleCmapAction)
        ImageMenu.addAction(self.saveLabColormapAction)
        ImageMenu.addSeparator()
        ImageMenu.addAction(self.zoomToObjsAction)
        ImageMenu.addAction(self.zoomOutAction)
        ImageMenu.addAction(self.launch3dViewerAction)
        
        annotMenu = self.menuBar().addMenu('&Annotations')
        annotMenu.setToolTipsVisible(True)
        self.annotationsMenu = annotMenu
        annotMenu.addAction(self.scalingLabelsAction)
        annotMenu.addAction(self.highLowResAction)
        annotMenu.addAction(self.editTextIDsColorAction)
        annotMenu.addAction(self.annotObjTrackSettingsAction)
        annotMenu.addSeparator()
        annotMenu.addAction(self.resetAnnotOptionsAction)
        annotMenu.addSeparator()
        annotMenu.addAction(self.phaseTranslatorAction)
        annotMenu.addAction(self.applyPhasesCellAction)
        annotMenu.addSeparator()
        annotMenu.addAction(self.maskUnanalyzedFramesAction)
        
        helpMenu = self.menuBar().addMenu('&Help')
        self.helpMenu = helpMenu
        helpMenu.addAction(self.userGuideAction)
        helpMenu.addAction(self.quickStartAction)
        helpMenu.addAction(self.acdcDocsAction)
        helpMenu.addSeparator()
        helpMenu.addAction(self.focusOnSearchAction)
        
        self.searchWidget = widgets.ButtonSearchWidget(
            self, loadingType=None
        )
        self.menuBar().setCornerWidget(self.searchWidget)
        self.searchWidget.setMinimumHeight(
            self.searchWidget.sizeHint().height()
        )
    
    def gui_createToolBars(self):
        fileToolBar = self.addToolBar("File")
        fileToolBar.setMovable(False)
        fileToolBar.addAction(self.openFolderAction)
        # Saving counterpart of loading (only the tables can be edited)
        fileToolBar.addAction(self.saveTablesAction)
        fileToolBar.addAction(self.openFileAction)
        fileToolBar.addAction(self.showInExplorerAction)
        fileToolBar.addAction(self.reloadAction)
        fileToolBar.addSeparator()
        fileToolBar.addAction(self.loadConfigurationAction)
        fileToolBar.addAction(self.saveConfigurationAction)
        fileToolBar.addAction(self.saveConfigurationAsAction)
        self.fileToolBar = fileToolBar
        
        # Navigation toolbar with the 'Viewer' mode tools as in guiWin
        navigateToolBar = widgets.ToolBar("Navigation", self)
        navigateToolBar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.addToolBar(navigateToolBar)
        self.navigateToolBar = navigateToolBar
        
        self.timepointsToolBar = widgets.ToolBar("Time points", self)
        self.timepointsToolBar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.timepointsToolBar.addAction(self.addTimepointIdentifierAction)
        self.timepointsToolBar.addAction(self.reevaluateAllTimepointsAction)
        self.timepointsToolBar.addAction(self.reevaluateCellTimepointsAction)
        # self.timepointsToolBar.addAction(self.editCellConstantsAction)
        self.timepointsToolBar.addSeparator()
        self.updateReevaluateTimepointsActions()
        self.addToolBar(self.timepointsToolBar)
        
        self.gui_createAnnotateToolbar(area=Qt.TopToolBarArea, visible=True)
        self.annotateToolbar.addAction(
            self.reevaluateAllCustomAnnotationsAction
        )
        self.annotateToolbar.addAction(
            self.reevaluateCellCustomAnnotationsAction
        )
        self.annotateToolbar.addSeparator()
        self.updateReevaluateCustomAnnotationActions()

        navigateToolBar.addAction(self.searchAction)
        
        self.zoomRectButton = QToolButton(self)
        self.zoomRectButton.setIcon(QIcon(":zoom_rect.svg"))
        self.zoomRectButton.setCheckable(True)
        self.zoomRectButton.setShortcut('Shift+Z')
        self.zoomRectButton.setToolTip(
            'Zoom to rectangular area\n\n'
            'Draw a rectangle with left-click to zoom into the area.\n'
            'Right-click to zoom back to the previous range.\n\n'
            'Shortcut: "Shift+Z"'
        )
        navigateToolBar.addWidget(self.zoomRectButton)
        self.leftClickButtons.append(self.zoomRectButton)
        self.widgetsWithShortcut['Zoom to rectangular area'] = (
            self.zoomRectButton
        )
        
        self.slideshowButton = QToolButton(self)
        self.slideshowButton.setIcon(QIcon(":eye-plus.svg"))
        self.slideshowButton.setCheckable(True)
        self.slideshowButton.setShortcut('Ctrl+W')
        self.slideshowButton.setToolTip(
            'Open a new window to browse frames or channels\n\n'
            'Shortcut: "Ctrl+W"'
        )
        navigateToolBar.addWidget(self.slideshowButton)
        
        navigateToolBar.addAction(self.launch3dViewerAction)
        
        self.overlayButton = widgets.rightClickToolButton(parent=self)
        self.overlayButton.setIcon(QIcon(":overlay.svg"))
        self.overlayButton.setCheckable(True)
        self.overlayButton.setToolTip(
            'Overlay channels\' images.\n\n'
            'Right-click on the button to overlay additional channels.\n\n'
            'Use the "Alpha" sliders to adjust the opacity.'
        )
        self.overlayButtonAction = navigateToolBar.addWidget(
            self.overlayButton
        )
        
        self.countObjsButton = QToolButton(self)
        self.countObjsButton.setIcon(QIcon(":count_objects.svg"))
        self.countObjsButton.setCheckable(True)
        self.countObjsButton.setShortcut('Ctrl+Shift+C')
        self.countObjsButton.setToolTip(
            'Count objects in the current view\n\n'
            'Shortcut: "Ctrl+Shift+C"'
        )
        navigateToolBar.addWidget(self.countObjsButton)
        
        self.overlayLabelsButton = widgets.rightClickToolButton(parent=self)
        self.overlayLabelsButton.setIcon(QIcon(":overlay_labels.svg"))
        self.overlayLabelsButton.setCheckable(True)
        self.overlayLabelsButton.setToolTip(
            'Add contours/masks of additional segmentation files.\n\n'
            'Right-click on the button to select which segmentation files '
            'to overlay.'
        )
        self.overlayLabelsButtonAction = navigateToolBar.addWidget(
            self.overlayLabelsButton
        )
        
        self.rulerButton = QToolButton(self)
        self.rulerButton.setIcon(QIcon(":ruler.svg"))
        self.rulerButton.setCheckable(True)
        self.rulerButton.setToolTip(
            'Draw a straight line and show its length.\n'
            'Length is displayed on the bottom-right corner.\n\n'
            'Hold "Ctrl" to snap the line to the closest 15° angle.'
        )
        navigateToolBar.addWidget(self.rulerButton)
        self.leftClickButtons.append(self.rulerButton)
        
        self.overlayColorButton = pg.ColorButton(self, color=(230,230,230))
        self.overlayColorButton.setDisabled(True)
        self.overlayColorButton.hide()
        
        self.overlayToolbar = widgets.OverlayToolbar(parent=self)
        self.addToolBar(Qt.TopToolBarArea, self.overlayToolbar)
        self.overlayToolbar.setVisible(False)
        
        self.phaseTranslatorToolbar = widgets.ToolBar("Phases", self)
        self.phaseTranslatorToolbar.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.phaseTranslatorToolbar.addAction(self.phaseTranslatorAction)
        self.phaseTranslatorToolbar.addAction(self.applyPhasesCellAction)
        self.addToolBar(Qt.TopToolBarArea, self.phaseTranslatorToolbar)
        
        self.setEnabledFileToolbar(False)
    
    def gui_connectActions(self):
        self.openFolderAction.triggered.connect(self.openFolder)
        self.openFileAction.triggered.connect(self.openFile)
        self.showInExplorerAction.triggered.connect(self.showInExplorer_cb)
        self.reloadAction.triggered.connect(self.reloadButtonClicked)
        self.phaseTranslatorAction.triggered.connect(
            self.configurePhaseTranslator
        )
        self.applyPhasesCellAction.triggered.connect(
            self.applyPhasesOnDisplayedCell
        )
        self.maskUnanalyzedFramesAction.toggled.connect(
            self.maskUnanalyzedFramesToggled
        )
        self.userGuideAction.triggered.connect(self.openUserGuide)
        self.quickStartAction.triggered.connect(self.showQuickStart)
        self.acdcDocsAction.triggered.connect(self.openAcdcDocs)
        self.exitAction.triggered.connect(self.close)
        self.loadConfigurationAction.triggered.connect(self.loadConfiguration)
        self.saveTablesAction.triggered.connect(self.saveTables)
        self.saveConfigurationAction.triggered.connect(self.saveConfiguration)
        self.saveConfigurationAsAction.triggered.connect(self.saveConfigurationAs)
        self.addTimepointIdentifierAction.triggered.connect(
            self.addTimepointIdentifier
        )
        self.reevaluateAllTimepointsAction.triggered.connect(
            self.reevaluateAllTimepoints
        )
        self.reevaluateCellTimepointsAction.triggered.connect(
            self.reevaluateDisplayedCellTimepoints
        )
        self.reevaluateAllCustomAnnotationsAction.triggered.connect(
            self.reevaluateAllCustomAnnotations
        )
        self.reevaluateCellCustomAnnotationsAction.triggered.connect(
            self.reevaluateDisplayedCellCustomAnnotations
        )
        # self.editCellConstantsAction.triggered.connect(self.editCellConstants)
        self.gui_connectCustomAnnotationsActions()
        
        self.scalingLabelsAction.toggled.connect(self.scalingLabelsToggled)
        self.highLowResAction.toggled.connect(self.highLowResToggled)
        self.editTextIDsColorAction.triggered.connect(
            self.toggleTextIDsColorButton
        )
        self.annotObjTrackSettingsAction.triggered.connect(
            self.setAnnotateObjTrackSettings
        )
        self.resetAnnotOptionsAction.triggered.connect(
            self.resetDefaultAnnotationOptions
        )
        
        self.toggleColorSchemeAction.triggered.connect(
            self.onToggleColorScheme
        )
        self.editShortcutsAction.triggered.connect(self.editShortcuts_cb)
        self.editOverlayColorAction.triggered.connect(
            self.toggleOverlayColorButton
        )
        self.viewCcaTableAction.triggered.connect(self.viewCcaTable)
        self.imgPropertiesAction.triggered.connect(self.editImgProperties)
        self.invertBwAction.toggled.connect(self.invertBw)
        self.invertBwAction.toggled.connect(self.recolorSignalsForBackground)
        self.addScaleBarAction.toggled.connect(self.addScaleBar)
        self.addTimestampAction.toggled.connect(self.addTimestamp)
        self.shuffleCmapAction.triggered.connect(self.shuffle_cmap)
        self.greedyShuffleCmapAction.triggered.connect(self.greedyShuffleCmap)
        self.saveLabColormapAction.triggered.connect(self.saveLabelsColormap)
        self.normalizeQActionGroup.triggered.connect(
            self.normaliseIntensitiesActionTriggered
        )
        self.zoomToObjsAction.triggered.connect(self.zoomToObjsActionCallback)
        self.zoomOutAction.triggered.connect(self.zoomOut)
        self.launch3dViewerAction.toggled.connect(
            self.launch3dViewerActionToggled
        )
        self.searchAction.triggered.connect(
            partial(self.focusOnSearch, blinking=True)
        )
        # Blink like the toolbar button: the focus alone is easy to miss
        self.focusOnSearchAction.triggered.connect(
            partial(self.focusOnSearch, blinking=True)
        )
        self.addAction(self.focusOnSearchAction)
        self.searchWidget.sigTriggerBlink.connect(self.onSearchTriggerBlink)
        self.searchWidget.sigSearchId.connect(self.onSearchId)
        self.nextAction.triggered.connect(self.nextActionTriggered)
        self.prevAction.triggered.connect(self.prevActionTriggered)
        
        self.zoomRectButton.toggled.connect(self.zoomRectActionToggled)
        self.slideshowButton.toggled.connect(self.launchSlideshow)
        self.overlayButton.toggled.connect(self.overlay_cb)
        self.overlayButton.sigRightClick.connect(self.showOverlayContextMenu)
        self.countObjsButton.toggled.connect(self.countObjectsCb)
        self.overlayLabelsButton.toggled.connect(self.overlayLabels_cb)
        self.overlayLabelsButton.sigRightClick.connect(
            self.showOverlayLabelsContextMenu
        )
        self.rulerButton.toggled.connect(self.ruler_cb)
        self.overlayColorButton.sigColorChanging.connect(
            self.changeOverlayColor
        )
        self.overlayColorButton.sigColorChanged.connect(self.saveOverlayColor)
        self.overlayToolbar.sigSetTranspacency.connect(
            self.setOverlayTransparency
        )
        self.overlayToolbar.sigSetSingleChannel.connect(
            self.setOverlaySingleChannel
        )
    
    def setEnabledFileToolbar(self, enabled):
        for action in getattr(self, 'annotationsActions', ()):
            action.setEnabled(enabled)
        for action in getattr(self, 'viewerToolsActions', ()):
            action.setEnabled(enabled)
        self.navigateToolBar.setEnabled(enabled)
        alwaysEnabledActions = (
            self.openFolderAction, self.openFileAction,
            self.loadConfigurationAction
        )
        for action in self.fileToolBar.actions():
            if action in alwaysEnabledActions:
                continue
            action.setEnabled(enabled)
        if enabled and self.isDataLoaded:
            self.imgPropertiesAction.setEnabled(True)
            self.viewCcaTableAction.setEnabled(self.hasCcaAnnotations())
        else:
            self.imgPropertiesAction.setEnabled(False)
            self.viewCcaTableAction.setEnabled(False)
        
        # As in guiWin the 3D viewer is available only for z-stacks
        isZstack = (
            enabled and self.isDataLoaded and self.posData().SizeZ > 1
        )
        for widget in (
                getattr(self, 'launch3dViewerButton', None), 
                self.launch3dViewerAction
            ):
            if widget is not None:
                widget.setDisabled(not isZstack)
                widget.setVisible(isZstack)
        self.updateFunctionalitySearchRecords()

    def updateFunctionalitySearchRecords(self):
        """Expose only enabled functionality available in this viewer."""
        records = {}
        recentPathActions = tuple(self.openRecentMenu.actions())

        def add_record(name, target):
            if any(target is action for action in recentPathActions):
                return
            name = name.replace('&', '').strip().rstrip('.')
            if not name or name == 'Search for ID or functionality':
                return
            records[name.casefold()] = {
                'display': name,
                'search_name': name,
                'target': target,
            }

        def add_menu_actions(menu):
            if menu is self.openRecentMenu or menu.title() == 'Open Recent':
                return
            for action in menu.actions():
                if action.isSeparator():
                    continue
                submenu = action.menu()
                if submenu is not None:
                    add_menu_actions(submenu)
                elif action.isEnabled():
                    add_record(action.text(), action)

        for action in self.menuBar().actions():
            menu = action.menu()
            if menu is not None:
                add_menu_actions(menu)
            elif action.isEnabled():
                add_record(action.text(), action)

        for name, target in self.widgetsWithShortcut.items():
            targets = target if isinstance(target, (list, tuple)) else (target,)
            if all(member.isEnabled() for member in targets):
                add_record(name, target)

        for button in (
                getattr(self, name, None)
                for name in (
                    'zoomRectButton', 'slideshowButton', 'overlayButton',
                    'countObjsButton', 'overlayLabelsButton', 'rulerButton',
                    'normalizeCheckbox', 'refreshButton',
                )
            ):
            if (
                    button is not None
                    and button.isEnabled()
                    and not button.isHidden()
                ):
                name = button.text().strip()
                if not name:
                    tooltip = button.toolTip()
                    name = (
                        tooltip.splitlines()[0].strip()
                        if tooltip else ''
                    )
                add_record(name, button)

        self.searchWidget.setSearchRecords(records)
    
    def isZsliceNavigationActive(self):
        """Next/Previous step z-slices instead of frames (z-stacks only)."""
        return (
            bool(self.data) and self.posData().SizeZ > 1
            and self.zSliceCheckbox.isChecked()
        )

    def nextActionTriggered(self):
        if self.isZsliceNavigationActive():
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )
        else:
            self.navigateScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )

    def prevActionTriggered(self):
        if self.isZsliceNavigationActive():
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepSub
            )
        else:
            self.navigateScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepSub
            )
    
    def launch3dViewerActionToggled(self, checked):
        self.launch3dViewerButton.setChecked(checked)
    
    def normaliseIntensitiesActionTriggered(self, action):
        how = action.text()
        self.df_settings.at['how_normIntensities', 'value'] = how
        self.df_settings.to_csv(self.settings_csv_path)
        if not self.isDataLoaded:
            return
        self.updateAllImages()
        self.updateImageValueFormatter()
    
    def getHardcodedShortcuts(self):
        """Shortcuts displayed (not editable) in the shortcuts editor, as
        {key sequence: (name, widget)}, the same format as
        `guiWin.getHardcodedShortcuts`."""
        hard_shortcuts = {}
        actions = (
            self.openFolderAction, self.reloadAction, self.saveTablesAction,
            self.saveConfigurationAction, self.saveConfigurationAsAction,
            self.focusOnSearchAction, self.editShortcutsAction, 
            self.viewCcaTableAction, 
            self.shuffleCmapAction, self.greedyShuffleCmapAction,
            self.slideshowButton, self.countObjsButton
        )
        for widget in actions:
            shortcut = widget.shortcut()
            if shortcut.isEmpty():
                continue
            text = widget.text() or widget.toolTip().split('\n')[0]
            name = text.replace('&', '').strip()
            if name.endswith('...'):
                name = name[:-3]
            hard_shortcuts[shortcut.toString()] = (name, widget)
        return hard_shortcuts
    
    def showInExplorer_cb(self):
        posData = self.posData()
        myutils.showInExplorer(posData.images_path)

    def openUserGuide(self, checked=False):
        """Open the measurements viewer documentation in the web browser."""
        self.logger.info(
            f'Opening the measurements viewer user guide '
            f'({urls.measurements_viewer_docs_url})...'
        )
        myutils.browse_url(urls.measurements_viewer_docs_url)

    def openAcdcDocs(self, checked=False):
        self.logger.info(
            f'Opening the Cell-ACDC documentation ({urls.docs_homepage})...'
        )
        myutils.browse_url(urls.docs_homepage)

    def showQuickStart(self, checked=False):
        """Short how-to (Help menu and info button of the measurements
        panel) with a button that opens the user guide."""
        self.logger.info('Showing the measurements viewer quick start')
        msg = widgets.myMessageBox()
        msg.information(
            self, 'Measurements viewer: quick start',
            html_utils.paragraph(MEASUREMENTS_VIEWER_QUICK_START),
            url_to_open=urls.measurements_viewer_docs_url,
            open_url_button_text='Open user guide'
        )

    def openRecentFile(self, path):
        if self.loadingInProgress():
            return
        self.logger.info(f'Opening recent folder: {path}')
        self.addToRecentPaths(path)
        self.openFolder(folderpath=path)
    
    def openFolder(self, checked=False, folderpath=None):
        if self.loadingInProgress():
            return
        if not folderpath:
            folderpath = QFileDialog.getExistingDirectory(
                self, 'Select experiment, Position or Images folder', 
                self.getMostRecentPath()
            )
        if not folderpath:
            return
        
        self.logger.info(f'Opening folder "{folderpath}"...')
        self.addToRecentPaths(folderpath)
        self.addPathToOpenRecentMenu(folderpath)
        
        images_paths = get_images_paths(folderpath)
        if not images_paths:
            self.warnLoading(
                f'No Position or Images folders found in "{folderpath}".'
            )
            return
        
        self.loadImagesPaths(images_paths)
    
    def openFile(self, checked=False, file_path=None):
        if self.loadingInProgress():
            return
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 'Select image file from an Images folder', 
                self.getMostRecentPath(), 
                'Images (*.tif *.tiff *.npz *.npy *.h5);;All Files (*)'
            )
        if not file_path:
            return
        
        self.logger.info(f'Opening file "{file_path}"...')
        images_path = os.path.dirname(file_path)
        if os.path.basename(images_path) != 'Images':
            self.warnLoading(
                f'The file "{file_path}" is not in an "Images" folder.\n\n'
                'Create the Cell-ACDC data structure first (e.g., by opening '
                'the file in the main GUI).'
            )
            return
        
        self.addToRecentPaths(images_path)
        self.addPathToOpenRecentMenu(images_path)
        channel = get_channel_from_filepath(file_path)
        self.loadImagesPaths([images_path], channel=channel)
    
    def loadImagesPaths(self, images_paths, channel=None):
        if self.loadingInProgress():
            return
        self._isLoadingData = True
        try:
            self._loadImagesPaths(images_paths, channel=channel)
        finally:
            self._isLoadingData = False

    def _loadImagesPaths(self, images_paths, channel=None):
        if self.refuseLoadingWhileProcessing():
            return
        try:
            chNames, segm_endnames = get_channels_and_segm_endnames(
                images_paths[0]
            )
        except Exception as err:
            self.warnLoading(str(err))
            return
        
        if not segm_endnames:
            self.warnLoading(
                f'No segmentation files found in "{images_paths[0]}".'
            )
            return
        
        if channel is None:
            channel = self.selectItem(
                'Select channel', 'Select channel to display:', chNames
            )
        if channel is None:
            return
        
        segm_endname = self.selectItem(
            'Select segmentation file', 'Select segmentation file to load:', 
            segm_endnames
        )
        if segm_endname is None:
            return

        if not self.askSaveBeforeLoading():
            return
        # Processing may have been started from the non-blocking dialogs
        if self.refuseLoadingWhileProcessing():
            return

        self.logger.info(
            f'Loading channel "{channel}" and segmentation '
            f'"{segm_endname}" of {len(images_paths)} position(s): '
            + ', '.join(f'"{path}"' for path in images_paths)
        )
        # In a worker thread: the window stays responsive (Positions that
        # cannot be loaded are skipped and reported in the terminal)
        data = self.runInThread(
            load_display_positions, images_paths, channel,
            segm_endname=segm_endname, reportProgress=True,
            text=f'Loading {len(images_paths)} position(s)...'
        )

        if not data:
            self.warnLoading(
                'None of the selected Positions could be loaded '
                '(see terminal for details).'
            )
            return

        # Before the predictions of `setData`
        mask, _ = self.resolveMaskUnanalyzedFrames(
            data, default=self.maskUnanalyzedFrames
        )
        if mask is None:
            return

        self.isSegm3D = None
        self.setMaskUnanalyzedFrames(mask, update=False)
        with self.deferPredictorPredictions():
            # The session prediction state belongs to the replaced data
            # (e.g. reopened after "Don't save"): rebuild it from the
            # visited column of the loaded tables.
            self.resetAutoPredictionTracking()
            self.setData(data)
            # The annotations and time point identifiers persist across
            # loading: saved assignments and time points of visited cells
            # must survive auto-prediction (as in `loadConfiguration`).
            self.seedVisitedCustomAnnotPredictions(
                self.customAnnotNames(), data
            )
            self.seedVisitedTimepointPredictions(data=data)
        self.markConfigurationSaved()
        self.logger.info(f'Loaded {len(data)} position(s)')

    def runInThread(
            self, func, *args, text='Working...', reportProgress=False,
            **kwargs
        ):
        """Run `func(*args, **kwargs)` in a worker `QThread` and return its
        result, or re-raise the exception it raised.

        A local event loop keeps the window repainting meanwhile (a long
        task on the GUI thread makes it "Not Responding", and Windows then
        shows a frozen copy of it), with its input disabled and a wait
        cursor. `text` is logged and shown in the status bar, followed by
        the progress texts that `func` reports through its ``progress``
        keyword argument (passed with `reportProgress`) and by the elapsed
        time. `func` must not create or touch Qt widgets nor show dialogs.

        The other windows of the application (launcher, main GUI, other
        viewers) are disabled meanwhile too: a dialog opened from them would
        run its own local event loop inside this one, and this call could
        return (re-enabling the window) only once that dialog closed.
        """
        worker = _ThreadTaskWorker(func, args, kwargs, reportProgress)
        thread = QThread()
        worker.moveToThread(thread)
        loop = QEventLoop()
        thread.started.connect(worker.run)
        # Queued to this (GUI) thread: received only while `loop` runs
        thread.finished.connect(loop.quit)
        worker.sigProgress.connect(self.showThreadTaskProgress)
        wasEnabled = self.isEnabled()
        self.setEnabled(False)
        otherWindows = [
            window for window in QApplication.topLevelWidgets()
            if window is not self and window.isVisible()
            and window.isEnabled() and not self.isAncestorOf(window)
        ]
        for window in otherWindows:
            window.setEnabled(False)
        self.setCursor(Qt.WaitCursor)
        self.logger.info(text)
        self.statusBar().showMessage(text)
        t0 = time.perf_counter()
        try:
            thread.start()
            loop.exec_()
        finally:
            thread.wait()
            thread.deleteLater()
            # `worker` moved back to this thread at the end of `run`
            worker.deleteLater()
            self.unsetCursor()
            for window in otherWindows:
                try:
                    window.setEnabled(True)
                except RuntimeError:
                    # Deleted meanwhile
                    pass
            if wasEnabled:
                self.setEnabled(True)
            self.statusBar().clearMessage()
        elapsed = time.perf_counter() - t0
        action = text.rstrip('.').strip()
        if worker.error is not None:
            self.logger.info(f'{action} failed after {elapsed:.1f} s')
            raise worker.error
        self.logger.info(f'{action} done in {elapsed:.1f} s')
        return worker.result

    def showThreadTaskProgress(self, text):
        """Show (and log) a progress text of a `runInThread` task."""
        self.logger.info(text)
        self.statusBar().showMessage(text)

    def selectItem(self, title, text, items):
        if len(items) == 1:
            return items[0]
        
        win = widgets.QDialogListbox(
            title, text, items, multiSelection=False, parent=self
        )
        win.exec_()
        if win.cancel or not win.selectedItemsText:
            return
        return win.selectedItemsText[0]
    
    def warnLoading(self, text):
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Loading data failed', text)
    
    def setData(self, data, pos_i=0, frame_i=None, ID=None):
        if self.data:
            # Turn the tools off while the previous data is still loaded,
            # so that their slots (e.g. `overlay_cb`) can tear down normally
            self.uncheckViewerTools()
        self.data = list(data) if data else []
        self.prunePendingPhaseIDs()
        hasData = len(self.data) > 0
        # `pos_i` of previously loaded data might be out of range
        pos_i = min(max(pos_i, 0), len(self.data)-1) if hasData else 0
        self.pos_i = pos_i
        self.centralWidget().setEnabled(hasData)
        self.setEnabledFileToolbar(hasData)
        
        self.posCombobox.blockSignals(True)
        self.posCombobox.clear()
        for posData in self.data:
            self.posCombobox.addItem(
                getattr(posData, 'pos_foldername', str(posData))
            )
        self.posCombobox.blockSignals(False)
        
        if not hasData:
            isMultiPos = False
            self.posLabel.setVisible(isMultiPos)
            self.posCombobox.setVisible(isMultiPos)
            return
        
        relPath = getattr(self.data[0], 'relPath', '')
        self.setWindowTitle(f'Cell-ACDC - Measurements viewer - {relPath}')
        self.pos_i = pos_i
        self.setPosDataDefaults()
        self.initLut()
        self.initViewerTools()
        self.loadPosition(pos_i, frame_i=frame_i, ID=ID)
        # As the main GUI (`loadingDataCompleted`): show the custom
        # annotations of every button, not only of the active one
        # (unchecking the eye button of the toolbar shows only the active
        # one and those whose "Hide annotations" option is off)
        self.viewAllCustomAnnotAction.setChecked(True)
        self.initViewerToolsAfterLoading()
        self.setEnabledFileToolbar(hasData)
        self.resizeBottomLayout()
        # On the first pass of the event loop, i.e. once the load finished:
        # the synchronous work after `setData` (e.g. `loadOverlayData` in
        # `loadConfiguration`) would otherwise delay the blink's timers
        # until it is over, hiding the blink.
        QTimer.singleShot(0, self.blinkCellIdControls)

    def blinkCellIdControls(self):
        """Blink the Cell ID spinbox and scrollbar after loading data, to
        show where the selected (highlighted) cell is changed."""
        if not self.data:
            return
        blinker = qutils.QControlBlink(
            [self.idLabel, self.idSpinBox, self.idScrollBar], qparent=self
        )
        blinker.start()
        self.statusBar().showMessage(
            'Select the cell to plot with the "Cell ID" controls (bottom) '
            'or click on a cell in the image.', 10000
        )
    
    def resizeBottomLayout(self):
        # As in `gui.guiWin.resizeGui`: fit the bottom layout to its 
        # content and then restore the height saved by the user
        self.bottomScrollArea._resizeVertical()
        self.setBottomLayoutHeight()
    
    def uncheckViewerTools(self):
        """Uncheck the viewer tools (with signals, so that their slots run
        their teardown). Called by `setData` while the previous data is
        still loaded."""
        for button in (
                self.overlayButton, self.overlayLabelsButton,
                self.rulerButton, self.zoomRectButton, self.countObjsButton,
                self.slideshowButton
            ):
            if button.isChecked():
                button.setChecked(False)
        if self.launch3dViewerAction.isChecked():
            self.launch3dViewerAction.setChecked(False)

    def resetViewerToolsWidgets(self):
        """Uncheck the tools and remove the overlay items of previously
        loaded data. The tools are normally already unchecked by `setData`
        while the previous data was still loaded (`uncheckViewerTools`)."""
        self.uncheckViewerTools()
        self.overlayToolbar.setVisible(False)
        for widget in self.overlayZsliceWidgets():
            widget.hide()
        
        self.removeOverlayItems()
        # Re-added by `gui_createOverlayItems`
        self.ax1.removeItem(self.rgbaImg1)
        for items in self.overlayLayersItems.values():
            imageItem, lutItem, alphaSB = items[:3]
            self.ax1.removeItem(imageItem)
            for widget in (alphaSB, alphaSB.label):
                self.bottomLeftLayout.removeWidget(widget)
                widget.deleteLater()
        alphaScrollbar = getattr(self.img1, 'alphaScrollbar', None)
        if alphaScrollbar is not None:
            for widget in (alphaScrollbar, alphaScrollbar.label):
                self.bottomLeftLayout.removeWidget(widget)
                widget.deleteLater()
            self.img1.alphaScrollbar = None
        self.alphaScrollbarRow = self.firstAlphaScrollbarRow
        
        for items in self.overlayLabelsItems.values():
            imageItem, contoursItem, gradItem = items
            self.ax1.removeItem(imageItem)
            self.ax1.removeItem(contoursItem)
            self.mainLayout.removeWidget(gradItem)
            gradItem.deleteLater()
        if hasattr(self, 'overlayLabelsContoursImages'):
            del self.overlayLabelsContoursImages
        
        self.overlayLayersItems = {}
        self.overlayLabelsItems = {}
        self.drawModeOverlayLabelsChannels = {}
        self.fluoDataChNameActions = []
        self.checkedOverlayChannels = set()
        self.overlayButton.setStyleSheet('')
        self.lutItemsLayout.addItem(self.imgGrad, row=0, col=0)
        
        if hasattr(self, 'zoomRectItem'):
            self.ax1.removeItem(self.zoomRectItem)
        self.tempSegmentON = False
        self.isMouseDragImg1 = False
        self.ax1_rulerPlotItem.setData([], [])
        self.ax1_rulerAnchorsItem.setData([], [])
        self.clearCursors()
    
    def initViewerTools(self):
        """Initialize the overlay of other channels and segmentation files 
        as in `gui.guiWin` after loading data"""
        self.resetViewerToolsWidgets()
        
        posData = self.posData()
        self.isSnapshot = posData.SizeT == 1
        try:
            self.ch_names, _ = get_channels_and_segm_endnames(
                posData.images_path
            )
        except Exception as err:
            self.ch_names = [self.user_ch_name]
        if self.user_ch_name not in self.ch_names:
            self.ch_names = [self.user_ch_name, *self.ch_names]
        
        existingSegmEndNames = set()
        for _posData in self.data:
            try:
                segm_files = load.get_segm_files(_posData.images_path)
                existingSegmEndNames.update(
                    load.get_endnames(_posData.basename, segm_files)
                )
            except Exception as err:
                # posData without files on disk --> no overlay labels
                pass
        self.existingSegmEndNames = existingSegmEndNames
        
        for _posData in self.data:
            _posData.ol_data_dict = {}
            _posData.ol_data = None
            _posData.ol_labels_data = None
            for ch in self.ch_names:
                _posData.lutLevels.setdefault(ch, {})
        
        self.setOverlayColors()
        self.createOverlayContextMenu()
        self.userChNameAction = QAction(self)
        self.userChNameAction.setCheckable(True)
        self.userChNameAction.setText(self.user_ch_name)
        self.gui_createOverlayColors()
        self.gui_createOverlayItems()
        
        self.createOverlayLabelsContextMenu(existingSegmEndNames)
        self.createOverlayLabelsItems(existingSegmEndNames)
        self.gui_addOverlayLayerItems()
        
        self.chNamesQActionGroup = QActionGroup(self)
        self.chNamesQActionGroup.addAction(self.userChNameAction)
        self.userChNameAction.setChecked(True)
        self.addActionsLutItemContextMenu(self.imgGrad)
        # Scrollbar for opacity of img1 (when overlaying)
        self.img1.alphaScrollbar = self.addAlphaScrollbar(
            self.user_ch_name, self.img1
        )
        
        self.guiTabControl.addChannels([self.user_ch_name])
        self.showPropsDockButton.setDisabled(False)
        self.setPermanentGreedyCmapPreferences()
    
    def initViewerToolsAfterLoading(self):
        self.updateImageValueFormatter()
        self.initPixelSizePropsDockWidget()
        if self.invertBwAction.isChecked() and not self.invertBwAlreadyCalledOnce:
            self.invertBw(True)
        self.gui_createZoomRectItem()
    
    def gui_createGraphicsPlots(self):
        self.graphLayout = pg.GraphicsLayoutWidget()
        if self.invertBwAction.isChecked():
            self.graphLayout.setBackground(graphLayoutBkgrColor)
            self.titleColor = 'black'
        else:
            self.graphLayout.setBackground(darkBkgrColor)
            self.titleColor = 'white'
        
        # Layout with the LUT items of the image and overlaid channels
        self.lutItemsLayout = self.graphLayout.addLayout(row=0, col=0)
        
        self.ax1 = widgets.MainPlotItem(ax_number=0)
        self.ax1.invertY(True)
        self.ax1.setAspectLocked(True)
        self.ax1.hideAxis('bottom')
        self.ax1.hideAxis('left')
        self.graphLayout.addItem(self.ax1, row=0, col=1)
        
        self.imgGrad = widgets.myHistogramLUTitem(parent=self, name='image')
        self.imgGrad.restoreState(self.df_settings)
        # Main GUI-only features
        imgGradMenu = self.imgGrad.gradient.menu
        imgGradMenu.removeAction(self.imgGrad.fontSizeMenu.menuAction())
        self.lutItemsLayout.addItem(self.imgGrad, row=0, col=0)
        for action in self.imgGrad.rescaleActionGroup.actions():
            if action.text() == self.defaultRescaleIntensHow:
                action.setChecked(True)
            self.rescaleIntensMenu.addAction(action)
        
        # Labels colormap used for the segm. masks overlay (as in guiWin)
        self.labelsGrad = widgets.labelsGradientWidget(parent=self)
        try:
            self.labelsGrad.restoreState(self.df_settings)
        except Exception as err:
            self.logger.info(
                'Failed to restore previously used colormap. '
                'Using default colormap "viridis"'
            )
            self.labelsGrad.item.loadPreset('viridis')
        self.labelsGrad.showRightImgAction.setChecked(False)
        self.hideRightImageLabelsGradActions()
        
        self.img1 = widgets.ParentImageItem()
        self.imgGrad.setImageItem(self.img1)
        self.img1.lutItem = self.imgGrad
        self.imgGrad.sigRescaleIntes.connect(self.rescaleIntensitiesLut)
        self.imgGrad.gradient.sigGradientChangeFinished.connect(
            self.imgGradLUTfinished_cb
        )
        self.ax1.addBaseImageItem(self.img1)
        
        # RGBA image for true transparency mode (added in 
        # `gui_createOverlayItems`)
        self.rgbaImg1 = pg.ImageItem()
        
        # Overwrite axes viewbox context menu
        self.ax1.vb.menu = self.imgGrad.gradient.menu
        
        self.gui_createAnnotationsItems()
        
        # Ruler plotItem and scatterItem
        rulerPen = pg.mkPen(color='r', style=Qt.DashLine, width=2)
        self.ax1_rulerPlotItem = widgets.RulerPlotItem(pen=rulerPen)
        self.ax1_rulerAnchorsItem = pg.ScatterPlotItem(
            symbol='o', size=9,
            brush=pg.mkBrush((255,0,0,50)),
            pen=pg.mkPen((255,0,0), width=2), tip=None
        )
        self.ax1.addItem(self.ax1_rulerPlotItem)
        self.ax1.addItem(self.ax1_rulerPlotItem.labelItem)
        self.ax1.addItem(self.ax1_rulerAnchorsItem)
        
        # Mirrored cursor from the slideshow window
        self.ax1_cursor = pg.ScatterPlotItem(
            symbol='d', pxMode=True, 
            pen=pg.mkPen('w', width=1),
            brush=pg.mkBrush((255, 255, 255, 70)), 
            size=16, tip=None
        )
        self.ax1.addItem(self.ax1_cursor)

        # Overlay channel layers (z-value 0) are added later, when data is
        # loaded, and annotation layers are at 0.5-0.6: keep the ruler and
        # the mirrored cursor above both, below the text IDs (z-value 1).
        # Items created on demand get the same z-value in
        # `raiseOnDemandAx1Items`.
        topItems = (
            self.ax1_rulerPlotItem, self.ax1_rulerPlotItem.labelItem,
            self.ax1_rulerAnchorsItem, self.ax1_cursor,
        )
        for item in topItems:
            item.setZValue(0.9)

        self.img1.hoverEvent = self.gui_hoverEventImg1
        self.img1.mousePressEvent = self.gui_mousePressEventImg1
        self.img1.mouseMoveEvent = self.gui_mouseDragEventImg1
        self.img1.mouseReleaseEvent = self.gui_mouseReleaseEventImg1
        self.img1.mouseClickEvent = self.gui_mouseClickEventImg1
        self.ax1.sigRangeChanged.connect(self.viewRangeChanged)
        
        self.labelsGrad.sigGradientChangeFinished.connect(self.updateLabelsCmap)
        self.labelsGrad.sigShuffleCmap.connect(self.shuffle_cmap)
        self.labelsGrad.sigGreeedyShuffleCmap.connect(self.greedyShuffleCmap)
        self.labelsGrad.permanentGreedyCmapAction.toggled.connect(
            self.permanentGreedyCmapToggled
        )
        self.labelsGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)
        self.labelsGrad.defaultSettingsAction.triggered.connect(
            self.restoreDefaultSettings
        )
        self.imgGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)
        self.imgGrad.defaultSettingsAction.triggered.connect(
            self.restoreDefaultSettings
        )
        self.imgGrad.sigAddScaleBar.connect(self.addScaleBarAction.setChecked)
        self.imgGrad.sigAddTimestamp.connect(self.addTimestampAction.setChecked)
    
    def hideRightImageLabelsGradActions(self):
        # The viewer has no right image (segm. image, duplicated left image, 
        # or next frame), hence the actions that only affect it are hidden
        lg = self.labelsGrad
        rightImageButtons = (lg.colorButton, lg.textColorButton)
        for action in lg.menu.actions():
            if not isinstance(action, QWidgetAction):
                continue
            widget = action.defaultWidget()
            if widget is None:
                continue
            if any(widget.isAncestorOf(b) for b in rightImageButtons):
                action.setVisible(False)
        
        for action in (
                lg.showLabelsImgAction, lg.showRightImgAction, 
                lg.showNextFrameAction, lg.fontSizeMenu.menuAction()
            ):
            action.setVisible(False)
    
    def gui_createAnnotationsItems(self):
        """Create the annotations items as in `gui.guiWin` (left image)"""
        # Text annotations color
        self.textIDsColorButton = pg.ColorButton(self)
        self.textIDsColorButton.hide()
        if 'textIDsColor' in self.df_settings.index:
            rgbString = self.df_settings.at['textIDsColor', 'value']
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.gui_createTextAnnotColors(r, g, b, custom=True)
            self.textIDsColorButton.setColor((r, g, b))
        else:
            self.gui_createTextAnnotColors(0, 0, 0, custom=False)
        
        self.gui_createContourPens()
        self.gui_createMothBudLinePens()
        
        # Overlay segm. masks item
        self.labelsLayerImg1 = widgets.BaseLabelsImageItem()
        self.ax1.addItem(self.labelsLayerImg1)
        self.setLut()
        self.initLabelsImageItems()
        
        self.ax1_contoursImageItem = pg.ImageItem()
        self.ax1_oldMothBudLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.oldMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax1_newMothBudLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.newMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax1_movementAgainstPrevLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.oldMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax1_movementAgainstPrevScatterItem = (
            self.gui_getLostObjScatterItem()
        )
        # Required to initialize the tracked lost objects text color
        self.gui_getTrackedLostObjScatterItem()
        
        self.ax1.addItem(self.ax1_contoursImageItem)
        self.ax1.addItem(self.ax1_oldMothBudLinesItem)
        self.ax1.addItem(self.ax1_newMothBudLinesItem)
        self.ax1.addItem(self.ax1_movementAgainstPrevLinesItem)
        self.ax1.addItem(self.ax1_movementAgainstPrevScatterItem)
        
        # Selected ID highlight items (same as searched ID in guiWin)
        self.highLightIDLayerImg1 = pg.ImageItem()
        self.ax1.addItem(self.highLightIDLayerImg1)
        self.searchedIDitemLeft = pg.ScatterPlotItem()
        self.searchedIDitemLeft.setData(
            [], [], symbol='s', pxMode=False, size=1,
            brush=pg.mkBrush(color=(255,0,0,150)),
            pen=pg.mkPen(width=3, color='r'), tip=None
        )
        self.ax1.addItem(self.searchedIDitemLeft)

        # Overlay channel layers are added later (when data is loaded) with
        # the default z-value 0, which would draw them above these items.
        # Keep annotations above every image layer and below the text IDs
        # (z-value 1, see `gui_annotations`). Custom annotation symbols are
        # at 0.6 (see `addCustomAnnnotScatterPlot`), the ruler and the
        # mirrored slideshow cursor at 0.9 (see `gui_createGraphicsPlots`).
        annotationItems = (
            self.labelsLayerImg1, self.ax1_contoursImageItem,
            self.ax1_oldMothBudLinesItem, self.ax1_newMothBudLinesItem,
            self.ax1_movementAgainstPrevLinesItem,
            self.ax1_movementAgainstPrevScatterItem,
            self.highLightIDLayerImg1, self.searchedIDitemLeft,
        )
        for i, item in enumerate(annotationItems):
            item.setZValue(0.5 + i * 0.01)

        self.textIDsColorButton.sigColorChanging.connect(
            self.updateTextAnnotColor
        )
        self.textIDsColorButton.sigColorChanged.connect(self.saveTextIDsColors)
        self.imgGrad.textColorButton.disconnect()
        self.imgGrad.textColorButton.clicked.connect(
            self.editTextIDsColorAction.trigger
        )
        self.imgGrad.labelsAlphaSlider.valueChanged.connect(
            self.updateLabelsAlpha
        )
    
    def initLabelsImageItems(self):
        lut = self.getLabelsImageLut()
        self.labelsLayerImg1.setLevels([0, len(lut)])
        self.labelsLayerImg1.setLookupTable(lut)
        alpha = self.imgGrad.labelsAlphaSlider.value()
        self.labelsLayerImg1.setOpacity(alpha)
    
    def updateLabelsAlpha(self, value):
        self.df_settings.at['overlaySegmMasksAlpha', 'value'] = value
        self.df_settings.to_csv(self.settings_csv_path)
        self.labelsLayerImg1.setOpacity(value)
    
    def gui_createAxesTextAnnotItems(self, allIDs):
        if 0 in self.textAnnot:
            self.textAnnot[0].removeFromPlotItem(self.ax1)
        self.gui_createTextAnnotItems(allIDs)
        self.gui_setTextAnnotColors()
        self.textAnnot[0].addToPlotItem(self.ax1)
        self.initTextAnnot()
        self.syncTextAnnotOptions()
    
    def syncTextAnnotOptions(self):
        """Set the text annotations flags from the annotation checkboxes"""
        for ax in self.annotatedAxesIndices():
            if ax not in self.textAnnot:
                continue
            textAnnot = self.textAnnot[ax]
            textAnnot.setLabelAnnot(self.annotIDsCheckbox(ax).isChecked())
            textAnnot.setCcaAnnot(self.isCellCycleInfoChecked(ax))
            textAnnot.setLineageAnnot(
                self.annotLineageInfoCheckbox(ax).isChecked()
            )
            textAnnot.setScaling(self.scalingLabelsAction.isChecked())
    
    def restoreViewerAnnotationOptions(self):
        """Restore the saved annotation options without triggering the 
        checkboxes callbacks (data might not be loaded yet)."""
        checkboxes = [
            checkbox for checkbox in self.annotOptionsCheckboxes[0].values()
            if hasattr(checkbox, 'setChecked')
        ]
        for checkbox in checkboxes:
            checkbox.blockSignals(True)
        try:
            self.restoreSavedAnnotationOptions()
        finally:
            for checkbox in checkboxes:
                checkbox.blockSignals(False)
        self.syncTextAnnotOptions()
    
    def resetDefaultAnnotationOptions(self, *args):
        super().resetDefaultAnnotationOptions(*args)
        self.syncTextAnnotOptions()
        if self.isDataLoaded:
            self.updateImages()
    
    def resetFocus(self):
        self.setFocus()
    
    def setTitleText(self, *args, **kwargs):
        # The viewer does not have the title label of the main GUI
        pass
    
    def setLookingGoodTitle(self):
        pass
    
    def changeFontSize(self):
        fontSize = self.fontSizeSpinBox.value()
        if fontSize == self.fontSize:
            return
        
        self.fontSize = fontSize
        self._set_setting_value('fontSize', self.fontSize)
        if not self.isDataLoaded:
            return
        
        for ax in self.annotatedAxesIndices():
            self.textAnnot[ax].changeFontSize(self.fontSize)
        self.updateImages()
    
    def changeTextResolution(self):
        if not self.isDataLoaded:
            return
        
        mode = 'high' if self.highLowResAction.isChecked() else 'low'
        posData = self.posData()
        img_shape = self.img1.image.shape[:2]
        scalingMode = self.scalingLabelsAction.isChecked()
        for ax in self.annotatedAxesIndices():
            self.textAnnot[ax].changeResolution(
                mode, posData.allIDs, self.ax1, img_shape, 
                scalingMode=scalingMode
            )
        self.updateImages()
    
    def highLowResToggled(self, checked=True):
        self._set_setting_value('highLowRes', self.highLowResAction.isChecked())
        self.changeTextResolution()
    
    def scalingLabelsToggled(self, checked=True):
        self._set_setting_value(
            'scalingLabels', self.scalingLabelsAction.isChecked()
        )
        if not self.isDataLoaded:
            return
        for annot in self.textAnnot.values():
            annot.setScaling(self.scalingLabelsAction.isChecked())
    
    def imgGradLUTfinished_cb(self):
        self.df_settings = self.imgGrad.saveState(self.df_settings)
        self.df_settings.to_csv(self.settings_csv_path)
    
    def initLut(self):
        """Initialize LUT levels as in `gui.guiWin` after loading data."""
        self.isDataLoaded = True
        posData = self.posData()
        self.user_ch_name = posData.user_ch_name
        self.rescaleIntensChannelHowMapper.setdefault(
            self.user_ch_name, 'Rescale each 2D image'
        )
        for _posData in self.data:
            if not hasattr(_posData, 'lutLevels'):
                _posData.lutLevels = {self.user_ch_name: {}}
        self.imgGrad.setAxisLabel(self.user_ch_name)
        self.img1.preComputedMinMaxValues(self.data)
        self.imgGrad.rescaleAcrossZstackAction.setDisabled(
            posData.SizeZ == 1
        )
        self.initImgGradRescaleIntensitiesHowPreference()
        self.rescaleIntensitiesLut(setImage=False)
    
    def gui_createNavigationWidgets(self):
        self.gui_createFramesNavigationWidgets()
        self.t_label.setText('Frame n.')
        self.gui_createZsliceNavigationWidgets()
        # z-projection lock is a main GUI-only feature
        self.zProjLockViewButton.hide()
        
        self.zSliceOverlay_SB = widgets.ScrollBar(Qt.Horizontal)
        self.overlay_z_label = QLabel('Overlay z-slice  ')
        self.overlay_z_label.setFont(_font)
        self.overlay_z_label.setDisabled(True)
        self.zProjOverlay_CB = widgets.ComboBox()
        self.zProjOverlay_CB.setFont(_font)
        self.zProjOverlay_CB.addItems([
            'single z-slice', 'max z-projection', 'mean z-projection',
            'median z-proj.', 'same as above'
        ])
        self.zProjOverlay_CB.setCurrentIndex(4)
        self.zSliceOverlay_SB.setDisabled(True)
        self.zSliceOverlay_SB.setMinimumWidth(200)
        for widget in self.overlayZsliceWidgets():
            widget.hide()

        self.idScrollBar = widgets.ScrollBar(Qt.Horizontal)
        self.idScrollBar.setMinimum(0)
        self.idLabel = QLabel('Cell ID')
        self.idLabel.setFont(_font)
        self.idSpinBox = widgets.SpinBox(disableKeyPress=True)
        self.idSpinBox.setMinimum(0)
        
        idSpinBoxHight = int(self.idSpinBox.sizeHint().height() * 0.8)
        self.idScrollBar.setFixedHeight(idSpinBoxHight)
        self.numIDsLabel = QLabel('')
        self.idNotPresentLabel = QLabel('')
        self.idNotPresentLabel.setFont(_font)
        self.followIDcheckbox = QCheckBox('Follow cell')
        self.followIDcheckbox.setToolTip(
            'If active, the view zooms on the selected\n'
            'cell also when changing frame or z-slice'
        )
        self.restrictFramesCheckbox = QCheckBox(
            'Only frames with selected cell'
        )
        self.restrictFramesCheckbox.setToolTip(
            'If active, the frames slider is restricted to the\n'
            'range of frames where the selected cell ID is present'
        )
        self.rescalePlotCheckbox = QCheckBox('Rescale plot')
        self.rescalePlotCheckbox.setToolTip(
            'If active, the plot is rescaled when changing Cell ID'
        )
        
        self.normalizeCheckbox = QCheckBox('Normalize measurements (0-1)')
        self.normalizeCheckbox.setToolTip(
            'Rescale each measurement to the range 0-1 to\n'
            'compare measurements with different scales'
        )
        self.showPhasesCheckbox = QCheckBox('Show phases')
        self.showPhasesCheckbox.setToolTip(
            'Show the evaluated phases (phase column) as a band below the\n'
            'time course and as point colors in Measurement vs Measurement.\n'
            'Set colors in the time point to phase translator.'
        )
        self.showPhasesCheckbox.setChecked(self._get_setting_value(
            'measurements_viewer_show_phases', True, cast=bool
        ))
        self.showPhasesCheckbox.setEnabled(False)
 
        
        self.posLabel = QLabel('Position')
        self.posCombobox = QComboBox()
        # Same font as the other labels of the label column and same look as
        # the other sliders of the slider column
        self.posLabel.setFont(_font)
        self.zSliceCheckbox.setFont(_font)
        self.zSliceOverlay_SB.setFixedHeight(idSpinBoxHight)

        # One left-aligned grid column each for the labels, spin boxes,
        # size labels and sliders (`NAV_*_COL`), so that they line up across
        # rows. The main GUI builds its own navigation layout (gui.py).
        navLayout = QGridLayout()
        labelCol, spinCol = self.NAV_LABEL_COL, self.NAV_SPINBOX_COL
        sizeCol, sliderCol = self.NAV_SIZE_COL, self.NAV_SLIDER_COL
        extraCol = self.NAV_EXTRA_COL
        row = 0
        navLayout.addWidget(self.posLabel, row, labelCol)
        # Spans to the slider column so that its width does not widen the
        # spin box column
        navLayout.addWidget(
            self.posCombobox, row, spinCol, 1, sliderCol - spinCol + 1,
            alignment=Qt.AlignLeft
        )

        row += 1
        navLayout.addWidget(self.t_label, row, labelCol)
        navLayout.addWidget(self.navSpinBox, row, spinCol)
        navLayout.addWidget(self.navSizeLabel, row, sizeCol)
        navLayout.addWidget(self.navigateScrollBar, row, sliderCol)
        # As the main GUI's last tracked frame label (`updateLastAnalyzedFrameLabel`)
        self.lastAnalyzedFrameLabel = QLabel('')
        self.lastAnalyzedFrameLabel.setFont(_font)
        self.lastAnalyzedFrameLabel.setToolTip(MASK_UNANALYZED_TOOLTIP)
        navLayout.addWidget(
            self.lastAnalyzedFrameLabel, row, extraCol,
            1, self.NAV_NUM_COLS - extraCol
        )

        row += 1
        navLayout.addWidget(self.zSliceCheckbox, row, labelCol)
        navLayout.addWidget(self.zSliceSpinbox, row, spinCol)
        navLayout.addWidget(self.SizeZlabel, row, sizeCol)
        navLayout.addWidget(self.zSliceScrollBar, row, sliderCol)
        navLayout.addWidget(self.zProjComboBox, row, extraCol)
        navLayout.addWidget(self.switchPlaneCombobox, row, extraCol + 1)
        navLayout.addWidget(self.launch3dViewerButton, row, extraCol + 2)

        row += 1
        # No spin box: the label (its text changes with the z-slice, see
        # `gui_overlay`) spans to the slider column
        navLayout.addWidget(
            self.overlay_z_label, row, labelCol, 1, sliderCol - labelCol
        )
        navLayout.addWidget(self.zSliceOverlay_SB, row, sliderCol)
        navLayout.addWidget(self.zProjOverlay_CB, row, extraCol)

        row += 1
        navLayout.addWidget(self.idLabel, row, labelCol)
        navLayout.addWidget(self.idSpinBox, row, spinCol)
        navLayout.addWidget(self.numIDsLabel, row, sizeCol)
        navLayout.addWidget(self.idScrollBar, row, sliderCol)

        # Below the grid, so that the overlay alpha sliders (grid rows added
        # by `addAlphaScrollbar`) follow the other sliders
        idOptionsLayout = QHBoxLayout()
        idOptionsLayout.addWidget(self.followIDcheckbox)
        idOptionsLayout.addWidget(self.restrictFramesCheckbox)
        idOptionsLayout.addWidget(self.rescalePlotCheckbox)

        idOptionsLayout.addWidget(self.normalizeCheckbox)
        idOptionsLayout.addWidget(self.showPhasesCheckbox)
        
        idOptionsLayout.addWidget(self.idNotPresentLabel)
        idOptionsLayout.addStretch(1)

        for scrollbar in self.navigationScrollbars():
            scrollbar.setMinimumWidth(200)

        # Only the sliders stretch
        for col in range(self.NAV_NUM_COLS):
            navLayout.setColumnStretch(col, int(col == sliderCol))

        # Overlay channels alpha scrollbars are added from this row on
        # (see `addAlphaScrollbar`)
        self.bottomLeftLayout = navLayout
        self.firstAlphaScrollbarRow = row + 1
        self.alphaScrollbarRow = row + 1
        self.h = self.navSpinBox.sizeHint().height()

        navGroupLayout = QVBoxLayout()
        navGroupLayout.addLayout(navLayout)
        navGroupLayout.addLayout(idOptionsLayout)
        self.navigationGroupbox = QGroupBox('Navigate')
        self.navigationGroupbox.setLayout(navGroupLayout)

    def addAlphaScrollbar(self, channelName, imageItem):
        """Add the alpha slider of an overlay channel
        (`OverlayChannelsGuiElements.addAlphaScrollbar`) and move it to the
        columns of the Navigate grid: label from the label column, slider
        in the slider column."""
        row = self.alphaScrollbarRow
        alphaScrollBar = super().addAlphaScrollbar(channelName, imageItem)
        # Same height as the other sliders of the column
        alphaScrollBar.setFixedHeight(self.idScrollBar.maximumHeight())
        label = alphaScrollBar.label
        for widget in (label, alphaScrollBar):
            self.bottomLeftLayout.removeWidget(widget)
        self.bottomLeftLayout.addWidget(
            label, row, self.NAV_LABEL_COL,
            1, self.NAV_SLIDER_COL - self.NAV_LABEL_COL
        )
        self.bottomLeftLayout.addWidget(
            alphaScrollBar, row, self.NAV_SLIDER_COL
        )
        return alphaScrollBar

    def overlayZsliceWidgets(self):
        return (self.overlay_z_label, self.zSliceOverlay_SB, self.zProjOverlay_CB)
    
    def navigationScrollbars(self):
        return (self.navigateScrollBar, self.zSliceScrollBar, self.idScrollBar)
    
    def gui_createAnnotationsWidgets(self):
        """Create the annotation options checkboxes and the quick settings 
        of `gui.guiWin` relevant for the display of the annotations."""
        self.gui_setupAnnotationOptionsLayout(0)
        
        self.annotLostObjsToggle = widgets.Toggle()
        annotLostObjsToggleTooltip = (
            'Toggle annotation of lost objects mode ON/OFF'
        )
        self.annotLostObjsToggle.setChecked(True)
        self.annotLostObjsToggle.setToolTip(annotLostObjsToggleTooltip)
        annotLostObjsLabel = QLabel('Annot. lost objects')
        annotLostObjsLabel.setToolTip(annotLostObjsToggleTooltip)
        self.annotLostObjsToggle.toggled.connect(self.annotLostObjsToggled)
        
        self.showAllContoursToggle = widgets.Toggle()
        showAllContoursTooltip = (
            'If active, all contours will be displayed, including\n'
            'inner contours (e.g. holes and sub-objects)'
        )
        self.showAllContoursToggle.setToolTip(showAllContoursTooltip)
        showAllContourLabel = QLabel('Show all contours')
        showAllContourLabel.setToolTip(showAllContoursTooltip)
        self.showAllContoursToggle.toggled.connect(
            self.showAllContoursToggled
        )
        
        self.fontSizeSpinBox = widgets.SpinBox()
        self.fontSizeSpinBox.setMinimum(1)
        self.fontSizeSpinBox.setMaximum(99)
        self.fontSizeSpinBox.setValue(self.fontSize)
        self.fontSizeSpinBox.editingFinished.connect(self.changeFontSize) 
        self.fontSizeSpinBox.sigUpClicked.connect(self.changeFontSize)
        self.fontSizeSpinBox.sigDownClicked.connect(self.changeFontSize)
        
        quickSettingsLayout = QHBoxLayout()
        quickSettingsLayout.addWidget(annotLostObjsLabel)
        quickSettingsLayout.addWidget(self.annotLostObjsToggle)
        quickSettingsLayout.addSpacing(15)
        quickSettingsLayout.addWidget(showAllContourLabel)
        quickSettingsLayout.addWidget(self.showAllContoursToggle)
        quickSettingsLayout.addSpacing(15)
        quickSettingsLayout.addWidget(QLabel('Font size'))
        quickSettingsLayout.addWidget(self.fontSizeSpinBox)
        quickSettingsLayout.addStretch(1)
        
        annotLayout = QVBoxLayout()
        annotLayout.addWidget(self.annotOptionsCheckboxes[0]['container'])
        annotLayout.addLayout(quickSettingsLayout)
        
        self.annotationsGroupbox = QGroupBox('Annotations')
        self.annotationsGroupbox.setLayout(annotLayout)
        
        self.restoreViewerAnnotationOptions()
    
    def gui_createMeasurementsPanel(self):
        self.measurementsPlot = pg.PlotWidget()
        if self.invertBwAction.isChecked():
            self.measurementsPlot.setBackground(graphLayoutBkgrColor)
        else:
            self.measurementsPlot.setBackground(darkBkgrColor)
        self.measurementsPlot.setLabel('bottom', 'Frame n.')
        self.measurementsPlot.showGrid(x=True, y=True, alpha=0.3)
        self.measurementsLegend = self.measurementsPlot.addLegend()
        
        self.currentFrameLine = pg.InfiniteLine(
            angle=90, movable=True, 
            pen=pg.mkPen(color=(255, 0, 0), width=2),
            hoverPen=pg.mkPen(color=(255, 100, 100), width=3)
        )
        self.currentFrameLine.setToolTip(
            'Current frame. Drag the line to change frame.'
        )
        self.currentFrameLine.setZValue(10)
        self.measurementsPlot.addItem(self.currentFrameLine, ignoreBounds=True)
        # Frames ignored by the calculations (`updateUnanalyzedRegions`)
        self.unanalyzedRegion = self.createUnanalyzedRegion()
        self.measurementsPlot.addItem(self.unanalyzedRegion, ignoreBounds=True)

        # Solid band of the selected cell's phases below the time course,
        # x-linked to the measurements plot
        self.phaseBandPlot = pg.PlotWidget()
        self.phaseBandPlot.setBackground(
            graphLayoutBkgrColor if self.invertBwAction.isChecked()
            else darkBkgrColor
        )
        self.phaseBandPlot.setFixedHeight(32)
        self.phaseBandPlot.setMouseEnabled(x=False, y=False)
        self.phaseBandPlot.setMenuEnabled(False)
        self.phaseBandPlot.hideButtons()
        self.phaseBandPlot.hideAxis('bottom')
        self.phaseBandPlot.getAxis('left').setStyle(showValues=False)
        self.phaseBandPlot.getAxis('left').setTicks([[]])
        self.phaseBandPlot.getAxis('left').setLabel('Phase')
        self.phaseBandPlot.setYRange(0, 1, padding=0)
        self.phaseBandPlot.setXLink(self.measurementsPlot)
        self.phaseBandFrameLine = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(color=(255, 0, 0), width=2)
        )
        self.phaseBandFrameLine.setZValue(10)
        self.phaseBandPlot.addItem(self.phaseBandFrameLine, ignoreBounds=True)
        self.phaseBandUnanalyzedRegion = self.createUnanalyzedRegion()
        self.phaseBandPlot.addItem(
            self.phaseBandUnanalyzedRegion, ignoreBounds=True
        )
        self.phaseBandItems = []
        self.phaseBandPlot.setVisible(False)
        # Keep the x positions aligned while the main left axis auto-sizes
        self.measurementsPlot.getAxis('left').geometryChanged.connect(
            self.syncPhaseBandAxisWidth
        )
        measurementsPlotLayout = QVBoxLayout()
        measurementsPlotLayout.setContentsMargins(0, 0, 0, 0)
        measurementsPlotLayout.setSpacing(0)
        measurementsPlotLayout.addWidget(self.measurementsPlot)
        measurementsPlotLayout.addWidget(self.phaseBandPlot)
        measurementsPlotContainer = QWidget()
        measurementsPlotContainer.setLayout(measurementsPlotLayout)
        
        self.columnsSearchLineEdit = QLineEdit()
        self.columnsSearchLineEdit.setPlaceholderText('Search column...')
        self.columnsSearchLineEdit.setClearButtonEnabled(True)
        
        self.columnsModel = widgets.MeasurementTreeModel(self)
        self.columnsSearchProxy = widgets.MeasurementTreeSearchProxy(self)
        self.columnsSearchProxy.setSourceModel(self.columnsModel)
        self.columnsSearchProxy.setDynamicSortFilter(True)
        self.columnsSearchProxy.sort(0, Qt.AscendingOrder)

        self.columnsListWidget = QTreeView()
        self.columnsListWidget.setHeaderHidden(True)
        self.columnsListWidget.setIndentation(6)
        self.columnsListWidget.setUniformRowHeights(True)
        self.columnsListWidget.setModel(self.columnsSearchProxy)
        self.columnsHighlightDelegate = _MeasurementColumnDelegate(
            self.columnsListWidget
        )
        self.columnsListWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.columnsListWidget.setItemDelegate(
            self.columnsHighlightDelegate
        )
        self.columnsListWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.columnsListWidget.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.columnsListWidget.setToolTip(
            'Click a column to plot/unplot it. Right-click on a\n'
            'column to change its color or conditional formatting.'
        )
        # Custom plotting colors set by the user {column: QColor}
        self.measurementsColors = {}
        # Index in SIGNAL_COLORS of the default color of each column that
        # was plotted, so that plotting another column does not recolor
        # the plotted ones {column: int}
        self._defaultSignalColorIndices = {}
        # Ordered rules {column: [{'condition': str, 'color': '#rrggbb'}]}
        self.conditionalFormatting = {}

        self.measurementsModeCombobox = QComboBox()
        self.measurementsModeCombobox.addItems(
            ['Time course', 'Measurement vs Measurement']
        )
        self.measurementsModeCombobox.setToolTip(
            'Time course: selected cell. Measurement '
            'vs Measurement: all cells and all frames\n'
            'of the current position. Hover for details; '
            'click a point to zoom to its cell.'
        )
        # Short how-to with a link to the user guide (also Help menu)
        self.measurementsHelpButton = widgets.infoPushButton()
        self.measurementsHelpButton.setToolTip(
            'How to use the measurements viewer (quick start and link to '
            'the user guide)'
        )
        self.measurementsHelpButton.clicked.connect(self.showQuickStart)
        self.measurementsAxes = {'X': '', 'Y': ''}
        self._measurementsMode = 0
        self._timeCourseColumns = set()
        
        self.refreshButton = widgets.reloadPushButton(' Reload table ')
        self.refreshButton.setToolTip(
            'Reload the acdc_output table from the file '
            '(e.g., after saving in the main GUI)'
        )
        self.processMeasurementsButton = widgets.setPushButton('Process measurements...')
        self.processMeasurementsButton.setToolTip(
            'Apply a processing recipe (e.g., smoothing, min-max normalization or\n'
            'ramp detection) to selected measurements of the current position.\n\n'
            'Each processed measurement is added as '
            'a new column (the measurement name plus\n'
            'the output suffix chosen in the dialog; '
            'the original columns are not changed),\n'
            'which can then be plotted and used by '
            'predictors. The recipe is recorded in\n'
            'the configuration and can be recalculated '
            'when the configuration is loaded.\n\n'
            'Use "Export measurements..." to save the new columns.'
        )
        self.processMeasurementsButton.clicked.connect(self.processMeasurements)
        self.combineMeasurementsButton = widgets.calcPushButton(
            'Combine measurements...'
        )
        self.combineMeasurementsButton.setToolTip(
            'Add columns calculated from the current position.\n'
            'Use Export measurements to save the results.'
        )
        self.combineMeasurementsButton.clicked.connect(self.combineMeasurements)
        self.exportMeasurementsButton = widgets.savePushButton(
            'Export measurements...'
        )
        self.exportMeasurementsButton.clicked.connect(self.exportMeasurements)
        
        timeCourseLayout = QVBoxLayout()
        timeCourseLayout.setContentsMargins(0, 0, 0, 0)
        timeCourseLayout.addWidget(QLabel('Columns to plot:'))
        timeCourseLayout.addWidget(self.columnsSearchLineEdit)
        timeCourseLayout.addWidget(self.columnsListWidget)
        self.timeCourseColumnsWidget = QWidget()
        self.timeCourseColumnsWidget.setLayout(timeCourseLayout)

        columnsLayout = QVBoxLayout()
        modeLayout = QHBoxLayout()
        modeLayout.setContentsMargins(0, 0, 0, 0)
        modeLayout.addWidget(self.measurementsModeCombobox, 1)
        modeLayout.addWidget(self.measurementsHelpButton)
        columnsLayout.addLayout(modeLayout)
        columnsLayout.addWidget(self.timeCourseColumnsWidget)
        columnsLayout.addWidget(self.processMeasurementsButton)
        columnsLayout.addWidget(self.combineMeasurementsButton)
        columnsLayout.addWidget(self.exportMeasurementsButton)
        columnsLayout.addWidget(self.refreshButton)
        columnsWidget = QWidget()
        columnsWidget.setLayout(columnsLayout)
        
        self.measurementsSplitter = QSplitter(Qt.Horizontal)
        self.measurementsSplitter.addWidget(measurementsPlotContainer)
        self.measurementsSplitter.addWidget(columnsWidget)
        self.measurementsSplitter.setStretchFactor(0, 3)
        self.measurementsSplitter.setStretchFactor(1, 1)
        
        self.measurementsGroupbox = QGroupBox('Measurements')
        measurementsGroupboxLayout = QVBoxLayout()
        measurementsGroupboxLayout.addWidget(self.measurementsSplitter)
        self.measurementsGroupbox.setLayout(measurementsGroupboxLayout)
    
    def gui_createLayout(self):
        self.gui_createRegionPropsDockWidget()
        self.gui_createShowPropsButton()
        
        # Column 0 is used by the overlay labels gradient items 
        # (see `OverlayLabelsGuiElements.createOverlayLabelsItems`)
        self.mainLayout = QGridLayout()
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.addLayout(self.leftSideDocksLayout, 0, 1, 3, 1)
        self.mainLayout.addWidget(self.graphLayout, 0, 2)
        # Labels colormap gradient editor on the right of the image (as in 
        # guiWin)
        self.mainLayout.addWidget(self.labelsGrad, 0, 3)
        
        # Resizable bottom layout (as in guiWin)
        self.resizeBottomLayoutLine = widgets.VerticalResizeHline()
        self.mainLayout.addWidget(self.resizeBottomLayoutLine, 1, 2)
        self.resizeBottomLayoutLine.dragged.connect(
            self.resizeBottomLayoutLineDragged
        )
        self.resizeBottomLayoutLine.clicked.connect(
            self.resizeBottomLayoutLineClicked
        )
        self.resizeBottomLayoutLine.released.connect(
            self.resizeBottomLayoutLineReleased
        )
        
        self.img1BottomGroupbox = QWidget()
        bottomLayout = QVBoxLayout()
        bottomLayout.setContentsMargins(0, 0, 0, 0)
        bottomLayout.addWidget(self.annotationsGroupbox)
        bottomLayout.addWidget(self.navigationGroupbox)
        bottomLayout.addStretch(1)
        self.img1BottomGroupbox.setLayout(bottomLayout)
        
        self.bottomScrollArea = widgets.ScrollArea(resizeVerticalOnShow=True)
        self.bottomScrollArea.setWidgetResizable(True)
        self.bottomScrollArea.setWidget(self.img1BottomGroupbox)
        self.mainLayout.addWidget(self.bottomScrollArea, 2, 2)
        self.mainLayout.setRowStretch(2, 0)
        self.bottomScrollArea._resizeVertical()
        self.mainLayout.setRowStretch(0, 1)
        self.mainLayout.setColumnStretch(2, 1)
        imagePanel = QWidget()
        imagePanel.setLayout(self.mainLayout)
        
        self.mainSplitter = QSplitter(Qt.Horizontal)
        self.mainSplitter.addWidget(imagePanel)
        self.mainSplitter.addWidget(self.measurementsGroupbox)
        self.mainSplitter.setStretchFactor(0, 1)
        self.mainSplitter.setStretchFactor(1, 1)
        
        self.setCentralWidget(self.mainSplitter)
        self.resize(1400, 800)
    
    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        self.statusBarLabel = QLabel('')
        self.statusbar.addWidget(self.statusBarLabel)
        # Permanent widget with mouse hover info (as in guiWin)
        self.wcLabel = QLabel('')
        self.statusbar.addPermanentWidget(self.wcLabel)
    
    def gui_connectViewerToolsSignals(self):
        self.launch3dViewerButton.toggled.connect(self.launch3dViewerToggled)
        
        self.guiTabControl.propsQGBox.idSB.valueChanged.connect(
            self.propsWidgetIDvalueChanged
        )
        self.guiTabControl.highlightCheckbox.toggled.connect(
            self.highlightIDonHoverCheckBoxToggled
        )
        self.guiTabControl.highlightSearchedCheckbox.toggled.connect(
            self.highlightSearchedIDcheckBoxToggled
        )
        intensMeasurQGBox = self.guiTabControl.intensMeasurQGBox
        intensMeasurQGBox.additionalMeasCombobox.currentTextChanged.connect(
            self.updatePropsWidget
        )
        intensMeasurQGBox.channelCombobox.currentTextChanged.connect(
            self.updatePropsWidget
        )
        propsQGBox = self.guiTabControl.propsQGBox
        propsQGBox.additionalPropsCombobox.currentTextChanged.connect(
            self.updatePropsWidget
        )
        self.showPropsDockButton.sigClicked.connect(self.showPropsDockWidget)
    
    def gui_connectSignals(self):
        self.posCombobox.currentIndexChanged.connect(
            self.posComboboxIndexChanged
        )
        
        self.navigateScrollBar.connectEvents({
            'valueChanged': self.framesScrollBarValueChanged
        })
        self.navSpinBox.connectValueChanged(self.navigateSpinboxValueChanged)
        
        self.zSliceScrollBar.valueChanged.connect(self.update_z_slice)
        self.zProjComboBox.currentTextChanged.connect(self.updateZproj)
        self.switchPlaneCombobox.sigPlaneChanged.connect(
            self.switchViewedPlane
        )
        
        self.idScrollBar.valueChanged.connect(self.idScrollBarValueChanged)
        self.idSpinBox.editingFinished.connect(self.idSpinBoxEditingFinished)
        # Arrows step to the next/previous existing ID (IDs can have gaps)
        self.idSpinBox.sigUpClicked.connect(partial(self.stepID, 1))
        self.idSpinBox.sigDownClicked.connect(partial(self.stepID, -1))
        self.followIDcheckbox.toggled.connect(self.followIDtoggled)
        self.restrictFramesCheckbox.toggled.connect(self.restrictFramesToggled)
        self.rescalePlotCheckbox.toggled.connect(self.rescalePlotToggled)
        
        self.currentFrameLine.sigPositionChangeFinished.connect(
            self.currentFrameLineMoved
        )
        self.columnsSearchLineEdit.textChanged.connect(self.filterColumns)
        self.columnsModel.itemChanged.connect(self.measurementColumnChanged)
        self.columnsListWidget.customContextMenuRequested.connect(
            self.showColumnsContextMenu
        )
        self.normalizeCheckbox.toggled.connect(self.normalizeToggled)
        self.normalizeCheckbox.toggled.connect(self.plotMeasurements)
        self.showPhasesCheckbox.toggled.connect(self.showPhasesToggled)
        self.measurementsModeCombobox.currentIndexChanged.connect(
            self.measurementsModeChanged
        )
        self.refreshButton.clicked.connect(self.reloadButtonClicked)
    
    def onSearchId(self, ID):
        if not self.isDataLoaded:
            self.logger.warning(
                'Data not loaded yet. Key pressing events are not connected.'
            )
            return

        self.get_data()
        self.findID(ID=ID)
    
    def loadPosition(self, pos_i, frame_i=None, ID=None):
        self.pos_i = pos_i
        posData = self.posData()
        self._lab = None
        self._labKey = None
        
        if self.isSegm3D is None:
            self.isSegm3D = posData.segm_data.ndim == 4
        
        if frame_i is None:
            frame_i = max(getattr(posData, 'frame_i', 0), 0)
        self.frame_i = min(frame_i, posData.SizeT-1)
        posData.frame_i = self.frame_i
        
        self.logger.info(
            f'Loading position {pos_i+1}/{len(self.data)} '
            f'("{getattr(posData, "pos_foldername", "")}"), '
            f'frame n. {self.frame_i+1}...'
        )
        self.posCombobox.blockSignals(True)
        self.posCombobox.setCurrentIndex(pos_i)
        self.posCombobox.blockSignals(False)
        isMultiPos = len(self.data) > 1
        self.posLabel.setVisible(isMultiPos)
        self.posCombobox.setVisible(isMultiPos)

        # Initialize the position before the navigation widgets, whose
        # range changes must never act on a partially loaded position
        allIDs = self.initPosAnnotData(posData)
        self._IDframesCache = {}
        self.refreshAcdcDf(plot=False)
        # Cells of this position whose phase evaluation was postponed
        self.schedulePendingPhaseAutoApply()
        self.initFramesNavigation()
        self.updateLastAnalyzedFrameLabel()
        self.initZsliceNavigation()
        self.initIDsNavigation(ID=ID)
        self.updateFramesRange()
        
        self.get_data()
        self.initContoursImage()
        self.gui_createAxesTextAnnotItems(allIDs)
        self.updateImages()
        self.ax1.autoRange()
        self.zoomToSelectedID()
        self._timecourseXRangeKey = None
        self.plotMeasurements()

        if self.rescalePlotCheckbox.isChecked():
            self.autoscaleMeasurementsPlot()
                
    def updateCcaTableWin(self):
        """Show the cell cycle annotations of the current frame in the open
        `ccaTableWin`. Frames without annotations show an empty table and
        only the zoomed IDs that are annotated are shown."""
        posData = self.posData()
        cca_df = posData.cca_df
        if cca_df is None:
            self.ccaTableWin.updateTable(self.getBaseCca_df(), IDs=None)
            return

        zoomIDs = self.getZoomIDs()
        if zoomIDs is not None:
            zoomIDs = [ID for ID in zoomIDs if ID in cca_df.index]
        self.ccaTableWin.updateTable(cca_df, IDs=zoomIDs)

    def onSigUpdateCcaTableWindow(self, *args):
        if not self.isDataLoaded or self.ccaTableWin is None:
            return
        self.updateCcaTableWin()

    def initFramesNavigation(self):
        posData = self.posData()
        SizeT = posData.SizeT
        # A smaller maximum clamps the current value and would emit
        # `valueChanged` (-> `setFrame`). The values are set below.
        self.navigateScrollBar.blockSignals(True)
        self.navSpinBox.blockSignals(True)
        self.navigateScrollBar.setMaximum(SizeT)
        self.navigateScrollBar.setAbsoluteMaximum(SizeT)
        self.navSpinBox.setMaximum(SizeT)
        self.navigateScrollBar.blockSignals(False)
        self.navSpinBox.blockSignals(False)
        self.navSizeLabel.setText(f'/{SizeT}')
        self.navigateScrollBar.setDisabled(SizeT == 1)
        self.navSpinBox.setDisabled(SizeT == 1)
        self.resetNavigateFramesScrollbar(frame_i=self.frame_i)
        self.navSpinBox.setValueNoEmit(self.frame_i+1)
    
    def initZsliceNavigation(self):
        posData = self.posData()
        isZstack = posData.SizeZ > 1
        self.imgGrad.rescaleAcrossZstackAction.setDisabled(not isZstack)
        for widget in self.zSliceNavigationWidgets():
            widget.setVisible(isZstack)

        # Reset the plane also for 2D data: a zy/zx plane left from a
        # previously loaded z-stack would otherwise stay active (hidden),
        # disabling custom annotations and projections.
        self.switchPlaneCombobox.blockSignals(True)
        self.switchPlaneCombobox.setPlane('xy')
        self.switchPlaneCombobox.blockSignals(False)
        self.zProjComboBox.setDisabled(False)

        if not isZstack:
            return

        # 2D segmentation masks cannot be displayed on planes other than xy
        self.switchPlaneCombobox.setDisabled(not self.isSegm3D)
        # A smaller maximum clamps the current z-slice and would emit
        # `valueChanged` (-> `update_z_slice`). The values are set below.
        self.zSliceScrollBar.blockSignals(True)
        self.zSliceSpinbox.blockSignals(True)
        self.setZsliceNavigationRange(posData.SizeZ)
        self.zSliceScrollBar.blockSignals(False)
        self.zSliceSpinbox.blockSignals(False)

        z = posData.SizeZ // 2
        zProjHow = 'single z-slice'
        segmInfo_df = getattr(posData, 'segmInfo_df', None)
        idx = (getattr(posData, 'filename', None), self.frame_i)
        try:
            z = int(segmInfo_df.at[idx, 'z_slice_used_gui'])
            zProjHow = segmInfo_df.at[idx, 'which_z_proj_gui']
        except Exception as err:
            pass
        
        self.zProjComboBox.blockSignals(True)
        self.zProjComboBox.setCurrentText(zProjHow)
        self.zProjComboBox.blockSignals(False)
        self.setZSliceWidgetsEnabled(zProjHow)
        
        self.zSliceScrollBar.blockSignals(True)
        self.zSliceScrollBar.setSliderPosition(z)
        self.zSliceScrollBar.blockSignals(False)
        self.zSliceSpinbox.setValueNoEmit(z+1)
    
    def zSliceNavigationWidgets(self):
        return (
            self.zSliceCheckbox, self.zSliceSpinbox, self.SizeZlabel, 
            self.zSliceScrollBar, self.zProjComboBox, self.switchPlaneCombobox
        )
    
    def setZSliceWidgetsEnabled(self, how):
        enabled = how == 'single z-slice'
        self.zSliceScrollBar.setEnabled(enabled)
        self.zSliceSpinbox.setEnabled(enabled)
        self.zSliceCheckbox.setEnabled(enabled)
    
    def replaceTimepoints(self, uid, timepoints, IDs, posData=None):
        """Update one identifier's boolean column in the in-memory acdc_df.

        First clear all marks for cells in `IDs`, then mark every
        (frame_i, Cell_ID) pair supplied in `timepoints`. Other marks are
        preserved. Missing columns are initialized to False; pairs without
        an existing table row are ignored (no rows are added). This is the central
        function to update the time points inside acdc_df. 

        Parameters
        ----------
        uid : int
            UID of an existing time point identifier. Its name determines
            the column to update: ``timepoint_<name>``.
        timepoints : dict
            Mapping of Cell_ID to an iterable of zero-based frame indices.
            Normally its keys are a subset of `IDs`. Keys outside `IDs`
            also get marked, but their previous marks are not cleared.
        IDs : iterable of int
            Cells whose previous marks should be cleared. A cell omitted
            from `timepoints` has all its marks removed.
        posData : optional
            Target position; defaults to the currently displayed position.

        Returns
        -------
        bool
            False if the position has no table, otherwise True, even when
            none of the supplied pairs match a row.

        Notes
        -----
        This only changes the table in memory. It does not write a CSV,
        record manual edits, update prediction/viewed-cell tracking, or
        redraw plot lines; callers handle those steps separately.

        For example, ``replaceTimepoints(uid, {7: [2, 5]}, [7, 9])`` replaces
        cell 7's marks with frames 2 and 5 and clears all marks for cell 9,
        leaving other cells and other identifiers unchanged.
        """
        acdc_df = self.timepointsDf(posData)
        if acdc_df is None:
            return False
        column = self.timepointColumn(self.timepointIdentifier(uid))
        values = (
            acdc_df[column].to_numpy().copy() if column in acdc_df.columns 
            else np.zeros(len(acdc_df), dtype=bool)
        )
        values[acdc_df.index.get_level_values('Cell_ID').isin(list(IDs))] = False
        pairs = [
            (frame_i, ID) for ID, frames in timepoints.items() 
            for frame_i in frames
        ]
        if pairs:
            values |= acdc_df.index.isin(pairs)
        acdc_df[column] = values
        if posData is None or posData is self.posData():
            self.schedulePhaseAutoApply(set(IDs).union(timepoints))
        return True

    def setAutoApplyPhases(self, enabled):
        self.autoApplyPhases = bool(enabled)
        if self.autoApplyPhases:
            self.ensurePhasePrediction()

    def setMaskUnanalyzedFrames(self, mask, update=True):
        """Set the "Ignore frames after the last analyzed frame" option
        (and its menu action, without emitting `toggled`). With `update`,
        the loaded data are updated if the option changed
        (`analysisFramesChanged`)."""
        mask = bool(mask)
        changed = mask != self.maskUnanalyzedFrames
        self.maskUnanalyzedFrames = mask
        action = getattr(self, 'maskUnanalyzedFramesAction', None)
        if action is not None and action.isChecked() != mask:
            action.blockSignals(True)
            action.setChecked(mask)
            action.blockSignals(False)
        if update and changed:
            self.analysisFramesChanged(
                self.partiallyAnalyzedPositions(self.data or [])
            )

    def maskUnanalyzedFramesToggled(self, checked):
        if bool(checked) == self.maskUnanalyzedFrames:
            return
        if self._isLoadingData or self.measurementProcessingThread is not None:
            # The load sets the option from its prompt or configuration;
            # running processing uses the frames of when it started
            self.setMaskUnanalyzedFrames(self.maskUnanalyzedFrames)
            self.statusBar().showMessage(
                'Wait for the data being loaded (or answer its dialog) first.'
                if self._isLoadingData else
                'Wait for measurement processing to finish first.',
                8000
            )
            return
        self.logger.info(
            'Ignore frames after the last analyzed frame '
            f'{"on" if checked else "off"}'
        )
        self.setMaskUnanalyzedFrames(checked)
        partial = self.partiallyAnalyzedPositions(self.data or [])
        if not partial:
            return
        self.statusBar().showMessage(
            ('Frames after the last analyzed frame are now ignored'
             if checked else 'All frames are now used')
            + ' by new calculations. Re-evaluate the predictors (and '
            'phases), or reload the configuration to recalculate the '
            'measurement steps.', 12000
        )

    def analysisFramesChanged(self, positions=None, replot=True):
        """Update the loaded `positions` (default: all) after the frames
        used by their calculations changed (option toggled, or table
        reloaded with a different last analyzed frame): the phases of all
        their cells are re-evaluated when they are evaluated automatically
        (the other outputs are not recalculated: their time points,
        annotations and measurement steps may have manual edits or
        parameters to review), the label and the plot are updated."""
        if not self.data:
            return
        if positions is None:
            positions = self.data
        positions = list(positions)
        for posData in positions:
            # The position's table (just reloaded, while `self.acdc_df` may
            # still be the previous one until `refreshAcdcDf`)
            acdc_df = getattr(posData, 'acdc_df', None)
            if acdc_df is None or acdc_df.empty:
                continue
            acdc_df = self._standardizeAcdcDf(acdc_df)
            if getattr(posData, 'phaseAutoEvaluatedIDs', None) is not None:
                posData.phaseAutoEvaluatedIDs.clear()
            self.schedulePhaseAutoApply(
                set(acdc_df.index.get_level_values('Cell_ID')), posData
            )
        self.updateLastAnalyzedFrameLabel()
        if replot and self.isDataLoaded and any(
                posData is self.posData() for posData in positions):
            self.plotMeasurements()

    def updateLastAnalyzedFrameLabel(self):
        """Show the last analyzed frame of the displayed position next to
        the frames scrollbar (as the main GUI's last tracked frame), when
        it is not analyzed until its last frame."""
        label = getattr(self, 'lastAnalyzedFrameLabel', None)
        if label is None:
            return
        posData = self.posData() if self.data else None
        if posData is None or analysis_last_frame(posData) is None:
            label.setText('')
            return
        kind = getattr(posData, 'lastAnalyzedKind', None)
        kind = 'cc annot.' if kind == LAST_ANALYZED_CCA else 'tracked'
        text = f'Last {kind} frame n. = {posData.lastAnalyzedFrame+1}'
        if self.maskUnanalyzedFrames:
            text = f'{text} (later frames ignored)'
        else:
            text = f'{text} (all frames used)'
        label.setText(text)

    def ensurePhasePrediction(self, posData=None, ID=None):
        """Schedule phase evaluation of the displayed cell (or `ID`) if it
        has no evaluated phase yet (no phase column or only empty phases).
        Each cell is tried once per session and position, so cells that no
        rule matches are not re-evaluated on every visit."""
        if not self.autoApplyPhases or not self.hasPhaseTranslation():
            return
        if not self.data:
            return
        if posData is None:
            posData = self.posData()
        if posData is not self.posData() or self.acdc_df is None:
            return
        if ID is None:
            ID = self.selectedID
        if ID is None:
            return
        ID = int(ID)
        tried = getattr(posData, 'phaseAutoEvaluatedIDs', None)
        if tried is None:
            tried = posData.phaseAutoEvaluatedIDs = set()
        if ID in tried:
            return
        tried.add(ID)
        try:
            df_ID = self.acdc_df.xs(ID, level='Cell_ID')
        except KeyError:
            return
        if self.phaseValues(df_ID).astype(bool).any():
            return
        self.schedulePhaseAutoApply([ID])

    def schedulePhaseAutoApply(self, IDs, posData=None):
        """Collect cells with changed time points (or custom annotations
        that affect phases, `annotationAffectsPhases`) and re-evaluate their
        phases once control returns to the event loop, so that batch
        updates (e.g., predictors on all cells) trigger one evaluation and
        callers holding a reference to the table are not disrupted.

        Pending cells are tracked per position (`posData`, default: the
        displayed one). Only the displayed position's cells are evaluated;
        those of another position (e.g., after switching position while
        measurement processing postponed the evaluation) stay pending
        until that position is displayed again, or are dropped when that
        position is no longer loaded (`prunePendingPhaseIDs`)."""
        if not self.autoApplyPhases or not self.hasPhaseTranslation():
            return
        if posData is None:
            if not self.data:
                return
            posData = self.posData()
        _, pending = self._pendingPhaseIDs.setdefault(
            id(posData), (posData, set())
        )
        pending.update(IDs)
        self.schedulePendingPhaseAutoApply()

    def schedulePendingPhaseAutoApply(self):
        """Start the (single) deferred phase auto-apply if the displayed
        position has pending cells."""
        if self._phaseAutoApplyScheduled or not self.data:
            return
        if id(self.posData()) not in self._pendingPhaseIDs:
            return
        self._phaseAutoApplyScheduled = True
        QTimer.singleShot(0, self.autoApplyPhaseTranslator)

    def autoApplyPhaseTranslator(self):
        self._phaseAutoApplyScheduled = False
        if self.measurementProcessingThread is not None:
            # Re-evaluated when measurement processing finishes
            return False
        if not self.autoApplyPhases or not self.hasPhaseTranslation():
            self._pendingPhaseIDs = {}
            return False
        self.prunePendingPhaseIDs()
        if not self.data or self.acdc_df is None:
            return False
        entry = self._pendingPhaseIDs.pop(id(self.posData()), None)
        if entry is None:
            return False
        IDs = entry[1].intersection(
            self.acdc_df.index.get_level_values('Cell_ID')
        )
        if not IDs:
            return False
        return self.applyPhaseTranslator(IDs=IDs, quiet=True)
    
    def reloadAcdcDfFile(self, posData=None):
        if posData is None:
            posData = self.posData()
        
        acdc_df_path = getattr(posData, 'acdc_output_csv_path', None)
        if not acdc_df_path or not os.path.exists(acdc_df_path):
            self.logger.warning(
                f'acdc_output CSV file not found: {acdc_df_path}'
            )
            self.statusBar().showMessage(
                f'acdc_output CSV file not found: {acdc_df_path}', 8000
            )
            return
        
        previous = getattr(posData, 'acdc_df', None)
        previousLast = self.analysisLastFrame(posData)
        posData.loadAcdcDf(acdc_df_path)
        # Results of measurement processing started before are stale.
        posData.acdcDfReloadCount = getattr(posData, 'acdcDfReloadCount', 0) + 1
        normalize_loaded_acdc_df(posData)
        # More frames may have been annotated in the main GUI meanwhile
        update_last_analyzed_frame(posData)
        posData.savedAcdcDf = acdc_df_snapshot(getattr(posData, 'acdc_df', None))
        posData.acdcDfFileStamp = acdc_df_file_stamp(acdc_df_path)
        self.restoreTimepointColumns(posData, previous)
        last = self.analysisLastFrame(posData)
        # Step outputs are empty after the frame they were calculated up to
        # (saved NaN is read back as 0 from tables with cell cycle columns)
        # and are ignored after the current last analyzed frame
        outputs = self.measurementStepOutputs(self.configurationSteps())
        stepsLast = self.measurementStepsLastFrames.get(position_key(posData))
        blankFrom = min_frame(stepsLast, last)
        self.blankStepOutputs(posData, outputs, blankFrom)
        self.recordStepsLastFrame(
            self.measurementStepsLastFrames, posData, blankFrom
        )
        if last != previousLast:
            text = (
                f'{position_name(getattr(posData, "images_path", ""))}: '
                + ('new calculations use all frames'
                   if last is None else
                   f'new calculations use frames 1-{last+1}')
                + ' (last analyzed frame changed).'
            )
            if outputs and blankFrom is not None and (
                    last is None or blankFrom < last):
                text += (
                    ' The measurement steps are calculated only up to frame '
                    f'n. {blankFrom+1}: load the configuration again and '
                    'choose "Calculate" to update them.'
                )
            self.logger.info(text)
            self.statusBar().showMessage(text, 12000)
            # The callers refresh (and replot) the displayed table
            self.analysisFramesChanged([posData], replot=False)
        else:
            # The last analyzed frame (or its kind) shown in the label may
            # have changed (e.g. with the option off)
            self.updateLastAnalyzedFrameLabel()
    
    @disableWindow
    def reloadButtonClicked(self, checked=False):
        self.logger.info(
            'Reloading table "'
            f'{getattr(self.posData(), "acdc_output_csv_path", "")}"...'
        )
        self.reloadAcdcDfFile()
        self.refreshAcdcDf()

    def measurementProcessingWarning(self, title, text, details=None):
        if details:
            self.logger.warning('%s: %s\n%s', title, text, details)
        else:
            self.logger.warning('%s: %s', title, text)
        msg = widgets.myMessageBox(wrapText=False)
        if details:
            msg.setDetailedText(details, wrap=False)
        msg.warning(self, title, text)

    def combineMeasurements(self, checked=False):
        if self.measurementProcessingThread is not None:
            self.measurementProcessingWarning(
                'Measurement processing running',
                'Wait for measurement processing to finish before combining measurements.'
            )
            return
        if self.acdc_df is None or self.acdc_df.empty:
            self.measurementProcessingWarning('No measurements', 'Load measurements first.')
            return
        posData = self.posData()
        source = self.acdc_df
        # Equations (and the dialog's test) see only the analyzed frames;
        # `source` stays the full table (identity check when applying)
        equationSource = self.analysisDf(source, posData).copy()
        self.logger.info('Opening "Combine measurements" dialog...')
        dialog = apps.CombineMetricsEquationDialog(
            posData.chNames, posData.SizeZ > 1, self.isSegm3D,
            parent=self, closeOnOk=False, acdc_df=equationSource,
            saveToConfig=False, allowCellAggregations=True,
            limitToLoadedColumns=True
        )
        # `SizeT` of the equations, as when the step is recalculated
        # (`replayMeasurementSteps`)
        dialog.SizeT = self.analysisSizeT(posData)
        dialog.setToolTip(
            'Results are added to the current position in\n'
            'memory. Use Export measurements to save them.'
        )
        dialog.sigOk.connect(
            partial(
                self.applyCombinedMeasurements, posData=posData,
                source=source, equationSource=equationSource,
                lastFrame=self.analysisLastFrame(posData)
            )
        )
        dialog.exec_()
        dialog.deleteLater()

    def applyCombinedMeasurements(
            self, dialog, *, posData, source, equationSource, lastFrame=None
        ):
        dialog.cancel = True
        if (
            self.measurementProcessingThread is not None
            or not any(pos is posData for pos in self.data)
            or self.posData() is not posData
            or self.acdc_df is not source
            # Frames used by the calculations changed (option toggled)
            or self.analysisLastFrame(posData) != lastFrame
        ):
            self.measurementProcessingWarning(
                'Measurements changed',
                'The source table changed. Reopen Combine measurements and try again.'
            )
            return
        equations, _ = dialog.getEquationsDict()
        collisions = set(equations).intersection(equationSource.columns)
        if collisions:
            self.measurementProcessingWarning(
                'Invalid measurement name',
                'Choose a name that does not overwrite existing columns: '
                + ', '.join(sorted(collisions))
            )
            return
        conflicts = self.recordedOutputConflicts(equations)
        if conflicts:
            self.recordedOutputConflictWarning(conflicts)
            return
        _, conflicts = self.measurementStepReplacementOrder(
            {'equations': equations}
        )
        if conflicts:
            self.recordedOrderConflictWarning(conflicts)
            return
        self.logger.info(
            'Combining measurements: ' + ', '.join(
                f'{name} = {expression}'
                for name, expression in equations.items()
            )
        )
        try:
            result = source.copy()
            for name, expression in equations.items():
                if equationSource.empty and not source.empty:
                    # No analyzed frame: empty outputs
                    result[name] = np.nan
                    continue
                values = dialog.evaluateEquation(equationSource, expression)
                if not isinstance(values, pd.Series):
                    raise ValueError('The equation must return one value per row.')
                if len(equationSource) < len(source):
                    # NaN on the frames not analyzed (aligned on the
                    # index): keep bool results numeric
                    values = float_if_bool(values)
                result[name] = values
        except Exception as err:
            self.logger.exception('Combining measurements failed')
            self.measurementProcessingWarning('Combining measurements failed', str(err))
            return
        posData.acdc_df = result
        self.acdc_df = result
        self.recordStepsLastFrame(
            self.measurementStepsLastFrames, posData, lastFrame
        )
        note = self.recordMeasurementStep(
            self.combinedMeasurementSteps, {'equations': equations}
        )
        self.updateColumnsList()
        self.plotMeasurements()
        dialog.cancel = False
        dialog.close()
        self.logger.info(
            f'Combined measurements added: {", ".join(equations)}'
        )
        self.statusBar().showMessage(
            'Combined measurements added. Use Export measurements to save them.'
            + (f' {note}' if note else ''),
            10000 if note else 5000
        )

    def processMeasurements(self, checked=False):
        if self.measurementProcessingThread is not None:
            return
        if self.refuseProcessingWhileLoading():
            return
        if self.acdc_df is None or self.acdc_df.empty:
            self.measurementProcessingWarning('No measurements', 'Load measurements first.')
            return
        columns = [
            col for col in self.acdc_df.select_dtypes(include='number').columns
            if col not in INDEX_COLUMNS
        ]
        if not columns:
            self.measurementProcessingWarning(
                'No measurements', 'The table has no numeric measurements to process.'
            )
            return
        dialog = apps_measurements_viewer.MeasurementProcessingRecipeDialog(
            columns, parent=self, acdc_df=self.acdc_df,
            allChNames=self.posData().chNames
        )
        self.logger.info('Opening "Process measurements" dialog...')
        posData = self.posData()
        dialog.exec_()
        if dialog.cancel:
            self.logger.info('Process measurements cancelled')
            return
        # The dialog does not block the window: other data may have been
        # loaded (or be loading) meanwhile
        if self.refuseProcessingWhileLoading():
            return
        if not self.data or self.posData() is not posData:
            self.measurementProcessingWarning(
                'Data changed',
                'Other data or another position was loaded while the '
                'dialog was open: start "Process measurements" again.'
            )
            return

        conflicts = self.recordedOutputConflicts(dialog.saveNames)
        if conflicts:
            self.recordedOutputConflictWarning(conflicts)
            return
        _, conflicts = self.measurementStepReplacementOrder({
            'columns': list(dialog.selectedChannels),
            'output_columns': list(dialog.saveNames)
        })
        if conflicts:
            self.recordedOrderConflictWarning(conflicts)
            return
        self.startMeasurementsProcessing(
            dialog.selectedChannels, dialog.selectedRecipe, dialog.saveNames
        )

    def startMeasurementsProcessing(self, columns, recipe, saveNames):
        posData = self.posData()
        self.logger.info(
            f'Processing measurements {", ".join(columns)} '
            '(recipe: ' + ', '.join(
                str(step.get('method') if isinstance(step, dict) else step)
                for step in recipe
            ) + ') '
            f'into {", ".join(saveNames)}...'
        )
        self._pendingMeasurementStep = copy.deepcopy({
            'columns': list(columns), 'recipe': recipe,
            'output_columns': list(saveNames)
        })
        # Results are merged into the position's table at completion unless
        # the table was reloaded from file or the input columns changed.
        # Only the analyzed frames are processed (NaN on the later ones);
        # the results are stale if they change meanwhile.
        lastFrame = self.analysisLastFrame(posData)
        self._measurementProcessingTarget = (
            posData, getattr(posData, 'acdcDfReloadCount', 0),
            self.acdc_df[list(columns)].copy(), list(saveNames), lastFrame
        )
        self.measurementProcessingThread = QThread(self)
        self.measurementProcessingWorker = workers.MeasurementProcessingWorkerGUI(
            QMutex(), QWaitCondition()
        )
        worker = self.measurementProcessingWorker
        worker.setupJob(
            df=self.acdc_df.copy(),
            cols=columns,
            recipe=recipe,
            new_colnames=saveNames,
            max_frame_i=lastFrame
        )
        worker.moveToThread(self.measurementProcessingThread)
        self.measurementProcessingThread.started.connect(worker.runJob)
        worker.sigDone.connect(self.measurementProcessingDone)
        worker.signals.critical.connect(self.measurementProcessingFailed)
        worker.signals.finished.connect(self.measurementProcessingThread.quit)
        worker.signals.finished.connect(worker.deleteLater)
        self.measurementProcessingThread.finished.connect(self.measurementProcessingFinished)
        self.processMeasurementsButton.setEnabled(False)
        self.combineMeasurementsButton.setEnabled(False)
        self.refreshButton.setEnabled(False)
        self.measurementProcessingThread.start()

    def measurementProcessingDone(self, processed):
        posData, _, inputs, outputs, _ = self._measurementProcessingTarget
        current = self.measurementProcessingTargetTable(posData)
        stale = current is None
        if not stale:
            columns = list(inputs.columns)
            stale = (
                not set(columns).issubset(current.columns)
                or not current[columns].sort_index().equals(inputs.sort_index())
            )
        if stale:
            self.measurementProcessingWarning(
                'Measurements changed',
                'The source table changed during processing. Results were not applied.'
            )
            return
        collisions = [col for col in outputs if col in current.columns]
        if collisions:
            # e.g. added meanwhile by a predictor or custom annotation
            self.measurementProcessingWarning(
                'Invalid measurement name',
                'Columns with the output names were added during processing. '
                'Results were not applied: ' + ', '.join(collisions)
            )
            return

        # Merge only the new columns, so that edits made meanwhile to the
        # other columns (time points, annotations, visited cells) are kept.
        for col in outputs:
            current[col] = processed[col].reindex(current.index)
        posData.acdc_df = current
        self.recordStepsLastFrame(
            self.measurementStepsLastFrames, posData,
            self._measurementProcessingTarget[4]
        )
        self.logger.info(
            f'Measurement processing done: added {", ".join(outputs)}'
        )
        note = self.recordMeasurementStep(
            self.measurementProcessingSteps, self._pendingMeasurementStep
        )
        if note:
            self.statusBar().showMessage(note, 10000)
        if self.posData() is posData:
            self.acdc_df = current
            self.updateColumnsList()
            self.plotMeasurements()

    def measurementProcessingFailed(self, out):
        worker, error = out
        details = ''.join(traceback.format_exception(
            type(error), error, error.__traceback__
        ))
        self.measurementProcessingWarning(
            'Measurement processing failed', str(error), details=details
        )

    def recordedOrderConflictWarning(self, conflicts):
        self.measurementProcessingWarning(
            'Invalid measurement step',
            'This step replaces a recorded step of the configuration whose '
            'outputs are read by other steps, but it reads outputs of steps '
            'recorded later, so the configuration could not be recalculated '
            'in order. Choose other output names or inputs: '
            + ', '.join(conflicts)
        )

    def recordedOutputConflictWarning(self, conflicts):
        self.measurementProcessingWarning(
            'Invalid measurement name',
            'These output names are already written by a different measurement '
            'step of the configuration (each output must come from exactly one '
            'step). Choose other names or re-apply that step with all of its '
            'outputs: ' + ', '.join(conflicts)
        )

    def measurementProcessingFinished(self):
        self.measurementProcessingThread.deleteLater()
        self.measurementProcessingThread = None
        self.measurementProcessingWorker = None
        self.processMeasurementsButton.setEnabled(True)
        self.combineMeasurementsButton.setEnabled(True)
        self.refreshButton.setEnabled(True)
        # Phase auto-apply (and clears) were postponed while processing was
        # running
        self.schedulePendingPhaseAutoApply()
        self.schedulePendingPhaseClears()

    def configurePhaseTranslator(self, checked=False):
        posData = self.posData() if self.data else None
        # The rule formula tests see the frames used by the evaluation
        options = {
            'acdc_df': self.dialogAnalysisDf(self.acdc_df, posData),
            'SizeT': self.dialogAnalysisSizeT(posData),
            'allChNames': getattr(posData, 'chNames', []),
            'isZstack': getattr(posData, 'SizeZ', 1) > 1,
            'isSegm3D': bool(self.isSegm3D),
        }
        self.logger.info('Opening time point to phase translator...')
        dialog = apps_measurements_viewer.PhaseTranslatorDialog(
            self.phaseTranslatorRules,
            [self.timepointColumn(item) for item in self.timepointIdentifiers],
            self.customAnnotNames(), options, parent=self,
            phaseColors=self.phaseColors,
            annotationOverrides=self.phaseOverrideAnnotations()
        )

        dialog.autoApplyCheckbox.setChecked(self.autoApplyPhases)

        def apply(rules, allCells):
            # Apply commits the colors and options together with the rules
            self.phaseColors = dialog.currentPhaseColors()
            self.setAutoApplyPhases(dialog.autoApplyCheckbox.isChecked())
            self.applyPhaseTranslator(rules, allCells)

        dialog.sigApply.connect(apply)
        dialog.exec_()
        if not dialog.cancel:
            self.phaseTranslatorRules = copy.deepcopy(dialog.rules)
            self.phaseColors = dialog.currentPhaseColors()
            self.setAutoApplyPhases(dialog.autoApplyCheckbox.isChecked())
            overrides = self.phaseOverrideAnnotations()
            self.logger.info(
                f'Phase translator rules set: {len(dialog.rules)} rule(s), '
                + (
                    'annotations overriding phases: '
                    + ', '.join(f'"{name}"' for name in overrides) + ', '
                    if overrides else ''
                )
                + f'auto apply {"on" if self.autoApplyPhases else "off"}'
            )
            self.plotMeasurements()
        else:
            self.logger.info('Phase translator closed without changes')
        dialog.deleteLater()

    def applyPhasesOnDisplayedCell(self, checked=False):
        """Evaluate the phases of the displayed cell with the configured
        phase translator rules and the custom annotations overriding phases
        (toolbar action and its shortcut)."""
        # Qt5 emits `triggered` also for a disabled action (shortcut)
        if not self.isDataLoaded or self.acdc_df is None:
            self.statusBar().showMessage('Load data first.', 5000)
            return False
        if not self.hasPhaseTranslation():
            self.statusBar().showMessage(
                'No phase rules or annotations overriding phases: add rules '
                'with "Time point to phase translator..." or enable '
                '"Overrides phases" of a custom annotation first.', 8000
            )
            self.logger.info(
                'Evaluate phases on displayed cell: no phase rules or '
                'annotations overriding phases defined'
            )
            return False
        return self.applyPhaseTranslator(allCells=False)

    def applyPhaseTranslator(
            self, rules=None, allCells=True, IDs=None, quiet=False
        ):
        """Evaluate phases on all cells, the displayed cell (`allCells`
        False) or the cells `IDs` (overrides `allCells`) of the current
        position, with `rules` (default the configured ones) and the
        current custom annotations overriding phases
        (`phaseOverrideAnnotations`). With `quiet`, errors and warnings go
        to the status bar instead of message boxes (automatic
        re-evaluation)."""
        def warn(title, text):
            if quiet:
                self.statusBar().showMessage(f'{title}: {text}', 8000)
            else:
                self.measurementProcessingWarning(title, text)

        if self.measurementProcessingThread is not None:
            warn(
                'Measurement processing running',
                'Wait for measurement processing to finish before evaluating phases.'
            )
            return False
        if self.acdc_df is None or self.acdc_df.empty:
            warn('No measurements', 'Load measurements first.')
            return False
        if IDs is None and not allCells:
            if self.selectedID is None:
                warn('No cell selected', 'Select a cell first.')
                return False
            IDs = [self.selectedID]
        if rules is None:
            rules = self.phaseTranslatorRules
        overrides = self.phaseOverrideAnnotations()
        posData = self.posData()
        if not quiet:
            self.logger.info(
                'Evaluating phases of '
                + ('all cells' if IDs is None else f'Cell ID(s) {list(IDs)}')
                + f' ({len(rules)} rule(s)'
                + (
                    ', annotations overriding phases: '
                    + ', '.join(f'"{name}"' for name in overrides)
                    if overrides else ''
                )
                + ')...'
            )
        previousDtypes = list(self.acdc_df.dtypes.items())
        try:
            evaluation = self.evaluatePositionPhases(posData, IDs, rules)
        except Exception as err:
            self.logger.exception('Phase evaluation failed')
            warn('Phase evaluation failed', str(err))
            return False
        self.phaseTranslatorRules = copy.deepcopy(rules)
        # Rebuilding the columns tree is slow on large tables: only do it
        # when the evaluation added, removed or retyped columns
        columnsChanged = (
            list(evaluation.table.dtypes.items()) != previousDtypes
        )
        if columnsChanged:
            self.updateColumnsList()
        self.plotMeasurements()
        if evaluation.warnings:
            warn('Phase evaluation warnings', '\n'.join(evaluation.warnings))
            if quiet:
                return True
        if not quiet:
            self.logger.info('Phases evaluated')
        self.statusBar().showMessage(
            'Phases evaluated. Use Export measurements to save the table.', 5000
        )
        return True

    def exportMeasurements(self, checked=False):
        if self.acdc_df is None or self.acdc_df.empty:
            self.measurementProcessingWarning('No measurements', 'Load measurements first.')
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, 'Export measurements', 'processed_measurements.csv', 'CSV (*.csv)'
        )
        if not filepath:
            return
        if self.hasPhaseTranslation():
            msg = widgets.myMessageBox(wrapText=False)
            cancel, skip, apply = msg.question(
                self, 'Evaluate phases before export?',
                'A phase translator (phase rules or custom annotations '
                'overriding phases) is configured. Evaluate all cells of the '
                'current position before exporting?',
                buttonsTexts=('Cancel', 'Export existing values', 'Evaluate and export')
            )
            if msg.cancel or msg.clickedButton == cancel:
                return
            if msg.clickedButton == apply and not self.applyPhaseTranslator():
                return
        try:
            export_df = self.acdc_df.copy()
            self.ensureTimepointColumns(export_df)
            self.ensureCustomAnnotColumns(export_df)
            self.logger.info(f'Exporting measurements to "{filepath}"...')
            export_df.to_csv(filepath)
        except (OSError, ValueError) as error:
            self.measurementProcessingWarning('Export failed', str(error))
            return
        self.logger.info(f'Measurements exported to "{filepath}"')
        self.statusBarLabel.setText(f'Measurements exported to {filepath}')
    
    def refreshAcdcDf(self, plot=True):
        self.acdc_df = self.getAcdcDf()
        self._IDframesCache = {}
        self.initLineageTree()
        self.updateColumnsList()
        if not plot:
            return
        self.initIDsNavigation(ID=self.selectedID)
        self.updateFramesRange()
        self.updateImages()
        self.plotMeasurements()
    
    def onMotherDaughterLine(self, checked, checkbox, ax=0):
        super().onMotherDaughterLine(checked, checkbox, ax=ax)
        if checked and self.initLineageTreeFrameData():
            self.drawAllLineageTreeLines()
    
    def updateColumnsList(self, renamed=None):
        """Repopulate the columns list keeping the checked columns.
        `renamed` ({old: new}) keeps renamed columns checked."""
        renamed = renamed or {}
        checkedColumns = {
            renamed.get(column, column) for column in self.checkedColumns()
        }
        isFirstInit = self.columnsModel.rowCount() == 0
        
        # Do not block the model signals: the search proxy and the view must 
        # be notified of the reset (otherwise they keep dangling indexes)
        self._isUpdatingColumnsList = True
        try:
            self._updateColumnsList(checkedColumns, isFirstInit)
        finally:
            self._isUpdatingColumnsList = False
    
    def _updateColumnsList(self, checkedColumns, isFirstInit):
        self.columnsModel.clear()
        if self.acdc_df is None:
            self.updateSignalAxes([])
            return
        
        numericColumns = [
            col for col in self.acdc_df.select_dtypes(
                include=['number', 'bool']
            ).columns
            if col not in INDEX_COLUMNS
        ]
        if isFirstInit and self.measurementsModeCombobox.currentIndex() == 0:
            for col in DEFAULT_MEASUREMENTS_COLUMNS:
                if col in numericColumns:
                    checkedColumns = {col}
                    break
            else:
                checkedColumns = set(numericColumns[:1])
        
        channels = getattr(self.posData(), 'chNames', ()) if self.data else ()
        self.columnsModel.populate(
            self.acdc_df[numericColumns], channels, checkable=True
        )
        for item in self.columnsModel.measurementItems():
            col = item.text()
            if col in checkedColumns:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
        self.columnsListWidget.expandAll()
        self.updateSignalAxes(numericColumns)
        self.filterColumns(self.columnsSearchLineEdit.text())

    def updateSignalAxes(self, columns):
        for axis, col in self.measurementsAxes.items():
            if col not in columns:
                self.measurementsAxes[axis] = ''
        self._timeCourseColumns.intersection_update(columns)
        self.updateColumnsIcons()
    
    def checkedColumns(self):
        columns = []
        for item in self.columnsModel.measurementItems():
            if item.checkState() == Qt.Checked:
                columns.append(item.text())
        return columns
    
    def filterColumns(self, text):
        self.columnsSearchProxy.setFilterText(text)
        self.columnsHighlightDelegate.setFilterText(text)
        if text.strip():
            self.columnsListWidget.expandAll()
        self.columnsListWidget.viewport().update()
    
    def initIDsNavigation(self, ID=None):
        self.IDs = self.getAllIDs()
        numIDs = len(self.IDs)
        self.numIDsLabel.setText(f'({numIDs} cells)')
        hasIDs = numIDs > 0
        self.idScrollBar.setEnabled(hasIDs)
        self.idSpinBox.setEnabled(hasIDs)
        if not hasIDs:
            self.selectedID = None
            return
        
        if ID not in self.IDs:
            # Select first ID present in current frame (if any)
            frameIDs = set(np.unique(self.getLabels()))
            ID = next((_ID for _ID in self.IDs if _ID in frameIDs), self.IDs[0])
        
        self.selectedID = ID
        self.idScrollBar.blockSignals(True)
        self.idScrollBar.setMaximum(numIDs-1)
        self.idScrollBar.setValue(self.IDs.index(ID))
        self.idScrollBar.blockSignals(False)
        self.idSpinBox.setMaximum(max(self.IDs))
        self.idSpinBox.setValueNoEmit(ID)
        self.ensureTimepointPredictions()
    
    def getImage(self, frame_i=None, raw=False):
        posData = self.posData()
        if frame_i is None:
            frame_i = self.frame_i
        img = posData.img_data[frame_i]
        if posData.SizeZ > 1:
            img = self.get_2Dimg_from_3D(img)
        return img
    
    def get_2Dimg_from_3D(self, imgData, isLayer0=True, frame_i=None):
        img = self.get_2Dimg_from_switched_plane(imgData)
        if img is not None:
            return img
        
        zProjHow = self.zProjComboBox.currentText()
        z = self.zSliceScrollBar.sliderPosition()
        if not isLayer0:
            zProjHowOverlay = self.zProjOverlay_CB.currentText()
            if zProjHowOverlay != 'same as above':
                zProjHow = zProjHowOverlay
                z = self.zSliceOverlay_SB.sliderPosition()
        return self.zProjectImage(imgData, zProjHow, z)
    
    def getHighlightedIDfromPropsDock(self):
        if not self.propsDockWidget.isVisible():
            return 0
        guiTabControl = self.guiTabControl
        doHighlight = (
            guiTabControl.highlightCheckbox.isChecked()
            or guiTabControl.highlightSearchedCheckbox.isChecked()
        )
        if not doHighlight:
            return 0
        return guiTabControl.propsQGBox.idSB.value()

    def updatePropsWidget(self, ID, fromHover=False):
        # Writing the Object ID from code must not select the cell
        wasUpdating = self._isUpdatingPropsDock
        self._isUpdatingPropsDock = True
        try:
            super().updatePropsWidget(ID, fromHover=fromHover)
        finally:
            self._isUpdatingPropsDock = wasUpdating

    def propsWidgetIDvalueChanged(self, ID):
        """The Object ID of the "Cell-ACDC objects" dock changed: a user edit
        selects that cell (as clicking it), code updates (hover, sync with
        the selected cell) only show its properties."""
        if self._isUpdatingPropsDock or not self.isDataLoaded:
            return
        ID = int(ID)
        if ID != self.selectedID and ID in self.IDs:
            self.logger.info(f'Selecting Cell ID {ID} from the objects dock')
            # Selecting updates the images, which sync the dock
            self.goToObjectID(ID)
            return
        super().propsWidgetIDvalueChanged(ID)

    def syncPropsDockToSelectedID(self):
        """Show the properties of the selected cell in the current frame in
        the "Cell-ACDC objects" dock (if visible), so that the dock and the
        viewer always refer to the same cell."""
        if not self.propsDockWidget.isVisible() or not self.selectedID:
            return
        posData = self.posData()
        if getattr(posData, 'rp', None) is None:
            return

        ID = int(self.selectedID)
        propsQGBox = self.guiTabControl.propsQGBox
        wasUpdating = self._isUpdatingPropsDock
        self._isUpdatingPropsDock = True
        try:
            # Force refresh: the values change with frame/z-slice
            self.currentPropsID = -1
            obj = posData.rp.get_obj_from_ID(ID, warn=False)
            propsQGBox.idSB.setValue(ID)
            if obj is None:
                # No values of another (previously shown) cell next to ID
                self.clearPropsDockValues()
                propsQGBox.notExistingIDLabel.setText(
                    f'Object ID {ID:d} does not exist'
                )
            elif not self.isObjInCurrSlice(obj.bbox):
                # 3D segmentation, single z-slice: the mixin's area of the
                # displayed slice would index outside the object
                self.clearPropsDockValues()
                propsQGBox.notExistingIDLabel.setText(
                    f'Object ID {ID:d} is not in this z-slice'
                )
            else:
                self.updatePropsWidget(ID)
        finally:
            self._isUpdatingPropsDock = wasUpdating

    def clearPropsDockValues(self):
        """Empty the value fields of the "Cell-ACDC objects" dock (shown
        when the selected cell has no values in the displayed frame or
        z-slice)."""
        propsQGBox = self.guiTabControl.propsQGBox
        intensMeasurQGBox = self.guiTabControl.intensMeasurQGBox
        fields = (
            propsQGBox.cellAreaPxlSB, propsQGBox.cellAreaUm2DSB,
            propsQGBox.cellVolVoxSB, propsQGBox.cellVolFlDSB,
            propsQGBox.cellVolVox3D_SB, propsQGBox.cellVolFl3D_DSB,
            propsQGBox.elongationDSB, propsQGBox.solidityDSB,
            propsQGBox.additionalPropsCombobox.indicator,
            intensMeasurQGBox.minimumDSB, intensMeasurQGBox.maximumDSB,
            intensMeasurQGBox.meanDSB, intensMeasurQGBox.medianDSB,
            intensMeasurQGBox.additionalMeasCombobox.indicator,
        )
        for field in fields:
            field.setText('')

    def updateImages(self):
        """Update image and annotations as `gui.guiWin.updateAllImages` in
        'Viewer' mode"""
        if not self.data:
            return

        self.get_data()
        self.syncPropsDockToSelectedID()
        self.highlightedID = (
            self.getHighlightedIDfromPropsDock() or self.selectedID or 0
        )
        self.clearAx1Items()
        
        self.rescaleIntensitiesLut(setImage=False)
        self.setImageImg1()
        
        posData = self.posData()
        self.currentLab2D = self.get_2Dlab(posData.lab, force_z=False)
        self.setOverlayImages()
        self.setOverlayLabelsItems()
        self.setOverlaySegmMasks()
        self.updateViewerWindow()
        
        self.setAllTextAnnotations()
        self.setAllContoursImages(compute=False)
        self.drawAllMothBudLines()
        if self.initLineageTreeFrameData():
            self.drawAllLineageTreeLines()
        
        if self.ccaTableWin is not None:
            self.updateCcaTableWin()
        
        self.highlightSelectedID()
        self.doCustomAnnotation(0)
        self.updateTimestampFrame()
        self.annotateAllObjectTracks()
        posData.visited = True
        self.updateObjectCounts()
        self.updateIDnotPresentLabel()
    
    def clearAx1Items(self):
        self.labelsLayerImg1.clear()
        self.labelsLayerImg1.setOpacity(self.imgGrad.labelsAlphaSlider.value())
        self.highLightIDLayerImg1.clear()
        self.searchedIDitemLeft.clear()
        self.ax1_contoursImageItem.clear()
        if 0 in self.textAnnot:
            self.textAnnot[0].clear()
        self.ax1_newMothBudLinesItem.setData([], [])
        self.ax1_oldMothBudLinesItem.setData([], [])
        self.ax1_movementAgainstPrevLinesItem.setData([], [])
        self.ax1_movementAgainstPrevScatterItem.setData([], [])
        self.clearObjTracks()
    
    def highlightSelectedID(self):
        """Highlight the selected ID as the searched ID in `gui.guiWin`"""
        ID = self.highlightedID
        if not ID:
            return
        
        posData = self.posData()
        obj = posData.rp.get_obj_from_ID(ID, warn=False)
        if obj is None:
            return
        
        if not self.isObjInCurrSlice(obj.bbox):
            return
        
        display_rp = self.get2DRP()
        obj = display_rp.get_obj_from_ID(ID, warn=False)
        if obj is None:
            return
        contours = self.getObjContours(
            obj,
            all_external=True,
            include_internal=self.showAllContoursToggle.isChecked()
        )
        for cont in contours:
            self.searchedIDitemLeft.addPoints(cont[:,0]+0.5, cont[:,1]+0.5)
    
    def updateAllImages(self, *args, **kwargs):
        self.updateImages()
    
    def setImageImg1(self):
        posData = self.posData()
        self.img1.setCurrentPosIndex(self.pos_i)
        self.img1.setCurrentFrameIndex(posData.frame_i)
        if posData.SizeZ > 1:
            zProjHow = self.zProjComboBox.currentText()
            if self.switchPlaneCombobox.depthAxes() != 'z':
                # Pre-computed min/max values refer to xy planes only
                z = None
            elif zProjHow == 'single z-slice':
                z = self.zSliceScrollBar.sliderPosition()
            else:
                z = zProjHow
            
            self.img1.setCurrentZsliceIndex(z)
        
        self.img1.setImage(self.getImage())
    
    def updateIDnotPresentLabel(self):
        if self.selectedID is None:
            self.idNotPresentLabel.setText('')
            return
        
        isPresent = np.any(self.getLabels() == self.selectedID)
        if isPresent:
            self.idNotPresentLabel.setText('')
        else:
            self.idNotPresentLabel.setText(
                f'<i>Cell ID {self.selectedID} is not present in frame '
                f'n. {self.frame_i+1}</i>'
            )
    
    def posComboboxIndexChanged(self, pos_i):
        self.loadPosition(pos_i, ID=self.selectedID)
    
    def navigateSpinboxValueChanged(self, value):
        self.navigateScrollBar.setSliderPosition(value)
    
    def framesScrollBarValueChanged(self, frame_n):
        self.setFrame(frame_n-1)
    
    def setFrame(self, frame_i):
        posData = self.posData()
        first_i, last_i = self.framesRange()
        frame_i = int(np.clip(frame_i, first_i, last_i))
        self.navSpinBox.setValueNoEmit(frame_i+1)
        if self.navigateScrollBar.sliderPosition() != frame_i+1:
            self.resetNavigateFramesScrollbar(frame_i=frame_i)
        
        if frame_i == self.frame_i:
            return
        
        self.frame_i = frame_i
        posData.frame_i = frame_i
        self.updateImages()
        self.zoomToSelectedID()
        self.updateCurrentFrameLine()
    
    def update_z_slice(self, z):
        self.zSliceSpinbox.setValueNoEmit(z+1)
        self.updateImages()
    
    def updateZproj(self, how):
        self.setZSliceWidgetsEnabled(how)
        self.updateImages()
    
    def switchViewedPlane(self, previousPlane, currentPlane):
        posData = self.posData()
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if depthAxes == 'z':
            self.zProjComboBox.setDisabled(False)
            depthSize = posData.SizeZ
        else:
            self.zProjComboBox.blockSignals(True)
            self.zProjComboBox.setCurrentText('single z-slice')
            self.zProjComboBox.blockSignals(False)
            self.setZSliceWidgetsEnabled('single z-slice')
            self.zProjComboBox.setDisabled(True)
            SizeY, SizeX = posData.img_data.shape[-2:]
            depthSize = SizeY if depthAxes == 'y' else SizeX
        
        self.zSliceScrollBar.blockSignals(True)
        self.setZsliceNavigationRange(depthSize)
        self.zSliceScrollBar.setSliderPosition(depthSize // 2)
        self.zSliceScrollBar.blockSignals(False)
        self.zSliceSpinbox.setValueNoEmit(depthSize // 2 + 1)
        
        self.initContoursImage()
        self.updateImages()
        self.ax1.autoRange()
        self.zoomToSelectedID()
    
    def idScrollBarValueChanged(self, idx):
        if not self.IDs:
            return
        ID = self.IDs[idx]
        self.idSpinBox.setValueNoEmit(ID)
        self.setSelectedID(ID)
        
        if self.rescalePlotCheckbox.isChecked():
            self.autoscaleMeasurementsPlot()
            
    def rescalePlotToggled(self, checked):
        self.logger.info(f'Rescale plot {"on" if checked else "off"}')
        if checked:
            self.autoscaleMeasurementsPlot()
        self._set_setting_value('rescalePlot', checked)
    
    def idSpinBoxEditingFinished(self):
        ID = self.idSpinBox.value()
        if ID == self.selectedID:
            return
        
        if ID not in self.IDs:
            self.statusBar().showMessage(f'Cell ID {ID} does not exist.', 3000)
            self.idSpinBox.setValueNoEmit(self.selectedID)
            return
        
        self.idScrollBar.setValue(self.IDs.index(ID))
    
    def setSelectedID(self, ID):
        if ID != self.selectedID:
            self.logger.info(f'Selected Cell ID {ID}')
        self.selectedID = ID
        comparisonData = self.comparisonPlotData()
        self.ensureTimepointPredictions()
        self.updateFramesRange()
        self.zoomToSelectedID(goToZslice=True)
        self.updateImages()
        if (comparisonData is not None
                and comparisonData.equals(self.comparisonPlotData())):
            # The comparison scatter shows all cells: replotting it is slow
            # on large tables and would reset the zoom
            return
        self.plotMeasurements()

    def comparisonPlotData(self):
        """X and Y columns of the plotted comparison scatter (copy), or None
        if not in comparison mode or nothing is plotted."""
        if self.measurementsModeCombobox.currentIndex() != 1:
            return
        if self.acdc_df is None or not self.measurementsPlot.listDataItems():
            return
        columns = [self.measurementsAxes['X'], self.measurementsAxes['Y']]
        if not all(col in self.acdc_df.columns for col in columns):
            return
        return self.acdc_df[columns].copy()
    
    def framesRange(self):
        """Get the (first, last) frame index the user can navigate to."""
        SizeT = self.posData().SizeT
        if not self.restrictFramesCheckbox.isChecked():
            return 0, SizeT-1
        
        if self.selectedID is None:
            return 0, SizeT-1
        
        frames = self.getIDframes(self.selectedID)
        frames = frames[(frames >= 0) & (frames < SizeT)]
        if len(frames) == 0:
            return 0, SizeT-1
        
        return int(frames[0]), int(frames[-1])
    
    def updateFramesRange(self):
        """Restrict the frames navigation widgets to `framesRange` and move 
        the current frame into the range if it is outside. 
        
        Returns True if the current frame changed (images not updated).
        """
        posData = self.posData()
        first_i, last_i = self.framesRange()
        
        self.navigateScrollBar.blockSignals(True)
        self.navigateScrollBar.setMinimum(first_i+1)
        self.navigateScrollBar.setMaximum(last_i+1)
        self.navigateScrollBar.blockSignals(False)
        self.navSpinBox.blockSignals(True)
        self.navSpinBox.setMinimum(first_i+1)
        self.navSpinBox.setMaximum(last_i+1)
        self.navSpinBox.blockSignals(False)
        
        isRangeZero = first_i == last_i
        self.navigateScrollBar.setDisabled(isRangeZero)
        self.navSpinBox.setDisabled(isRangeZero)
        
        frame_i = int(np.clip(self.frame_i, first_i, last_i))
        isFrameChanged = frame_i != self.frame_i
        self.frame_i = frame_i
        posData.frame_i = frame_i
        self.navSpinBox.setValueNoEmit(frame_i+1)
        self.resetNavigateFramesScrollbar(frame_i=frame_i)
        if isFrameChanged:
            self.updateCurrentFrameLine()
        return isFrameChanged
    
    def restrictFramesToggled(self, checked):
        self.logger.info(
            f'Only frames with selected cell {"on" if checked else "off"}'
        )
        self._set_setting_value('restrictFrames', checked)
        if not self.data:
            return
        isFrameChanged = self.updateFramesRange()
        if not isFrameChanged:
            return
        self.updateImages()
        if self.followIDcheckbox.isChecked():
            self.zoomToSelectedID()

    def followIDtoggled(self, checked):
        self.logger.info(f'Follow cell {"on" if checked else "off"}')
        if checked:
            self.zoomToSelectedID()
            
        self._set_setting_value('followID', checked)
    
    def goToZsliceID(self, ID):
        """Move the z-slice scrollbar to the slice of `ID` (along the 
        current depth axis) if the ID is not visible in the current slice.
        """
        posData = self.posData()
        if not self.isSegm3D or posData.SizeZ == 1:
            return
        
        if self.zProjComboBox.currentText() != 'single z-slice':
            return
        
        lab = self.getLabels()
        lab2D = self.get_2Dlab(lab, force_z=False)
        if np.any(lab2D == ID):
            return
        
        objMask = (lab == ID).astype(np.uint8)
        if not np.any(objMask):
            return
        
        obj = skimage.measure.regionprops(objMask)[0]
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if depthAxes == 'z':
            current_z = self.zSliceScrollBar.sliderPosition()
            z = core.nearest_nonzero_z_idx_from_z_centroid(
                obj, current_z=current_z
            )
        elif depthAxes == 'y':
            z = round(obj.centroid[1])
        else:
            z = round(obj.centroid[2])
        
        self.zSliceScrollBar.setSliderPosition(int(z))
    
    def zoomToSelectedID(self, goToZslice=False, force=False):
        ID = self.selectedID
        if ID is None:
            return
        
        if not self.followIDcheckbox.isChecked() and not force:
            return
        
        if goToZslice or self.followIDcheckbox.isChecked():
            self.goToZsliceID(ID)
        
        lab = self.getLabels()
        lab2D = self.get_2Dlab(lab, force_z=False)
        objMask = None
        if np.any(lab2D == ID):
            objMask = lab2D == ID
        elif self.switchPlaneCombobox.depthAxes() == 'z':
            # ID not visible on current slice --> zoom on its projection
            objMask = lab == ID
            if objMask.ndim == 3:
                objMask = objMask.any(axis=0)
        
        if objMask is None or not np.any(objMask):
            return
        
        rows, cols = np.nonzero(objMask)
        obj_bbox = (rows.min(), cols.min(), rows.max()+1, cols.max()+1)
        self.setViewRangeToObjBbox(obj_bbox, padding=10)
    
    def updateCurrentFrameLine(self):
        self.currentFrameLine.blockSignals(True)
        self.currentFrameLine.setValue(self.frame_i+1)
        self.currentFrameLine.blockSignals(False)
        self.phaseBandFrameLine.setValue(self.frame_i+1)
    
    def currentFrameLineMoved(self, line):
        frame_n = int(round(line.value()))
        self.setFrame(frame_n-1)
        self.updateCurrentFrameLine()
    
            # Per-cell constants values {name: {Cell_ID: value}}
    #         posData.cellConstants = {}
    #         posData.cellConstantsSource = None
    
    # def positionCellConstants(self, posData=None, acdc_df=None):
    #     """Values of the per-cell constants of a position 
    #     {name: {Cell_ID: value}} (or a Series indexed by (frame_i, Cell_ID) 
    #     for per-frame constants), recomputed if the constants or the 
    #     measurements table changed."""
    #     if posData is None:
    #         posData = self.posData()
    #     self.initPositionTimepointState(posData)
    #     if not self.cellConstants:
    #         posData.cellConstants = {}
    #         posData.cellConstantsSource = None
    #         return posData.cellConstants
    #     if acdc_df is None:
    #         acdc_df = (
    #             self.acdc_df if posData is self.posData() 
    #             else self.getAcdcDf(posData)
    #         )
    #     constants = tuple(
    #         tuple(timepoint_predictors.normalize_constant(constant).items())
    #         for constant in self.cellConstants
    #     )
    #     source = posData.cellConstantsSource
    #     if source is not None and source[0] is acdc_df and source[1] == constants:
    #         return posData.cellConstants
    #     try:
    #         values = timepoint_predictors.compute_cell_constants(
    #             acdc_df, self.cellConstants, SizeT=posData.SizeT
    #         )
    #     except Exception as err:
    #         self.logger.exception('Computing per-cell constants failed')
    #         self.measurementProcessingWarning(
    #             'Per-cell constants failed', 
    #             f'Computing the per-cell constants failed:\n\n{err}'
    #         )
    #         values = {}
    #     posData.cellConstants = values
    #     # Keep a reference to the table to detect changes by identity
    #     posData.cellConstantsSource = (acdc_df, constants)
    #     return values
    
    # def acdcDfWithConstants(self, acdc_df=None, posData=None):
    #     """Copy of `acdc_df` (default current table) with the per-cell 
    #     constants added as columns."""
    #     if not self.data:
    #         return acdc_df if acdc_df is not None else self.acdc_df
    #     if posData is None:
    #         posData = self.posData()
    #     if acdc_df is None and posData is self.posData():
    #         acdc_df = self.acdc_df
    #     if acdc_df is None:
    #         return None
    #     values = self.positionCellConstants(posData, acdc_df=acdc_df)
    #     return timepoint_predictors.add_constants_columns(acdc_df, values)
    
    # def setCellConstants(self, constants):
    #     timepoint_predictors.validate_constants(constants)
    #     self.cellConstants = [
    #         timepoint_predictors.normalize_constant(constant)
    #         for constant in constants
    #     ]
    #     for posData in self.data or []:
    #         if getattr(posData, 'timepointsPredicted', None) is not None:
    #             posData.cellConstants = {}
    #             posData.cellConstantsSource = None
    
    # def askReevaluateAfterConstantsChanged(self):
    #     txt = html_utils.paragraph(
    #         'The per-cell constants changed.<br><br>'
    #         'Do you want to <b>re-evaluate the time point predictors</b> on '
    #         'all cells now?'
    #     )
    #     msg = widgets.myMessageBox(wrapText=False)
    #     _, yesButton = msg.question(
    #         self, 'Re-evaluate predictors?', txt, 
    #         buttonsTexts=('No', 'Yes, re-evaluate')
    #     )
    #     return msg.clickedButton == yesButton
    
    # def editCellConstants(self, checked=False):
    #     posData = self.posData() if self.data else None
    #     dialog = apps.CellConstantsDialog(
    #         constants=self.cellConstants, acdc_df=self.acdc_df,
    #         SizeT=getattr(posData, 'SizeT', None), parent=self
    #     )
    #     dialog.exec_()
    #     if dialog.cancel or dialog.constants is None:
    #         return False
    #     newConstants = [
    #         timepoint_predictors.normalize_constant(constant) 
    #         for constant in dialog.constants
    #     ]
    #     if newConstants == self.cellConstants:
    #         return True
    #     self.setCellConstants(newConstants)
    #     if posData is not None:
    #         values = self.positionCellConstants(posData)
    #         self.statusBar().showMessage(
    #             f'Computed {len(values)} per-cell constant(s).', 5000
    #         )
    #     if posData is not None and self.predictorTimepointUids():
    #         # Other positions are re-predicted when they are loaded
    #         for otherPosData in self.data:
    #             if otherPosData is posData:
    #                 continue
    #             if getattr(otherPosData, 'timepointsPredicted', None) is not None:
    #                 otherPosData.timepointsPredicted.difference_update(
    #                     self.predictorTimepointUids()
    #                 )
    #                 positionPath = os.path.normcase(
    #                     os.path.abspath(otherPosData.pos_path)
    #                 )
    #                 for uid in self.predictorTimepointUids():
    #                     self.timepointAutoPredictions.pop((positionPath, uid), None)
    #         if self.askReevaluateAfterConstantsChanged():
    #             self.reevaluateAllTimepoints()
    #     return True
    
    def reservedShortcuts(
            self, excludeUid=None, excludeCustomAnnot=None, shiftDerived=True
        ):
        """Key sequences already used in the viewer {sequence: used by}.

        The shortcuts (including the Shift+ auto shortcut) of the time
        point identifier `excludeUid` and of the custom annotation button
        `excludeCustomAnnot` are not included (e.g., when editing them).

        If `shiftDerived` is True, a key K is also reserved when ``Shift+K``
        is used, because K would get ``Shift+K`` as auto shortcut (see
        `auto_shortcut_text`)."""
        reserved = {}
        for name in ('Left', 'Right', 'Up', 'Down', 'Space', 'Z', 'Esc'):
            reserved[name] = 'Viewer navigation'
        for n in range(1, 10):
            reserved[str(n)] = 'Overlay channels toggle'
        zoomOutKey = QKeySequence(self.zoomOutKeyValue).toString()
        reserved[zoomOutKey] = 'Zoom out'
        timepointActions = {
            action: uid for uid, action in self.timepointActions.items()
        }
        timepointActions.update({
            action: uid for uid, action in self.timepointAutoActions.items()
        })
        excludedActions = set(
            self.customAnnotShortcutActions.get(excludeCustomAnnot, ())
        )
        widgets_ = self.findChildren(QAction) + self.findChildren(QAbstractButton)
        for widget in widgets_:
            if timepointActions.get(widget, -1) == excludeUid:
                continue
            if widget in excludedActions:
                continue
            shortcut = widget.shortcut().toString()
            if not shortcut:
                continue
            text = widget.text() or widget.toolTip().split('\n')[0]
            reserved[shortcut] = text.replace('&', '') or shortcut
        for name, widget in self.widgetsWithShortcut.items():
            shortcut = getattr(widget, 'keyPressShortcut', None)
            if isinstance(shortcut, QKeySequence) and not shortcut.isEmpty():
                reserved[shortcut.toString()] = name
        if not shiftDerived:
            return reserved
        for shortcut, usedBy in list(reserved.items()):
            if 'Shift+' not in shortcut:
                continue
            base = QKeySequence(shortcut.replace('Shift+', '', 1)).toString()
            if not base or base in reserved:
                continue
            if auto_shortcut_text(base) == shortcut:
                reserved[base] = f'{usedBy} ({shortcut})'
        return reserved

    def autoShortcutConflict(self, shortcut, excludeUid=None,
            excludeCustomAnnot=None):
        """Return (auto shortcut text, used by) of `shortcut`. `used by` is
        None if the auto shortcut is free (or empty)."""
        autoText = auto_shortcut_text(shortcut)
        if not autoText:
            return autoText, None
        reserved = self.reservedShortcuts(
            excludeUid=excludeUid, excludeCustomAnnot=excludeCustomAnnot
        )
        return autoText, reserved.get(autoText)
    
    def defaultTimepointIdentifier(self):
        names = {identifier.name for identifier in self.timepointIdentifiers}
        n = len(self.timepointIdentifiers) + 1
        while f'Time point {n}' in names:
            n += 1
        palette = timepoint_predictors.DEFAULT_COLORS
        usedColors = {
            tuple(identifier.color) for identifier in self.timepointIdentifiers
        }
        color = next(
            (c for c in palette if c not in usedColors), 
            palette[len(self.timepointIdentifiers) % len(palette)]
        )
        reserved = self.reservedShortcuts()
        shortcut = next(
            (key for key in timepoint_predictors.DEFAULT_SHORTCUTS 
             if key not in reserved), ''
        )
        return TimepointIdentifier(
            name=f'Time point {n}', color=color, shortcut=shortcut
        )
    
    def execTimepointIdentifierDialog(self, identifier, title):
        posData = self.posData() if self.data else None
        names = {
            other.name for other in self.timepointIdentifiers
            if other.uid != identifier.uid
        }
        dialog = apps_measurements_viewer.TimepointIdentifierDialog(
            identifier.name, identifier.color, identifier.shortcut,
            predictor=identifier.predictor, existingNames=names,
            reservedShortcuts=self.reservedShortcuts(excludeUid=identifier.uid),
            acdc_df=(
                self.dialogAnalysisDf(self.timepointsDf(posData), posData)
                if posData is not None else None
            ),
            SizeT=self.dialogAnalysisSizeT(posData),
            allChNames=getattr(posData, 'chNames', []),
            isZstack=getattr(posData, 'SizeZ', 1) > 1,
            isSegm3D=bool(self.isSegm3D), title=title, parent=self
        )
        dialog.validatePredictorDependencies = (
            lambda name, predictor: self.validatePredictorDependencies(
                self.timepointColumn(identifier), f'{TIMEPOINT_PREFIX}{name}',
                predictor
            )
        )
        dialog.autoApplyCheckBox.setChecked(identifier.autoApply)
        dialog.exec_()
        if dialog.cancel:
            return
        return dialog.identifier
    
    def addTimepointIdentifier(self, checked=False):
        identifier = self.defaultTimepointIdentifier()
        values = self.execTimepointIdentifierDialog(
            identifier, 'Add time point identifier'
        )
        if values is None:
            return
        identifier = TimepointIdentifier(**values)
        column = self.timepointColumn(identifier)
        self.logger.info(
            f'Adding time point identifier "{identifier.name}" '
            f'(column "{column}")'
        )
        preexisting = [
            posData for posData in self.data or []
            if any(column in table for table in self.positionTables(posData))
        ]
        self.addTimepointIdentifierItem(identifier)
        # Saved time points of visited cells in an existing column are kept
        # (a brand-new column has none to keep).
        self.seedVisitedTimepointPredictions([identifier.uid], preexisting)
        self.ensureTimepointPredictions()
        self.updateColumnsList()
        self.plotMeasurements()
        self.updateTimepointLines()
        return identifier
    
    def addTimepointIdentifierItem(self, identifier):
        self.timepointIdentifiers.append(identifier)
        for acdc_df in self.allAcdcDfs():
            self.ensureTimepointColumns(acdc_df)
        # Shortcut: hovered plot frame, click: current frame
        action = _ShortcutAwareAction(self)
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        action.triggered.connect(
            partial(self.timepointActionTriggered, identifier.uid)
        )
        self.timepointActions[identifier.uid] = action
        self.timepointsToolBar.addAction(action)
        
        # Shift+shortcut re-evaluates the predictor for the selected cell.
        autoAction = QAction(self)
        autoAction.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        autoAction.triggered.connect(
            lambda checked=False: self.reevaluateDisplayedCellTimepoints(
                uids=[identifier.uid]
            )
        )
        self.addAction(autoAction)
        self.timepointAutoActions[identifier.uid] = autoAction
        
        self.updateTimepointAction(identifier)
        self.updateReevaluateTimepointsActions()
        button = self.timepointsToolBar.widgetForAction(action)
        if button is not None:
            button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            button.customContextMenuRequested.connect(
                partial(self.showTimepointActionContextMenu, identifier.uid)
            )
        return action
    
    def updateTimepointAction(self, identifier):
        action = self.timepointActions[identifier.uid]
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(pg.mkPen(color=identifier.color, width=4))
        painter.drawLine(16, 2, 16, 30)
        painter.setBrush(pg.mkBrush(identifier.color))
        painter.drawEllipse(9, 9, 14, 14)
        painter.end()
        action.setIcon(QIcon(pixmap))
        action.setText(identifier.name)
        action.setShortcut(QKeySequence(identifier.shortcut))
        autoText, autoUsedBy = self.autoShortcutConflict(
            identifier.shortcut, excludeUid=identifier.uid
        )
        if autoUsedBy is not None:
            self.logger.warning(
                f'Auto shortcut "{autoText}" of time point '
                f'"{identifier.name}" is not set because it is already used '
                f'by "{autoUsedBy}".'
            )
        if not identifier.shortcut:
            shortcutText = 'No shortcut'
        elif autoUsedBy is not None:
            shortcutText = (
                f'Shortcut: "{identifier.shortcut}", Auto: none '
                f'("{autoText}" is used by "{autoUsedBy}")'
            )
        elif autoText:
            shortcutText = (
                f'Shortcut: "{identifier.shortcut}", Auto: "{autoText}" '
                '(re-evaluate the predictor on the selected cell)'
            )
        else:
            shortcutText = f'Shortcut: "{identifier.shortcut}", Auto: none'
        predictorText = timepoint_predictors.TimepointPredictor.describe(
            identifier.predictor
        )
        action.setToolTip(qutils.wrap_tooltip(
            f'Time point: {identifier.name}\n\n'
            'Click to add it for the selected cell at the current frame.\n'
            'Hover over the measurements plot and press the shortcut to add '
            'it at the hovered frame.\n'
            'Right-click for more options.\n\n'
            f'{shortcutText}\nPredictor: {predictorText}'
        ))
        
        # update AutoAction
        autoAction = self.timepointAutoActions[identifier.uid]
        autoAction.setText(f'Re-evaluate "{identifier.name}" predictor')
        autoAction.setShortcut(
            QKeySequence(autoText if autoUsedBy is None else '')
        )

    
    def showTimepointActionContextMenu(self, uid, pos):
        identifier = self.timepointIdentifier(uid)
        button = self.timepointsToolBar.widgetForAction(
            self.timepointActions[uid]
        )
        menu = QMenu(button)
        menu.addAction('Edit...', partial(self.editTimepointIdentifier, uid))
        hasPredictor = (
            identifier.predictor['type'] != timepoint_predictors.PREDICTOR_NONE
        )
        predictAllAction = menu.addAction(
            'Re-evaluate predictor on all cells', 
            partial(self.reevaluateTimepoints, [uid])
        )
        predictCellAction = menu.addAction(
            'Re-evaluate predictor on displayed cell', 
            partial(self.reevaluateDisplayedCellTimepoints, uids=[uid])
        )
        predictAllAction.setEnabled(hasPredictor and bool(self.data))
        predictCellAction.setEnabled(
            hasPredictor and bool(self.data) and self.selectedID is not None
        )
        menu.addAction(
            'Clear time points of current position', 
            partial(self.clearTimepoints, uid)
        )
        menu.addSeparator()
        menu.addAction('Remove', partial(self.removeTimepointIdentifier, uid))
        menu.exec_(button.mapToGlobal(pos))
    
    def editTimepointIdentifier(self, uid):
        identifier = self.timepointIdentifier(uid)
        values = self.execTimepointIdentifierDialog(
            identifier, f'Edit time point identifier "{identifier.name}"'
        )
        if values is None:
            return
        predictorChanged = values['predictor'] != identifier.predictor
        oldColumn = self.timepointColumn(identifier)
        self.logger.info(
            f'Editing time point identifier "{identifier.name}"'
            + (f' (renamed to "{values["name"]}")'
               if values['name'] != identifier.name else '')
            + (' (predictor changed)' if predictorChanged else '')
        )
        identifier.name = values['name']
        newColumn = self.timepointColumn(identifier)
        if newColumn != oldColumn:
            self.renamePhaseTranslatorInput(oldColumn, newColumn)
            for acdc_df in self.allAcdcDfs():
                if oldColumn not in acdc_df.columns:
                    continue
                acdc_df.drop(columns=newColumn, inplace=True, errors='ignore')
                acdc_df.rename(columns={oldColumn: newColumn}, inplace=True)
            self.renameMeasurementColumnReferences(oldColumn, newColumn)
            self.plotMeasurements()
        identifier.color = values['color']
        identifier.shortcut = values['shortcut']
        identifier.predictor = values['predictor']
        identifier.autoApply = values['autoApply']
        self.updateTimepointAction(identifier)
        self.updateReevaluateTimepointsActions()
        if predictorChanged:
            for key in list(self.timepointAutoPredictions):
                if key[1] == uid:
                    del self.timepointAutoPredictions[key]
            # Visited cells keep their time points (as for annotations).
            self.seedVisitedTimepointPredictions([uid])
        self.updateTimepointLines()
    
    def removeTimepointIdentifier(self, uid, ask=True, updateColumns=True):
        identifier = self.timepointIdentifier(uid)
        if ask:
            column = self.timepointColumn(identifier)
            # The column's own formatting is removed with it
            ownFormatting = f'conditional formatting of "{column}"'
            dependents = [
                dependent
                for dependent in self.columnDependents(
                    column, excludeOutput=column, includePhaseInputs=True
                )
                if dependent != ownFormatting
            ]
            note = (
                f'\n\nNote: "{column}" is used by {", ".join(dependents)}, '
                'which will fail until edited.' if dependents else ''
            )
            msg = widgets.myMessageBox(wrapText=False)
            _, yes = msg.question(
                self, 'Remove time point identifier?',
                f'Remove "{identifier.name}" and its time points of all '
                f'positions?{note}', buttonsTexts=('Cancel', 'Remove')
            )
            if msg.clickedButton != yes:
                return
            self.logger.info(
                f'Removing time point identifier "{identifier.name}"'
            )
        action = self.timepointActions.pop(uid)
        self.timepointsToolBar.removeAction(action)
        # Still a child until deleted: free its key for `reservedShortcuts`
        action.setShortcut(QKeySequence())
        action.deleteLater()
        self.timepointIdentifiers.remove(identifier)
        for key in list(self.timepointAutoPredictions):
            if key[1] == uid:
                del self.timepointAutoPredictions[key]
        column = self.timepointColumn(identifier)
        for acdc_df in self.allAcdcDfs():
            acdc_df.drop(columns=column, inplace=True, errors='ignore')
        # The column's own plot settings go with it (as for custom
        # annotations). On load, the configuration's formatting is assigned
        # after `setTimepointIdentifiers`, so it is not wiped here.
        self.measurementsColors.pop(column, None)
        self._defaultSignalColorIndices.pop(column, None)
        self.conditionalFormatting.pop(column, None)
        for posData in self.data:
            getattr(posData, 'timepointsEdited', {}).pop(uid, None)
            
        autoAction = self.timepointAutoActions.pop(uid)
        self.removeAction(autoAction)
        autoAction.setShortcut(QKeySequence())
        autoAction.deleteLater()

        self.updateReevaluateTimepointsActions()
        if updateColumns:
            self.updateColumnsList()
            self.plotMeasurements()
        self.updateTimepointLines()

    def clearTimepoints(self, uid):
        if not self.data:
            return
        self.logger.info(
            f'Clearing time points "{self.timepointIdentifier(uid).name}" '
            'of all cells'
        )
        self.replaceTimepoints(uid, {}, self.timepointCellIDs(uid))
        self.editedTimepointCells()[uid] = set()
        self.refreshPlotForColumns(
            [self.timepointColumn(self.timepointIdentifier(uid))]
        )
    
    def setTimepointIdentifiers(self, identifiers):
        for identifier in list(self.timepointIdentifiers):
            self.removeTimepointIdentifier(
                identifier.uid, ask=False, updateColumns=False
            )
        for identifier in identifiers:
            self.addTimepointIdentifierItem(identifier)
        self.updateColumnsList()
        self.plotMeasurements()

    def runTimepointPredictor(
            self, uid, posData=None, replot=False, IDs=None, skipIDs=(),
            force=False
        ):
        """Predict the time points of `uid` for the cells of `posData` 
        (default current position), replacing the existing ones.
        
        If `IDs` is not None, only these cells are predicted. Cells in 
        `skipIDs` keep their current time points. Unless `force` is True,
        require `autoApply` and skip cells already predicted in this session
        for the same position path and identifier. Returns False if the
        prediction failed."""
        identifier = self.timepointIdentifier(uid)
        if not self.data or identifier is None:
            return False
        if not force and not identifier.autoApply:
            return True
        
        if posData is None:
            posData = self.posData()
        self.initPositionTimepointState(posData)
        if identifier.predictor['type'] == timepoint_predictors.PREDICTOR_NONE:
            return True
        acdc_df = (
            self.acdc_df if posData is self.posData() 
            else self.getAcdcDf(posData)
        )
        predictionKey = (os.path.normcase(os.path.abspath(posData.pos_path)), uid)
        alreadyPredicted = self.timepointAutoPredictions[predictionKey]
        skipIDs = set(skipIDs)
        if IDs is not None:
            IDs = set(IDs) - skipIDs
        if not force:
            if IDs is None:
                IDs = (
                    set(acdc_df.index.get_level_values('Cell_ID'))
                    if acdc_df is not None else set()
                )
            IDs -= alreadyPredicted | skipIDs
        if IDs is not None and not IDs:
            return True
        try:
            # All cells are passed since predictors can use other cells
            # Only the analyzed frames (`analysisDf`): cut on every call,
            # since prerequisites write their outputs into the full table
            predicted = timepoint_predictors.TimepointPredictor.predict(
                self.analysisDf(acdc_df, posData), identifier.predictor,
                SizeT=self.analysisSizeT(posData), IDs=IDs
            )
        except Exception as err:
            self.logger.exception('Time point prediction failed')
            self.measurementProcessingWarning(
                'Time point prediction failed', 
                f'Predicting "{identifier.name}" failed:\n\n{err}'
            )
            return False
        
        if IDs is None:
            replacedIDs = (
                self.timepointCellIDs(uid, posData).union(predicted) - skipIDs
            )
        else:
            replacedIDs = set(IDs) - skipIDs
        self.replaceTimepoints(
            uid, {ID: predicted[ID] for ID in replacedIDs if ID in predicted},
            replacedIDs, posData
        )
        posData.timepointsEdited[uid].difference_update(replacedIDs)

        completedIDs = (
            set(acdc_df.index.get_level_values('Cell_ID'))
            if acdc_df is not None else set()
        )
        if IDs is not None:
            completedIDs.intersection_update(IDs)
        alreadyPredicted.update(completedIDs - skipIDs)

        numCells = sum(1 for ID in replacedIDs if ID in predicted)
        numTimepoints = sum(len(predicted[ID]) for ID in replacedIDs if ID in predicted)
        self.statusBar().showMessage(
            f'Predicted {numTimepoints} "{identifier.name}" time point(s) '
            f'for {numCells} cell(s).', 5000
        )
        if replot:
            self.refreshPlotForColumns([self.timepointColumn(identifier)])
        return True
    
    def updateReevaluateTimepointsActions(self):
        hasPredictors = bool(self.predictorTimepointUids())
        self.reevaluateAllTimepointsAction.setEnabled(hasPredictors)
        self.reevaluateCellTimepointsAction.setEnabled(hasPredictors)

    def updateReevaluateCustomAnnotationActions(self):
        hasPredictors = any(
            self.customAnnotHasPredictor(button)
            for button in self.customAnnotDict
        )
        self.reevaluateAllCustomAnnotationsAction.setEnabled(hasPredictors)
        self.reevaluateCellCustomAnnotationsAction.setEnabled(hasPredictors)
    
    def askOverwriteManualTimepoints(self, editedCells, scope, what='Time points'):
        """Warn that re-evaluating overwrites manual edits.
        
        Parameters
        ----------
        editedCells : dict
            {identifier name: sorted list of edited Cell_IDs}
        scope : str
            Description of the re-evaluated cells.
        what : str, optional
            What has been edited (e.g., 'Custom annotations').

        Returns
        -------
        str
            'overwrite', 'skip' (keep the manually edited cells) or 'cancel'.
        """
        details = '\n'.join(
            f'{name}: Cell ID {", ".join(str(ID) for ID in IDs)}'
            for name, IDs in editedCells.items()
        )
        txt = html_utils.paragraph(
            f'{what} of {scope} have been <b>edited manually</b>.<br><br>'
            'Re-evaluating the predictors will <b>overwrite</b> these edits.'
            '<br><br>Do you want to overwrite the manual edits or skip the '
            'edited cells?'
        )
        msg = widgets.myMessageBox(wrapText=False)
        _, skipButton, overwriteButton = msg.warning(
            self, 'Overwrite manual edits?', txt, detailsText=details,
            buttonsTexts=('Cancel', 'Skip edited cells', 'Overwrite')
        )
        if msg.clickedButton == overwriteButton:
            return 'overwrite'
        if msg.clickedButton == skipButton:
            return 'skip'
        return 'cancel'
    
    def reevaluateTimepoints(self, uids=None, IDs=None, posData=None):
        """Re-run the predictors of `uids` (default all identifiers with 
        a predictor) on cells `IDs` (default all cells) of `posData`, 
        warning if manual edits would be overwritten. Returns False if 
        cancelled."""
        if not self.data:
            return False
        if posData is None:
            posData = self.posData()
        uids = self.predictorTimepointUids(uids)
        if not uids:
            self.statusBar().showMessage(
                'No time point identifier with a predictor.', 5000
            )
            return False
        columns = [
            self.timepointColumn(self.timepointIdentifier(uid)) for uid in uids
        ]
        self.logger.info(
            f'Re-evaluating time points {", ".join(columns)} of '
            + ('all cells' if IDs is None else f'Cell ID(s) {list(IDs)}')
        )
        return self.evaluatePredictorColumns(columns, IDs, posData)
    
    def reevaluateAllTimepoints(self, checked=False):
        return self.reevaluateTimepoints()
    
    def reevaluateDisplayedCellTimepoints(self, checked=False, uids=None):
        if self.selectedID is None:
            self.statusBar().showMessage('Select a cell first.', 5000)
            return False
        return self.reevaluateTimepoints(uids=uids, IDs=[self.selectedID])
    
    def ensureTimepointPredictions(self, posData=None):
        """Evaluate automatic predictors only on the newly displayed cell."""
        self.ensurePredictorPredictions(posData)

    @contextmanager
    def deferPredictorPredictions(self):
        previous = getattr(self, '_deferPredictorPredictions', False)
        self._deferPredictorPredictions = True
        try:
            yield
        finally:
            self._deferPredictorPredictions = previous
        if not previous:
            self.ensurePredictorPredictions()

    def renameMeasurementColumnReferences(self, oldColumn, newColumn):
        """Move the plot selection, axes, color and conditional formatting
        of the renamed table column `oldColumn` to `newColumn` and update
        the columns list."""
        if oldColumn == newColumn:
            return
        if oldColumn in self._timeCourseColumns:
            self._timeCourseColumns.discard(oldColumn)
            self._timeCourseColumns.add(newColumn)
        for axis, column in self.measurementsAxes.items():
            if column == oldColumn:
                self.measurementsAxes[axis] = newColumn
        for mapping in (
                self.measurementsColors, self.conditionalFormatting,
                self._defaultSignalColorIndices):
            if oldColumn in mapping:
                mapping[newColumn] = mapping.pop(oldColumn)
        self.updateColumnsList(renamed={oldColumn: newColumn})

    def evaluatePredictorColumns(
            self, columns, IDs, posData, automatic=False
        ):
        definitions = self.predictorDefinitions()
        predictors = {
            column: definition[2] for column, definition in definitions.items()
            if definition[2].get('type') != timepoint_predictors.PREDICTOR_NONE
        }
        try:
            ordered = timepoint_predictors.PredictorDependencies.order(
                predictors, columns
            )
        except ValueError as err:
            self.measurementProcessingWarning('Invalid predictors', str(err))
            return False
        if automatic:
            ordered = [column for column in ordered if definitions[column][3]]
        scopes = {column: None if IDs is None else set(IDs) for column in ordered}
        for column in reversed(ordered):
            inputs = timepoint_predictors.PredictorDependencies.inputs(
                predictors[column]
            )
            for dependency, otherCells in inputs.items():
                if dependency in scopes and (
                        otherCells or scopes[column] is None):
                    scopes[dependency] = None
        edited = {}
        editedCells = {}
        for column in ordered:
            kind, key, _, _ = definitions[column]
            cells = (
                self.editedTimepointCells(posData)[key]
                if kind == 'timepoint' else
                self.editedCustomAnnotCells(posData)[column]
            )
            scopeIDs = scopes[column]
            edited[column] = set(cells) if scopeIDs is None else cells & scopeIDs
            if edited[column]:
                name = (
                    self.timepointIdentifier(key).name
                    if kind == 'timepoint' else column
                )
                editedCells[name] = sorted(edited[column])
        # Automatic predictions never overwrite manual edits (no prompt):
        # edited cells, including those of prerequisites, are skipped
        skip = automatic
        if editedCells and not automatic:
            allEdited = set().union(*editedCells.values())
            scope = (
                f'the displayed cell (ID {next(iter(IDs))})'
                if IDs is not None and len(IDs) == 1 and allEdited <= set(IDs)
                else f'{len(allEdited)} cell(s)'
            )
            if posData is not self.posData():
                posName = getattr(posData, 'pos_foldername', 'another position')
                scope = f'{scope} of position "{posName}"'
            kinds = {definitions[column][0] for column in ordered}
            if kinds == {'timepoint'}:
                answer = self.askOverwriteManualTimepoints(editedCells, scope)
            else:
                what = (
                    'Custom annotations' if kinds == {'annotation'}
                    else 'Time points and custom annotations'
                )
                answer = self.askOverwriteManualTimepoints(
                    editedCells, scope, what=what
                )
            if answer == 'cancel':
                return False
            skip = answer == 'skip'
        failed = set()
        names = []
        for column in ordered:
            inputs = timepoint_predictors.PredictorDependencies.inputs(
                predictors[column]
            )
            if failed.intersection(inputs):
                failed.add(column)
                self.statusBar().showMessage(
                    f'Skipped "{column}": a prerequisite prediction failed.', 5000
                )
                continue
            kind, key, _, _ = definitions[column]
            run = (
                self.runTimepointPredictor if kind == 'timepoint'
                else self.runCustomAnnotPredictor
            )
            if not run(
                    key, posData=posData, IDs=scopes[column],
                    skipIDs=edited[column] if skip else (),
                    force=not automatic
                ):
                failed.add(column)
            if kind == 'annotation':
                names.append(column)
        if automatic and failed:
            path = os.path.normcase(os.path.abspath(posData.pos_path))
            acdc_df = self.timepointsDf(posData)
            for column in failed:
                scopeIDs = scopes[column]
                if scopeIDs is None:
                    scopeIDs = (
                        set(acdc_df.index.get_level_values('Cell_ID'))
                        if acdc_df is not None else set()
                    )
                kind, key, _, _ = definitions[column]
                predictions = (
                    self.timepointAutoPredictions if kind == 'timepoint'
                    else self.customAnnotAutoPredictions
                )
                predictions[(path, key if kind == 'timepoint' else column)].update(
                    scopeIDs
                )
        if names:
            # Redraw the annotations, the plot is refreshed below
            self.doCustomAnnotation(0)
        self.refreshPlotForColumns(ordered)
        return not failed

    def ensurePredictorPredictions(self, posData=None):
        if getattr(self, '_deferPredictorPredictions', False):
            return
        if not self.data or self.selectedID is None:
            return
        if posData is None:
            posData = self.posData()
        if posData is not self.posData():
            return
        ID = int(self.selectedID)
        acdc_df = self.timepointsDf(posData)
        if acdc_df is None:
            return
        if VISITED_COLUMN not in acdc_df:
            acdc_df[VISITED_COLUMN] = False
        rows = acdc_df.index.get_level_values('Cell_ID') == ID
        newCell = not parse_flags(acdc_df.loc[rows, VISITED_COLUMN]).any()
        acdc_df.loc[rows, VISITED_COLUMN] = True
        posData.acdc_df = acdc_df
        self.initPositionTimepointState(posData)
        columns = []
        for column, definition in self.predictorDefinitions().items():
            kind, key, predictor, autoApply = definition
            if (not autoApply
                    or predictor.get('type') == timepoint_predictors.PREDICTOR_NONE):
                continue
            if kind == 'timepoint':
                if not newCell:
                    continue
            elif ID in self.customAnnotAutoPredictions[
                    self.customAnnotPredictionKey(posData, column)]:
                continue
            columns.append(column)
        if columns:
            self.evaluatePredictorColumns(columns, [ID], posData, automatic=True)
        self.ensurePhasePrediction(posData, ID)
    
    # --- Custom annotations -------------------------------------------------
    # Overrides of the `gui_custom_annotations.CustomAnnotationsGuiElements` 
    # hooks. The annotations are stored as 0/1 columns of the acdc_df of each 
    # position (the frame tables of `posData.allData_li` are temporary copies).
    def addCustomAnnotation(self):
        if not self.data:
            self.statusBar().showMessage(
                'Load data to add custom annotations.', 5000
            )
            return
        return super().addCustomAnnotation()
    
    def loadCustomAnnotations(self):
        if not self.data:
            self.statusBar().showMessage(
                'Load data to load custom annotations.', 5000
            )
            return
        with self.deferPredictorPredictions():
            return super().loadCustomAnnotations()
    
    def schedulePhaseOverrideCells(self, names, phases=()):
        """Re-evaluate (deferred, `schedulePhaseAutoApply`) the cells of
        every loaded position annotated with `names` or with one of
        `phases`, e.g. after an annotation started or stopped overriding
        phases or was renamed or removed.

        When neither phase rules nor other overriding annotations are left
        (`hasPhaseTranslation`), nothing could evaluate these cells again:
        their `phases` are cleared instead (deferred,
        `clearPendingPhases`), also with automatic evaluation off."""
        if getattr(self, '_isBulkSettingCustomAnnotations', False):
            return
        if not self.hasPhaseTranslation():
            phases = set(phases)
            for posData in self.data or []:
                if not self.phaseOverrideCells(posData, (), phases):
                    continue
                _, pending = self._pendingPhaseClears.setdefault(
                    id(posData), (posData, set())
                )
                pending.update(phases)
            self.schedulePendingPhaseClears()
            return
        for posData in self.data or []:
            IDs = self.phaseOverrideCells(posData, names, phases)
            if IDs:
                self.schedulePhaseAutoApply(IDs, posData)

    def schedulePendingPhaseClears(self):
        """Start the (single) deferred `clearPendingPhases`."""
        if self._phaseClearScheduled or not self._pendingPhaseClears:
            return
        self._phaseClearScheduled = True
        QTimer.singleShot(0, self.clearPendingPhases)

    def clearPendingPhases(self, positions=None):
        """Clear the phases recorded by `schedulePhaseOverrideCells` (the
        names of removed or switched off overriding annotations) in the
        loaded `positions` (default: all), by evaluating their cells
        without rules and overrides (`phase_translator.evaluate` then
        resets them and drops the unused ``phase_<name>`` outputs). If
        phases can be evaluated again by now, the cells are scheduled for
        automatic evaluation instead. Postponed while measurement
        processing runs (`measurementProcessingFinished`)."""
        self._phaseClearScheduled = False
        if self.measurementProcessingThread is not None:
            return
        loaded = {id(posData): posData for posData in self.data or []}
        if positions is None:
            positions = list(loaded.values())
        displayedChanged = False
        for posData in positions:
            entry = self._pendingPhaseClears.pop(id(posData), None)
            if entry is None or loaded.get(id(posData)) is not entry[0]:
                continue
            phases = entry[1]
            IDs = self.phaseOverrideCells(posData, (), phases)
            if not IDs:
                continue
            if self.hasPhaseTranslation():
                self.schedulePhaseAutoApply(IDs, posData)
                continue
            try:
                self.evaluatePositionPhases(posData, IDs, rules=[])
            except Exception as err:
                self.logger.exception('Clearing phases failed')
                self.statusBar().showMessage(
                    f'Clearing phases failed: {err}', 8000
                )
                continue
            self.logger.info(
                'Cleared the phases '
                + ', '.join(f'"{phase}"' for phase in sorted(phases))
                + f' of Cell ID(s) {sorted(IDs)} of '
                + f'"{getattr(posData, "pos_foldername", "position")}"'
            )
            displayedChanged = displayedChanged or posData is self.posData()
        # Positions no longer loaded are forgotten
        self._pendingPhaseClears = {
            key: entry for key, entry in self._pendingPhaseClears.items()
            if loaded.get(key) is entry[0]
        }
        if displayedChanged:
            self.updateColumnsList()
            self.plotMeasurements()

    def flushPendingPhaseEvaluations(self, positions):
        """Run now the deferred phase evaluations of `positions` (automatic
        evaluation of changed cells, `schedulePhaseAutoApply`, which waits
        for a position to be displayed, and phase clears), e.g. before
        their tables are saved, so that no table is saved with phases that
        do not match its annotations. Skipped while measurement processing
        runs."""
        if self.measurementProcessingThread is not None or not self.data:
            return
        self.clearPendingPhases(positions)
        if not self.autoApplyPhases or not self.hasPhaseTranslation():
            return
        self.prunePendingPhaseIDs()
        displayed = self.posData()
        for posData in positions:
            if posData is displayed:
                # Updates the display too
                if id(displayed) in self._pendingPhaseIDs:
                    self.autoApplyPhaseTranslator()
                continue
            entry = self._pendingPhaseIDs.pop(id(posData), None)
            if entry is None:
                continue
            table = self.timepointsDf(posData)
            if table is None or table.empty:
                continue
            IDs = entry[1].intersection(
                table.index.get_level_values('Cell_ID')
            )
            if not IDs:
                continue
            try:
                self.evaluatePositionPhases(posData, IDs)
            except Exception as err:
                self.logger.exception('Phase evaluation failed')
                self.statusBar().showMessage(
                    f'Phase evaluation failed: {err}', 8000
                )

    def reinitCustomAnnot(self):
        with self.bulkSettingCustomAnnotations():
            super().reinitCustomAnnot()

    def customAnnotIsVisibleMode(self):
        return bool(self.data)
    
    def defaultCustomAnnotShortcut(self, reservedShortcuts):
        for key in custom_annotation_predictors.DEFAULT_SHORTCUTS:
            if QKeySequence(key).toString() not in reservedShortcuts:
                return key
        return ''
    
    def customAnnotationDialogKwargs(self, state=None):
        posData = self.posData()
        button = (
            None if state is None 
            else self.customAnnotButtonByName(state['name'])
        )
        reservedShortcuts = self.reservedShortcuts(excludeCustomAnnot=button)
        kwargs = {
            'predictorOptions': {
                'acdc_df': self.dialogAnalysisDf(
                    self.timepointsDf(posData), posData
                ),
                'SizeT': self.dialogAnalysisSizeT(posData),
                'allChNames': getattr(posData, 'chNames', []),
                'isZstack': getattr(posData, 'SizeZ', 1) > 1,
                'isSegm3D': bool(self.isSegm3D),
                'validateDependencies': (
                    lambda name, predictor: self.validatePredictorDependencies(
                        None if state is None else state['name'],
                        name, predictor
                    )
                ),
            },
            'reservedShortcuts': reservedShortcuts,
            'existingNames': [
                name for name in self.customAnnotNames()
                if state is None or name != state['name']
            ],
        }
        if state is None:
            kwargs['state'] = {
                'type': 'Single time-point',
                'name': '',
                'symbol': '',
                'shortcut': self.defaultCustomAnnotShortcut(reservedShortcuts),
                'description': '',
            }
        return kwargs
    
    def customAnnotToolTip(self, state):
        shortcut = state.get('shortcut') or ''
        autoText = auto_shortcut_text(shortcut)
        if shortcut:
            autoDescr = (
                f'Auto: "{autoText}" (re-evaluate the predictor on the '
                'selected cell)' if autoText else 'Auto: none'
            )
            shortcutText = (
                f'Shortcut: "{shortcut}" (annotate/un-annotate the selected '
                f'cell across its entire video history)\n{autoDescr}'
            )
        else:
            shortcutText = 'No shortcut'
        predictor = CustomAnnotationPredictor.describe(state.get('predictor'))
        if state.get('autoApply', False):
            predictor = f'{predictor} (auto apply on unvisited cells)'
        lines = [
            f'Custom annotation: {state["name"]}', '',
            'Usage: select a cell and press the shortcut, or\n'
            'activate the button and RIGHT-click on a cell.',
            shortcutText, f'Predictor: {predictor}'
        ]
        if state.get('overridesPhases', False):
            lines.append(
                f'Overrides phases: annotated cells get the phase '
                f'"{state["name"]}" on all their frames (before the phase '
                'rules; the first overriding annotation of the toolbar set '
                'in a cell wins)'
            )
        description = state.get('description', '')
        if description:
            lines.extend(['', f'Description: {description}'])
        lines.extend([
            '', 'RIGHT-click on the button to modify,\n'
                're-evaluate, clear or remove the annotation.'
        ])
        # Names, descriptions and predictors are entered by the user
        return qutils.wrap_tooltip('\n'.join(lines))
    
    def setCustomAnnotShortcut(self, button, shortcut):
        """The shortcut annotates the selected cell across its entire history
        (instead of activating the button) and Shift+shortcut re-evaluates
        the predictor on the selected cell.

        The Shift+shortcut is not set if `shortcut` contains Shift or if it
        is already used (e.g., by a configuration created before it was
        reserved). Neither shortcut is set if `shortcut` itself is already
        used by another action (e.g., an annotation from 'Load
        annotation...' or a configuration, which skip the dialog checks),
        because Qt would trigger neither of two actions sharing a key. In
        both cases a warning is logged and a note is appended to the button
        tool tip, which must be set before calling this method."""
        button.setShortcut(QKeySequence())
        actions = self.customAnnotShortcutActions.get(button)
        if actions is None:
            toggleAction = QAction(self)
            toggleAction.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            toggleAction.triggered.connect(
                partial(self.customAnnotShortcutTriggered, button)
            )
            autoAction = QAction(self)
            autoAction.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            autoAction.triggered.connect(
                partial(self.reevaluateDisplayedCellCustomAnnot, button)
            )
            self.addAction(toggleAction)
            self.addAction(autoAction)
            actions = (toggleAction, autoAction)
            self.customAnnotShortcutActions[button] = actions
        toggleAction, autoAction = actions
        if isinstance(shortcut, QKeySequence):
            shortcut = shortcut.toString()
        keySequence = QKeySequence(shortcut or '')
        name = getattr(button, 'name', '')
        keyText = keySequence.toString()
        usedBy = None
        if keyText:
            usedBy = self.reservedShortcuts(
                excludeCustomAnnot=button, shiftDerived=False
            ).get(keyText)
        if usedBy is not None:
            self.logger.warning(
                f'Shortcut "{keyText}" of custom annotation '
                f'"{name}" is not set because it is already used by '
                f'"{usedBy}".'
            )
            button.setToolTip(qutils.wrap_tooltip(
                f'{button.toolTip()}\n\nNote: the shortcut "{keyText}" '
                f'(and its auto shortcut) is not set because it is already '
                f'used by "{usedBy}".'
            ))
            shortcut = ''
            keySequence = QKeySequence()
        autoText, autoUsedBy = self.autoShortcutConflict(
            shortcut, excludeCustomAnnot=button
        )
        if autoUsedBy is not None:
            self.logger.warning(
                f'Auto shortcut "{autoText}" of custom annotation '
                f'"{name}" is not set because it is already used by '
                f'"{autoUsedBy}".'
            )
            button.setToolTip(qutils.wrap_tooltip(
                f'{button.toolTip()}\n\nNote: the auto shortcut "{autoText}" '
                f'is not set because it is already used by "{autoUsedBy}".'
            ))
            autoText = ''
        toggleAction.setShortcut(keySequence)
        autoAction.setShortcut(QKeySequence(autoText))
        toggleAction.setText(f'Annotate "{name}"')
        autoAction.setText(f'Re-evaluate "{name}" predictor')

    def addCustomAnnnotScatterPlot(self, symbolColor, symbol, toolButton):
        super().addCustomAnnnotScatterPlot(symbolColor, symbol, toolButton)
        # Overlay layers are re-added at z-value 0 on every load: keep the
        # symbols above them (annotation items 0.5-0.57) and below the
        # text IDs (1)
        scatterPlotItem = self.customAnnotDict[toolButton]['scatterPlotItem']
        scatterPlotItem.setZValue(0.6)

    def initCustomAnnotColumn(self, button, name):
        # Positions whose table already holds assignments of `name` (e.g.,
        # exported by an earlier session): their visited cells are not
        # auto-predicted (see `onCustomAnnotationChanged`)
        self.customAnnotPreexistingColumns[name] = [
            posData for posData in self.data or []
            if any(
                name in table.columns
                for table in self.positionTables(posData)
            )
        ]
        for acdc_df in self.allAcdcDfs():
            self.ensureCustomAnnotColumn(acdc_df, name)
        self.updateColumnsList()

    def renameCustomAnnotColumn(self, oldName, newName):
        self.renamePhaseTranslatorInput(oldName, newName)
        for acdc_df in self.allAcdcDfs():
            if oldName in acdc_df.columns:
                acdc_df.rename(columns={oldName: newName}, inplace=True)
        self.renameMeasurementColumnReferences(oldName, newName)
        self.plotMeasurements()
    
    def customAnnotColumnRemovalNote(self, name):
        # The column's own formatting is removed with it
        ownFormatting = f'conditional formatting of "{name}"'
        dependents = [
            dependent
            for dependent in self.columnDependents(
                name, excludeOutput=name, includePhaseInputs=True
            )
            if dependent != ownFormatting
        ]
        notes = []
        if dependents:
            notes.append(
                f'"{name}" is used by {", ".join(dependents)}, which will '
                'fail until edited if the column is removed.'
            )
        overrides = self.phaseOverrideAnnotations()
        if name in overrides:
            if not self.phaseTranslatorRules and overrides == [name]:
                # Nothing left to evaluate phases with
                # (`schedulePhaseOverrideCells`)
                notes.append(
                    f'"{name}" overrides phases: once it is removed, the '
                    f'phase "{name}" of its annotated cells is cleared.'
                )
            else:
                notes.append(
                    f'"{name}" overrides phases: once it is removed, its '
                    'annotated cells get their phases from the phase rules '
                    'again'
                    + (' (re-evaluated automatically).' if self.autoApplyPhases
                       else ' when phases are evaluated.')
                )
        if not notes:
            return ''
        return 'Note: ' + ' '.join(notes)

    def removeCustomAnnotColumn(self, name):
        for acdc_df in self.allAcdcDfs():
            acdc_df.drop(columns=name, errors='ignore', inplace=True)
        # The column's own plot settings go with it (`updateColumnsList`
        # drops it from the plotted columns and comparison axes)
        self.measurementsColors.pop(name, None)
        self._defaultSignalColorIndices.pop(name, None)
        self.conditionalFormatting.pop(name, None)
        self.updateColumnsList()
        self.plotMeasurements()
    
    def onCustomAnnotationChanged(self, button, oldState=None):
        state = self.customAnnotDict[button]['state']
        name = state['name']
        oldName = name if oldState is None else oldState['name']
        if oldState is None:
            self.logger.info(f'Custom annotation "{name}" added')
        else:
            self.logger.info(
                f'Custom annotation "{oldName}" edited'
                + (f' (renamed to "{name}")' if oldName != name else '')
            )
        overrides = bool(state.get('overridesPhases', False))
        oldOverrides = oldState is not None and bool(
            oldState.get('overridesPhases', False)
        )
        if overrides != oldOverrides:
            self.logger.info(
                f'Custom annotation "{name}" '
                + ('overrides phases' if overrides else
                   'no longer overrides phases')
            )
        if oldName != name:
            for key in list(self.customAnnotAutoPredictions.keys()):
                if key[1] == oldName:
                    self.customAnnotAutoPredictions[(key[0], name)] = (
                        self.customAnnotAutoPredictions.pop(key)
                    )
            for posData in self.data or []:
                edited = self.editedCustomAnnotCells(posData)
                if oldName in edited:
                    edited[name] = edited.pop(oldName)
            if oldOverrides:
                # The color of the phase named after the annotation follows
                # the rename (unless a rule also assigns the old name)
                rulePhases = phase_translator.phase_names(
                    self.phaseTranslatorRules
                )
                if (oldName in self.phaseColors
                        and oldName not in rulePhases
                        and name not in self.phaseColors):
                    self.phaseColors[name] = self.phaseColors.pop(oldName)
        predictorChanged = (
            oldState is None 
            or oldState.get('predictor') != state.get('predictor')
        )
        preexisting = self.customAnnotPreexistingColumns.pop(name, [])
        if predictorChanged:
            self.clearCustomAnnotPredictionsTracking(name)
            # Like time points, automatic prediction only runs on cells not
            # viewed yet: a new or edited predictor keeps the (saved or
            # manual) assignments of visited cells. Only a brand-new column
            # is predicted on visited cells.
            if oldState is not None:
                self.seedVisitedCustomAnnotPredictions([name])
            elif preexisting:
                self.seedVisitedCustomAnnotPredictions([name], preexisting)
        elif (
            not oldState.get('autoApply', False)
            and state.get('autoApply', False)
        ):
            # Cells viewed while auto apply was off were never recorded as
            # predicted: enabling it keeps their assignments too.
            self.seedVisitedCustomAnnotPredictions([name])

        button.extraContextMenuActions = []
        if self.customAnnotHasPredictor(button):
            button.extraContextMenuActions.extend([
                (
                    'Re-evaluate predictor on all cells', 
                    self.reevaluateAllCellsCustomAnnot
                ),
                (
                    'Re-evaluate predictor on displayed cell', 
                    self.reevaluateDisplayedCellCustomAnnot
                ),
            ])
        button.extraContextMenuActions.append(
            ('Clear annotations of current position', self.clearCustomAnnotations)
        )
        self.updateReevaluateCustomAnnotationActions()
        if not self.data:
            return
        if (overrides or oldOverrides) and (
                overrides != oldOverrides or oldName != name):
            # The option was switched, an overriding annotation was renamed
            # (its cells get the new name) or a new overriding annotation
            # already has annotated cells: re-evaluate the annotated cells
            # and the cells with the annotation's (old) name as phase. Not
            # while annotations are set in bulk (configuration load).
            self.schedulePhaseOverrideCells([name], {oldName, name})
        self.ensureCustomAnnotPredictions()
        self.refreshCustomAnnotations([name])

    def onCustomAnnotationRemoved(self, button, state):
        for action in self.customAnnotShortcutActions.pop(button, ()):
            self.removeAction(action)
            # Still a child until deleted: free its key for the
            # `reservedShortcuts` check of annotations added right after
            action.setShortcut(QKeySequence())
            action.deleteLater()
        name = state['name']
        self.logger.info(f'Custom annotation "{name}" removed')
        self.clearCustomAnnotPredictionsTracking(name)
        for posData in self.data or []:
            self.editedCustomAnnotCells(posData).pop(name, None)
        self.updateReevaluateCustomAnnotationActions()
        if state.get('overridesPhases', False):
            # Its cells get their phases from the rules (or the next
            # overriding annotation) again: the cells with its name as
            # phase (its column may already be removed) and, if the column
            # is kept, the annotated cells. Not in bulk (`reinitCustomAnnot`)
            self.schedulePhaseOverrideCells([name], {name})
    
    def replaceCustomAnnotations(self, name, annotated, IDs, posData=None):
        """Replace the cell-wide assignments of IDs from predictor results.

        The phases of the cells whose assignment changed are re-evaluated
        (`schedulePhaseAutoApplyForAnnotation`)."""
        acdc_df = self.timepointsDf(posData)
        if acdc_df is None:
            return False
        if name in acdc_df.columns:
            values = parse_flags(acdc_df[name]).to_numpy().astype(int)
        else:
            values = np.zeros(len(acdc_df), dtype=int)
        previous = values.copy()
        cellIDs = acdc_df.index.get_level_values('Cell_ID')
        values[cellIDs.isin(list(IDs))] = 0
        markedIDs = {
            ID for ID, frames in annotated.items() if frames and ID in IDs
        }
        values[cellIDs.isin(markedIDs)] = 1
        acdc_df[name] = values
        changed = previous != values
        if changed.any():
            self.schedulePhaseAutoApplyForAnnotation(
                name, {int(ID) for ID in cellIDs[changed]}, posData
            )
        return True

    def schedulePhaseAutoApplyForAnnotation(self, name, IDs, posData=None):
        """Re-evaluate (deferred) the phases of the cells `IDs` of `posData`
        (default: the displayed position) whose custom annotation `name`
        changed, if it affects phases (`annotationAffectsPhases`). Nothing is
        scheduled while annotations are set in bulk."""
        if getattr(self, '_isBulkSettingCustomAnnotations', False):
            return
        if not IDs or not self.annotationAffectsPhases(name):
            return
        self.schedulePhaseAutoApply(IDs, posData)
    
    def refreshCustomAnnotations(self, names=None):
        """Redraw the annotations and the plot if it (or its conditional
        formatting) reads one of `names`."""
        self.doCustomAnnotation(0)
        if names is None:
            names = self.customAnnotNames()
        if self.plotReadsColumns(names):
            self.plotMeasurements()
    
    def toggleCustomAnnotation(self, button, ID):
        """Toggle one Cell_ID across its entire history in this position.

        Works on any frame, also where the cell is absent: the annotation is
        one value for the whole cell, so only a cell without any row in the
        measurements table is refused."""
        name = self.customAnnotDict[button]['state']['name']
        acdc_df = self.acdc_df
        if acdc_df is None:
            return False
        mask = acdc_df.index.get_level_values('Cell_ID') == ID
        if not mask.any():
            self.statusBar().showMessage(
                f'Cell ID {ID} is not in the measurements table.', 5000
            )
            return False
        self.ensureCustomAnnotColumn(acdc_df, name)
        annotated = bool(parse_flags(acdc_df.loc[mask, name]).any())
        acdc_df.loc[mask, name] = 0 if annotated else 1
        self.editedCustomAnnotCells()[name].add(int(ID))
        self.schedulePhaseAutoApplyForAnnotation(name, [int(ID)])
        action = 'Removed' if annotated else 'Added'
        self.logger.info(
            f'{action} custom annotation "{name}" for Cell ID {ID}'
        )
        self.statusBar().showMessage(
            f'{action} "{name}" annotation for Cell ID {ID} across its '
            'entire video history.', 5000
        )
        return True
    
    def customAnnotShortcutTriggered(self, button, checked=False):
        if not self.data or button not in self.customAnnotDict:
            return
        if self.selectedID is None:
            self.statusBar().showMessage('Select a cell to annotate.', 5000)
            return
        if self.toggleCustomAnnotation(button, int(self.selectedID)):
            self.refreshCustomAnnotations([button.name])
    
    def annotateClickedObject(self, xdata, ydata):
        """Annotate the object under the right-click with the active custom 
        annotation (as in `gui.guiWin` right-click with an active button)."""
        button = self.customAnnotButton
        posData = self.posData()
        lab2D = self.get_2Dlab(posData.lab)
        try:
            ID = int(lab2D[ydata, xdata])
        except IndexError:
            ID = 0
        if ID == 0:
            self.statusBar().showMessage(
                'RIGHT-click on a cell to annotate it.', 5000
            )
            return
        self.doCustomAnnotation(ID)
        if not button.keepToolActive:
            button.setChecked(False)
    
    def doCustomAnnotation(self, ID):
        """Toggle the annotation of `ID` across its entire history with the
        active button (pass 0 to only redraw) and draw the annotations of 
        the current frame."""
        if not self.data or not self.customAnnotDict:
            return
        if self.switchPlaneCombobox.depthAxes() != 'z':
            self.clearCustomAnnot()
            return
        
        if ID > 0 and self.customAnnotButton in self.customAnnotDict:
            button = self.customAnnotButton
            if self.toggleCustomAnnotation(button, ID):
                if self.plotReadsColumns([button.name]):
                    self.plotMeasurements()
        
        if getattr(self.posData(), 'rp', None) is None:
            # Position still loading (e.g., predictions run by
            # `initIDsNavigation`): `updateImages` draws the annotations
            # once its objects are computed (`get_data`)
            return
        buttons = self.customAnnotButtonsToDraw()
        for button, annot in self.customAnnotDict.items():
            if button not in buttons:
                self.clearScatterPlotCustomAnnotButton(button)
                continue
            IDs = self.customAnnotFrameIDs(annot['state']['name'])
            self.drawCustomAnnotationIDs(button, IDs)
        if buttons:
            return buttons[0]
    
    def runCustomAnnotPredictor(
            self, button, posData=None, IDs=None, skipIDs=(), force=False
        ):
        """Same as `runTimepointPredictor` for a custom annotation: annotate 
        the entire cell history after boolean consolidation.
        
        If `IDs` is not None, only these cells are predicted. Cells in 
        `skipIDs` keep their current annotations. Unless `force` is True, 
        require `autoApply` and skip cells already predicted for the same 
        position path and annotation. Returns False if the prediction 
        failed."""
        if not self.data or button not in self.customAnnotDict:
            return False
        state = self.customAnnotDict[button]['state']
        if not force and not state.get('autoApply', False):
            return True
        if not self.customAnnotHasPredictor(button):
            return True
        if posData is None:
            posData = self.posData()
        acdc_df = self.timepointsDf(posData)
        if acdc_df is None:
            return True
        name = state['name']
        predictionKey = self.customAnnotPredictionKey(posData, name)
        alreadyPredicted = self.customAnnotAutoPredictions[predictionKey]
        allIDs = {
            int(ID) for ID in acdc_df.index.get_level_values('Cell_ID').unique()
        }
        IDs = allIDs if IDs is None else allIDs.intersection(IDs)
        IDs -= set(skipIDs)
        if not force:
            IDs -= alreadyPredicted
        if not IDs:
            return True
        try:
            # All cells are passed since predictors can use other cells
            # Only the analyzed frames (see `runTimepointPredictor`)
            predicted = CustomAnnotationPredictor.predict(
                self.analysisDf(acdc_df, posData), state['predictor'],
                SizeT=self.analysisSizeT(posData), IDs=IDs
            )
        except Exception as err:
            if not force:
                # Do not retry (and warn again) automatically
                alreadyPredicted.update(IDs)
            self.logger.exception('Custom annotation prediction failed')
            self.measurementProcessingWarning(
                'Custom annotation prediction failed', 
                f'Predicting "{name}" failed:\n\n{err}'
            )
            return False
        
        self.replaceCustomAnnotations(name, predicted, IDs, posData)
        self.editedCustomAnnotCells(posData)[name].difference_update(IDs)
        alreadyPredicted.update(IDs)
        
        numFrames = sum(len(frames) for frames in predicted.values())
        self.statusBar().showMessage(
            f'Predicted {numFrames} "{name}" annotation(s) for '
            f'{len(predicted)} cell(s).', 5000
        )
        return True
    
    def reevaluateCustomAnnotations(self, buttons=None, IDs=None, posData=None):
        """Re-run the predictors of the custom annotations `buttons` 
        (default all with a predictor) on cells `IDs` (default all cells), 
        warning if manual edits would be overwritten. Returns False if 
        cancelled."""
        if not self.data:
            return False
        if posData is None:
            posData = self.posData()
        if buttons is None:
            buttons = list(self.customAnnotDict.keys())
        buttons = [
            button for button in buttons
            if button in self.customAnnotDict 
            and self.customAnnotHasPredictor(button)
        ]
        if not buttons:
            self.statusBar().showMessage(
                'No custom annotation with a predictor.', 5000
            )
            return False
        columns = [self.customAnnotDict[button]['state']['name'] for button in buttons]
        self.logger.info(
            f'Re-evaluating custom annotations {", ".join(columns)} of '
            + ('all cells' if IDs is None else f'Cell ID(s) {list(IDs)}')
        )
        return self.evaluatePredictorColumns(columns, IDs, posData)
    
    def reevaluateAllCellsCustomAnnot(self, button=None, checked=False):
        buttons = None if button is None else [button]
        return self.reevaluateCustomAnnotations(buttons=buttons)

    def reevaluateAllCustomAnnotations(self, checked=False):
        return self.reevaluateAllCellsCustomAnnot()

    def reevaluateDisplayedCellCustomAnnotations(self, checked=False):
        return self.reevaluateDisplayedCellCustomAnnot()

    def reevaluateDisplayedCellCustomAnnot(self, button=None, checked=False):
        if not self.data:
            return False
        if self.selectedID is None:
            self.statusBar().showMessage('Select a cell first.', 5000)
            return False
        buttons = None if button is None else [button]
        return self.reevaluateCustomAnnotations(
            buttons=buttons, IDs=[int(self.selectedID)]
        )
    
    def clearCustomAnnotations(self, button, checked=False):
        """Remove all the annotations of `button` in the current position."""
        if not self.data or button not in self.customAnnotDict:
            return
        acdc_df = self.timepointsDf()
        if acdc_df is None:
            return
        name = self.customAnnotDict[button]['state']['name']
        self.logger.info(
            f'Clearing custom annotation "{name}" of all cells'
        )
        IDs = set(acdc_df.index.get_level_values('Cell_ID'))
        self.replaceCustomAnnotations(name, {}, IDs)
        self.editedCustomAnnotCells()[name] = set()
        self.refreshCustomAnnotations([name])
    
    def ensureCustomAnnotPredictions(self, posData=None):
        """Evaluate the automatic predictors of the custom annotations on 
        the displayed cell if not predicted yet."""
        self.ensurePredictorPredictions(posData)
    
    def setCustomAnnotations(self, states):
        """Replace the custom annotations with `states` (e.g., from a
        configuration). Requires loaded data.

        Adding them does not re-evaluate phases (the tables keep their
        saved phases, `bulkSettingCustomAnnotations`); the automatic
        predictions of the displayed cell, run afterwards, do when they
        change an annotation that affects phases."""
        self.setUncheckedAllCustomAnnotButtons()
        self.reinitCustomAnnot()
        if not self.data or not states:
            return
        self.readSavedCustomAnnot()
        with self.deferPredictorPredictions(), \
                self.bulkSettingCustomAnnotations():
            for savedState in states:
                symbolColor = pg.mkColor(tuple(savedState['symbolColor']))
                state = {
                    'type': 'Single time-point', 'symbol': '', 'shortcut': '',
                    'description': '', 'keepActive': True, 'isHideChecked': True,
                    **savedState, 'symbolColor': symbolColor
                }
                button = self.addCustomAnnotationItems(
                    custom_annot_symbol(state), symbolColor, state['shortcut'],
                    self.customAnnotToolTip(state), state['name'],
                    state['keepActive'], state['isHideChecked'], state
                )
                self.onCustomAnnotationChanged(button)
        self.saveCustomAnnot(only_temp=True)
    
    def hoveredMeasurementsPlotFrame(self):
        """Frame index under the mouse cursor in the measurements plot or 
        None if the cursor is not over the plot."""
        plot = self.measurementsPlot
        if not self.data or not plot.isVisible():
            return
        pos = plot.mapFromGlobal(QCursor.pos())
        if not plot.rect().contains(pos):
            return
        viewPos = plot.getViewBox().mapSceneToView(plot.mapToScene(pos))
        return self.clipFrame(int(round(viewPos.x())) - 1)
    
    def timepointActionTriggered(self, uid, checked=False):
        """Add a time point of the identifier `uid` for the selected cell.

        Its shortcut adds it at the frame under the mouse cursor in the
        measurements plot (nothing when the plot is not hovered). A click on
        its toolbar button (or any other trigger) adds it at the current
        frame. The shortcut is recognised by the action
        (`_ShortcutAwareAction`), not by the cursor position, which may be
        anywhere when the button is clicked (e.g., toolbar extension).
        """
        if not self.data:
            self.statusBar().showMessage('Load data to add time points.', 5000)
            return
        identifier = self.timepointIdentifier(uid)
        action = self.timepointActions.get(uid)
        if getattr(action, 'triggeredByShortcut', False):
            frame_i = self.hoveredMeasurementsPlotFrame()
            if frame_i is None:
                self.statusBar().showMessage(
                    f'Hover over the measurements plot to add a '
                    f'"{identifier.name}" time point with its shortcut, or '
                    'click its button to add it at the current frame.', 5000
                )
                return
        else:
            frame_i = self.frame_i
        self.addTimepoint(uid, frame_i)
    
    def addTimepoint(self, uid, frame_i, ID=None):
        identifier = self.timepointIdentifier(uid)
        if not self.data or frame_i is None:
            self.statusBar().showMessage('Load data to add time points.', 5000)
            return False
        if self.measurementsModeCombobox.currentIndex() == 1:
            self.statusBar().showMessage(
                'Switch to "Time course" mode to add time points.', 5000
            )
            return False
        if ID is None:
            ID = self.selectedID
        if ID is None:
            self.statusBar().showMessage('Select a cell to add time points.', 5000)
            return False
        frame_i = self.clipFrame(frame_i)
        cellTimepoints = self.cellTimepoints(uid, ID)
        if frame_i in cellTimepoints:
            self.statusBar().showMessage(
                f'"{identifier.name}" is already at frame n. {frame_i+1} '
                f'for Cell ID {ID}.', 5000
            )
            return False
        if not self.cellHasRow(ID, frame_i):
            self.statusBar().showMessage(
                f'Cell ID {ID} is not in the measurements table at frame n. '
                f'{frame_i+1}.', 5000
            )
            return False
        self.logger.info(
            f'Adding time point "{identifier.name}" at frame n. {frame_i+1} '
            f'for Cell ID {ID}'
        )
        self.replaceTimepoints(uid, {ID: cellTimepoints + [frame_i]}, [ID])
        self.markTimepointEdited(uid, ID)
        self.refreshPlotForColumns([self.timepointColumn(identifier)])
        self.statusBar().showMessage(
            f'Added "{identifier.name}" at frame n. {frame_i+1} for '
            f'Cell ID {ID}.', 5000
        )
        return True
    
    def removeTimepoint(self, uid, ID, frame_i):
        frames = [f for f in self.cellTimepoints(uid, ID) if f != frame_i]
        self.replaceTimepoints(uid, {ID: frames}, [ID])
    
    def timepointLineRemoveRequested(self, line):
        self.logger.info(
            f'Removing time point "{self.timepointIdentifier(line.uid).name}" '
            f'at frame n. {line.frame_i+1} of Cell ID {line.ID}'
        )
        self.removeTimepoint(line.uid, line.ID, line.frame_i)
        self.markTimepointEdited(line.uid, line.ID)
        self.refreshPlotForColumns(
            [self.timepointColumn(self.timepointIdentifier(line.uid))]
        )

    def timepointLineMoved(self, line):
        new_frame_i = self.clipFrame(int(round(line.value())) - 1)
        editedColumns = []
        # A time point can only sit on a frame where the cell has a row
        if (new_frame_i != line.frame_i
                and self.cellHasRow(line.ID, new_frame_i)):
            self.logger.info(
                'Moving time point '
                f'"{self.timepointIdentifier(line.uid).name}" of Cell ID '
                f'{line.ID} from frame n. {line.frame_i+1} to '
                f'{new_frame_i+1}'
            )
            self.removeTimepoint(line.uid, line.ID, line.frame_i)
            frames = self.cellTimepoints(line.uid, line.ID)
            self.replaceTimepoints(
                line.uid, {line.ID: frames + [new_frame_i]}, [line.ID]
            )
            self.markTimepointEdited(line.uid, line.ID)
            editedColumns.append(
                self.timepointColumn(self.timepointIdentifier(line.uid))
            )
        # Rebuild the lines after the drag event has been fully handled
        QTimer.singleShot(
            0, partial(self.refreshPlotForColumns, editedColumns)
        )
    
    def updateTimepointLines(self):
        for line in self.timepointLines:
            self.measurementsPlot.removeItem(line)
        self.timepointLines = []
        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        if not self.data or isComparison or self.selectedID is None:
            return
        for identifier in self.timepointIdentifiers:
            frames = self.cellTimepoints(identifier.uid, self.selectedID)
            for frame_i in frames:
                line = _TimepointLine(identifier, self.selectedID, frame_i)
                line.sigPositionChangeFinished.connect(self.timepointLineMoved)
                line.sigRemoveRequested.connect(
                    self.timepointLineRemoveRequested
                )
                self.measurementsPlot.addItem(line, ignoreBounds=True)
                self.timepointLines.append(line)

    def plotReadsColumns(self, columns):
        """True if one of the table `columns` is plotted or is read by a
        conditional formatting condition of a plotted column."""
        columns = set(columns)
        if not columns:
            return False
        plotted = set(self.checkedColumns()) | set(self.measurementsAxes.values())
        if plotted & columns:
            return True
        Dependencies = timepoint_predictors.PredictorDependencies
        for formatted in plotted:
            for rule in self.conditionalFormatting.get(formatted, ()):
                inputs = Dependencies.inputs({
                    'type': timepoint_predictors.PREDICTOR_EQUATION,
                    'equation': rule.get('condition', '')
                })
                if columns.intersection(inputs):
                    return True
        return False

    def refreshPlotForColumns(self, columns):
        """After editing the table `columns` (time points or annotations),
        replot if the plot or its conditional formatting reads one of them,
        otherwise only redraw the time point lines."""
        if self.data and self.plotReadsColumns(columns):
            self.plotMeasurements()
        else:
            self.updateTimepointLines()

    def plotMeasurements(self, *args):
        for item in self.measurementsPlot.listDataItems():
            self.measurementsPlot.removeItem(item)
        self.measurementsLegend.clear()
        self.updateCurrentFrameLine()
        self.updateColumnsIcons()

        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        self.currentFrameLine.setVisible(not isComparison)
        self.updateTimepointLines()
        self.updatePhaseBand()
        self.updateUnanalyzedRegions()
        if not self.data:
            self.measurementsPlot.setTitle('No data loaded')
            return
        if isComparison:
            self.plotMeasurementsComparison()
            return

        self.measurementsPlot.setLabel('bottom', 'Frame n.')
        isNormalized = self.normalizeCheckbox.isChecked()
        self.measurementsPlot.setLabel(
            'left', 'Normalized value' if isNormalized else 'Value'
        )
        
        posData = self.posData()
        # Frame range 1..SizeT on the first time course plot of a position
        # (or after comparison mode), kept while nothing can be autoscaled
        # (no column plotted or no values for the selected cell)
        xRangeKey = (self.pos_i, posData.SizeT)
        isNewXRange = getattr(self, '_timecourseXRangeKey', None) != xRangeKey
        if isNewXRange:
            self.measurementsPlot.setXRange(
                1, max(posData.SizeT, 2), padding=0.02
            )
            self._timecourseXRangeKey = xRangeKey

        df_ID = self.getMeasurementsDf()
        columns = self.checkedColumns()
        if self.selectedID is None:
            self.measurementsPlot.setTitle('')
        else:
            self.measurementsPlot.setTitle(f'Cell ID {self.selectedID}')

        # Autoscale x and y (`autoscaleMeasurementsPlot`, as the plot's "A"
        # button) when the plotted curves change (column plotted or
        # removed, normalization toggled) or the frame range was reset; the
        # plot then follows the data until the user zooms or pans by hand.
        # The key is stored only once there was data to fit (cell with data).
        rangeKey = (frozenset(columns), isNormalized)
        if isNewXRange or not columns:
            self._timecourseRangeKey = None
        autoscale = getattr(self, '_timecourseRangeKey', None) != rangeKey

        if df_ID is None or not columns:
            return

        plottedValues = []

        frames_n = df_ID.index.get_level_values(0).to_numpy() + 1
        phases = self.phaseValues(df_ID)
        for c, col in enumerate(columns):
            if col not in df_ID.columns:
                continue
            values = pd.to_numeric(df_ID[col], errors='coerce').to_numpy()
            rawValues = values.astype(float)
            values = rawValues
            if isNormalized:
                values = self._normalize(values)
            color = self.getSignalColor(col)
            rules = self.conditionalFormatting.get(col, [])
            matched = self.cellConditionalFormatting(df_ID, col, rules)
            pointData = [
                {
                    'frame_i': int(frame_n-1),
                    'Cell_ID': self.selectedID,
                    'values': {col: (raw, value)},
                    'normalized': isNormalized,
                    'formatting': (
                        rules[m]['condition'] if m >= 0 else None
                    ),
                    'phase': phase,
                }
                for frame_n, raw, value, m, phase
                in zip(frames_n, rawValues, values, matched, phases)
            ]
            brushes = [
                pg.mkBrush(
                    formatting_color(rules[m]['color'], color) if m >= 0
                    else color
                )
                for m in matched
            ]
            # Segment i (point i to i+1) takes the format of point i. The
            # main curve draws only unformatted segments; each rule draws
            # its segments as separate pairs below the main curve's points.
            finite = np.isfinite(values)
            segments = finite[:-1] & finite[1:]
            connect = np.append(segments & (matched[:-1] < 0), False)
            for r, rule in enumerate(rules):
                idx = np.flatnonzero(segments & (matched[:-1] == r))
                if not len(idx):
                    continue
                pairs = np.column_stack((idx, idx + 1)).ravel()
                segmentsItem = self.measurementsPlot.plot(
                    frames_n[pairs], values[pairs], connect='pairs',
                    pen=pg.mkPen(
                        color=formatting_color(rule['color'], color), width=2
                    )
                )
                segmentsItem.setZValue(-1)
            item = self.measurementsPlot.plot(
                frames_n, values, name=col,
                pen=pg.mkPen(color=color, width=2), connect=connect,
                symbol='o', symbolSize=5, symbolBrush=brushes, symbolPen=None,
                data=pointData
            )
            self.connectSignalPoints(item)
            plottedValues.append(values)

        hasValues = any(np.isfinite(vals).any() for vals in plottedValues)
        if autoscale and hasValues:
            self.autoscaleMeasurementsPlot()
            self._timecourseRangeKey = rangeKey

    def autoscaleMeasurementsPlot(self):
        """Autoscale the measurements plot exactly like its "A" button
        (`PlotItem.autoBtnClicked`): enable pyqtgraph's auto range on both
        axes, so x and y fit the plotted data (time course or Measurement
        vs Measurement) and keep following it (e.g. another cell or a table
        edit) until the user zooms or pans by hand.

        The y auto range uses only the data in the visible x range
        (`ViewBox.setAutoVisible`), so it still follows the data when the
        user pans or zooms only the x axis. pyqtgraph applies the range on
        the next paint (tests call ``viewBox.updateAutoRange()``).

        Every automatic rescaling of this plot must go through this method
        (column plotted or removed, normalization toggled, ``Rescale plot``,
        Measurement vs Measurement axes changed).
        """
        viewBox = self.measurementsPlot.getViewBox()
        viewBox.setAutoVisible(y=True)
        self.measurementsPlot.enableAutoRange()

    def measurementsModeChanged(self, *args):
        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        self.logger.info(
            f'Plot mode: "{self.measurementsModeCombobox.currentText()}"'
        )
        if self._measurementsMode == 0:
            self._timeCourseColumns = set(self.checkedColumns())
        self._measurementsMode = self.measurementsModeCombobox.currentIndex()
        columns = (
            set(self.measurementsAxes.values()) if isComparison
            else self._timeCourseColumns
        )
        self.columnsModel.blockSignals(True)
        for item in self.columnsModel.measurementItems():
            item.setCheckState(
                Qt.Checked if item.text() in columns else Qt.Unchecked
            )
        self.columnsModel.blockSignals(False)
        self.columnsListWidget.setToolTip(
            'Click two columns to plot: first X, then Y.\n'
            'Click a selected column again to free its axis.'
            if isComparison else
            'Click a column to plot/unplot it. Right-click on a\n'
            'column to change its color or conditional formatting.'
        )
        self.plotMeasurements()

    def measurementColumnChanged(self, item):
        if getattr(self, '_isUpdatingColumnsList', False):
            return
        if not item.isCheckable():
            return
        if self.measurementsModeCombobox.currentIndex() == 1:
            col = item.text()
            if item.checkState() != Qt.Checked:
                for axis, measurement in self.measurementsAxes.items():
                    if measurement == col:
                        self.measurementsAxes[axis] = ''
                        self.logger.info(f'Removing "{col}" from the {axis} axis')
            elif col not in self.measurementsAxes.values():
                axis = next(
                    (axis for axis, measurement in self.measurementsAxes.items()
                     if not measurement), None
                )
                if axis is None:
                    self.columnsModel.blockSignals(True)
                    item.setCheckState(Qt.Unchecked)
                    self.columnsModel.blockSignals(False)
                    self.statusBar().showMessage(
                        'Two measurements are already selected. Uncheck X or Y '
                        'before selecting another measurement.', 5000
                    )
                else:
                    self.measurementsAxes[axis] = col
                    self.logger.info(f'Plotting "{col}" on the {axis} axis')
        elif item.checkState() == Qt.Checked:
            self.logger.info(f'Plotting "{item.text()}"')
        else:
            self.logger.info(f'Removing "{item.text()}" from the plot')
        self.plotMeasurements()

    def plotMeasurementsComparison(self):
        xCol = self.measurementsAxes['X']
        yCol = self.measurementsAxes['Y']
        isNormalized = self.normalizeCheckbox.isChecked()
        suffix = ' (normalized)' if isNormalized else ''
        self.measurementsPlot.setLabel('bottom', f'{xCol}{suffix}')
        self.measurementsPlot.setLabel('left', f'{yCol}{suffix}')
        self.measurementsPlot.setTitle('All cells, all frames (current position)')
        self.autoscaleMeasurementsPlot()
        self._timecourseXRangeKey = None
        if self.acdc_df is None:
            self.measurementsPlot.setTitle('No measurements available')
            return
        # A column removed from the table (e.g., time point identifier) is
        # an unset axis until the columns list is updated
        if (not xCol or not yCol or xCol not in self.acdc_df.columns
                or yCol not in self.acdc_df.columns):
            self.measurementsPlot.setTitle('Select X and Y measurements from the column list')
            return

        df = self.acdc_df
        frames = df.index.get_level_values('frame_i').to_numpy()
        IDs = df.index.get_level_values('Cell_ID').to_numpy()
        inRange = (frames >= 0) & (frames < self.posData().SizeT) & (IDs > 0)
        df = df.loc[inRange]
        frames, IDs = frames[inRange], IDs[inRange]
        rawX = pd.to_numeric(df[xCol], errors='coerce').to_numpy(dtype=float)
        rawY = pd.to_numeric(df[yCol], errors='coerce').to_numpy(dtype=float)
        phases = self.phaseValues(df)
        finite = np.isfinite(rawX) & np.isfinite(rawY)
        frames, IDs = frames[finite], IDs[finite]
        rawX, rawY = rawX[finite], rawY[finite]
        phases = phases[finite]
        if not len(rawX):
            self.measurementsPlot.setTitle('No finite paired measurements')
            return
        x = self._normalize(rawX) if isNormalized else rawX
        y = self._normalize(rawY) if isNormalized else rawY
        pointData = [
            {
                'frame_i': int(frame_i),
                'Cell_ID': int(ID),
                'values': {xCol: (rx, vx), yCol: (ry, vy)},
                'normalized': isNormalized,
                'phase': phase,
            }
            for frame_i, ID, rx, ry, vx, vy, phase
            in zip(frames, IDs, rawX, rawY, x, y, phases)
        ]
        brush = self.getSignalColor(yCol)
        if self.isShowingPhases():
            brush = self.comparisonPhaseBrushes(phases)
        item = self.measurementsPlot.plot(
            x, y, pen=None, symbol='o', symbolSize=7,
            symbolBrush=brush, symbolPen=None, data=pointData
        )
        self.connectSignalPoints(item)

    def isShowingPhases(self):
        return self.showPhasesCheckbox.isChecked() and self.hasPhaseColumn()

    def normalizeToggled(self, checked):
        self.logger.info(f'Normalize measurements {"on" if checked else "off"}')

    def showPhasesToggled(self, checked):
        self.logger.info(f'Show phases {"on" if checked else "off"}')
        self._set_setting_value(
            'measurements_viewer_show_phases', 'Yes' if checked else 'No'
        )
        self.plotMeasurements()

    def comparisonPhaseBrushes(self, phases):
        """Per-point brushes by phase; adds a legend entry per shown phase."""
        colors = self.phaseColorMap()
        unassigned = pg.mkBrush(150, 150, 150, 120)
        brushes = {name: pg.mkBrush(color) for name, color in colors.items()}
        present = set(phases)
        for name, brush in brushes.items():
            if name in present:
                self.measurementsLegend.addItem(
                    pg.ScatterPlotItem(symbol='o', brush=brush, pen=None),
                    name
                )
        if '' in present:
            self.measurementsLegend.addItem(
                pg.ScatterPlotItem(symbol='o', brush=unassigned, pen=None),
                'No phase'
            )
        return [brushes.get(phase, unassigned) for phase in phases]

    def syncPhaseBandAxisWidth(self):
        width = self.measurementsPlot.getAxis('left').width()
        self.phaseBandPlot.getAxis('left').setWidth(int(width))

    def updatePhaseBand(self):
        """Draw the selected cell's phases as colored bars below the time
        course. Hidden in comparison mode, without phases or when disabled."""
        for item in self.phaseBandItems:
            self.phaseBandPlot.removeItem(item)
        self.phaseBandItems = []
        self.phaseBandPlot.setToolTip('')
        self.showPhasesCheckbox.setEnabled(self.hasPhaseColumn())
        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        df_ID = None
        if self.data and not isComparison and self.isShowingPhases():
            df_ID = self.getMeasurementsDf()
        runs = []
        if df_ID is not None:
            frames_n = df_ID.index.get_level_values(0).to_numpy() + 1
            runs = self.phaseRuns(frames_n, self.phaseValues(df_ID))
        self.phaseBandPlot.setVisible(bool(runs))
        if not runs:
            return
        colors = self.phaseColorMap()
        bars = pg.BarGraphItem(
            x0=[start - 0.5 for start, _, _ in runs],
            x1=[end + 0.5 for _, end, _ in runs],
            y0=0, height=1, pen=None,
            brushes=[pg.mkBrush(colors[phase]) for _, _, phase in runs]
        )
        self.phaseBandPlot.addItem(bars)
        self.phaseBandItems.append(bars)
        for start, end, phase in runs:
            color = colors[phase]
            textColor = 'k' if color.lightness() > 140 else 'w'
            text = pg.TextItem(phase, color=textColor, anchor=(0.5, 0.5))
            text.setPos((start + end) / 2, 0.5)
            self.phaseBandPlot.addItem(text, ignoreBounds=True)
            self.phaseBandItems.append(text)
        self.phaseBandPlot.setToolTip(qutils.wrap_tooltip('\n'.join(
            f'{phase}: frames {start}-{end}' for start, end, phase in runs
        )))
        self.phaseBandPlot.setYRange(0, 1, padding=0)
        self.syncPhaseBandAxisWidth()

    @staticmethod
    def createUnanalyzedRegion():
        """Gray, non-movable region over the frames ignored by the
        calculations (red hues are reserved for the frame lines). Below the
        data, time point and frame lines; added with ``ignoreBounds=True``
        so that it never changes the auto range."""
        region = pg.LinearRegionItem(
            values=(0, 1), movable=False,
            brush=pg.mkBrush(128, 128, 128, 60), pen=pg.mkPen(None)
        )
        region.setZValue(-10)
        region.setVisible(False)
        return region

    def updateUnanalyzedRegions(self):
        """Shade the frames after the last analyzed frame of the displayed
        position (`analysisLastFrame`) in the time course and phase band
        (frame numbers are 1-based: frame index ``i`` is drawn at
        ``i + 1``). Hidden in comparison mode and when all frames are used.
        """
        regions = (self.unanalyzedRegion, self.phaseBandUnanalyzedRegion)
        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        last = self.analysisLastFrame() if self.data else None
        if last is None or isComparison:
            for region in regions:
                region.setVisible(False)
            return
        SizeT = self.posData().SizeT
        # No tool tip: it would hide the points' hover info (shown with
        # the ViewBox tool tip) and the phase band tool tip on these frames
        for region in regions:
            region.setRegion((last + 1.5, SizeT + 0.5))
            region.setVisible(True)

    def connectSignalPoints(self, item):
        # PlotDataItem does not forward hover options to its scatter item.
        item.scatter.setData(
            x=item.xData, y=item.yData, data=item.opts['data'],
            brush=item.opts['symbolBrush'],
            hoverable=True, tip=self.measurementPointTip,
            hoverSize=11, hoverBrush=pg.mkBrush(255, 100, 100)
        )
        item.scatter.sigClicked.connect(self.measurementPointClicked)

    @staticmethod
    def measurementPointTip(x, y, data):
        lines = [
            f'Cell ID: {data["Cell_ID"]}',
            f'Frame: {data["frame_i"]+1}',
        ]
        for col, (raw, value) in data['values'].items():
            lines.append(f'{col}: {raw:.6g}')
            if data['normalized']:
                lines.append(f'{col} (normalized): {value:.6g}')
        if data.get('phase'):
            lines.append(f'Phase: {data["phase"]}')
        if data.get('formatting'):
            lines.append(f'Formatting: {data["formatting"]}')
        # Long column names, phase names or formatting conditions
        return qutils.wrap_tooltip('\n'.join(lines))

    def measurementPointClicked(self, item, points, event):
        if event.button() != Qt.LeftButton or not len(points):
            return
        data = points[0].data()
        ID, frame_i = data['Cell_ID'], data['frame_i']
        if ID not in self.IDs or not 0 <= frame_i < self.posData().SizeT:
            self.statusBar().showMessage(
                f'Cell ID {ID} at frame {frame_i+1} is not available.', 5000
            )
            return
        # Navigating replots the measurements, deleting the clicked scatter item
        # while pyqtgraph is still dispatching the click to it --> defer
        QTimer.singleShot(
            100, partial(self.navigateToSignalPoint, ID, frame_i)
        )

    def navigateToSignalPoint(self, ID, frame_i):
        self.idScrollBar.setValue(self.IDs.index(ID))
        self.setFrame(frame_i)
        self.zoomToSelectedID(goToZslice=True, force=True)
    
    def getSignalColor(self, col):
        """Get the plotting color of column `col`: the custom color if set
        by the user, otherwise its stable default color
        (`defaultSignalColorIndex`)."""
        color = self.measurementsColors.get(col)
        if color is not None:
            return color
        return signal_color(
            self.defaultSignalColorIndex(col),
            light_background=self.invertBwAction.isChecked()
        )

    def defaultSignalColorIndex(self, col):
        """Index of the default color of column `col` (see `signal_color`).

        A column keeps its color while it is plotted, and gets it back when
        plotted again unless another plotted column took it meanwhile. A
        newly plotted column takes the first color that no other plotted
        column uses. In comparison mode the columns plotted in the time
        course count as plotted, so that the comparison (Y column color)
        does not reassign their colors.
        """
        indices = self._defaultSignalColorIndices
        if self._measurementsMode == 0:
            plotted = set(self.checkedColumns())
        else:
            plotted = self._timeCourseColumns
        used = {
            index for column, index in indices.items()
            if column != col and column in plotted
        }
        index = indices.get(col)
        if index is None or index in used:
            index = next(i for i in range(len(used)+1) if i not in used)
            indices[col] = index
        return index
    
    def updateColumnsIcons(self):
        """Show color swatches or comparison-axis markers beside columns."""
        columns = self.checkedColumns()
        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        self.columnsModel.blockSignals(True)
        for item in self.columnsModel.measurementItems():
            col = item.text()
            item.setCheckable(True)
            if isComparison:
                axes = '/'.join(
                    axis for axis, measurement in self.measurementsAxes.items()
                    if measurement == col
                )
                item.setData(axes, Qt.AccessibleDescriptionRole)
                if not axes:
                    item.setIcon(QIcon())
                    continue
                pixmap = QPixmap(24, 16)
                pixmap.fill(Qt.transparent)
                painter = QPainter(pixmap)
                painter.setPen(self.columnsListWidget.palette().text().color())
                font = self.columnsListWidget.font()
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(pixmap.rect(), Qt.AlignCenter, axes)
                painter.end()
                item.setIcon(QIcon(pixmap))
                continue
            item.setData('', Qt.AccessibleDescriptionRole)
            if col not in columns:
                item.setIcon(QIcon())
                continue
            pixmap = QPixmap(12, 12)
            pixmap.fill(self.getSignalColor(col))
            item.setIcon(QIcon(pixmap))
        self.columnsModel.blockSignals(False)
        # Signals were blocked to avoid itemChanged, so repaint explicitly
        self.columnsListWidget.viewport().update()
    
    def showColumnsContextMenu(self, pos):
        proxy_index = self.columnsListWidget.indexAt(pos)
        if not proxy_index.isValid():
            return
        source_index = self.columnsSearchProxy.mapToSource(proxy_index)
        item = self.columnsModel.itemFromIndex(source_index)
        if not item.isCheckable():
            return
        
        col = item.text()
        menu = QMenu(self.columnsListWidget)

        plotAction = QAction('Plot', menu)
        plotAction.setCheckable(True)
        plotAction.setChecked(item.checkState() == Qt.Checked)
        plotAction.toggled.connect(
            lambda checked, item=item: item.setCheckState(
                Qt.Checked if checked else Qt.Unchecked
            )
        )
        menu.addAction(plotAction)
        if self.measurementsModeCombobox.currentIndex() == 1:
            menu.popup(self.columnsListWidget.viewport().mapToGlobal(pos))
            return
        menu.addSeparator()
        
        changeColorAction = QAction('Change color...', menu)
        changeColorAction.triggered.connect(
            lambda checked=False, col=col: self.changeSignalColor(col)
        )
        menu.addAction(changeColorAction)
        
        resetColorAction = QAction('Reset to default color', menu)
        resetColorAction.setEnabled(col in self.measurementsColors)
        resetColorAction.triggered.connect(
            lambda checked=False, col=col: self.resetSignalColor(col)
        )
        menu.addAction(resetColorAction)
        menu.addSeparator()

        formattingAction = QAction('Conditional formatting...', menu)
        formattingAction.triggered.connect(
            lambda checked=False, col=col: self.editConditionalFormatting(col)
        )
        menu.addAction(formattingAction)

        clearFormattingAction = QAction('Clear conditional formatting', menu)
        clearFormattingAction.setEnabled(col in self.conditionalFormatting)
        clearFormattingAction.triggered.connect(
            lambda checked=False, col=col: self.setConditionalFormatting(
                col, []
            )
        )
        menu.addAction(clearFormattingAction)

        menu.popup(self.columnsListWidget.viewport().mapToGlobal(pos))
    
    def changeSignalColor(self, col):
        initialColor = pg.mkColor(self.getSignalColor(col))
        color = QColorDialog.getColor(
            initialColor, self, f'Select color of "{col}"'
        )
        if not color.isValid():
            return
        
        self.logger.info(f'Changing color of "{col}" to {color.name()}')
        self.measurementsColors[col] = color
        self.plotMeasurements()
    
    def resetSignalColor(self, col):
        self.logger.info(f'Resetting color of "{col}" to the default')
        self.measurementsColors.pop(col, None)
        self.plotMeasurements()

    def cellConditionalFormatting(self, df_ID, col, rules):
        """Rule index per row of the selected cell for column `col` (-1: no
        rule). Errors are reported in the status bar, and the column is then
        plotted without formatting."""
        if not rules:
            return np.full(len(df_ID), -1, dtype=int)
        try:
            return self.evaluateConditionalFormatting(
                df_ID, rules, ID=self.selectedID,
                SizeT=getattr(self.posData(), 'SizeT', None),
                last_frame=self.analysisLastFrame()
            )
        except Exception as err:
            message = f'Conditional formatting of "{col}" failed: {err}'
            if message != getattr(self, '_lastFormattingError', None):
                self.logger.warning(message)
            self._lastFormattingError = message
            self.statusBar().showMessage(message, 10000)
            return np.full(len(df_ID), -1, dtype=int)

    def editFormattingCondition(self, condition):
        posData = self.posData() if self.data else None
        dialog = _FormattingConditionDialog(
            getattr(posData, 'chNames', []),
            getattr(posData, 'SizeZ', 1) > 1, bool(self.isSegm3D),
            acdc_df=self.dialogAnalysisDf(self.acdc_df, posData),
            SizeT=self.dialogAnalysisSizeT(posData),
            condition=condition, parent=self
        )
        dialog.exec_()
        if dialog.cancel:
            return None
        return dialog.equation

    def editConditionalFormatting(self, col):
        dialog = ConditionalFormattingDialog(
            col, self.conditionalFormatting.get(col, []),
            validateCondition=self.validateFormattingCondition,
            editCondition=self.editFormattingCondition, parent=self
        )
        dialog.exec_()
        if dialog.cancel:
            return
        self.setConditionalFormatting(col, dialog.rules)

    def setConditionalFormatting(self, col, rules):
        if rules:
            self.logger.info(
                f'Conditional formatting of "{col}" set: {len(rules)} rule(s)'
            )
            self.conditionalFormatting[col] = copy.deepcopy(rules)
        else:
            self.logger.info(f'Conditional formatting of "{col}" cleared')
            self.conditionalFormatting.pop(col, None)
        self._lastFormattingError = None
        self.plotMeasurements()
    
    def stepID(self, step):
        if not self.IDs:
            return
        idx = int(np.clip(
            self.idScrollBar.value() + step, 0, len(self.IDs)-1
        ))
        self.idScrollBar.setValue(idx)
        # The spinbox arrows already changed its value by +-1
        self.idSpinBox.setValueNoEmit(self.currID())

    def currID(self):
        if not self.IDs:
            return None
        idx = self.idScrollBar.value()
        return self.IDs[idx]
    
    def updateScrollbars(self):
        """Sync the navigation widgets after `posData.frame_i` was changed 
        externally (e.g., by `SearchIDGuiElements.findID`)"""
        frame_i = self.frame_i
        self.navSpinBox.setValueNoEmit(frame_i+1)
        self.resetNavigateFramesScrollbar(frame_i=frame_i)
        self._lab = None
        self._labKey = None
        self.updateCurrentFrameLine()
        self.plotMeasurements()
        self.updateIDnotPresentLabel()
    
    def invertBw(self, checked, update=True):
        # `ColorSchemeGuiElements.invertBw` inverts the text annotations of 
        # both images of guiWin while the viewer has only the left image
        textAnnot = self.textAnnot
        hasLeftTextAnnot = 0 in textAnnot
        if not hasLeftTextAnnot:
            del self.textAnnot
        else:
            textAnnot[1] = _NoOpItem()
        try:
            super().invertBw(checked, update=update and self.isDataLoaded)
        finally:
            self.textAnnot = textAnnot
            textAnnot.pop(1, None)
        
        self.isBwInverted = checked
        background = graphLayoutBkgrColor if checked else darkBkgrColor
        self.measurementsPlot.setBackground(background)
        self.phaseBandPlot.setBackground(background)

    def recolorSignalsForBackground(self, checked=False):
        """Redraw the columns' default colors, which depend on the plot
        background (`signal_color`), after `Invert black/white` toggled."""
        if not self.isDataLoaded or self._isLoadingData:
            return
        self.updateColumnsIcons()
        self.plotMeasurements()

    def highlightSearchedID(self, ID, force=False, greyOthers=True):
        self.highlightedID = ID
        self.highLightIDLayerImg1.clear()
        self.searchedIDitemLeft.clear()
        if ID == 0:
            return
        self.highlightSelectedID()
    
    def goToObjectID(self, ID):
        if ID not in self.IDs:
            self.statusBar().showMessage(
                f'Cell ID {ID} does not exist in the acdc_output table.', 
                3000
            )
            return
        
        idx = self.IDs.index(ID)
        if self.idScrollBar.value() == idx:
            self.setSelectedID(ID)
        else:
            self.idScrollBar.setValue(idx)
    
    def goToLostObjectID(self, ID, color=None):
        self.goToObjectID(ID)

    def goToAcceptedLostObjectID(self, ID):
        self.goToObjectID(ID)
    
    def initLookupTableLab(self):
        self.initLabelsImageItems()
    
    def updateLookuptable(self, *args, **kwargs):
        self.initLabelsImageItems()
    
    def clearCursors(self):
        self.ax1_cursor.setData([], [])
    
    def connectLeftClickButtons(self):
        # The viewer does not have editing tools that need to be reconnected
        pass
    
    def disconnectLeftClickButtons(self):
        pass
    
    def uncheckLeftClickButtons(self, sender):
        for button in self.leftClickButtons:
            if button == sender:
                continue
            if button.isChecked():
                button.setChecked(False)
    
    def drawPointsLayers(self, **kwargs):
        # Points layers are a main GUI-only feature
        pass
    
    def addPointsLayersToVolumeViewer(self, *args, **kwargs):
        pass
    
    def resetCursor(self):
        while self.app.overrideCursor() is not None:
            self.app.restoreOverrideCursor()
    
    def raiseOnDemandAx1Items(self):
        """Keep the ax1 items created on demand (object tracks, scale bar,
        timestamp) above the overlay channel layers, which are re-added
        (z-value 0) on every load, at the same z-value as the ruler."""
        items = []
        tracksPlotItem = getattr(self, 'tracksPlotItems', {}).get(0)
        if tracksPlotItem is not None:
            items.append(tracksPlotItem)
        scaleBar = getattr(self, 'scaleBar', None)
        if scaleBar is not None:
            items.extend((scaleBar.plotItem, scaleBar.labelItem))
        timestamp = getattr(self, 'timestamp', None)
        if timestamp is not None:
            items.append(timestamp)
        for item in items:
            item.setZValue(0.9)

    def annotateAllObjectTracks(self, settings=None):
        super().annotateAllObjectTracks(settings=settings)
        self.raiseOnDemandAx1Items()

    def addTimestamp(self, checked):
        super().addTimestamp(checked)
        self.raiseOnDemandAx1Items()

    def addScaleBar(self, checked):
        super().addScaleBar(checked)
        self.raiseOnDemandAx1Items()
        # Rescale once the scale bar items and the dialog are settled
        QTimer.singleShot(0, self.autoRange)

    def autoRange(self):
        self.ax1.autoRange()
        if self.followIDcheckbox.isChecked():
            # Keep the view on the selected cell (e.g., after resizing the 
            # bottom layout)
            self.zoomToSelectedID()
    
    def clearComboBoxFocus(self, *args):
        self.setFocus()
    
    def isScaleBarOrTimestampHighlighted(self):
        for name in ('scaleBar', 'timestamp'):
            item = getattr(self, name, None)
            if item is not None and item.isHighlighted():
                return item
    
    def resetCountKeyPress(self):
        self.doubleKeyTimeElapsed = True
        self.countKeyPress = 0
    
    def isHoveringRoiHandle(self, roi):
        # Same as `gui.guiWin.gui_isHoveringRoiHandle`
        if roi is None:
            return False
        try:
            handles = roi.getHandles()
        except Exception:
            return False

        for handle in handles:
            try:
                if handle.currentPen == handle.hoverPen:
                    return True
            except Exception:
                continue
        return False
    
    def gui_hoverEventImg1(self, event):
        if not self.isDataLoaded:
            return
        
        if event.isExit():
            self.xHoverImg, self.yHoverImg = None, None
            self.wcLabel.setText('')
            self.resetCursor()
            if self.slideshowWin is not None:
                self.slideshowWin.setMirroredCursorPos(None, None)
            return
        
        x, y = event.pos()
        self.xHoverImg, self.yHoverImg = x, y
        
        if self.slideshowWin is not None:
            self.slideshowWin.setMirroredCursorPos(x, y)
        
        if self.rulerButton.isChecked() and self.tempSegmentON:
            self.drawTempRulerLine(event)
        
        Y, X = self.currentLab2D.shape
        xdata, ydata = int(x), int(y)
        if not myutils.is_in_bounds(xdata, ydata, X, Y):
            self.wcLabel.setText('')
            return
        
        ID = self.currentLab2D[ydata, xdata]
        self.updatePropsWidget(ID, fromHover=True)
        activeToolButton = self.getActiveToolButton()
        txt = self.hoverValuesFormatted(xdata, ydata, activeToolButton, True)
        self.checkHighlightScaleBar(x, y, activeToolButton)
        self.checkHighlightTimestamp(x, y, activeToolButton)
        self.wcLabel.setText(txt)
    
    def gui_mousePressEventImg1(self, event):
        if not self.isDataLoaded:
            pg.ImageItem.mousePressEvent(self.img1, event)
            return
        
        if event.button() not in STANDARD_MOUSE_BUTTONS:
            # As in guiWin: only bound buttons are forwarded (the window's
            # `QMainWindow.mousePressEvent` does not accept scene events)
            if event.button() in self.mouseBindings.values():
                self.mousePressEvent(event)
            else:
                event.ignore()
            return

        left_click = event.button() == Qt.MouseButton.LeftButton
        right_click = event.button() == Qt.MouseButton.RightButton
        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)

        isZoomRectActive = self.zoomRectButton.isChecked()
        if isZoomRectActive and self.isHoveringRoiHandle(self.zoomRectItem):
            event.ignore()
            return
        
        if left_click and self.rulerButton.isChecked():
            if not self.tempSegmentON:
                self.ax1_rulerAnchorsItem.setData([xdata], [ydata])
                self.tempSegmentON = True
            else:
                modifiers = QGuiApplication.keyboardModifiers()
                ctrl = modifiers == Qt.ControlModifier
                self.tempSegmentON = False
                xxRA, yyRA = self.ax1_rulerAnchorsItem.getData()
                x0, y0 = xxRA[0], yyRA[0]
                if ctrl:
                    x1, y1 = transformation.snap_xy_to_closest_angle(
                        x0, y0, xdata, ydata
                    )
                else:
                    x1, y1 = xdata, ydata
                lengthText = self.getRulerLengthText()
                self.ax1_rulerPlotItem.setData(
                    [x0, x1], [y0, y1], lengthText=lengthText
                )
                self.ax1_rulerAnchorsItem.setData([x0, x1], [y0, y1])
            return
        
        if isZoomRectActive and (left_click or right_click):
            if left_click:
                self.zoomRectItem.setPos((xdata, ydata))
                self.isMouseDragImg1 = True
            else:
                try:
                    xRange, yRange = self.zoomRectItem.getLastRange()
                    self.ax1.setRange(xRange=xRange, yRange=yRange, padding=0)
                except Exception as err:
                    QTimer.singleShot(100, self.autoRange)
            return
        
        highlightedItem = self.isScaleBarOrTimestampHighlighted()
        if left_click and highlightedItem is not None:
            highlightedItem.mousePressed(x, y)
            return
        
        if right_click and self.customAnnotButton is not None:
            self.annotateClickedObject(xdata, ydata)
            return
        
        if right_click:
            screenPos = event.screenPos()
            self.imgGradShowContextMenu(screenPos.x(), screenPos.y())
            return
        
        pg.ImageItem.mousePressEvent(self.img1, event)
        event.ignore()
    
    def imgGradShowContextMenu(self, x, y):
        # Same as `gui.guiWin.gui_imgGradShowContextMenu`
        highlightedItem = self.isScaleBarOrTimestampHighlighted()
        if highlightedItem is not None:
            highlightedItem.showContextMenu(x, y)
            return
        self.imgGrad.gradient.menu.popup(QPoint(int(x), int(y)))
    
    def gui_mouseDragEventImg1(self, event):
        if not self.isDataLoaded:
            pg.ImageItem.mouseMoveEvent(self.img1, event)
            return
        
        x, y = event.pos().x(), event.pos().y()
        
        if hasattr(self, 'scaleBar'):
            if self.scaleBarDialog is not None:
                self.scaleBarDialog.locCombobox.setCurrentText('Custom')
            if self.scaleBar.isHighlighted() and self.scaleBar.clicked:
                self.scaleBar.setLocationProperty('custom')
                self.scaleBar.move(x, y)
                return
        
        if hasattr(self, 'timestamp'):
            if self.timestampDialog is not None:
                self.timestampDialog.locCombobox.setCurrentText('Custom')
            if self.timestamp.isHighlighted() and self.timestamp.clicked:
                self.timestamp.setLocationProperty('custom')
                self.timestamp.move(x, y)
                return
        
        if self.isMouseDragImg1 and self.zoomRectButton.isChecked():
            xdata, ydata = int(x), int(y)
            x0, y0 = self.zoomRectItem.pos()
            w, h = (xdata-x0), (ydata-y0)
            self.zoomRectItem.setSize((w, h))
            return
        
        pg.ImageItem.mouseMoveEvent(self.img1, event)
    
    def gui_mouseReleaseEventImg1(self, event):
        if not self.isDataLoaded:
            pg.ImageItem.mouseReleaseEvent(self.img1, event)
            return
        
        if hasattr(self, 'scaleBar'):
            if self.scaleBar.isHighlighted() and self.scaleBar.clicked:
                self.scaleBar.clicked = False
                return
        
        if hasattr(self, 'timestamp'):
            if self.timestamp.isHighlighted() and self.timestamp.clicked:
                self.timestamp.clicked = False
                return
        
        if self.isMouseDragImg1 and self.zoomRectButton.isChecked():
            self.isMouseDragImg1 = False
            self.zoomRectDone()
            return
        
        pg.ImageItem.mouseReleaseEvent(self.img1, event)
    
    def gui_mouseClickEventImg1(self, event):
        """Select the clicked cell (pyqtgraph sends click events only if 
        the mouse was not dragged, hence panning does not select)."""
        if not self.isDataLoaded:
            return
        
        if event.button() != Qt.MouseButton.LeftButton:
            return
        
        isToolActive = (
            self.rulerButton.isChecked() 
            or self.zoomRectButton.isChecked()
            or self.isScaleBarOrTimestampHighlighted() is not None
        )
        if isToolActive:
            return
        
        x, y = event.pos().x(), event.pos().y()
        lab2D = self.currentLab2D
        Y, X = lab2D.shape[-2:]
        xdata, ydata = int(x), int(y)
        if not (0 <= xdata < X and 0 <= ydata < Y):
            return
        
        ID = int(lab2D[ydata, xdata])
        if ID == 0:
            return
        
        event.accept()
        self.goToObjectID(ID)
    
    def mousePressEvent(self, event):
        # Custom mouse bindings (e.g., "Next"/"Previous" on mouse side 
        # buttons) as in guiWin
        for name, button in self.mouseBindings.items():
            if event.button() != button:
                continue
            widget = self.widgetsWithShortcut.get(name)
            if widget is None:
                continue
            if hasattr(widget, 'click'):
                widget.click()
            else:
                widget.trigger()
            # Handled: a forwarded scene event must not propagate to the
            # window again (triggering the action twice)
            event.accept()
            return
        if not isinstance(event, QMouseEvent):
            # Scene events forwarded by `gui_mousePressEventImg1`
            event.ignore()
            return
        super().mousePressEvent(event)
    
    def keyPressEvent(self, event):
        if not self.data:
            super().keyPressEvent(event)
            return
        key = event.key()
        modifiers = event.modifiers()
        isNoModifier = modifiers == Qt.NoModifier
        if self.overlayButton.isChecked() and isNoModifier:
            isButtonClicked = self.checkOverlayToolbuttonClicked(event)
            if isButtonClicked:
                return
        
        if key == self.zoomOutKeyValue and isNoModifier:
            doubleKeyPress = False
            if self.countKeyPress == 0:
                self.countKeyPress = 1
                self.doubleKeyTimeElapsed = False
                QTimer.singleShot(400, self.resetCountKeyPress)
            elif self.countKeyPress == 1 and not self.doubleKeyTimeElapsed:
                doubleKeyPress = True
                self.countKeyPress = 0
                
            if self.measurementsPlot.underMouse():
                if not doubleKeyPress:
                    self.autoscaleMeasurementsPlot()
                else:
                    posData = self.posData()
                    max_frame_index = posData.SizeT
                    self.measurementsPlot.setXRange(0, max_frame_index)
                return
            
            if not self.img1.isUnderMouse():
                return
            # Over the image: press --> zoom to objects,
            # double press --> zoom out
            if not doubleKeyPress:
                self.zoomToCells(enforce=True)
            else:
                self.ax1.autoRange()
                
        elif key == Qt.Key_Space and isNoModifier:
            # Same as guiWin: double press --> toggle "Do not annotate"
            if self.countKeyPress == 0:
                # Single press --> wait that it's not double press
                self.isKeyDoublePress = False
                self.countKeyPress = 1
                self.doubleKeyTimeElapsed = False
                QTimer.singleShot(300, self.doubleKeySpacebarTimerCallback)
            elif self.countKeyPress == 1 and not self.doubleKeyTimeElapsed:
                self.isKeyDoublePress = True
                # Double press --> toggle draw nothing
                self.onDoubleSpaceBar()
                self.countKeyPress = 0
        elif key == Qt.Key_Escape:
            if self.zoomRectButton.isChecked():
                self.zoomRectCancelled()
            elif self.rulerButton.isChecked() and self.tempSegmentON:
                self.tempSegmentON = False
                self.ax1_rulerPlotItem.setData([], [])
                self.ax1_rulerAnchorsItem.setData([], [])
            elif self.customAnnotButton is not None:
                self.setUncheckedAllCustomAnnotButtons()
            else:
                self.uncheckLeftClickButtons(None)
        elif key == Qt.Key_Z:
            if self.posData().SizeZ > 1:
                self.zSliceCheckbox.setChecked(
                    not self.zSliceCheckbox.isChecked()
                )
        elif key in (Qt.Key_Left, Qt.Key_Right):
            step = 1 if key == Qt.Key_Right else -1
            if self.isZsliceNavigationActive():
                z = self.zSliceScrollBar.sliderPosition() + step
                self.zSliceScrollBar.setSliderPosition(z)
            else:
                self.setFrame(self.frame_i + step)
        elif key in (Qt.Key_Up, Qt.Key_Down):
            self.stepID(1 if key == Qt.Key_Up else -1)
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        if self._isLoadingData:
            # The load continues after its dialogs/worker thread return
            text = 'Wait for the data being loaded before closing the viewer.'
            self.logger.warning(text)
            self.statusBar().showMessage(text, 8000)
            event.ignore()
            return
        if self.measurementProcessingThread is not None:
            self.measurementProcessingWarning(
                'Measurement processing running',
                'Wait for measurement processing to finish before closing the viewer.'
            )
            event.ignore()
            return
        if self.isDataLoaded:
            # The prompt does not block the window: no load may replace the
            # data it saves (`loadingInProgress`) while it is open
            self._isClosingPrompt = True
            try:
                proceed = self.askSaveBeforeClosing()
            finally:
                self._isClosingPrompt = False
            if not proceed:
                event.ignore()
                return
            self.saveAnnotationOptions()
        if self.slideshowWin is not None:
            self.slideshowWin.close()
        if self.ccaTableWin is not None:
            self.ccaTableWin.close()
        if self.countObjsWindow is not None:
            self.countObjsWindow.close()
        volume_renderer = getattr(self, '_volume_renderer', None)
        if volume_renderer is not None:
            self._volume_renderer = None
            volume_renderer.force_close()
        self.saveWindowGeometry(setting_suffix='_measurements_viewer')
        self.logger.info('Measurements viewer closed')
        super().closeEvent(event)
        self.sigClosed.emit(self)
        
        
def run(folderpath=None):
    """Launch the measurements viewer as a standalone application, optionally 
    loading `folderpath` (experiment, Position or Images folder)."""
    from ._run import _setup_app
    app, _ = _setup_app()
    win = MeasurementsViewerWin()
    if folderpath:
        # The folder is loaded directly: the startup configuration prompt
        # would open on top of the folder loading dialogs.
        win._startupConfigurationPrompted = True
    win.show()
    if folderpath:
        win.openFolder(folderpath=folderpath)
    app.exec_()

if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else None)
