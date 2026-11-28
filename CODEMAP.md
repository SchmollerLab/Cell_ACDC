# CODEMAP

Where the code of Cell-ACDC lives, for people and for Claude Code sessions:
find the module here, then grep inside it. Conventions and gotchas are in
`CLAUDE.md`, this file is the map. Paths are relative to `cellacdc/` unless
they start with `cellacdc/`, `tests/` or `tools/`; `path::Name` names a
class, function or method defined in that file.

Keep it complete: after adding, moving or deleting a module or package, run
`python tools/codemap.py` (rewrites the package tree of section 3, keeps the
descriptions already there, refreshes the line counts) and describe the new
entries. `tests/test_codemap.py` fails when a package or module is not
listed, or when a path or `path::Name` named here does not exist.

## 1. How the code is organized

- **Program parts**: one package per part of the program, with its window
  and the dialogs, widgets and workers that only this part uses:
  `launcher/`, `data_structure/`, `data_prep/`, `segmentation_module/`,
  `gui/` (main GUI), `measurements_viewer/`, and the batch utilities in
  `utils/` (one module per utility, with its dialogs and worker).
- **Shared code**: `shared/dialogs/`, `shared/widgets/`, `shared/workers/`,
  one module per theme, for code used by two or more parts, by a library
  module, by a plugin or by spotMAX. Cohesive families stay together
  (the PushButton family in `shared/widgets/push_buttons.py`).
- **Library**: top-level modules without a window (`load.py`, `core.py`,
  `measurements.py`, `cli.py`, `timepoint_predictors.py`, `whitelist.py`,
  ...) and the helper package `myutils/` (imports without Qt).
- **Plugins**, discovered by folder name: `models/`, `trackers/`,
  `promptable_models/`, `measurement_processing/` (CLAUDE.md "Plugin
  conventions"). **Services**: `service/` (model server and environments),
  `volume_renderer/` (3D viewer), `help/`.
- **`gui/mixins/` is shared too**: the measurements viewer window composes
  these mixins, and the dialogs and widgets in `gui/mixins/dialogs.py` and
  `gui/mixins/widgets.py` are reached only through them, so changes there
  affect both windows. Code that only the main GUI uses is in
  `gui/main_window/`, `gui/dialogs/`, `gui/widgets/`, `gui/workers/` and
  `gui/helpers.py`; `gui/mixins/` and `measurements_viewer/` never import
  them.
- **Old import paths (compat)**: the old modules (`cellacdc.apps`,
  `cellacdc.widgets`, `cellacdc.workers`, `cellacdc.gui_navigation`,
  `cellacdc._main`, ...) have no files any more: an import hook of
  `_compat.py` (`OldPathFinder`, installed by `__init__.py`, table
  `OLD_MODULES` in `_compat_moves.py`) serves them, and the `__getattr__`
  of `gui/__init__.py`, `measurements_viewer/__init__.py` and
  `myutils/__init__.py` forwards the old names of those packages. Both give
  the objects at their new homes, with a `MovedWarning`. They exist for
  external code (spotMAX, user plugins, notebooks). Cell-ACDC code and tests
  never use them: pytest turns the warning into an error and
  `tests/test_no_old_paths.py` scans for old paths (and fails when a file
  exists at an old module path). Exception: the plugin API names of
  `cellacdc.myutils` (`PLUGIN_API` in `_compat_moves.py`, e.g.
  `myutils.check_install_package`) stay reachable without a warning for
  external models.

## 2. Where new code goes

Placement rules:

1. Used by one program part: that part's package (`<part>/dialogs`,
   `<part>/widgets`, `<part>/workers`), or the module of its single user.
2. Reached only through the GUI mixins that the viewer also composes:
   `gui/mixins/dialogs.py`, `gui/mixins/widgets.py`. Code of the main GUI
   only: `gui/main_window/`, `gui/dialogs/`, `gui/widgets/`, `gui/workers/`,
   `gui/helpers.py`.
3. Used by one batch utility: `utils/<utility>.py`; by several utilities:
   `utils/base.py`.
4. Used by several parts, a library module, a plugin or spotMAX:
   `shared/<kind>/<theme>.py`. Cohesive families stay together.
5. Foundation leaves import no module of the import cycle at module level
   (section 6).
6. Modules import modules (aliases with attribute access), absolute outside
   the part package, relative inside it (section 6).
7. Windows: `QMainWindow` first, then the window-only mixins, then the
   shared mixins; Qt-named methods, `__init__`, `super()` users, Signals and
   overriding class attributes stay in the class body (section 5).

| New code | Goes into |
| --- | --- |
| dialog, widget or worker used by one program part | that part: `<part>/dialogs.py`, `<part>/widgets.py`, `<part>/workers.py` (a package by theme when large), or the module of its single user |
| ... reached only from the viewer-shared GUI mixins | `gui/mixins/dialogs.py`, `gui/mixins/widgets.py` |
| ... used by several parts, a library module, a plugin or spotMAX | `shared/dialogs/<theme>.py`, `shared/widgets/<theme>.py`, `shared/workers/<theme>.py` |
| ... of one batch utility | `utils/<utility>.py`; used by several utilities: `utils/base.py` |
| helper without a window | the library module of its theme, or `myutils/<theme>.py` (must import without Qt) |
| main GUI method | the `<Topic>GuiElements` mixin of its topic in `gui/main_window/<topic>.py`; the class body in `gui/main_window/window.py` only for the cases of section 5 |
| method both windows need | the mixin of its topic in `gui/mixins/`, with a host hook (section 5) where the windows differ |
| measurements viewer method | the `Viewer<Topic>GuiElements` mixin in `measurements_viewer/main_window/<topic>.py`; widget-free logic in `measurements_viewer/viewer_utils.py`, `measurements_viewer/position_data.py` or `measurements_viewer/annotation_data.py` |
| new utility | `utils/<utility>.py` + its action in `launcher/main_window.py` + help text in `info.py` (`utilsInfo`) |
| new module | a module docstring whose first sentence names its theme, then `python tools/codemap.py` |

## 3. Package tree

One line per package, module and data folder (indented by folder level), its
line count (`1.4k` = about 1400 lines) and its theme. Not listed:
`qrc_resources*.py` (generated Qt resources), the
contents of `docs/` and `bioformats/` (vendored) and the plugin folders
(`models/<Name>/`, ...: one folder per plugin).

<!-- codemap:tree:start -->
```
cellacdc/                                    the package root; library modules (no program part) live here
  __init__.py                          940   package globals: paths, settings files, GUI_INSTALLED, printl, exception_handler, disableWindow
  __main__.py                          170   entry point run() of cellacdc/acdc/Cell-ACDC: launcher GUI, or headless CLI with --params
  _base_widgets.py                      40   QBaseDialog (non-modal exec_), leaf base of every dialog
  _compat.py                           450   compat layer of old import paths: import hook OldPathFinder (install), MovedWarning, alias_module, moved_module, package_compat
  _compat_moves.py                    2.0k   GENERATED tables old module -> new home of every moved name (do not edit)
  _core.py                             180   small numeric helpers: unit conversion, rotational volume, significant digits
  _debug.py                            110   debugging helpers (callers, lineage tree, mother-bud line)
  _equations.py                         65   input-token detection of pandas measurement equations (backtick names)
  _get_app_palette.py                   24   dev script: prints the Fusion palette colors
  _model_startup.py                     29   debug-mode timings of model environment preparation and worker startup
  _palettes.py                         250   light/dark color palettes and text/highlight colors of the Qt app
  _process.py                           36   script: runs comma-separated commands in a separate process
  _run.py                              720   app setup (_setup_app, GUI libraries, qrc resources, model params) and the CLI workflows (run_cli)
  _types.py                            110   typed parameter markers of model/function signatures (Vector, FolderPath, NotGUIParam, ...)
  _view_all_buttons.py                  71   dev script: shows every PushButton of shared/widgets/push_buttons.py
  _view_all_icons.py                    66   dev script: shows every icon of the qrc resources
  _warnings.py                         630   warning message boxes and warning classes shared by all parts (warnTooManyItems, ...)
  _window_stacking.py                  280   keeps Cell-ACDC windows above their parents instead of above all apps
  acdc_regex.py                         95   regexes for floats, function names and alphanumeric names
  annotate.py                         1.3k   text annotations of the images (IDs, cell cycle info), object tracks item
  autopilot.py                         120   AutoPilot: replays the dialog answers of the last loading (AutoPilotProfile)
  background_values.py                 410   SPEC (docstring): background intensity of channels and normalization constants
  cca_functions.py                    1.0k   cell cycle analysis of acdc_output tables (downstream data, relatives, per-phase)
  cli.py                              2.9k   headless kernels: SegmKernel, ComputeMeasurementsKernel (Set measurements -> saved columns)
  colors.py                            790   colormaps, gradients (gradient_state_to_setting), LUT helpers, color math
  config.py                            300   ConfigParser and the preprocessing recipe <-> INI mappers
  core.py                             3.8k   core image analysis: segmentation post-processing, tracking helpers, label edits
  core_split_IDs.py                   1.3k   splitting merged objects (2D cuts, 3D convexity-defect planes)
  data.py                              310   example datasets (_Data subclasses) used by tests and demos
  debugutils.py                        550   benchmarks, call stacks, qt_debug_trace, check_unused_methods
  exporters.py                         330   image/SVG/video exporters of the current view (ffmpeg)
  features.py                          210   feature-range filters, combined metrics and custom post-processing of acdc_df
  fonts.py                              14   default QFont objects
  html_utils.py                        400   HTML text helpers for message boxes (paragraph, tag, href_tag, rst_to_html)
  info.py                              150   help texts: utilsInfo (key = utility action text)
  io.py                                370   file I/O helpers: measurement selections, image saving, file moves
  lineage_tree_integrity.py           1.5k   SPEC (docstring): lineage tree column checks of acdc_output tables, CLI
  load.py                             4.9k   loadData (= posData, the per-Position container), paths, images, segm, acdc_output I/O
  measure.py                            94   rotational volume and label separation
  measurements.py                     2.9k   measurement names and functions, custom and combined metrics, unselected_measurement_columns
  path.py                              170   path helpers: listdir, Position folder walks, show in file manager
  plot.py                              960   plotting helpers: imshow (preview window), heatmaps, matplotlib LUTs
  preprocess.py                        780   image preprocessing functions (steps of preprocessing recipes)
  prompts.py                           520   Qt prompts of library code (channel selection, SAM embeddings, model params)
  QtScoped.py                           90   scoped Qt enum values that differ between PyQt5 and PyQt6
  qutils.py                            530   Qt helpers: key event routing, QControlBlink, tool tip wrapping, setRetainSizePolicy
  record.py                            280   screen recorder window
  regionprops.py                      1.3k   acdcRegionprops: lazy regionprops with 3D/projection extensions
  rst_utils.py                         250   rst helpers; get_tooltips_from_docs reads docs/source/tooltips.rst
  sam_embeddings.py                    240   persists SAM embeddings without PyTorch in the client
  segm_utils.py                        380   segmentation helpers: overlaps, bounding boxes, single-cell segmentation
  syntax.py                            190   Python syntax highlighter of code editors
  test_segm_model.py                   270   dev script: test a segmentation model outside the GUI (TestSegmModelInitalDialog)
  test_tracker.py                      160   dev script: test a tracker outside the GUI
  timepoint_predictors.py             3.7k   SPEC (docstring): time point identifiers and predictors (CellEquation, filters, ...)
  transformation.py                    400   label/image transformations: resize, crop, retracking, TrackMate XML
  urls.py                               37   URLs (docs pages, GitHub, forum)
  whitelist.py                         940   Whitelist data class (used by load) and nearest_ID_to_centroid
  _deprecated/                               deprecated code, not imported
  _html/                                     HTML pages of the welcome guide
  _profile/                                  profiling scripts, not imported
  acdc_bioio_bioformats/                     Bio-Formats readers through bioio (raw microscopy files of the data structure module)
    __init__.py                         45   JAVA_HOME setup and the extension -> bioio plugin package table
    _init_reader.py                     26   initializes the bioio reader
    _read_metadata.py                   43   reads the metadata of a raw file
    _read_sample_data.py               100   reads sample image data of a raw file
    _readers.py                        110   reader subprocess helpers
    _save_data.py                      190   saves the channels of a raw file as TIFFs
    _save_data_single_channel.py       190   saves one channel of a raw file
    _utils.py                          350   argument parsing, file names, symlinks, supported extensions
    install.py                          66   installs bioio-bioformats and the extra format packages
    reader.py                          300   ImageReader and the OME metadata classes
  bioformats/                                vendored python-bioformats (contents not listed)
  data_prep/                                 PART data prep module: z-slices, frames, crops, alignment, background ROIs
    dialogs.py                         310   data-prep-only dialogs (sub-crops paths, frame-range crop tool)
    main_window.py                    3.6k   dataPrepWin, the data prep window
    workers.py                         270   data prep workers: align, save background data, crop
  data_structure/                            PART data structure module: raw files or loose TIFFs -> Position folders
    data_restruct.py                   590   re-structuring loose TIFF files into Positions
    data_struct.py                    2.4k   createDataStructWin: raw microscopy files -> Positions (Bio-Formats)
    helpers.py                         430   Java/Bio-Formats/javabridge install and checks, git check
    workers.py                          68   data structure workers (symlinked Positions)
    dialogs/                                 dialogs of the data structure module only
      metadata.py                     1.5k   OME-XML metadata and Fiji macro conversion dialogs, installJavaDialog
      setup.py                        1.6k   setup dialogs of data structure and TIFF re-structuring
  docs/                                      Sphinx user docs (contents not listed; docs/source/tooltips.rst is read at runtime)
  fiji_macros/                               Fiji macros (.ijm)
    __init__.py                        190   running Fiji macros and the Fiji download/command helpers
  gui/                                       PART main GUI (guiWin): annotate and correct segmentation, tracking, cell cycle
    helpers.py                         250   non-UI helpers of guiWin only (border objects, ID formatting, distances)
    dialogs/                                 dialogs of guiWin only, by theme
      data_io.py                      1.0k   load, save, restore, export dialogs of the main GUI
      editing.py                      2.1k   object and annotation editing dialogs (edit ID, cca table, ...) of the main GUI
      points_layers.py                 780   AddPointsLayerDialog
      segmentation.py                  990   segmentation/tracking dialogs of the main GUI (selectTrackerGUI, ...)
    main_window/                             guiWin (window.py) and its window-only <Topic>GuiElements mixins
      cca_annotation.py               2.2k   manual cell cycle annotation: clicks in Cell cycle analysis mode
      cca_data.py                     2.0k   cca tables: auto annotation, integrity checker, annotations after segm edits
      constants.py                      55   module constants of the main GUI (colors, sizes, _font: read/write as constants._font)
      drawing_tools.py                1.5k   brush, eraser, magic wand, curvature, expand/move label, manual background
      exports.py                       560   exports of the current view, videos and images
      frame_data.py                   1.7k   store_data/get_data, labels, regionprops, undo/redo
      frame_navigation.py             1.1k   frame, Position and z-slice navigation of guiWin only
      hover_cursors.py                1.1k   tool cursors and image hover handlers
      image_display.py                1.2k   image and labels display, title label with IDs, status bar label
      image_processing.py             1.1k   preprocessing recipes and combine channels in the GUI (CombineGuiElements)
      keyboard.py                      640   key handling helpers, shortcut registry (reservedShortcuts, getHardcodedShortcuts)
      layout.py                       1.5k   window layout: bottom widgets, sliders, plots, image items, right image
      lineage_tree.py                 1.1k   lineage tree mode (backup, askLineageTreeChanges, resetLin_tree_future)
      loading.py                      2.0k   loading data: open file/folder, load worker, init after loading, recovery
      magic_prompts.py                1.3k   magic prompts (promptable models) workflow
      menus_actions.py                1.1k   menu bar, actions and connections, terminal widget, tooltips
      modes.py                         810   mode switching (changeMode) and tool button states
      mouse_events.py                 1.1k   mouse events except the left image: click helpers, right image
      mouse_left_image.py             1.8k   mouse press/drag/release on the left image (editing tools)
      object_editing.py               1.5k   edit ID, merge, delete, separate, propagate, Keep IDs
      points_layers.py                1.4k   points layers
      prompt_annotations.py           1.3k   box and brush-stroke prompts of the magic prompts
      rois.py                         1.5k   deletion ROIs, label ROI / freehand region tool
      saving.py                       1.2k   save data and its prompts, autosave, Set measurements
      segmentation.py                 1.7k   repeat segmentation, model execution, post-processing, segment lost IDs
      toolbars.py                     1.6k   toolbars, controls toolbar, tool settings and right-click menus
      tracking.py                     2.0k   real-time and repeat tracking, lost/new objects, manual tracking
      whitelist_ids.py                1.1k   Whitelist IDs tool (WhitelistGUIElements)
      window.py                       1.0k   class guiWin: bases, Signals, __init__, Qt overrides (keyPressEvent, eventFilter, ...)
      window_setup.py                  640   run(), initGlobalAttr, session settings, help/about/log menus, progress callbacks
    mixins/                                  mixins ALSO COMPOSED BY THE MEASUREMENTS VIEWER: changes affect both windows
      annotations.py                  1.7k   annotation layers: contours, masks, IDs, cca and lineage tree texts
      custom_annotations.py            920   custom annotations (CustomAnnotationsGuiElements) and their host hooks
      dialogs.py                      2.3k   dialogs opened only from these mixins (customAnnotationDialog, ShortcutEditorDialog, ...)
      navigation.py                    860   frames, z-slices, zoom to object, recent paths, LutGuiElements (per-channel LUT SPEC)
      object_info.py                   960   cca table, object counts, search ID, hover info, region properties
      overlay.py                      1.7k   overlay channels and overlay labels
      viewer_tools.py                  960   zoom, ruler, scale bar/timestamp, color scheme, labels cmap, slideshow, 3D viewer
      widgets.py                      1.5k   widgets reached from these mixins (FontSizeWidget, TimeWidget, ...)
      window_core.py                   440   settings CSV values, geometry, workers, shortcuts, default settings (GUICoreElements, ...)
    widgets/                                 widgets of guiWin only (old gui.widgets.X names forwarded for external code)
      edit_toolbars.py                1.4k   editing toolbars (manual tracking, copy lost object, whitelist, ...)
      main_window_items.py             460   ghost contour/mask items, label ROI items, clipSelemMask
      points_layer_widgets.py         1.3k   points layers and magic prompts toolbars and widgets
    workers/                                 workers of guiWin only (old gui.workers.X names forwarded for external code)
      annotation_checks.py             380   CcaIntegrityCheckerWorker
      image_processing.py              270   preprocessing and combine-channels workers of the GUI
      loading_saving.py                650   loadDataWorker, LazyLoader, autosave/state workers, are_acdc_dfs_equal
      segmentation_tracking.py        1.4k   segmWorker, segmVideoWorker, trackingWorker, magic prompts workers
  help/                                      help windows
    about.py                           260   About dialog and the version commands it shows
    welcome.py                        1.2k   welcome guide window and example data download
  launcher/                                  PART launcher: module buttons and the Utilities menu
    dialogs.py                         220   launcher-only dialogs (user profile folder, sub-cellular tracking setup)
    main_window.py                    2.6k   mainWin: module buttons, Utilities menu actions and their launch* slots, self-update
    workers.py                          78   MigrateUserProfileWorker
  measurement_processing/                    measurement processing plugins, one folder per step (<Name>/acdcProcess.py)
    __init__.py                        250   plugin loading (_load_processors) and apply_recipe
  measurements_viewer/                       PART measurements viewer (MeasurementsViewerWin)
    __init__.py                        700   the viewer SPEC (docstring) + lazy compat of old names
    __main__.py                         10   python -m cellacdc.measurements_viewer [folder]
    annotation_data.py                 630   AnnotationDataUtils: widget-free time points, annotations, predictors, phases
    custom_annotation_predictors.py    270   boolean per-cell annotation predictors (All/None/Most)
    phase_translator.py                900   time point boundaries -> phase / phase_<name> columns
    position_data.py                   540   PositionDataUtils: widget-free per-position and per-frame data
    viewer_utils.py                   1.8k   MeasurementsViewerUtils + module functions (load_position, signal_color, get_logger, ...)
    widgets.py                         540   _TimeCoursePlotItem, extra y-axis view boxes, time point lines, _ShortcutAwareAction
    workers.py                          97   _ThreadTaskWorker (runInThread), MeasurementProcessingWorkerGUI
    dialogs/                                 dialogs of the viewer only
      conditional_formatting.py        330   conditional formatting dialogs and relative colors
      phase_translator_dialogs.py      500   PhaseTranslatorDialog and its rule dialog
      predictor_editors.py            1.8k   TimepointIdentifierDialog, predictor/filter equation editors, disabled_while_nested
      premade_setup.py                 500   setup dialog of premade configurations (placeholders, background values)
      processing_recipe.py             270   measurement processing recipe dialog and parameter widgets
    main_window/                             MeasurementsViewerWin (window.py) and its Viewer<Topic>GuiElements mixins
      columns.py                       360   Columns to plot list: colors (getSignalColor), y axes, conditional formatting
      configuration.py                 990   configurations: save/select/load, premade templates (instantiatePremadeConfiguration)
      custom_annotations.py            680   custom annotations of the viewer (hook overrides of gui/mixins/custom_annotations.py)
      data_io.py                      1.0k   load/save/reload/export, runInThread, loading guards, unsaved-changes prompts
      image_panel.py                   870   image panel: images, text annotations, LUTs, viewer tools, status bar label
      navigation.py                    560   frame, z-slice, Cell ID navigation, arrow keys (navigateWithArrowKey)
      phases.py                        390   phase translator UI and phase evaluation scheduling
      plotting.py                      770   plotMeasurements, comparison scatter, extra y axes, autoscale, phase band
      processing.py                    380   Process measurements and Combine measurements
      timepoints.py                   1.1k   time point identifiers UI and predictor evaluation
      ui_setup.py                     1.9k   builds the window: actions, menus, toolbars, widgets, docks
      window.py                        660   class MeasurementsViewerWin (bases, class attributes, Qt/super() overrides) and run()
    premade_configurations/                  premade configuration templates (*.json)
  metrics/                                   example custom metric scripts for users
  models/                                    segmentation model plugins, one folder per model (<Name>/acdcSegment.py)
    _env_config.py                     350   isolated environment configuration of the models (packages, min_version)
    _yeaz_v2_weights.py                 65   weights registry of YeaZ v2 and its parameter widget
    _cellpose_base/                          shared Cellpose model base of the cellpose plugins
      _directML.py                      82   DirectML device setup for Cellpose
      acdcSegment.py                   720   Cellpose Model base class
  myutils/                                   helper library by theme, imports without Qt (cellacdc.myutils.X plugin API kept without warning)
    app_info.py                        590   version (read_version), git info, logging (setupLogger, Logger), log files
    downloads.py                       550   URL/zip helpers, model weights (get_model_path, download_model), manual, ffmpeg
    filesystem.py                      560   Position/Images detection, listdir, channel/basename lookup, recent paths, browse_url
    image_utils.py                     370   dtype conversions (to_uint8, img_to_float), TIFF writing, iterate_along_axes
    installs.py                       1.2k   pip/conda installs: check_install_package, _install_pip_package, is_server_mode
    misc.py                            230   standard-library-only helpers: text formatting, iteration, try_kwargs
    model_installs.py                  630   check_install_<model> functions, Cellpose versions, GPU checks
    model_modules.py                   290   Qt-free model module introspection (its own ArgSpec), import acdcSegment
    segm_models.py                     790   model/tracker registry and discovery, ArgSpec, import/init models and trackers
    table_utils.py                     260   acdc_df helpers: base tables, checked_reset_index, df_ctc_to_acdc_df
    torch_utils.py                     560   PyTorch install, wheel discovery, version checks, get_torch_device
  napari_utils/                              napari plugins integration
    arboretum.py                       130   napari-arboretum lineage tree viewer dialog
  precompiled/                               prebuilt Cython extensions (precompiled_functions)
  promptable_models/                         promptable model plugins, one folder per model (<name>/acdcPromptSegment.py)
    micro_sam_base.py                  420   prompt handling shared by micro-sam_v1 and micro-sam_v2
    saved_models.py                    410   registry of trained promptable models saved from the GUI
    utils.py                           440   brush strokes, mask building and insertion of model outputs
  resources/                                 icons (<name>.svg + <name>_dark.svg), logos, fonts
  scripts/                                   standalone user scripts, not imported
  segmentation_module/                       PART segmentation and tracking module (whole experiments, outside the GUI)
    dialogs.py                         530   stop frame and z-plane prompts of the module
    main_window.py                    1.7k   segmWin and its segmentation/tracking worker flow
  service/                                   model service: HTTP worker in a model env, client and env management
    __init__.py                         55   re-exports ModelClient, the backends and PROTOCOL_VERSION
    client.py                          850   ModelClient, LocalBackend, RemoteBackend (client side)
    environment.py                    1.4k   model environments: create, check (_check_environment_dependencies), repair
    model_execution.py                 560   decides where a segmentation model runs and creates it
    payload.py                         270   encodes model call arguments for HTTP
    prompts.py                         290   client-side prompts (CliHandler)
    protocol.py                         34   PROTOCOL_VERSION of client <-> worker
    schema.py                           54   response schemas of the worker
    setup_models.py                     46   acdc-setup-models: prepares model environments
    worker.py                         1.0k   HTTP worker that serves one model (python -m cellacdc.service.worker --server)
  shared/                                    SHARED code used by several parts, library modules or plugins
    dialogs/                                 shared dialogs, by theme
      base.py                           13   QBaseDialog, base of the dialogs (foundation leaf: imports only Qt, _base_widgets)
      combine_measurements.py         1.3k   _CombineMetricsDialog, CombineMetricsEquationDialog (equation editors)
      file_selection.py               1.3k   file/folder/Position selection (filenameDialog, SelectFoldersToAnalyse)
      generic.py                      1.4k   combobox, line edit, tree and progress dialogs (QDialogWorkerProgress, TreeSelectorDialog)
      image_tools.py                  1.3k   imageViewer (slideshow), crop tools, points/overlay appearance
      metadata.py                      640   QDialogMetadata
      model_params.py                 2.2k   QDialogModelParams, post-processing and feature-range widgets
      model_selection.py              1.4k   QDialogSelectModel, downloadModel, InstallPyTorchDialog
      preprocessing.py                1.9k   FunctionParamsDialog, ArgWidget, preprocessing recipe dialogs
      set_measurements.py             2.2k   SetMeasurementsDialog, askDeleteUncheckedExistingMeasurements, _metricsQGBox
      tracker_params.py                550   Cell-ACDC, BayesianTracker and Delta tracker parameter windows
    widgets/                                 shared widgets and pyqtgraph items, by theme (__init__ imports style)
      basic_widgets.py                 600   separators, spacers, ScrollArea, labels, progress bars, log console
      combo_boxes.py                   400   ComboBox (previousText), typable/centered combos, PreProcessingSelector
      imshow_window.py                1.2k   the image preview window of plot.imshow (ImShow)
      item_views.py                    520   QDialogListbox (allowEmptySelection), listWidget, TreeWidget
      line_edits.py                   1.0k   validated line edits, path controls, forms, ShortcutLineEdit, is_mac key texts
      lut_items.py                    1.2k   histogram LUT items (baseHistogramLUTitem), colormap menus, gradient editors
      message_box.py                   900   myMessageBox and its content helpers
      numeric_inputs.py                730   SpinBox/DoubleSpinBox, sliders, scroll bars, RangeSelector
      plot_items.py                   1.1k   MainPlotItem, image items (ParentImageItem), PlotCurveItem, ruler
      push_buttons.py                 1.4k   PushButton family of icon buttons, getPushButton (icon by keyword), myColorButton
      search_widgets.py                990   ButtonSearchWidget, fuzzy search completers
      style.py                         150   style constants, fonts, colormap registry (cmaps, Gradients); run at import
      toolbars.py                      650   ToolBar (overflow shortcuts), OverlayToolbar, tool buttons
    workers/                                 shared worker infrastructure and workers
      base.py                          340   worker_exception_handler, workerLogger, signals, SimpleWorker, AutoPilotWorker
      saving.py                        380   saveDataWorker (main GUI and Compute measurements utility)
  trackers/                                  tracker plugins, one folder per tracker (<Name>/<Name>_tracker.py)
  utils/                                     PART batch utilities: one module per utility with its dialogs and worker
    acdcOutputToCtcTable.py            140   acdc_output -> Cell Tracking Challenge table
    acdcToSymDiv.py                    420   acdc_output -> symmetric division lineage
    align.py                           350   align frames of Positions
    applyTrackFromTable.py             220   apply tracking info from a table
    applyTrackFromTrackMateXML.py      130   apply tracking info from TrackMate XML
    base.py                           1.2k   NewThreadMultipleExpBaseUtil, BaseWorkerUtil, dialogs shared by utilities
    combineChannels.py                 290   combine channels utility
    compute.py                        1.1k   Compute measurements utility (ComputeMetricsWorker)
    computeBackgroundValues.py         890   background intensity utility (core: background_values.py)
    computeMultiChannel.py             710   combine measurements of several channels/tables
    concat.py                         1.0k   Concatenate acdc_output tables (AllPos_*)
    convert.py                         700   convert file formats, images -> Positions
    countObjects.py                    120   count objects in segmentation files
    createConnected3Dsegm.py            71   connected 3D segmentation from z-slices
    customPreprocess.py                240   custom preprocessing utility
    editMetadata.py                    410   edit metadata of Positions
    fillHolesInSegm.py                 170   fill holes in segmentation masks
    filterObjFromCoordsTable.py        190   filter objects with a coordinates table
    fromImageJroiToSegm.py             430   ImageJ ROIs -> segmentation
    fucciPreprocess.py                 300   FUCCI preprocessing
    generateMothBudTotalTable.py       600   mother+bud total table
    rename.py                          360   rename files
    repeat.py                          500   repeat data prep
    segm_workers.py                    550   workers of segmentation-file utilities whose names clash with their windows
    splitVideoIntoFrameTiffs.py        250   split videos into per-frame TIFFs
    stack2Dinto3Dsegm.py                71   stack 2D segmentations into 3D
    toImageJroi.py                     180   segmentation -> ImageJ ROIs
    toObjCoords.py                     130   segmentation -> object coordinates table
    toTrackastraInputData.py           470   Trackastra input data export
    trackSubCellObjects.py             330   track sub-cellular objects
    resize/                                  resize Positions utility
      __init__.py                      410   resizing functions (images, segm, tables, metadata)
      util.py                          250   ResizePositionsUtil window and worker
  volume_renderer/                           3D volume viewer (vispy)
    _dialogs.py                        350   animation setup dialog
    _widgets.py                        410   toolbars, overlays, points layer context menu
    canvas.py                         2.9k   vispy canvas, lazy volumes, labels and points layers
    gl_blend.py                         66   OpenGL blend state of volume layers
    utils.py                            88   frame stacking, display scale, demo data download
    _demos/                                  demo scripts, not imported
```
<!-- codemap:tree:end -->

## 4. Task → location

| Task | Where |
| --- | --- |
| package globals: paths, settings folder, `printl`, `exception_handler` | `cellacdc/__init__.py` |
| app start, GUI or CLI entry | `cellacdc/__main__.py::run`, `_run.py::_setup_app`, `_run.py::run_cli` |
| standard message box; button icons by keyword | `shared/widgets/message_box.py::myMessageBox`, `shared/widgets/push_buttons.py::getPushButton` |
| list selection dialog | `shared/widgets/item_views.py::QDialogListbox` |
| combobox, line edit, progress and tree dialogs | `shared/dialogs/generic.py::QDialogCombobox`, `shared/dialogs/generic.py::QLineEditDialog`, `shared/dialogs/generic.py::QDialogWorkerProgress`, `shared/dialogs/generic.py::TreeSelectorDialog` |
| base class of the dialogs (non-modal `exec_`) | `shared/dialogs/base.py::QBaseDialog`, `_base_widgets.py::QBaseDialog` |
| file, folder and Position selection | `shared/dialogs/file_selection.py::filenameDialog`, `shared/dialogs/file_selection.py::SelectFoldersToAnalyse` |
| colormaps, gradients, stylesheets, fonts | `shared/widgets/style.py`, `colors.py` |
| LUT items and the colormap menu | `shared/widgets/lut_items.py::baseHistogramLUTitem` |
| toolbars and their overflow shortcuts | `shared/widgets/toolbars.py::ToolBar` |
| shortcut texts, macOS key names | `shared/widgets/line_edits.py::ShortcutLineEdit`, `shared/widgets/line_edits.py::windowsShortcutToMac` |
| Set measurements (dialog -> saved columns) | `shared/dialogs/set_measurements.py::SetMeasurementsDialog`, `cli.py::ComputeMeasurementsKernel`, `measurements.py` |
| combine-measurements equation editors | `shared/dialogs/combine_measurements.py::_CombineMetricsDialog` |
| model selection, installation, parameters; tracker parameters | `shared/dialogs/model_selection.py::QDialogSelectModel`, `shared/dialogs/model_params.py::QDialogModelParams`, `shared/dialogs/tracker_params.py` |
| preprocessing recipes and function parameters | `shared/dialogs/preprocessing.py::FunctionParamsDialog`, `shared/widgets/combo_boxes.py::PreProcessingSelector`, `preprocess.py` |
| slideshow window | `shared/dialogs/image_tools.py::imageViewer` |
| worker infrastructure (exception handler, logger) | `shared/workers/base.py::worker_exception_handler`, `shared/workers/base.py::workerLogger` |
| main GUI: saving | `gui/main_window/saving.py`, `shared/workers/saving.py::saveDataWorker` |
| main GUI: loading data | `gui/main_window/loading.py::LoadDataGuiElements.loadingDataCompleted`, `gui/workers/loading_saving.py::loadDataWorker`, `load.py::loadData` |
| main GUI: class body, key events, app event filter | `gui/main_window/window.py::guiWin` (`keyPressEvent`, `eventFilter`, `KEY_EVENT_KEYS`) |
| main GUI: shortcuts and reserved keys | `gui/main_window/keyboard.py::KeyboardGuiElements.reservedShortcuts`, `gui/mixins/window_core.py::ShortcutsGuiElements`, `gui/mixins/dialogs.py::ShortcutEditorDialog` |
| main GUI: mode switching | `gui/main_window/modes.py::ModesGuiElements.changeMode` |
| main GUI: frame and Position navigation | `gui/main_window/frame_navigation.py`, `gui/mixins/navigation.py` (shared widgets) |
| main GUI: per-frame data, undo/redo | `gui/main_window/frame_data.py::FrameDataGuiElements.store_data` |
| main GUI: drawing tools, object edits | `gui/main_window/drawing_tools.py`, `gui/main_window/object_editing.py`, `gui/main_window/mouse_left_image.py` |
| main GUI: segmentation and tracking | `gui/main_window/segmentation.py::SegmentationGuiElements.repeatSegm`, `gui/main_window/tracking.py`, `gui/workers/segmentation_tracking.py::trackingWorker`, `gui/dialogs/segmentation.py` |
| main GUI: cell cycle annotation clicks vs tables | `gui/main_window/cca_annotation.py` (clicks), `gui/main_window/cca_data.py::AnnotationsAfterEditsGuiElements.warnEditingWithCca_df` (tables, annotations after edits) |
| main GUI: lineage tree mode | `gui/main_window/lineage_tree.py::LineageTreeGuiElements.askLineageTreeChanges`, `trackers/CellACDC_normal_division/CellACDC_normal_division_tracker.py::normal_division_lineage_tree` |
| main GUI: magic prompts (promptable models) | `gui/main_window/magic_prompts.py`, `gui/main_window/prompt_annotations.py`, `promptable_models/` |
| per-channel LUTs and display settings (spec) | `gui/mixins/navigation.py::LutGuiElements` |
| overlay channels and labels | `gui/mixins/overlay.py::OverlayChannelsGuiElements` |
| custom annotations (both windows) | `gui/mixins/custom_annotations.py::CustomAnnotationsGuiElements`, `gui/mixins/dialogs.py::customAnnotationDialog`, `cellacdc/code_structure.md` |
| viewer tools: zoom, ruler, scale bar, 3D viewer | `gui/mixins/viewer_tools.py::Volume3DViewerGuiElements`, `volume_renderer/` |
| measurements viewer: spec | `measurements_viewer/__init__.py` (docstring), `cellacdc/docs/source/measurements-viewer.rst` |
| measurements viewer: window and loading | `measurements_viewer/main_window/window.py::MeasurementsViewerWin`, `measurements_viewer/main_window/data_io.py::ViewerDataIOGuiElements.runInThread` |
| measurements viewer: configurations, premade templates | `measurements_viewer/main_window/configuration.py::ViewerConfigurationGuiElements.instantiatePremadeConfiguration`, `measurements_viewer/premade_configurations/`, `measurements_viewer/dialogs/premade_setup.py` |
| measurements viewer: plotting | `measurements_viewer/main_window/plotting.py::ViewerPlottingGuiElements.plotMeasurements`, `measurements_viewer/widgets.py::_TimeCoursePlotItem` |
| measurements viewer: time points UI | `measurements_viewer/main_window/timepoints.py`, `measurements_viewer/dialogs/predictor_editors.py::TimepointIdentifierDialog` |
| measurements viewer: widget-free logic | `measurements_viewer/viewer_utils.py::MeasurementsViewerUtils`, `measurements_viewer/position_data.py::PositionDataUtils`, `measurements_viewer/annotation_data.py::AnnotationDataUtils` |
| predictors and phases (libraries) | `timepoint_predictors.py`, `measurements_viewer/custom_annotation_predictors.py`, `measurements_viewer/phase_translator.py` |
| measurement processing plugins | `measurement_processing/__init__.py::apply_recipe`, `measurements_viewer/dialogs/processing_recipe.py` |
| package installs | `myutils/installs.py::check_install_package`, `myutils/installs.py::_install_pip_package`, `myutils/model_installs.py` |
| model and tracker discovery, init | `myutils/segm_models.py::get_list_of_models`, `myutils/segm_models.py::init_tracker`, `myutils/model_modules.py` |
| model environments and the model server | `service/environment.py::get_or_create_model_env`, `models/_env_config.py`, `service/worker.py`, `service/client.py::LocalBackend` |
| Java and Bio-Formats | `data_structure/helpers.py`, `acdc_bioio_bioformats/` |
| version, logging | `myutils/app_info.py::read_version`, `myutils/app_info.py::setupLogger` |
| launcher actions and the Utilities menu | `launcher/main_window.py::mainWin` |
| batch utility pattern | `utils/base.py::NewThreadMultipleExpBaseUtil`, `utils/base.py::BaseWorkerUtil`, `utils/computeBackgroundValues.py` (example), `info.py::utilsInfo` |
| old import paths (compat) | `_compat.py`, `_compat_moves.py` |

## 5. Window classes

- `gui/main_window/window.py::guiWin` bases: `QMainWindow`, then the
  window-only `<Topic>GuiElements` mixins of `gui/main_window/` (in the
  order of the topics), then the shared mixins of `gui/mixins/`, ending with
  `GUICoreElements`. A name resolves in the first base that defines it, so
  a window-only mixin overrides a shared one.
- `measurements_viewer/main_window/window.py::MeasurementsViewerWin` bases:
  `QMainWindow`, the `Viewer<Topic>GuiElements` mixins of
  `measurements_viewer/main_window/`, `MeasurementsViewerUtils` (widget-free;
  bases `PositionDataUtils`, `AnnotationDataUtils`), then the same shared
  mixins of `gui/mixins/`. The viewer overrides a shared method by defining
  it in one of its own mixins (e.g. `customAnnotColumnRemovalNote` in
  `measurements_viewer/main_window/custom_annotations.py`).
- What stays in the class body: `__init__`; methods with a Qt name
  (`keyPressEvent`, `eventFilter`, `closeEvent`, `showEvent`,
  `setWindowTitle`, ...: `QMainWindow` comes first in the bases, so a mixin
  method with a Qt name would never be called); methods calling `super()`
  (zero-argument `super()` needs the class body); Signals (`sigClosed`); and
  class attributes that override a mixin's (the viewer's
  `showDeleteObjectShortcut = False`). Everything else goes into the mixin of
  its topic. So the viewer's overrides that extend a shared mixin method
  with `super()` (`setShortcuts`, `overlay_cb`, `invertBw`,
  `updatePropsWidget`, `addCustomAnnnotScatterPlot`, ...) are in its class
  body, its other overrides in its `Viewer<Topic>GuiElements` mixins.
- Host hooks: shared code asks the window through attributes and methods
  with main-GUI defaults, overridden by the viewer:
  `gui/mixins/window_core.py::ShortcutsGuiElements.showDeleteObjectShortcut`,
  `gui/mixins/custom_annotations.py::CustomAnnotationsGuiElements.canDeleteSavedCustomAnnot`,
  `customAnnotColumnRemovalNote`, `customAnnotColumnConflict`, the optional
  `bottomLayoutFont` read by `gui/mixins/overlay.py`, and
  `limitToLoadedColumns` of the combine dialogs (`True` only in the
  viewer). The main GUI lists windows parented elsewhere in
  `gui/main_window/keyboard.py::KeyboardGuiElements.ownedWindowsOutsideParentChain`.
- Checked by `tests/test_gui_window_layout.py` and
  `tests/test_viewer_window_layout.py` (helpers in `tests/window_layout.py`):
  their `OVERRIDES` pin the class that defines each name defined in two or
  more classes; after an intended new override, print the new dictionary
  with `python -m tests.test_gui_window_layout` (or
  `tests.test_viewer_window_layout`).
- Finding a method: `grep -rn "def <name>" cellacdc/gui cellacdc/measurements_viewer`
  (a method defined in the window body and in a mixin: the body wins).

## 6. Conventions

- **Module aliases**: modules import other cellacdc modules, not names, and
  use them with attribute access, so that one test patch on the owner module
  reaches every caller and the import cycle of the GUI modules stays safe:
  `from cellacdc.shared.widgets import message_box as widgets_message_box`,
  then `widgets_message_box.myMessageBox(...)`. Do the same in new code.
  From-imports of cellacdc names remain for the package globals
  (`from cellacdc import printl, exception_handler, is_mac, ...`),
  constants, a window's own mixin classes, and base classes and functions
  of older code (`from cellacdc.utils.base import NewThreadMultipleExpBaseUtil`,
  `from cellacdc.plot import imshow`): such a name is bound in the
  importing module (patch it there).
- **Canonical aliases**: shared modules `<kind>_<module>`
  (`widgets_message_box`, `dialogs_generic`, `workers_base`); main GUI
  modules `gui_main_<module>` (`gui_main_constants`), `gui_dialogs_<module>`,
  `gui_widgets_<module>`, `gui_workers_<module>`, `gui_mixin_dialogs`,
  `gui_mixin_widgets`; the mixins moved whole keep their old names
  (`gui_navigation`, `gui_annotations`, `gui_custom_annotations`,
  `gui_overlay`; also `gui_viewer_tools`, `gui_object_info`,
  `gui_window_core`); viewer modules `viewer_<module>` (`viewer_utils`,
  `viewer_widgets`, `viewer_configuration`) and `viewer_dialogs_<module>`;
  windows `gui_window`, `viewer_window`, `launcher_window`,
  `data_prep_window`, `segm_window`, `data_struct`; `myutils` submodules by
  their bare name (`installs`, `filesystem`, `segm_models`); utilities
  `utils_<module>`. Imports that existed before the refactor kept their
  local name for modules moved whole
  (`from cellacdc.data_prep import main_window as dataPrep`, also `segm`,
  `_main`, `dataStruct`, `myutils_models`); new imports use the canonical
  alias.
- **Imports**: absolute for modules outside the importing file's part
  package, relative inside it. A window module imports its mixins; mixin
  modules never import their window module.
- **Patching in tests**: patch the attribute in the module whose code looks
  it up: the owner module for alias-qualified names
  (`monkeypatch.setattr(widgets_message_box, 'myMessageBox', ...)`), every
  importing module for names bound by from-imports (`printl`, `is_mac`,
  Qt classes). Never patch a name through another module's namespace.
- **Foundation leaves**: modules whose names other modules bind at import
  time (base classes, decorators, constants) import only the standard
  library, third-party packages, Qt, `cellacdc` globals and other leaves at
  module level: `shared/dialogs/base.py`, `shared/widgets/style.py`,
  `shared/widgets/numeric_inputs.py`, `shared/widgets/search_widgets.py`,
  `shared/widgets/plot_items.py`, `shared/widgets/push_buttons.py`,
  `shared/widgets/toolbars.py`, `shared/workers/base.py`,
  `gui/main_window/constants.py`. Anything else is a function-level import
  (the module-level imports kept from the old modules are listed in
  `KNOWN_CYCLE_IMPORTS` of `tests/test_import_layering.py`: never add new
  ones). `gui/main_window/constants.py` `_font` is read and written
  through the module (`gui_main_constants._font`), never with `global`.
- **No old paths** in `cellacdc/`, `tests/` and notebooks (section 1,
  old import paths).
- **Checks**: `tests/test_import_paths.py` (every import names something
  that exists, no relative import leaves its part),
  `tests/test_import_layering.py` (library and `shared/` modules never import
  a part package at module level; imports between parts are listed in its
  `CROSS_PART_IMPORTS`; foundation leaves), `tests/test_no_old_paths.py`,
  `tests/test_package_init_files.py` (every folder with modules has an
  `__init__.py`: the wheel leaves out the others), `tests/test_compat.py`
  (old paths still resolve) and `tests/test_codemap.py`.
- **Docstrings**: every module has one whose first sentence names its theme
  (the default description of section 3). Package docstrings name files
  with slash paths (`gui/mixins/navigation.py`), never dotted `cellacdc.`
  names (`tests/related_tests.py` reads those as imports).

## 7. Old name → new location

The package-layout refactor moved names without renaming them (old
modules in bold, relative to `cellacdc/`):

| Old module | New location |
| --- | --- |
| **_main.py** | `launcher/main_window.py` |
| **dataStruct.py**, **dataReStruct.py** | `data_structure/data_struct.py`, `data_structure/data_restruct.py` |
| **dataPrep.py** | `data_prep/main_window.py` |
| **segm.py** | `segmentation_module/main_window.py` |
| **gui.py** | `gui/main_window/window.py` (class body of `guiWin`) and the mixins of `gui/main_window/`; module constants in `gui/main_window/constants.py`; `qt_debug_trace` in `debugutils.py` |
| **gui_navigation.py**, **gui_annotations.py**, **gui_custom_annotations.py**, **gui_overlay.py** | `gui/mixins/navigation.py`, `gui/mixins/annotations.py`, `gui/mixins/custom_annotations.py`, `gui/mixins/overlay.py` |
| **gui_viewer_tools.py**, **gui_core.py** | `gui/mixins/viewer_tools.py`, `gui/mixins/object_info.py`, `gui/mixins/window_core.py` |
| **gui_combine.py** | `gui/main_window/image_processing.py` |
| **gui_utils.py** | `whitelist.py` |
| **whitelist.py** (**WhitelistGUIElements**) | `gui/main_window/whitelist_ids.py` (the `Whitelist` class stays) |
| **measurements_viewer.py** | spec: `measurements_viewer/__init__.py`; `MeasurementsViewerWin`: `measurements_viewer/main_window/window.py` and the mixins of `measurements_viewer/main_window/`; helpers: `measurements_viewer/widgets.py`, `measurements_viewer/workers.py` |
| **measurements_viewer_utils.py** | `measurements_viewer/viewer_utils.py`, `measurements_viewer/position_data.py`, `measurements_viewer/annotation_data.py` |
| **apps_measurements_viewer.py** | `measurements_viewer/dialogs/` |
| **custom_annotation_predictors.py**, **phase_translator.py**, **measurements_viewer_premade_configurations/** | `measurements_viewer/custom_annotation_predictors.py`, `measurements_viewer/phase_translator.py`, `measurements_viewer/premade_configurations/` |
| **apps.py** | by user: `shared/dialogs/`, `gui/dialogs/`, `gui/mixins/dialogs.py`, `launcher/dialogs.py`, `data_structure/dialogs/`, `data_prep/dialogs.py`, `segmentation_module/dialogs.py`, `utils/<utility>.py` |
| **widgets.py** | `shared/widgets/`, `gui/mixins/widgets.py`, `gui/widgets/`, `measurements_viewer/widgets.py`, single-user widgets next to their dialog |
| **workers.py** | `shared/workers/`, `gui/workers/`, `launcher/workers.py`, `data_structure/workers.py`, `data_prep/workers.py`, `measurements_viewer/workers.py`, `utils/<utility>.py`, `utils/segm_workers.py` |
| **myutils.py** | `myutils/` by theme; single-part helpers in `gui/helpers.py`, `data_structure/helpers.py`, `launcher/main_window.py`, `fiji_macros/__init__.py`, `help/`, `utils/<utility>.py`, `qutils.py` |
| **myutils_models.py**, **_torch_utils.py** | `myutils/model_modules.py`, `myutils/torch_utils.py` |

Where one old name went:

    python -c "from cellacdc._compat_moves import NAME_MOVES as M; print(M['cellacdc.widgets']['QDialogListbox'])"

(`NAME_MOVES[old module][name]` = `(home module, attribute)`; modules moved
whole are in `MODULE_ALIASES`). A former `guiWin` or `MeasurementsViewerWin`
method is still a method of the class: find its mixin with
`grep -rn "def <name>"` (section 5).

## 8. Specs (docstrings to read before changing behaviour)

- Measurements viewer: `measurements_viewer/__init__.py`; user docs
  `cellacdc/docs/source/measurements-viewer.rst` and
  `cellacdc/docs/source/equation-editor.rst`.
- Per-channel LUTs and channel display settings:
  `gui/mixins/navigation.py::LutGuiElements`.
- Predictors and phases: `timepoint_predictors.py`,
  `measurements_viewer/custom_annotation_predictors.py`,
  `measurements_viewer/phase_translator.py`.
- Background values: `background_values.py`. Lineage tree checks:
  `lineage_tree_integrity.py`.
- Custom annotations in the main GUI: `cellacdc/code_structure.md`.
- Compat layer: `_compat.py`. Model environments and the model server:
  `cellacdc/docs/source/model-environments.rst`.

## 9. History

- Files moved whole keep their history across the move:
  `git log --follow -- <new path>`. For code split out of a large module
  use `git blame -C -C -C <new path>` (finds lines moved between files) or
  `git log -L :<function>:<old path> <commit before the refactor>`.
- `.git-blame-ignore-revs` lists the commits of the refactor that only
  rewrote imports, references and docstrings; `git blame` uses it after
  `git config blame.ignoreRevsFile .git-blame-ignore-revs` (GitHub does it
  automatically).
- Porting a branch started on the old layout: `tools/refactor/README.md`
  ("Porting an open branch", while `tools/refactor/` exists), then compare
  every item the branch changed with an `ast` script (CLAUDE.md "Main
  GUI").
