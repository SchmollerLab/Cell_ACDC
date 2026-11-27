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
- **Compat shims**: the files at the old module paths (`apps.py`,
  `widgets.py`, `workers.py`, `gui_navigation.py`, `_main.py`, ...) and the
  `__getattr__` of `gui/__init__.py`, `measurements_viewer/__init__.py` and
  `myutils/__init__.py` only forward old names to their new homes, with a
  `MovedWarning` (`_compat.py`, tables in `_compat_moves.py`). They exist
  for external code (spotMAX, user plugins, notebooks). Cell-ACDC code and
  tests never use them: pytest turns the warning into an error and
  `tests/test_no_old_paths.py` scans for old paths. Exception: the plugin
  API names of `cellacdc.myutils` (`PLUGIN_API` in `_compat_moves.py`, e.g.
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
line count (`1.4k` = about 1400 lines) and its theme. Not listed: the
compat shims at old paths, `qrc_resources*.py` (generated Qt resources), the
contents of `docs/` and `bioformats/` (vendored) and the plugin folders
(`models/<Name>/`, ...: one folder per plugin).

<!-- codemap:tree:start -->
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
  compat shims).
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
