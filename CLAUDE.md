# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Python environment

**Always use this Python executable** for running code, tests, and scripts:

```
%USERPROFILE%\miniforge3\envs\acdc\python.exe
```

That is the `acdc` conda environment of the miniforge installation in the user's home folder (Python 3.13, PyQt6, pytest installed; no flake8/pyflakes, see Testing). Its editable install points to a different checkout (`~/repos/Cell_ACDC`), so run Python/pytest from this repo's root (`python -m ...` puts the cwd first on `sys.path`); scripts run from elsewhere (e.g. the scratchpad) need `PYTHONPATH=<this repo>` or they import the other clone. Write it as `~/miniforge3/envs/acdc/python.exe` in Bash and `$env:USERPROFILE\miniforge3\envs\acdc\python.exe` in PowerShell. Do not use a bare `python`/`pytest` from PATH. Examples (Bash):

```bash
~/miniforge3/envs/acdc/python.exe -m pytest tests/test_phase_translator.py -q
~/miniforge3/envs/acdc/python.exe -m pytest tests -k "measurement_viewer"
~/miniforge3/envs/acdc/python.exe -m cellacdc            # launch main GUI
~/miniforge3/envs/acdc/python.exe -m cellacdc.measurements_viewer [folder]
```

## Project overview

Cell-ACDC (Cell segmentation, tracking and event annotation) is a PyQt (via `qtpy`) + `pyqtgraph` desktop application for live-cell imaging analysis, from the Schmoller lab (Helmholtz Munich). Package: `cellacdc/`. Install for development with `pip install -e .` plus `requirements_gui_pyqt6.txt` (or the `gui` extra). Version comes from `setuptools_scm` (`cellacdc/_version.py`, generated).

Entry points (`pyproject.toml`): `cellacdc` / `acdc` / `Cell-ACDC` → `cellacdc.__main__:run` (GUI, or headless CLI when `--params <ini>` is given, see `cli.py` / `_run.py`); `acdc-setup-models` → `cellacdc.service.setup_models:main`.

## Architecture

The codebase is large; many core files are huge single modules (`gui.py` ~31k lines, `apps.py` ~25k, `widgets.py` ~15k, `workers.py` ~7.5k, `myutils.py` ~6.5k). Search with Grep rather than reading whole files.

- **Launcher** — `_main.py` (`mainWin`): the main launcher window that opens the modules (data prep, segmentation, GUI, utilities).
- **Main GUI** — `gui.py` (`guiWin`): the annotation/correction GUI. It is composed of many mixin classes from `gui_navigation.py`, `gui_annotations.py`, `gui_custom_annotations.py`, `gui_viewer_tools.py`, `gui_overlay.py`, `gui_combine.py`, `whitelist.py`. Mixins are shared with other windows, so changes there affect multiple windows. When merging `main`, its edits to methods that this branch moved out of `gui.py` show up as delete/modify conflicts in `gui.py`: port them into the mixin version (find it by method name) instead of re-adding the method to `gui.py`. Host-specific behaviour of shared code goes through hooks with main-GUI defaults, e.g. `ShortcutsGuiElements.showDeleteObjectShortcut`, `CustomAnnotationsGuiElements.canDeleteSavedCustomAnnot()` / `customAnnotColumnRemovalNote()`, the optional `bottomLayoutFont` read by `gui_overlay`, and `limitToLoadedColumns` of the combine dialogs (`True` only in the viewer).
- **Measurements viewer** — `measurements_viewer.py` (`MeasurementsViewerWin`, active development on branch `signal_gui`): standalone window that plots per-cell time courses of `acdc_output` columns, measurement-vs-measurement comparisons, combined measurements, measurement processing, time point identifiers, custom annotations and phases. It reuses the GUI mixins above and loads its own data; configurations are saved as JSON in the settings folder. The module docstring is an extensive spec—read it before changing behavior. Users open it from the launcher (Utilities > Measurements > Measurements viewer..., `_main.mainWin.launchMeasurementsViewer`).
  - Overlay channel image items are added to `ax1` when data loads (after the annotation items), so annotation layers need explicit z-values to stay visible (`gui_createAnnotationsWidgets`: 0.5–0.6; custom annotation symbols 0.6 via the viewer's `addCustomAnnnotScatterPlot` override; ruler and mirrored cursor 0.9 in `gui_createGraphicsPlots`; on-demand items (object tracks, scale bar, timestamp) 0.9 via `raiseOnDemandAx1Items`, so add new on-demand ax1 items there; text IDs are at 1).
  - `getAcdcDf` rebinds `posData.acdc_df` on every position load, so never detect stale tables by DataFrame identity: measurement processing merges only its output columns at completion and checks `posData.acdcDfReloadCount` (bumped in `reloadAcdcDfFile`). Tables are normalized on load/reload by `normalize_loaded_acdc_df` (`load._load_acdc_df_file` fills NaN with 0 in non-cca columns when cca columns exist, so phases go through `phase_translator.normalize_loaded_outputs`); `saveAcdcDfs` checks `posData.acdcDfFileStamp` for files changed on disk. Read stored flag columns (`timepoint_*`, `measurements_viewer_visited`, custom annotations, `phase_<name>`) only with `timepoint_predictors.parse_flags` (`strict=True` to also recognise whether a column is a flag column, as for phase markers), never `astype(bool)` or `pd.to_numeric(...) != 0`: the main GUI's `store_data` reindexes with `fill_value=0`, so they come back as `'True'`/`'False'`/0 strings. Each measurement-step output column comes from exactly one recorded step (`recordMeasurementStep` replaces an exact-output match, ordered by `measurementStepReplacementOrder`; `validateMeasurementSteps` rejects duplicates and steps reading outputs of a later step). Session auto-prediction tracking (`timepointAutoPredictions` / `customAnnotAutoPredictions`) is reset by `resetAutoPredictionTracking` whenever data loads and rebuilt from the visited column via `seedVisited*`; enabling auto apply or editing a predictor must not re-predict visited cells. Removed QActions stay children until `deleteLater` runs and `reservedShortcuts` (findChildren) still sees them: clear their shortcut first.
  - Shortcuts: a time point/annotation shortcut K gets the auto (re-evaluate) shortcut Shift+K (`auto_shortcut_text`), so K is reserved when Shift+K is taken (`reservedShortcuts`); premade templates are checked by `test_premade_shortcuts_do_not_collide`. Renaming a predictor output column is refused while other predictors/formatting reference it (`columnDependents`), because equations are not rewritten.
  - Time-course plotting (`plotMeasurements`): `connectSignalPoints` re-calls `scatter.setData`, so per-point options (brushes) must be passed again there. With a pyqtgraph `connect` array, non-finite points are back-filled and lines bridge NaN gaps unless the neighbouring segments are masked out (see conditional formatting).
  - Phase names that CSV reading turns into numbers/booleans/NA (e.g. `'1'`, `'NA'`) are recovered from the boolean `phase_<name>` columns (`phase_translator._phase_strings`), so phase names need no extra validation.
  - Phase display: `phaseBandPlot` is a separate `PlotWidget` x-linked below `measurementsPlot` (left axis width synced via `syncPhaseBandAxisWidth`); redrawn by `updatePhaseBand` from `plotMeasurements`. Colors: `phase_translator.phase_colors` + `self.phaseColors` overrides.
  - Persist display-only per-user toggles (e.g. "Show phases") with `gui_core.GUICoreElements._get_setting_value` / `_set_setting_value` (settings CSV); options that change analysis behaviour (e.g. `auto_apply_phases`) belong in the viewer configuration (`configuration()` / `validateConfiguration` / `loadConfiguration`).
  - Premade configurations: templates in `cellacdc/measurements_viewer_premade_configurations/*.json` with `channel_placeholders`, optional typed `input_placeholders` (values asked at setup, with info text) and `{NAME}` tokens; on load they are resolved and copied into the settings folder (`instantiatePremadeConfiguration`). Resolve templates in code/tests with `MeasurementsViewerWin.resolvePremadeTemplate` (fills input defaults), not by popping `channel_placeholders` by hand. In template equations/predictors/conditions wrap token-built column names in backticks (`` `{RED}_mean` ``): substitution unquotes plain identifiers, and channels that pandas cannot parse even quoted (digit first or right after a dash, e.g. `561`, `GFP-488`) are rejected (`isEquationName`; the mapping dialog's `validate` callback). Keep such checks explicit (regex), never based on `tokenize`/pandas parsing alone: Python 3.10/3.11 tokenize `561_mean` without error, 3.12+ rejects it. Python 3.11/3.12 interpreters without project deps are under `~/.pyenv/pyenv-win/versions/` for syntax/tokenizer checks. `channel` may be empty in any configuration (e.g. FUCCI: both reporters are overlays): the display channel is asked when loading (`selectDisplayChannel` / `resolveInputChannel`), so code after input selection can rely on a non-empty `channel`.
  - `plotMeasurements` resets the time-course x range only when `_timecourseXRangeKey` changes (cleared on `loadPosition` and in comparison mode), so replots keep the user's zoom.
  - Predictor fast paths (`OtherCellMeasurements` block emission and `_equality_candidates` value index for lineage sources, `TimepointPredictor._evaluate_all_cells` for row-wise equations via `CellEquation.is_row_wise`) must stay exactly equivalent to the per-target/per-cell paths; `tests/test_predictor_fast_paths.py` disables them to compare.
  - `timepoint_predictors.py`: time point identifiers and predictors (equations over acdc_df columns with per-cell functions like `MAX`, `DIFF`, `SHIFT`; filters; other-cell aggregates; frame windows; Python-file `predictor(df)`), dependency ordering of interdependent predictors. Classes: `TimepointIdentifier`, `TimepointPredictor`, `CellEquation`, `CellFilter`, `OtherCellMeasurements`, `CellTable`, `FrameRange`, `PredictorDependencies`.
  - `custom_annotation_predictors.py`: boolean per-cell annotations built on the same machinery, consolidated with All/None/Most.
  - `phase_translator.py`: translates ordered time point boundaries into `phase` / `phase_<name>` columns per rule.
  - `measurement_processing/`: plugin steps (see below).
  - `_equations.py`: equation evaluation used by "Combine measurements".
- **Dialogs and widgets** — `apps.py` (dialogs/windows), `widgets.py` (custom widgets), `_base_widgets.py`. Dialogs used only by the measurements viewer (phase translator, time point identifier and predictor/filter equation editors, custom annotation predictor group box, measurement processing recipe, channel/input setup, conditional formatting) live in `apps_measurements_viewer.py`. Shared bases stay in `apps.py` (`_CombineMetricsDialog`, `CombineMetricsEquationDialog`, `customAnnotationDialog`). `apps.py` must not import `apps_measurements_viewer` at module level (circular); `customAnnotationDialog` imports the predictor group box locally, only when `predictorOptions` is given (viewer).
- **Background work** — `workers.py`: `QObject` workers run in `QThread`s with signals for progress/logging.
- **Data I/O** — `load.py` (`loadData` = per-Position data container `posData`; paths, images, segm, `acdc_output` CSV, metadata), `io.py`, `path.py`, `dataStruct.py`, `dataPrep.py`, `dataReStruct.py`.
- **Core algorithms** — `core.py`, `segm.py`, `segm_utils.py`, `measure.py`, `measurements.py`, `regionprops.py`, `features.py`, `cca_functions.py` (cell cycle analysis), `preprocess.py`, `transformation.py`. Cython: `precompiled_functions.pyx` (built via `precompile_functions.py`; prebuilt in `precompiled/`).
- **Utilities** — `myutils.py` (grab bag: model/tracker discovery and import, installs, settings), `cellacdc/__init__.py` defines globals such as `printl` (debug print with caller location), `settings_folderpath` (`~/acdc-appdata/.acdc-settings` or `<repo>/.acdc-settings`), `user_profile_path`, `exception_handler` decorators.
- **Model service** — `service/`: FastAPI/uvicorn server + client to run segmentation models in separate environments (startup and environment config live outside it: `cellacdc/_model_startup.py`, `cellacdc/models/_env_config.py`).
- **Batch utilities** — `utils/`: standalone utility tools (convert, align, concat, ImageJ ROI, Trackastra export, etc.) launched from the launcher.
- **Resources** — `qrc_resources*.py` are generated Qt resource files (~340k lines); never edit or read them. Icons live in `resources/`.
- **Docs** — `cellacdc/docs/source` (Sphinx), `UserManual/`, `cellacdc/code_structure.md` (notes on custom annotations in the main GUI).

## Plugin conventions (discovered by folder name)

- **Segmentation models** — `cellacdc/models/<Name>/acdcSegment.py` with a `Model` class exposing `__init__(...)` and `segment(image, ...)` returning a label array. Init/segment keyword arguments with defaults become GUI parameters. Folders starting with `_` are ignored. See `models/thresholding/acdcSegment.py` for a minimal example.
- **Trackers** — `cellacdc/trackers/<Name>/<Name>_tracker.py` with a `tracker` class and `track(segm_video, signals=None, ...)`.
- **Promptable models** — `cellacdc/promptable_models/<name>/`.
- **Measurement processing** — `cellacdc/measurement_processing/<Name>/acdcProcess.py` with module-level `name` and a `Model` class with flags `is_timelapse`, `is_bulk_processing`, `numeric_input` and a `@staticmethod` `run(...)` (`Model` is never instantiated; a non-static `run` is skipped at load). See `SmoothMeasurement`. Recipes in configurations reference the module `name` (e.g. `"Ramp detection"`), not the folder name. NumPy-style `run` docstring parameters become the per-option info buttons; JSON kwargs are passed without type conversion, but recipe INI files store every kwarg as a string: cast them with `MeasurementProcessingParamsWidget.castRecipeValue` (bools via `ConfigParser.BOOLEAN_STATES`), never `bool(str)`. Plugins that fail to import, lack a non-empty `name` or `Model.run`, or repeat a `name` are skipped with a `RuntimeWarning` (first wins); `_load_processors(folder, package)` can target a tmp package in tests. Existing: `SmoothMeasurement`, `MinMaxNormalization` (fixed constants), `RampDetection` (FUCCI ramps, used by `fucci_cell_cycle.json`).
- `measurement_processing.apply_recipe(df, cols, recipe, new_colnames)` appends one output column per input and rejects mismatched, duplicate or already existing output names; `workers.MeasurementProcessingWorkerGUI` is one-shot (`setupJob` + `runJob`, result via `sigDone`).

## Data layout

Experiments are organized as `<exp_folder>/Position_<n>/Images/` containing `<basename><channel>.tif`, segmentation `<basename>segm*.npz`, the annotations/measurements table `<basename>acdc_output*.csv` (one row per `frame_i` × `Cell_ID`), and metadata CSV/JSON files. `acdc_df` in code refers to the `acdc_output` table (indexed by `frame_i`, `Cell_ID`).

## Testing

- Tests live in `tests/` (pytest; config in `pyproject.toml`, which ignores `cellacdc/test_tracker.py` and DeprecationWarnings). Shared helpers in `tests/utils/`, sample data in `tests/test_data/`.
- Qt tests create a module-scoped `QApplication.instance() or QApplication([])` fixture and drive widgets with `QTest`; dialogs that would block are monkeypatched (see `tests/test_measurement_viewer_configuration.py`). No pytest-qt.
- UI tests must run offscreen and never block (user requirement): `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen` (override the env var to watch the UI) and a watchdog auto-answers any dialog still blocking after 0.3 s (OK/default button, `ok_cb`, accept), reported as `AutoAnsweredDialogWarning` (handles `myMessageBox`, plain `QMessageBox` and other dialogs; see `tests/test_conftest_dialog_watchdog.py`). Still monkeypatch dialogs whose answer matters. The conftest also redirects user settings (`settings_folderpath` and the module-level settings/metadata/recent-paths/shortcuts/custom-annotation paths, QSettings) to `tmp_path` (autouse `_isolate_user_settings`) and destroys `deleteLater()`'d widgets after each test (`flushDeferredDeletes`; outside an event loop `processEvents()` never deletes them).
- Keep tests fast (user requirement): a test file should take seconds, not minutes. Use small synthetic tables (tens of cells/frames), keep parametrized grids and random parity/fuzz cases to a handful, avoid long `qWait`s, and check new tests with `pytest --durations=15`. Benchmarks and large fuzz runs belong in scratch scripts run with a timeout, not in the suite. Slowest files now (~1-2 min each): `test_equation_inputs.py`, `test_measurement_viewer_configuration.py`, `test_predictor_fast_paths.py`, `test_timepoint_identifiers.py`, `test_measurement_viewer.py`.
- Never assert inside a `QTimer.singleShot` slot in tests: an exception there aborts the whole pytest process. Record values in the slot and assert after the blocking call returns.
- The identity `Dummy` measurement processor is test-only (`tests/utils/measurement_processing.py`); tests get it through the `dummy_processor` fixture.
- Some tests need optional model environments/GPU/network (`test_all_model_environments.py`, `test_gpu_availability.py`, `tests/segm`, `tests/prompt_segm`); prefer running targeted test files.
- Local lint: flake8/pyflakes are not installed in the `acdc` env. Install them into a scratch folder without touching the env (`python -m pip install --target <dir> flake8 pyflakes`) and run `PYTHONPATH=<dir> python -m flake8 --select=E9,F63,F7,F82 <files>` (the CI check) / `python -m pyflakes <file>`; `python -m py_compile <file>` for a quick syntax check.
- Run Qt test files one at a time (`pytest tests/<file>`): since the conftest flushes deleted widgets, one pytest process over the whole suite also passes (as in CI), but a crash there may not print a summary.
- The measurements viewer's recipe dialog offers only measurement leaves of a hierarchical model (group headings are not selectable); tests select leaves by name and must enter a non-colliding output suffix before `ok_cb()`.
- Importing `cellacdc.measurements_viewer` in a bare `python -c` script prints a caught `regionprops.py` traceback (`app.mainWindow`) but works; smoke-test widgets through pytest instead.
- CI (`.github/workflows/`) runs `flake8 --select=E9,F63,F7,F82` (syntax errors / undefined names must not occur) and `pytest` on Python 3.10–3.13: PyQt5 and PyQt6 on Windows and macOS, PyQt5 only on Ubuntu.
- Platform pitfalls seen in CI: on macOS shortcut texts use macOS key names (`KeySequenceFromText.toString()`, `ShortcutLineEdit` display text, `ShortcutEditorDialog.new_hard_shortcuts` give `Command+O`; `ShortcutLineEdit.text()` gives `Ctrl+O`), so compare `QKeySequence`s or use `widgets.windowsShortcutToMac` in tests, and simulate with `monkeypatch.setattr(widgets, 'is_mac', True)`; macOS ignores `QMessageBox.setWindowTitle`; Qt5 `QAction.trigger()` emits `triggered` even for a disabled action, so handlers must not assume loaded data; Windows CI checks out on `D:` while tmp is on `C:` (`os.path.commonpath`/`relpath` raise `ValueError`); Python 3.10/3.11 `tokenize` recovers from errors (e.g. an apostrophe in a backtick name) that 3.12+ raise on.
- To test PyQt5 / Python 3.11 locally: make a venv from `~/.pyenv/pyenv-win/versions/3.11.9/python.exe` and `pip install pytest -e . -r requirements_gui_pyqt5.txt`. On Windows, showing a plain `QMessageBox` with Qt5 on the offscreen platform crashes the interpreter (also in CI, killing the whole pytest run): tests that show one get the `shows_qmessagebox` skip mark (`test_conftest_dialog_watchdog.py`); use `QT_QPA_PLATFORM=windows` to run them locally.

## Code style notes

- Use `qtpy` imports (not `PyQt6` directly) so PyQt5/PyQt6/PySide all work.
- GUI code uses Qt-style camelCase method names; newer non-GUI modules (predictors, phase translator) use snake_case functions and class-based APIs with thorough docstrings.
- Keep module docstrings in sync when changing behavior of the measurements viewer and predictor modules—they serve as the specification.
- Main branch for PRs: `main`.

## Maintaining this file

When you learn something during a task that would have saved time or prevented a mistake, and it is likely to matter again, add it here before finishing the task.

**Add:** non-obvious conventions, gotchas, where an important piece of logic lives, how to run or test something, environment quirks, and corrections from the user ("don't do X", "always do Y").

**Don't add:** what is obvious from reading the code, one-off task details, git history, or anything temporary. Keep entries to one or two lines and point to files (`module.py` / `ClassName`) instead of copying code.

**How:**

- Put each entry in the section it belongs to (Architecture, Plugin conventions, Data layout, Testing, Code style). If none fits, create a new `##` section rather than appending to the end of the file.
- Update or delete an existing entry that is outdated or wrong; never keep two entries that contradict each other.
- If an entry describes a whole subsystem and grows beyond a few lines, put the details in a doc next to the code (like `cellacdc/code_structure.md`) and link to it from here.
- In your reply to the user, say briefly what you added or changed in this file.
