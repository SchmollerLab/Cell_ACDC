# tools/refactor: the package-layout refactor as a reproducible pipeline

This folder turns the hand-reviewed mapping (`mapping/*.json`) into the
commits of the package-layout refactor (plan: CODEMAP.md, "Commit
sequence"), checks the mapping, and documents how to port branches that
were started on the old layout. Everything here uses only the standard
library (`ast`, `symtable`, `tokenize`, `json`, `subprocess`) and parses on
Python 3.10 (CI runs flake8 E9/F63/F7/F82 on the whole repository); run it
with the development interpreter (Python >= 3.12 recommended: f-string
positions).

```
tools/refactor/
  README.md            this file
  mapping.py           validate the shards, emit the name table, coverage report
  run.py               apply / check / replay (the pipeline)
  qt_names.json        dir(QMainWindow) of PyQt5 and PyQt6 (plan_check)
  mapping/<shard>.json the source of truth (one shard per area)
  acdc_refactor/       the codemod (source, items, scopes, imports, plan,
                       move, shims, plan_check, rewrite) + templates/_compat.py
  verify/              verification scripts (run_all(commit))
  fixups/<commit>/     scripted manual edits, run after the automatic step
  commit_messages/     <commit>.txt, the commit messages used by replay
  tests/               unit tests on synthetic packages (not in testpaths)
  out/                 logs and generated files (git-ignored)
```

## Quick start

```bash
# from the repository root, with the development interpreter
python tools/refactor/mapping.py validate             # schema, keys, coverage, aliases
python tools/refactor/mapping.py coverage-report -v   # per old module
python tools/refactor/mapping.py name-table --out tools/refactor/out/name_table.json
python -m acdc_refactor.plan_check                    # (cwd tools/refactor) all static checks

python tools/refactor/run.py apply C3                 # one commit's automatic step on the working tree
python tools/refactor/run.py apply C3 --fixups        # ... plus tools/refactor/fixups/C3/*.py
python tools/refactor/run.py check C3                 # plan_check + verify scripts
python tools/refactor/run.py replay --base <sha> --until C16 --branch refactor/package-layout-rebuilt

# unit tests (without the repository conftest)
python -m pytest tools/refactor/tests -q -p no:cacheprovider --rootdir tools/refactor --noconftest
```

A replay of one's own (parallel work, scratch outputs):

```bash
ACDC_REFACTOR_OUT=<scratch>/out_x python tools/refactor/run.py replay \
    --base <sha> --until C12 --branch tmp/x-1
# worktree: <scratch>/out_x/replay/tmp_x-1; afterwards
# git worktree remove --force <path> && git branch -D tmp/x-1
```

The verify scripts of `run.py check <commit>` read the name table of that
commit from `$ACDC_REFACTOR_OUT/<commit>/name_table.json` (where every old
name lives at that commit, so the fingerprints of C3-C9 compare with the
intermediate homes; `check` writes it, and the final table to
`$ACDC_REFACTOR_OUT/name_table.json`), the edits logged by the replay from
`$ACDC_REFACTOR_OUT/<commit>/changes.jsonl` (run `check` with the
`ACDC_REFACTOR_OUT` of the replay) and the base snapshots from
`$ACDC_REFACTOR_OUT/baseline/` (copy `tools/refactor/out/baseline`).
`check` writes its snapshots into the same folders, so parallel checks
need one `ACDC_REFACTOR_OUT` copy each.

Dry runs belong in a scratch worktree (`git worktree add <scratch>/wt/x
HEAD --detach`, then `run.py --repo <scratch>/wt/x apply C3`), never in a
checkout other people work in. All `run.py` commands take `--repo` and
`--mapping`; the mapping keys are always read at the base commit
(`4c4e498b`, override with the environment variable `ACDC_REFACTOR_BASE`
or `mapping.py --base`), with `git show`, so the tools work at any
checked-out commit.

## Commit ids

| id | automatic step (run.py apply) |
|----|-------------------------------|
| C0 | none (replay copies tools/refactor into the new branch) |
| C1 | none: preparation on the old layout is done by fixups |
| C2 | `cellacdc/_compat.py` from `acdc_refactor/templates/_compat.py`, empty `cellacdc/_compat_moves.py` |
| C3 | whole-module moves (`git mv`), package `__init__`s, alias/split shims, relative imports of moved files |
| C4 | split of gui_viewer_tools, gui_core, gui_combine, gui_utils, whitelist |
| C5-C8 | widgets.py, apps.py, workers.py, myutils.py (C8 also moves myutils_models.py and _torch_utils.py into myutils/) |
| C9 | measurements viewer dialogs |
| C10, C11 | guiWin and MeasurementsViewerWin (+ MeasurementsViewerUtils) into mixins |
| C12 | `acdc_refactor/rewrite.py --notebooks --drop-unused-imports --external-origins` over cellacdc/, tests/ (hidden `.test_*.py` included) and the code cells of notebooks/, then the intermediate compat blocks are removed; report `out/C12/rewrite_report.json`. Fixup 11 then repairs `cellacdc/_compat_moves.py` for the rewritten tree |
| C13 | none: enforcement (new tests, related_tests.py, the pytest filter that turns MovedWarning into an error) by fixups |
| C14 | module docstrings of `modules` entries with `doc_commit: "C14"`; docstrings, comments and user strings by fixups |
| C15 | none: docs (CODEMAP.md, CLAUDE.md, rst docs, code_structure.md) by fixups |
| C16 | `.git-blame-ignore-revs` (after the C16 fixups) listing C12 and C14 with their final SHAs: replay passes them, `apply C16` takes `ACDC_REFACTOR_SHAS` or finds the commits in HEAD's history by the subjects of `commit_messages/` |
| C17 | none: the import hook of `_compat.py` replaces the shim files at old paths (fixup C17/01 applies `fixups/C17/import_hook.patch`) |

Every C3-C11 step regenerates `cellacdc/_compat_moves.py` for the state
after that commit.

## Mapping format

Shards are JSON files (UTF-8, 2-space indent) in `mapping/`. Paths are
POSIX paths from the repository root; item and class keys always use the
file path at the base commit, the tool follows module moves of earlier
commits.

```jsonc
{
  "shard": "widgets",
  "default_commit": "C5",               // commit of entries without one
  "module_moves": [                     // shard "moves" only: whole-file git mv
    {"old": "cellacdc/gui_navigation.py", "new": "cellacdc/gui/mixins/navigation.py",
     "shim": "alias", "commit": "C3"}  // shim: alias | split | none | package
  ],                                    // a folder move uses folder paths
  "shims": {"cellacdc/apps.py": {"kind": "split", "commit": "C6"}},
  "packages": {"cellacdc/gui/mixins": {"doc": "...", "commit": "C3",
                "init_code": "optional extra lines of the __init__",
                "forward": "optional old module whose names it forwards"}},
  "modules": {"cellacdc/shared/dialogs/generic.py": {"doc": "...", "alias": "dialogs_generic",
               "code": "optional: create a module without items (with commit)",
               "commit": "C3", "doc_commit": "C14", "forward": "cellacdc/widgets.py"}},
  "items": {
    "cellacdc/apps.py::QDialogCombobox": "cellacdc/shared/dialogs/generic.py",
    "cellacdc/gui_utils.py::nearest_ID_to_centroid": {"to": "cellacdc/whitelist.py", "commit": "C4"},
    "cellacdc/apps.py::stmt:try:": "DELETE"
  },
  "classes": {
    "cellacdc/gui.py::guiWin": {
      "host": "cellacdc/gui/main_window/window.py",
      "mixins": {"cellacdc/gui/main_window/layout.py::LayoutGuiElements": {"doc": "...", "commit": "C10"}},
      "methods": {"gui_createPlots": "cellacdc/gui/main_window/layout.py::LayoutGuiElements"},
      "bases": ["QMainWindow", "cellacdc/gui/main_window/layout.py::LayoutGuiElements", "..."]
    }
  },
  "plugin_api": ["check_install_package", "..."],   // or {"<old file>": [names]}
  "package_compat": {"cellacdc/gui/__init__.py": "cellacdc/gui.py"},
  "attribute_forwarders": {"cellacdc/gui/widgets/__init__.py":
                           {"old": "cellacdc/gui.py::widgets", "table": "cellacdc/widgets.py", "commit": "C5"}},
  "allow": [{"check": "globals", "match": "_font", "reason": "C10 fixup writes constants._font"}],
  "notes": ["free text"]
}
```

Item keys (`<old file>::<key>`):

- def / class: its name; assignment or annotated assignment: the first
  target name (`cmaps` for `cmaps, Gradients = addGradients()`);
- any other statement: `stmt:` + its first source line stripped
  (`stmt:renamePgCmaps()`, `stmt:try:`, `stmt:if __name__ == '__main__':`);
- the module docstring: `__doc__` (or its `stmt:` key);
- repeats of a key, in source order: `<key> #2`, `<key> #3` (also for names
  defined twice);
- imports are never items (imports are recomputed), except an import with
  side effects that must move as text: `import:` + first line;
- top-level `if`/`try`/`with` blocks that contain only imports are
  *import blocks*: optional items (not needed for coverage). Mapped, they
  move as text; their imports stay available to every destination (copied
  with the reconstructed guard) unless the block is `DELETE`d.
- `"DELETE"` removes dead code (logged).

Coverage: every non-import, non-docstring top-level statement of a module
that disappears as code (split shim without module move, or replaced by a
same-name package) must be mapped. A module moved whole keeps every item
that is not listed. Class splits move the listed methods (all definitions
of a name, so property getter/setter pairs stay together); unlisted
methods and every non-method statement of the class body stay in the
host. `bases` is the complete new bases list (`"QMainWindow"` or
`"<new file>::<Class>"`).

The canonical alias of a module (`modules[...].alias`) names it wherever
moved code reaches it (`widgets_message_box.myMessageBox`). A module
without alias gets a derived one (`shared/widgets/x.py` ->
`widgets_x`, `gui/main_window/x.py` -> `gui_main_window_x`, root modules
their name) and a plan_check warning. Package docstrings name files with
slash paths (`gui/mixins/navigation.py`), never dotted cellacdc names
(tests/related_tests.py scans those).

## The name table

`mapping.py name-table [--commit C]` (default: final state) writes the
contract with `acdc_refactor/rewrite.py` and the verify scripts:

- `modules`: whole-module moves, old dotted -> new dotted;
- `names`: per old module, every top-level name that the module the old
  dotted name now resolves to no longer defines, mapped to its home: the
  defining module for cellacdc names (re-exports are followed), the new
  module that binds it for Qt/third-party names;
- `entries`: the same with `[home, attribute]` (attribute `null`: the name
  is the module itself, listed in `module_values`);
- `runtime`: what `_compat_moves.NAME_MOVES` uses: import-bound names
  resolve through the module that binds them now, so per-module patches
  (conftest `IsolatedQSettings`, test patches) stay visible;
- `intermediate`: modules at new paths that lose names in a later commit
  (compat blocks until C12);
- `reexports` (rewrite view only, never in the compat tables): names that
  an old module bound by importing them from cellacdc and that `names`
  leaves out: every cellacdc import of a module moved whole
  (`gui_custom_annotations.apps` is the old `cellacdc.apps`) and the old
  globals of `gui`/`measurements_viewer` that clash with a new submodule
  (`measurements_viewer.widgets` was the old `cellacdc.widgets`; the
  literal path `cellacdc.measurements_viewer.widgets` is the new module).
  With this key the rewriter never guesses (no `reexport_guessed`);
- `fanout`: old module -> the new modules holding its code (the compat
  layer's `PATCH_FANOUT`: where a patch of an old global must land);
- `defined`: the names each old module defined at the base (a patch of a
  re-exported name never lands in its defining module);
- `aliases`, `packages`, `plugin_api`, `classes`, `shims`.

## What `move.py` does

For one commit id, in a single process:

1. **Module moves**: `git mv` of files and folders (parents created), the
   package `__init__.py` files of the commit (docstring, `init_code`,
   `package_compat` for packages that replace an old module), `modules`
   entries with `code`. Every from-import of a moved file (also inside
   functions) is recomputed: relative inside the destination part
   (`cellacdc/<part>/`), absolute outside it, so `from . import widgets`
   keeps meaning `cellacdc.widgets`. Imports of modules moved with an
   alias shim (or none) use the new path with `as <old name>`; imports of
   modules with a split shim keep the old path (the shim keeps working
   while later commits move names out).
2. **Items and class splits**: items are located in their current module
   (by key, verified against the base text) and cut with their decorators,
   comment block and trailing comments, byte-exact (each line keeps its
   EOL: CRLF working trees stay CRLF; encoding and BOM are kept). They are
   appended to their destination in source order (an existing destination
   gets them before a trailing `if __name__ == '__main__':`). Free global
   names of the moved code (symtable, scope-aware; decorators, defaults,
   bases, evaluated annotations and class bodies included) are resolved
   against the source module:
   - bound by an import: the import statement is copied (filtered to the
     needed names, in source order, retargeted for the new location, with
     the reconstructed `if`/`try` guard and the guard's own names);
     `pg.setConfigOption(...)` after `import pyqtgraph as pg` is copied
     into every new module that needs `pg` (logged `companion-copy`);
   - defined by an item that lives elsewhere after the commit:
     `Name` -> `alias.Name` with `from <pkg> import <module> as <alias>`;
     names that end up in the same module stay bare;
   - references in the code that stays in the source to names that left
     are qualified the same way.
   Mixins are created as `class <Mixin>:` + docstring with the methods
   byte-exact; the host class header gets the `bases` list (mixins
   imported by name; existing base expressions are kept). A destination
   that from-imports a name it now defines loses that import.
   Module-level assignments arriving where the same name is already bound
   by the same statement text (byte-identical copies in two old modules,
   e.g. apps.py's and widgets.py's `TREEWIDGET_STYLESHEET`) are dropped
   (`merged-duplicate`). A moved module docstring (`__doc__` item) is
   merged into the destination's docstring, moved text first
   (`docstring-merged`; the viewer spec into `measurements_viewer/__init__`).
   After the step, `run.py` adds the `doc` of `modules` entries whose
   `doc_commit` is this commit to modules that have no docstring.
3. **Shims and compat** (`shims.py`): shims at old paths, `__getattr__`
   compat blocks in modules that lost names, forwarders, and
   `cellacdc/_compat_moves.py`.

Edits other than moving text, import statements, qualification and import
depth are logged in `out/<commit>/changes.jsonl` (`category: "other"`),
together with all the others (`move`, `import`, `qualify`, `depth`);
`summary.json` counts them. The tool never rewraps lines (lines longer than
127 characters after a qualification are logged as `long-line`) and never
removes imports that became unused in a source module (the C12 rewrite
drops those of old modules and moved names).

## What `rewrite.py` does in C12

The module docstring has the rules; in short, file by file and
scope-aware: imports of old modules become imports of the modules that
define the names now (canonical aliases, attribute access), `old.X`
becomes `<home alias>.X`, `from old import X` is split per home, dotted
`'cellacdc.<old>.X'` strings are rewritten and patch targets go to the
modules whose code looks the name up. In C12 run.py also passes:

- `--notebooks`: the code cells of `*.ipynb` are rewritten as one module
  (cells joined in order, IPython magics masked); the JSON is written back
  in its format (indent 1, key order, line endings) only if it
  round-trips;
- `--drop-unused-imports`: an import of an old module, or of a name
  imported from one, without any use left is removed (moved code no
  longer needs it, e.g. `workers` in the main window module; also an old
  module that still exists, such as whitelist.py, once every use names
  another module; imports that had no use before are left alone). Kept:
  `noqa` lines, `__all__` entries, names in string annotations,
  module-level bindings that another rewritten file reads through the
  module (use index over all files; reads through an old module are
  redirected by the table and do not count) and, in tests, imports of
  modules moved whole (imported for their side effects). The report's
  `dropped` lists them;
- `--external-origins`: third-party names read through a cellacdc module
  (`measurements_viewer.QFileDialog`, `widgets.QStandardItem`,
  `myutils.subprocess`) are imported from their origin (qtpy, the stdlib),
  so a shared module never imports a program part for a Qt name.

Also always (with a source root):

- modules that stay where they are but bound an old module that other
  files read through them (`model_execution.myutils.X` in a test) are
  followed like old modules (`find_reexports`; the report's `reexports`);
- patch lookups: the modules holding the old module's code (`fanout`) that
  bind the name and read it, plus its home when the old module defined it
  (`defined`). Several lookup modules are narrowed to those the enclosing
  test function uses (a window class counts as all its mixin modules),
  else the statement is repeated per module under the comment line
  `# moved: x is read in each of these modules` (`setattr(m, 'x',
  <expression>)`: the copies set `<first module>.x`, so every module gets
  the same object; assignments `old.x = v` are repeated too);
  `for name in ('a', 'b'): setattr(old, name, ...)` is retargeted when all
  names have the same lookup module; `mod = <old split module>` in a
  function goes and its `mod.X` uses name the home modules;
- Python code in strings of test files (scripts run in a subprocess,
  `textwrap.dedent`-ed or not) is rewritten like a module, unless it
  checks the layout (`__path__`: the old branch is meant); an unused
  `import cellacdc.<moved whole>` becomes the import of the new path.

Reported, not rewritten (fixups/C12 handles the cases of this
repository): old split modules used as values (`getattr(widgets, name)`,
`(module, 'name')` pairs), dynamic patches, string annotations naming old
aliases.

## The compat layer

`templates/_compat.py` becomes `cellacdc/_compat.py` in C2 (stdlib only,
Python 3.10):

- `alias_module(__name__)` in a 1:1 shim: `sys.modules[old]` becomes the
  module at the new path (identity, patches reach the code);
- `__getattr__, __dir__ = moved_module(__name__)` in split shims and in
  modules that lost names (compat blocks); lazy, never cached, resolved
  through the home module's binding; unknown names and dunders (except
  `__all__`) raise `AttributeError` without importing anything;
- `package_compat(__name__)` in `gui/`, `measurements_viewer/`, `myutils/`:
  submodules imported on demand, plugin API names without warning;
- `forward_module(__name__, old)` for modules that shadow an old attribute
  path (`cellacdc.gui.widgets.X`);
- `CompatModule.__setattr__/__delattr__` forward patches of moved names to
  the home module (`monkeypatch.setattr`, `mock.patch` round trips work)
  and to the same object bound in every other module that holds code of
  the old module (`_compat_moves.PATCH_FANOUT`): the old module had one
  global, the codemod copies import-bound names into each destination, so
  `monkeypatch.setattr(measurements_viewer, 'get_channels_and_segm_endnames',
  ...)` still reaches the mixin modules after C11. Every set through an
  old path pushes the bindings it replaced on a per-name undo stack;
  deleting a name while such a patch is pending restores them (pytest >=
  9.2 undoes `monkeypatch.setattr` of a name missing from the module's
  `__dict__`, as moved names always are, with `delattr`; `mock.patch`
  undoes non-local patches the same way), setting the replaced value again
  pops the patch (older pytest), and only a delete without a pending patch
  deletes;
- `MovedWarning(DeprecationWarning)`, message prefix
  `Cell-ACDC import path moved: `, reported at the first frame outside
  importlib, `_compat` and the shims (`warnings.warn_explicit`).

Compat blocks in modules that are not old paths are marked
`# cellacdc-refactor: intermediate compat block (removed in C12)`; run.py
removes them after the C12 rewrite (`shims.remove_intermediate_blocks`).

## plan_check

`python -m acdc_refactor.plan_check [--quick] [--info]` (from
tools/refactor, or `run.py check <commit>`) reports, with a non-zero exit
code on errors:

- schema and duplicates, unknown keys, coverage of fully split modules;
- destinations at old paths, folders without package entries, gitignored
  new paths, module/package dotted-name collisions;
- alias uniqueness, aliases that are top-level names of their importing
  module or local names where a qualified reference lands (scope-aware);
- eager import cycles of the final layout (an import-time use of a module
  in the same import cycle, with the cycle path; the host's mixin bases
  count as eager; cycles of unchanged base modules are not reported);
- methods moved into mixins that are Qt names (`qt_names.json`, PyQt5 and
  PyQt6), use zero-argument `super()`, `__class__`, name mangling or
  `global`, use class-scope names in decorators/defaults, are used by
  class-body statements, or would stop overriding a base; bases lists that
  drop a base or miss a mixin;
- `global X` writers that end up apart from X or its readers;
- name-table homes missing from the final layout;
- warnings: derived aliases, module sizes outside 150-3000 lines, plugin API
  names that do not exist, identical duplicates.

Known, intended violations go into a shard's `allow` list (`check`,
`match` = substring of the message, `reason`); they become warnings.
`--update-qt-names` adds the installed Qt binding to `qt_names.json`.

## run.py replay and fixups

`replay` creates a worktree from `--base` on a new branch and, for each
commit up to `--until`: the automatic step, the fixups, `git add -A`, a
commit (message from `commit_messages/<commit>.txt` or a default; empty
steps are skipped). The tools, mapping and fixups always come from this
checkout; C0 copies them into the new branch. The resulting SHAs are in
`out/replay_shas.json`.

A fixup is a Python script `fixups/<commit>/<nn>_<what>.py`, run as
`python <script> <repo root>` with the repository as working directory and
`ACDC_REFACTOR_COMMIT`, `ACDC_REFACTOR_REPO`, `ACDC_REFACTOR_BASE`,
`ACDC_REFACTOR_SHAS` (JSON {commit: sha} so far) set. Scripts run in name
order; a non-zero exit stops the pipeline. Keep them idempotent and
specific (replace exact text, fail loudly when the text is not found), so
that a rebuild on a newer main reports what changed. Fixups of C1-C11
are behaviour-preserving edits the codemod cannot do (C1 preparation,
`_font` -> `constants._font`, `__file__` paths, ...). `fixups/_fixuplib.py`
has the helpers (`replace`, `ensure_line_after`) and `log_change`, which
appends the edited items to `out/<commit>/changes.jsonl` so that
`verify/fingerprint.py` excuses them. `ACDC_REFACTOR_FIXUPS` points
run.py to another fixups folder (the unit tests use an empty one).

Fixups so far:

| commit | script | edit |
|--------|--------|------|
| C1 | 01_file_relative_paths | `__file__`-relative paths of dataStruct.py, apps.py (2 tracker dialogs), myutils.read_version and the viewer's `PREMADE_CONFIGURATIONS_PATH` -> `cellacdc_path` |
| C1 | 02_drop_sys_path_append | `sys.path.append(cellacdc_path)` of utils/convert.py and utils/rename.py |
| C1 | 03_print_call_stack | `myutils.print_call_stack()` (never existed) -> `debugutils` in plot.py and whitelist.py |
| C1 | 04_colors_unused_imshow | unused `from cellacdc.plot import imshow` in colors.py (cycle colors -> plot -> ... -> html_utils -> colors broke `import cellacdc` after C8) |
| C1 | 05_regionprops_explicit_import | `import skimage.measure._regionprops` in regionprops.py (it read the lazily loaded private submodule; failed without Qt after C8) |
| C1 | 06_tests_docs_paths | three tests read the Sphinx docs relative to `cellacdc_path` instead of the folder of `_main`/`measurements_viewer` |
| C1 | 07_shortcut_filepath | `shortcut_filepath` defined once in `cellacdc/__init__.py`; gui.py and gui_viewer_tools.py bind it from there, the launcher reads it from cellacdc |
| C1 | 08_conftest_settings_isolation | tests/conftest.py settings isolation reads bindings with `vars()`, redirects modules first imported during a test, adds `custom_cmaps_filepath`; test_conftest_isolation checks the globals where the code reads them |
| C1 | 09_check_unused_methods_mro | `debugutils.check_unused_methods` reads the sources of the classes of the guiWin MRO instead of parsing gui.py |
| C1 | 10_datarestruct_type_checking | dataReStruct.py imports the launcher (`_main`) only under `TYPE_CHECKING` (string annotation of `run`) |
| C1 | 11_source_scan_tests | the three slow source scans of the main GUI (test_channel_luts, test_linked_right_image_layers, test_list_selection_required) find the code wherever it lives |
| C1 | 12_segm_dead_imshow_tk | removes segm.py's dead `segmWin.debugSegmWorker` (called the non-existent `apps.imshow_tk`) |
| C1 | 13_unused_imports | unused imports of cellacdc modules: `gui` in utils/base.py and utils/acdcToSymDiv.py, `apps` in the trackpy tracker, `myutils` in DeepSea/BABY and html_utils.py |
| C1 | 14_datastruct_bioformats_import | dataStruct.py `readMetadata` imports `cellacdc.bioformats` (was the non-existent `cellacdc.bioformat`) |
| C1 | 15_volume_renderer_guard | the optional volume renderer import of `cellacdc/__init__.py` re-raises ModuleNotFoundError of cellacdc modules |
| C1 | 16_related_tests_module_path | test_related_tests' repository selection takes the phase translator module from the imports of its test file (no hard-coded path) |
| C3 | 01_premade_configurations_path | the viewer's premade folder moved to `measurements_viewer/premade_configurations` |
| C3 | 02_welcome_lazy_guiwin | help/welcome.py imports the gui window inside the two methods that create it (synthesis R6) |
| C5 | 01_buttons_source_path | `load.get_all_buttons_names` reads `shared/widgets/push_buttons.py` |
| C6 | 01_test_segm_model_dialog_first | `TestSegmModelInitalDialog` placed after the dev script's imports (it is used at module level) |
| C8 | 01_torch_utils_import_test | `test_torch_utils_import_does_not_load_gui_or_torch` allows the lazy `cellacdc.myutils` package and blocks its other submodules |
| C8 | 02_model_env_layout_check | subprocess templates run in model environments check `hasattr(cellacdc.myutils, "__path__")` and work with both myutils layouts |
| C8 | 03_remove_leftover_myutils_module | deletes the leftover cellacdc/myutils.py (imports and compat block) next to the myutils/ package |
| C10 | 01_font_via_constants | `resizeSlidersArea` writes `gui_main_constants._font` (no `global`) |
| C12 | 01_string_annotations | the three string annotations naming `widgets.` (load.py, shared/dialogs/preprocessing.py, set_measurements.py) point at the new homes (flake8 F821 after the rewrite) |
| C12 | 02_sam_exec_namespaces | the SAM/SAM2 embedding tests exec the plugin classes in a namespace with `image_utils` (what the plugins import now) instead of `myutils` |
| C12 | 03_gpu_availability_widget_modules | test_directml_install_uses_target_python_and_version swaps the `widgets_message_box`/`widgets_push_buttons` globals of myutils/model_installs.py (was `myutils.widgets`) |
| C12 | 04_keep_ids_widget_classes | test_keep_ids builds its parametrized widgets from a `WIDGET_CLASSES` table of the new homes (was `getattr(widgets, cls)`; ids unchanged) |
| C12 | 05_torch_version_legacy_test | test_legacy_imports_reexport_torch_helpers compares the myutils facade with myutils/torch_utils.py (not the `_torch_utils` alias shim) |
| C12 | 06_timepoint_identifiers_constants_dialog | test_constants_are_disabled checks the equation dialog modules for `CellConstantsDialog` (was `hasattr(apps, ...)`) |
| C12 | 07_view_all_buttons_push_buttons | the dev script _view_all_buttons.py reads the button classes from shared/widgets/push_buttons.py (was `getattr(widgets, name)`) |
| C12 | 08_headless_server_script | test_headless_server_config_import: the Qt-blocked script imports every myutils module, checks `segm_models.config` and that no gui/, shared/widgets/ or shared/dialogs/ module is loaded (was myutils, widgets, apps) |
| C12 | 09_torch_utils_import_script | test_torch_utils_import_does_not_load_gui_or_torch blocks the gui/, shared/widgets/ and shared/dialogs/ modules (was the old widgets and apps) |
| C12 | 10_timepoint_identifiers_long_line | wraps the 142-character loop tuple of test_viewer_deletes_identifier_and_condition_dialogs that the qualification produced (the rewrite's only remaining `long_lines` entry) |
| C12 | 11_compat_tables_after_rewrite | `_compat_moves.py` for the tree after the rewrite: the entries of the 5 intermediate modules go (their compat blocks were removed), the names whose binding import `--drop-unused-imports` removed (about 75: `measurements_viewer.load_position`, `gui.setupLogger`, `apps.widgets`, ...) point to their defining module (name table `entries`), `except ... as e` names go; without it these old paths raise AttributeError from C12 on (tests/test_compat.py, C13) |
| C13 | 01_new_tests | adds the layout tests from fixups/C13/files (test_compat, test_no_old_paths, test_import_paths, test_import_layering, window layout tests, test_package_init_files, read_version and plot.imshow tests) |
| C13 | 02_related_tests | replaces tests/related_tests.py and tests/test_related_tests.py (compat shims are no importers, importers counted per program part) |
| C13 | 03_pytest_old_paths_error | pyproject.toml: `error:Cell-ACDC import path moved:DeprecationWarning` after `ignore::DeprecationWarning` |
| C13 | 04_viewer_main_test | test that `python -m cellacdc.measurements_viewer` runs the viewer window (tests/test_launcher_measurements_viewer.py) |
| C14 | 01_old_module_mentions | docstrings, comments and user-visible strings of cellacdc/ that name old modules (the rewrite report's `mentions`) |
| C14 | 02_tests_conftest_mentions | the docstring and comments of tests/conftest.py name `app_info.setupLogger` and `segm_models.get_list_of_promptable_models` (were `myutils.`) |
| C15 | 01_codemap | CODEMAP.md (template + tools/codemap.py), tools/codemap.py and tests/test_codemap.py |
| C15 | 02_claude_md | CLAUDE.md for the new layout (three-way merge of fixups/C15/files/CLAUDE.md) |
| C15 | 03_docs | model-environments.rst, models.rst, contributing.rst and cellacdc/code_structure.md name the new modules |
| C17 | 01_import_hook | applies `fixups/C17/import_hook.patch` (`git apply --index --3way`): deletes the 22 shim files at old paths, `_compat.OldPathFinder` (installed by `cellacdc/__init__.py`, table `OLD_MODULES`) serves those paths; tests and docs. Regenerate the patch from the C17 commit with `git diff <C16> <C17> --binary -- . ":(exclude)tools" > tools/refactor/fixups/C17/import_hook.patch` |

To rebuild the PR on a newer main: update the base (`--base`), run
`mapping.py validate` (new or renamed items of main show up as coverage or
key errors: map them), `run.py replay`, then the checks of every commit.

## Status (integration run of 2026-11-28)

`run.py replay --base bac8ae07 --until C16` applies the whole chain
(branch `tmp/integrate3`). Gates (Windows, Python 3.13, PyQt6, pytest
9.0), run on a replay whose commits C1-C16 have the same trees outside
tools/refactor:

| commits | checks | result |
|---------|--------|--------|
| C1-C16 | flake8 E9/F63/F7/F82 of cellacdc, tests, tools; `import cellacdc` and the window modules (launcher, guiWin, MeasurementsViewerWin, data prep, data structure, segmentation module, cli) in fresh interpreters, from C12 on also with the MovedWarning as an error | all pass |
| C1-C16 | `run.py check`: plan_check (0 errors), fingerprint (0 changed/lost/new), MRO snapshot (0 errors; the warnings are `body_changed` of qualified references), headless | all pass (C14 after fixup C14/01 logs the items it edits) |
| C1, C3, C8, C11, C12 | push tier | 2700 passed each |
| C13 | push tier | 2758 passed |
| C16 | full suite | 3510 passed, 48 skipped (base: 3438 passed, 48 skipped) |
| C16 | push tier on Python 3.11, PyQt5, pytest 9.2 | 2758 passed, 51 skipped (base, same environment: 2696 passed, 51 skipped); `-m infra` (slow included): 126 passed, 5 skipped |
| C16 | every file of cellacdc, tests, tools compiled by Python 3.11 and parsed with the 3.10 grammar | no error |
| C16 | verify compares with `out/baseline`: import_all (gui, noqt), headless, collect_ids, importtime, smoke | all ok: no module fails that imported at the base (gui: the same 9 failures; noqt: the new failures are modules that need Qt), 0 lost test ids (73 new), import times within the tolerance, smoke failures only for dialogs that need arguments and are found now because they moved into utils/ |
| C16 | `python tests/related_tests.py --base 4c4e498b` | `tests` (`cellacdc/__init__.py` changed) |

pytest 9.2 restores a `monkeypatch.setattr` of a name missing from the
module `__dict__` with `delattr`; before the patch-undo stack of
`CompatModule` (see "The compat layer") this deleted moved names at their
home through old paths (tests/test_compat.py failed under pytest 9.2).

Remaining issues (none blocks the PR):

- The C12 rewrite keeps the old local names of imports that existed
  before the refactor (`... as dataPrep`, `as segm`, `as _main`,
  `as dataStruct`, `as dataReStruct`, `as myutils_models`); CODEMAP.md
  documents them. Renaming them is a follow-up (code, not a mention).
- `port.py` (plan) does not exist: open branches are ported by hand with
  the steps of "Porting an open branch".
- Not run here: Python 3.10, PyQt5 on macOS/Linux, the real app with
  the `run` skill, `wheel_diff` and the spotMAX check of the plan's
  "Verification" list; CI does not run for PRs stacked on another branch.
- Pre-existing at the base, unchanged: `cellacdc/record.py` uses a
  relative import beyond the top-level package (never imported), and
  `_deprecated/filters.py` has relative imports that only resolve from
  `cellacdc/` itself (dead code, excluded from the import sweeps); the
  rewriter reports both as `unresolved_relative_import`.

## Porting an open branch (started on the old layout)

1. Merge main up to the refactor's parent into the branch (resolve as
   usual).
2. Merge the refactor. Files moved whole follow by rename detection (also
   use `git merge -X find-renames=50%` if a rename is missed): since C17
   no file stays at their old paths, so the branch's edits land in the new
   files. Split modules (apps.py, widgets.py, workers.py, myutils.py, ...)
   are deleted: keep the deletion (no file may come back at an old path,
   `tests/test_no_old_paths.py`) and keep the branch's changes aside.
3. For every item the branch changed in a split module (diff the old
   module between the merge base and the branch tip, item by item:
   `items.module_items` gives the spans), find its new home with
   `python -c "from cellacdc._compat_moves import NAME_MOVES as M;
   print(M['cellacdc.widgets']['QDialogListbox'])"` (or the name table), and
   apply the branch's change to the item there. New items of the branch:
   add them to the matching module (CODEMAP.md, placement rules).
4. Rewrite the branch's references to old paths with
   `python tools/refactor/acdc_refactor/rewrite.py --table
   tools/refactor/out/name_table.json --src-root . [--notebooks]
   [--drop-unused-imports] <changed files>`; tests/test_no_old_paths.py
   enforces it. Review the report's `unhandled` entries (section "What
   `rewrite.py` does in C12").
5. Compare every touched item with the `ast` script of CLAUDE.md (watch for
   the trap where taking one side of a hunk drops the last lines of the
   preceding method).

Open branches at the time of writing: `feat/acdc-data-cli` (apps,
dataStruct, dataReStruct), `channel_lut_settings` (gui, gui_overlay,
gui_viewer_tools, measurements_viewer, widgets), `fix_font_size_menu`,
`workflow-gui`. Merging them into main before the refactor lands is
cheaper: replay then absorbs them.

## Limitations

- Until C16 the old paths are shim files (C3 creates them in the same
  commit as the moves), so the single commits C3-C16 show the moves as
  copies (`git diff -B -M --summary`, `git log --follow -B`, `git blame
  -C -C` follow them). C17 deletes the shims (import hook instead), so the
  whole refactor (base..C17), which merges and the pull request show, has
  the 1:1 moves as renames.
- Rewritten import lines are regenerated (`from a import (x, y)` style)
  when a statement must be split; comments inside such statements are
  lost (logged). No line is rewrapped.
- The import-cycle analysis is static and models the final layout from
  the base code: import-order effects of intermediate commits (a module
  that used to be imported first becoming lazy, e.g. the myutils facade)
  and dynamic imports are only caught by `verify/import_all.py`.
- `global` writers, `__file__`-relative paths, string references to module
  names and `getattr(module, name)` are not rewritten (plan_check / the
  rewrite report list them; fixups handle them).
- Names bound only dynamically (star imports, `globals()[...] =`) are
  logged as `unresolved-global` and left as they are.
- A copied import guard whose condition uses a name defined in the source
  module (not imported) is an error: copy it by hand.
