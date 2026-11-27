"""Shared helpers of the refactor verification scripts (tools/refactor/verify).

- Paths: the checkout holding these tools (``REPO_ROOT``), the output folder
  ``tools/refactor/out/<commit>/`` and the mapping shards
  ``tools/refactor/mapping/*.json``.
- The one exclusion list of the import sweeps (``EXCLUDED_IMPORT_PREFIXES``,
  ``PLUGIN_PACKAGES``, ``is_import_excluded``): dev scripts, docs, generated
  resources, plugin packages and modules that run code at import.
- Listing a tree's modules statically (git-tracked plus untracked, not ignored
  files), without importing anything.
- The name table (``mapping.py name-table``) and the shards, wrapped by
  ``Mapping``: it maps old module/name/method keys of the base layout to their
  canonical (new) keys, so that snapshots taken before and after a refactor
  step compare item by item.
- Child-process helpers: running a command with a hard timeout, a small
  parallel pool, and the settings isolation used by the scripts that construct
  windows (``install_settings_isolation``, the same redirections as
  ``tests/conftest.py`` ``_isolate_user_settings`` plus every module-level path
  inside the user's settings folder, also for modules imported later).

Every script here parses on Python 3.10 and uses only the standard library.
"""
import concurrent.futures
import datetime
import json
import os
import platform
import re
import subprocess
import sys
import tempfile

VERIFY_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(VERIFY_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(TOOLS_DIR))
OUT_DIR = os.environ.get('ACDC_REFACTOR_OUT') or os.path.join(TOOLS_DIR, 'out')
MAPPING_DIR = os.path.join(TOOLS_DIR, 'mapping')
PACKAGE = 'cellacdc'

# --- Import sweep exclusions (the ONE list of the plan) ---------------------

# Module name prefixes never imported by the sweeps (a prefix matches the
# module itself and its submodules; ``cellacdc.qrc_resources`` also matches
# ``qrc_resources_dark``/``_light``)
EXCLUDED_IMPORT_PREFIXES = (
    # Generated Qt resources (~340k lines; registering them has side effects)
    'cellacdc.qrc_resources',
    # Sphinx docs, scripts plotting citations
    'cellacdc.docs',
    # resources/to_dark_mode_svg.py runs inkscape at import
    'cellacdc.resources',
    # Dev-only profiling scripts and deprecated code
    'cellacdc._profile',
    'cellacdc._deprecated',
    # Dev scripts that open windows at import
    'cellacdc._view_all_buttons',
    'cellacdc._view_all_icons',
    'cellacdc._get_app_palette',
    'cellacdc._q_debug',
    'cellacdc.test_segm_model',
    'cellacdc.test_tracker',
    'cellacdc.scripts',
    'cellacdc.volume_renderer._demos',
    # Starts the Java VM at import
    'cellacdc.bioformats.tests',
    # Command line scripts run in subprocesses (argparse at import)
    'cellacdc._process',
    'cellacdc.acdc_bioio_bioformats._init_reader',
    'cellacdc.acdc_bioio_bioformats._read_metadata',
    'cellacdc.acdc_bioio_bioformats._read_sample_data',
    'cellacdc.acdc_bioio_bioformats._save_data',
    'cellacdc.acdc_bioio_bioformats._save_data_single_channel',
)

# Plugin packages: their sub-packages (one folder per model/tracker/processor)
# import optional third-party libraries and are never imported by the sweeps;
# the package itself and its plain modules (e.g. models/_env_config.py) are.
PLUGIN_PACKAGES = (
    'cellacdc.models',
    'cellacdc.trackers',
    'cellacdc.promptable_models',
    'cellacdc.measurement_processing',
)

# Imported by default as "entry modules" (their sys.modules sets are compared)
# as (label, [candidate module names, new layout first])
ENTRY_MODULES = (
    ('cellacdc', ['cellacdc']),
    ('launcher', ['cellacdc.launcher.main_window', 'cellacdc._main']),
    ('gui_window', ['cellacdc.gui.main_window.window', 'cellacdc.gui']),
    ('viewer_window', [
        'cellacdc.measurements_viewer.main_window.window',
        'cellacdc.measurements_viewer',
    ]),
    ('data_prep', ['cellacdc.data_prep.main_window', 'cellacdc.dataPrep']),
    ('data_struct', [
        'cellacdc.data_structure.data_struct', 'cellacdc.dataStruct'
    ]),
    ('segm', ['cellacdc.segmentation_module.main_window', 'cellacdc.segm']),
    ('cli', ['cellacdc.cli']),
    ('utils_base', ['cellacdc.utils.base']),
    ('service_worker', ['cellacdc.service.worker']),
    ('lineage_tree_integrity', ['cellacdc.lineage_tree_integrity']),
)

# Window classes (label -> candidate 'module:Class', new layout first)
WINDOW_CLASSES = {
    'guiWin': [
        'cellacdc.gui.main_window.window:guiWin', 'cellacdc.gui:guiWin'
    ],
    'MeasurementsViewerWin': [
        'cellacdc.measurements_viewer.main_window.window:MeasurementsViewerWin',
        'cellacdc.measurements_viewer:MeasurementsViewerWin',
    ],
    'mainWin': [
        'cellacdc.launcher.main_window:mainWin', 'cellacdc._main:mainWin'
    ],
    'dataPrepWin': [
        'cellacdc.data_prep.main_window:dataPrepWin',
        'cellacdc.dataPrep:dataPrepWin',
    ],
    'createDataStructWin': [
        'cellacdc.data_structure.data_struct:createDataStructWin',
        'cellacdc.dataStruct:createDataStructWin',
    ],
    'CreateSymLinkToPosWin': [
        'cellacdc.data_structure.data_struct:CreateSymLinkToPosWin',
        'cellacdc.dataStruct:CreateSymLinkToPosWin',
    ],
    'segmWin': [
        'cellacdc.segmentation_module.main_window:segmWin',
        'cellacdc.segm:segmWin',
    ],
}

QT_BINDINGS = ('qtpy', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'sip')


def is_import_excluded(modname, is_package=False):
    """True if the import sweeps never import `modname`."""
    if modname.startswith('cellacdc.qrc_resources'):
        return True
    parts = modname.split('.')
    if parts[-1] == '__main__':
        # Runs the application (cellacdc/__main__.py deletes old installs)
        return True
    if any(not part.isidentifier() for part in parts):
        # e.g. promptable_models/micro-sam_v1: not importable by name
        return True
    for prefix in EXCLUDED_IMPORT_PREFIXES:
        if modname == prefix or modname.startswith(prefix + '.'):
            return True
    for pkg in PLUGIN_PACKAGES:
        if not modname.startswith(pkg + '.'):
            continue
        depth = len(modname.split('.')) - len(pkg.split('.'))
        if depth >= 2 or is_package:
            return True
    return False


def is_plugin_module(modname):
    """True for modules inside a plugin sub-package (models/<Name>/...)."""
    parts = modname.split('.')
    if any(not part.isidentifier() for part in parts):
        return False
    return any(
        modname.startswith(pkg + '.') for pkg in PLUGIN_PACKAGES
    )


# --- Generic I/O --------------------------------------------------------------

def write_json(path, obj, indent=1):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as file:
        json.dump(obj, file, indent=indent, sort_keys=False)
        file.write('\n')
    os.replace(tmp, path)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def out_dir(commit):
    return os.path.join(OUT_DIR, commit)


def git(repo, *args, timeout=60, check=True):
    result = subprocess.run(
        ['git', '-C', repo, *args], capture_output=True, text=True,
        timeout=timeout, encoding='utf-8', errors='replace'
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f'git {" ".join(args)} failed: {result.stderr.strip()}'
        )
    return result.stdout


def head_sha(repo):
    try:
        return git(repo, 'rev-parse', 'HEAD').strip()
    except Exception:
        return None


def is_dirty(repo):
    try:
        return bool(git(repo, 'status', '--porcelain', '--', PACKAGE).strip())
    except Exception:
        return None


def meta(repo):
    """Description of the snapshotted tree, stored in every snapshot."""
    return {
        'repo': os.path.abspath(repo),
        'head': head_sha(repo),
        'dirty_cellacdc': is_dirty(repo),
        'python': sys.version.split()[0],
        'executable': sys.executable,
        'platform': platform.platform(),
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
    }


# --- Static listing of a tree -------------------------------------------------

def list_package_files(repo):
    """Repo-relative slash paths of the ``.py`` files of the package.

    Tracked and untracked files not ignored by git (so local ignored scripts
    such as ``cellacdc/_q_debug.py`` and generated ``_version.py`` are left
    out, as in a clean worktree); a plain folder walk outside git.
    """
    try:
        out = git(
            repo, 'ls-files', '--cached', '--others', '--exclude-standard',
            '-z', '--', PACKAGE
        )
        files = [path for path in out.split('\0') if path]
        files = [
            path for path in files
            if os.path.exists(os.path.join(repo, path))
        ]
    except Exception:
        files = []
        for root, dirs, names in os.walk(os.path.join(repo, PACKAGE)):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for name in names:
                path = os.path.join(root, name)
                files.append(os.path.relpath(path, repo))
    files = [path.replace('\\', '/') for path in files]
    return sorted(
        path for path in files
        if path.endswith('.py') and '/__pycache__/' not in path
    )


def module_name(relpath):
    """Dotted module name of a repo-relative ``.py`` path."""
    parts = relpath[:-3].split('/')
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def module_path(modname, files_set):
    """Repo-relative path of `modname` in `files_set` (None if missing)."""
    base = modname.replace('.', '/')
    for candidate in (base + '.py', base + '/__init__.py'):
        if candidate in files_set:
            return candidate
    return None


def importable_modules(repo, files=None):
    """{module name: (relpath, is_package)} of the importable modules.

    A module is importable when every folder from ``cellacdc/`` down to its
    file holds an ``__init__.py`` (regular packages, as ``pkgutil``).
    """
    files = files if files is not None else list_package_files(repo)
    files_set = set(files)
    result = {}
    for path in files:
        parts = path.split('/')
        folders = ['/'.join(parts[:i]) for i in range(1, len(parts))]
        if not all(f + '/__init__.py' in files_set for f in folders):
            continue
        result[module_name(path)] = (path, path.endswith('/__init__.py'))
    return result


# --- Mapping (name table + shards) --------------------------------------------

def default_name_table(commit=None):
    """Path of the name table to use by default (None if there is none)."""
    candidates = []
    if commit:
        # the table of that commit (run.py check writes name_table.json)
        candidates += [
            os.path.join(OUT_DIR, commit, 'name-table.json'),
            os.path.join(OUT_DIR, commit, 'name_table.json'),
        ]
    candidates += [
        os.path.join(OUT_DIR, 'name-table.json'),
        os.path.join(OUT_DIR, 'name_table.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def path_to_module(path):
    path = path.replace('\\', '/')
    if path.endswith('.py'):
        return module_name(path)
    return path.rstrip('/').replace('/', '.')


class Mapping:
    """Old -> new keys from the name table and the mapping shards.

    Keys of items are ``'<module>:<Qual.name>'``. The canonical key of an
    item is its key in the target layout: top-level names follow the name
    table, methods of classes split into mixins are keyed by their host class
    (``'<host module>:<HostClass>.<method>'``), on both sides, so that a
    method moved from ``guiWin`` into a mixin keeps its key.

    `side` is ``'before'`` or ``'after'``. Module-level mapping (whole-module
    moves) applies only to modules of plain files (never to a package
    ``__init__``), because the old ``gui.py``/``measurements_viewer.py``/
    ``myutils.py`` modules become packages of the same dotted name.
    """

    def __init__(self, table=None, shards=None):
        table = table or {}
        self.table_path = None
        self.modules = dict(table.get('modules', {}))
        self.names = {
            mod: dict(names) for mod, names in table.get('names', {}).items()
        }
        self.aliases = dict(table.get('aliases', {}))
        self.packages = set(table.get('packages', []))
        # The state compared: shard entries of later commits are ignored
        # (the name table of `mapping.py name-table --commit C` records C)
        self.commit = table.get('commit')
        self.shards = shards or []
        # (new module, mixin name) -> (host class name, host module)
        self.mixin_hosts = {}
        # (old module, class name) -> host module
        self.class_hosts = {}
        # host (old module, class) -> planned bases (list of str)
        self.class_bases = {}
        # (old module, 'stmt:...') -> new module (or 'DELETE')
        self.items = {}
        self.deleted = set()
        self.shim_modules = set()
        self.new_package_modules = set()
        self.file_moves = {}
        for shard in self.shards:
            self._add_shard(shard)
        # module moves of the shards complete the table's modules
        self.folder_moves = {}
        for old, new in self.file_moves.items():
            oldmod, newmod = path_to_module(old), path_to_module(new)
            if old.endswith('.py'):
                self.modules.setdefault(oldmod, newmod)
            else:
                self.folder_moves[oldmod] = newmod

    def _applies(self, commit):
        """True when an entry of `commit` is part of the compared state."""
        if not self.commit or not commit:
            return True
        now, then = commit_index(self.commit), commit_index(commit)
        return now is None or then is None or then <= now

    def _add_shard(self, shard):
        default = shard.get('default_commit')
        for move in shard.get('module_moves', []):
            if not self._applies(move.get('commit', default)):
                continue
            old, new = move.get('old'), move.get('new')
            if old and new:
                self.file_moves[old.rstrip('/')] = new.rstrip('/')
            if old and move.get('shim', 'none') != 'none':
                self.shim_modules.add(path_to_module(old))
        for path in shard.get('shims', {}):
            self.shim_modules.add(path_to_module(path))
        for path in shard.get('packages', {}):
            self.new_package_modules.add(path_to_module(path))
        for key, value in shard.get('items', {}).items():
            if '::' not in key:
                continue
            oldfile, itemkey = key.split('::', 1)
            to = value.get('to') if isinstance(value, dict) else value
            commit = (value.get('commit') if isinstance(value, dict)
                      else None) or default
            if not self._applies(commit):
                continue
            oldmod = path_to_module(oldfile)
            if to == 'DELETE':
                self.deleted.add((oldmod, itemkey))
                self.items[(oldmod, itemkey)] = 'DELETE'
            elif to:
                self.items[(oldmod, itemkey)] = path_to_module(to)
        for key, spec in shard.get('classes', {}).items():
            if '::' not in key:
                continue
            oldfile, cls = key.split('::', 1)
            commits = [m.get('commit', default)
                       for m in (spec.get('mixins') or {}).values()
                       if isinstance(m, dict)] or [default]
            if not any(self._applies(c) for c in commits):
                continue
            oldmod = path_to_module(oldfile)
            host = spec.get('host')
            hostmod = path_to_module(host) if host else None
            if hostmod:
                self.class_hosts[(oldmod, cls)] = hostmod
            self.class_bases[(oldmod, cls)] = list(spec.get('bases', []))
            mixins = set(spec.get('mixins', {}))
            mixins.update(spec.get('methods', {}).values())
            for mixin in mixins:
                if '::' not in mixin:
                    continue
                mfile, mname = mixin.split('::', 1)
                self.mixin_hosts[(path_to_module(mfile), mname)] = (
                    cls, hostmod or path_to_module(mfile)
                )

    @classmethod
    def load(cls, table_path=None, mapping_dir=MAPPING_DIR):
        table = read_json(table_path) if table_path else None
        shards = []
        if mapping_dir and os.path.isdir(mapping_dir):
            for name in sorted(os.listdir(mapping_dir)):
                if name.endswith('.json'):
                    try:
                        shards.append(read_json(os.path.join(mapping_dir, name)))
                    except Exception as err:
                        print(
                            f'[verify] cannot read shard {name}: {err}',
                            file=sys.stderr
                        )
        mapping = cls(table, shards)
        mapping.table_path = table_path
        return mapping

    def map_module(self, mod, is_pkg_init=False):
        """New dotted name of module `mod` (longest moved prefix)."""
        if mod in self.modules:
            if is_pkg_init:
                # A package with the dotted name of an old module is new
                return mod
            return self.modules[mod]
        # Submodules follow only folder moves (an old plain module such as
        # gui.py has no submodules: cellacdc.gui.mixins is new)
        best = None
        for old in self.folder_moves:
            if mod.startswith(old + '.'):
                if best is None or len(old) > len(best):
                    best = old
        if best is None:
            return mod
        return self.folder_moves[best] + mod[len(best):]

    def map_def(self, mod, name, is_pkg_init=False):
        """(module, name) of the canonical home of top-level `name`."""
        home = self.names.get(mod, {}).get(name)
        if home:
            return home, name
        return self.map_module(mod, is_pkg_init), name

    def canonical_key(self, mod, qual, is_pkg_init=False):
        """Canonical key of item `qual` of module `mod` (either side)."""
        if qual.startswith('stmt:'):
            # repeats (' #2') are numbered per module: the canonical key
            # drops the number, fingerprint.compare pairs repeats itself
            base = re.sub(r' #\d+$', '', qual)
            target = self.items.get((mod, qual)) or self.items.get(
                (mod, base)
            )
            if target and target != 'DELETE':
                return f'{target}:{base}'
            return f'{self.map_module(mod, is_pkg_init)}:{base}'
        top, _, rest = qual.partition('.')
        top_base = top.split(' #')[0]
        host = self.mixin_hosts.get((mod, top_base))
        if host is not None and rest:
            cls, hostmod = host
            return f'{hostmod}:{cls}.{rest}'
        if rest and (mod, top_base) in self.class_hosts:
            return f'{self.class_hosts[(mod, top_base)]}:{top}.{rest}'
        newmod, _ = self.map_def(mod, top_base, is_pkg_init)
        return f'{newmod}:{qual}'

    def map_dotted(self, value):
        """Map a dotted ``cellacdc.x[.Name]`` string (module or name)."""
        if value in self.modules:
            return self.modules[value]
        mod, _, name = value.rpartition('.')
        home = self.names.get(mod, {}).get(name)
        if home:
            return f'{home}.{name}'
        return self.map_module(value)

    def is_mixin_shell(self, mod, qual):
        return '.' not in qual and (mod, qual.split(' #')[0]) in self.mixin_hosts

    def is_compat_module(self, mod, relpath=''):
        """True for compat-only modules of the target layout."""
        base = mod.split('.')[-1]
        if base in ('_compat', '_compat_moves'):
            return True
        if mod in self.shim_modules:
            return True
        if relpath.endswith('__init__.py') and mod in self.new_package_modules:
            return True
        return False


def load_changes(paths):
    """Identifiers listed in ``changes.jsonl`` files (any string value).

    Each line is a JSON object; every string value in it (recursively) is
    an identifier that excuses a difference or an unmatched item, e.g.
    ``{"item": "cellacdc/gui.py::guiWin.foo", "reason": "_font fixup"}``.
    Accepted identifier forms: ``<file>::<Qual>``, ``<module>:<Qual>``
    (old, new or canonical). Plain-text lines are taken whole.
    """
    listed = set()

    def collect(value):
        if isinstance(value, str):
            listed.add(value.strip())
        elif isinstance(value, dict):
            for v in value.values():
                collect(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                collect(v)

    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    collect(json.loads(line))
                except json.JSONDecodeError:
                    listed.add(line)
    return listed


def commit_index(commit):
    match = re.fullmatch(r'C(\d+)', commit or '')
    return int(match.group(1)) if match else None


def changes_files_up_to(commit):
    """``out/C<k>/changes.jsonl`` of every commit up to `commit`."""
    paths = []
    idx = commit_index(commit)
    if not os.path.isdir(OUT_DIR):
        return paths
    for name in sorted(os.listdir(OUT_DIR)):
        k = commit_index(name)
        if k is None or idx is None or k > idx:
            if name != commit:
                continue
        path = os.path.join(OUT_DIR, name, 'changes.jsonl')
        if os.path.exists(path):
            paths.append(path)
    return paths


# --- Subprocesses ---------------------------------------------------------

def child_env(repo, extra=None, offscreen=True):
    env = dict(os.environ)
    if offscreen:
        env['QT_QPA_PLATFORM'] = 'offscreen'
    env['PYTHONHASHSEED'] = '0'
    env['PYTHONIOENCODING'] = 'utf-8'
    # The snapshotted tree comes first, also for children run from elsewhere
    pythonpath = [os.path.abspath(repo)]
    if env.get('PYTHONPATH'):
        pythonpath.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(pythonpath)
    env.pop('PYTHONSTARTUP', None)
    if extra:
        env.update(extra)
    return env


def run_child(cmd, cwd, env, timeout):
    """Run `cmd`; returns dict(returncode, stdout, stderr, timed_out, time)."""
    import time
    start = time.perf_counter()
    kwargs = {}
    if sys.platform.startswith('win'):
        kwargs['creationflags'] = getattr(
            subprocess, 'CREATE_NEW_PROCESS_GROUP', 0
        )
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = b'', b''
    return {
        'returncode': proc.returncode,
        'stdout': stdout.decode('utf-8', errors='replace'),
        'stderr': stderr.decode('utf-8', errors='replace'),
        'timed_out': timed_out,
        'time': round(time.perf_counter() - start, 3),
    }


def kill_tree(proc):
    if sys.platform.startswith('win'):
        subprocess.run(
            ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
            capture_output=True
        )
    else:
        proc.kill()


def parallel(func, items, workers=8):
    """[func(item) for item in items] on a thread pool, in input order."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, items))


def tail(text, lines=15):
    return '\n'.join(text.strip().splitlines()[-lines:])


def scratch_root():
    """Folder for temporary files: ``$ACDC_VERIFY_SCRATCH`` or the temp dir."""
    root = os.environ.get('ACDC_VERIFY_SCRATCH') or os.path.join(
        tempfile.gettempdir(), 'acdc_refactor_verify'
    )
    os.makedirs(root, exist_ok=True)
    return root


def scratch_dir(name):
    path = os.path.join(scratch_root(), name)
    os.makedirs(path, exist_ok=True)
    return path


# --- Settings isolation (run inside child processes) -------------------------

# Same as tests/conftest.py USER_SETTINGS_PATHS (used when it cannot be
# imported), plus the custom colormaps file the plan adds
FALLBACK_USER_SETTINGS_PATHS = {
    'settings_csv_path': 'settings.csv',
    'shortcut_filepath': 'shortcuts.ini',
    'custom_annot_path': 'custom_annotations.json',
    'MEASUREMENTS_CONFIGURATIONS_PATH': 'measurements_viewer_configurations',
    'last_entries_metadata_path': 'last_entries_metadata.csv',
    'additional_metadata_path': 'additional_metadata.json',
    'last_selected_measurements_ini_path': 'last_selected_measurements.ini',
    'recentPaths_path': 'recentPaths.csv',
    'logs_path': 'logs',
    'saved_promptable_models_path': 'saved_promptable_models',
    'promptable_models_list_file_path': 'custom_promptable_models_paths.ini',
    'custom_cmaps_filepath': 'custom_colormaps.ini',
}


class SettingsIsolation:
    """Redirect the user's settings of every ``cellacdc`` module to `folder`.

    - the names of ``tests/conftest.py`` ``USER_SETTINGS_PATHS`` (and
      ``custom_cmaps_filepath``) and ``settings_folderpath``;
    - every other module-level string that is a path inside the real settings
      folder or the real logs folder;
    - ``QSettings(org, app)`` writes an INI file in `folder`;
    - modules imported later get the same treatment (import hook).
    """

    def __init__(self, folder):
        import cellacdc
        self.folder = folder
        self.real_settings = os.path.normcase(
            os.path.abspath(cellacdc.settings_folderpath)
        )
        self.real_logs = os.path.normcase(
            os.path.abspath(getattr(cellacdc, 'logs_path', '') or folder)
        )
        names = dict(FALLBACK_USER_SETTINGS_PATHS)
        try:
            from tests import conftest
            names.update(getattr(conftest, 'USER_SETTINGS_PATHS', {}))
        except Exception:
            pass
        self.names = names
        os.makedirs(folder, exist_ok=True)
        scheme = getattr(cellacdc, 'scheme', 'light')
        with open(os.path.join(folder, 'settings.csv'), 'w') as file:
            file.write(f'setting,value\ncolorScheme,{scheme}\n')
        self.QSettings = None
        self.IsolatedQSettings = None
        self.redirected = []

    def _redirect_path(self, value):
        norm = os.path.normcase(os.path.abspath(value))
        for real, target in (
            (self.real_settings, self.folder),
            (self.real_logs, os.path.join(self.folder, 'logs')),
        ):
            if norm == real or norm.startswith(real + os.sep):
                rest = os.path.abspath(value)[len(real):].lstrip('\\/')
                return os.path.join(target, rest) if rest else target
        return None

    def apply(self, module):
        for name, filename in self.names.items():
            if isinstance(getattr(module, name, None), str):
                setattr(module, name, os.path.join(self.folder, filename))
        if isinstance(getattr(module, 'settings_folderpath', None), str):
            module.settings_folderpath = self.folder
        for name, value in list(vars(module).items()):
            if not isinstance(value, str) or name in self.names:
                continue
            if len(value) < 3 or not os.path.isabs(value):
                continue
            new = self._redirect_path(value)
            if new is not None and new != value:
                setattr(module, name, new)
                self.redirected.append(f'{module.__name__}.{name}')
        if self.IsolatedQSettings is not None:
            if getattr(module, 'QSettings', None) is self.QSettings:
                module.QSettings = self.IsolatedQSettings

    def install(self):
        if 'qtpy' in sys.modules or self._can_import_qt():
            self._make_qsettings()
        for name, module in list(sys.modules.items()):
            if module is not None and (
                name == PACKAGE or name.startswith(PACKAGE + '.')
            ):
                self.apply(module)
        sys.meta_path.insert(0, _IsolationFinder(self))
        return self

    @staticmethod
    def _can_import_qt():
        try:
            import qtpy  # noqa: F401
            return True
        except Exception:
            return False

    def _make_qsettings(self):
        from qtpy.QtCore import QSettings
        iniFormat = getattr(QSettings, 'IniFormat', None)
        if iniFormat is None:
            iniFormat = QSettings.Format.IniFormat
        folder = self.folder

        class IsolatedQSettings(QSettings):
            def __init__(self, *args, **kwargs):
                isOrgApp = (
                    len(args) == 2 and not kwargs
                    and all(isinstance(arg, str) for arg in args)
                )
                if isOrgApp:
                    filename = '_'.join(args) + '.ini'
                    args = (os.path.join(folder, filename), iniFormat)
                super().__init__(*args, **kwargs)

        self.QSettings = QSettings
        self.IsolatedQSettings = IsolatedQSettings


class _IsolationFinder:
    """Meta path finder applying the isolation to cellacdc modules on import."""

    def __init__(self, isolation):
        self.isolation = isolation
        self._busy = set()

    def find_spec(self, fullname, path=None, target=None):
        if not (fullname == PACKAGE or fullname.startswith(PACKAGE + '.')):
            return None
        if fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            import importlib.util
            spec = importlib.util.find_spec(fullname)
        except Exception:
            return None
        finally:
            self._busy.discard(fullname)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        exec_module = getattr(loader, 'exec_module', None)
        if exec_module is None:
            return spec
        isolation = self.isolation

        class _Loader:
            def __getattr__(self, name):
                return getattr(loader, name)

            def create_module(self, spec):
                return loader.create_module(spec)

            def exec_module(self, module):
                exec_module(module)
                isolation.apply(module)

        spec.loader = _Loader()
        return spec


def install_settings_isolation(folder=None):
    """Isolate the user's settings in this (child) process; returns folder."""
    folder = folder or tempfile.mkdtemp(
        prefix='settings_', dir=scratch_dir('isolated_settings')
    )
    SettingsIsolation(folder).install()
    return folder


def resolve_attr(spec):
    """Import 'module:Qual.name' and return the object."""
    import importlib
    modname, _, qual = spec.partition(':')
    obj = importlib.import_module(modname)
    for part in qual.split('.'):
        obj = getattr(obj, part)
    return obj


def existing_candidate(candidates, repo):
    """First 'module[:attr]' candidate whose module file exists in `repo`."""
    files_set = set(list_package_files(repo))
    for candidate in candidates:
        modname = candidate.split(':')[0]
        if module_path(modname, files_set):
            return candidate
    return None
