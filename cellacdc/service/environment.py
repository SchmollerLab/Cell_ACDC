import contextlib
import os
import re
from typing import Optional, Union, Tuple, Literal, List
import shutil
import subprocess
import json
import hashlib
from importlib.metadata import distribution, PackageNotFoundError
from typing import Optional, Callable, Any, List, Literal, Union
import sys

from cellacdc import is_conda_env, user_profile_path
from cellacdc._model_startup import startup_timer
from cellacdc._torch_utils import (
    _torch_import_check, ensure_torch, torch_version_specifier,
)


def run_streaming(
    cmd: Union[list, str], logger_func: Callable = print, check: bool = False
) -> int:
    """Run `cmd`, forwarding each output line (stdout+stderr) to `logger_func`
    as it is produced. Returns the exit code."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    # pip/conda redraw progress with '\r'; universal newlines turns those into lines
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            logger_func(line)
    proc.stdout.close()
    returncode = proc.wait()
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return returncode


def get_model_env_name(model_name: str) -> str:
    """Return standard environment name for a model, e.g. '_acdc_cellpose_v3'."""
    clean_name = model_name.replace(" ", "_").replace("-", "_")
    return f"_acdc_{clean_name}"


def get_install_method() -> Literal["conda", "venv"]:
    """Return 'conda' if running inside conda/mamba, else 'venv'."""
    return "conda" if is_conda_env() else "venv"


def get_venv_dir(model_name: str, env_root: Optional[str] = None) -> str:
    """Return filesystem path for a model's venv."""
    root = env_root or os.path.join(user_profile_path, "envs")
    return os.path.join(root, get_model_env_name(model_name))


def get_python_from_env_dir(env_dir: str) -> Optional[str]:
    """Find python executable inside an environment directory."""
    if not env_dir or not os.path.isdir(env_dir):
        return None

    candidates = []
    if sys.platform == "win32":
        candidates = [
            os.path.join(env_dir, "python.exe"),
            os.path.join(env_dir, "Scripts", "python.exe"),
        ]
    else:
        candidates = [
            os.path.join(env_dir, "bin", "python"),
            os.path.join(env_dir, "python"),
        ]

    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def get_conda_env_dir(model_name: str) -> Optional[str]:
    """Search for existing conda environment matching model name."""
    env_name = get_model_env_name(model_name)
    conda_exe = shutil.which("conda") or os.environ.get("CONDA_EXE")
    if not conda_exe:
        return None
    try:
        res = subprocess.run(
            [conda_exe, "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout)
        envs = data.get("envs", [])
        for env_path in envs:
            if os.path.basename(os.path.normpath(env_path)) == env_name:
                return env_path
    except Exception:
        pass
    return None


def get_model_python_exec(
    model_name: str,
    install_method: Optional[Literal["conda", "venv"]] = None,
    env_root: Optional[str] = None,
) -> Optional[str]:
    """Return existing python executable for model environment if present."""
    method = install_method or get_install_method()
    if method == "conda":
        env_dir = get_conda_env_dir(model_name)
        if env_dir:
            return get_python_from_env_dir(env_dir)
        return None
    else:
        env_dir = get_venv_dir(model_name, env_root=env_root)
        return get_python_from_env_dir(env_dir)


def _get_cellacdc_source_root() -> Optional[str]:
    """Return the repo root if Cell-ACDC runs from a source checkout, else None."""
    import cellacdc

    root = os.path.dirname(os.path.dirname(os.path.abspath(cellacdc.__file__)))
    return root if os.path.isfile(os.path.join(root, "pyproject.toml")) else None


def link_cellacdc_to_env(
    python_exec: str, logger_func: Callable = print
) -> bool:
    """Make a source checkout of cellacdc importable in the environment via a
    .pth file. No-op (returns False) when cellacdc is not a source checkout,
    since the .pth would otherwise expose the host's whole site-packages."""
    repo_root = _get_cellacdc_source_root()
    if repo_root is None:
        return False
    try:
        script = (
            "import site; "
            "sps = [p for p in site.getsitepackages() if 'site-packages' in p.lower()]; "
            "print(sps[0] if sps else (site.getsitepackages()[0] if site.getsitepackages() else site.getusersitepackages()))"
        )
        res = subprocess.run(
            [python_exec, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        site_packages = res.stdout.strip()
        if not site_packages or not os.path.isdir(site_packages):
            logger_func(f"Could not locate site-packages in {python_exec}")
            return False

        pth_path = os.path.join(site_packages, "cellacdc.pth")
        with open(pth_path, "w", encoding="utf-8") as f:
            f.write(repo_root + "\n")
        logger_func(f"Linked Cell-ACDC ({repo_root}) into {pth_path}")
        return True
    except Exception as e:
        logger_func(f"Warning: Failed to link cellacdc to {python_exec}: {e}")
        return False


# Minimal runtime needed to import cellacdc.core/myutils/service.worker and run
# models. Deliberately excludes GUI packages and heavy extras of the full
# Cell-ACDC install (boto3, matplotlib, scikit-learn, imagecodecs, ...).
WORKER_RUNTIME_DEPENDENCIES = [
    "numpy",
    "pandas<3.0",
    "scikit-image>=0.18",
    "opencv-python-headless",
    "natsort",
    "tqdm",
    "h5py",
    "psutil",
    "requests",
    "sympy",
    "fastapi",
    "uvicorn",
    "pydantic",
    "python-multipart",
]


def managed_model_env_dir(python_exec: str) -> Optional[str]:
    """The directory of the model environment created by Cell-ACDC (named
    `_acdc_<model>`, see `get_model_env_name`) that `python_exec` belongs
    to, or None if it belongs to another environment.

    The path is not resolved: in a venv on Linux/macOS, `bin/python` is a
    link to the base interpreter.
    """
    directory = os.path.dirname(os.path.abspath(python_exec))
    # <env>/bin/python, <env>/Scripts/python.exe, <env>/python.exe (conda)
    if os.path.basename(directory).lower() in ("bin", "scripts"):
        directory = os.path.dirname(directory)
    if os.path.basename(directory).startswith("_acdc_"):
        return directory
    return None


def is_managed_model_env(python_exec: str) -> bool:
    """True if `python_exec` belongs to a model environment created by
    Cell-ACDC (see `managed_model_env_dir`).

    Only these environments are repaired automatically (Cell-ACDC is
    reinstalled in them); other environments belong to the user.
    """
    return managed_model_env_dir(python_exec) is not None


# Run in a model environment: print which Cell-ACDC it imports. Old Cell-ACDC
# versions without `cellacdc.service.protocol` use protocol 1.
_RUNTIME_MARKER = "__ACDC_RUNTIME__"
_RUNTIME_IDENTITY_CODE = f'''
def _acdc_runtime_identity():
    import json, os, sys
    import cellacdc
    try:
        from cellacdc.service.protocol import PROTOCOL_VERSION as protocol
    except ImportError:
        protocol = 1
    # Version and direct_url of the installed distribution whose package is
    # the imported one (dist_file: its __init__.py), all None if the imported
    # Cell-ACDC is not an installed package (e.g. a checkout in the working
    # directory). Metadata without a RECORD (e.g. the cellacdc.egg-info of a
    # checkout) was not written by an installer and does not count
    file = os.path.abspath(cellacdc.__file__)
    package_dir = os.path.normcase(os.path.realpath(os.path.dirname(file)))
    dist_version = direct_url = dist_file = None
    try:
        from importlib.metadata import distributions
        for dist in distributions(name="cellacdc"):
            if dist.read_text("RECORD") is None:
                continue
            located = str(dist.locate_file("cellacdc"))
            if os.path.normcase(os.path.realpath(located)) != package_dir:
                continue
            dist_version = dist.version
            direct_url = json.loads(dist.read_text("direct_url.json") or "null")
            dist_file = os.path.join(os.path.abspath(located), "__init__.py")
            break
    except Exception:
        pass
    print({_RUNTIME_MARKER!r} + json.dumps({{
        "protocol": protocol, "version": dist_version,
        "direct_url": direct_url, "file": file, "dist_file": dist_file,
        "prefix": sys.prefix,
    }}), flush=True)
_acdc_runtime_identity()
'''
_RUNTIME_IMPORTS_OK = "__ACDC_RUNTIME_IMPORTS_OK__"
# Run in a model environment, whose Cell-ACDC may still have the
# single-module myutils.py of older versions: imports every module of
# the myutils package (as `import cellacdc.myutils` imported the whole
# old module) and binds `installs` to the module with the package spec
# helpers. No try/except ImportError: missing dependencies must fail
_MYUTILS_IMPORT_CODE = '''
import cellacdc.myutils
if hasattr(cellacdc.myutils, "__path__"):
    import importlib, pkgutil
    for _acdc_module in pkgutil.iter_modules(cellacdc.myutils.__path__):
        importlib.import_module("cellacdc.myutils." + _acdc_module.name)
    from cellacdc.myutils import installs
else:
    from cellacdc import myutils as installs
'''
_RUNTIME_CHECK_CODE = _RUNTIME_IDENTITY_CODE + f'''
import cellacdc.core
{_MYUTILS_IMPORT_CODE}
import cellacdc.service.worker
print({_RUNTIME_IMPORTS_OK!r})
'''


def _parse_runtime_identity(output: str) -> Optional[dict]:
    for line in output.splitlines():
        if line.startswith(_RUNTIME_MARKER):
            try:
                return json.loads(line[len(_RUNTIME_MARKER):])
            except ValueError:
                return None
    return None


def _same_path(path, other) -> bool:
    return (
        os.path.normcase(os.path.realpath(path))
        == os.path.normcase(os.path.realpath(other))
    )


def _is_within(path: str, root: str) -> bool:
    """True if `path` is `root` or inside it."""
    path = os.path.normcase(os.path.realpath(path))
    root = os.path.normcase(os.path.realpath(root))
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _installed_cellacdc_version() -> Optional[str]:
    """Version of the installed Cell-ACDC distribution, or None."""
    from importlib.metadata import version

    try:
        return version("cellacdc")
    except PackageNotFoundError:
        return None


def _installed_cellacdc_direct_url() -> Optional[dict]:
    """Where the installed Cell-ACDC was installed from (PEP 610
    `direct_url.json`), or None (e.g. installed from PyPI)."""
    try:
        text = distribution("cellacdc").read_text("direct_url.json")
        return json.loads(text) if text else None
    except (PackageNotFoundError, ValueError):
        return None


def _build_mismatch(identity: dict) -> Optional[str]:
    """Why the installed (not linked) Cell-ACDC build of `identity` differs
    from this one, or None.

    Compares what `_installed_cellacdc_spec` installs: the commit of a VCS
    install, the URL of an archive install, otherwise the version (a VCS
    build's version string can change for the same commit, e.g. when the
    commit is tagged later).

    The environment's version and direct_url are those of the installed
    distribution that the imported Cell-ACDC (`identity['file']`) belongs
    to (`identity['dist_file']`, see `_RUNTIME_IDENTITY_CODE`). If it
    belongs to none, e.g. Python imports a checkout in the working
    directory before the installed package, the environment does not run
    this build even if an installed package matches.
    """
    expected_url = _installed_cellacdc_direct_url() or {}
    expected_version = _installed_cellacdc_version()
    if not expected_url and expected_version is None:
        # This Cell-ACDC has no distribution metadata to compare with
        return None
    imported, dist_file = identity.get("file"), identity.get("dist_file")
    if not dist_file or not imported or not _same_path(imported, dist_file):
        return (
            f"it imports Cell-ACDC from {imported}, which is not an "
            f"installed Cell-ACDC package"
        )
    actual_url = identity.get("direct_url") or {}
    commit = (expected_url.get("vcs_info") or {}).get("commit_id")
    if commit:
        actual_commit = (actual_url.get("vcs_info") or {}).get("commit_id")
        if actual_commit != commit:
            return (
                f"it has Cell-ACDC {identity.get('version')} from "
                f"{actual_url.get('url') or 'a package index'}"
                f"{f' (commit {actual_commit})' if actual_commit else ''}, "
                f"but this Cell-ACDC is commit {commit}"
            )
        return None
    if expected_url.get("archive_info") is not None and expected_url.get("url"):
        if actual_url.get("url") != expected_url["url"]:
            return (
                f"it has Cell-ACDC {identity.get('version')} from "
                f"{actual_url.get('url') or 'a package index'}, but this "
                f"Cell-ACDC was installed from {expected_url['url']}"
            )
        return None
    if expected_version is not None and identity.get("version") != expected_version:
        return (
            f"it has Cell-ACDC {identity.get('version')}, but this is "
            f"Cell-ACDC {expected_version}"
        )
    return None


def _runtime_mismatch(identity: Optional[dict]) -> Optional[str]:
    """Why the Cell-ACDC of a model environment (`identity`, printed by
    `_RUNTIME_IDENTITY_CODE`) differs from this one, or None if it is the
    same.

    From a source checkout, the environment must import this checkout (it is
    linked, see `link_cellacdc_to_env`). Otherwise it must have the same
    Cell-ACDC build installed (see `_build_mismatch`).
    """
    from cellacdc.service.protocol import PROTOCOL_VERSION

    if identity is None:
        return "Cell-ACDC cannot be imported"
    imported_root = os.path.dirname(os.path.dirname(identity.get("file") or ""))
    if imported_root and _same_path(imported_root, os.getcwd()):
        # Python searches the working directory first. Reinstalling cannot
        # fix this, so say why
        cwd_note = (
            f" (it is the Cell-ACDC in the current working directory "
            f"{os.getcwd()}; start Cell-ACDC from another directory)"
        )
    else:
        cwd_note = ""
    if identity.get("protocol") != PROTOCOL_VERSION:
        return (
            f"it uses model worker protocol {identity.get('protocol')}, but "
            f"this Cell-ACDC uses protocol {PROTOCOL_VERSION}{cwd_note}"
        )
    source_root = _get_cellacdc_source_root()
    if source_root is not None:
        if not imported_root or not _same_path(imported_root, source_root):
            return (
                f"it imports Cell-ACDC from {identity.get('file')}, not from "
                f"this source checkout ({source_root}){cwd_note}"
            )
        return None
    mismatch = _build_mismatch(identity)
    return f"{mismatch}{cwd_note}" if mismatch else None


def _read_cellacdc_runtime(python_exec: str) -> Tuple[Optional[dict], bool, str]:
    """Run `_RUNTIME_CHECK_CODE` in `python_exec`.

    Returns the identity of its Cell-ACDC (None if it cannot be imported),
    whether the worker runtime can be imported, and the process output.
    """
    try:
        res = subprocess.run(
            [python_exec, "-c", _RUNTIME_CHECK_CODE, "--server"],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as err:
        return None, False, f"{type(err).__name__}: {err}"
    output = (res.stdout or "") + (res.stderr or "")
    imports_ok = (
        res.returncode == 0
        and _RUNTIME_IMPORTS_OK in (res.stdout or "").splitlines()
    )
    return _parse_runtime_identity(res.stdout or ""), imports_ok, output


def cellacdc_runtime_mismatch(python_exec: str) -> Optional[str]:
    """Why the Cell-ACDC runtime in `python_exec` cannot run this Cell-ACDC's
    model workers (missing, another version, or broken dependencies), or
    None if it can."""
    identity, imports_ok, output = _read_cellacdc_runtime(python_exec)
    mismatch = _runtime_mismatch(identity)
    if mismatch is None and not imports_ok:
        mismatch = "its Cell-ACDC runtime dependencies cannot be imported"
    if mismatch is not None and not imports_ok and output.strip():
        mismatch = f"{mismatch}:\n{output.strip()}"
    return mismatch


def is_cellacdc_cli_installed(python_exec: str) -> bool:
    """Check if the lightweight Cell-ACDC runtime is importable in the environment.

    Does not check its version, see `cellacdc_runtime_mismatch`.
    """
    test_code = (
        "import cellacdc\n"
        "import cellacdc.core\n"
        + _MYUTILS_IMPORT_CODE
        + "import cellacdc.service.worker\n"
    )
    try:
        res = subprocess.run(
            [python_exec, "-c", test_code, "--server"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return res.returncode == 0
    except Exception:
        return False


# Run in a model environment: where `import cellacdc` would load it from,
# without executing the package
_CELLACDC_ORIGIN_CODE = (
    "import importlib.util, json, sys; "
    "spec = importlib.util.find_spec('cellacdc'); "
    "print(json.dumps({'origin': spec.origin if spec else None, "
    "'prefix': sys.prefix}))"
)


def _uninstall_shadowing_cellacdc(
    python_exec: str, logger_func: Callable = print
):
    """Uninstall a Cell-ACDC copy installed in the environment, which is
    imported instead of the linked source checkout (site-packages come
    before the `.pth` link on `sys.path`).

    The copy is located without importing it (it may be broken, e.g. when
    its dependencies are missing).
    """
    source_root = _get_cellacdc_source_root()
    try:
        res = subprocess.run(
            [python_exec, "-c", _CELLACDC_ORIGIN_CODE],
            capture_output=True, text=True, timeout=60,
        )
        found = json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        return
    imported, prefix = found.get("origin"), found.get("prefix")
    if not imported or not prefix or not _is_within(imported, prefix):
        return
    if source_root is not None and _is_within(imported, source_root):
        return
    logger_func(f"Uninstalling the Cell-ACDC copy at {imported}...")
    run_streaming(
        [python_exec, "-m", "pip", "uninstall", "-y",
         "--disable-pip-version-check", "cellacdc"],
        logger_func,
    )


def install_cellacdc_cli(
    python_exec: str, logger_func: Callable = print, reinstall: bool = False,
) -> bool:
    """Install a lightweight Cell-ACDC runtime into the target environment.

    Cell-ACDC itself is linked (source checkout) or installed with `--no-deps`
    (released package), then only `WORKER_RUNTIME_DEPENDENCIES` are installed.
    With `reinstall`, an installed Cell-ACDC is replaced: the package is
    reinstalled (`--force-reinstall`), or, from a source checkout, a copy
    that hides the linked checkout is uninstalled.
    """
    pip_cmd = [python_exec, "-m", "pip", "install", "--disable-pip-version-check"]
    try:
        if _get_cellacdc_source_root() is not None:
            if reinstall:
                _uninstall_shadowing_cellacdc(python_exec, logger_func)
            link_cellacdc_to_env(python_exec, logger_func=logger_func)
        else:
            spec = _installed_cellacdc_spec()
            options = ["--no-deps"] + (["--force-reinstall"] if reinstall else [])
            logger_func(f"Installing {spec} (no dependencies)...")
            if run_streaming(pip_cmd + options + [spec], logger_func) != 0:
                logger_func(f"Warning: pip install of {spec} failed.")
                return False

        logger_func("Installing lightweight Cell-ACDC runtime dependencies...")
        if run_streaming(pip_cmd + WORKER_RUNTIME_DEPENDENCIES, logger_func) != 0:
            logger_func("Warning: pip install of runtime dependencies failed.")
            return False
        return True
    except Exception as e:
        logger_func(
            f"Warning: Failed to install Cell-ACDC runtime into {python_exec}: {e}"
        )
        return False


def _installed_cellacdc_spec() -> str:
    """Return a pip spec that reproduces this installed Cell-ACDC build."""
    from cellacdc import __version__ as acdc_version

    try:
        dist = distribution('cellacdc')
    except PackageNotFoundError as err:
        raise RuntimeError(
            'Cannot reproduce this Cell-ACDC build in the model environment: '
            'the installed distribution metadata is unavailable.'
        ) from err

    direct_url_text = dist.read_text('direct_url.json')
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        url = direct_url.get('url')
        vcs_info = direct_url.get('vcs_info') or {}
        if url and vcs_info.get('commit_id'):
            vcs = vcs_info.get('vcs', 'git')
            spec = f'cellacdc @ {vcs}+{url}@{vcs_info["commit_id"]}'
            subdirectory = direct_url.get('subdirectory')
            if subdirectory:
                spec += f'#subdirectory={subdirectory}'
            return spec
        if url and direct_url.get('archive_info') is not None:
            return f'cellacdc @ {url}'

    if re.fullmatch(r'\d+(\.\d+)*', str(acdc_version)):
        return f'cellacdc=={acdc_version}'
    raise RuntimeError(
        'Cannot reproduce this development Cell-ACDC build in the model '
        'environment. Install Cell-ACDC from a VCS URL or wheel that records '
        'PEP 610 direct_url.json provenance, or run from a source checkout.'
    )


def repair_cellacdc_runtime(
    python_exec: str, logger_func: Callable = print
) -> bool:
    """(Re)install this Cell-ACDC build in the environment of `python_exec`
    and check the result.

    Used when the environment has no Cell-ACDC, another version (e.g. it was
    created by an older Cell-ACDC), or a worker with another protocol version
    (see `protocol.py`). Returns False if it still differs afterwards.

    Several processes (e.g. cluster jobs sharing an environment) may try to
    repair the same environment: a lock file lets one of them reinstall,
    and the others then find the environment repaired.
    """
    try:
        with _repair_lock(python_exec, logger_func) as waited:
            if waited and cellacdc_runtime_mismatch(python_exec) is None:
                return True
            if not install_cellacdc_cli(
                python_exec, logger_func=logger_func, reinstall=True
            ):
                return False
            mismatch = cellacdc_runtime_mismatch(python_exec)
    except TimeoutError as err:
        logger_func(f"Warning: {err}")
        return False
    if mismatch is not None:
        logger_func(
            f"Warning: Cell-ACDC in {python_exec} still differs from this "
            f"Cell-ACDC after reinstalling it: {mismatch}"
        )
        return False
    return True


_REPAIR_LOCK_NAME = ".cellacdc-repair.lock"
# How long to wait for another process that repairs the same environment
_REPAIR_LOCK_TIMEOUT_SECONDS = 30 * 60


def _try_lock_file(fd: int) -> bool:
    """Take the exclusive lock of the operating system on the open file `fd`
    without waiting.

    Returns False if another process (or another open file in this process)
    holds it. Raises OSError if the file system does not support locks. The
    lock is released by `_unlock_file`, or by the operating system when the
    process exits, also when it is killed.
    """
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except PermissionError:  # EACCES: locked by another file handle
            return False
        return True
    import errno
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as err:
        # EACCES: how some network file systems (e.g. CIFS) report a lock
        # held by another process
        if err.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
            return False
        raise
    return True


def _unlock_file(fd: int):
    """Release the lock taken by `_try_lock_file`."""
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def _repair_lock(
    python_exec: str, logger_func: Callable = print,
    timeout: float = _REPAIR_LOCK_TIMEOUT_SECONDS, poll: float = 1.0,
):
    """Hold the repair lock of the environment of `python_exec`.

    The lock is a lock of the operating system on the lock file (see
    `_try_lock_file`): only one process holds it, and the operating system
    releases it when that process exits, also when it is killed. The lock
    file itself is never deleted: deleting it would let a process lock a new
    file while another one still holds the old one.

    On shared file systems whose locks apply only within one computer (e.g.
    NFS mounted with `nolock` or `local_lock`, Lustre with `localflock`),
    processes on different computers do not exclude each other.

    Yields whether another process held it first (then the environment may
    already be repaired). Raises TimeoutError after `timeout` seconds. If the
    lock file cannot be opened or locked (e.g. a read-only folder or a file
    system without locks), logs a warning and yields without holding a lock.
    """
    import time

    env_dir = managed_model_env_dir(python_exec) or os.path.dirname(
        os.path.abspath(python_exec)
    )
    path = os.path.join(env_dir, _REPAIR_LOCK_NAME)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    except OSError as err:
        fd = None
        logger_func(
            f"Warning: cannot open {path} ({err}); repairing Cell-ACDC in "
            f"{env_dir} without a lock."
        )
    if fd is None:
        yield False
        return
    try:
        t0 = time.monotonic()
        waited = False
        while True:
            try:
                locked = _try_lock_file(fd)
            except OSError as err:
                locked = False
                logger_func(
                    f"Warning: cannot lock {path} ({err}); repairing "
                    f"Cell-ACDC in {env_dir} without a lock."
                )
                break
            if locked:
                break
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(
                    f"Another process has been repairing Cell-ACDC in "
                    f"{env_dir} for too long (lock file {path})."
                )
            if not waited:
                logger_func(
                    f"Waiting for another process that repairs Cell-ACDC in "
                    f"{env_dir}..."
                )
            waited = True
            time.sleep(poll)
        try:
            yield waited
        finally:
            if locked:
                _unlock_file(fd)
    finally:
        os.close(fd)


def ensure_cellacdc_cli(
    python_exec: str, logger_func: Callable = print
) -> bool:
    """Ensure the environment runs this Cell-ACDC build.

    Installs the lightweight Cell-ACDC runtime if it is missing, and
    reinstalls it if the environment has another Cell-ACDC version or
    protocol version (see `cellacdc_runtime_mismatch`).
    """
    with startup_timer("Cell-ACDC runtime version check", logger_func):
        mismatch = cellacdc_runtime_mismatch(python_exec)
    if mismatch is None:
        return True
    logger_func(f"Cell-ACDC in {python_exec}: {mismatch}")
    with startup_timer("Cell-ACDC runtime installation", logger_func):
        return repair_cellacdc_runtime(python_exec, logger_func=logger_func)


def load_model_env_config(model_name: str) -> dict:
    """Return the entry of `cellacdc/models/_env_config.py` for the model.

    The file is loaded by path so that the model package (whose `__init__`
    may install dependencies) is never imported. Returns `{}` if the model
    has no entry.
    """
    import importlib.util

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "_env_config.py",
    )
    spec = importlib.util.spec_from_file_location(
        "_acdc_model_env_config", config_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    envs = module.MODEL_ENVS
    if model_name in envs:
        return dict(envs[model_name])

    def _norm(name):
        return name.lower().replace(" ", "_").replace("-", "_")

    for key, value in envs.items():
        if _norm(key) == _norm(model_name):
            return dict(value)
    return {}


def model_installed_in_main_env(model_name: str) -> bool:
    """Check distribution versions without importing model or dependency modules."""
    from importlib.metadata import PackageNotFoundError, version
    from packaging.specifiers import SpecifierSet

    config = load_model_env_config(model_name)
    packages = config.get("main_env_packages", config.get("packages", []))
    if not packages:
        # Built-in or unconfigured models are not evidence of a preinstallation.
        return False
    specs = _normalize_package_specs(packages)
    if config.get("torch"):
        specs += [
            {"pkg_name": "torch",
             "version_specifier": torch_version_specifier(config.get("torch_version"))},
            {"pkg_name": "torchvision"},
        ]
    for spec in specs:
        distribution = spec.get("distribution_name") or spec.get("pypi_name") or spec["pkg_name"]
        constraints = []
        if spec.get("version_specifier"):
            constraints.append(spec["version_specifier"])
        if spec.get("exact_version"):
            constraints.append(f"=={spec['exact_version']}")
        for key, operator, inclusive_key in (
            ("min_version", ">", "include_lower_version"),
            ("max_version", "<", "include_higher_version"),
        ):
            if spec.get(key):
                bound = operator + ("=" if spec.get(inclusive_key) else "")
                constraints.append(f"{bound}{spec[key]}")
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            return False
        if installed not in SpecifierSet(",".join(constraints)):
            return False
    return True


def missing_required_env(
    model_name: str, init_kwargs: Optional[dict] = None
) -> list:
    """Return the `required_env` specs of the model that are not set."""
    init_kwargs = init_kwargs or {}
    missing = []
    for spec in load_model_env_config(model_name).get("required_env", []):
        if os.environ.get(spec["name"]):
            continue
        overrides = spec.get("unless_init_kwargs", [])
        if any(
            init_kwargs.get(key) not in (None, "", "None") for key in overrides
        ):
            continue
        missing.append(spec)
    return missing


def ensure_required_env(
    model_name: str,
    init_kwargs: Optional[dict] = None,
    prompt_func: Optional[Callable] = None,
) -> bool:
    """Ask for missing runtime environment variables of the model.

    Values are stored in `os.environ` of this process only (never written to
    disk or logged), so that the worker started afterwards inherits them.
    Returns False if the user cancelled or no prompt is available.
    """
    from cellacdc.service import prompts

    for spec in missing_required_env(model_name, init_kwargs):
        name = spec["name"]
        if prompt_func is None:
            return False
        text = spec.get("description") or f"{model_name} requires {name}."
        value = prompt_func(prompts.text_input(
            f"{model_name}: {name} required", f"{text}\n\n{name}:",
            extra={"secret": bool(spec.get("secret"))},
        ))
        value = (value or "").strip()
        if not value:
            return False
        os.environ[name] = value
    return True


def required_env_fingerprint(model_name: str) -> str:
    """Fingerprint required values inherited by a newly started worker."""
    digest = hashlib.sha256()
    for spec in load_model_env_config(model_name).get("required_env", []):
        name = spec["name"]
        digest.update(name.encode())
        digest.update(b'\0')
        digest.update(os.environ.get(name, '').encode())
        digest.update(b'\0')
    return digest.hexdigest()


def _normalize_package_specs(packages) -> list:
    return [
        {"pkg_name": pkg} if isinstance(pkg, str) else dict(pkg)
        for pkg in packages
    ]


_WORKER_BASE_PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("python-multipart", "multipart"),
    ("requests", "requests"),
    ("numpy", "numpy"),
]


def _check_environment_dependencies(
    model_name: str, python_exec: str, logger_func: Callable = print
) -> bool:
    """Validate an existing environment in one disposable, non-installing process."""
    from cellacdc.config import parser_args

    config = load_model_env_config(model_name)
    packages = [
        {"pkg_name": name, "import_pkg_name": import_name}
        for name, import_name in _WORKER_BASE_PACKAGES
    ] + _normalize_package_specs(config.get("packages", []))
    script = f'''
import json
from importlib import import_module
from importlib.metadata import version, PackageNotFoundError
from cellacdc._model_startup import startup_timer
from cellacdc.config import parser_args
parser_args["debug"] = {bool(parser_args["debug"])!r}
with startup_timer("Cell-ACDC runtime imports"):
    import cellacdc.core
    exec({_MYUTILS_IMPORT_CODE!r})
    import cellacdc.service.worker
{_RUNTIME_IDENTITY_CODE}
if {bool(config.get("torch"))!r}:
    with startup_timer("torch/torchvision imports (shared process)"):
        exec({_torch_import_check(config.get("torch_version"))!r})
defaults = dict(min_version="", max_version="", exact_version="",
                include_lower_version=False, include_higher_version=False)
specs = installs._normalize_pkg_specs(json.loads({json.dumps(packages)!r}), "", "", defaults)
for spec in specs:
    with startup_timer("dependency check (shared process): " + spec["import_pkg_name"]):
        module = import_module(spec["import_pkg_name"])
        installed = getattr(module, "__version__", None)
        if not installed or installed == 'not-installed':
            try:
                installed = version(spec["pkg_name"])
            except PackageNotFoundError:
                installed = ""
        installs._check_package_version_constraints(spec, installed)
print("__ACDC_ENV_OK__")
'''
    with startup_timer(f"{model_name}: combined environment checks", logger_func):
        result = subprocess.run(
            [python_exec, "-c", script, "--server"],
            capture_output=True, text=True,
        )
    for line in result.stdout.splitlines():
        if line.startswith("[model startup]"):
            logger_func(line)
    if result.returncode == 0 and "__ACDC_ENV_OK__" in result.stdout.splitlines():
        mismatch = _runtime_mismatch(_parse_runtime_identity(result.stdout))
        if mismatch is None:
            return True
        logger_func(
            f"{model_name}: Cell-ACDC in {python_exec} differs from this "
            f"Cell-ACDC ({mismatch}); reinstalling it."
        )
        return False
    logger_func(
        f"{model_name}: combined environment check failed; "
        f"checking dependencies individually for repair.\n"
        f"{result.stderr or result.stdout}"
    )
    return False


def ensure_model_dependencies(
    model_name: str,
    python_exec: str,
    install_method: Optional[Literal["conda", "venv"]] = None,
    logger_func: Callable = print,
    parent: Any = None,
    auto_install: bool = True,
    prompt_func: Optional[Callable] = None,
    found_cuda_version: Optional[Union[Tuple[int, int], Literal[False]]] = None,
    possible_links: Optional[Tuple[List[Tuple[str, Optional[str]]], int]] = None,
    platform: Optional[str] = None,
):
    """Ensure specific dependencies for a model are installed in its environment."""
    from cellacdc import myutils

    config = load_model_env_config(model_name)
    if config.get("torch"):
        torch_ok = ensure_torch(
            python_exec, logger_func=logger_func, prompt_func=prompt_func,
            platform=platform or sys.platform,
            found_cuda_version=found_cuda_version,
            possible_links=possible_links,
            torch_version=config.get("torch_version"),
        )
        if not torch_ok:
            raise RuntimeError(
                f"{model_name}: PyTorch is not installed in {python_exec} "
                "(installation cancelled or failed, see the log above). "
                "The model cannot run without it."
            )

    specs = _normalize_package_specs(config.get("packages", []))
    if specs:
        method = install_method or get_install_method()
        installer = "conda" if method == "conda" else "pip"
        print(specs)
        myutils.check_install_package(
            specs,
            installer=installer,
            python_exec=python_exec,
            logger_func=logger_func,
            parent=parent,
            is_cli=(parent is None),
            auto_install=auto_install,
        )
    _ensure_model_setup_steps(
        model_name, python_exec, config, logger_func, auto_install, prompt_func
    )


def _ensure_model_setup_steps(
    model_name, python_exec, config, logger_func, auto_install, prompt_func
):
    for step in config.get("setup_steps", []):
        name = step["name"]
        check = step.get("check")
        # Optional steps (e.g., dependencies of only some model types) do
        # not block the model when they fail or are declined
        optional = step.get("optional", False)
        if check:
            result = subprocess.run(
                [python_exec, "-c", check, "--server"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                logger_func(f"{model_name}: {name} already complete.")
                continue
            logger_func(f"{model_name}: {name} check failed:\n{result.stderr or result.stdout}")
        if not auto_install:
            from . import prompts
            approved = prompt_func is not None and prompt_func(prompts.confirm(
                f"Set up {model_name}", step.get("description", name),
            ))
            if not approved and optional:
                logger_func(
                    f"{model_name}: optional setup step {name} skipped."
                )
                continue
            if prompt_func is None:
                raise RuntimeError(f"{model_name}: {name} requires setup approval.")
            if not approved:
                raise RuntimeError(f"{model_name}: {name} setup cancelled.")
        logger_func(f"{model_name}: {step.get('description', name)}")
        try:
            for args in step["commands"]:
                run_streaming([python_exec, *args], logger_func, check=True)
            if check:
                result = subprocess.run(
                    [python_exec, "-c", check, "--server"],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"{model_name}: {name} verification failed in {python_exec}:\n"
                        f"{result.stderr or result.stdout}"
                    )
        except (subprocess.CalledProcessError, RuntimeError, OSError) as err:
            if not optional:
                raise
            logger_func(
                f"[WARNING]: {model_name}: the optional setup step {name} "
                f"failed ({err}). Continuing without it."
            )


def ensure_worker_dependencies(
    python_exec: str,
    install_method: Optional[Literal["conda", "venv"]] = None,
    logger_func: Callable = print,
    parent: Any = None,
    auto_install: bool = True,
):
    """Ensure base dependencies needed by the model worker service are installed."""
    from cellacdc import myutils

    if not ensure_cellacdc_cli(python_exec, logger_func=logger_func):
        raise RuntimeError(
            f"Cell-ACDC could not be installed in {python_exec} "
            "(see the log above). The model worker cannot start without it."
        )

    method = install_method or get_install_method()
    installer = "conda" if method == "conda" else "pip"
    base_pkgs_names = _WORKER_BASE_PACKAGES
    
    base_pkgs_import_names = [import_name for _, import_name in base_pkgs_names]
    base_pkgs = [pkg_name for pkg_name, _ in base_pkgs_names]
    myutils.check_install_package(
        base_pkgs,
        import_pkg_name=base_pkgs_import_names,
        installer=installer,
        python_exec=python_exec,
        logger_func=logger_func,
        parent=parent,
        is_cli=(parent is None),
        auto_install=auto_install,
    )


def _python_version_of(python_exec: str) -> Optional[str]:
    try:
        res = subprocess.run(
            [python_exec, "-c",
             "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=20,
        )
        return res.stdout.strip() if res.returncode == 0 else None
    except Exception:
        return None


def _which_all(name: str) -> List[str]:
    """All executables called `name` on PATH (shutil.which returns only the first)."""
    exts = [""]
    if sys.platform == "win32":
        exts = [""] + os.environ.get("PATHEXT", ".EXE").lower().split(os.pathsep)
    found = []
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        for ext in exts:
            path = os.path.join(directory.strip('"'), name + ext)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                found.append(path)
    return found


def find_python_interpreter(version: str) -> Optional[str]:
    """Find an installed interpreter for `version` (e.g. '3.11').

    Checks the current interpreter, the Windows `py` launcher, `pythonX.Y`,
    `pythonXY`, `python3` and `python` on PATH, and pyenv. Each candidate is
    run to verify its actual version."""
    version = ".".join(version.split(".")[:2])
    if f"{sys.version_info.major}.{sys.version_info.minor}" == version:
        return sys.executable

    candidates = []
    if sys.platform == "win32":
        py_launcher = shutil.which("py")
        if py_launcher:
            try:
                res = subprocess.run(
                    [py_launcher, f"-{version}", "-c", "import sys; print(sys.executable)"],
                    capture_output=True, text=True, timeout=20,
                )
                if res.returncode == 0:
                    candidates.append(res.stdout.strip())
            except Exception:
                pass
    for name in (
        f"python{version}", f"python{version.replace('.', '')}",
        "python3", "python",
    ):
        candidates.extend(_which_all(name))

    pyenv_root = os.environ.get("PYENV_ROOT") or os.path.join(
        os.path.expanduser("~"), ".pyenv", "pyenv-win" if sys.platform == "win32" else ""
    )
    versions_dir = os.path.join(pyenv_root, "versions")
    if os.path.isdir(versions_dir):
        for name in sorted(os.listdir(versions_dir), reverse=True):
            if name == version or name.startswith(version + "."):
                candidates.append(get_python_from_env_dir(os.path.join(versions_dir, name)))

    # Windows Store aliases (WindowsApps) may open the Store; try them last
    candidates.sort(key=lambda c: bool(c) and "windowsapps" in c.lower())
    checked = set()
    for candidate in candidates:
        if not candidate or not os.path.isfile(candidate):
            continue
        key = os.path.normcase(os.path.realpath(candidate))
        if key in checked:
            continue
        checked.add(key)
        if _python_version_of(candidate) == version:
            return candidate
    return None


def create_venv(
    env_dir: str,
    python_version: Optional[str] = None,
    logger_func: Callable = print,
) -> None:
    """Create a venv at `env_dir`, optionally with a specific Python version.

    If no matching interpreter is installed, falls back to `uv`, which can
    download the requested Python. Raises RuntimeError if neither works."""
    if python_version is None:
        run_streaming([sys.executable, "-m", "venv", env_dir], logger_func, check=True)
        return

    interpreter = find_python_interpreter(python_version)
    if interpreter:
        run_streaming([interpreter, "-m", "venv", env_dir], logger_func, check=True)
        return

    uv = shutil.which("uv")
    if uv:
        logger_func(f"Python {python_version} not found locally, using uv to fetch it...")
        run_streaming(
            [uv, "venv", "--seed", "--python", python_version, env_dir],
            logger_func, check=True
        )
        return

    raise RuntimeError(
        f"Python {python_version} was not found. Install it (or install `uv`, "
        f"which can download it automatically) and try again."
    )


def create_model_env(
    model_name: str,
    install_method: Optional[Literal["conda", "venv"]] = None,
    python_version: Optional[str] = None,
    env_root: Optional[str] = None,
    ensure_deps: bool = True,
    auto_install: bool = True,
    logger_func: Callable = print,
    parent: Any = None,
    prompt_func: Optional[Callable] = None,
) -> str:
    """Create a new dedicated conda or venv environment for the given model."""
    method = install_method or get_install_method()
    env_name = get_model_env_name(model_name)
    if python_version is None:
        python_version = load_model_env_config(model_name).get("python")
    py_ver =  python_version or f"{sys.version_info.major}.{sys.version_info.minor}"

    if method == "conda":
        conda_exe = shutil.which("conda") or os.environ.get("CONDA_EXE", "conda")
        logger_func(f'Creating conda environment "{env_name}" (Python {py_ver})...')
        cmd = [conda_exe, "create", "-y", "-n", env_name, f"python={py_ver}"]
        run_streaming(cmd, logger_func, check=True)
        py_exec = get_model_python_exec(model_name, install_method="conda")
        if not py_exec:
            raise RuntimeError(
                f"Conda environment {env_name} created but python executable not found."
            )
    else:
        env_dir = get_venv_dir(model_name, env_root=env_root)
        os.makedirs(os.path.dirname(env_dir), exist_ok=True)
        logger_func(f'Creating venv environment at "{env_dir}" (Python {py_ver})...')
        create_venv(env_dir, py_ver if python_version else None, logger_func)
        py_exec = get_python_from_env_dir(env_dir)
        if not py_exec or not os.path.isfile(py_exec):
            raise RuntimeError(
                f"Venv created at {env_dir} but python executable not found."
            )

    link_cellacdc_to_env(py_exec, logger_func=logger_func)
    if ensure_deps:
        ensure_worker_dependencies(
            py_exec,
            install_method=method,
            logger_func=logger_func,
            parent=parent,
            auto_install=auto_install,
        )
        ensure_model_dependencies(
            model_name,
            py_exec,
            install_method=method,
            logger_func=logger_func,
            parent=parent,
            auto_install=auto_install,
            prompt_func=prompt_func,
            platform=sys.platform,
        )
    return py_exec


def get_or_create_model_env(
    model_name: str,
    install_method: Optional[Literal["conda", "venv"]] = None,
    python_version: Optional[str] = None,
    env_root: Optional[str] = None,
    ensure_deps: bool = True,
    auto_install: bool = True,
    logger_func: Callable = print,
    parent: Any = None,
    prompt_func: Optional[Callable] = None,
) -> str:
    """Get existing environment's python or create a new dedicated one."""
    method = install_method or get_install_method()
    with startup_timer(f"{model_name}: environment discovery", logger_func):
        py_exec = get_model_python_exec(
            model_name, install_method=method, env_root=env_root
        )
    if py_exec and os.path.isfile(py_exec):
        with startup_timer(f"{model_name}: source linking", logger_func):
            link_cellacdc_to_env(py_exec, logger_func=logger_func)
        if ensure_deps:
            if _check_environment_dependencies(model_name, py_exec, logger_func):
                _ensure_model_setup_steps(
                    model_name, py_exec, load_model_env_config(model_name),
                    logger_func, auto_install, prompt_func,
                )
                return py_exec
            ensure_worker_dependencies(
                py_exec,
                install_method=method,
                logger_func=logger_func,
                parent=parent,
                auto_install=auto_install,
            )
            ensure_model_dependencies(
                model_name,
                py_exec,
                install_method=method,
                logger_func=logger_func,
                parent=parent,
                auto_install=auto_install,
                prompt_func=prompt_func,
                platform=sys.platform,
            )
        return py_exec

    return create_model_env(
        model_name,
        install_method=method,
        python_version=python_version,
        env_root=env_root,
        ensure_deps=ensure_deps,
        auto_install=auto_install,
        logger_func=logger_func,
        parent=parent,
        prompt_func=prompt_func,
    )
