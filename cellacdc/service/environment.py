import os
import re
import sys
import shutil
import subprocess
import json
from typing import Optional, Callable, Any, List, Literal

from cellacdc import is_conda_env, user_profile_path


def run_streaming(
    cmd: list, logger_func: Callable = print, check: bool = False
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


def is_cellacdc_cli_installed(python_exec: str) -> bool:
    """Check if the lightweight Cell-ACDC runtime is importable in the environment."""
    test_code = (
        "import cellacdc; "
        "import cellacdc.core; "
        "import cellacdc.myutils; "
        "import cellacdc.service.worker"
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


def install_cellacdc_cli(
    python_exec: str, logger_func: Callable = print
) -> bool:
    """Install a lightweight Cell-ACDC runtime into the target environment.

    Cell-ACDC itself is linked (source checkout) or installed with `--no-deps`
    (released package), then only `WORKER_RUNTIME_DEPENDENCIES` are installed.
    """
    pip_cmd = [python_exec, "-m", "pip", "install", "--disable-pip-version-check"]
    try:
        if _get_cellacdc_source_root() is not None:
            link_cellacdc_to_env(python_exec, logger_func=logger_func)
        else:
            from cellacdc import __version__ as acdc_version

            is_release = re.fullmatch(r"\d+(\.\d+)*", str(acdc_version)) is not None
            spec = f"cellacdc=={acdc_version}" if is_release else "cellacdc"
            logger_func(f"Installing {spec} (no dependencies)...")
            if run_streaming(pip_cmd + ["--no-deps", spec], logger_func) != 0:
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


def ensure_cellacdc_cli(
    python_exec: str, logger_func: Callable = print
) -> bool:
    """Ensure the lightweight Cell-ACDC runtime is installed and importable."""
    if is_cellacdc_cli_installed(python_exec):
        return True
    return install_cellacdc_cli(python_exec, logger_func=logger_func)


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
        specs += [{"pkg_name": "torch"}, {"pkg_name": "torchvision"}]
    for spec in specs:
        distribution = spec.get("distribution_name") or spec.get("pypi_name") or spec["pkg_name"]
        constraints = []
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


def detect_torch_index() -> Optional[str]:
    """Pick the PyTorch wheel index matching this machine.

    Uses the CUDA version reported by `nvidia-smi` (the highest CUDA the
    driver supports). Returns None to use the default PyPI wheels (CPU on
    Windows/macOS, which is also all that exists on macOS).
    """
    base = "https://download.pytorch.org/whl/"
    if sys.platform == "darwin":
        return None

    cuda_version = None
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.run(
                [nvidia_smi], capture_output=True, text=True, timeout=20
            ).stdout
            match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
            if match:
                cuda_version = (int(match.group(1)), int(match.group(2)))
        except Exception:
            pass

    if cuda_version is None:
        # Plain PyPI on Linux ships CUDA builds, so ask for CPU explicitly
        return base + "cpu" if sys.platform.startswith("linux") else None

    for minimum, tag in (
        ((12, 8), "cu128"), ((12, 6), "cu126"), ((12, 4), "cu124"),
        ((12, 1), "cu121"), ((11, 8), "cu118"),
    ):
        if cuda_version >= minimum:
            return base + tag
    return base + "cpu"


TORCH_CUDA_TAGS = ("cu128", "cu126", "cu124", "cu121", "cu118")
_TORCH_BASE_URL = "https://download.pytorch.org/whl/"


def get_torch_choices() -> dict:
    """Map of label -> pip index URL (None = default PyPI wheels)."""
    choices = {"CPU": _TORCH_BASE_URL + "cpu"}
    if sys.platform == "darwin":
        return {"CPU": None}
    for tag in reversed(TORCH_CUDA_TAGS):
        choices[f"CUDA {tag[2:4]}.{tag[4:]} (NVIDIA GPU)"] = _TORCH_BASE_URL + tag
    return choices


def ensure_torch(
    python_exec: str,
    logger_func: Callable = print,
    prompt_func: Optional[Callable] = None,
) -> bool:
    """Install PyTorch into the model environment.

    The build (CPU or CUDA) is detected from `nvidia-smi`. If `prompt_func`
    is given (see `cellacdc.service.prompts`), the user is asked on behalf
    of the worker with the detected build preselected.
    """
    check = subprocess.run(
        [python_exec, "-c", "import torch, torchvision"], capture_output=True
    )
    if check.returncode == 0:
        return True

    index_url = detect_torch_index()
    if prompt_func is not None:
        from cellacdc.service import prompts

        choices = get_torch_choices()
        default = next(
            (k for k, v in choices.items() if v == index_url),
            next(iter(choices)),
        )
        answer = prompt_func(prompts.choice(
            "Install PyTorch",
            "This model needs PyTorch in its isolated environment.\n"
            f"Detected: {default}.\nChoose the build to install:",
            list(choices), default,
            dialog="pytorch",
            extra={"platforms": choices, "python": python_exec},
        ))
        if answer is None:
            logger_func("PyTorch installation cancelled.")
            return False
        index_url = choices[answer]
    base_cmd = [python_exec, "-m", "pip", "install", "torch", "torchvision"]
    attempts = []
    if index_url:
        logger_func(f"Installing PyTorch from {index_url}")
        attempts.append(base_cmd + ["--index-url", index_url])
    attempts.append(base_cmd)
    for cmd in attempts:
        try:
            run_streaming(cmd, logger_func, check=True)
            return True
        except Exception as err:
            logger_func(f"PyTorch install failed ({err}), trying fallback...")
    return False


def _normalize_package_specs(packages) -> list:
    return [
        {"pkg_name": pkg} if isinstance(pkg, str) else dict(pkg)
        for pkg in packages
    ]


def ensure_model_dependencies(
    model_name: str,
    python_exec: str,
    install_method: Optional[Literal["conda", "venv"]] = None,
    logger_func: Callable = print,
    parent: Any = None,
    auto_install: bool = True,
    prompt_func: Optional[Callable] = None,
):
    """Ensure specific dependencies for a model are installed in its environment."""
    from cellacdc import myutils

    config = load_model_env_config(model_name)
    if config.get("torch"):
        torch_ok = ensure_torch(
            python_exec, logger_func=logger_func, prompt_func=prompt_func
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
        myutils.check_install_package(
            specs,
            installer=installer,
            python_exec=python_exec,
            logger_func=logger_func,
            parent=parent,
            is_cli=(parent is None),
            auto_install=auto_install,
        )
    for step in config.get("setup_steps", []):
        name = step["name"]
        check = step.get("check")
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
            if prompt_func is None:
                raise RuntimeError(f"{model_name}: {name} requires setup approval.")
            if not prompt_func(prompts.confirm(
                f"Set up {model_name}", step.get("description", name),
            )):
                raise RuntimeError(f"{model_name}: {name} setup cancelled.")
        logger_func(f"{model_name}: {step.get('description', name)}")
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
    base_pkgs_names = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("python-multipart", "multipart"),
        ("requests", "requests"),
        ("numpy", "numpy"),
    ]
    
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
    py_exec = get_model_python_exec(
        model_name, install_method=method, env_root=env_root
    )
    if py_exec and os.path.isfile(py_exec):
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
