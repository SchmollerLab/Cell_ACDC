"""Installing third-party packages: version checks of installed packages,
pip and conda commands and their runners (_install_pip_package,
_run_command), normalising package specs, check_install_package with the
special installers (_SPECIAL_INSTALLERS, _PKG_SPEC_KEYS), the
command-line and GUI install prompts, and the checks for server mode and
a running GUI. Plugin API, forwarded by myutils/__init__.py without a
warning: check_install_package, is_server_mode, _raise_server_missing,
_install_package_msg, get_package_version. Under if GUI_INSTALLED it
imports QCoreApplication and the shared message box of the GUI install
prompts (shared/widgets/message_box.py); it imports no dialog.
"""
import os
import re
import shlex
from typing import Literal, List, Callable, Optional, Union
import sys
import platform
import shutil
import traceback
import subprocess
import importlib
from importlib import import_module
from packaging import version as packaging_version
from cellacdc import GUI_INSTALLED, is_conda_env
from cellacdc import html_utils, is_mac
from cellacdc import printl
from cellacdc import try_input_install_package
if GUI_INSTALLED:
    from qtpy.QtCore import QCoreApplication
    from cellacdc.shared.widgets import message_box as widgets_message_box


def testQcoreApp():
    print(QCoreApplication.instance())


def get_package_version(import_pkg_name):
    import importlib.metadata
    version =  importlib.metadata.version(import_pkg_name)
    return version


def check_napari_plugin(plugin_name, module_name, parent=None):
    try:
        import_module(module_name)
    except ModuleNotFoundError as e:
        url = 'https://napari.org/stable/plugins/find_and_install_plugin.html#find-and-install-plugins'
        href = html_utils.href_tag('this guide', url)
        txt = html_utils.paragraph(f"""
            To correctly use this napari utility you need to <b>install the 
            plugin</b> called <code>{plugin_name}</code>.<br><br>
            Please, read {href} on how to install plugins in napari.<br><br>
            You will need to <b>restart</b> both napari and Cell-ACDC after installing 
            the plugin.<br><br>
            NOTE: in the text box in napari you will need to write the full name 
            <code>{plugin_name}</code> becasue it is NOT A SEARCH BOX.
        """)
        msg = widgets_message_box.myMessageBox()
        msg.critical(parent, f'Napari plugin required', txt)
        raise e

def _run_command_streaming(command: List[str], logger: Callable = print):
    """Run `command`, forwarding each output line to `logger` as it appears.
    Raises CalledProcessError on non-zero exit."""
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace', bufsize=1
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            logger(line)
    proc.stdout.close()
    returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)

def _install_pip_package(
        pkg_name: Union[str, List[str]],
        logger: Callable = print,
        install_dependencies: bool = True,
        force_binary: bool = True,
        pref_binary: bool = True,
        python_exec: Optional[str] = None,
        ) -> None:
    """Install one requirement or, if a list is given, all of them with a 
    single pip command (so pip can resolve them together)."""
    if isinstance(pkg_name, str):
        pkg_names = [pkg_name.replace('"', '')]
    else:
        pkg_names = [p.replace('"', '') for p in pkg_name]
    pkg_name = ' '.join(pkg_names)
    py_exec = python_exec or sys.executable
    command = [py_exec, '-m', 'pip', 'install', *pkg_names]
    if force_binary:
        command.append('--only-binary=:all:')
    elif pref_binary:
        command.append('--prefer-binary')
    if not install_dependencies:
        command.append('--no-deps')
    try:
        _run_command_streaming(command, logger)
    except subprocess.CalledProcessError as e:
        if "--only-binary=:all:" in str(e):
            logger(f"Error: {pkg_name} does not have a binary distribution available, trying preferred binary.")
            _install_pip_package(
                pkg_name=pkg_names,
                logger=logger,
                install_dependencies=install_dependencies,
                force_binary=False,
                pref_binary=True,
                python_exec=python_exec,
            )
        elif "--prefer-binary" in str(e):
            logger(f"Error: {pkg_name} does not have a preferred binary distribution available, trying source.")
            command.remove('--prefer-binary')
            command.append('--no-binary=:all:')
            _install_pip_package(
                pkg_name=pkg_names,
                logger=logger,
                install_dependencies=install_dependencies,
                force_binary=False,
                pref_binary=False,
                python_exec=python_exec,
            )
        else:
            logger(f"""Error: {pkg_name} installation failed. Please check the error message. This is probably due to the package 
                   not being available for your platform or python version.""")
            raise e

def uninstall_pip_package(pkg_name):
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', pkg_name]
    )


def purge_module(module_name):
    to_delete = [mod for mod in sys.modules if mod == module_name or mod.startswith(module_name + '.')]
    for mod in to_delete:
        del sys.modules[mod]
        
    importlib.invalidate_caches()
    importlib.import_module(module_name)
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        raise ModuleNotFoundError(f"Module '{module_name}' not found in sys.modules.")

def is_second_version_greater(
        target_version: str,
        current_version: str,
):
    """
    Compares two model versions and returns True if the current version is
    greater than or equal to the target version.
    """
    target_version = packaging_version.parse(target_version)
    current_version = packaging_version.parse(current_version)
    
    return current_version >= target_version

def is_pkg_version_within_range(
        package_version: str, min_version='', max_version=''
    ):
    package_version_number = packaging_version.parse(package_version)
    is_greater_than_min = True
    if min_version:
        min_version_number = packaging_version.parse(min_version)
        is_greater_than_min = package_version_number >= min_version_number
    
    is_less_than_max = True
    if max_version:
        max_version_number = packaging_version.parse(max_version)
        is_less_than_max = package_version_number <= max_version_number
    
    return is_greater_than_min and is_less_than_max


def is_gui_running():
    if not GUI_INSTALLED:
        return False
    
    return QCoreApplication.instance() is not None

def check_pkg_version(import_pkg_name, min_version, include_lower_version, raise_err=True):
    is_version_correct = False
    try:
        installed_version = get_package_version(import_pkg_name)
        if include_lower_version:
            is_version_correct = (
                packaging_version.parse(installed_version) 
                >= packaging_version.parse(min_version)
            )
        else:   
            is_version_correct = (
                packaging_version.parse(installed_version) 
                > packaging_version.parse(min_version)
            )
    except Exception as err:
        is_version_correct = False
    
    if raise_err and not is_version_correct:
        raise ModuleNotFoundError(
            f'{import_pkg_name}>{min_version} not installed.'
        )
    else:
        return is_version_correct

def check_pkg_exact_version(import_pkg_name, version: str, raise_err=True):
    is_version_correct = False
    try:
        installed_version = get_package_version(import_pkg_name)
        is_version_correct = (
            packaging_version.parse(installed_version) 
            == packaging_version.parse(version)
        )
    except Exception as err:
        is_version_correct = False
    
    if raise_err and not is_version_correct:
        raise ModuleNotFoundError(
            f'{import_pkg_name}=={version} not installed.'
        )
    else:
        return is_version_correct

def check_pkg_max_version(
        import_pkg_name, max_version, include_higher_version, raise_err=True
    ):
    is_version_correct = False
    try:
        from packaging import version
        installed_version = get_package_version(import_pkg_name)  
        if include_higher_version:
            is_version_correct = (
                packaging_version.parse(installed_version) 
                <= packaging_version.parse(max_version)
            )
        else:
            is_version_correct = (
                packaging_version.parse(installed_version) 
                < packaging_version.parse(max_version)
            )
    except Exception as err:
        is_version_correct = False
    
    if raise_err and not is_version_correct:
        raise ModuleNotFoundError(
            f'{import_pkg_name}<={max_version} not installed.'
        )
    else:
        return is_version_correct

def install_package_conda(
        conda_pkg_name, 
        channel='conda-forge', 
        env_name=None, 
        env_prefix=None
    ):
    if not is_conda_env() and env_name is None and env_prefix is None:
        raise EnvironmentError(
            'Cell-ACDC is not running in a `conda` environment.'
        )
    
    if isinstance(conda_pkg_name, (list, tuple)):
        # Keep quotes so that specifiers like ">=1.0" survive the shell
        conda_pkg_name = ' '.join(conda_pkg_name)
    else:
        conda_pkg_name = conda_pkg_name.replace('"', '')
    if env_name is not None:
        conda_exe = shutil.which('conda') or os.environ.get('CONDA_EXE', 'conda')
        command = f'{conda_exe} install -y -n {env_name} -c {channel} {conda_pkg_name}'
    elif env_prefix is not None:
        conda_exe = shutil.which('conda') or os.environ.get('CONDA_EXE', 'conda')
        command = f'{conda_exe} install -y -p "{env_prefix}" -c {channel} {conda_pkg_name}'
    else:
        conda_prefix, pip_prefix = get_pip_conda_prefix()
        conda_prefix = re.sub(
            r'(-c\sconda-forge\s?|--channel=conda-forge\s?)', f'-c {channel} ', 
            conda_prefix
        )
        command = f'{conda_prefix} -y {conda_pkg_name}'
    _subprocess_run_command(command)

def _subprocess_run_command(command, shell=True, callback='check_call'):
    func = getattr(subprocess, callback)
    try:
        out = func(command, shell=shell)
    except Exception as err:
        commad_parts = shlex.split(command)
        print(
            f'[WARNING]: Command `{command}` failed. '
            f'Trying with `{commad_parts}`...'
        )
        out = func(commad_parts, shell=False)
    
    return out


def _run_command(command: str | list[str], shell=False):
    if not isinstance(command, (str, list)):
        raise TypeError(
            f'Command must be a string or a list of strings, not {type(command)}'
        )
    
    command_str = None
    if isinstance(command, str):
        args_list = [command]
        command_str = command
    else:
        args_list = command
        if len(command) == 1:
            command_str = command[0]
        
    try:
        subprocess.check_call(args_list, shell=shell)
        return
    except Exception as err:
        pass
    
    if command_str is None:
        return
    
    try:
        subprocess.check_call(command_str, shell=shell)
        return
    except Exception as err:
        pass
    
    try:
        from cellacdc import acdc_regex
        args = acdc_regex.RE_SPLIT_SPACES_IGNORE_QUOTES.split(command_str)[1::2]
        subprocess.check_call(args, shell=shell)
        return
    except Exception as err:
        pass


def is_server_mode():
    """True when running inside a model worker process (`service.worker`).

    In this mode nothing is installed and no prompt is shown: the client
    prepares the environment (see `cellacdc/models/_env_config.py`).
    """
    import cellacdc
    return cellacdc.RUNNING_AS_SERVER

def _raise_server_missing(what):
    raise ModuleNotFoundError(
        f'The model worker environment is missing or has an incompatible '
        f'version of: {what}. The worker does not install packages by '
        f'itself. Add the package to this model\'s entry in '
        f'cellacdc/models/_env_config.py and restart, or install it in the '
        f'environment: {sys.executable}'
    )
def _install_tensorflow(max_version='', min_version='', python_exec=None):
    cpu = platform.processor()
    pkg_command = _get_pkg_command_pip_install(
        'tensorflow', 
        max_version=max_version, 
        min_version=min_version
    )
    pkg_command = pkg_command.replace('"', '')

    py_exec = python_exec or sys.executable
    if is_mac and cpu == 'arm' and py_exec == sys.executable:
        conda_prefix, pip_prefix = get_pip_conda_prefix()
        args = [f'{conda_prefix} "{pkg_command}"']
        shell = True
    else:
        args = [py_exec, '-m', 'pip', 'install', '-U', pkg_command]
        shell = False
    subprocess.check_call(args, shell=shell)
    
    if py_exec == sys.executable:
        print('TensorFlow installed. Restart Cell-ACDC before using it; '
              'NumPy cannot safely be reloaded in this process.')

def _install_segment_anything(python_exec=None):
    py_exec = python_exec or sys.executable
    args = [
        py_exec, '-m', 'pip', 'install',
        '-U', '--use-pep517',
        'git+https://github.com/facebookresearch/segment-anything.git'
    ]
    subprocess.check_call(args)

def _install_sam2(python_exec=None):
    py_exec = python_exec or sys.executable
    args = [
        py_exec, '-m', 'pip', 'install',
        '-U', '--use-pep517',
        'git+https://github.com/facebookresearch/sam2.git'
    ]
    subprocess.check_call(args)

def _install_deepsea(python_exec=None):
    py_exec = python_exec or sys.executable
    subprocess.check_call(
        [py_exec, '-m', 'pip', 'install', 'deepsea']
    )


_PKG_SPEC_KEYS = (
    'pkg_name', 'import_pkg_name', 'pypi_name', 'min_version', 'max_version',
    'exact_version', 'include_higher_version', 'include_lower_version',
    # Only used to detect an installation (see models/_env_config.py);
    # installation uses `pkg_name`/`pypi_name`.
    'distribution_name',
)
_SPECIAL_INSTALLERS = {
    'tensorflow': _install_tensorflow,
    'deepsea': _install_deepsea,
    'segment_anything': _install_segment_anything,
    'sam2': _install_sam2,
}


def _align_to_packages(value, n_pkgs: int, arg_name: str) -> list:
    """Return a list with one entry per package.

    A list/tuple must have exactly `n_pkgs` entries (use '' for "use the
    default"). A non-empty plain string is only allowed for a single package.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != n_pkgs:
            raise ValueError(
                f'`{arg_name}` has {len(value)} entries but {n_pkgs} '
                'packages were given'
            )
        return list(value)
    if value and n_pkgs > 1:
        raise ValueError(
            f'`{arg_name}` must be a list with one entry per package '
            f'when checking {n_pkgs} packages'
        )
    return [value] * n_pkgs


def _normalize_pkg_specs(
        pkg_name, import_pkg_name, pypi_name, defaults: dict
    ) -> list:
    """Turn str | dict | list[str | dict] into a list of full spec dicts.
    `import_pkg_name` and `pypi_name` are str or lists aligned with the
    packages. Keys given in a dict entry override everything else."""
    entries = [pkg_name] if isinstance(pkg_name, (str, dict)) else list(pkg_name)
    n_pkgs = len(entries)
    import_names = _align_to_packages(
        import_pkg_name, n_pkgs, 'import_pkg_name'
    )
    pypi_names = _align_to_packages(pypi_name, n_pkgs, 'pypi_name')

    specs = []
    for entry, imp_name, pypi in zip(entries, import_names, pypi_names):
        spec = dict(defaults)
        spec['import_pkg_name'] = imp_name
        spec['pypi_name'] = pypi
        if isinstance(entry, str):
            spec['pkg_name'] = entry
        else:
            unknown = set(entry) - set(_PKG_SPEC_KEYS)
            if unknown:
                raise TypeError(f'Unknown package spec keys: {unknown}')
            spec.update(entry)
        if not spec['import_pkg_name']:
            spec['import_pkg_name'] = spec['pkg_name']
        spec['import_pkg_name'] = spec['import_pkg_name'].replace('-', '_')
        specs.append(spec)
    return specs


def _check_package_version_constraints(spec: dict, installed_version: str):
    """Apply the version bounds used for external environment checks."""
    if not installed_version:
        return
    if installed_version == 'not-installed':
        import traceback
        traceback.print_stack()
        print(installed_version)
        raise ModuleNotFoundError(
            f"{spec['pkg_name']} is not installed"
        )

    installed = packaging_version.parse(installed_version)
    if spec['exact_version']:
        if installed != packaging_version.parse(spec['exact_version']):
            raise ModuleNotFoundError(
                f"{spec['pkg_name']}: expected {spec['exact_version']}, "
                f"found {installed_version}"
            )
    for key, inclusive_key, lower in (
        ('min_version', 'include_lower_version', True),
        ('max_version', 'include_higher_version', False),
    ):
        if not spec[key]:
            continue
        bound = packaging_version.parse(spec[key])
        outside = installed < bound if lower else installed > bound
        if outside or (installed == bound and not spec[inclusive_key]):
            raise ModuleNotFoundError(
                f"{spec['pkg_name']}: {key} {spec[key]} not satisfied "
                f"(found {installed_version})"
            )


def _check_single_package(spec: dict, force_upgrade: bool, python_exec):
    """Check one package.

    Returns
    -------
    (satisfied, needs_upgrade) : tuple[bool, bool]
        `satisfied` is True if nothing has to be installed.
        `needs_upgrade` is True if the package is installed but must be
        upgraded/downgraded.
    """
    pkg_name = spec['pkg_name']
    import_pkg_name = spec['import_pkg_name']
    exact_version = spec['exact_version']
    min_version = spec['min_version']
    max_version = spec['max_version']
    include_higher_version = spec['include_higher_version']
    include_lower_version = spec['include_lower_version']

    imported = False
    try:
        if python_exec is not None and python_exec != sys.executable:
            code = (
                f"import {import_pkg_name}\n"
                f"import importlib.metadata as m\n"
                f"try:\n"
                f"    v = getattr({import_pkg_name}, '__version__', None)\n"
                f"    if not v or v == 'not-installed':\n"
                f"        v = m.version('{pkg_name}')\n"
                f"except Exception:\n"
                f"    v = ''\n"
                f"print(v)\n"
            )
            proc = subprocess.run(
                [python_exec, '-c', code], capture_output=True, text=True
            )
            if proc.returncode != 0:
                raise ModuleNotFoundError(
                    f"Package '{pkg_name}' not installed in {python_exec}"
                )
            imported = True
            installed_version = proc.stdout.strip()
            if force_upgrade:
                raise ModuleNotFoundError(
                    f'User requested to forcefully upgrade "{pkg_name}"'
                )
            _check_package_version_constraints(spec, installed_version)
        else:
            import_module(import_pkg_name)
            imported = True
            if force_upgrade:
                raise ModuleNotFoundError(
                    f'User requested to forcefully upgrade "{pkg_name}"'
                )
            if exact_version:
                check_pkg_exact_version(import_pkg_name, exact_version)
            if min_version:
                check_pkg_version(
                    import_pkg_name, min_version, include_lower_version
                )
            if max_version:
                check_pkg_max_version(
                    import_pkg_name, max_version, include_higher_version
                )
    except ModuleNotFoundError:
        # If the import worked, the failure was a version mismatch -> upgrade
        return False, imported
    return True, False


def _describe_spec(spec: dict) -> str:
    """Human readable 'name (constraints)' string for messages."""
    constraints = []
    if spec['exact_version']:
        constraints.append(f"=={spec['exact_version']}")
    if spec['min_version']:
        constraints.append(f">={spec['min_version']}")
    if spec['max_version']:
        constraints.append(f"<={spec['max_version']}")
    name = spec['pypi_name'] or spec['pkg_name']
    return f"{name} ({', '.join(constraints)})" if constraints else name


def check_install_package(
        pkg_name: Union[str, dict, list],
        import_pkg_name: Union[str, List[str]]='',
        pypi_name: Union[str, List[str]]='',
        note='',
        parent=None,
        raise_on_cancel=True,
        logger_func=print,
        is_cli=False,
        caller_name='Cell-ACDC',
        force_upgrade=False,
        upgrade=False,
        min_version='',
        max_version='',
        exact_version='',
        install_dependencies=True,
        return_outcome=False,
        installer: Literal['pip', 'conda']='pip',
        include_higher_version: bool = False,
        include_lower_version: bool = False,
        python_exec: Optional[str] = None,
        conda_env_name: Optional[str] = None,
        conda_env_prefix: Optional[str] = None,
        auto_install: bool = False
    ):
    """Try to import one or more packages. Each package is checked
    individually; all missing/outdated ones are then installed together after
    a single confirmation prompt.

    Parameters
    ----------
    pkg_name : str, dict or list of (str or dict)
        Package(s) to check. A plain string is the name displayed to the user
        (and imported if `import_pkg_name` is empty). A dict may contain any of
        the per-package keys: 'pkg_name' (required), 'import_pkg_name',
        'pypi_name', 'min_version', 'max_version', 'exact_version',
        'include_higher_version', 'include_lower_version'. Keys that are not
        given fall back to the function arguments of the same name.
    import_pkg_name : str or list of str, optional
        Name(s) used to import the package(s) (case sensitive). If a list, it
        must have one entry per package in `pkg_name` (same order); use ''
        for packages that should be imported by their `pkg_name`. A plain
        string is only valid when checking a single package. A 'import_pkg_name'
        key in a dict entry takes precedence. Default is ''
    pypi_name : str or list of str, optional
        Name(s) used with pip, same list rules as `import_pkg_name`. If
        empty, the entry's `pkg_name` is installed. Default is ''
    note : str, optional
        Additional text to display to the user. Default is ''
    parent : QObject, optional
        Calling QtWidget. Default is None
    raise_on_cancel : bool, optional
        Raise exception if processed cancelled. Default is True
    logger_func : callable, optional
        Function used to log text. Default is print
    is_cli : bool, optional
        If True, message will be displayed in the terminal.
        If False, message will be displayed in a Qt message box.
        Default is False
    caller_name : str, optional
        Program calling this function. Default is 'Cell-ACDC'
    force_upgrade : bool, optional
        If True, we force the upgrade of all packages even if installed.
    upgrade : bool, optional
        If True, pip will upgrade the packages. This value is True if
        `force_upgrade` is True or any package needs an upgrade.
    min_version, max_version, exact_version : str, optional
        Default version constraints (`major[.minor][.patch]`) for entries
        that do not define their own. A violated constraint forces an upgrade.
    install_dependencies : bool, optional
        If False, the `--no-deps` flag will be added to the pip command.
        Applies to the whole batch.
    return_outcome : bool, optional
        If True, returns True after a successful installation.
    installer : str, optional
        Package manager to use to install the packages. Either 'pip' or
        'conda'. Default is 'pip'
    include_higher_version : bool, optional
        If True, a higher installed version will not be downgraded.
        Default is False
    include_lower_version : bool, optional
        If True, a lower installed version will not be upgraded.
        Default is False
    python_exec : str, optional
        Target python executable in which to check and install the packages.
        If None, sys.executable is used. Default is None.
    conda_env_name : str, optional
        Target conda environment name when installer='conda'.
    conda_env_prefix : str, optional
        Target conda environment prefix path when installer='conda'.
    auto_install : bool, optional
        If True, install without asking the user. Default is False

    Raises
    ------
    ModuleNotFoundError
        Error raised if process is cancelled and `raise_on_cancel=True`.
    """
    if not is_gui_running():
        is_cli = True

    if python_exec is not None and python_exec != sys.executable:
        # `conda install` without an env name targets the active env, so 
        # for another interpreter always use its own pip
        installer = 'pip'

    defaults = {
        'pkg_name': '',
        'min_version': min_version,
        'max_version': max_version,
        'exact_version': exact_version,
        'include_higher_version': include_higher_version,
        'include_lower_version': include_lower_version,
    }
    specs = _normalize_pkg_specs(
        pkg_name, import_pkg_name, pypi_name, defaults
    )

    from cellacdc._model_startup import startup_timer

    # 1) Check every package individually
    missing_specs = []
    for spec in specs:
        with startup_timer(
            f"dependency check: {spec['import_pkg_name']}", logger_func
        ):
            satisfied, needs_upgrade = _check_single_package(
                spec, force_upgrade, python_exec
            )
        if satisfied:
            continue
        missing_specs.append(spec)
        if needs_upgrade:
            upgrade = True

    if force_upgrade:
        upgrade = True

    if not missing_specs:
        return

    if is_server_mode() and python_exec in (None, sys.executable):
        _raise_server_missing(
            ', '.join(_describe_spec(s) for s in missing_specs)
        )

    # 2) Ask once for everything that is missing
    display_name = ', '.join(_describe_spec(s) for s in missing_specs)
    pkg_commands = [
        _get_pkg_command_pip_install(
            spec['pypi_name'] or spec['pkg_name'],
            exact_version=spec['exact_version'],
            max_version=spec['max_version'],
            min_version=spec['min_version'],
            including_higher_version=spec['include_higher_version'],
            including_lower_version=spec['include_lower_version'],
        )
        for spec in missing_specs
    ]
    if auto_install:
        proceed = True
    else:
        proceed = _install_package_msg(
            display_name,
            note=note,
            parent=parent,
            upgrade=upgrade,
            is_cli=is_cli,
            caller_name=caller_name,
            logger_func=logger_func,
            installer=installer,
            pkg_commands=pkg_commands
        )

    if not proceed:
        if raise_on_cancel:
            raise ModuleNotFoundError(
                f'User aborted installation of: {display_name}'
            )
        return f'User aborted installation of: {display_name}'

    # 3) Install: special packages individually, everything else in one go
    try:
        batch_commands = []
        tensorflow_batched = False
        for spec, command in zip(missing_specs, pkg_commands):
            name = spec['pypi_name'] or spec['pkg_name']
            special_installer = _SPECIAL_INSTALLERS.get(name)
            if name == 'tensorflow' and python_exec not in (None, sys.executable):
                # Resolve model and runtime constraints together in isolated envs.
                special_installer = None
                tensorflow_batched = installer == 'pip'
            if special_installer is None:
                batch_commands.append(command)
            elif name == 'tensorflow':
                special_installer(
                    max_version=spec['max_version'],
                    min_version=spec['min_version'],
                    python_exec=python_exec
                )
            else:
                special_installer(python_exec=python_exec)

        if batch_commands:
            # `batch_commands` is a list of requirement strings. The helpers
            # must append them as separate arguments (one per requirement).
            if installer == 'pip':
                _install_pip_package(
                    batch_commands,
                    install_dependencies=install_dependencies,
                    logger=logger_func,
                    python_exec=python_exec
                )
            else:
                install_package_conda(
                    batch_commands,
                    env_name=conda_env_name,
                    env_prefix=conda_env_prefix
                )

        if (installer == 'pip' and len(specs) > 1 and not tensorflow_batched
                and python_exec not in (None, sys.executable)):
            # A package installed above can pin a dependency of another
            # requested package below its minimum (e.g. TF -> typing-extensions).
            # Pip cannot resolve such conflicts, so force the requested
            # version without touching dependencies.
            for spec in specs:
                satisfied, _ = _check_single_package(spec, False, python_exec)
                if satisfied:
                    continue
                printl(spec)
                command = _get_pkg_command_pip_install(
                    spec['pypi_name'] or spec['pkg_name'],
                    exact_version=spec['exact_version'],
                    max_version=spec['max_version'],
                    min_version=spec['min_version'],
                    including_higher_version=spec['include_higher_version'],
                    including_lower_version=spec['include_lower_version'],
                )
                _install_pip_package(
                    command,
                    install_dependencies=False,
                    logger=logger_func,
                    python_exec=python_exec
                )
    except Exception:
        printl(traceback.format_exc())
        if is_cli or (python_exec is not None and python_exec != sys.executable):
            logger_func(
                f'Installation of {display_name} failed in '
                f'{python_exec or sys.executable}.'
            )
            raise
        _inform_install_package_failed(
            display_name, parent=parent, do_exit=raise_on_cancel
        )

    if return_outcome:
        return True

def check_install_custom_dependencies(custom_install_requires, *args, **kwargs):
    """Used to install a package with custom dependencies, usefull if they have
    random pinned versions for their dependencies.
    
    For *args and **kwargs see `check_install_package`.

    Parameters
    ----------
    custom_install_requires : list
        list of dependencies. Check either requirements.txt, setup.py, 
        setup.cfg, pyproject.toml, or any other file that lists the dependencies.
        For formatting of the dependencies with min max version, 
        use _get_pkg_command_pip_install.
    """
    kwargs['install_dependencies'] = False
    kwargs['return_outcome'] = True
    success = check_install_package(*args, **kwargs)
    if not success:
        return
    for pkg_name in custom_install_requires:
        _install_pip_package(pkg_name)


def check_matplotlib_version(qparent=None):
    mpl_version = get_package_version('matplotlib')  
    mpl_version_digits = mpl_version.split('.')

    mpl_major = int(mpl_version_digits[0])
    mpl_minor = int(mpl_version_digits[1])
    is_less_than_3_5 = (
        mpl_major < 3 or (mpl_major >= 3 and mpl_minor < 5)
    )
    if not is_less_than_3_5:
        return 
    
    proceed = _install_package_msg('matplotlib', parent=qparent, upgrade=True)
    if not proceed:
        raise ModuleNotFoundError(
            f'User aborted "matplotlib" installation'
        )
    import subprocess
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-U', 'matplotlib']
        )
    except Exception as e:
        printl(traceback.format_exc())
        _inform_install_package_failed(
            'matplotlib', parent=qparent, do_exit=False
            )

def _inform_install_package_failed(pkg_name, parent=None, do_exit=True):
    conda_prefix, pip_prefix = get_pip_conda_prefix()

    install_command = f'<code>{pip_prefix} --upgrade {pkg_name}</code>'
    txt = html_utils.paragraph(f"""
        Unfortunately, <b>installation of</b> <code>{pkg_name}</code> <b>returned an error</b>.<br><br>
        Try restarting Cell-ACDC. If it doesn't work, 
        please close Cell-ACDC and, with the <code>acdc</code> <b>environment ACTIVE</b>, 
        install <code>{pkg_name}</code> manually using the follwing command:<br><br>
        {install_command}<br><br>
        Thank you for your patience.
    """)
    msg = widgets_message_box.myMessageBox()
    msg.critical(parent, f'{pkg_name} installation failed', txt)
    print('*'*50)
    print(
        f'[ERROR]: Installation of "{pkg_name}" failed. '
        f'Please, close Cell-ACDC and run the command '
        f'{pip_prefix} --upgrade {pkg_name}`'
    )
    print('^'*50)


def _install_package_msg(
        pkg_name, note='', parent=None, upgrade=False, caller_name='Cell-ACDC',
        is_cli=False, pkg_command='', logger_func=print, 
        exact_version='', max_version='', min_version='', 
        installer: Literal['pip', 'conda']='pip',
        include_higher_version: bool = False,
        include_lower_version: bool = False,
        pkg_commands: Optional[List[str]] = None
    ):
    if is_cli:
        proceed = _install_package_cli_msg(
            pkg_name, note=note, upgrade=upgrade, caller_name=caller_name,
            pkg_command=pkg_command, pkg_commands=pkg_commands,
            exact_version=exact_version,
            max_version=max_version, 
            min_version=min_version, logger_func=logger_func, 
            installer=installer,
            include_higher_version=include_higher_version,
            include_lower_version=include_lower_version
        )
    else:
        proceed = _install_package_gui_msg(
            pkg_name, note=note, parent=parent, upgrade=upgrade, 
            caller_name=caller_name, pkg_command=pkg_command,
            pkg_commands=pkg_commands, exact_version=exact_version,
            max_version=max_version, min_version=min_version, 
            logger_func=logger_func, installer=installer,
            including_higher_version=include_higher_version,
            including_lower_version=include_lower_version
        )
    return proceed

def get_cli_multi_choice_question(question, choices):
    choices_format = [f'{i+1}) {choice}.' for i, choice in enumerate(choices)]
    choices_format = ' '.join(choices_format)
    choices_opts = '/'.join([str(i) for i in range(1, len(choices)+1)])
    text = f'{question} {choices_format} q) Quit. ({choices_opts})?: '
    return text


def _get_pkg_command_pip_install(
        pkg_command, 
        exact_version='',
        max_version='', 
        min_version='',
        including_lower_version=False, 
        including_higher_version=False
    ):
    if exact_version:
        pkg_command = f'{pkg_command}=={exact_version}'
        pkg_command = f'"{pkg_command}"'
        return pkg_command
    
    if including_higher_version:
        sign_max = "<="
    else:
        sign_max = "<"
    if including_lower_version:
        sign_min = ">="
    else:
        sign_min = ">"
    if min_version:
        pkg_command = f'{pkg_command}{sign_min}{min_version}'
        if max_version:
            pkg_command = f'{pkg_command},'
    
    if max_version:
        pkg_command = f'{pkg_command}{sign_max}{max_version}'
        
    pkg_command = f'"{pkg_command}"'    
    return pkg_command

def _install_package_cli_msg(
        pkg_name, note='', upgrade=False, caller_name='Cell-ACDC',
        logger_func=print, pkg_command='', exact_version='', max_version='', 
        min_version='', installer: Literal['pip', 'conda']='pip',
        include_lower_version=False,
        include_higher_version=False,
        pkg_commands=None
    ):
    if pkg_commands:
        pkg_command = ' '.join(pkg_commands)
    else:
        if not pkg_command:
            pkg_command = pkg_name
        
        pkg_command = _get_pkg_command_pip_install(
            pkg_command, exact_version=exact_version, 
            max_version=max_version, min_version=min_version,
            including_lower_version=include_lower_version,
            including_higher_version=include_higher_version
        )
    
    if upgrade:
        action = 'upgrade'
    else:
        action = 'install'
    
    conda_prefix, pip_prefix = get_pip_conda_prefix()

    if installer == 'conda':
        install_command = f'{conda_prefix} {pkg_command}'
    else:
        install_command = f'{pip_prefix} --upgrade {pkg_command}'
        
    separator = '-'*60
    txt = (
        f'{separator}\n{caller_name} needs to {action} {pkg_name}\n\n'
        'You can choose to install it now or stop the process and install it '
        'later with the following command:\n\n'
        f'{install_command}\n'
    )
    logger_func(txt)

    while True:
        answer = try_input_install_package(pkg_name, install_command)
        if not answer or answer.lower() == 'y':
            return True
        
        if answer.lower() == 'n':
            return False
        
        logger_func(
            f'{answer} is not a valid answer. Valid answers are "y" for Yes and '
            '"n" for No.'
        )

def _install_package_gui_msg(
        pkg_name, note='', parent=None, upgrade=False, caller_name='Cell-ACDC', 
        pkg_command='', logger_func=None, exact_version='', 
        max_version='', min_version='',
        including_lower_version=False, including_higher_version=False,
        installer: Literal['pip', 'conda']='pip',
        pkg_commands=None
    ):
    msg = widgets_message_box.myMessageBox(parent=parent)
    if upgrade:
        install_text = 'upgrade'
    else:
        install_text = 'install'
    if pkg_name == 'BayesianTracker':
        pkg_name = 'btrack'
    
    if pkg_commands:
        pkg_command = ' '.join(pkg_commands)
    else:
        if not pkg_command:
            pkg_command = pkg_name
        
        pkg_command = _get_pkg_command_pip_install(
            pkg_command, exact_version=exact_version,
            max_version=max_version, min_version=min_version,
            including_lower_version=including_lower_version,
            including_higher_version=including_higher_version
        )
    
    conda_prefix, pip_prefix = get_pip_conda_prefix()

    if installer == 'pip':
        command = f'{pip_prefix} --upgrade {pkg_command}'
    elif installer == 'conda':
        command = f'{conda_prefix} {pkg_command}'
        
    command_html = command.lower().replace('<', '&lt;').replace('>', '&gt;')
    
    txt = html_utils.paragraph(f"""
        {caller_name} is going to <b>download and {install_text}</b>
        <code>{pkg_name}</code>.<br><br>
        Make sure you have an <b>active internet connection</b>,
        before continuing.<br>
        Progress will be displayed on the terminal<br><br>
        You might have to <b>restart {caller_name}</b>.<br><br>
        Alternatively, you can cancel the process and try later.<br><br>
        To install later, or if the installation fails, run the following 
        command:
    """)
    if note:
        txt = f'{txt}{note}'
    _, okButton = msg.information(
        parent, f'Install {pkg_name}', txt, 
        buttonsTexts=('Cancel', 'Ok'), 
        commands=(command_html,)
    )
    return msg.clickedButton == okButton


def get_pip_conda_prefix(list_return=False, exec_path=None):
    from cellacdc.config import parser_args
    try:
        cp = parser_args
        if cp["install_details"] is not None:
            no_cli_install = True
            install_details = cp["install_details"]
            venv_path = install_details["venv_path"]
            conda_path = install_details["conda_path"]
            if ' ' not in conda_path:
                conda_path = conda_path.strip('"').strip("'")
        else:
            no_cli_install = False
    except:
        no_cli_install = False
        pass

    exec_path = exec_path or sys.executable
    if exec_path != sys.executable:
        uses_different_exec = True
    else:
        uses_different_exec = False
        
    if no_cli_install:
        conda_prefix = f'{conda_path} install -y -p {venv_path} -c conda-forge'
        if ' ' in exec_path:
            exec_path = f'"{exec_path}"'
        pip_prefix = f"{exec_path} -m pip install"
    else:
        conda_prefix = 'conda install -y -c conda-forge'
        if uses_different_exec:
            pip_prefix = f'"{exec_path}" -m pip install'
        else:
            pip_prefix = 'pip install'
    
    pip_list = [exec_path, '-m', 'pip', 'install']
    if no_cli_install:
        conda_list = [conda_path.strip('"').strip("'"), 'install', '-y', '-p', venv_path.strip('"').strip("'"), '-c', 'conda-forge']
    else:
        conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
        conda_list = [conda, 'install', '-y', '-c', 'conda-forge']
    if list_return:
        return conda_list, pip_list
    else:
        return conda_prefix, pip_prefix
