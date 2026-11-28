"""About the running Cell-ACDC: its version (read_version, from
setuptools_scm or _version.py), release date, git branch and remotes,
the Linux distribution, the version and system info text, and the log
files (the Logger stdout tee, StdErr, setupLogger, get_logs_path and
deleting old log files).
"""
import os
import re
import sys
import platform
import traceback
import logging
import datetime
import subprocess
import importlib
from uuid import uuid4
from importlib import metadata
import requests
import json
from cellacdc import GUI_INSTALLED
from cellacdc import is_linux
from cellacdc import cellacdc_path, printl, logs_path
from cellacdc import user_profile_path
from cellacdc import qrc_resources_path
from cellacdc import settings_folderpath
from cellacdc import cellacdc_installation_path
if GUI_INSTALLED:
    from qtpy.QtWidgets import QPlainTextEdit
from . import installs


def get_logs_path():
    return logs_path

class Logger(logging.Logger):
    def __init__(
            self,
            module='base', 
            name='cellacdc-logger', 
            level=logging.DEBUG,
            QLogWidget: 'QPlainTextEdit'=None
        ):
        super().__init__(f'{name}-{module}', level=level)
        self.propagate = False  # prevent UnicodeEncodeError via root StreamHandler
        self._stdout = sys.stdout
        self._stderr = StdErr(logger=self)
        sys.stderr = self._stderr
        self._levelToName = {
            50: "CRITICAL",
            40: "ERROR",
            30: "WARNING",
            20: "INFO",
            10: "DEBUG",
            0: "NOTSET"
        }
        self._q_log_widget = QLogWidget
        
    def write(self, text, log_to_file=True, write_to_stdout=True):
        """Capture print statements, print to terminal and log text to 
        the open log file

        Parameters
        ----------
        text : str
            Text to log
        log_to_file : bool, optional
            If True, call `info` method with `text`. Default is True
        """     
        if write_to_stdout:
            try:
                self._stdout.write(text)
            except UnicodeEncodeError:
                self._stdout.write(text.encode(
                    self._stdout.encoding, errors='replace'
                ).decode(self._stdout.encoding))
        
        if self._q_log_widget is not None:
            try:
                # Log thread-safely to the QPlainTextEdit widget
                from qtpy.QtCore import QThread
                if QThread.currentThread() == self._q_log_widget.thread():
                    self._q_log_widget.appendPlainText(text)
                    self._q_log_widget.verticalScrollBar().setValue(
                        self._q_log_widget.verticalScrollBar().maximum()
                    )
            except Exception:
                pass
        
        if not log_to_file:
            return
        
        if text == '\n':
            return
        
        if not text:
            return 

        try:
            self.debug(text)
        except UnicodeEncodeError:
            self.debug(text.encode('ascii', errors='replace').decode('ascii'))

    def close(self):
        for handler in self.handlers:
            handler.close()
            self.removeHandler(handler)
        sys.stdout = self._stdout
        self._stderr.close()
    
    def __del__(self):
        sys.stdout = self._stdout
        self._stderr.close()
    
    def info(self, text, *args, **kwargs):
        super().info(text, *args, **kwargs)
        try:
            self.write(f'{text}\n', log_to_file=False)
        except TypeError:
            # Sometimes the logger is patched (e.g., by spotiflow), which
            # triggers the TypeError because the patching function does not have 
            # log_to_file argument
            self.write(f'{text}\n')
    
    def warning(self, text, *args, **kwargs):
        super().warning(text, *args, **kwargs)
        try:
            self.write(f'[WARNING]: {text}\n', log_to_file=False)
        except TypeError:
            # Sometimes the logger is patched (e.g., by spotiflow), which
            # triggers the TypeError because the patching function does not have 
            # log_to_file argument
            self.write(f'[WARNING]: {text}\n')
    
    def error(self, text, *args, write_traceback=True, **kwargs):
        super().error(text, *args, **kwargs)
        self.write(traceback.format_exc())
        try:
            self.write(f'[ERROR]: {text}\n', log_to_file=False)
        except TypeError:
            # Sometimes the logger is patched (e.g., by spotiflow), which
            # triggers the TypeError because the patching function does not have 
            # log_to_file argument
            self.write(f'[ERROR]: {text}\n')
    
    def plain(self, text, write_to_stdout=False):
        orig_formatters = [handler.formatter for handler in self.handlers]
        for handler in self.handlers:
            handler.setFormatter(logging.Formatter('%(message)s'))
        self.write(text, write_to_stdout=write_to_stdout)
        for handler in self.handlers:
            handler.setFormatter(orig_formatters.pop(0))
    
    def critical(self, text, *args, **kwargs):
        super().critical(text, *args, **kwargs)
        try:
            self.write(f'[CRITICAL]: {text}\n', log_to_file=False)
        except TypeError:
            # Sometimes the logger is patched (e.g., by spotiflow), which
            # triggers the TypeError because the patching function does not have 
            # log_to_file argument
            self.write(f'[CRITICAL]: {text}\n')
    
    def exception(self, text, *args, write_traceback=True, **kwargs):
        super().exception(text, *args, **kwargs)
        self.write(traceback.format_exc())
        try:
            self.write(f'[ERROR]: {text}\n', log_to_file=False)
        except TypeError:
            # Sometimes the logger is patched (e.g., by spotiflow), which
            # triggers the TypeError because the patching function does not have 
            # log_to_file argument
            self.write(f'[ERROR]: {text}\n')
    
    def log(self, level, text):
        if not isinstance(level, int):
            printl(level, text, type(level), type(text), sep='\n')
        super().log(level, text)
        levelName = self._levelToName.get(level, 'INFO')
        getattr(self, levelName.lower())(text)
    
    def flush(self):
        self._stdout.flush()

class StdErr:
    def __init__(self, logger: Logger=None):
        self._sys_stderr = sys.stderr
        self._err_msg_line_buffer = []
        self._logger = logger
    
    def write(self, text: str):     
        if text.startswith('Traceback'):
            print('-'*100)
            
        self._sys_stderr.write(text)
        
        if not text:
            return
        
        self._err_msg_line_buffer.append(text)
        if not text.endswith('\n'):
            return
        
        # If the line ends with a newline, flush the buffer
        err_line = ''.join(self._err_msg_line_buffer)
        if self._logger is not None:
            self._logger.plain(err_line, write_to_stdout=False)
        else:
            print(err_line)
        
        self._err_msg_line_buffer = []
            
    def flush(self):
        self._sys_stderr.flush()
    
    def close(self):
        """Close the StdErr stream"""
        sys.stderr = self._sys_stderr

def delete_older_log_files(logs_path):
    if not os.path.exists(logs_path):
        return
    
    log_files = os.listdir(logs_path)
    for log_file in log_files:
        if not log_file.endswith('.log'):
            continue
        
        log_filepath = os.path.join(logs_path, log_file)
        try:
            mtime = os.path.getmtime(log_filepath)
        except Exception as err:
            continue
        
        mdatetime = datetime.datetime.fromtimestamp(mtime)
        days = (datetime.datetime.now() - mdatetime).days
        if days < 7:
            continue

        try:
            os.remove(log_filepath)
        except Exception as err:
            continue

def get_info_version_text(is_cli=False, cli_formatted_text=True):
    version = read_version()
    release_date = get_date_from_version(version, package='cellacdc')
    py_ver = sys.version_info
    env_folderpath = sys.prefix
    python_version = f'{py_ver.major}.{py_ver.minor}.{py_ver.micro}'
    info_txts = [
        f'Version: {version}',
        f'Released on: {release_date}',
        f'Installed in: "{cellacdc_path}"',
        f'Environment folder: "{env_folderpath}"',
        f'User profile folder: "{user_profile_path}"',
        f'Settings folder: "{settings_folderpath}"',
        f'Python version: {python_version}',
        f'Platform: {platform.platform()}',
        f'System: {platform.system()}',
    ]
    if is_linux:
        try:
            distro_name = get_linux_distribution_name()
        except Exception as err:
            distro_name = 'Undetermined'
        
        info_txts.append(f'Linux distribution: {distro_name}')
    
    if GUI_INSTALLED and not is_cli:
        info_txts.append(f'Icons from: "{qrc_resources_path}"')
        try:
            from qtpy import QtCore
            info_txts.append(f'Qt version: {QtCore.__version__}')
        except Exception as err:
            info_txts.append('Qt: Not installed')
    
    try:
        branch_name = get_git_branch_name()
        info_txts.append(f'Git branch: "{branch_name}"')
    except Exception as err:
        pass

    from cellacdc import IS_CLONED
    if IS_CLONED:        
        try:
            remote_repos = get_git_remote_repos_names()
            indentation = ' '*8
            remote_repos_txt = '\n'.join(
                [f'{indentation}* {repo}' for repo in remote_repos])
            info_txts.append(f'Git remote repositories:\n{remote_repos_txt}')
        except Exception as err:
            pass

    info_txts.append(f'Working directory: {os.getcwd()}')
    
    if not cli_formatted_text:
        return info_txts
    
    info_txts = [f'  - {txt}' for txt in info_txts]
    info_txt = '\n'.join(info_txts)
    info_txts = [line.expandtabs() for line in info_txt.splitlines()]
    max_len = max([len(line) for line in info_txts]) + 2
    
    formatted_info_txts = []
    for txt in info_txts:
        horiz_spacing = ' '*(max_len - len(txt))
        txt = f'{txt}{horiz_spacing}|'
        formatted_info_txts.append(txt)
    
    formatted_info_txts.insert(0, 'Cell-ACDC info:\n')
    formatted_info_txts.insert(0, '='*max_len)
    formatted_info_txts.append('='*max_len)
    info_txt = '\n'.join(formatted_info_txts)
    
    try:
        from spotmax.utils import get_info_version_text as smax_info
        smax_info_txt = smax_info(include_platform=False, is_cli=is_cli)
        info_txt += '\n\n' + smax_info_txt
    except ImportError:
        pass
    
    return info_txt

def _log_system_info(logger, log_path, is_cli=False, also_spotmax=False):
    logger.info(f'Initialized log file "{log_path}"')
    
    info_txt = get_info_version_text(is_cli=is_cli)
    
    logger.info(info_txt)
    
    if not also_spotmax:
        return
    
    from spotmax.utils import get_info_version_text as smax_info
    smax_info_txt = smax_info(include_platform=False)
    logger.info(smax_info_txt)

def setupLogger(
        module='base', 
        logs_path=None, 
        caller='Cell-ACDC', 
        QLogWidget=None
    ):
    if logs_path is None:
        logs_path = get_logs_path()
    
    logger = Logger(module=module, QLogWidget=QLogWidget)
    sys.stdout = logger
    
    delete_older_log_files(logs_path)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)

    date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    id = uuid4()
    log_filename = f'{date_time}_{module}_{id}_stdout.log'
    log_path = os.path.join(logs_path, log_filename)

    output_file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')

    # Format your logs (optional)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s:\n'
        '------------------------\n'
        '%(message)s\n'
        '------------------------\n',
        datefmt='%d-%m-%Y, %H:%M:%S'
    )
    output_file_handler.setFormatter(formatter)

    logger.addHandler(output_file_handler)
    
    _log_system_info(logger, log_path, also_spotmax=caller!='Cell-ACDC')
    
    # if module == 'gui' and GUI_INSTALLED:
    #     qt_handler = widgets.QtHandler()
    #     qt_handler.setFormatter(logging.Formatter("%(message)s"))
    #     logger.addHandler(qt_handler)

    return logger, logs_path, log_path, log_filename


def read_version(logger=None, return_success=False):
    cellacdc_parent_path = os.path.dirname(cellacdc_path)
    cellacdc_parent_folder = os.path.basename(cellacdc_parent_path)
    if cellacdc_parent_folder == 'site-packages':
        from cellacdc import __version__
        version = __version__
        success = True
    else:
        try:
            from setuptools_scm import get_version
            version = get_version(
                root='..',
                relative_to=os.path.join(cellacdc_path, '__init__.py')
            )
            success = True
        except Exception as e:
            if logger is None:
                logger = print
            logger('*'*40)
            logger(traceback.format_exc())
            logger('-'*40)
            logger(
                '[WARNING]: Cell-ACDC could not determine the current version. '
                'Returning the version determined at installation time. '
                'See details above.'
            )
            logger('='*40)
            try:
                from cellacdc import _version
                version = _version.version
                success = False
            except Exception as e:
                version = 'ND'
                success = False
    
    if return_success:
        return version, success
    else:
        return version

def get_date_from_version(version: str, package='cellacdc', debug=False):
    try:
        response = requests.get(
            f'https://pypi.org/pypi/{package}/json', 
            timeout=2
        )
        res_json = response.json()
        pypi_releases_json = res_json['releases']
        version_json = pypi_releases_json[version][0]
        upload_time = version_json['upload_time_iso_8601']
        date = datetime.datetime.strptime(
            upload_time, r'%Y-%m-%dT%H:%M:%S.%fZ'
        )
        date_str = date.strftime(r'%A %d %B %Y at %H:%M')
        return date_str
    except Exception as err:
        if debug:
            traceback.print_exc()
    
    try:
        # Locate the direct_url.json file for the package 
        # installed with pip git+
        dist = importlib.metadata.distribution(package)
        dist_info_dir = dist._path  # internal path to .dist-info
        direct_url_path = os.path.join(dist_info_dir, "direct_url.json")

        with open(direct_url_path) as f:
            data = json.load(f)

        vcs_info = data["vcs_info"]
        commit_id = vcs_info.get("commit_id")
        url = data.get("url")

        parts = url.split("github.com/")[1].split(".git")[0]
        owner, repo = parts.split("/", 1)

        # Query GitHub API for commit date
        api_url = (
            f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_id}"
        )
        response = requests.get(api_url)
        response.raise_for_status()

        commit_data = response.json()
        date_utc = commit_data["commit"]["committer"]["date"]
        
        date_str = format_commit_date_utc(date_utc)
        
        return date_str
    except Exception as err:
        if debug:
            traceback.print_exc()
            
    try:
        if package == 'cellacdc':
            pkg_path = cellacdc_path
        elif package == 'spotmax':
            from spotmax import spotmax_path
            pkg_path = spotmax_path
        commit_hash = re.findall(r'\+g([A-Za-z0-9]+)(\.d)?', version)[0][0]
        git_path = os.path.dirname(pkg_path)
        command = f'git -C {git_path} show {commit_hash}'
        commit_log = installs._subprocess_run_command(
            command, shell=False, callback='check_output'
        )
        commit_log = commit_log.decode() 
        date_log = re.findall(r'Date:(.*) \+', commit_log)[0].strip()
        date = datetime.datetime.strptime(date_log, r'%a %b %d %H:%M:%S %Y')
        date_str = date.strftime(r'%A %d %B %Y at %H:%M')
        return date_str
    except Exception as err:
        if debug:
            traceback.print_exc()
    
    return 'ND'  

def get_git_branch_name():
    # First, try the existing Git checkout approach.
    try:
        result = subprocess.run(
            [
                'git',
                '-C',
                str(cellacdc_installation_path),
                'rev-parse',
                '--abbrev-ref',
                'HEAD',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        branch_name = result.stdout.strip()

        if branch_name and branch_name != 'HEAD':
            return branch_name

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fall back to pip's installation metadata.
    # Find the distribution corresponding to the cellacdc package.
    distributions = metadata.packages_distributions().get(
        'cellacdc', []
    )

    for dist_name in distributions:
        try:
            dist = metadata.distribution(dist_name)
            direct_url = dist.read_text('direct_url.json')

            if not direct_url:
                continue

            data = json.loads(direct_url)
            vcs_info = data.get('vcs_info', {})

            if vcs_info.get('vcs') == 'git':
                return vcs_info.get('requested_revision')

        except (metadata.PackageNotFoundError, json.JSONDecodeError):
            continue

    return None

def get_git_remote_repos_names():
    command = (
        f'git -C "{cellacdc_installation_path}" remote -v'
    )
    output = installs._subprocess_run_command(
        command, shell=False, callback='check_output'
    )
    remote_names = [
        line.expandtabs() for line in output.decode().strip().splitlines()
    ]
    return remote_names


def format_commit_date_utc(utc_str):
    # Parse the UTC date string (ISO 8601 format)
    dt = datetime.datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    
    # Convert to your local time zone (optional)
    local_dt = dt.astimezone()  # removes UTC offset if local
    
    # Format nicely
    return local_dt.strftime(r"%A %d %B %Y at %H:%M")

def get_linux_distribution_name():
    import csv
    RELEASE_DATA = {}
    with open("/etc/os-release") as f:
        reader = csv.reader(f, delimiter="=")
        for row in reader:
            if row:
                RELEASE_DATA[row[0]] = row[1]
    if RELEASE_DATA["ID"] in ["debian", "raspbian"]:
        with open("/etc/debian_version") as f:
            DEBIAN_VERSION = f.readline().strip()
        major_version = DEBIAN_VERSION.split(".")[0]
        version_split = RELEASE_DATA["VERSION"].split(" ", maxsplit=1)
        if version_split[0] == major_version:
            # Just major version shown, replace it with the full version
            RELEASE_DATA["VERSION"] = " ".join([DEBIAN_VERSION] + version_split[1:])
    
    name_version = f'{RELEASE_DATA["NAME"]} {RELEASE_DATA["VERSION"]}'
    
    return name_version
