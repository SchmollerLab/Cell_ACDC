"""Small git helpers (subprocess, no third-party packages)."""
import os
import subprocess

from . import RefactorError


def git(repo, *args, check=True, input=None):
    proc = subprocess.run(
        ['git', '-C', repo, *args], capture_output=True, input=input
    )
    if check and proc.returncode != 0:
        raise RefactorError(
            f'git {" ".join(args)} failed in {repo}:\n'
            f'{proc.stderr.decode("utf-8", "replace")}'
        )
    return proc


def git_text(repo, *args, check=True):
    return git(repo, *args, check=check).stdout.decode('utf-8', 'replace')


class BaseTree:
    """Read-only access to the files of a commit (git show / ls-tree).

    Contents come from the object store (index form: LF line endings),
    which is what the mapping keys and the coverage checks need. Works in
    any worktree of the repository and at any checked out commit.
    """

    def __init__(self, repo, commit):
        self.repo = repo
        self.commit = commit
        self._cache = {}
        self._files = None

    def files(self):
        if self._files is None:
            out = git_text(
                self.repo, 'ls-tree', '-r', '--name-only', self.commit
            )
            self._files = set(out.splitlines())
        return self._files

    def exists(self, path):
        return path in self.files()

    def is_dir(self, path):
        prefix = path.rstrip('/') + '/'
        return any(f.startswith(prefix) for f in self.files())

    def read_bytes(self, path):
        if path not in self._cache:
            proc = git(
                self.repo, 'show', f'{self.commit}:{path}', check=False
            )
            if proc.returncode != 0:
                raise RefactorError(
                    f'{path} does not exist at {self.commit}'
                )
            self._cache[path] = proc.stdout
        return self._cache[path]

    def read_text(self, path):
        from .source import SourceText
        return SourceText.from_bytes(self.read_bytes(path)).text


def git_mv(repo, old, new):
    """git mv old -> new, creating the parent folders of `new`."""
    parent = os.path.dirname(os.path.join(repo, new))
    os.makedirs(parent, exist_ok=True)
    git(repo, 'mv', old, new)


def git_rm(repo, path):
    git(repo, 'rm', '-q', '-r', path)


def git_add(repo, *paths):
    if paths:
        git(repo, 'add', '--', *paths)


def is_ignored(repo, paths):
    """The subset of `paths` that .gitignore rules ignore."""
    if not paths:
        return set()
    proc = git(
        repo, 'check-ignore', '--no-index', '--stdin', check=False,
        input='\n'.join(paths).encode('utf-8')
    )
    return set(proc.stdout.decode('utf-8').splitlines())
