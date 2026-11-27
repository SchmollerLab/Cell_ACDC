"""Reproducible pipeline of the package-layout refactor.

Commands (run from any directory; --repo defaults to this checkout):

    python tools/refactor/run.py apply <commit> [--fixups] [--repo R]
        Apply the automatic step of one commit to the working tree:
          C2       write cellacdc/_compat.py (from
                   acdc_refactor/templates/_compat.py) and an empty
                   cellacdc/_compat_moves.py
          C3..C11  module moves, item moves, class splits, shims and
                   compat blocks (acdc_refactor/move.py, shims.py), then
                   cellacdc/_compat_moves.py for the new state
          C12      acdc_refactor/rewrite.py over cellacdc/, tests/ (hidden
                   tests/**/.test_*.py included) and the code cells of
                   notebooks/*.ipynb (imports of old modules and moved
                   names left unused are dropped, third-party names are
                   imported from their origin), then the intermediate
                   compat blocks are removed
          C14      module docstrings of modules entries with doc_commit C14
          C16      .git-blame-ignore-revs from the SHAs of C12 and C14
                   (after the C16 fixups; replay passes the SHAs, `apply`
                   reads ACDC_REFACTOR_SHAS or finds the commits in the
                   history by the subjects of commit_messages/)
          C0, C1, C13, C15: nothing automatic (fixups only)
        With --fixups, the scripts of tools/refactor/fixups/<commit>/ run
        afterwards. Edits are logged to tools/refactor/out/<commit>/.

    python tools/refactor/run.py check <commit> [--slow] [--quick]
        plan_check of the mapping, then the verify scripts
        (tools/refactor/verify, run_all) when present; writes the name
        table of the commit to <out>/<commit>/name_table.json and the
        final one to <out>/name_table.json for them and compares with
        the snapshots in <out>/baseline/.

    python tools/refactor/run.py replay --base <sha> --until <commit>
            --branch <name> [--worktree <path>] [--from C3]
        Rebuild the commits in a fresh worktree created from <base>: for
        every commit up to <until>, the automatic step, then the fixups
        (tools/refactor/fixups/<commit>/*.py, in name order), then
        `git add -A` and a commit with the message of
        tools/refactor/commit_messages/<commit>.txt (or a default). C0
        copies tools/refactor (without out/) into the worktree. The tools,
        mapping and fixups are always taken from this checkout; the mapping
        keys are read at <base> (ACDC_REFACTOR_BASE).

    python tools/refactor/run.py name-table [--commit C] [--out FILE]

Fixup scripts: plain Python scripts, run as `python <script> <repo root>`
with the repository root as working directory and the environment variables
ACDC_REFACTOR_COMMIT (commit id), ACDC_REFACTOR_REPO (repo root),
ACDC_REFACTOR_BASE (base sha) and ACDC_REFACTOR_SHAS (JSON {commit: sha} of
the commits made so far by replay, e.g. for .git-blame-ignore-revs in C16).
A script that exits with a non-zero code stops the pipeline.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from acdc_refactor import (  # noqa: E402
    COMMITS, MOVE_COMMITS, DEFAULT_REPO_ROOT, RefactorError, commit_le,
    base_commit
)

# ACDC_REFACTOR_FIXUPS: another fixups folder (tests use an empty one)
FIXUPS_DIR = os.environ.get('ACDC_REFACTOR_FIXUPS') or os.path.join(HERE, 'fixups')
MESSAGES_DIR = os.path.join(HERE, 'commit_messages')
MAPPING_DIR = os.path.join(HERE, 'mapping')
OUT_DIR = os.environ.get('ACDC_REFACTOR_OUT') or os.path.join(HERE, 'out')
#: The folders rewritten in C12 (rewrite.py walks dotfiles such as
#: tests/segm/.test_sam.py; --notebooks adds the code cells of *.ipynb)
REWRITE_PATHS = ['cellacdc', 'tests', 'notebooks']
#: Commits listed in .git-blame-ignore-revs (C16): the reference rewrite
#: and the docstring/comment update. The move commits are not listed:
#: `git blame -C -C` follows moved code to its old file.
BLAME_IGNORE_COMMITS = ['C12', 'C14']
BLAME_BEGIN = '# >>> package-layout refactor (tools/refactor/run.py, C16)'
BLAME_END = '# <<< package-layout refactor'
BLAME_HEADER = [
    '# Commits that only rewrote import paths, references, docstrings and',
    '# comments for the package layout of CODEMAP.md. `git blame` skips',
    '# them with `git config blame.ignoreRevsFile .git-blame-ignore-revs`',
    '# (GitHub reads this file by itself). The commits that moved code are',
    '# not listed: `git blame -C -C <file>` follows lines to their old',
    '# module.',
]
DEFAULT_MESSAGES = {
    'C0': 'tools: codemod, mapping and verification of the package layout',
    'C1': 'refactor: prepare the old layout for the package moves',
    'C2': 'refactor: add the compat layer for moved import paths',
    'C3': 'refactor: move whole modules into program-part packages',
    'C4': 'refactor: split the existing GUI mixin modules',
    'C5': 'refactor: split widgets.py',
    'C6': 'refactor: split apps.py',
    'C7': 'refactor: split workers.py',
    'C8': 'refactor: split myutils.py into the myutils package',
    'C9': 'refactor: split the measurements viewer dialogs',
    'C10': 'refactor: split guiWin into mixins',
    'C11': 'refactor: split the measurements viewer window into mixins',
    'C12': 'refactor: use the new import paths in cellacdc and the tests',
    'C13': 'test: enforce the new import paths',
    'C14': 'docs: update docstrings and comments for the new layout',
    'C15': 'docs: CODEMAP.md and the docs for the new layout',
    'C16': 'chore: .git-blame-ignore-revs for the reference rewrites',
}


def log(message):
    print(f'[run.py] {message}', flush=True)


def load_plan(repo, mapping_dir=None):
    from acdc_refactor.plan import Plan
    return Plan(repo, mapping_dir=mapping_dir or MAPPING_DIR)


# ------------------------------------------------------------------ apply
def write_name_table(repo, commit, out=None, mapping_dir=None):
    plan = load_plan(repo, mapping_dir)
    table = plan.name_table(commit)
    out = out or os.path.join(OUT_DIR, commit, 'name_table.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as file:
        json.dump(table, file, indent=1, sort_keys=True)
    return out


def run_rewrite(repo, commit, paths, mapping_dir=None):
    """rewrite.py over `paths` of `repo`, code cells of notebooks
    included; imports of old modules and of moved names that are unused
    afterwards are dropped (--drop-unused-imports), and third-party names
    read through cellacdc modules (Qt classes, stdlib modules) are
    imported from their origin (--external-origins: shared modules never
    import a program part for a Qt name). The report is
    out/<commit>/rewrite_report.json ("dropped": the removed imports)."""
    from acdc_refactor import rewrite
    table = write_name_table(repo, COMMITS[-1],
                             os.path.join(OUT_DIR, commit,
                                          'name_table.json'), mapping_dir)
    report = os.path.join(OUT_DIR, commit, 'rewrite_report.json')
    existing = [p for p in paths if os.path.exists(os.path.join(repo, p))]
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        code = rewrite.main(['--table', table, '--report', report,
                             '--src-root', repo, '--notebooks',
                             '--drop-unused-imports', '--external-origins']
                            + existing)
    finally:
        os.chdir(cwd)
    if code:
        raise RefactorError(f'rewrite.py exited with {code}')
    return table


def resolve_shas(repo, shas=None):
    """{commit: sha} of the refactor commits: `shas` (replay), then
    ACDC_REFACTOR_SHAS (JSON), then, for the commits still missing, the
    newest commit of HEAD's history whose subject is the first line of
    commit_messages/<commit>.txt."""
    found = {}
    env = os.environ.get('ACDC_REFACTOR_SHAS')
    if env:
        try:
            found.update(json.loads(env))
        except ValueError:
            pass
    found.update(shas or {})
    missing = [c for c in BLAME_IGNORE_COMMITS if not found.get(c)]
    if missing:
        log_lines = git(repo, 'log', '--format=%H %s', check=False)
        for line in log_lines.splitlines():
            sha, _, subject = line.partition(' ')
            for commit in list(missing):
                if subject == commit_message(commit).splitlines()[0]:
                    found[commit] = sha
                    missing.remove(commit)
    return found


def blame_ignore_text(shas, existing=''):
    """Text of .git-blame-ignore-revs: the refactor block (header, then
    per commit of BLAME_IGNORE_COMMITS a comment with its id and subject
    and its SHA) replacing an earlier refactor block of `existing`; other
    entries of `existing` are kept."""
    lines = [BLAME_BEGIN] + BLAME_HEADER
    for commit in BLAME_IGNORE_COMMITS:
        subject = commit_message(commit).splitlines()[0]
        lines += ['', f'# {commit}: {subject}', shas[commit]]
    lines.append(BLAME_END)
    block = '\n'.join(lines)
    existing = existing.replace('\r\n', '\n')
    if BLAME_BEGIN in existing and BLAME_END in existing:
        before, _, rest = existing.partition(BLAME_BEGIN)
        _, _, after = rest.partition(BLAME_END)
        parts = [before.strip('\n'), block, after.strip('\n')]
    else:
        parts = [existing.strip('\n'), block]
    return '\n\n'.join(p for p in parts if p) + '\n'


def write_blame_ignore_revs(repo, shas=None):
    """Write .git-blame-ignore-revs (C16, after everything else) from the
    final SHAs of the commits in BLAME_IGNORE_COMMITS."""
    shas = resolve_shas(repo, shas)
    missing = [c for c in BLAME_IGNORE_COMMITS if not shas.get(c)]
    if missing:
        raise RefactorError('.git-blame-ignore-revs: no SHA for '
                            + ', '.join(missing) + ' (no commit with its '
                            'subject in the history; set ACDC_REFACTOR_SHAS)')
    path = os.path.join(repo, '.git-blame-ignore-revs')
    existing = ''
    if os.path.exists(path):
        with open(path, encoding='utf-8') as file:
            existing = file.read()
    with open(path, 'w', encoding='utf-8', newline='\n') as file:
        file.write(blame_ignore_text(shas, existing))
    log('wrote .git-blame-ignore-revs ('
        + ', '.join(f'{c} {shas[c][:10]}' for c in BLAME_IGNORE_COMMITS)
        + ')')
    return path


def apply_docstrings(repo, plan, commit):
    """Add the `doc` of modules entries with doc_commit == commit to the
    modules that have no docstring."""
    import ast
    from acdc_refactor.source import SourceText
    from acdc_refactor.textutil import format_docstring
    done = []
    for path, spec in sorted(plan.modules.items()):
        if spec.get('doc_commit') != commit or not spec.get('doc'):
            continue
        full = os.path.join(repo, plan.module_path(path, commit))
        if not os.path.exists(full):
            full = os.path.join(repo, path)
        if not os.path.exists(full):
            continue
        src = SourceText.read(full)
        tree = ast.parse(src.text)
        if ast.get_docstring(tree, clean=False) is not None:
            continue
        doc = format_docstring(spec['doc']).replace('\n', src.eol) + src.eol
        lines = src.lines
        k = 0
        while k < len(lines) and lines[k].lstrip().startswith('#') and \
                ('coding' in lines[k] or lines[k].startswith('#!')):
            k += 1
        new = ''.join(lines[:k]) + doc + ''.join(lines[k:])
        src.with_text(new).write(full)
        done.append(path)
    return done


def apply(commit, repo, mapping_dir=None, fixups=False, shas=None):
    from acdc_refactor import shims
    from acdc_refactor import move
    os.environ.setdefault('ACDC_REFACTOR_BASE', base_commit())
    if commit == 'C2':
        path = shims.write_compat_module(repo)
        plan = load_plan(repo, mapping_dir)
        shims.write_compat_moves(repo, plan, 'C2')
        log(f'wrote {path} and cellacdc/_compat_moves.py')
    elif commit in MOVE_COMMITS:
        plan = load_plan(repo, mapping_dir)
        errors = [i for i in plan.issues if i.level == 'error']
        if errors:
            raise RefactorError('The mapping has errors:\n' + '\n'.join(
                str(e) for e in errors))
        out = os.path.join(OUT_DIR, commit)
        changes = move.apply_commit(repo, plan, commit, out_dir=out)
        log(f'{commit}: {len(changes.entries)} edits logged in {out}')
        # modules entries with doc_commit == commit (e.g. the viewer
        # window.py once its docstring moved to the package in C11)
        done = apply_docstrings(repo, plan, commit)
        if done:
            log(f'{commit}: added {len(done)} module docstrings: '
                + ', '.join(done))
    elif commit == 'C12':
        run_rewrite(repo, commit, REWRITE_PATHS, mapping_dir)
        removed = shims.remove_intermediate_blocks(repo)
        log(f'removed {len(removed)} intermediate compat blocks')
    elif commit == 'C14':
        plan = load_plan(repo, mapping_dir)
        done = apply_docstrings(repo, plan, commit)
        log(f'added {len(done)} module docstrings')
    elif commit == 'C16':
        pass    # written after the C16 fixups (below), from the final SHAs
    else:
        log(f'{commit}: no automatic step')
    if fixups:
        run_fixups(commit, repo, shas)
    if commit == 'C16':
        write_blame_ignore_revs(repo, shas)


# ----------------------------------------------------------------- fixups
def fixup_scripts(commit):
    folder = os.path.join(FIXUPS_DIR, commit)
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, name) for name in sorted(os.listdir(folder))
            if name.endswith('.py') and not name.startswith('_')]


def run_fixups(commit, repo, shas=None):
    env = dict(os.environ)
    env['ACDC_REFACTOR_COMMIT'] = commit
    env['ACDC_REFACTOR_REPO'] = repo
    env['ACDC_REFACTOR_BASE'] = base_commit()
    env['ACDC_REFACTOR_SHAS'] = json.dumps(shas or {})
    for script in fixup_scripts(commit):
        log(f'fixup {os.path.relpath(script, HERE)}')
        proc = subprocess.run([sys.executable, script, repo], cwd=repo,
                              env=env, timeout=600)
        if proc.returncode != 0:
            raise RefactorError(f'fixup {script} failed ({proc.returncode})')


# ------------------------------------------------------------------ check
def check(commit, repo, slow=False, quick=False, mapping_dir=None):
    from acdc_refactor import plan_check
    code = plan_check.run(repo, commit=commit, deep=not quick,
                          mapping_dir=mapping_dir or MAPPING_DIR)
    verify_dir = os.path.join(HERE, 'verify')
    if os.path.isfile(os.path.join(verify_dir, '__init__.py')):
        # the verify scripts read the name table of the checked commit
        # (<out>/<commit>/name_table.json: where every old name lives at
        # that commit, so that the fingerprints of C3-C9 compare with the
        # intermediate homes), the final one (<out>/name_table.json) and
        # the base snapshots (baseline/) from OUT_DIR
        write_name_table(repo, COMMITS[-1],
                         os.path.join(OUT_DIR, 'name_table.json'),
                         mapping_dir)
        write_name_table(repo, commit,
                         os.path.join(OUT_DIR, commit, 'name_table.json'),
                         mapping_dir)
        if not os.path.isdir(os.path.join(OUT_DIR, 'baseline')):
            log(f'{OUT_DIR}/baseline is missing: copy '
                'tools/refactor/out/baseline there (snapshots of the base)')
        try:
            import verify
        except ImportError as err:
            log(f'cannot import verify: {err}')
            return code or 1
        if hasattr(verify, 'run_all'):
            reports = verify.run_all(commit, repo=repo, slow=slow)
        else:
            reports = {}
            for name in sorted(os.listdir(verify_dir)):
                if not name.endswith('.py') or name.startswith('_'):
                    continue
                module = __import__(f'verify.{name[:-3]}',
                                    fromlist=['run'])
                if hasattr(module, 'run'):
                    reports[name[:-3]] = module.run(commit)
        failed = []
        for name, report in reports.items():
            ok = report.get('ok', True) if isinstance(report, dict) else \
                bool(report)
            log(f'verify {name}: {"ok" if ok else "FAILED"}')
            if not ok:
                failed.append(name)
        if failed:
            code = code or 1
    return code


# ----------------------------------------------------------------- replay
def git(repo, *args, check=True):
    proc = subprocess.run(['git', '-C', repo, *args], capture_output=True,
                          text=True)
    if check and proc.returncode != 0:
        raise RefactorError(f'git {" ".join(args)}: {proc.stderr}')
    return proc.stdout.strip()


def copy_tools(dest_repo):
    dest = os.path.join(dest_repo, 'tools', 'refactor')
    if os.path.abspath(dest) == os.path.abspath(HERE):
        return
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(HERE, dest, ignore=shutil.ignore_patterns(
        'out', '__pycache__', '*.pyc'))


def commit_message(commit):
    path = os.path.join(MESSAGES_DIR, f'{commit}.txt')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as file:
            return file.read().strip() + '\n'
    return DEFAULT_MESSAGES.get(commit, f'refactor: {commit}') + \
        f'\n\nGenerated by tools/refactor/run.py ({commit}).\n'


def replay(base, until, branch, worktree=None, start=None,
           mapping_dir=None, repo=DEFAULT_REPO_ROOT):
    """Rebuild the commits in a new worktree of `repo` (see the module
    docstring); returns {commit: sha}."""
    os.environ['ACDC_REFACTOR_BASE'] = base
    worktree = os.path.abspath(worktree or os.path.join(
        OUT_DIR, 'replay', branch.replace('/', '_')))
    if os.path.exists(worktree):
        raise RefactorError(f'{worktree} exists: remove it first (git '
                            'worktree remove --force)')
    git(repo, 'worktree', 'add', '-b', branch, worktree, base)
    log(f'worktree {worktree} on {branch} from {base}')
    shas = {}
    start = start or COMMITS[0]
    for commit in COMMITS:
        if not commit_le(start, commit):
            continue
        if not commit_le(commit, until):
            break
        log(f'--- {commit}')
        if commit == 'C0':
            copy_tools(worktree)
            run_fixups(commit, worktree, shas)
        else:
            apply(commit, worktree, mapping_dir=mapping_dir, fixups=True,
                  shas=shas)
        git(worktree, 'add', '-A')
        if not git(worktree, 'status', '--porcelain'):
            log(f'{commit}: nothing to commit')
            continue
        message_file = os.path.join(OUT_DIR, f'{commit}_message.txt')
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(message_file, 'w', encoding='utf-8') as file:
            file.write(commit_message(commit))
        git(worktree, 'commit', '-q', '-F', message_file)
        shas[commit] = git(worktree, 'rev-parse', 'HEAD')
        log(f'{commit}: {shas[commit]}')
    with open(os.path.join(OUT_DIR, 'replay_shas.json'), 'w',
              encoding='utf-8') as file:
        json.dump({'base': base, 'branch': branch, 'worktree': worktree,
                   'shas': shas}, file, indent=1)
    return shas


# ------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Package-layout refactor pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--repo', default=DEFAULT_REPO_ROOT)
    parser.add_argument('--mapping', default=None,
                        help='mapping folder (default: tools/refactor/'
                        'mapping of this checkout)')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('apply')
    p.add_argument('commit', choices=COMMITS)
    p.add_argument('--fixups', action='store_true')
    p = sub.add_parser('check')
    p.add_argument('commit', choices=COMMITS)
    p.add_argument('--slow', action='store_true')
    p.add_argument('--quick', action='store_true',
                   help='plan_check without the import-cycle analysis')
    p = sub.add_parser('replay')
    p.add_argument('--base', required=True)
    p.add_argument('--until', default=COMMITS[-1], choices=COMMITS)
    p.add_argument('--branch', required=True)
    p.add_argument('--worktree', default=None)
    p.add_argument('--from', dest='start', default=None, choices=COMMITS)
    p = sub.add_parser('name-table')
    p.add_argument('--commit', default=COMMITS[-1], choices=COMMITS)
    p.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    repo = os.path.abspath(args.repo)
    try:
        if args.cmd == 'apply':
            apply(args.commit, repo, args.mapping, fixups=args.fixups)
            return 0
        if args.cmd == 'check':
            return check(args.commit, repo, slow=args.slow,
                         quick=args.quick, mapping_dir=args.mapping)
        if args.cmd == 'replay':
            replay(args.base, args.until, args.branch, args.worktree,
                   args.start, args.mapping, repo=repo)
            return 0
        if args.cmd == 'name-table':
            out = write_name_table(repo, args.commit, args.out,
                                   args.mapping)
            log(f'wrote {out}')
            return 0
    except RefactorError as err:
        print(f'ERROR: {err}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
