"""Helpers of the fixup scripts (tools/refactor/fixups/<commit>/*.py).

Fixups are exact, idempotent text edits: each replacement names the old
text and the new text; a file that already holds the new text (and not the
old one) is left alone, a file holding neither stops the pipeline, so a
rebuild on a newer main reports what changed. Line endings of the file are
kept (CRLF working trees stay CRLF): the texts are given with '\n'.
"""
import inspect
import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FixupError(Exception):
    pass


def _commit():
    commit = os.environ.get('ACDC_REFACTOR_COMMIT')
    if commit:
        return commit
    # run by hand: the folder of the calling script (fixups/<commit>/x.py)
    for frame in inspect.stack():
        folder = os.path.basename(os.path.dirname(frame.filename))
        if folder.startswith('C') and folder[1:].isdigit():
            return folder
    return 'C?'


def log_change(relpath, items, detail, kind='fixup'):
    """Record a fixup edit in <out>/<commit>/changes.jsonl (the file the
    codemod writes for the commit; verify/fingerprint.py excuses the listed
    items). `items` are `<file at that commit>::<Qual>` identifiers."""
    if isinstance(items, str):
        items = [items]
    out = os.environ.get('ACDC_REFACTOR_OUT') or os.path.join(TOOLS_DIR,
                                                               'out')
    commit = _commit()
    folder = os.path.join(out, commit)
    os.makedirs(folder, exist_ok=True)
    entry = {'commit': commit, 'category': 'other', 'kind': kind,
             'file': relpath, 'detail': detail, 'items': list(items),
             'script': os.path.basename(sys.argv[0])}
    with open(os.path.join(folder, 'changes.jsonl'), 'a',
              encoding='utf-8') as file:
        file.write(json.dumps(entry, sort_keys=True) + '\n')


def repo_root():
    if len(sys.argv) > 1:
        return os.path.abspath(sys.argv[1])
    return os.environ.get('ACDC_REFACTOR_REPO') or os.getcwd()


def _read(path):
    with open(path, 'rb') as file:
        data = file.read()
    eol = '\r\n' if b'\r\n' in data else '\n'
    text = data.decode('utf-8')
    return text, eol


def _write(path, text):
    with open(path, 'wb') as file:
        file.write(text.encode('utf-8'))


def replace(repo, relpath, old, new, count=1, items=()):
    """Replace `old` by `new` exactly `count` times in `relpath`.

    Idempotent: returns False (nothing to do) when the file holds `new`
    `count` times and `old` no longer. `items` (identifiers of the edited
    items, `<relpath>::<Qual>`) are logged with log_change.
    """
    path = os.path.join(repo, relpath)
    text, eol = _read(path)
    old_e = old.replace('\n', eol)
    new_e = new.replace('\n', eol)
    found = text.count(old_e)
    if found == 0 and (new_e in text if new else True):
        print(f'  {relpath}: already done')
        return False
    if found != count:
        raise FixupError(f'{relpath}: expected {count} occurrence(s) of '
                         f'{old!r}, found {found}')
    _write(path, text.replace(old_e, new_e))
    print(f'  {relpath}: replaced {count}x {old.strip()[:60]!r}')
    log_change(relpath, list(items) or [relpath],
               f'replaced {old.strip()[:80]!r}')
    return True


def ensure_line_after(repo, relpath, anchor, line):
    """Insert `line` (without EOL) right after the line `anchor` (exact
    line text without EOL), unless the file already has that line."""
    path = os.path.join(repo, relpath)
    text, eol = _read(path)
    lines = text.split(eol)
    if line in lines:
        print(f'  {relpath}: already has {line!r}')
        return False
    try:
        k = lines.index(anchor)
    except ValueError:
        raise FixupError(f'{relpath}: anchor line {anchor!r} not found')
    lines.insert(k + 1, line)
    _write(path, eol.join(lines))
    print(f'  {relpath}: inserted {line!r}')
    log_change(relpath, [relpath], f'inserted import {line!r}',
               kind='fixup-import')
    return True


def has_line(repo, relpath, line):
    text, eol = _read(os.path.join(repo, relpath))
    return line in text.split(eol)


def run(main):
    try:
        main(repo_root())
    except FixupError as err:
        print(f'FIXUP FAILED: {err}', file=sys.stderr)
        sys.exit(1)
