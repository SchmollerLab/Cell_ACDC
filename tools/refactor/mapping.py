"""Mapping shards: validate them, emit the name table, report coverage.

Usage (from the repository root, any commit; the base commit is read with
`git show`):

    python tools/refactor/mapping.py validate
    python tools/refactor/mapping.py name-table --out tools/refactor/out/name_table.json [--commit C11]
    python tools/refactor/mapping.py coverage-report [--module cellacdc/apps.py]

Options: --repo <path> (default: this checkout), --mapping <folder>
(default: tools/refactor/mapping), --base <sha> (default: 4c4e498b or
$ACDC_REFACTOR_BASE).

The name table (contract with acdc_refactor/rewrite.py):

    {"modules": {old dotted: new dotted}          whole-module moves
     "names": {old dotted module: {name: home dotted module}}
              every top-level name of an old module that the module the old
              dotted name now resolves to no longer defines (re-exports are
              mapped to the module that defines the name; Qt/third-party
              names to the new module that binds them)
     "entries": same as names, values [home, attribute] (attribute null:
              the name is the home module itself, see "module_values")
     "module_values": {old: {name: module}}  names whose value is a module
     "intermediate": {new dotted: {name: [home, attr]}}  modules at new
              paths that later lost names (compat blocks until C12)
     "aliases": {dotted: canonical alias}, "packages": [new packages],
     "plugin_api": {old: [names]}, "classes": {old class: {host, methods}},
     "shims": {old: kind}, "commit": ..., "base": ...}
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from acdc_refactor import (  # noqa: E402
    COMMITS, DEFAULT_REPO_ROOT
)
from acdc_refactor.plan import Plan  # noqa: E402
from acdc_refactor.plan_check import Checker, format_report  # noqa: E402


def load_plan(args):
    if args.base:
        os.environ['ACDC_REFACTOR_BASE'] = args.base
    return Plan(args.repo, mapping_dir=args.mapping)


def cmd_validate(args):
    plan = load_plan(args)
    checker = Checker(plan, deep=False)
    checker.check_keys()
    checker.check_coverage()
    checker.check_dests()
    checker.check_paths()
    checker.check_aliases()
    checker.check_names()
    checker.apply_allow()
    print(format_report(checker.issues, show_info=args.info))
    return 1 if any(i.level == 'error' for i in checker.issues) else 0


def cmd_name_table(args):
    plan = load_plan(args)
    table = plan.name_table(args.commit)
    text = json.dumps(table, indent=1, sort_keys=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                    exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as file:
            file.write(text + '\n')
        n_names = sum(len(v) for v in table['names'].values())
        print(f'Wrote {args.out}: {len(table["modules"])} module moves, '
              f'{len(table["names"])} old modules, {n_names} names')
    else:
        print(text)
    errors = [i for i in plan.issues if i.level == 'error']
    for issue in errors:
        print(issue, file=sys.stderr)
    return 1 if errors else 0


def cmd_coverage(args):
    plan = load_plan(args)
    files = plan.old_module_files()
    if args.module:
        files = [f for f in files if f == args.module]
    fully = set(plan.fully_split_modules())
    for path in files:
        bmod = plan.base_module(path)
        items = [i for i in bmod.items if i.kind not in ('import',
                                                          'docstring')]
        mapped = [i for i in items if (path, i.key) in plan.items]
        required = [i for i in items if not i.optional]
        missing = [i for i in required if (path, i.key) not in plan.items]
        move = plan.module_move_of(path)
        kind = ('fully split' if path in fully else
                f'moved to {move.new}' if move else 'partly split')
        print(f'{path} ({kind}): {len(mapped)}/{len(items)} items mapped'
              + (f', {len(missing)} required items unmapped'
                 if path in fully and missing else ''))
        dests = {}
        for i in mapped:
            dest = plan.items[(path, i.key)].dest or 'DELETE'
            dests[dest] = dests.get(dest, 0) + 1
        for dest in sorted(dests):
            print(f'    {dests[dest]:5d} -> {dest}')
        if path in fully and missing and args.verbose:
            for i in missing:
                print(f'    unmapped: {i.key} (line {i.code_first})')
        for (old_file, cls), split in plan.classes.items():
            if old_file != path or bmod.class_node(cls) is None:
                continue
            methods = [m for m in bmod.class_items(cls) if m.kind == 'method']
            names = {m.key for m in methods}
            moved = names & set(split.methods)
            print(f'    class {cls}: {len(moved)}/{len(names)} methods '
                  f'into {len(split.mixins)} mixins; stay: '
                  f'{sorted(names - moved)[:20]}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Validate the mapping shards, emit the name table.')
    parser.add_argument('--repo', default=DEFAULT_REPO_ROOT)
    parser.add_argument('--mapping', default=None)
    parser.add_argument('--base', default=None)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('validate')
    p.add_argument('--info', action='store_true')
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser('name-table')
    p.add_argument('--commit', default=COMMITS[-1])
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_name_table)
    p = sub.add_parser('coverage-report')
    p.add_argument('--module', default=None)
    p.add_argument('-v', '--verbose', action='store_true')
    p.set_defaults(func=cmd_coverage)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
