"""Attribute resolution snapshot of the two composed windows.

Usage::

    python tools/refactor/verify/mro_snapshot.py snapshot --out <dir> [--repo <tree>]
        [--class guiWin] [--class MeasurementsViewerWin] [--mapping-table <t>]
    python tools/refactor/verify/mro_snapshot.py compare --before <dir> --after <dir>
        [--mapping-table <name-table.json>] [--mapping-dir <shards>]

``snapshot`` imports ``guiWin`` and ``MeasurementsViewerWin`` in a child
process (``QT_QPA_PLATFORM=offscreen``, user settings isolated, new import
paths tried first, see ``common.WINDOW_CLASSES``; with a name table the
table's home of the old name comes first) and records, for every name in
``dir(cls)`` defined by at least one ``cellacdc`` class of the MRO:

- ``definers``: every ``cellacdc`` class of the MRO defining it, in MRO
  order (the first one is what the name resolves to), each with the kind
  (function, property, staticmethod, classmethod, signal, class, attr), the
  identity of the defined function (``module:qualname`` of the undecorated
  function), a canonical code hash and its refs, and the strict hash of the
  code (``co_code`` + ``co_consts`` recursively without file names/line
  numbers + ``co_names`` + ``co_varnames``);
- ``qt``: the non-``cellacdc`` classes of the MRO defining it too (Qt names
  shadowed by a mixin method).

The canonical code hash comes from ``dis``: globals and ``module.attr``
chains are folded into refs to the object they load (classes/functions by
``__module__.__qualname__``, modules by name, other values by the module
variable), imports inside the function are dropped and their aliases folded
the same way, jump targets become labels; so rewriting references to moved
names does not change it when the refs map to the same new homes.

Also recorded: the MRO, the Signals with their signatures, the
``staticMetaObject`` methods and ``overrides`` (names defined in two or more
``cellacdc`` classes of the MRO).

``compare`` maps identities, classes and refs through the name table and
the mapping shards (``common.Mapping``: a method moved from ``guiWin`` into a
mixin keeps its canonical identity). Errors: a name resolving to a different
function than before (``resolution_changed``, e.g. a shared mixin's method
now winning over a window-only override), a changed chain of definers or of
shadowed Qt classes, lost or new names, MRO order of the old classes,
Signals and meta-object methods. A function whose identity is unchanged but
whose canonical hash or refs differ is a warning (``body_changed``): the
static fingerprint compare is decisive for bodies.
"""
import argparse
import dis
import hashlib
import json
import os
import re
import sys

try:
    from . import common
except ImportError:
    import common

SNAPSHOT_FILE = 'mro.json'
DEFAULT_CLASSES = ('guiWin', 'MeasurementsViewerWin')

IGNORED_NAMES = {
    '__doc__', '__module__', '__dict__', '__weakref__', '__qualname__',
    '__firstlineno__', '__static_attributes__', '__annotations__',
    '__orig_bases__', '__parameters__', '__abstractmethods__', '_abc_impl',
    '__annotate__', '__annotations_cache__',
}

SKIP_OPS = {
    'CACHE', 'RESUME', 'NOP', 'EXTENDED_ARG', 'MAKE_CELL', 'COPY_FREE_VARS',
    'RESUME_CHECK', 'NOT_TAKEN',
}
LOAD_FAST_OPS = {
    'LOAD_FAST', 'LOAD_FAST_CHECK', 'LOAD_FAST_AND_CLEAR', 'LOAD_DEREF',
    'LOAD_CLOSURE', 'LOAD_FAST_BORROW',
}
STORE_OPS = {'STORE_FAST', 'STORE_DEREF', 'STORE_NAME', 'STORE_GLOBAL'}
DOUBLE_OPS = {
    'LOAD_FAST_LOAD_FAST': ('LOAD_FAST', 'LOAD_FAST'),
    'STORE_FAST_STORE_FAST': ('STORE_FAST', 'STORE_FAST'),
    'STORE_FAST_LOAD_FAST': ('STORE_FAST', 'LOAD_FAST'),
    'LOAD_FAST_BORROW_LOAD_FAST_BORROW': ('LOAD_FAST', 'LOAD_FAST'),
}
ADDRESS = re.compile(r' at 0x[0-9A-Fa-f]+')
DOTTED_CELLACDC = re.compile(r'cellacdc(\.[A-Za-z_][A-Za-z0-9_]*)+')


def _sha(text):
    return hashlib.sha1(text.encode('utf-8', 'replace')).hexdigest()[:20]


def _safe_repr(value):
    if isinstance(value, frozenset):
        return 'frozenset(' + repr(sorted(map(_safe_repr, value))) + ')'
    if isinstance(value, tuple):
        return '(' + ', '.join(_safe_repr(v) for v in value) + ')'
    try:
        text = repr(value)
    except Exception:
        text = f'<{type(value).__name__}>'
    return ADDRESS.sub('', text)


# --- Code hashing (runs in the child) ------------------------------------

def strict_hash(code):
    parts = [code.co_code.hex(), repr(code.co_names), repr(code.co_varnames)]
    for const in code.co_consts:
        if hasattr(const, 'co_code'):
            parts.append('CODE:' + strict_hash(const))
        else:
            parts.append(_safe_repr(const))
    return _sha('\n'.join(parts))


class CodeCanon:
    """Canonical token stream of code objects (see the module docstring)."""

    def __init__(self):
        import builtins
        self.builtins = vars(builtins)

    # objects -> refs ---------------------------------------------------------
    @staticmethod
    def obj_ref(value, where, name):
        import types
        if isinstance(value, types.ModuleType):
            return ['M', value.__name__]
        module = getattr(value, '__module__', None)
        qual = getattr(value, '__qualname__', None)
        if isinstance(module, str) and isinstance(qual, str) and (
            isinstance(value, type) or callable(value)
        ):
            if module == 'cellacdc' or module.startswith('cellacdc.'):
                return ['D', module, qual]
            return ['E', f'{module}.{qual}']
        return ['V', where, name]

    def resolve_dotted(self, path):
        """(object or None, ref) for a dotted import path."""
        import importlib
        parts = path.split('.')
        for i in range(len(parts), 0, -1):
            modname = '.'.join(parts[:i])
            module = sys.modules.get(modname)
            if module is None and modname.split('.')[0] == 'cellacdc':
                try:
                    module = importlib.import_module(modname)
                except Exception:
                    module = None
            if module is None:
                continue
            obj = module
            ref = ['M', module.__name__]
            for attr in parts[i:]:
                try:
                    obj2 = getattr(obj, attr)
                except Exception:
                    return None, ['U', getattr(obj, '__name__', path), attr]
                ref = self.obj_ref(obj2, getattr(obj, '__name__', ''), attr)
                obj = obj2
            return obj, ref
        return None, ['E', path]

    # imports inside code ------------------------------------------------------
    def import_aliases(self, instrs, globals_):
        """(indices to drop, {local name: dotted path}) of the import blocks."""
        drop = set()
        aliases = {}
        package = globals_.get('__package__') or ''
        n = len(instrs)
        for idx, ins in enumerate(instrs):
            if ins.opname != 'IMPORT_NAME' or idx < 2:
                continue
            level_ins, from_ins = instrs[idx - 2], instrs[idx - 1]
            if level_ins.opname != 'LOAD_CONST' or from_ins.opname != (
                'LOAD_CONST'
            ):
                continue
            level, fromlist = level_ins.argval, from_ins.argval
            name = ins.argval or ''
            if level:
                base = package.split('.')
                if level > 1:
                    base = base[:-(level - 1)]
                absname = '.'.join(base + ([name] if name else []))
            else:
                absname = name
            cur = absname if fromlist else absname.split('.')[0]
            drop.update((idx - 2, idx - 1, idx))
            j = idx + 1
            while j < n:
                op = instrs[j].opname
                if op == 'IMPORT_FROM':
                    val = f'{cur}.{instrs[j].argval}' if cur else (
                        instrs[j].argval
                    )
                    drop.add(j)
                    j += 1
                    if j < n and instrs[j].opname in STORE_OPS:
                        aliases[instrs[j].argval] = val
                        drop.add(j)
                        j += 1
                    elif j < n and instrs[j].opname in ('SWAP', 'POP_TOP'):
                        cur = val
                    continue
                if op in STORE_OPS:
                    aliases[instrs[j].argval] = cur
                    drop.add(j)
                    j += 1
                    break
                if op in ('SWAP', 'POP_TOP'):
                    drop.add(j)
                    j += 1
                    if op == 'POP_TOP':
                        break
                    continue
                break
        return drop, aliases

    # token stream ----------------------------------------------------------
    def code_tokens(self, code, globals_, inherited, tokens, refs):
        raw = [
            ins for ins in dis.get_instructions(code)
            if ins.opname not in SKIP_OPS
        ]
        instrs = []
        for ins in raw:
            split = DOUBLE_OPS.get(ins.opname)
            if split and isinstance(ins.argval, tuple):
                for opname, argval in zip(split, ins.argval):
                    instrs.append(_Ins(opname, argval, ins))
            else:
                instrs.append(_Ins(ins.opname, ins.argval, ins))
        drop, alias_paths = self.import_aliases(instrs, globals_)
        aliases = dict(inherited)
        for name, path in alias_paths.items():
            aliases[name] = self.resolve_dotted(path)
        targets = sorted({
            ins.target for ins in instrs if ins.target is not None
        })
        labels = {offset: i for i, offset in enumerate(targets)}
        where = globals_.get('__name__', '')
        i = 0
        n = len(instrs)
        while i < n:
            ins = instrs[i]
            op = ins.opname
            if i in drop:
                i += 1
                continue
            obj = ref = None
            if op in ('LOAD_GLOBAL', 'LOAD_NAME'):
                name = ins.argval
                if name in globals_:
                    obj = globals_[name]
                    ref = self.obj_ref(obj, where, name)
                elif name in self.builtins:
                    obj, ref = self.builtins[name], ['B', name]
                else:
                    obj, ref = None, ['U', where, name]
            elif op in LOAD_FAST_OPS and ins.argval in aliases:
                obj, ref = aliases[ins.argval]
            if ref is not None:
                i += 1
                while (obj is not None and _is_module(obj) and i < n
                       and instrs[i].opname in ('LOAD_ATTR', 'LOAD_METHOD')):
                    attr = instrs[i].argval
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            obj2 = getattr(obj, attr)
                    except Exception:
                        ref = ['U', obj.__name__, attr]
                        obj = None
                        i += 1
                        break
                    ref = self.obj_ref(obj2, obj.__name__, attr)
                    obj = obj2
                    i += 1
                tokens.append('REF')
                refs.append(ref)
                continue
            if op == 'LOAD_CONST' and hasattr(ins.argval, 'co_code'):
                tokens.append('CODE{')
                self.code_tokens(ins.argval, globals_, aliases, tokens, refs)
                tokens.append('}')
            elif op in ('LOAD_CONST', 'RETURN_CONST', 'LOAD_SMALL_INT'):
                value = ins.argval
                if i + 1 < n and instrs[i + 1].argval == '__firstlineno__':
                    # class bodies store their first line number (3.13+)
                    tokens.append(f'{op} <firstlineno>')
                elif isinstance(value, str) and DOTTED_CELLACDC.fullmatch(
                    value
                ):
                    tokens.append(op + ' SREF')
                    refs.append(['S', value])
                else:
                    tokens.append(f'{op} {_safe_repr(value)}')
            elif ins.target is not None:
                tokens.append(f'{op} L{labels[ins.target]}')
            elif op in ('LOAD_ATTR', 'LOAD_METHOD', 'STORE_ATTR',
                        'LOAD_SUPER_ATTR', 'DELETE_ATTR'):
                tokens.append(f'{op} {ins.argval}')
            elif ins.argval is None:
                tokens.append(op)
            else:
                tokens.append(f'{op} {_safe_repr(ins.argval)}')
            i += 1


def _is_module(obj):
    import types
    return isinstance(obj, types.ModuleType)


class _Ins:
    __slots__ = ('opname', 'argval', 'target')

    def __init__(self, opname, argval, ins):
        self.opname = opname
        self.argval = argval
        target = getattr(ins, 'jump_target', None)
        if target is None and ins.opcode in set(dis.hasjrel) | set(
            dis.hasjabs
        ):
            target = ins.argval
        self.target = target


# --- Snapshot (child) -----------------------------------------------------

def _unwrap(func):
    seen = set()
    chain = [func]
    while hasattr(func, '__wrapped__') and id(func) not in seen:
        seen.add(id(func))
        func = func.__wrapped__
        chain.append(func)
    return chain


def _describe_function(func, canon, repo):
    chain = _unwrap(func)
    inner = chain[-1]
    code = getattr(inner, '__code__', None)
    entry = {
        'id': f"{getattr(inner, '__module__', '?')}:"
              f"{getattr(inner, '__qualname__', '?')}",
    }
    if code is None:
        entry['kind_detail'] = type(inner).__name__
        return entry
    tokens, refs = [], []
    canon.code_tokens(code, getattr(inner, '__globals__', {}), {}, tokens,
                      refs)
    entry['hash'] = _sha('\n'.join(tokens))
    entry['refs'] = refs
    entry['strict'] = strict_hash(code)
    filename = code.co_filename
    try:
        filename = os.path.relpath(filename, repo).replace('\\', '/')
    except ValueError:
        pass
    entry['file'] = f'{filename}:{code.co_firstlineno}'
    if len(chain) > 1:
        wrapper = chain[0]
        wcode = getattr(wrapper, '__code__', None)
        if wcode is not None:
            wtokens, wrefs = [], []
            canon.code_tokens(wcode, getattr(wrapper, '__globals__', {}), {},
                              wtokens, wrefs)
            entry['wrapper'] = {
                'code': f"{wcode.co_name}",
                'hash': _sha('\n'.join(wtokens)), 'refs': wrefs,
            }
    return entry


def _describe(cls, name, raw, canon, repo):
    entry = {'cls': f'{cls.__module__}:{cls.__qualname__}'}
    kind = type(raw).__name__
    if isinstance(raw, staticmethod):
        entry['kind'] = 'staticmethod'
        entry.update(_describe_function(raw.__func__, canon, repo))
    elif isinstance(raw, classmethod):
        entry['kind'] = 'classmethod'
        entry.update(_describe_function(raw.__func__, canon, repo))
    elif isinstance(raw, property):
        entry['kind'] = 'property'
        parts = {}
        for part in ('fget', 'fset', 'fdel'):
            func = getattr(raw, part)
            if func is not None:
                parts[part] = _describe_function(func, canon, repo)
        entry['parts'] = parts
        if 'fget' in parts:
            entry['id'] = parts['fget']['id']
    elif kind in ('pyqtSignal', 'Signal', 'SignalInstance'):
        entry['kind'] = 'signal'
        sig = getattr(raw, 'signatures', None)
        entry['signature'] = list(sig) if sig else _safe_repr(raw)
    elif isinstance(raw, type):
        entry['kind'] = 'class'
        entry['id'] = f'{raw.__module__}:{raw.__qualname__}'
    elif callable(raw) and hasattr(raw, '__code__'):
        entry['kind'] = 'function'
        entry.update(_describe_function(raw, canon, repo))
    else:
        entry['kind'] = 'attr'
        entry['type'] = kind
        text = _safe_repr(raw)
        entry['value'] = text if len(text) <= 200 else _sha(text)
    return entry


def snapshot_class(cls, repo):
    canon = CodeCanon()
    names = {}
    overrides = []
    signals = {}
    for name in sorted(dir(cls)):
        if name in IGNORED_NAMES:
            continue
        definers = []
        qt = []
        for klass in cls.__mro__:
            if name not in vars(klass):
                continue
            module = klass.__module__ or ''
            if module == 'cellacdc' or module.startswith('cellacdc.'):
                definers.append(_describe(
                    klass, name, vars(klass)[name], canon, repo
                ))
            else:
                qt.append(f'{module}:{klass.__qualname__}')
        if not definers:
            continue
        names[name] = {'definers': definers, 'qt': qt}
        if len(definers) > 1:
            overrides.append(name)
        if definers[0].get('kind') == 'signal':
            signals[name] = definers[0].get('signature')
    meta_methods = []
    meta = getattr(cls, 'staticMetaObject', None)
    if meta is not None:
        try:
            for i in range(meta.methodCount()):
                sig = meta.method(i).methodSignature()
                meta_methods.append(bytes(sig).decode('utf-8', 'replace'))
        except Exception as err:
            meta_methods.append(f'<error {err!r}>')
    return {
        'cls': f'{cls.__module__}:{cls.__qualname__}',
        'mro': [f'{k.__module__}:{k.__qualname__}' for k in cls.__mro__],
        'names': names,
        'overrides': overrides,
        'signals': signals,
        'meta_methods': meta_methods,
    }


def child_main(args):
    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import warnings
    warnings.simplefilter('ignore', DeprecationWarning)
    import cellacdc  # noqa: F401
    common.install_settings_isolation()
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    result = {'classes': {}, 'errors': {}}
    candidates = json.loads(args.candidates)
    for label, specs in candidates.items():
        cls = None
        errors = []
        for spec in specs:
            try:
                cls = common.resolve_attr(spec)
                break
            except Exception as err:
                errors.append(f'{spec}: {type(err).__name__}: {err}')
        if cls is None:
            result['errors'][label] = errors
            continue
        data = snapshot_class(cls, repo)
        data['imported_from'] = spec
        result['classes'][label] = data
    result['packages'] = sorted(
        name for name, module in sys.modules.items()
        if module is not None and (name == 'cellacdc' or name.startswith(
            'cellacdc.'
        )) and hasattr(module, '__path__')
    )
    with open(args.child_out, 'w', encoding='utf-8') as file:
        json.dump(result, file)
    return 0


# --- Parent ---------------------------------------------------------------

def class_candidates(labels, mapping=None):
    candidates = {}
    for label in labels:
        specs = list(common.WINDOW_CLASSES.get(label, []))
        if ':' in label:
            specs = [label]
        if mapping is not None:
            for spec in list(specs):
                mod, _, qual = spec.partition(':')
                home = mapping.names.get(mod, {}).get(qual)
                if home and f'{home}:{qual}' not in specs:
                    specs.insert(0, f'{home}:{qual}')
        candidates[label.split(':')[-1]] = specs
    return candidates


def snapshot(repo, out, labels=DEFAULT_CLASSES, table=None, timeout=300):
    mapping = common.Mapping.load(table) if table else None
    candidates = class_candidates(labels, mapping)
    os.makedirs(out, exist_ok=True)
    child_out = os.path.join(out, SNAPSHOT_FILE + '.child')
    cmd = [
        sys.executable, os.path.abspath(__file__), '_child', '--repo', repo,
        '--child-out', child_out, '--candidates', json.dumps(candidates),
    ]
    result = common.run_child(cmd, repo, common.child_env(repo), timeout)
    if result['timed_out'] or result['returncode'] != 0 or not os.path.exists(
        child_out
    ):
        raise RuntimeError(
            f"mro snapshot child failed (rc={result['returncode']}, "
            f"timeout={result['timed_out']}):\n{common.tail(result['stderr'])}"
        )
    data = common.read_json(child_out)
    os.remove(child_out)
    data['meta'] = common.meta(repo)
    path = os.path.join(out, SNAPSHOT_FILE)
    common.write_json(path, data, indent=None)
    summary = {
        label: {
            'names': len(info['names']), 'mro': len(info['mro']),
            'overrides': len(info['overrides']),
            'signals': len(info['signals']),
            'qt_shadowed': sum(1 for v in info['names'].values() if v['qt']),
            'imported_from': info['imported_from'],
        }
        for label, info in data['classes'].items()
    }
    print(f'[mro] {path}: {summary} errors={data["errors"]}')
    return summary


class _RefCanon:
    def __init__(self, mapping, packages):
        self.mapping = mapping
        self.packages = set(packages)

    def ref(self, ref):
        kind = ref[0]
        m = self.mapping
        if kind in ('D', 'V', 'U'):
            mod, qual = ref[1], ref[2]
            top, _, rest = qual.partition('.')
            newmod, newtop = m.map_def(mod, top, mod in self.packages)
            return f'{kind}|{newmod}|' + (
                f'{newtop}.{rest}' if rest else newtop
            )
        if kind == 'M':
            return 'M|' + m.map_module(ref[1], ref[1] in self.packages)
        if kind == 'S':
            return 'S|' + m.map_dotted(ref[1])
        return '|'.join(ref)

    def refs(self, refs):
        return [self.ref(r) for r in refs]


def _definer_identity(canon, entry, name):
    if entry.get('kind') in ('signal', 'attr', 'class') or not entry.get('id'):
        mod, _, qual = entry['cls'].partition(':')
        return canon.mapping.canonical_key(
            mod, f'{qual}.{name}', mod in canon.packages
        )
    mod, _, qual = entry['id'].partition(':')
    return canon.mapping.canonical_key(mod, qual, mod in canon.packages)


def _body(canon, entry):
    if entry.get('kind') == 'property':
        return tuple(
            (part, d.get('hash'), tuple(canon.refs(d.get('refs', []))))
            for part, d in sorted(entry.get('parts', {}).items())
        )
    if entry.get('kind') == 'signal':
        return ('signal', json.dumps(entry.get('signature')))
    if entry.get('kind') == 'attr':
        return ('attr', entry.get('type'), entry.get('value'))
    return (entry.get('hash'), tuple(canon.refs(entry.get('refs', []))))


def _class_canon(mapping, spec, packages):
    mod, _, qual = spec.partition(':')
    if mapping.is_mixin_shell(mod, qual):
        return None
    newmod, newqual = mapping.map_def(mod, qual.split('.')[0], mod in packages)
    rest = qual.partition('.')[2]
    return f'{newmod}:{newqual}' + (f'.{rest}' if rest else '')


def compare(before, after, mapping):
    report = {'ok': True, 'errors': [], 'warnings': [], 'info': [],
              'classes': {}}
    cb = _RefCanon(mapping, before.get('packages', []))
    ca = _RefCanon(mapping, after.get('packages', []))
    for label in sorted(set(before['classes']) | set(after['classes'])):
        b = before['classes'].get(label)
        a = after['classes'].get(label)
        if b is None or a is None:
            report['errors'].append(f'{label}: missing on one side')
            continue
        res = {'resolution_changed': [], 'body_changed': [],
               'chain_changed': [], 'qt_changed': [], 'lost': [], 'new': [],
               'rewritten_only': 0, 'same': 0}
        # MRO order of the old classes
        bmro = [_class_canon(mapping, c, cb.packages) for c in b['mro']]
        amro = [_class_canon(mapping, c, ca.packages) for c in a['mro']]
        amro_known = [c for c in amro if c is not None]
        bset = [c for c in bmro if c is not None]
        missing = [c for c in bset if c not in amro_known]
        order = [c for c in amro_known if c in bset]
        expected = [c for c in bset if c in amro_known]
        if missing:
            report['errors'].append(f'{label}: MRO classes lost: {missing}')
        if order != expected:
            report['errors'].append(f'{label}: MRO order of the old classes '
                                    'changed')
        res['mro_new_classes'] = [
            c for c in a['mro']
            if _class_canon(mapping, c, ca.packages) is None
            or _class_canon(mapping, c, ca.packages) not in bset
        ]
        for name in sorted(set(b['names']) | set(a['names'])):
            bn, an = b['names'].get(name), a['names'].get(name)
            if bn is None:
                res['new'].append(name)
                continue
            if an is None:
                res['lost'].append(name)
                continue
            bids = [_definer_identity(cb, d, name) for d in bn['definers']]
            aids = [_definer_identity(ca, d, name) for d in an['definers']]
            if bids[0] != aids[0]:
                entry = {'name': name, 'before': bids[0], 'after': aids[0]}
                if aids[0] in bids:
                    entry['note'] = (
                        f'now resolves to before-definer #{bids.index(aids[0])}'
                    )
                res['resolution_changed'].append(entry)
                continue
            if bids != aids:
                res['chain_changed'].append(
                    {'name': name, 'before': bids, 'after': aids}
                )
            bq = [_class_canon(mapping, c, cb.packages) or c for c in bn['qt']]
            aq = [_class_canon(mapping, c, ca.packages) or c for c in an['qt']]
            if bq != aq:
                res['qt_changed'].append(
                    {'name': name, 'before': bn['qt'], 'after': an['qt']}
                )
            bd, ad = bn['definers'][0], an['definers'][0]
            if _body(cb, bd) != _body(ca, ad):
                res['body_changed'].append({
                    'name': name, 'identity': bids[0],
                    'before': bd.get('file'), 'after': ad.get('file'),
                })
            elif bd.get('strict') != ad.get('strict'):
                res['rewritten_only'] += 1
            else:
                res['same'] += 1
        sig_b, sig_a = b.get('signals', {}), a.get('signals', {})
        if sig_b != sig_a:
            report['errors'].append(f'{label}: Signals differ')
        if sorted(b.get('meta_methods', [])) != sorted(
            a.get('meta_methods', [])
        ):
            report['errors'].append(f'{label}: staticMetaObject methods differ')
        for key in ('resolution_changed', 'chain_changed', 'qt_changed',
                    'lost', 'new'):
            for item in res[key]:
                report['errors'].append(f'{label}: {key}: {item}')
        for item in res['body_changed']:
            report['warnings'].append(f'{label}: body_changed: {item}')
        res['summary'] = {
            k: (len(v) if isinstance(v, list) else v) for k, v in res.items()
            if k != 'summary'
        }
        report['classes'][label] = res
    report['ok'] = not report['errors']
    return report


def load_snapshot(folder):
    return common.read_json(os.path.join(folder, SNAPSHOT_FILE))


def run(commit, repo=None, before='baseline', table=None, mapping_dir=None):
    repo = repo or common.REPO_ROOT
    out = common.out_dir(commit)
    table = table or common.default_name_table(commit)
    snapshot(repo, out, table=table)
    if not before or before == commit:
        return None
    before_dir = common.out_dir(before)
    if not os.path.exists(os.path.join(before_dir, SNAPSHOT_FILE)):
        return None
    mapping = common.Mapping.load(table, mapping_dir or common.MAPPING_DIR)
    report = compare(load_snapshot(before_dir), load_snapshot(out), mapping)
    common.write_json(os.path.join(out, 'mro_compare.json'), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('snapshot')
    p.add_argument('--out', required=True)
    p.add_argument('--repo', default=common.REPO_ROOT)
    p.add_argument('--class', dest='classes', action='append')
    p.add_argument('--mapping-table')
    p.add_argument('--timeout', type=int, default=300)
    p = sub.add_parser('compare')
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--mapping-table')
    p.add_argument('--mapping-dir', default=common.MAPPING_DIR)
    p.add_argument('--report')
    p.add_argument('--max-print', type=int, default=40)
    p = sub.add_parser('_child')
    p.add_argument('--repo', required=True)
    p.add_argument('--child-out', required=True)
    p.add_argument('--candidates', required=True)
    args = parser.parse_args(argv)
    if args.cmd == '_child':
        return child_main(args)
    if args.cmd == 'snapshot':
        snapshot(os.path.abspath(args.repo), args.out,
                 args.classes or DEFAULT_CLASSES, args.mapping_table,
                 args.timeout)
        return 0
    mapping = common.Mapping.load(args.mapping_table, args.mapping_dir)
    report = compare(
        load_snapshot(args.before), load_snapshot(args.after), mapping
    )
    path = args.report or os.path.join(args.after, 'mro_compare.json')
    common.write_json(path, report)
    for label, res in report['classes'].items():
        print(f"[mro] {label}: {res['summary']}")
    for line in report['warnings'][:args.max_print]:
        print('WARNING:', line)
    for line in report['errors'][:args.max_print]:
        print('ERROR:', line)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
