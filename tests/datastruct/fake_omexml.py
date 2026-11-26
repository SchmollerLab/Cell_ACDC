"""`FakeOMEXML`: duck-typed legacy metadata object for the metadata cascade.

It answers exactly the accessor expressions of
`dataStruct.bioFormatsWorker.readMetadata`, each one returning a value or
raising per a JSON-able spec:

    metadata.instrument().Objective.LensNA         spec['LensNA']
    metadata.get_image_count()                     spec['image_count']
    metadata.image().Name                          spec['Name']
    metadata.image().Pixels.SizeZ / SizeT / SizeC  spec['SizeZ'] ...
    metadata.image().Pixels.PhysicalSizeX/Y/Z      spec['PhysicalSizeX'] ...
    metadata.image().Pixels.node.get(key)          spec['node'][key]
    metadata.image().Pixels.Channel(c).Name        spec['channels'][c]['Name']
    metadata.image().Pixels.Channel(c).node.get('EmissionWavelength')
                                                   spec['channels'][c]['EmissionWavelength']

Rules:
- A value equal to a raise marker (`raises(...)`, e.g. RAISE) makes that
  accessor raise. The levels 'instrument', 'Objective', 'image', 'Pixels',
  'node' and 'Channel' (the call `Channel(c)` for every c) can also be set to
  a raise marker.
- A missing attribute key raises AttributeError (the accessor is absent).
- `node.get(key)` returns None for a missing key (ElementTree / dict `get`).
- `Channel(c)` raises IndexError when c >= len(spec['channels']); an entry may
  itself be a raise marker.
- A value {'__np_str__': 'x'} is returned as np.str_('x').
- 'omexml_string', when present, is exposed as the attribute of that name.

`accessed` lists every accessor evaluated, in order (e.g.
'image().Pixels.SizeZ', 'image().Pixels.Channel(1).node.get(EmissionWavelength)').
`cascade_grid()` is the shared named grid of specs (the cascade golden and
the metadata extraction tests).
"""
from __future__ import annotations

import builtins
import copy

import numpy as np

RAISE_KEY = '__raise__'
NP_STR_KEY = '__np_str__'
MICRO_SIGN_UM = '\u00b5m'      # U+00B5, what OME-XML writers use
GREEK_MU_UM = '\u03bcm'        # U+03BC, the GUI label and legacy default


def raises(exception: str = 'AttributeError', message: str = '') -> dict:
    """Raise marker: the accessor raises builtins.<exception>(message)."""
    return {RAISE_KEY: exception, 'message': message}


RAISE = raises()


def np_str(text: str) -> dict:
    """Marker for a value returned as np.str_(text)."""
    return {NP_STR_KEY: text}


def _is_raise(value) -> bool:
    return isinstance(value, dict) and RAISE_KEY in value


def _decode(value, accessor: str):
    if _is_raise(value):
        exc_type = getattr(builtins, value[RAISE_KEY])
        message = value.get('message') or f'FakeOMEXML: {accessor} raised'
        raise exc_type(message)
    if isinstance(value, dict) and NP_STR_KEY in value:
        return np.str_(value[NP_STR_KEY])
    return value


class _Accessor:
    """Object whose attributes are looked up in a spec dict."""

    def __init__(self, root: 'FakeOMEXML', spec: dict, prefix: str,
                 attributes: tuple[str, ...]) -> None:
        self._root = root
        self._spec = spec
        self._prefix = prefix
        self._attributes = attributes

    def __getattr__(self, name: str):
        if name.startswith('_') or name not in self._attributes:
            raise AttributeError(name)
        accessor = f'{self._prefix}.{name}'
        self._root.accessed.append(accessor)
        if name not in self._spec:
            raise AttributeError(f'FakeOMEXML: {accessor} is absent')
        return _decode(self._spec[name], accessor)


class _Node:
    def __init__(self, root: 'FakeOMEXML', spec: dict, prefix: str) -> None:
        self._root = root
        self._spec = spec
        self._prefix = prefix

    def get(self, key: str, default=None):
        accessor = f'{self._prefix}.get({key})'
        self._root.accessed.append(accessor)
        if key not in self._spec:
            return default
        return _decode(self._spec[key], accessor)


class _Channel:
    def __init__(self, root: 'FakeOMEXML', spec: dict, prefix: str) -> None:
        self._root = root
        self._spec = spec
        self._prefix = prefix

    @property
    def Name(self):
        accessor = f'{self._prefix}.Name'
        self._root.accessed.append(accessor)
        if 'Name' not in self._spec:
            raise AttributeError(f'FakeOMEXML: {accessor} is absent')
        return _decode(self._spec['Name'], accessor)

    @property
    def node(self) -> _Node:
        node_spec = {
            key: value for key, value in self._spec.items() if key != 'Name'
        }
        return _Node(self._root, node_spec, f'{self._prefix}.node')


class _Pixels(_Accessor):
    ATTRIBUTES = (
        'SizeZ', 'SizeT', 'SizeC', 'SizeY', 'SizeX',
        'PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ',
    )

    def __init__(self, root: 'FakeOMEXML', spec: dict) -> None:
        super().__init__(root, spec, 'image().Pixels', self.ATTRIBUTES)

    @property
    def node(self) -> _Node:
        accessor = 'image().Pixels.node'
        _decode(self._spec.get('node'), accessor)
        return _Node(self._root, self._spec.get('node') or {}, accessor)

    def Channel(self, c: int = 0) -> _Channel:
        accessor = f'image().Pixels.Channel({c})'
        self._root.accessed.append(accessor)
        _decode(self._spec.get('Channel'), accessor)
        channels = self._spec.get('channels') or []
        if not 0 <= c < len(channels):
            raise IndexError(f'FakeOMEXML: {accessor} out of range')
        _decode(channels[c], accessor)
        return _Channel(self._root, channels[c], accessor)


class _Image(_Accessor):
    def __init__(self, root: 'FakeOMEXML', spec: dict) -> None:
        super().__init__(root, spec, 'image()', ('Name',))

    @property
    def Pixels(self) -> _Pixels:
        _decode(self._spec.get('Pixels'), 'image().Pixels')
        return _Pixels(self._root, self._spec)


class _Instrument:
    def __init__(self, root: 'FakeOMEXML', spec: dict) -> None:
        self._root = root
        self._spec = spec

    @property
    def Objective(self) -> _Accessor:
        accessor = 'instrument().Objective'
        _decode(self._spec.get('Objective'), accessor)
        return _Accessor(self._root, self._spec, accessor, ('LensNA',))


class FakeOMEXML:
    """Legacy metadata object answering per `spec` (see the module docstring)."""

    def __init__(self, spec: dict) -> None:
        self.spec = copy.deepcopy(spec)
        self.accessed: list[str] = []
        if 'omexml_string' in self.spec:
            self.omexml_string = self.spec['omexml_string']

    def instrument(self) -> _Instrument:
        self.accessed.append('instrument()')
        _decode(self.spec.get('instrument'), 'instrument()')
        return _Instrument(self, self.spec)

    def get_image_count(self):
        accessor = 'get_image_count()'
        self.accessed.append(accessor)
        if 'image_count' not in self.spec:
            raise AttributeError(f'FakeOMEXML: {accessor} is absent')
        return _decode(self.spec['image_count'], accessor)

    def image(self) -> _Image:
        self.accessed.append('image()')
        _decode(self.spec.get('image'), 'image()')
        return _Image(self, self.spec)

    def __repr__(self) -> str:
        return f'FakeOMEXML({self.spec!r})'


FULL_SPEC = {
    'LensNA': 1.4,
    'image_count': 1,
    'Name': 'Scene #1',
    'SizeZ': 2, 'SizeT': 3, 'SizeC': 2, 'SizeY': 16, 'SizeX': 20,
    'PhysicalSizeX': 0.1, 'PhysicalSizeY': 0.1, 'PhysicalSizeZ': 0.5,
    'node': {
        'TimeIncrement': 7.5, 'TimeIncrementUnit': 's',
        'PhysicalSizeXUnit': MICRO_SIGN_UM,
    },
    'channels': [
        {'Name': 'GFP', 'EmissionWavelength': 510.0},
        {'Name': 'mCherry', 'EmissionWavelength': 610.0},
    ],
}
"""Every accessor answers (python numbers, as the bioio OMEXML object)."""

STRING_SPEC = {
    'LensNA': '1.4',
    'image_count': 1,
    'Name': 'Scene #1',
    'SizeZ': '2', 'SizeT': '3', 'SizeC': '2', 'SizeY': '16', 'SizeX': '20',
    'PhysicalSizeX': '0.1', 'PhysicalSizeY': '0.1', 'PhysicalSizeZ': '0.5',
    'node': {
        'TimeIncrement': '7.5', 'TimeIncrementUnit': 's',
        'PhysicalSizeXUnit': MICRO_SIGN_UM,
    },
    'channels': [
        {'Name': 'GFP', 'EmissionWavelength': '510.0'},
        {'Name': 'mCherry', 'EmissionWavelength': '610.0'},
    ],
}
"""Every accessor answers with XML attribute strings (as load.OMEXML)."""

ATTRIBUTE_FIELDS = (
    'LensNA', 'image_count', 'Name', 'SizeZ', 'SizeT', 'SizeC',
    'PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ',
)
NODE_FIELDS = ('TimeIncrement', 'TimeIncrementUnit', 'PhysicalSizeXUnit')
LEVELS = ('instrument', 'Objective', 'image', 'Pixels', 'node', 'Channel')


class _Delete:
    def __repr__(self) -> str:
        return 'DELETE'


DELETE = _Delete()


def spec_with(base: dict | None = None, **changes) -> dict:
    """Deep copy of `base` (default FULL_SPEC) with top-level `changes`;
    a change whose value is `DELETE` removes the key."""
    spec = copy.deepcopy(FULL_SPEC if base is None else base)
    for key, value in changes.items():
        if value is DELETE:
            spec.pop(key, None)
        else:
            spec[key] = copy.deepcopy(value)
    return spec


def _node_with(**changes) -> dict:
    node = copy.deepcopy(FULL_SPEC['node'])
    for key, value in changes.items():
        if value is DELETE:
            node.pop(key, None)
        else:
            node[key] = value
    return node


def cascade_grid() -> dict[str, dict]:
    """Named FakeOMEXML specs covering every branch of the legacy cascade.

    Names are stable (goldens key on them): 'full', 'strings', 'empty',
    'raise:<field>' / 'none:<field>' / 'bad:<field>' for every attribute
    field, 'raise:node.<key>' / 'none:node.<key>' / 'bad:node.TimeIncrement',
    'raise:<level>' for every accessor level, channel variants
    ('channels:*') and unit variants ('units:*'). JSON-serializable.
    """
    grid = {
        'full': spec_with(),
        'strings': copy.deepcopy(STRING_SPEC),
        'empty': {},
        'scenes:3': spec_with(image_count=3),
    }
    for field in ATTRIBUTE_FIELDS:
        grid[f'raise:{field}'] = spec_with(**{field: RAISE})
        grid[f'none:{field}'] = spec_with(**{field: None})
        grid[f'absent:{field}'] = spec_with(**{field: DELETE})
        if field != 'Name':
            grid[f'bad:{field}'] = spec_with(**{field: 'abc'})
    for key in NODE_FIELDS:
        grid[f'raise:node.{key}'] = spec_with(
            node=_node_with(**{key: raises('ValueError')})
        )
        grid[f'none:node.{key}'] = spec_with(node=_node_with(**{key: DELETE}))
    grid['bad:node.TimeIncrement'] = spec_with(
        node=_node_with(TimeIncrement='abc')
    )
    for level in LEVELS:
        grid[f'raise:{level}'] = spec_with(**{level: RAISE})
    grid['channels:short'] = spec_with(channels=FULL_SPEC['channels'][:1])
    grid['channels:none'] = spec_with(channels=DELETE)
    grid['channels:np_str'] = spec_with(channels=[
        {'Name': np_str('GFP'), 'EmissionWavelength': 510.0},
        {'Name': np_str('mCherry'), 'EmissionWavelength': 610.0},
    ])
    grid['channels:name_none'] = spec_with(channels=[
        {'Name': None, 'EmissionWavelength': 510.0},
        {'Name': 'mCherry', 'EmissionWavelength': 610.0},
    ])
    grid['channels:name_raise'] = spec_with(channels=[
        {'Name': RAISE, 'EmissionWavelength': 510.0},
        {'Name': 'mCherry', 'EmissionWavelength': 610.0},
    ])
    grid['channels:no_emission'] = spec_with(channels=[
        {'Name': 'GFP'}, {'Name': 'mCherry', 'EmissionWavelength': 610.0},
    ])
    grid['channels:bad_emission'] = spec_with(channels=[
        {'Name': 'GFP', 'EmissionWavelength': 'abc'},
        {'Name': 'mCherry', 'EmissionWavelength': 610.0},
    ])
    grid['channels:entry_raise'] = spec_with(channels=[
        RAISE, {'Name': 'mCherry', 'EmissionWavelength': 610.0},
    ])
    grid['channels:size_c_3'] = spec_with(SizeC=3)
    for name, unit in (
            ('ms', 'ms'), ('min', 'min'), ('h', 'h'),
            ('us', '\u00b5s')):
        grid[f'units:time_{name}'] = spec_with(
            node=_node_with(TimeIncrementUnit=unit)
        )
    for name, unit in (
            ('nm', 'nm'), ('mm', 'mm'), ('um_ascii', 'um'),
            ('um_micro_sign', MICRO_SIGN_UM), ('um_greek_mu', GREEK_MU_UM)):
        grid[f'units:length_{name}'] = spec_with(
            node=_node_with(PhysicalSizeXUnit=unit)
        )
    return grid
