"""Metadata model of Module 0 and the command line.

- ``RawFileMetadata``: what the worker read from one raw file (lossless,
  None = not found).
- ``resolve_metadata``: legacy defaults and "not found" warnings
  (``ResolvedMetadata``, file units, what the metadata dialog shows).
- ``ConversionSettings``: confirmed settings (seconds, micrometers,
  sanitized channel names), built from the dialog
  (``settings_from_dialog_values``), from the previous file
  (``settings_for_next_unit``, Trust/Reuse buttons) or from the file plus
  command line overrides (``settings_from_metadata``).
- Parsers of the command line time-range and positions specs.

Time ranges are 1-based and inclusive ``(t0, t1)``; ``None`` means every
frame of the file being written.
"""
from __future__ import annotations

import csv
import dataclasses
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, NoReturn, Sequence

from . import errors, naming, sources, units

RAW_METADATA_SCHEMA = 1
_RAW_KEYS = (
    'schema', 'path', 'source', 'reader', 'lens_na', 'size_s', 'size_z',
    'size_t', 'time_increment', 'time_increment_unit', 'size_c',
    'physical_size_x', 'physical_size_y', 'physical_size_z',
    'physical_size_unit', 'image_name', 'channel_names', 'em_wavelens',
    'scene_names', 'size_y', 'size_x', 'dtype', 'metadata_xml',
)
_RAW_SEQUENCE_KEYS = ('channel_names', 'em_wavelens', 'scene_names')


@dataclass(frozen=True)
class RawFileMetadata:
    """Metadata values read from one raw file by the legacy accessor
    expressions; None = the legacy expression raised (not found)."""
    path: str
    source: Literal['ome_tif', 'bioio']
    reader: str
    lens_na: float | None
    size_s: int | None
    size_z: int | None
    size_t: int | None
    time_increment: float | None
    time_increment_unit: str | None
    size_c: int | None
    physical_size_x: float | None
    physical_size_y: float | None
    physical_size_z: float | None
    physical_size_unit: str | None
    image_name: str | None
    channel_names: tuple[str | None, ...]
    em_wavelens: tuple[float | None, ...]
    scene_names: tuple[str, ...] = ()
    size_y: int | None = None
    size_x: int | None = None
    dtype: str | None = None
    metadata_xml: str = ''
    schema: int = RAW_METADATA_SCHEMA

    def to_dict(self) -> dict:
        """JSON-safe dict with the keys of ``raw_metadata`` (sequences as
        lists)."""
        data = {}
        for key in _RAW_KEYS:
            value = getattr(self, key)
            data[key] = list(value) if key in _RAW_SEQUENCE_KEYS else value
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> 'RawFileMetadata':
        """Strict inverse of ``to_dict``.

        Raises
        ------
        ValueError
            Unknown or missing keys, or a schema other than 1.
        """
        unknown = sorted(set(data) - set(_RAW_KEYS))
        if unknown:
            raise ValueError(f'Unknown raw metadata keys: {unknown}')
        schema = data.get('schema', RAW_METADATA_SCHEMA)
        if schema != RAW_METADATA_SCHEMA:
            raise ValueError(
                f'Unsupported raw metadata schema {schema!r} '
                f'(expected {RAW_METADATA_SCHEMA})'
            )
        kwargs = dict(data)
        for key in _RAW_SEQUENCE_KEYS:
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        try:
            return cls(**kwargs)
        except TypeError as err:
            raise ValueError(f'Invalid raw metadata: {err}') from err


WARNING_FIELDS = (
    'LensNA', 'SizeS', 'SizeZ', 'SizeT', 'TimeIncrement',
    'TimeIncrementUnit', 'SizeC', 'PhysicalSizeX', 'PhysicalSizeY',
    'PhysicalSizeZ', 'PhysicalSizeUnit', 'Image Name'
)
_WARNING_SEPARATOR = '=' * 51


@dataclass(frozen=True)
class MetadataWarning:
    """A metadata field that was not found and fell back to its default."""
    field: str      # one of WARNING_FIELDS (legacy label)
    path: str       # raw file path as the legacy code printed it

    @property
    def message(self) -> str:
        return f'WARNING: {self.field} not found in metadata.'

    def gui_lines(self) -> tuple[str, str, str, str]:
        """The four log lines the legacy worker emitted."""
        return (
            _WARNING_SEPARATOR, self.path, self.message, _WARNING_SEPARATOR
        )


@dataclass(frozen=True)
class ResolvedMetadata:
    """Metadata with legacy defaults applied, in FILE units; what the
    metadata dialog is prefilled with."""
    lens_na: float
    size_s: int
    size_z: int
    size_t: int
    time_increment: float
    time_increment_unit: str
    size_c: int
    physical_size_x: float
    physical_size_y: float
    physical_size_z: float
    physical_size_unit: str
    image_name: str
    channel_names: tuple[str, ...]  # raw (unsanitized), '' if unknown
    em_wavelens: tuple[float, ...]
    defaulted: frozenset[str]       # every field that fell back


def resolve_metadata(
        raw: RawFileMetadata, *, layout: sources.RawLayout,
        one_per_channel: sources.OnePerChannelIndex | None = None
    ) -> tuple[ResolvedMetadata, tuple[MetadataWarning, ...]]:
    """Apply the legacy defaults of the metadata reader
    (``bioFormatsWorker.readMetadata``) with its warnings, in legacy order.

    Defaults: LensNA 1.4; SizeS 1; SizeZ 1; SizeT 1; TimeIncrement 1.0;
    TimeIncrementUnit 's'; SizeC 1; PhysicalSizeX/Y/Z 1.0;
    PhysicalSizeUnit 'μm'; Image Name ''; channel name '' and emission
    wavelength 0.0 per channel (no warning). One file per channel: SizeS is
    the number of positions (no warning), channel names are the raw tokens
    and every emission wavelength is 500.0; the SizeC warning is kept when
    the file has no SizeC, exactly as the legacy reader did.

    ``defaulted`` lists every field that fell back, with the warning labels
    plus 'chNames' and 'EmissionWavelength' for the silent per-channel
    defaults.
    """
    one_per_channel_layout = (
        layout == sources.RawLayout.ONE_FILE_PER_CHANNEL
    )
    if one_per_channel_layout and one_per_channel is None:
        raise ValueError(
            'resolve_metadata: one_per_channel is required for the '
            'one-file-per-channel layout'
        )
    warnings: list[MetadataWarning] = []
    defaulted: set[str] = set()

    def value_or_default(field: str, value, default):
        if value is not None:
            return value
        warnings.append(MetadataWarning(field, raw.path))
        defaulted.add(field)
        return default

    lens_na = value_or_default('LensNA', raw.lens_na, 1.4)
    if one_per_channel_layout:
        size_s = len(one_per_channel.position_ids)
    else:
        size_s = value_or_default('SizeS', raw.size_s, 1)
    size_z = value_or_default('SizeZ', raw.size_z, 1)
    size_t = value_or_default('SizeT', raw.size_t, 1)
    time_increment = value_or_default(
        'TimeIncrement', raw.time_increment, 1.0
    )
    time_increment_unit = value_or_default(
        'TimeIncrementUnit', raw.time_increment_unit, 's'
    )
    size_c = value_or_default('SizeC', raw.size_c, 1)
    physical_size_x = value_or_default(
        'PhysicalSizeX', raw.physical_size_x, 1.0
    )
    physical_size_y = value_or_default(
        'PhysicalSizeY', raw.physical_size_y, 1.0
    )
    physical_size_z = value_or_default(
        'PhysicalSizeZ', raw.physical_size_z, 1.0
    )
    physical_size_unit = value_or_default(
        'PhysicalSizeUnit', raw.physical_size_unit, 'μm'
    )
    image_name = value_or_default('Image Name', raw.image_name, '')

    if one_per_channel_layout:
        defaulted.discard('SizeC')
        channel_names = tuple(one_per_channel.channel_tokens)
        size_c = len(channel_names)
        em_wavelens = (500.0,) * size_c
    else:
        channel_names = tuple(
            _item(raw.channel_names, c) or '' for c in range(size_c)
        )
        emissions = [_item(raw.em_wavelens, c) for c in range(size_c)]
        em_wavelens = tuple(
            0.0 if emission is None else emission for emission in emissions
        )
        if any(not name for name in channel_names):
            defaulted.add('chNames')
        if any(emission is None for emission in emissions):
            defaulted.add('EmissionWavelength')

    resolved = ResolvedMetadata(
        lens_na=lens_na, size_s=size_s, size_z=size_z, size_t=size_t,
        time_increment=time_increment,
        time_increment_unit=time_increment_unit, size_c=size_c,
        physical_size_x=physical_size_x, physical_size_y=physical_size_y,
        physical_size_z=physical_size_z,
        physical_size_unit=physical_size_unit, image_name=image_name,
        channel_names=channel_names, em_wavelens=em_wavelens,
        defaulted=frozenset(defaulted),
    )
    return resolved, tuple(warnings)


def _item(values: Sequence, index: int):
    return values[index] if index < len(values) else None


_CONFIRM_VALUE_KEYS = (
    'LensNA', 'SizeT', 'SizeZ', 'SizeC', 'SizeS', 'TimeIncrement',
    'PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ', 'chNames',
    'emWavelens',
)


def confirm_signal_args(
        resolved: ResolvedMetadata, *, filename: str, raw_path: str,
        sample: object, values: Mapping[str, object] | None = None
    ) -> tuple:
    """The 17 positional arguments of ``bioFormatsWorker.confirmMetadata``
    in signal order: filename, LensNA, SizeT, SizeZ, SizeC, SizeS,
    TimeIncrement, TimeIncrementUnit, PhysicalSizeX, PhysicalSizeY,
    PhysicalSizeZ, PhysicalSizeUnit, chNames, emWavelens, ImageName,
    rawFilePath, sampleImgData.

    ``values`` (keys LensNA, SizeT, SizeZ, SizeC, SizeS, TimeIncrement,
    PhysicalSizeX/Y/Z, chNames, emWavelens) overrides the resolved values
    after an "eye" re-read (dialog-edited, not unit-converted).
    ``filename`` must be ``unit.metadata_file`` for every layout.
    """
    current = {
        'LensNA': resolved.lens_na,
        'SizeT': resolved.size_t,
        'SizeZ': resolved.size_z,
        'SizeC': resolved.size_c,
        'SizeS': resolved.size_s,
        'TimeIncrement': resolved.time_increment,
        'PhysicalSizeX': resolved.physical_size_x,
        'PhysicalSizeY': resolved.physical_size_y,
        'PhysicalSizeZ': resolved.physical_size_z,
        'chNames': list(resolved.channel_names),
        'emWavelens': list(resolved.em_wavelens),
    }
    if values is not None:
        unknown = sorted(set(values) - set(_CONFIRM_VALUE_KEYS))
        if unknown:
            raise ValueError(f'Unknown confirm values: {unknown}')
        current.update(values)
        current['chNames'] = list(current['chNames'])
        current['emWavelens'] = list(current['emWavelens'])
    return (
        filename, current['LensNA'], current['SizeT'], current['SizeZ'],
        current['SizeC'], current['SizeS'], current['TimeIncrement'],
        resolved.time_increment_unit, current['PhysicalSizeX'],
        current['PhysicalSizeY'], current['PhysicalSizeZ'],
        resolved.physical_size_unit, current['chNames'],
        current['emWavelens'], resolved.image_name, raw_path, sample
    )


# ---------------------------------------------------------- time ranges
_TIME_RANGE_HINT = (
    'Use START:END, START:, :END or N (frames are 1-based and inclusive, '
    'e.g. 5:20).'
)


def validate_time_range(
        t0: int, t1: int, size_t: int | None, *, flag: str = 'time range'
    ) -> None:
    """Check a 1-based inclusive frame range.

    Raises
    ------
    SettingsError
        Unless ``1 <= t0 <= t1`` and (``size_t`` is None or
        ``t1 <= size_t``); the message names ``flag``.
    """
    problems = []
    if t0 < 1:
        problems.append(
            f'the first frame ({t0}) must be at least 1 (frames are '
            '1-based)'
        )
    if t1 < t0:
        problems.append(
            f'the last frame ({t1}) must not be before the first frame '
            f'({t0})'
        )
    if size_t is not None and t1 > size_t:
        problems.append(
            f'the last frame ({t1}) exceeds the number of frames ({size_t})'
        )
    if problems:
        raise errors.SettingsError(
            f'Invalid {flag} {t0}:{t1}: {"; ".join(problems)}.',
            problems=[f'{flag}: {problem}' for problem in problems],
            hint=_TIME_RANGE_HINT
        )


def _parse_frame(text: str, spec: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    try:
        frame = int(text)
    except ValueError:
        frame = None
    if frame is None or frame < 1:
        raise errors.UsageError(
            f'Invalid --time-range "{spec}": "{text}" is not a frame number '
            '(an integer >= 1).',
            hint=_TIME_RANGE_HINT
        )
    return frame


def parse_time_range_spec(spec: str) -> tuple[int | None, int | None]:
    """Parse ``--time-range``: 'START:END', 'START:', ':END' or 'N'
    (= N:N), 1-based inclusive. Open ends are None.

    Raises
    ------
    UsageError
        Malformed spec, a frame below 1 or START > END.
    """
    if spec.count(':') > 1:
        raise errors.UsageError(
            f'Invalid --time-range "{spec}": more than one ":".',
            hint=_TIME_RANGE_HINT
        )
    if ':' in spec:
        start_text, _, end_text = spec.partition(':')
        start = _parse_frame(start_text, spec)
        end = _parse_frame(end_text, spec)
    else:
        start = end = _parse_frame(spec, spec)
    if start is None and end is None:
        raise errors.UsageError(
            f'Invalid --time-range "{spec}": no frame given.',
            hint=_TIME_RANGE_HINT
        )
    if start is not None and end is not None and start > end:
        raise errors.UsageError(
            f'Invalid --time-range "{spec}": START ({start}) is after END '
            f'({end}).',
            hint=_TIME_RANGE_HINT
        )
    return start, end


def resolve_time_range(
        spec: tuple[int | None, int | None] | None, size_t: int
    ) -> tuple[int, int] | None:
    """Fill the open ends of a parsed spec with 1 and ``size_t`` and
    validate it (``flag='--time-range'``). None, or a range equal to
    ``(1, size_t)``, gives None (all frames)."""
    if spec is None:
        return None
    t0 = 1 if spec[0] is None else spec[0]
    t1 = size_t if spec[1] is None else spec[1]
    validate_time_range(t0, t1, size_t, flag='--time-range')
    if (t0, t1) == (1, size_t):
        return None
    return t0, t1


# ------------------------------------------------------------- positions
_POSITION_RE = re.compile(r'^(?:position_)?(\d+)$', re.IGNORECASE)
_POSITION_RANGE_RE = re.compile(
    r'^(?:position_)?(\d+)\s*-\s*(?:position_)?(\d+)$', re.IGNORECASE
)
_POSITIONS_HINT = (
    'Use comma-separated N, Position_N or A-B items (1-based), '
    'e.g. 1,3,5-7.'
)


def parse_positions_spec(spec: str) -> frozenset[int]:
    """Parse ``--positions``: comma-separated 'N', 'Position_N' or 'A-B'
    items (input position ids, 1-based), e.g. '1,3,5-7'.

    Raises
    ------
    UsageError
        Empty spec or item, malformed item, id below 1 or A > B.
    """
    def fail(reason: str) -> NoReturn:
        raise errors.UsageError(
            f'Invalid --positions "{spec}": {reason}.', hint=_POSITIONS_HINT
        )

    ids: set[int] = set()
    items = spec.split(',')
    for item in items:
        item = item.strip()
        if not item:
            fail('empty item')
        single = _POSITION_RE.match(item)
        span = _POSITION_RANGE_RE.match(item)
        if single:
            first = last = int(single.group(1))
        elif span:
            first, last = int(span.group(1)), int(span.group(2))
            if first > last:
                fail(f'"{item}" is a decreasing range')
        else:
            fail(f'"{item}" is not N, Position_N or A-B')
        if first < 1:
            fail(f'"{item}": positions start at 1')
        ids.update(range(first, last + 1))
    return frozenset(ids)


def positions_from_gui_labels(labels: Iterable[str]) -> frozenset[int] | None:
    """Selected positions of the metadata dialog: None (all) when
    'All Positions' (capital P) is present, else the k of every
    'Position_k'. Other labels, such as the list box item 'All positions',
    are ignored (legacy ``saveToPosFolder``)."""
    labels = list(labels)
    if 'All Positions' in labels:
        return None
    selected = set()
    for label in labels:
        match = re.fullmatch(r'Position_(\d+)', label)
        if match:
            selected.add(int(match.group(1)))
    return frozenset(selected)


# -------------------------------------------------------------- settings
ApplyToNext = Literal['ask', 'trust', 'reuse']


@dataclass(frozen=True)
class ConversionSettings:
    """Confirmed (dialog) or derived (command line) settings of one
    metadata unit: TimeIncrement in s, PhysicalSize in um, sanitized
    channel names."""
    to_h5: bool
    lens_na: float
    size_t: int
    size_z: int
    size_c: int
    size_s: int
    time_increment_s: float
    physical_size_x_um: float
    physical_size_y_um: float
    physical_size_z_um: float
    channel_names: tuple[str, ...]          # sanitized, len == size_c
    save_channels: tuple[bool, ...]         # len == size_c
    em_wavelens: tuple[float, ...]          # len == size_c
    time_range: tuple[int, int] | None = None   # 1-based inclusive
    selected_positions: frozenset[int] | None = None   # input ids
    add_image_name: bool = False

    def frames(self, size_t: int | None = None) -> tuple[int, int]:
        """``time_range``, or all frames ``(1, size_t)`` (``size_t``
        defaults to the confirmed SizeT)."""
        if self.time_range is not None:
            return self.time_range
        return 1, self.size_t if size_t is None else size_t

    def saved_size_t(self, size_t: int | None = None) -> int:
        """Number of saved frames (metadata.csv SizeT)."""
        t0, t1 = self.frames(size_t)
        return t1 - t0 + 1

    def errors(self) -> list[str]:
        """Every problem of the settings ([] = valid): list lengths equal
        to size_c, saved channel names non-empty and unique ignoring case,
        at least one saved channel, time range shape ``1 <= t0 <= t1``
        (its upper bound is checked against the file, not against
        ``size_t``) and a non-empty position selection."""
        problems = []
        for name in ('channel_names', 'save_channels', 'em_wavelens'):
            length = len(getattr(self, name))
            if length != self.size_c:
                problems.append(
                    f'{length} {name.replace("_", " ")} for '
                    f'{self.size_c} channels (SizeC).'
                )
        problems.extend(naming.validate_channel_names(
            self.channel_names, self.save_channels
        ))
        if not any(self.save_channels):
            problems.append(
                'No channel selected: select at least one channel to save.'
            )
        if self.time_range is not None:
            if len(self.time_range) != 2:
                problems.append(
                    f'Invalid time range {self.time_range}: expected '
                    '(first frame, last frame).'
                )
            else:
                t0, t1 = self.time_range
                if not 1 <= t0 <= t1:
                    problems.append(
                        f'Invalid time range {t0}:{t1}: frames are 1-based '
                        'and the first frame must not be after the last one.'
                    )
        if self.selected_positions is not None and not self.selected_positions:
            problems.append('No position selected.')
        return problems

    def validate(self, *, context: str = '') -> None:
        """Raise ``SettingsError`` (with ``problems``) when ``errors()`` is
        not empty; ``context`` (e.g. the file name) goes in the message."""
        problems = self.errors()
        if not problems:
            return
        where = f' for {context}' if context else ''
        details = '\n'.join(f'  - {problem}' for problem in problems)
        raise errors.SettingsError(
            f'Invalid conversion settings{where}:\n{details}',
            problems=problems
        )

    def to_dict(self) -> dict:
        """JSON-able dict (inverse of ``from_dict``)."""
        data = dataclasses.asdict(self)
        for key in ('channel_names', 'save_channels', 'em_wavelens'):
            data[key] = list(data[key])
        if self.time_range is not None:
            data['time_range'] = list(self.time_range)
        if self.selected_positions is not None:
            data['selected_positions'] = sorted(self.selected_positions)
        return data

    @classmethod
    def from_dict(cls, d: Mapping) -> 'ConversionSettings':
        """Settings from ``to_dict()`` output.

        Raises
        ------
        ValueError
            Unknown or missing keys.
        """
        names = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise ValueError(f'Unknown conversion settings keys: {unknown}')
        kwargs = dict(d)
        for key in ('channel_names', 'save_channels', 'em_wavelens'):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        if kwargs.get('time_range') is not None:
            kwargs['time_range'] = tuple(kwargs['time_range'])
        if kwargs.get('selected_positions') is not None:
            kwargs['selected_positions'] = frozenset(
                kwargs['selected_positions']
            )
        try:
            return cls(**kwargs)
        except TypeError as err:
            raise ValueError(f'Invalid conversion settings: {err}') from err


def settings_from_dialog_values(
        win: object, *, file_size_t: int | None
    ) -> tuple[ConversionSettings, ApplyToNext]:
    """Settings from the attributes of a closed ``QDialogMetadataXML``
    (after ``ok_cb``: TimeIncrement already in s, PhysicalSize in um,
    chNames sanitized). Duck-typed and pure.

    Time range: ``timeRangeToSave`` (1-based inclusive, exactly what the
    legacy worker passed to the writer) becomes None (all frames) only when
    ``file_size_t`` is known, ``win.SizeT == file_size_t`` and the range is
    ``(1, file_size_t)``; otherwise it is kept explicitly, also when the
    widget was hidden because SizeT was edited to 1.

    The second item is 'reuse' ("Use the above metadata for all the next
    positions"), else 'trust' ("Trust metadata reader"), else 'ask'.
    """
    time_range = tuple(win.timeRangeToSave)
    if (
        file_size_t is not None and win.SizeT == file_size_t
        and time_range == (1, file_size_t)
    ):
        time_range = None
    settings = ConversionSettings(
        to_h5=bool(win.to_h5),
        lens_na=win.LensNA,
        size_t=win.SizeT,
        size_z=win.SizeZ,
        size_c=win.SizeC,
        size_s=win.SizeS,
        time_increment_s=win.TimeIncrement,
        physical_size_x_um=win.PhysicalSizeX,
        physical_size_y_um=win.PhysicalSizeY,
        physical_size_z_um=win.PhysicalSizeZ,
        channel_names=tuple(naming.sanitize_name(n) for n in win.chNames),
        save_channels=tuple(bool(save) for save in win.saveChannels),
        em_wavelens=tuple(win.emWavelens),
        time_range=time_range,
        selected_positions=positions_from_gui_labels(win.selectedPos),
        add_image_name=bool(win.addImageName),
    )
    if getattr(win, 'overWrite', False):
        apply_to_next = 'reuse'
    elif getattr(win, 'trust', False):
        apply_to_next = 'trust'
    else:
        apply_to_next = 'ask'
    return settings, apply_to_next


def settings_for_next_unit(
        mode: ApplyToNext, previous: ConversionSettings,
        resolved: ResolvedMetadata
    ) -> ConversionSettings | None:
    """Settings of the next file without asking ("Trust metadata reader" /
    "Use the above metadata for all the next positions").

    'ask' -> None. 'reuse' -> ``previous`` unchanged. 'trust' -> the
    reader's LensNA, sizes, TimeIncrement converted to s, pixel sizes
    converted to um and emission wavelengths, with the sticky user choices
    of ``previous`` (names, save flags, positions, h5, image name, time
    range). Both return None (the caller asks for this file) when the
    file's channel count differs from ``previous``, when an explicit time
    range ends after the file's last frame, or (trust) when a unit is
    unknown.
    """
    if mode == 'ask':
        return None
    if resolved.size_c != len(previous.channel_names):
        return None
    if (
        previous.time_range is not None
        and previous.time_range[1] > resolved.size_t
    ):
        return None
    if mode == 'reuse':
        return previous
    try:
        time_increment_s = units.time_increment_to_seconds(
            resolved.time_increment, resolved.time_increment_unit
        )
        physical_sizes_um = [
            units.physical_size_to_um(value, resolved.physical_size_unit)
            for value in (
                resolved.physical_size_x, resolved.physical_size_y,
                resolved.physical_size_z
            )
        ]
    except errors.SettingsError:
        return None
    return dataclasses.replace(
        previous,
        lens_na=resolved.lens_na,
        size_t=resolved.size_t,
        size_z=resolved.size_z,
        size_c=resolved.size_c,
        size_s=resolved.size_s,
        time_increment_s=time_increment_s,
        physical_size_x_um=physical_sizes_um[0],
        physical_size_y_um=physical_sizes_um[1],
        physical_size_z_um=physical_sizes_um[2],
        em_wavelens=tuple(resolved.em_wavelens),
    )


# ------------------------------------------------- command line overrides
@dataclass(frozen=True)
class MetadataOverrides:
    """Command line flags / ``--metadata-csv`` values; None keeps the file
    value."""
    lens_na: float | None = None
    time_increment_s: float | None = None
    physical_size_x_um: float | None = None
    physical_size_y_um: float | None = None
    physical_size_z_um: float | None = None
    channel_names: tuple[str, ...] | None = None    # unsanitized
    em_wavelens: tuple[float, ...] | None = None
    save_channels: tuple[str, ...] | None = None    # FINAL names to save
    time_range: tuple[int | None, int | None] | None = None
    positions: frozenset[int] | None = None
    to_h5: bool = False
    add_image_name: bool = False

    def merged_over(self, lower: 'MetadataOverrides') -> 'MetadataOverrides':
        """``self`` on top of ``lower`` (flags over CSV): every None of
        ``self`` takes the value of ``lower``; flags are or-ed."""
        values = {}
        for field in dataclasses.fields(self):
            mine = getattr(self, field.name)
            theirs = getattr(lower, field.name)
            if isinstance(mine, bool):
                values[field.name] = mine or theirs
            else:
                values[field.name] = theirs if mine is None else mine
        return MetadataOverrides(**values)


_CSV_FLOAT_KEYS = {
    'LensNA': 'lens_na',
    'TimeIncrement': 'time_increment_s',
    'PhysicalSizeX': 'physical_size_x_um',
    'PhysicalSizeY': 'physical_size_y_um',
    'PhysicalSizeZ': 'physical_size_z_um',
}
_CSV_IGNORED_KEYS = ('basename', 'SizeT', 'SizeZ')
_CSV_CHANNEL_RE = re.compile(r'^channel_(\d+)_(name|emWavelen)$')
_CSV_ALLOWED_TEXT = (
    'LensNA, TimeIncrement (s), PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ '
    '(um), channel_<c>_name, channel_<c>_emWavelen (and the ignored '
    'basename, SizeT, SizeZ)'
)


def overrides_from_metadata_csv(
        path: str
    ) -> tuple[MetadataOverrides, list[str]]:
    """Overrides from a ``Description,values`` CSV (as written by
    ``acdc-data metadata --format csv`` or found in Position folders).

    Keys: LensNA, TimeIncrement (s), PhysicalSizeX/Y/Z (um),
    channel_{c}_name, channel_{c}_emWavelen (c contiguous from 0).
    basename, SizeT and SizeZ are ignored with a warning (returned).

    Raises
    ------
    SettingsError
        Unreadable file, wrong header, duplicate or unknown key, value that
        is not a number, non-contiguous channel indices.
    """
    hint = 'Fix the file passed to --metadata-csv.'
    try:
        with open(path, 'r', newline='', encoding='utf-8') as file:
            rows = [row for row in csv.reader(file) if row]
    except (OSError, UnicodeDecodeError, csv.Error) as err:
        raise errors.SettingsError(
            f'Cannot read the metadata CSV {path}: {err}', hint=hint
        ) from err
    if not rows or [cell.strip() for cell in rows[0]] != [
            'Description', 'values']:
        raise errors.SettingsError(
            f'The metadata CSV {path} must start with the header '
            '"Description,values".', hint=hint
        )
    problems = []
    warnings = []
    values: dict[str, str] = {}
    for row in rows[1:]:
        if len(row) != 2:
            problems.append(f'row {row} does not have 2 columns')
            continue
        key, value = row[0].strip(), row[1].strip()
        if key in values:
            problems.append(f'"{key}" is given more than once')
        values[key] = value

    kwargs: dict[str, object] = {}
    channel_values: dict[str, dict[int, str]] = {'name': {}, 'emWavelen': {}}
    for key, value in values.items():
        if key in _CSV_IGNORED_KEYS:
            warnings.append(
                f'The metadata CSV row "{key}" is ignored (it is computed '
                'from the file and the selected frames).'
            )
        elif key in _CSV_FLOAT_KEYS:
            number = _csv_float(key, value, problems)
            if number is not None:
                kwargs[_CSV_FLOAT_KEYS[key]] = number
        elif _CSV_CHANNEL_RE.match(key):
            match = _CSV_CHANNEL_RE.match(key)
            channel_values[match.group(2)][int(match.group(1))] = value
        else:
            problems.append(f'unknown row "{key}"')

    names = _csv_channel_list(channel_values['name'], 'name', problems)
    if names is not None:
        kwargs['channel_names'] = tuple(names)
    wavelens = _csv_channel_list(
        channel_values['emWavelen'], 'emWavelen', problems
    )
    if wavelens is not None:
        numbers = [
            _csv_float(f'channel_{c}_emWavelen', value, problems)
            for c, value in enumerate(wavelens)
        ]
        if None not in numbers:
            kwargs['em_wavelens'] = tuple(numbers)
    if problems:
        details = '\n'.join(f'  - {problem}' for problem in problems)
        raise errors.SettingsError(
            f'Invalid metadata CSV {path}:\n{details}\n'
            f'Allowed rows: {_CSV_ALLOWED_TEXT}.',
            problems=problems, hint=hint
        )
    return MetadataOverrides(**kwargs), warnings


def _csv_float(key: str, value: str, problems: list[str]) -> float | None:
    try:
        return float(value)
    except ValueError:
        problems.append(f'"{key}" value "{value}" is not a number')
        return None


def _csv_channel_list(
        by_index: Mapping[int, str], kind: str, problems: list[str]
    ) -> list[str] | None:
    if not by_index:
        return None
    indices = sorted(by_index)
    if indices != list(range(len(indices))):
        problems.append(
            f'channel_<c>_{kind} rows must be numbered 0, 1, 2, ... without '
            f'gaps (found {indices})'
        )
        return None
    return [by_index[c] for c in indices]


def settings_from_metadata(
        resolved: ResolvedMetadata, overrides: MetadataOverrides | None, *,
        layout: sources.RawLayout, context: str,
        scene_names: Sequence[str] | None = None
    ) -> tuple[ConversionSettings, list[str]]:
    """Command line resolution: trust the file, apply the overrides.

    Units are converted to s/um. Channel names come from the overrides or
    from the file and are sanitized; a file name that is empty after
    sanitizing becomes 'channel_{c}' with a warning. Lists must have SizeC
    items. ``save_channels`` names (after sanitizing, ignoring case) become
    flags. The time range is resolved against the file's SizeT.
    ``add_image_name`` needs a non-empty scene name in ``scene_names``
    (``RawFileMetadata.scene_names``; default: the resolved Image Name).

    Returns
    -------
    (ConversionSettings, list of warning texts)

    Raises
    ------
    SettingsError
        Unknown unit, wrong list lengths, unknown channel to save, invalid
        time range, no image name for ``add_image_name``, or invalid
        settings (``validate(context=context)``).
    """
    o = MetadataOverrides() if overrides is None else overrides
    warnings = []
    where = f'{context}: ' if context else ''
    size_c = resolved.size_c
    if layout == sources.RawLayout.ONE_FILE_PER_CHANNEL:
        channels_text = (
            f'the {size_c} channels found in the file names'
        )
    else:
        channels_text = f'the {size_c} channels (SizeC) of the file'

    lens_na = resolved.lens_na if o.lens_na is None else o.lens_na
    if o.time_increment_s is None:
        time_increment_s = units.time_increment_to_seconds(
            resolved.time_increment, resolved.time_increment_unit
        )
    else:
        time_increment_s = o.time_increment_s
    physical_sizes_um = []
    for axis in ('x', 'y', 'z'):
        value = getattr(o, f'physical_size_{axis}_um')
        if value is None:
            value = units.physical_size_to_um(
                getattr(resolved, f'physical_size_{axis}'),
                resolved.physical_size_unit
            )
        physical_sizes_um.append(value)

    if o.channel_names is None:
        channel_names = []
        for c, name in enumerate(resolved.channel_names):
            sanitized = naming.sanitize_name(name)
            if not sanitized:
                sanitized = f'channel_{c}'
                warnings.append(
                    f'{where}channel {c} has no name in the file metadata; '
                    f'using "{sanitized}" (set the names with --channels).'
                )
            channel_names.append(sanitized)
    else:
        if len(o.channel_names) != size_c:
            raise errors.SettingsError(
                f'{where}{len(o.channel_names)} channel names given for '
                f'{channels_text}.',
                hint='Give one name per channel with --channels.'
            )
        channel_names = [naming.sanitize_name(n) for n in o.channel_names]

    if o.em_wavelens is None:
        em_wavelens = tuple(resolved.em_wavelens)
    else:
        if len(o.em_wavelens) != size_c:
            raise errors.SettingsError(
                f'{where}{len(o.em_wavelens)} emission wavelengths given '
                f'for {channels_text}.',
                hint='Give one wavelength per channel with --em-wavelens.'
            )
        em_wavelens = tuple(o.em_wavelens)

    if o.save_channels is None:
        save_channels = (True,) * size_c
    else:
        keys = [name.casefold() for name in channel_names]
        wanted = {naming.sanitize_name(n).casefold() for n in o.save_channels}
        unknown = sorted(
            name for name in o.save_channels
            if naming.sanitize_name(name).casefold() not in keys
        )
        if unknown:
            raise errors.SettingsError(
                f'{where}unknown channels to save: {", ".join(unknown)} '
                f'(channels: {", ".join(channel_names)}).',
                hint='--save-channels takes channel names (after --channels).'
            )
        save_channels = tuple(key in wanted for key in keys)

    try:
        time_range = resolve_time_range(o.time_range, resolved.size_t)
    except errors.SettingsError as err:
        raise errors.SettingsError(
            f'{where}{err.message}', problems=err.problems, hint=err.hint
        ) from None

    if o.add_image_name:
        if scene_names is None:
            scene_names = (resolved.image_name,)
        if not any(scene_names):
            raise errors.SettingsError(
                f'{where}the file has no image name, so --add-image-name '
                'cannot add it to the file names.',
                hint='Remove --add-image-name.'
            )

    settings = ConversionSettings(
        to_h5=o.to_h5,
        lens_na=lens_na,
        size_t=resolved.size_t,
        size_z=resolved.size_z,
        size_c=size_c,
        size_s=resolved.size_s,
        time_increment_s=time_increment_s,
        physical_size_x_um=physical_sizes_um[0],
        physical_size_y_um=physical_sizes_um[1],
        physical_size_z_um=physical_sizes_um[2],
        channel_names=tuple(channel_names),
        save_channels=save_channels,
        em_wavelens=em_wavelens,
        time_range=time_range,
        selected_positions=o.positions,
        add_image_name=o.add_image_name,
    )
    settings.validate(context=context)
    return settings, warnings
