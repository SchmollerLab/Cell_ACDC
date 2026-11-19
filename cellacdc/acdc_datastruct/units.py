"""Unit conversion of the frame interval (to seconds) and the pixel size (to
micrometers) for Module 0 and the command line (standard library only).

The "Re-structure image files" dialog keeps its own time-unit table.
"""
from __future__ import annotations

from . import errors

GUI_TIME_UNIT_LABELS = ('ms', 'seconds', 'minutes', 'hours')
GUI_LENGTH_UNIT_LABELS = ('nm', 'μm', 'mm', 'cm')
CLI_TIME_UNITS = ('ms', 's', 'min', 'h')
CLI_LENGTH_UNITS = ('nm', 'um', 'mm', 'cm')

_MILLISECONDS = frozenset({'ms', 'millisecond', 'milliseconds'})
_SECONDS = frozenset({'s', 'sec', 'second', 'seconds'})
_MINUTES = frozenset({'min', 'minute', 'minutes'})
_HOURS = frozenset({'h', 'hour', 'hours'})
_MICROSECONDS = frozenset({'us', 'µs', 'μs'})
_NANOSECONDS = frozenset({'ns'})

_NANOMETERS = frozenset({'nm'})
_MICROMETERS = frozenset({
    'um', 'µm', 'μm', 'micron', 'microns', 'micrometer',
    'micrometers',
})
_MILLIMETERS = frozenset({'mm'})
_CENTIMETERS = frozenset({'cm'})


def time_increment_to_seconds(value: float, unit: str) -> float:
    """Convert a frame interval to seconds.

    ``unit`` (surrounding whitespace ignored) is a GUI label or an OME unit
    symbol: ms/millisecond(s) -> value / 1000; s/sec/second(s) ->
    float(value); min/minute(s) -> value * 60; h/hour(s) -> value * 3600;
    us/µs -> value / 1e6; ns -> value / 1e9 (same arithmetic as the metadata
    dialog's ``convertUnits``).

    Raises
    ------
    SettingsError
        Unknown unit.
    """
    value = float(value)
    key = str(unit).strip()
    if key in _MILLISECONDS:
        return value / 1000
    if key in _SECONDS:
        return value
    if key in _MINUTES:
        return value * 60
    if key in _HOURS:
        return value * 3600
    if key in _MICROSECONDS:
        return value / 1e6
    if key in _NANOSECONDS:
        return value / 1e9
    raise errors.SettingsError(
        f'Unknown time unit "{unit}" (known: ms, s, min, h, us, ns).',
        hint='Set the frame interval with --time-increment and --time-unit.'
    )


def physical_size_to_um(value: float, unit: str) -> float:
    """Convert a pixel size to micrometers.

    nm -> value / 1000; um/µm/micron(s)/micrometer(s) -> float(value);
    mm -> value * 1000; cm -> value * 1e4.

    Raises
    ------
    SettingsError
        Unknown unit.
    """
    value = float(value)
    key = str(unit).strip()
    if key in _NANOMETERS:
        return value / 1000
    if key in _MICROMETERS:
        return value
    if key in _MILLIMETERS:
        return value * 1000
    if key in _CENTIMETERS:
        return value * 1e4
    raise errors.SettingsError(
        f'Unknown length unit "{unit}" (known: nm, um, mm, cm).',
        hint=(
            'Set the pixel size with --physical-size-x/-y/-z and '
            '--physical-size-unit.'
        )
    )
