"""acdc_datastruct.units: frame interval to s, pixel size to um."""
from __future__ import annotations

import pytest

from cellacdc.acdc_datastruct import errors, units

VALUES = (0.0, 1.0, 0.1, 0.3, 1e-9, 2.5, 7, 13.37, 1234.5678, 1e12)


def _legacy_time(value, timeUnit):
    """Inline copy of the time part of QDialogMetadataXML.convertUnits."""
    TimeIncrement = value
    if timeUnit == 'ms':
        TimeIncrement /= 1000
    elif timeUnit == 'minutes':
        TimeIncrement *= 60
    elif timeUnit == 'hours':
        TimeIncrement *= 3600
    return TimeIncrement


def _intended_length(value, unit):
    """The physical-size arithmetic of convertUnits, comparing the
    physical unit (main compared the time unit, GV-03)."""
    if unit == 'nm':
        value /= 1000
    elif unit == 'mm':
        value *= 1000
    elif unit == 'cm':
        value *= 1e4
    return value


def test_labels():
    assert units.GUI_TIME_UNIT_LABELS == ('ms', 'seconds', 'minutes', 'hours')
    assert units.GUI_LENGTH_UNIT_LABELS == ('nm', 'μm', 'mm', 'cm')
    assert units.CLI_TIME_UNITS == ('ms', 's', 'min', 'h')
    assert units.CLI_LENGTH_UNITS == ('nm', 'um', 'mm', 'cm')


@pytest.mark.parametrize('label', units.GUI_TIME_UNIT_LABELS)
def test_time_bit_exact_vs_dialog(label):
    for value in VALUES:
        converted = units.time_increment_to_seconds(value, label)
        assert converted == _legacy_time(value, label)
        assert isinstance(converted, float)


@pytest.mark.parametrize('unit, factor', [
    ('ms', 1e-3), ('millisecond', 1e-3), ('milliseconds', 1e-3),
    ('s', 1.0), ('sec', 1.0), ('second', 1.0), ('seconds', 1.0),
    (' s ', 1.0),
    ('min', 60.0), ('minute', 60.0), ('minutes', 60.0),
    ('h', 3600.0), ('hour', 3600.0), ('hours', 3600.0),
    ('us', 1e-6), ('µs', 1e-6), ('μs', 1e-6),
    ('ns', 1e-9),
])
def test_time_symbols(unit, factor):
    assert units.time_increment_to_seconds(2.0, unit) == pytest.approx(
        2.0 * factor
    )


def test_time_exact_arithmetic():
    assert units.time_increment_to_seconds(0.3, 'us') == 0.3 / 1e6
    assert units.time_increment_to_seconds(7, 'ns') == 7 / 1e9
    assert units.time_increment_to_seconds(3, 's') == 3.0
    assert isinstance(units.time_increment_to_seconds(3, 's'), float)


@pytest.mark.parametrize('label', units.GUI_LENGTH_UNIT_LABELS)
def test_length_gui_labels(label):
    for value in VALUES:
        assert (
            units.physical_size_to_um(value, label)
            == _intended_length(value, label)
        )


@pytest.mark.parametrize('unit, factor', [
    ('nm', 1e-3), ('um', 1.0), ('µm', 1.0), ('μm', 1.0),
    ('micron', 1.0), ('microns', 1.0), ('micrometer', 1.0),
    ('micrometers', 1.0), ('mm', 1e3), ('cm', 1e4), (' nm', 1e-3),
])
def test_length_symbols(unit, factor):
    assert units.physical_size_to_um(0.5, unit) == pytest.approx(0.5 * factor)


def test_length_fixes_dialog_bug():
    """GV-03: nm/mm/cm are converted (main left the number unchanged)."""
    assert units.physical_size_to_um(65.0, 'nm') == 0.065
    assert units.physical_size_to_um(0.002, 'mm') == 2.0
    assert units.physical_size_to_um(3, 'um') == 3.0
    assert isinstance(units.physical_size_to_um(3, 'um'), float)


@pytest.mark.parametrize('unit', ['', 'd', 'Ms', 'sx', 'MIN', 'years'])
def test_unknown_time_unit(unit):
    with pytest.raises(errors.SettingsError) as info:
        units.time_increment_to_seconds(1.0, unit)
    assert '--time-unit' in info.value.hint
    assert isinstance(info.value, errors.UsageError)


@pytest.mark.parametrize('unit', ['', 'm', 'pixel', 'inch', 'km'])
def test_unknown_length_unit(unit):
    with pytest.raises(errors.SettingsError) as info:
        units.physical_size_to_um(1.0, unit)
    assert '--physical-size-unit' in info.value.hint
