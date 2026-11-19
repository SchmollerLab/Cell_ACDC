"""acdc_datastruct.metadata: raw metadata model, legacy defaults and
warnings, settings (dialog, trust/reuse, command line) and spec parsers."""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from cellacdc.acdc_datastruct import errors, metadata, sources

RawLayout = sources.RawLayout

FULL = dict(
    schema=1, path='C:\\raw\\exp1.czi', source='bioio', reader='bioio_czi',
    lens_na=1.3, size_s=3, size_z=5, size_t=10, time_increment=500.0,
    time_increment_unit='ms', size_c=2, physical_size_x=0.1,
    physical_size_y=0.1, physical_size_z=0.5, physical_size_unit='\u00b5m',
    image_name='Scene #1', channel_names=['GFP', 'mCherry'],
    em_wavelens=[510.0, 610.0], scene_names=['Scene #1', '', 'S3'],
    size_y=16, size_x=20, dtype='uint16', metadata_xml='<OME/>',
)
INDEX = sources.parse_one_per_channel_filenames([
    'ASY015_1_GFP.tif', 'ASY015_1_mNeon.tif', 'ASY015_2_GFP.tif',
    'ASY015_2_mNeon.tif', 'ASY015_3_GFP.tif', 'ASY015_3_mNeon.tif',
])


def _raw(**changes) -> metadata.RawFileMetadata:
    return metadata.RawFileMetadata.from_dict({**FULL, **changes})


def _resolved(**changes) -> metadata.ResolvedMetadata:
    resolved, _ = metadata.resolve_metadata(
        _raw(**changes), layout=RawLayout.ONE_FILE_PER_POS
    )
    return resolved


def _settings(**changes) -> metadata.ConversionSettings:
    values = dict(
        to_h5=False, lens_na=1.3, size_t=10, size_z=5, size_c=2, size_s=1,
        time_increment_s=0.5, physical_size_x_um=0.1, physical_size_y_um=0.1,
        physical_size_z_um=0.5, channel_names=('GFP', 'mCherry'),
        save_channels=(True, True), em_wavelens=(510.0, 610.0),
    )
    values.update(changes)
    return metadata.ConversionSettings(**values)


# ---------------------------------------------------------- raw metadata
def test_raw_metadata_round_trip_json():
    raw = _raw(
        lens_na=None, channel_names=['GFP', None], em_wavelens=[None, 0.1],
        image_name='Ünïcode µ \u4e2d', time_increment=1 / 3,
        metadata_xml='<OME>\u00b5m</OME>'
    )
    data = json.loads(json.dumps(raw.to_dict(), allow_nan=True))
    assert list(data) == list(metadata._RAW_KEYS)
    again = metadata.RawFileMetadata.from_dict(data)
    assert again == raw
    assert again.channel_names == ('GFP', None)
    assert again.time_increment == 1 / 3


def test_raw_metadata_from_dict_is_strict():
    with pytest.raises(ValueError, match='Unknown'):
        metadata.RawFileMetadata.from_dict({**FULL, 'extra': 1})
    with pytest.raises(ValueError, match='schema'):
        metadata.RawFileMetadata.from_dict({**FULL, 'schema': 2})
    missing = dict(FULL)
    del missing['size_t']
    with pytest.raises(ValueError):
        metadata.RawFileMetadata.from_dict(missing)
    optional = {k: v for k, v in FULL.items() if k not in (
        'scene_names', 'size_y', 'size_x', 'dtype', 'metadata_xml', 'schema'
    )}
    raw = metadata.RawFileMetadata.from_dict(optional)
    assert raw.scene_names == () and raw.metadata_xml == ''


# ------------------------------------------------------------- resolving
def test_resolve_full_metadata_has_no_warning():
    resolved, warnings = metadata.resolve_metadata(
        _raw(), layout=RawLayout.SINGLE_FILE_MULTI_POS
    )
    assert warnings == ()
    assert resolved.defaulted == frozenset()
    assert resolved == metadata.ResolvedMetadata(
        lens_na=1.3, size_s=3, size_z=5, size_t=10, time_increment=500.0,
        time_increment_unit='ms', size_c=2, physical_size_x=0.1,
        physical_size_y=0.1, physical_size_z=0.5,
        physical_size_unit='\u00b5m', image_name='Scene #1',
        channel_names=('GFP', 'mCherry'), em_wavelens=(510.0, 610.0),
        defaulted=frozenset(),
    )


def test_resolve_all_missing_legacy_defaults_and_order():
    nothing = {
        key: None for key in (
            'lens_na', 'size_s', 'size_z', 'size_t', 'time_increment',
            'time_increment_unit', 'size_c', 'physical_size_x',
            'physical_size_y', 'physical_size_z', 'physical_size_unit',
            'image_name',
        )
    }
    raw = _raw(**nothing, channel_names=[None], em_wavelens=[None])
    resolved, warnings = metadata.resolve_metadata(
        raw, layout=RawLayout.ONE_FILE_PER_POS
    )
    assert [w.field for w in warnings] == list(metadata.WARNING_FIELDS)
    assert all(w.path == FULL['path'] for w in warnings)
    assert (
        resolved.lens_na, resolved.size_s, resolved.size_z, resolved.size_t,
        resolved.time_increment, resolved.time_increment_unit,
        resolved.size_c, resolved.physical_size_x, resolved.physical_size_y,
        resolved.physical_size_z, resolved.physical_size_unit,
        resolved.image_name, resolved.channel_names, resolved.em_wavelens,
    ) == (
        1.4, 1, 1, 1, 1.0, 's', 1, 1.0, 1.0, 1.0, '\u03bcm', '', ('',),
        (0.0,)
    )
    assert resolved.defaulted == frozenset(
        metadata.WARNING_FIELDS + ('chNames', 'EmissionWavelength')
    )


@pytest.mark.parametrize('key, field', [
    ('lens_na', 'LensNA'), ('size_s', 'SizeS'), ('size_z', 'SizeZ'),
    ('size_t', 'SizeT'), ('time_increment', 'TimeIncrement'),
    ('time_increment_unit', 'TimeIncrementUnit'),
    ('physical_size_x', 'PhysicalSizeX'), ('physical_size_y', 'PhysicalSizeY'),
    ('physical_size_z', 'PhysicalSizeZ'),
    ('physical_size_unit', 'PhysicalSizeUnit'), ('image_name', 'Image Name'),
])
def test_resolve_single_missing_field(key, field):
    resolved, warnings = metadata.resolve_metadata(
        _raw(**{key: None}), layout=RawLayout.ONE_FILE_PER_POS
    )
    assert [w.field for w in warnings] == [field]
    assert resolved.defaulted == {field}


def test_resolve_missing_size_c_uses_one_channel():
    resolved, warnings = metadata.resolve_metadata(
        _raw(size_c=None, channel_names=['GFP'], em_wavelens=[510.0]),
        layout=RawLayout.ONE_FILE_PER_POS
    )
    assert [w.field for w in warnings] == ['SizeC']
    assert resolved.size_c == 1
    assert resolved.channel_names == ('GFP',)


def test_resolve_per_channel_defaults_are_silent():
    resolved, warnings = metadata.resolve_metadata(
        _raw(channel_names=[None, 'RFP'], em_wavelens=[510.0, None]),
        layout=RawLayout.ONE_FILE_PER_POS
    )
    assert warnings == ()
    assert resolved.channel_names == ('', 'RFP')
    assert resolved.em_wavelens == (510.0, 0.0)
    assert resolved.defaulted == {'chNames', 'EmissionWavelength'}
    # Fewer channel entries than SizeC (legacy: '' and 0.0 for the rest)
    resolved, _ = metadata.resolve_metadata(
        _raw(size_c=3), layout=RawLayout.ONE_FILE_PER_POS
    )
    assert resolved.channel_names == ('GFP', 'mCherry', '')
    assert resolved.em_wavelens == (510.0, 610.0, 0.0)


def test_resolve_one_per_channel():
    resolved, warnings = metadata.resolve_metadata(
        _raw(size_s=None, size_c=5), layout=RawLayout.ONE_FILE_PER_CHANNEL,
        one_per_channel=INDEX
    )
    assert warnings == ()
    assert resolved.size_s == 3
    assert resolved.size_c == 2
    assert resolved.channel_names == ('GFP', 'mNeon')
    assert resolved.em_wavelens == (500.0, 500.0)
    # The legacy reader warned about a missing SizeC also in this layout
    resolved, warnings = metadata.resolve_metadata(
        _raw(size_c=None), layout=RawLayout.ONE_FILE_PER_CHANNEL,
        one_per_channel=INDEX
    )
    assert [w.field for w in warnings] == ['SizeC']
    assert resolved.size_c == 2 and 'SizeC' not in resolved.defaulted
    with pytest.raises(ValueError):
        metadata.resolve_metadata(
            _raw(), layout=RawLayout.ONE_FILE_PER_CHANNEL
        )


def test_metadata_warning_text():
    warning = metadata.MetadataWarning('SizeT', 'C:\\raw\\a.tif')
    assert warning.message == 'WARNING: SizeT not found in metadata.'
    assert warning.gui_lines() == (
        '=' * 51, 'C:\\raw\\a.tif', 'WARNING: SizeT not found in metadata.',
        '=' * 51
    )


def test_confirm_signal_args():
    resolved = _resolved()
    args = metadata.confirm_signal_args(
        resolved, filename='exp1.czi', raw_path='C:\\raw\\exp1.czi',
        sample={'x': 1}
    )
    assert args == (
        'exp1.czi', 1.3, 10, 5, 2, 3, 500.0, 'ms', 0.1, 0.1, 0.5, '\u00b5m',
        ['GFP', 'mCherry'], [510.0, 610.0], 'Scene #1', 'C:\\raw\\exp1.czi',
        {'x': 1}
    )
    edited = metadata.confirm_signal_args(
        resolved, filename='f', raw_path='p', sample=None,
        values={'SizeT': 4, 'TimeIncrement': 2.0, 'chNames': ('a', 'b'),
                'emWavelens': (1.0, 2.0)}
    )
    assert edited[2] == 4 and edited[6] == 2.0 and edited[7] == 'ms'
    assert edited[12] == ['a', 'b'] and edited[13] == [1.0, 2.0]
    with pytest.raises(ValueError):
        metadata.confirm_signal_args(
            resolved, filename='f', raw_path='p', sample=None,
            values={'SizeX': 1}
        )


# ------------------------------------------------------------ time range
def test_validate_time_range():
    metadata.validate_time_range(1, 10, 10)
    metadata.validate_time_range(3, 3, None)
    for t0, t1, size_t in ((0, 3, 10), (4, 3, 10), (2, 11, 10), (-1, 2, None)):
        with pytest.raises(errors.SettingsError) as info:
            metadata.validate_time_range(t0, t1, size_t, flag='--time-range')
        assert '--time-range' in info.value.message
        assert info.value.problems


@pytest.mark.parametrize('spec, expected', [
    ('5:20', (5, 20)), ('5:', (5, None)), (':20', (None, 20)),
    ('7', (7, 7)), (' 2 : 3 ', (2, 3)), ('1:1', (1, 1)),
])
def test_parse_time_range_spec(spec, expected):
    assert metadata.parse_time_range_spec(spec) == expected


@pytest.mark.parametrize('spec', [
    '', ':', '0:3', '3:2', 'a:b', '1:2:3', '-1', '1.5', '0',
])
def test_parse_time_range_spec_errors(spec):
    with pytest.raises(errors.UsageError, match='--time-range'):
        metadata.parse_time_range_spec(spec)


def test_resolve_time_range():
    assert metadata.resolve_time_range(None, 10) is None
    assert metadata.resolve_time_range((None, None), 10) is None
    assert metadata.resolve_time_range((1, 10), 10) is None
    assert metadata.resolve_time_range((1, None), 10) is None
    assert metadata.resolve_time_range((2, None), 10) == (2, 10)
    assert metadata.resolve_time_range((None, 3), 10) == (1, 3)
    with pytest.raises(errors.SettingsError, match='--time-range'):
        metadata.resolve_time_range((5, 12), 10)


# ------------------------------------------------------------- positions
@pytest.mark.parametrize('spec, expected', [
    ('1', {1}), ('1,3,5-7', {1, 3, 5, 6, 7}), ('Position_3', {3}),
    ('position_2, 4', {2, 4}), ('2-2', {2}),
    ('Position_1-Position_3', {1, 2, 3}),
])
def test_parse_positions_spec(spec, expected):
    assert metadata.parse_positions_spec(spec) == frozenset(expected)


@pytest.mark.parametrize('spec', ['', '1,,2', '0', '3-1', 'x', 'Pos_1', '1-'])
def test_parse_positions_spec_errors(spec):
    with pytest.raises(errors.UsageError, match='--positions'):
        metadata.parse_positions_spec(spec)


def test_positions_from_gui_labels():
    assert metadata.positions_from_gui_labels(['All Positions']) is None
    assert metadata.positions_from_gui_labels(
        ['All positions', 'Position_1', 'Position_3']
    ) == {1, 3}
    assert metadata.positions_from_gui_labels(['All positions']) == frozenset()
    assert metadata.positions_from_gui_labels([]) == frozenset()


# -------------------------------------------------------------- settings
def test_settings_frames_and_saved_size_t():
    settings = _settings()
    assert settings.frames() == (1, 10)
    assert settings.frames(7) == (1, 7)
    assert settings.saved_size_t(7) == 7
    settings = _settings(time_range=(2, 4))
    assert settings.frames(7) == (2, 4)
    assert settings.saved_size_t() == 3


def test_settings_errors():
    assert _settings().errors() == []
    # The range is not bounded by the (dialog-edited) size_t
    assert _settings(size_t=1, time_range=(1, 5)).errors() == []
    cases = {
        'channel names': _settings(channel_names=('GFP',)),
        'save channels': _settings(save_channels=(True,)),
        'em wavelens': _settings(em_wavelens=(1.0, 2.0, 3.0)),
        'empty': _settings(channel_names=('', 'mCherry')),
        'same name': _settings(channel_names=('GFP', 'gfp')),
        'No channel selected': _settings(save_channels=(False, False)),
        'Invalid time range 0:3': _settings(time_range=(0, 3)),
        'Invalid time range 4:3': _settings(time_range=(4, 3)),
        'No position selected': _settings(selected_positions=frozenset()),
    }
    for text, settings in cases.items():
        problems = settings.errors()
        assert any(text in p for p in problems), (text, problems)
        with pytest.raises(errors.SettingsError) as info:
            settings.validate(context='exp1.czi')
        assert 'exp1.czi' in info.value.message
        assert info.value.problems == problems
    # Unsaved channels are not validated
    assert _settings(
        channel_names=('', 'mCherry'), save_channels=(False, True)
    ).errors() == []


def test_settings_dict_round_trip():
    settings = _settings(
        time_range=(2, 4), selected_positions=frozenset({3, 1}),
        add_image_name=True, to_h5=True
    )
    data = json.loads(json.dumps(settings.to_dict()))
    assert data['selected_positions'] == [1, 3]
    assert metadata.ConversionSettings.from_dict(data) == settings
    assert metadata.ConversionSettings.from_dict(
        json.loads(json.dumps(_settings().to_dict()))
    ) == _settings()
    with pytest.raises(ValueError):
        metadata.ConversionSettings.from_dict({**data, 'bogus': 1})


def _dialog(**changes):
    values = dict(
        to_h5=False, LensNA=1.4, SizeT=10, SizeZ=5, SizeC=2, SizeS=1,
        timeRangeToSave=(1, 10), TimeIncrement=0.5, PhysicalSizeX=0.1,
        PhysicalSizeY=0.1, PhysicalSizeZ=0.5, selectedPos=['All Positions'],
        chNames=['GFP', 'mNeon_Green'], saveChannels=[True, False],
        emWavelens=[510.0, 520.0], addImageName=False, trust=False,
        overWrite=False,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_settings_from_dialog_values():
    settings, mode = metadata.settings_from_dialog_values(
        _dialog(), file_size_t=10
    )
    assert mode == 'ask'
    assert settings == metadata.ConversionSettings(
        to_h5=False, lens_na=1.4, size_t=10, size_z=5, size_c=2, size_s=1,
        time_increment_s=0.5, physical_size_x_um=0.1, physical_size_y_um=0.1,
        physical_size_z_um=0.5, channel_names=('GFP', 'mNeon_Green'),
        save_channels=(True, False), em_wavelens=(510.0, 520.0),
        time_range=None, selected_positions=None, add_image_name=False,
    )


@pytest.mark.parametrize('changes, file_size_t, expected', [
    ({}, 10, None),
    ({}, None, (1, 10)),
    ({'timeRangeToSave': (2, 10)}, 10, (2, 10)),
    # SizeT edited 10 -> 5: the range widget was reset to (1, 5)
    ({'SizeT': 5, 'timeRangeToSave': (1, 5)}, 10, (1, 5)),
    # SizeT edited 5 -> 1: hidden widget keeps the stale (1, 5)
    ({'SizeT': 1, 'timeRangeToSave': (1, 5)}, 5, (1, 5)),
    # The user raised SizeT above the file's frame count
    ({'SizeT': 12, 'timeRangeToSave': (1, 12)}, 10, (1, 12)),
])
def test_settings_from_dialog_time_range_rule(changes, file_size_t, expected):
    settings, _ = metadata.settings_from_dialog_values(
        _dialog(**changes), file_size_t=file_size_t
    )
    assert settings.time_range == expected
    assert settings.errors() == []


def test_settings_from_dialog_buttons_and_positions():
    _, mode = metadata.settings_from_dialog_values(
        _dialog(trust=True), file_size_t=10
    )
    assert mode == 'trust'
    _, mode = metadata.settings_from_dialog_values(
        _dialog(trust=True, overWrite=True), file_size_t=10
    )
    assert mode == 'reuse'
    settings, _ = metadata.settings_from_dialog_values(
        _dialog(selectedPos=['All positions', 'Position_2'],
                addImageName=True, to_h5=True),
        file_size_t=10
    )
    assert settings.selected_positions == {2}
    assert settings.add_image_name and settings.to_h5


def test_settings_for_next_unit_ask_and_reuse():
    previous = _settings(time_range=(2, 8))
    resolved = _resolved(size_t=9)
    assert metadata.settings_for_next_unit('ask', previous, resolved) is None
    assert metadata.settings_for_next_unit(
        'reuse', previous, resolved
    ) is previous


@pytest.mark.parametrize('mode', ['trust', 'reuse'])
def test_settings_for_next_unit_reasks(mode):
    previous = _settings(time_range=(2, 8))
    # different number of channels
    assert metadata.settings_for_next_unit(
        mode, previous,
        _resolved(size_c=3, channel_names=['a', 'b', 'c'],
                  em_wavelens=[1.0, 2.0, 3.0])
    ) is None
    # explicit time range ending after the file's last frame
    assert metadata.settings_for_next_unit(
        mode, previous, _resolved(size_t=7)
    ) is None
    # a full range means all frames: any SizeT fits
    assert metadata.settings_for_next_unit(
        mode, _settings(), _resolved(size_t=3)
    ) is not None


def test_settings_for_next_unit_trust_converts_units():
    previous = _settings(
        channel_names=('A', 'B'), save_channels=(True, False),
        time_range=(1, 3), selected_positions=frozenset({1}), to_h5=True,
        add_image_name=True
    )
    resolved = _resolved(
        time_increment=2.0, time_increment_unit='min', size_t=4,
        physical_size_x=65.0, physical_size_y=130.0, physical_size_z=300.0,
        physical_size_unit='nm', lens_na=0.75, size_z=7,
        em_wavelens=[500.0, 600.0]
    )
    trusted = metadata.settings_for_next_unit('trust', previous, resolved)
    assert trusted == dataclasses.replace(
        previous, lens_na=0.75, size_t=4, size_z=7, size_c=2, size_s=3,
        time_increment_s=120.0, physical_size_x_um=0.065,
        physical_size_y_um=0.13, physical_size_z_um=0.3,
        em_wavelens=(500.0, 600.0)
    )
    assert metadata.settings_for_next_unit(
        'trust', previous, _resolved(time_increment_unit='fortnight')
    ) is None
    assert metadata.settings_for_next_unit(
        'trust', previous, _resolved(physical_size_unit='inch')
    ) is None


# --------------------------------------------------- command line overrides
def test_overrides_merged_over():
    flags = metadata.MetadataOverrides(
        lens_na=1.2, channel_names=('a', 'b'), to_h5=True
    )
    csv_values = metadata.MetadataOverrides(
        lens_na=0.9, time_increment_s=3.0, channel_names=('x', 'y'),
        add_image_name=True
    )
    merged = flags.merged_over(csv_values)
    assert merged == metadata.MetadataOverrides(
        lens_na=1.2, time_increment_s=3.0, channel_names=('a', 'b'),
        to_h5=True, add_image_name=True
    )


def _write(path, text):
    path.write_text(text, encoding='utf-8')
    return str(path)


def test_overrides_from_metadata_csv(tmp_path):
    path = _write(tmp_path / 'm.csv', (
        'Description,values\n'
        'LensNA,1.2\nSizeT,5\nSizeZ,3\nTimeIncrement,0.25\n'
        'PhysicalSizeZ,0.5\nPhysicalSizeY,0.1\nPhysicalSizeX,0.11\n'
        'basename,exp_s1_\n'
        'channel_0_name,GFP\nchannel_1_name,mNeon-Green\n'
        'channel_0_emWavelen,510.0\nchannel_1_emWavelen,520\n'
    ))
    overrides, warnings = metadata.overrides_from_metadata_csv(path)
    assert overrides == metadata.MetadataOverrides(
        lens_na=1.2, time_increment_s=0.25, physical_size_x_um=0.11,
        physical_size_y_um=0.1, physical_size_z_um=0.5,
        channel_names=('GFP', 'mNeon-Green'), em_wavelens=(510.0, 520.0)
    )
    assert len(warnings) == 3
    assert all('ignored' in w for w in warnings)


@pytest.mark.parametrize('text, message', [
    ('a,b\nLensNA,1\n', 'header'),
    ('Description,values\nSizeC,3\n', 'unknown row "SizeC"'),
    ('Description,values\nLensNA,abc\n', 'not a number'),
    ('Description,values\nLensNA,1\nLensNA,2\n', 'more than once'),
    ('Description,values\nchannel_1_name,GFP\n', 'without gaps'),
    ('Description,values\nchannel_0_emWavelen,x\n', 'not a number'),
    ('Description,values\nLensNA\n', '2 columns'),
])
def test_overrides_from_metadata_csv_errors(tmp_path, text, message):
    path = _write(tmp_path / 'bad.csv', text)
    with pytest.raises(errors.SettingsError) as info:
        metadata.overrides_from_metadata_csv(path)
    assert message in info.value.message
    assert '--metadata-csv' in info.value.hint
    with pytest.raises(errors.SettingsError):
        metadata.overrides_from_metadata_csv(str(tmp_path / 'missing.csv'))


def test_settings_from_metadata_defaults():
    resolved = _resolved()
    settings, warnings = metadata.settings_from_metadata(
        resolved, None, layout=RawLayout.ONE_FILE_PER_POS, context='exp1.czi'
    )
    assert warnings == []
    assert settings == metadata.ConversionSettings(
        to_h5=False, lens_na=1.3, size_t=10, size_z=5, size_c=2, size_s=3,
        time_increment_s=0.5, physical_size_x_um=0.1, physical_size_y_um=0.1,
        physical_size_z_um=0.5, channel_names=('GFP', 'mCherry'),
        save_channels=(True, True), em_wavelens=(510.0, 610.0),
    )


def test_settings_from_metadata_overrides():
    overrides = metadata.MetadataOverrides(
        lens_na=1.0, time_increment_s=2.0, physical_size_x_um=0.2,
        channel_names=('phase contrast', 'mNeon-Green'),
        em_wavelens=(0.0, 520.0), save_channels=('mNeon-Green',),
        time_range=(2, None), positions=frozenset({1, 3}), to_h5=True,
        add_image_name=True,
    )
    settings, warnings = metadata.settings_from_metadata(
        _resolved(), overrides, layout=RawLayout.SINGLE_FILE_MULTI_POS,
        context='exp1.czi', scene_names=('', 'Scene'),
    )
    assert warnings == []
    assert settings.channel_names == ('phase_contrast', 'mNeon_Green')
    assert settings.save_channels == (False, True)
    assert settings.em_wavelens == (0.0, 520.0)
    assert settings.time_range == (2, 10)
    assert settings.selected_positions == {1, 3}
    assert (settings.lens_na, settings.time_increment_s) == (1.0, 2.0)
    assert settings.physical_size_x_um == 0.2
    assert settings.physical_size_y_um == 0.1
    assert settings.to_h5 and settings.add_image_name


def test_settings_from_metadata_csv_precedence(tmp_path):
    path = _write(tmp_path / 'm.csv', (
        'Description,values\nLensNA,0.9\nTimeIncrement,7\n'
        'channel_0_name,a\nchannel_1_name,b\n'
    ))
    csv_overrides, _ = metadata.overrides_from_metadata_csv(path)
    flags = metadata.MetadataOverrides(lens_na=1.1)
    settings, _ = metadata.settings_from_metadata(
        _resolved(), flags.merged_over(csv_overrides),
        layout=RawLayout.ONE_FILE_PER_POS, context='x'
    )
    assert settings.lens_na == 1.1               # flag > CSV
    assert settings.time_increment_s == 7.0      # CSV > file
    assert settings.channel_names == ('a', 'b')
    assert settings.physical_size_z_um == 0.5    # file


def test_settings_from_metadata_unnamed_channels_warn():
    settings, warnings = metadata.settings_from_metadata(
        _resolved(channel_names=[None, '!!']), None,
        layout=RawLayout.ONE_FILE_PER_POS, context='a.tif'
    )
    assert settings.channel_names == ('channel_0', 'channel_1')
    assert len(warnings) == 2 and '--channels' in warnings[0]
    assert warnings[0].startswith('a.tif: ')


def test_settings_from_metadata_errors():
    resolved = _resolved()
    layout = RawLayout.ONE_FILE_PER_POS
    bad = [
        (metadata.MetadataOverrides(channel_names=('a',)), '--channels'),
        (metadata.MetadataOverrides(em_wavelens=(1.0,)), '--em-wavelens'),
        (metadata.MetadataOverrides(save_channels=('RFP',)),
         '--save-channels'),
        (metadata.MetadataOverrides(time_range=(3, 12)), '--time-range'),
        (metadata.MetadataOverrides(add_image_name=True), '--add-image-name'),
        (metadata.MetadataOverrides(channel_names=('a!', 'A?')), None),
        (metadata.MetadataOverrides(channel_names=('', 'b')), None),
    ]
    for overrides, flag in bad:
        with pytest.raises(errors.SettingsError) as info:
            metadata.settings_from_metadata(
                resolved, overrides, layout=layout, context='f.czi',
                scene_names=('', '')
            )
        text = f'{info.value.message} {info.value.hint}'
        assert flag is None or flag in text, (flag, text)
    with pytest.raises(errors.SettingsError) as info:
        metadata.settings_from_metadata(
            _resolved(time_increment_unit='d'), None, layout=layout,
            context='f'
        )
    assert '--time-unit' in info.value.hint
    # One file per channel: the channel count message names the tokens
    resolved_2, _ = metadata.resolve_metadata(
        _raw(), layout=RawLayout.ONE_FILE_PER_CHANNEL, one_per_channel=INDEX
    )
    with pytest.raises(errors.SettingsError, match='found in the file names'):
        metadata.settings_from_metadata(
            resolved_2, metadata.MetadataOverrides(channel_names=('a',)),
            layout=RawLayout.ONE_FILE_PER_CHANNEL, context='ASY015_'
        )


def test_add_image_name_uses_resolved_image_name_by_default():
    settings, _ = metadata.settings_from_metadata(
        _resolved(), metadata.MetadataOverrides(add_image_name=True),
        layout=RawLayout.ONE_FILE_PER_POS, context='x'
    )
    assert settings.add_image_name
    with pytest.raises(errors.SettingsError, match='--add-image-name'):
        metadata.settings_from_metadata(
            _resolved(image_name=None),
            metadata.MetadataOverrides(add_image_name=True),
            layout=RawLayout.ONE_FILE_PER_POS, context='x'
        )
