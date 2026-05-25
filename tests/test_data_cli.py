import pytest

from cellacdc.data_cli import (
    HeadlessMetadataHandler,
    ParsedRawMetadata,
    RESTRUCTURE_LAYOUTS,
    build_convert_state,
    build_data_parser,
    get_start_pos_n,
    guess_basename_from_filepath,
    layout_to_raw_data_struct,
    metadata_to_dataframe,
    on_existing_to_worker_flags,
    parse_one_per_channel_files,
    parse_time_range,
    read_filename_pattern,
)


def test_build_data_parser_prog():
    assert build_data_parser().prog == 'acdc-data'


def test_layout_to_raw_data_struct():
    assert layout_to_raw_data_struct('single-multi-pos') == 0
    assert layout_to_raw_data_struct('one-per-pos') == 1
    assert layout_to_raw_data_struct('one-per-channel') == 2
    with pytest.raises(ValueError):
        layout_to_raw_data_struct('invalid-layout')


def test_parse_time_range():
    assert parse_time_range(None) == (0, None)
    assert parse_time_range('0:99') == (0, 99)
    assert parse_time_range('120') == (0, 120)


def test_on_existing_to_worker_flags():
    assert on_existing_to_worker_flags('overwrite') == {
        'overwrite': True, 'add_files': False, 'create_new': False,
    }
    assert on_existing_to_worker_flags('add') == {
        'overwrite': False, 'add_files': True, 'create_new': False,
    }
    assert on_existing_to_worker_flags('create-new') == {
        'overwrite': False, 'add_files': False, 'create_new': True,
    }


def test_parse_one_per_channel_files():
    files = ['ASY015_1_GFP.tif', 'ASY015_1_mNeon.tif', 'ASY015_2_GFP.tif']
    basename, pos_nums, ch_names = parse_one_per_channel_files(files)
    assert basename == 'ASY015_'
    assert pos_nums == [1, 2]
    assert set(ch_names) == {'GFP', 'mNeon'}


def test_headless_metadata_handler_from_args():
    handler = HeadlessMetadataHandler.from_parser_args({
        'trust_metadata': True,
        'format': 'h5',
        'channels': 'phase_contrast,GFP',
        'time_range': '0:10',
        'lens_na': 1.4,
    })
    assert handler.to_h5 is True
    assert handler.channels == ['phase_contrast', 'GFP']
    assert handler.time_range_end == 10
    assert handler.lens_na == 1.4


def test_metadata_to_dataframe():
    parsed = ParsedRawMetadata(
        lens_na=1.4,
        size_t=10,
        size_z=1,
        channel_names=['phase_contrast', 'GFP'],
        em_wavelens=[500.0, 525.0],
    )
    df = metadata_to_dataframe(parsed, 'test_exp_')
    assert df.at['basename', 'values'] == 'test_exp_'
    assert df.at['SizeT', 'values'] == 10
    assert df.at['channel_0_name', 'values'] == 'phase_contrast'
    assert df.at['channel_1_emWavelen', 'values'] == 525.0


def test_guess_basename_from_filepath():
    assert guess_basename_from_filepath('/path/to/Example1.czi') == 'Example1_'


def test_metadata_parser_accepts_required_args():
    args = build_data_parser().parse_args([
        'metadata', '--input', '/tmp/file.czi',
    ])
    assert args.command == 'metadata'
    assert args.input == '/tmp/file.czi'


def test_convert_parser_accepts_required_args():
    args = build_data_parser().parse_args([
        'convert',
        '--input', '/tmp/raw',
        '--output', '/tmp/exp',
        '--layout', 'one-per-pos',
    ])
    assert args.command == 'convert'
    assert args.layout == 'one-per-pos'
    assert args.trust_metadata is True


def test_restructure_parser_accepts_required_args():
    args = build_data_parser().parse_args([
        'restructure',
        '--input', '/tmp/raw',
        '--output', '/tmp/exp',
        '--layout', 'multi-timepoint',
        '--channels', 'GFP,mCherry',
    ])
    assert args.command == 'restructure'
    assert args.layout in RESTRUCTURE_LAYOUTS
    assert args.channels == 'GFP,mCherry'


def test_read_filename_pattern():
    pos, frame, ch = read_filename_pattern('pos1_GFP_01.tif')
    assert pos == 'pos1'
    assert frame == '01'
    assert ch == 'GFP'


def test_build_convert_state():
    parsed = ParsedRawMetadata(
        lens_na=1.4,
        size_t=10,
        size_z=1,
        channel_names=['phase_contrast'],
        em_wavelens=[500.0],
    )
    handler = HeadlessMetadataHandler(channels=['GFP'], to_h5=True)
    state = build_convert_state(parsed, handler)
    assert state.channel_names == ['GFP']
    assert state.to_h5 is True
    assert state.size_c == 1


def test_get_start_pos_n(tmp_path):
    assert get_start_pos_n(str(tmp_path), 'overwrite') == 1
    (tmp_path / 'Position_1').mkdir()
    (tmp_path / 'Position_3').mkdir()
    assert get_start_pos_n(str(tmp_path), 'create-new') == 4
