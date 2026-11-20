"""acdc_datastruct.sources: listing, checks, selection, one-per-channel
parsing and metadata units (no bioio: supported extensions are
monkeypatched)."""
from __future__ import annotations

import json
import os
import random
import shutil

import pytest
from natsort import natsorted

from cellacdc import io, myutils
from cellacdc.acdc_bioio_bioformats import _utils as bioio_utils
from cellacdc.acdc_datastruct import errors, sources

from . import conditions

RawLayout = sources.RawLayout
SUPPORTED = frozenset({'.tif', '.tiff', '.czi', '.nd2', '.ome.tiff'})


@pytest.fixture
def supported(monkeypatch):
    """get_supported_image_extensions without bioio (independent of the
    installed reader plugins)."""
    monkeypatch.setattr(
        bioio_utils, 'get_supported_image_extensions',
        lambda: {'.TIF', '.tiff', '.czi ', '.nd2', '.ome.tiff'}
    )
    return SUPPORTED


def _touch(folder, *names):
    os.makedirs(folder, exist_ok=True)
    for name in names:
        with open(os.path.join(folder, name), 'wb') as file:
            file.write(name.encode('utf-8'))
    return str(folder)


# ------------------------------------------------------------------ layout
def test_raw_layout_cli_names():
    assert RawLayout.from_cli('single-multi-pos') is RawLayout(0)
    assert RawLayout.from_cli('one-per-pos') is RawLayout(1)
    assert RawLayout.from_cli('one-per-channel') is RawLayout(2)
    for layout in RawLayout:
        assert RawLayout.from_cli(layout.cli_name) is layout
    with pytest.raises(errors.UsageError):
        RawLayout.from_cli('one_per_pos')


# ----------------------------------------------------------------- listing
def test_list_source_files_filters_like_the_dialog(tmp_path):
    src = _touch(
        tmp_path / 'src', 'a10.tif', 'a2.tif', '.hidden.tif', 'desktop.ini',
        'recovery', 'x.new.npz', 'B1.tif'
    )
    os.makedirs(os.path.join(src, 'subfolder'))
    assert sources.list_source_files(src) == ['B1.tif', 'a2.tif', 'a10.tif']
    with pytest.raises(errors.SourceError):
        sources.list_source_files(str(tmp_path / 'missing'))


@pytest.mark.parametrize('filename, expected', [
    ('a.tif', '.tif'), ('a.TIF', '.tif'), ('a.ome.tif', '.ome.tif'),
    ('a.OME.TIFF', '.ome.tiff'), ('a.ome.zarr', '.ome.zarr'),
    ('a.czi', '.czi'), ('noext', ''), ('a.b.nd2', '.nd2'),
])
def test_normalized_extension(filename, expected):
    assert sources.normalized_extension(filename) == expected


def test_extension_counts_keeps_case():
    counts = sources.extension_counts(['a.tif', 'b.TIF', 'c.ome.tif'])
    assert counts == {'.tif': 2, '.TIF': 1}


def test_find_invalid_filenames():
    files = [
        'a b.tif', 'a-b_c.czi', 'x.ome.tif', 'a!b.tif', 'a.b.tif',
        'x.OME.tif', 'éa.tif', 'a,b.tif', 'a.b.c.d.tif'
    ]
    assert sources.find_invalid_filenames(files) == [
        'a!b.tif', 'a,b.tif', 'a.b.c.d.tif'
    ]


def legacy_rename_files_replace_invalid_chars(
        files, src_path, replacement_char='_'):
    """`io.rename_files_replace_invalid_chars` on main (what the setup
    dialog's rename buttons called): renames file by file, no collision
    check."""
    renamed_files = []
    for file in files:
        invalid_chars = io._validate_filename(file, is_path=False)
        new_file = file
        for char in invalid_chars:
            new_file = new_file.replace(char, replacement_char)
        if new_file != file:
            src_filepath = os.path.join(src_path, file)
            dst_filepath = os.path.join(src_path, new_file)
            os.rename(src_filepath, dst_filepath)
        renamed_files.append(new_file)
    return renamed_files


INVALID_NAMES = (
    'a b.tif', 'a!b#.tif', 'ok.tif', 'x (1).czi', 'é ü.nd2',
    'a,b;c.tif', 'x.ome.tif', 'a +=@&.tif', 'μm 5.tif'
)


@pytest.mark.parametrize('replacement', ['_', '-'])
def test_invalid_name_renames_equal_main(tmp_path, replacement):
    """planned_valid_filename and plan_invalid_name_renames (read-only,
    order of the files) give the names of main's io function."""
    names = list(INVALID_NAMES)
    oracle = _touch(tmp_path / 'oracle', *names)
    expected = legacy_rename_files_replace_invalid_chars(
        names, oracle, replacement_char=replacement
    )
    planned = [sources.planned_valid_filename(n, replacement) for n in names]
    assert planned == expected

    folder = _touch(tmp_path / 'copy', *names)
    before = conditions.tree_snapshot(folder)
    renames = sources.plan_invalid_name_renames(folder, names, replacement)
    assert conditions.tree_snapshot(folder) == before
    assert list(renames.items()) == [
        (old, new) for old, new in zip(names, expected) if new != old
    ]


def test_invalid_name_renames_list_every_collision(tmp_path):
    """'a b.tif' -> 'a_b.tif' would replace the raw file a_b.tif (main's io
    function: FileExistsError on Windows, after renaming the files listed
    before it, if any; the file silently replaced on Linux/macOS): every
    collision is listed and nothing is renamed."""
    names = ['a b.tif', 'a_b.tif', 'p!.tif', 'p#.tif', 'x+y.tif']
    src = _touch(tmp_path / 'src', *names)
    before = conditions.tree_snapshot(src)
    with pytest.raises(errors.SourceError, match='overwrite') as info:
        sources.plan_invalid_name_renames(src, names, '_')
    message = info.value.message
    assert 'a b.tif -> a_b.tif: a_b.tif already exists' in message
    assert 'a b.tif and a_b.tif would both be named a_b.tif' in message
    assert 'p!.tif and p#.tif would both be named p_.tif' in message
    assert 'x+y.tif' not in message
    # Only the target exists (not among the files to rename)
    with pytest.raises(errors.SourceError, match='a_b.tif already exists'):
        sources.plan_invalid_name_renames(src, ['a b.tif', 'x+y.tif'], '_')
    assert conditions.tree_snapshot(src) == before


def test_invalid_name_renames_case_variants(tmp_path):
    """A new name that differs from another name only in case collides;
    unchanged names that differ only in case do not (they already coexist
    on a case-sensitive file system; the files need not exist)."""
    src = str(tmp_path)
    assert sources.plan_invalid_name_renames(
        src, ['Pos1.czi', 'pos1.czi', 'x+y.czi'], '_'
    ) == {'x+y.czi': 'x_y.czi'}
    for files in (['A b.czi', 'a_b.czi'], ['a_b.czi', 'A b.czi'],
                  ['A b.czi', 'a b.czi'], ['a-b.czi', 'A b.czi']):
        replacement = '-' if 'a-b.czi' in files else '_'
        with pytest.raises(errors.SourceError,
                           match='which differ only in case'):
            sources.plan_invalid_name_renames(src, files, replacement)


def _collisions(error):
    """The problems listed by a collision error, without the bullets."""
    return [line[len('  - '):] for line in error.message.splitlines()[1:]]


def test_invalid_name_renames_report_every_colliding_pair(tmp_path):
    """'a+b.czi' -> 'a_b.czi' collides with 'a_b.czi' (same name) and with
    'A_B.czi' (same name ignoring case); the unchanged 'A_B.czi' and
    'a_b.czi' do not collide (the files need not exist)."""
    with pytest.raises(errors.SourceError) as info:
        sources.plan_invalid_name_renames(
            str(tmp_path), ['A_B.czi', 'a+b.czi', 'a_b.czi'], '_'
        )
    assert _collisions(info.value) == [
        'A_B.czi and a+b.czi would be named A_B.czi and a_b.czi, which '
        'differ only in case',
        'a+b.czi and a_b.czi would both be named a_b.czi',
    ]


def test_name_collisions_brute_force():
    """Every pair of files whose final names are equal ignoring case is
    reported once, in the order of its second file, unless both names are
    unchanged and the files are not moved; the words say which case it
    is."""
    rng = random.Random(3)
    names = ['a.czi', 'A.czi', 'b.czi', 'B.czi', 'a+.czi', 'x!.czi']
    targets = ['a.czi', 'A.czi', 'b.czi', 'B.czi', 'c.czi']
    for _ in range(500):
        files = rng.sample(names, rng.randint(2, len(names)))
        final = {
            f: rng.choice(targets) if rng.random() < 0.4 else f
            for f in files
        }
        moved_to = rng.choice([None, 'dst'])
        expected = []
        for j, second in enumerate(files):
            for first in files[:j]:
                a, b = final[first], final[second]
                pair = f'{first} and {second}'
                if a.casefold() != b.casefold():
                    continue
                if a == b:
                    expected.append(f'{pair} would both be named {a}')
                elif a != first or b != second:
                    expected.append(
                        f'{pair} would be named {a} and {b}, which differ '
                        'only in case'
                    )
                elif moved_to is not None:
                    expected.append(
                        f'{pair} differ only in case and cannot both be '
                        'moved to dst (its file system may ignore case)'
                    )
        assert sources.name_collisions(
            files, final, moved_to=moved_to
        ) == expected, (files, final, moved_to)


def test_supported_extensions_lowercases_and_strips(supported):
    assert sources.supported_extensions() == supported


def test_find_unsupported_files(supported):
    files = ['a.TIF', 'b.ome.tif', 'c.OME.TIFF', 'd.txt', 'e.CZI', 'f']
    assert sources.find_unsupported_files(files) == ['d.txt', 'f']
    assert sources.find_unsupported_files(
        files, supported=['.TXT', '.ome.tif']
    ) == ['a.TIF', 'c.OME.TIFF', 'e.CZI', 'f']
    assert sources.find_unsupported_files(['z.ome.zarr'], ['.ome.zarr']) == []


def test_one_file_per_extension():
    assert sources.one_file_per_extension(['a.czi', 'b.czi', 'c.nd2']) == [
        'a.czi', 'c.nd2'
    ]
    assert sources.one_file_per_extension(
        ['x.ome.tif', 'y.tif', 'z.OME.TIF', 'w.TIF']
    ) == ['x.ome.tif', 'y.tif']
    assert sources.one_file_per_extension([]) == []


# --------------------------------------------------------------- selection
def _select(src, layout=RawLayout.ONE_FILE_PER_POS, **kwargs):
    before = conditions.tree_snapshot(src)
    try:
        return sources.select_source_files(src, layout, **kwargs)
    finally:
        assert conditions.tree_snapshot(src) == before


def test_select_all_files(tmp_path, supported):
    src = _touch(tmp_path / 'src', 'p2.tif', 'p10.tif', 'p1.tif')
    selection = _select(src)
    assert selection.files == ('p1.tif', 'p2.tif', 'p10.tif')
    assert selection.renames == {}
    assert selection.ignored == ()


def test_select_missing_or_empty_source(tmp_path, supported):
    with pytest.raises(errors.SourceError, match='does not exist'):
        sources.select_source_files(
            str(tmp_path / 'nope'), RawLayout.ONE_FILE_PER_POS
        )
    empty = str(tmp_path / 'empty')
    os.makedirs(empty)
    with pytest.raises(errors.SourceError, match='does not contain'):
        _select(empty)


def test_select_mixed_extensions(tmp_path, supported):
    src = _touch(tmp_path / 'src', 'a.czi', 'b.CZI', 'notes.txt')
    with pytest.raises(errors.SourceError) as info:
        _select(src)
    assert '.czi: 1' in info.value.message and '.txt: 1' in info.value.message
    assert '--ext' in info.value.hint

    selection = _select(src, ext='.CZI')
    assert selection.files == ('a.czi', 'b.CZI')
    assert selection.ignored == ('notes.txt',)
    assert _select(src, ext='czi').files == ('a.czi', 'b.CZI')

    with pytest.raises(errors.SourceError, match='No file with the extension'):
        _select(src, ext='.nd2')


def test_select_ext_includes_ome_tif_like_the_dialog(tmp_path, supported):
    src = _touch(tmp_path / 'src', 'a.tif', 'b.ome.tif', 'c.czi')
    assert _select(src, ext='.tif').files == ('a.tif', 'b.ome.tif')
    assert _select(src, ext='.ome.tif').files == ('b.ome.tif',)


def test_select_requested_files(tmp_path, supported):
    src = _touch(tmp_path / 'src', 'a.czi', 'b.czi', 'c.nd2')
    selection = _select(src, files=['c.nd2', 'a.czi', 'a.czi'])
    assert selection.files == ('a.czi', 'c.nd2')
    assert selection.ignored == ('b.czi',)
    with pytest.raises(errors.SourceError, match='not found') as info:
        _select(src, files=['a.czi', 'zz.czi'])
    assert 'zz.czi' in info.value.message
    with pytest.raises(errors.SourceError, match='extension'):
        _select(src, files=['a.czi', 'c.nd2'], ext='.czi')
    assert _select(src, files=['a.czi'], ext='.czi').files == ('a.czi',)


def test_select_single_multi_pos_needs_one_file(tmp_path, supported):
    src = _touch(tmp_path / 'src', 'a.czi', 'b.czi')
    with pytest.raises(errors.SourceError) as info:
        _select(src, RawLayout.SINGLE_FILE_MULTI_POS)
    assert '--files' in info.value.hint
    selection = _select(src, RawLayout.SINGLE_FILE_MULTI_POS, files=['b.czi'])
    assert selection.files == ('b.czi',)
    assert selection.ignored == ('a.czi',)


def test_select_unsupported(tmp_path, supported):
    src = _touch(tmp_path / 'src', 'a.xyz', 'b.xyz')
    with pytest.raises(errors.SourceError, match='not supported') as info:
        _select(src)
    assert 'a.xyz' in info.value.message


def test_select_invalid_names(tmp_path, supported):
    src = _touch(tmp_path / 'src', 'a b.tif', 'c!d.tif', 'ok.tif')
    with pytest.raises(errors.SourceError, match='invalid characters') as info:
        _select(src)
    assert 'c!d.tif' in info.value.message
    assert 'a b.tif' not in info.value.message
    assert '--rename-invalid' in info.value.hint

    # Like the dialog, every selected file is renamed (also 'a b.tif')
    selection = _select(src, rename_invalid='_')
    assert selection.files == ('a b.tif', 'c!d.tif', 'ok.tif')
    assert selection.renames == {'a b.tif': 'a_b.tif', 'c!d.tif': 'c_d.tif'}
    selection = _select(src, rename_invalid='-')
    assert selection.renames == {'a b.tif': 'a-b.tif', 'c!d.tif': 'c-d.tif'}


def test_select_valid_names_are_not_renamed(tmp_path, supported):
    src = _touch(tmp_path / 'src', 'a b.tif', 'ok.tif')
    assert _select(src, rename_invalid='_').renames == {}


def test_select_rename_collisions(tmp_path, supported):
    src = _touch(tmp_path / 'src', 'a!.tif', 'a_.tif')
    with pytest.raises(errors.SourceError, match='overwrite'):
        _select(src, rename_invalid='_')
    src = _touch(tmp_path / 'src2', 'a!.tif', 'a#.tif')
    with pytest.raises(errors.SourceError, match='would both be named'):
        _select(src, rename_invalid='_')
    # The target exists although it is not selected
    src = _touch(tmp_path / 'src3', 'x!.tif', 'x_.tif')
    with pytest.raises(errors.SourceError, match='overwrite'):
        _select(src, files=['x!.tif'], rename_invalid='_')


def test_rename_collisions_ignore_unchanged_case_variants(tmp_path):
    """Names that differ only in case coexist on a case-sensitive file
    system (Linux): they collide only when one of them comes from a rename,
    and then the new name collides with each of them, whatever the order of
    the files (the files need not exist)."""
    src = str(tmp_path)
    sources._check_rename_collisions(
        src, ['Pos1.czi', 'pos1.czi', 'x+y.czi'], {'x+y.czi': 'x_y.czi'}
    )
    cases = {
        ('Pos1.czi', 'pos1.czi', 'p+1.czi'): [
            'Pos1.czi and p+1.czi would be named Pos1.czi and POS1.czi, '
            'which differ only in case',
            'pos1.czi and p+1.czi would be named pos1.czi and POS1.czi, '
            'which differ only in case',
        ],
        ('Pos1.czi', 'p+1.czi', 'pos1.czi'): [
            'Pos1.czi and p+1.czi would be named Pos1.czi and POS1.czi, '
            'which differ only in case',
            'p+1.czi and pos1.czi would be named POS1.czi and pos1.czi, '
            'which differ only in case',
        ],
        ('p+1.czi', 'Pos1.czi', 'pos1.czi'): [
            'p+1.czi and Pos1.czi would be named POS1.czi and Pos1.czi, '
            'which differ only in case',
            'p+1.czi and pos1.czi would be named POS1.czi and pos1.czi, '
            'which differ only in case',
        ],
    }
    for selected, expected in cases.items():
        with pytest.raises(errors.SourceError) as info:
            sources._check_rename_collisions(
                src, selected, {'p+1.czi': 'POS1.czi'}
            )
        assert _collisions(info.value) == expected, selected


# ------------------------------------------------------- one per channel
def _random_stems(n, seed=7):
    rng = random.Random(seed)
    alphabet = 'ab_0123_9x-._'
    out = [
        'ASY015_1_GFP', 'exp_01_mNeon-Green', 'img_1_mNeon_2', 'a_1_ch_2_x',
        'nopattern', '1_GFP', '12', '_1_', 'a1_b', 'x_10_', 'p_007_RFP.ome',
    ]
    while len(out) < n:
        out.append(''.join(rng.choice(alphabet) for _ in range(12)))
    return out


def test_split_position_channel_parity_with_findalliter():
    for stem in _random_stems(500):
        m_iter = myutils.findalliter(r'(\d+)_(.+)', stem)
        expected = None if len(m_iter) <= 1 else m_iter[-2][0]
        assert sources.split_position_channel(stem) == expected, stem


@pytest.mark.parametrize('stem, expected', [
    ('ASY015_1_GFP', ('1', 'GFP')),
    ('exp_01_mNeon-Green', ('01', 'mNeon-Green')),
    ('img_1_mNeon_2', ('1', 'mNeon_2')),
    ('nopattern', None),
])
def test_split_position_channel(stem, expected):
    assert sources.split_position_channel(stem) == expected


def test_parse_basic():
    files = [
        'ASY015_1_GFP.tif', 'ASY015_1_mNeon.tif', 'ASY015_2_GFP.tif',
        'ASY015_2_mNeon.tif'
    ]
    index = sources.parse_one_per_channel_filenames(files)
    assert index.basename == 'ASY015_'
    assert index.position_ids == (1, 2)
    assert index.position_tokens == {1: '1', 2: '2'}
    assert index.channel_tokens == ('GFP', 'mNeon')
    assert index.files[(2, 'mNeon')] == 'ASY015_2_mNeon.tif'
    assert index.missing() == []


def test_parse_ome_tif_and_zero_padding():
    files = [
        'exp_01_GFP.ome.tif', 'exp_01_RFP.ome.tif', 'exp_10_GFP.ome.tif',
        'exp_10_RFP.OME.TIF'
    ]
    index = sources.parse_one_per_channel_filenames(files)
    assert index.basename == 'exp_'
    assert index.position_ids == (1, 10)
    assert index.position_tokens == {1: '01', 10: '10'}
    assert index.channel_tokens == ('GFP', 'RFP')
    assert index.files[(10, 'RFP')] == 'exp_10_RFP.OME.TIF'


def test_parse_exact_tokens_and_natsorted_order():
    files = [
        'a_1_GFP2.tif', 'a_1_GFP.tif', 'a_1_GFP-2.tif', 'a_1_ch10.tif',
        'a_1_ch9.tif', 'a_2_GFP.tif'
    ]
    index = sources.parse_one_per_channel_filenames(files)
    assert index.channel_tokens == ('GFP', 'GFP2', 'GFP-2', 'ch9', 'ch10')
    assert index.channel_tokens == tuple(natsorted(index.channel_tokens))
    assert index.files[(1, 'GFP')] == 'a_1_GFP.tif'
    assert index.files[(1, 'GFP2')] == 'a_1_GFP2.tif'
    assert index.missing() == [
        (2, 'GFP2'), (2, 'GFP-2'), (2, 'ch9'), (2, 'ch10')
    ]
    assert index.missing(['GFP']) == []


def test_parse_errors():
    with pytest.raises(errors.FilenamePatternError) as info:
        sources.parse_one_per_channel_filenames(['a_1_GFP.tif', 'notes.tif'])
    assert info.value.files == ['notes.tif']
    with pytest.raises(errors.FilenamePatternError, match='same position'):
        sources.parse_one_per_channel_filenames(
            ['a_1_GFP.tif', 'a_01_GFP.tif']
        )
    with pytest.raises(errors.FilenamePatternError, match='basename'):
        sources.parse_one_per_channel_filenames(['1_GFP.tif', '2_GFP.tif'])
    with pytest.raises(errors.FilenamePatternError):
        sources.parse_one_per_channel_filenames([])
    assert issubclass(errors.FilenamePatternError, errors.SourceError)


def test_index_dict_round_trip():
    index = sources.parse_one_per_channel_filenames(
        ['exp_01_GFP.tif', 'exp_02_GFP.tif', 'exp_02_RFP.tif']
    )
    data = json.loads(json.dumps(index.to_dict()))
    assert sources.OnePerChannelIndex.from_dict(data) == index


# -------------------------------------------------------- metadata units
def test_metadata_units():
    units = sources.metadata_units(RawLayout(0), ['a.czi'], None)
    assert units == [sources.MetadataUnit(0, 'a.czi', 'a.czi', ('a.czi',),
                                          None)]
    with pytest.raises(errors.SourceError):
        sources.metadata_units(RawLayout(0), ['a.czi', 'b.czi'], None)

    units = sources.metadata_units(RawLayout(1), ['a.tif', 'b.tif'], None)
    assert [(u.index, u.label, u.metadata_file, u.files, u.file_ordinal)
            for u in units] == [
        (0, 'a.tif', 'a.tif', ('a.tif',), 1),
        (1, 'b.tif', 'b.tif', ('b.tif',), 2),
    ]

    files = ['s_2_GFP.tif', 's_10_GFP.tif', 's_1_GFP.tif']
    (unit,) = sources.metadata_units(RawLayout(2), files, None)
    assert unit.label == 's_'
    assert unit.metadata_file == 's_1_GFP.tif'
    assert unit.files == tuple(files)
    assert unit.file_ordinal is None
    index = sources.parse_one_per_channel_filenames(files)
    assert sources.metadata_units(RawLayout(2), files, index) == [unit]


def test_sources_never_change_the_tree(tmp_path, supported):
    src = _touch(tmp_path / 'src', 'a b!.tif', 'k.czi')
    shutil.copytree(src, str(tmp_path / 'copy'))
    before = conditions.tree_snapshot(src)
    sources.select_source_files(
        src, RawLayout.ONE_FILE_PER_POS, ext='.tif', rename_invalid='_'
    )
    sources.list_source_files(src)
    assert conditions.tree_snapshot(src) == before


# ------------------------------- io.rename_files_replace_invalid_chars
def _contents(folder):
    """{file name: content} (`_touch` writes the original name)."""
    contents = {}
    for name in os.listdir(folder):
        with open(os.path.join(folder, name), 'rb') as file:
            contents[name] = file.read()
    return contents


@pytest.mark.parametrize('replacement', ['_', '-'])
def test_io_rename_files_replace_invalid_chars_plans_first(
        tmp_path, replacement):
    """Main's public helper (what the setup dialog's rename buttons called)
    gives main's new names, but plans with plan_invalid_name_renames: when
    a new name is taken nothing is renamed (main: FileExistsError on
    Windows, the file silently replaced on Linux/macOS)."""
    names = list(INVALID_NAMES)
    oracle = _touch(tmp_path / 'oracle', *names)
    expected = legacy_rename_files_replace_invalid_chars(
        names, oracle, replacement_char=replacement
    )
    folder = _touch(tmp_path / 'copy', *names)
    renamed = io.rename_files_replace_invalid_chars(
        names, folder, replacement_char=replacement
    )
    assert renamed == expected
    assert _contents(folder) == _contents(oracle)

    taken = f'a{replacement}b.tif'
    src = _touch(tmp_path / 'src', 'a b.tif', taken, 'x+y.tif')
    before = conditions.tree_snapshot(src)
    with pytest.raises(errors.SourceError, match=f'{taken} already exists'):
        io.rename_files_replace_invalid_chars(
            ['a b.tif', taken, 'x+y.tif'], src, replacement_char=replacement
        )
    assert conditions.tree_snapshot(src) == before
