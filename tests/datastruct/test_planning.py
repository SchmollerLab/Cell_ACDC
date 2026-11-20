"""acdc_datastruct.planning: numbering, selection, one-per-channel files,
other files, conflicts, source mutations and PositionJob JSON."""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import random
import re

import pytest

from cellacdc.acdc_datastruct import errors, metadata, planning, sources

from . import conditions

RawLayout = sources.RawLayout
OnExisting = planning.OnExisting

LAYOUT_FILES = {
    RawLayout.SINGLE_FILE_MULTI_POS: ['exp.czi'],
    RawLayout.ONE_FILE_PER_POS: ['a.tif', 'b.tif', 'c.tif'],
    RawLayout.ONE_FILE_PER_CHANNEL: [
        's_1_GFP.tif', 's_1_RFP.tif', 's_2_GFP.tif', 's_2_RFP.tif',
        's_3_GFP.tif', 's_3_RFP.tif',
    ],
}


def _touch(folder, *names):
    os.makedirs(folder, exist_ok=True)
    for name in names:
        with open(os.path.join(folder, name), 'w') as file:
            file.write(name)


def _make_existing(dst, numbers, empty=()):
    for n in numbers:
        _touch(os.path.join(dst, f'Position_{n}', 'Images'), 'old.tif')
    for n in empty:
        os.makedirs(os.path.join(dst, f'Position_{n}', 'Images'))


def _ctx(tmp_path, layout, files, *, on_existing=OnExisting.ADD,
         existing=(), move_raw=False, other_files='leave', extra_src=(),
         use_symlinks=False, renames=None, start_pos_n=None):
    src = str(tmp_path / 'src')
    dst = str(tmp_path / 'exp')
    _touch(src, *files, *extra_src)
    _make_existing(dst, existing)
    existing_numbers = planning.existing_position_numbers(dst)
    if start_pos_n is None:
        start_pos_n = planning.start_position_number(
            on_existing, existing_numbers
        )
    opc = None
    if layout == RawLayout.ONE_FILE_PER_CHANNEL:
        opc = sources.parse_one_per_channel_filenames(files)
    return planning.PlanContext(
        layout=layout, src_dir=src, dst_dir=dst, files=tuple(files),
        on_existing=on_existing, existing=tuple(existing_numbers),
        start_pos_n=start_pos_n, use_symlinks=use_symlinks, lazy_load=True,
        other_files=other_files, one_per_channel=opc,
        mutations=planning.plan_source_mutations(
            src, files, dst_dir=dst, move_raw=move_raw, renames=renames
        ),
        source_listing=tuple(sources.list_source_files(src)),
    )


def _raw(path='x', **changes):
    values = dict(
        path=path, source='bioio', reader='r', lens_na=1.4, size_s=3,
        size_z=4, size_t=6, time_increment=1.0, time_increment_unit='s',
        size_c=2, physical_size_x=1.0, physical_size_y=1.0,
        physical_size_z=1.0, physical_size_unit='um', image_name='',
        channel_names=('GFP', 'RFP'), em_wavelens=(1.0, 2.0),
    )
    values.update(changes)
    return metadata.RawFileMetadata(**values)


def _settings(**changes):
    values = dict(
        to_h5=False, lens_na=1.4, size_t=6, size_z=4, size_c=2, size_s=3,
        time_increment_s=1.0, physical_size_x_um=0.1, physical_size_y_um=0.2,
        physical_size_z_um=0.3, channel_names=('GFP', 'RFP'),
        save_channels=(True, True), em_wavelens=(510.0, 610.0),
    )
    values.update(changes)
    return metadata.ConversionSettings(**values)


def _plan_all(ctx, settings, raw=None):
    """Plan every unit of ctx (layout 1: one unit per file)."""
    jobs, skipped = [], []
    units = sources.metadata_units(ctx.layout, ctx.files, ctx.one_per_channel)
    for unit in units:
        unit_jobs, unit_skipped = planning.plan_unit_positions(
            ctx, unit, raw or _raw(), settings, xml_ref=f'xml:{unit.index}'
        )
        jobs.extend(unit_jobs)
        skipped.extend(unit_skipped)
    return jobs, skipped


# --------------------------------------------------------------- basics
def test_on_existing_from_gui_flags():
    flags = dict(overwrite=False, add_files=False, create_new=False)
    assert OnExisting.from_gui_flags(**flags) is OnExisting.ADD
    assert OnExisting.from_gui_flags(
        **{**flags, 'add_files': True}) is OnExisting.ADD
    assert OnExisting.from_gui_flags(
        **{**flags, 'overwrite': True}) is OnExisting.OVERWRITE
    assert OnExisting.from_gui_flags(
        **{**flags, 'create_new': True}) is OnExisting.CREATE_NEW
    assert OnExisting('create-new') is OnExisting.CREATE_NEW
    assert OnExisting.ERROR == 'error'


def test_existing_position_numbers_and_start(tmp_path):
    dst = str(tmp_path / 'exp')
    assert planning.existing_position_numbers(dst) == []
    _make_existing(dst, [10, 2], empty=[3])
    _touch(dst, 'Position_7')
    os.makedirs(os.path.join(dst, 'Position_x', 'Images'))
    assert planning.existing_position_numbers(dst) == [2, 10]
    assert planning.start_position_number(OnExisting.CREATE_NEW, [2, 10]) == 11
    assert planning.start_position_number(OnExisting.CREATE_NEW, []) == 1
    for mode in (OnExisting.ERROR, OnExisting.ADD, OnExisting.OVERWRITE):
        assert planning.start_position_number(mode, [2, 10]) == 1


def test_num_positions_and_csv_spec():
    settings = _settings(size_s=5, time_range=(2, 4))
    index = sources.parse_one_per_channel_filenames(
        LAYOUT_FILES[RawLayout.ONE_FILE_PER_CHANNEL]
    )
    assert planning.num_positions(
        RawLayout(0), settings, num_files=1, one_per_channel=None) == 5
    assert planning.num_positions(
        RawLayout(1), settings, num_files=7, one_per_channel=None) == 7
    assert planning.num_positions(
        RawLayout(2), settings, num_files=6, one_per_channel=index) == 3
    spec = planning.metadata_csv_spec(settings, basename='b_s1_',
                                      saved_size_t=3)
    assert spec == planning.MetadataCsvSpec(
        lens_na=1.4, saved_size_t=3, size_z=4, time_increment_s=1.0,
        physical_size_z_um=0.3, physical_size_y_um=0.2,
        physical_size_x_um=0.1, basename='b_s1_',
        channel_names=('GFP', 'RFP'), save_channels=(True, True),
        em_wavelens=(510.0, 610.0),
    )
    assert planning.MetadataCsvSpec.from_dict(
        json.loads(json.dumps(spec.to_dict()))
    ) == spec


# ------------------------------------------------------ numbering matrix
@pytest.mark.parametrize('layout', list(RawLayout))
@pytest.mark.parametrize('on_existing', list(OnExisting))
@pytest.mark.parametrize('subset', [None, frozenset({1, 3})])
def test_plan_matrix(tmp_path, layout, on_existing, subset):
    files = LAYOUT_FILES[layout]
    ctx = _ctx(tmp_path, layout, files, on_existing=on_existing,
               existing=(1, 2))
    settings = _settings(selected_positions=subset)
    jobs, skipped = _plan_all(ctx, settings)

    ids = [1, 2, 3] if subset is None else [1, 3]
    assert [job.input_id for job in jobs] == ids
    assert [s.input_id for s in skipped] == ([] if subset is None else [2])
    assert all(s.size_c == 2 for s in skipped)
    offset = 2 if on_existing == OnExisting.CREATE_NEW else 0
    assert [job.output_number for job in jobs] == [i + offset for i in ids]
    for job in jobs:
        assert job.num_positions == 3
        assert job.pos_str == str(job.output_number)
        assert job.images_path == os.path.join(
            ctx.dst_dir, f'Position_{job.output_number}', 'Images'
        )
        assert job.clear_images == (on_existing == OnExisting.OVERWRITE)
        assert job.series == (
            job.input_id - 1 if layout == RawLayout.SINGLE_FILE_MULTI_POS
            else 0
        )
        assert job.time_range == (1, 6)
        assert job.metadata_csv.saved_size_t == 6
        assert job.metadata_csv.basename == job.basename
        assert job.output == 'tif'
        assert [c.name for c in job.channels] == ['GFP', 'RFP']
        assert [c.csv_index for c in job.channels] == [0, 1]

    conflicts = planning.find_conflicts(jobs, ctx.existing, on_existing)
    if on_existing == OnExisting.ERROR:
        assert conflicts == [1, 2] if subset is None else [1]
        with pytest.raises(errors.ExistingPositionsError) as info:
            planning.check_conflicts(jobs, ctx.existing, on_existing)
        assert info.value.existing == conflicts
        assert info.value.choices == ('overwrite', 'add', 'create-new')
        assert 'Position_1' in info.value.message
        assert '--on-existing' in info.value.hint
        assert isinstance(info.value, FileExistsError)
    else:
        assert conflicts == []
        planning.check_conflicts(jobs, ctx.existing, on_existing)


def test_layout_specific_names_and_sources(tmp_path):
    ctx = _ctx(tmp_path / '0', RawLayout(0), ['exp.ome.tif'])
    jobs, _ = _plan_all(ctx, _settings(size_s=2))
    assert [job.basename for job in jobs] == ['exp_ome_s1_', 'exp_ome_s2_']
    assert jobs[0].filename_no_ext == 'exp.ome'
    assert jobs[0].source_file == 'exp.ome.tif'
    assert [(c.raw_path, c.file_channel_index) for c in jobs[1].channels] == [
        (os.path.join(ctx.src_dir, 'exp.ome.tif'), 0),
        (os.path.join(ctx.src_dir, 'exp.ome.tif'), 1),
    ]

    ctx = _ctx(tmp_path / '1', RawLayout(1), ['a.tif', 'b.tif'])
    jobs, _ = _plan_all(ctx, _settings(save_channels=(False, True)))
    assert [job.basename for job in jobs] == ['a_s1_', 'b_s2_']
    assert [job.source_file for job in jobs] == ['a.tif', 'b.tif']
    assert [[c.name for c in job.channels] for job in jobs] == [['RFP']] * 2
    assert jobs[1].channels[0].file_channel_index == 1
    assert jobs[1].unit_index == 1 and jobs[1].metadata_xml_ref == 'xml:1'

    files = LAYOUT_FILES[RawLayout(2)]
    ctx = _ctx(tmp_path / '2', RawLayout(2), files)
    jobs, _ = _plan_all(ctx, _settings(channel_names=('green', 'red')))
    assert jobs[1].basename == 's__s2_'
    assert jobs[1].filename_no_ext == 's_'
    assert jobs[1].source_file == 's_2_GFP.tif'
    assert [(c.name, c.raw_path, c.file_channel_index)
            for c in jobs[1].channels] == [
        ('green', os.path.join(ctx.src_dir, 's_2_GFP.tif'), 0),
        ('red', os.path.join(ctx.src_dir, 's_2_RFP.tif'), 0),
    ]
    assert all(job.size_z is None for job in jobs)


def test_padding_follows_the_number_of_positions(tmp_path):
    files = [f'f{i}.tif' for i in range(1, 13)]
    ctx = _ctx(tmp_path, RawLayout(1), files, on_existing=OnExisting.CREATE_NEW,
               existing=(1, 2))
    jobs, _ = _plan_all(ctx, _settings())
    assert [job.output_number for job in jobs] == list(range(3, 15))
    assert [job.pos_str for job in jobs][:2] == ['03', '04']
    assert jobs[-1].pos_str == '14'
    assert jobs[0].basename == 'f1_s03_'


def test_layout_2_offset_uses_lowest_position(tmp_path):
    files = ['p_5_GFP.tif', 'p_7_GFP.tif']
    ctx = _ctx(tmp_path, RawLayout(2), files, on_existing=OnExisting.CREATE_NEW,
               existing=(1, 2, 3))
    jobs, _ = _plan_all(ctx, _settings(size_c=1, channel_names=('GFP',),
                                       save_channels=(True,),
                                       em_wavelens=(500.0,)))
    assert [(j.input_id, j.output_number) for j in jobs] == [(5, 4), (7, 6)]
    ctx = _ctx(tmp_path / 'add', RawLayout(2), files, existing=(1,))
    jobs, _ = _plan_all(ctx, _settings(size_c=1, channel_names=('GFP',),
                                       save_channels=(True,),
                                       em_wavelens=(500.0,)))
    assert [j.output_number for j in jobs] == [5, 7]
    assert [j.pos_str for j in jobs] == ['5', '7']


@pytest.mark.parametrize('layout, selected', [
    (RawLayout(0), {4}), (RawLayout(1), {1, 4}), (RawLayout(2), {0, 2}),
])
def test_unknown_positions(tmp_path, layout, selected):
    ctx = _ctx(tmp_path, layout, LAYOUT_FILES[layout])
    with pytest.raises(errors.PlanError, match='Unknown positions') as info:
        _plan_all(ctx, _settings(selected_positions=frozenset(selected)))
    assert 'available: 1-3' in info.value.message


def test_layout_1_selection_is_by_file_ordinal(tmp_path):
    ctx = _ctx(tmp_path, RawLayout(1), ['a.tif', 'b.tif', 'c.tif'])
    jobs, skipped = _plan_all(ctx, _settings(selected_positions={2}))
    assert [(j.input_id, j.source_file) for j in jobs] == [(2, 'b.tif')]
    assert [s.input_id for s in skipped] == [1, 3]


# ---------------------------------------------------------- frames and Z
def test_time_range_and_size_z(tmp_path):
    ctx = _ctx(tmp_path, RawLayout(1), ['a.tif'])
    unit = sources.metadata_units(ctx.layout, ctx.files, None)[0]

    def plan(settings, raw):
        return planning.plan_unit_positions(
            ctx, unit, raw, settings, xml_ref='x'
        )[0][0]

    job = plan(_settings(size_t=99), _raw(size_t=6, size_z=4))
    assert (job.time_range, job.metadata_csv.saved_size_t) == ((1, 6), 6)
    assert job.size_z == 4 and job.metadata_csv.size_z == 4
    job = plan(_settings(size_t=5, size_z=2), _raw(size_t=None, size_z=None))
    assert (job.time_range, job.size_z, job.metadata_csv.size_z) == (
        (1, 5), 2, 2
    )
    job = plan(_settings(size_z=9), _raw(size_z=4))
    assert job.size_z == 4 and job.metadata_csv.size_z == 9
    # SizeT edited to 1 in the dialog: the stale (1, 5) range is kept
    job = plan(_settings(size_t=1, time_range=(1, 5)), _raw(size_t=5))
    assert (job.time_range, job.metadata_csv.saved_size_t) == ((1, 5), 5)
    job = plan(_settings(time_range=(2, 3)), _raw(size_t=6))
    assert (job.time_range, job.metadata_csv.saved_size_t) == ((2, 3), 2)
    with pytest.raises(errors.SettingsError, match='a.tif'):
        plan(_settings(time_range=(2, 7)), _raw(size_t=6))
    # Unknown file SizeT: the range is checked by the writer instead
    job = plan(_settings(size_t=12, time_range=(1, 12)), _raw(size_t=None))
    assert job.time_range == (1, 12)


@pytest.mark.parametrize('use_symlinks, to_h5, expected', [
    (False, False, 'tif'), (False, True, 'h5'), (True, False, 'symlink'),
    (True, True, 'symlink'),
])
def test_output_kind(tmp_path, use_symlinks, to_h5, expected):
    ctx = _ctx(tmp_path, RawLayout(1), ['a.tif'], use_symlinks=use_symlinks)
    jobs, _ = _plan_all(ctx, _settings(to_h5=to_h5))
    assert jobs[0].output == expected
    assert jobs[0].physical_size_zyx_um == (0.3, 0.2, 0.1)
    assert jobs[0].time_increment_s == 1.0 and jobs[0].lazy_load is True


def test_image_name(tmp_path):
    ctx = _ctx(tmp_path, RawLayout(0), ['exp.ome.tif'])
    raw = _raw(scene_names=('Scene #1', '', 'S 3!'))
    jobs, _ = _plan_all(ctx, _settings(add_image_name=True), raw)
    assert [j.image_name for j in jobs] == ['Scene__1', '', 'S_3']
    assert [j.basename for j in jobs] == [
        'exp_ome_Scene__1_s1_', 'exp_ome_s2_', 'exp_ome_S_3_s3_'
    ]
    jobs, _ = _plan_all(ctx, _settings(), raw)
    assert [j.image_name for j in jobs] == ['', '', '']
    ctx = _ctx(tmp_path / 'one', RawLayout(1), ['a.tif', 'b.tif'])
    jobs, _ = _plan_all(ctx, _settings(add_image_name=True), _raw())
    assert [j.image_name for j in jobs] == ['', '']


# -------------------------------------------------------- one per channel
def _settings_2(names, save=None):
    return _settings(
        size_c=len(names), channel_names=tuple(names),
        save_channels=tuple(save or [True] * len(names)),
        em_wavelens=(500.0,) * len(names)
    )


def test_layout_2_size_c_above_tokens(tmp_path):
    ctx = _ctx(tmp_path, RawLayout(2), LAYOUT_FILES[RawLayout(2)])
    with pytest.raises(errors.SettingsError) as info:
        _plan_all(ctx, _settings_2(['a', 'b', 'c']))
    assert info.value.message == (
        'One file per channel: SizeC cannot exceed the 2 channels found in '
        'the file names.'
    )
    jobs, _ = _plan_all(ctx, _settings_2(['only_gfp']))
    assert [[c.raw_path for c in j.channels] for j in jobs][0] == [
        os.path.join(ctx.src_dir, 's_1_GFP.tif')
    ]


def test_layout_2_missing_pairs_among_saved_channels(tmp_path):
    files = [
        'e_1_GFP.tif', 'e_1_mNeon-Green.tif', 'e_2_GFP.tif', 'e_3_GFP.tif'
    ]
    ctx = _ctx(tmp_path, RawLayout(2), files)
    with pytest.raises(errors.MissingChannelFilesError) as info:
        _plan_all(ctx, _settings_2(['GFP', 'mNeon_Green']))
    assert info.value.missing == [(2, 'mNeon-Green'), (3, 'mNeon-Green')]
    assert isinstance(info.value, errors.SourceError)
    jobs, _ = _plan_all(
        ctx, _settings_2(['GFP', 'mNeon_Green'], save=[True, False])
    )
    assert len(jobs) == 3
    # Output names differ from the raw tokens; lookup by exact token (GV-08)
    ctx = _ctx(tmp_path / 'b', RawLayout(2), files[:2])
    jobs, _ = _plan_all(ctx, _settings_2(['g', 'mNeon_Green']))
    assert [(c.name, os.path.basename(c.raw_path))
            for c in jobs[0].channels] == [
        ('g', 'e_1_GFP.tif'), ('mNeon_Green', 'e_1_mNeon-Green.tif')
    ]


def test_layout_2_other_files(tmp_path):
    files = ['s_01_GFP.tif', 's_01_RFP.tif', 's_2_GFP.tif', 's_2_RFP.tif']
    extra = [
        's_1_notes.txt', 's_01_extra.txt', 's_10_x.txt', 's_2_log.txt',
        'xs_1_bad.txt', 's_1_GFP.txt', 'notes.txt',
    ]
    for mode in ('leave', 'copy', 'move'):
        ctx = _ctx(tmp_path / mode, RawLayout(2), files, other_files=mode,
                   extra_src=extra, move_raw=True)
        jobs, _ = _plan_all(ctx, _settings_2(['GFP', 'RFP'], [True, False]))
        if mode == 'leave':
            assert all(job.other_files == () for job in jobs)
            continue
        position_1 = jobs[0]
        assert [
            (os.path.basename(a.src), os.path.dirname(a.src),
             os.path.basename(a.dst), os.path.dirname(a.dst), a.mode)
            for a in position_1.other_files
        ] == [
            ('s_01_extra.txt', ctx.src_dir, 's__s1_extra.txt',
             position_1.images_path, mode),
            ('s_1_notes.txt', ctx.src_dir, 's__s1_notes.txt',
             position_1.images_path, mode),
        ]
        assert [os.path.basename(a.src) for a in jobs[1].other_files] == [
            's_2_log.txt'
        ]
        # raw files were moved, other files stay in the original folder
        assert position_1.channels[0].raw_path == os.path.join(
            ctx.dst_dir, 'raw_microscopy_files', 's_01_GFP.tif'
        )


# ------------------------------------------------------- source mutations
def test_plan_source_mutations_without_move(tmp_path):
    src = str(tmp_path / 'src')
    _touch(src, 'a.tif')
    m = planning.plan_source_mutations(
        src, ['a.tif'], dst_dir=str(tmp_path / 'exp'), move_raw=False
    )
    assert m.is_empty()
    assert m.final_paths == {'a.tif': os.path.join(src, 'a.tif')}


def test_plan_source_mutations_moves_with_companion(tmp_path):
    src = str(tmp_path / 'src')
    dst = str(tmp_path / 'exp')
    _touch(src, 'a.vsi', 'b.vsi')
    os.makedirs(os.path.join(src, '_a_'))
    before = conditions.tree_snapshot(src, dst)
    m = planning.plan_source_mutations(
        src, ['a.vsi', 'b.vsi'], dst_dir=dst, move_raw=True
    )
    assert conditions.tree_snapshot(src, dst) == before
    raw_dir = os.path.join(dst, 'raw_microscopy_files')
    assert m.moves == (
        planning.RawMove(
            os.path.join(src, 'a.vsi'), os.path.join(raw_dir, 'a.vsi'),
            os.path.join(src, '_a_'), os.path.join(raw_dir, '_a_')
        ),
        planning.RawMove(
            os.path.join(src, 'b.vsi'), os.path.join(raw_dir, 'b.vsi'),
            None, None
        ),
    )
    assert m.final_paths['b.vsi'] == os.path.join(raw_dir, 'b.vsi')
    assert not m.is_empty()


def test_plan_source_mutations_in_raw_microscopy_files(tmp_path):
    src = str(tmp_path / 'exp' / 'raw_microscopy_files')
    _touch(src, 'a.tif')
    for folder in (src, src + os.sep):
        m = planning.plan_source_mutations(
            folder, ['a.tif'], dst_dir=str(tmp_path / 'exp'), move_raw=True
        )
        assert m.moves == ()


def test_plan_source_mutations_collisions(tmp_path):
    src = str(tmp_path / 'src')
    dst = str(tmp_path / 'exp')
    raw_dir = os.path.join(dst, 'raw_microscopy_files')
    _touch(src, 'a.tif', 'b.vsi', 'c d.tif', 'c_d.tif', 'e f.vsi', 'g!.tif')
    os.makedirs(os.path.join(src, '_b_'))
    os.makedirs(os.path.join(src, '_e f_'))
    _touch(raw_dir, 'a.tif')
    os.makedirs(os.path.join(raw_dir, '_b_'))
    cases = [
        (dict(files=['a.tif'], move_raw=True), 'a.tif'),
        (dict(files=['b.vsi'], move_raw=True), 'companion folder of b.vsi'),
        (dict(files=['c d.tif'], renames={'c d.tif': 'c_d.tif'}),
         'cannot rename c d.tif'),
        (dict(files=['e f.vsi'], renames={'e f.vsi': 'e_f.vsi'}),
         'detach its companion folder'),
        (dict(files=['g!.tif', 'c_d.tif'], renames={'g!.tif': 'C_D.tif'}),
         'g!.tif and c_d.tif would be named C_D.tif and c_d.tif, which '
         'differ only in case'),
        (dict(files=['c_d.tif', 'g!.tif'], renames={'g!.tif': 'c_d.tif'}),
         'c_d.tif and g!.tif would both be named c_d.tif'),
    ]
    for kwargs, text in cases:
        kwargs.setdefault('move_raw', False)
        with pytest.raises(errors.PlanError) as info:
            planning.plan_source_mutations(src, dst_dir=dst, **kwargs)
        assert text in info.value.message, (kwargs, info.value.message)


def test_plan_source_mutations_rename_then_move(tmp_path):
    src = str(tmp_path / 'src')
    dst = str(tmp_path / 'exp')
    _touch(src, 'a b.tif')
    m = planning.plan_source_mutations(
        src, ['a b.tif'], dst_dir=dst, move_raw=True,
        renames={'a b.tif': 'a_b.tif'}
    )
    assert m.renames == (
        (os.path.join(src, 'a b.tif'), os.path.join(src, 'a_b.tif')),
    )
    assert m.moves[0].src == os.path.join(src, 'a_b.tif')
    assert m.final_paths == {
        'a b.tif': os.path.join(dst, 'raw_microscopy_files', 'a_b.tif')
    }


@pytest.mark.parametrize('move_raw', [False, True])
def test_plan_source_mutations_unchanged_case_variants(tmp_path, move_raw):
    """Names that differ only in case coexist on a case-sensitive file
    system (Linux), so unchanged ones do not collide, unless both files are
    moved: raw_microscopy_files may be on a file system that ignores case
    (network share, exFAT, macOS), where the second move would replace the
    first file. The files need not exist: nothing is renamed or moved."""
    src = str(tmp_path / 'src')
    dst = str(tmp_path / 'exp')
    os.makedirs(src)
    files = ['Pos1.czi', 'pos1.czi']
    if not move_raw:
        m = planning.plan_source_mutations(
            src, files, dst_dir=dst, move_raw=False
        )
        assert m.renames == ()
        assert len(set(m.final_paths.values())) == 2
        return
    raw_dir = os.path.join(dst, 'raw_microscopy_files')
    with pytest.raises(errors.PlanError) as info:
        planning.plan_source_mutations(src, files, dst_dir=dst, move_raw=True)
    assert info.value.message.splitlines()[1:] == [
        f'  - Pos1.czi and pos1.czi differ only in case and cannot both be '
        f'moved to {raw_dir} (its file system may ignore case)'
    ]
    # Not moved when the source is itself raw_microscopy_files
    m = planning.plan_source_mutations(
        raw_dir, files, dst_dir=dst, move_raw=True
    )
    assert m.moves == ()


@pytest.mark.parametrize('move_raw', [False, True])
@pytest.mark.parametrize(
    'files', list(itertools.permutations(['p+1.czi', 'Pos1.czi', 'pos1.czi']))
)
def test_plan_source_mutations_reports_every_colliding_pair(
        tmp_path, files, move_raw):
    """A new name collides with every name that differs from it only in
    case: each colliding pair is reported once, and only those, whatever
    the order of the files ('Pos1.czi' and 'pos1.czi' themselves collide
    only when they are moved)."""
    src = str(tmp_path / 'src')
    os.makedirs(src)
    with pytest.raises(errors.PlanError) as info:
        planning.plan_source_mutations(
            src, list(files), dst_dir=str(tmp_path / 'exp'),
            move_raw=move_raw, renames={'p+1.czi': 'POS1.czi'}
        )
    reported = {}
    for line in info.value.message.splitlines()[1:]:
        first, second, text = re.fullmatch(
            r'  - (\S+) and (\S+) (.+)', line
        ).groups()
        assert frozenset((first, second)) not in reported, line
        reported[frozenset((first, second))] = text
    renamed = {
        frozenset(('p+1.czi', 'Pos1.czi')): 'Pos1.czi',
        frozenset(('p+1.czi', 'pos1.czi')): 'pos1.czi',
    }
    assert set(reported) - set(renamed) == (
        {frozenset(('Pos1.czi', 'pos1.czi'))} if move_raw else set()
    )
    for pair, other in renamed.items():
        names = ' and '.join(
            'POS1.czi' if f == 'p+1.czi' else other for f in files
            if f in pair
        )
        assert reported[pair] == (
            f'would be named {names}, which differ only in case'
        )
    if move_raw:
        assert reported[frozenset(('Pos1.czi', 'pos1.czi'))].startswith(
            'differ only in case and cannot both be moved to '
        )


def test_rename_and_mutation_plans_report_the_same_collisions(tmp_path):
    """The rename buttons of the setup dialog and --rename-invalid
    (sources.plan_invalid_name_renames) report the same colliding pairs,
    in the same words, as plan_source_mutations without moves (the files
    need not exist)."""
    src = str(tmp_path / 'src')
    dst = str(tmp_path / 'exp')
    names = [
        'a b.czi', 'a_b.czi', 'A_B.czi', 'a+b.czi', 'A b.czi', 'Pos1.czi',
        'pos1.czi', 'p+1.czi', 'P+1.czi', 'p_1.czi',
    ]
    rng = random.Random(5)
    for size in (2, 3, 4):
        for files in itertools.combinations(names, size):
            files = rng.sample(files, size)
            new_names = {
                f: sources.planned_valid_filename(f, '_') for f in files
            }
            renames = {f: new for f, new in new_names.items() if new != f}
            try:
                sources.plan_invalid_name_renames(src, files, '_')
                expected = []
            except errors.SourceError as err:
                expected = err.message.splitlines()[1:]
            try:
                planning.plan_source_mutations(
                    src, files, dst_dir=dst, move_raw=False, renames=renames
                )
                reported = []
            except errors.PlanError as err:
                reported = err.message.splitlines()[1:]
            assert reported == expected, files


def test_basename_follows_the_renamed_file(tmp_path):
    ctx = _ctx(tmp_path, RawLayout(1), ['a b.tif'],
               renames={'a b.tif': 'a_b.tif'})
    jobs, _ = _plan_all(ctx, _settings())
    assert jobs[0].basename == 'a_b_s1_'
    assert jobs[0].filename_no_ext == 'a_b'
    assert jobs[0].source_file == 'a b.tif'
    assert jobs[0].channels[0].raw_path == os.path.join(ctx.src_dir,
                                                        'a_b.tif')


@pytest.mark.parametrize('other_files', ['leave', 'copy'])
def test_layout_2_basename_follows_the_renamed_files(tmp_path, other_files):
    """One file per channel with --rename-invalid: the outputs are
    named after the renamed files, as when they are renamed first; other
    files (not renamed) are still matched by their original names."""
    files = [
        'exp(1)_1_GFP.czi', 'exp(1)_1_mCh.czi', 'exp(1)_2_GFP.czi',
        'exp(1)_2_mCh.czi',
    ]
    renames = {f: sources.planned_valid_filename(f, '_') for f in files}
    ctx = _ctx(tmp_path, RawLayout(2), files, renames=renames,
               other_files=other_files, extra_src=['exp(1)_1_notes.txt'])
    jobs, _ = _plan_all(ctx, _settings_2(['GFP', 'mCh']))
    assert [job.basename for job in jobs] == ['exp_1___s1_', 'exp_1___s2_']
    assert all(job.filename_no_ext == 'exp_1__' for job in jobs)
    assert jobs[0].source_file == 'exp(1)_1_GFP.czi'
    assert jobs[0].channels[0].raw_path == os.path.join(
        ctx.src_dir, 'exp_1__1_GFP.czi'
    )
    if other_files == 'copy':
        assert [
            (os.path.basename(a.src), os.path.basename(a.dst))
            for a in jobs[0].other_files
        ] == [('exp(1)_1_notes.txt', 'exp_1___s1_notes.txt')]
    # Renamed first (what the GUI does): the same names
    renamed = tmp_path / 'renamed'
    ctx = _ctx(renamed, RawLayout(2), sorted(renames.values()))
    first = _plan_all(ctx, _settings_2(['GFP', 'mCh']))[0]
    assert [job.basename for job in first] == [job.basename for job in jobs]


@pytest.mark.parametrize('token', ['ch1+2', '405+nm'])
def test_layout_2_renames_that_break_the_pattern(tmp_path, token):
    """A rename can break the pattern of a channel token: 'exp_2_ch1_2'
    reads as position 1, channel '2' and 'exp_2_405_nm' as position 405,
    channel 'nm', so the renamed names do not parse (several files per
    position and channel). The renames leave the basename 'exp_' unchanged:
    the outputs keep it, positions and channels come from the original
    names. When the renames change the basename too, the error names the
    renames."""
    files = [f'exp_{pos}_{name}.czi' for pos in (1, 2)
             for name in ('DIC', token)]
    renames = {f: sources.planned_valid_filename(f, '_') for f in files
               if '+' in f}
    ctx = _ctx(tmp_path / 'a', RawLayout(2), files, renames=renames)
    tokens = ctx.one_per_channel.channel_tokens
    jobs, _ = _plan_all(ctx, _settings_2(['A', 'B']))
    assert [job.basename for job in jobs] == ['exp__s1_', 'exp__s2_']
    assert [job.input_id for job in jobs] == [1, 2]
    assert [
        os.path.basename(channel.raw_path) for channel in jobs[1].channels
    ] == [renames.get(f'exp_2_{t}.czi', f'exp_2_{t}.czi') for t in tokens]

    files = [f'e+{name}' for name in files]
    renames = {f: sources.planned_valid_filename(f, '_') for f in files}
    ctx = _ctx(tmp_path / 'b', RawLayout(2), files, renames=renames)
    with pytest.raises(errors.FilenamePatternError) as info:
        _plan_all(ctx, _settings_2(['A', 'B']))
    message = info.value.message
    assert message.startswith(
        'The names that --rename-invalid would give the files cannot be '
        'read as one file per channel '
        '(<basename><position number>_<channel>.<ext>), so the outputs '
        'cannot be named after them:\n'
    )
    for old, new in renames.items():
        assert f'  - {old} -> {new}\n' in message
    assert 'With the new names: Several files have the same position' in (
        message
    )
    assert info.value.files == files


def test_relative_paths_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        planning.plan_source_mutations(
            'src', ['a.tif'], dst_dir=str(tmp_path), move_raw=False
        )
    with pytest.raises(ValueError):
        planning.plan_source_mutations(
            str(tmp_path), ['a.tif'], dst_dir='exp', move_raw=False
        )
    with pytest.raises(ValueError):
        planning.ChannelSource('GFP', 0, 'a.tif', 0)
    with pytest.raises(ValueError):
        planning.OtherFileAction(str(tmp_path / 'a'), 'b', 'copy')
    with pytest.raises(ValueError):
        planning.RawMove(str(tmp_path / 'a'), str(tmp_path / 'b'), 'c', None)
    with pytest.raises(ValueError):
        planning.SourceMutations((), (), {'a.tif': 'a.tif'})
    ctx = _ctx(tmp_path, RawLayout(1), ['a.tif'])
    with pytest.raises(ValueError):
        dataclasses.replace(ctx, dst_dir='exp')
    job = _plan_all(ctx, _settings())[0][0]
    with pytest.raises(ValueError):
        dataclasses.replace(job, images_path='Position_1/Images')


def test_position_job_json_round_trip(tmp_path):
    files = ['s_1_GFP.tif', 's_1_RFP.tif']
    ctx = _ctx(tmp_path, RawLayout(2), files, other_files='copy',
               extra_src=['s_1_notes.txt'])
    jobs, _ = _plan_all(ctx, _settings(time_range=(2, 3)))
    job = jobs[0]
    assert job.other_files
    data = json.loads(json.dumps(job.to_dict()))
    assert planning.PositionJob.from_dict(data) == job
    assert data['metadata_csv']['channel_names'] == ['GFP', 'RFP']

    ctx = _ctx(tmp_path / 'b', RawLayout(1), ['a.tif'])
    job = _plan_all(ctx, _settings())[0][0]
    assert planning.PositionJob.from_dict(
        json.loads(json.dumps(job.to_dict()))
    ) == job


def test_plan_does_not_touch_the_file_system(tmp_path):
    files = LAYOUT_FILES[RawLayout(2)]
    ctx = _ctx(tmp_path, RawLayout(2), files, move_raw=True,
               other_files='move', extra_src=['s_1_x.txt'],
               on_existing=OnExisting.OVERWRITE, existing=(1,))
    before = conditions.tree_snapshot(ctx.src_dir, ctx.dst_dir)
    jobs, _ = _plan_all(ctx, _settings())
    planning.check_conflicts(jobs, ctx.existing, ctx.on_existing)
    assert conditions.tree_snapshot(ctx.src_dir, ctx.dst_dir) == before


# ------------------------------------------ contract with the pixel writer
@pytest.mark.parametrize('add_image_name', [False, True])
@pytest.mark.parametrize('layout, files', [
    (RawLayout.SINGLE_FILE_MULTI_POS, ['exp.v2.czi']),
    (RawLayout.ONE_FILE_PER_POS, ['a.1.tif', 'b.tif']),
    (RawLayout.ONE_FILE_PER_CHANNEL,
     LAYOUT_FILES[RawLayout.ONE_FILE_PER_CHANNEL]),
])
def test_basename_matches_the_pixel_writer(
        tmp_path, layout, files, add_image_name):
    """metadata.csv/XML (``job.basename``) and the image files named by
    ``_utils.getFilename`` from the job's writer fields share one prefix."""
    from cellacdc.acdc_bioio_bioformats import _utils
    ctx = _ctx(tmp_path, layout, files)
    raw = _raw(scene_names=('Scene #1', 'Scene #2', 'Scene #3'))
    jobs, _ = _plan_all(ctx, _settings(add_image_name=add_image_name), raw)
    assert jobs
    for job in jobs:
        assert bool(job.image_name) == add_image_name
        _, writer_basename = _utils.getFilename(
            job.filename_no_ext, job.pos_str, 'GFP', job.series, '.tif',
            return_basename=True, image_name=job.image_name
        )
        assert writer_basename == job.basename == job.metadata_csv.basename
