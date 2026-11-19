"""End to end with real bioio + bioio-tifffile.

Conversions of synthetic files in a folder whose path has spaces, a comma
and non-ASCII characters: layout 0 (`.ome.tif`, 3 scenes, also as
symlinks), layout 1 (ImageJ TIFFs, h5) and layout 2 (zero-padded
`.ome.tif` names). Every Position is opened again with
`load.loadData` -> `loadImgData` -> `loadOtherFiles(load_metadata=True)`
(what the GUI modules do) and checked with the decoded pixel origins and the
metadata.csv values. The raw metadata read in a worker child equals the
in-process read.
"""
from __future__ import annotations

import os

import pytest

from cellacdc.acdc_datastruct import backends, convert, sources

from . import conditions, synthetic

pytestmark = conditions.requires_bioio

ODD_FOLDER = 'raw dàta, µ x'


def convert_folder(src, dst, layout, files, *, to_h5=False, **kwargs):
    from cellacdc.acdc_datastruct import metadata
    request = convert.ConvertRequest(
        src_dir=str(src), dst_dir=str(dst), layout=layout,
        files=tuple(files), **kwargs
    )
    confirm = convert.confirm_from_metadata(
        metadata.MetadataOverrides(to_h5=to_h5)
    )
    result = convert.convert(
        request, callbacks=convert.ConvertCallbacks(confirm=confirm),
        backend=backends.InProcessBackend()
    )
    assert result.ok
    return result


def load_position(written: convert.WrittenPosition, channel: str):
    """A Position as the GUI modules open it."""
    from cellacdc import load
    suffixes = (f'_{channel}.tif', f'_{channel}.h5', '_symlink.ini')
    image_file = next(
        path for path in written.files if path.endswith(suffixes)
    )
    pos_data = load.loadData(image_file, channel)
    pos_data.getBasenameAndChNames()
    pos_data.buildPaths()
    pos_data.loadImgData()
    pos_data.loadOtherFiles(load_segm_data=False, load_metadata=True)
    return pos_data


def origins(pos_data) -> list[tuple[int, int, int, int]]:
    """Decoded (s, t, c, z) of every plane of the loaded image data (an
    h5 file is only opened by `loadImgData`: read and closed here)."""
    data = pos_data.dset[()]
    h5f = getattr(pos_data, 'h5f', None)
    if h5f is not None:
        h5f.close()
    planes = data.reshape(-1, *data.shape[-2:])
    return [synthetic.decode_pixel(plane[0, 0]) for plane in planes]


@pytest.mark.parametrize('use_symlinks', [False, True])
def test_layout0_ome_tif_scenes(tmp_path, use_symlinks):
    src = tmp_path / ODD_FOLDER / 'src'
    dst = tmp_path / ODD_FOLDER / 'exp'
    src.mkdir(parents=True)
    synthetic.make_ome_tif(str(src / 'exp.ome.tif'), series=3)
    result = convert_folder(
        src, dst, sources.RawLayout.SINGLE_FILE_MULTI_POS, ['exp.ome.tif'],
        use_symlinks=use_symlinks
    )
    assert [w.output_number for w in result.written] == [1, 2, 3]
    for scene, written in enumerate(result.written):
        pos_data = load_position(written, 'mCherry')
        assert pos_data.img_data.shape == (3, 2, 16, 20)
        assert origins(pos_data) == [
            (scene, t, 1, z) for t in range(3) for z in range(2)
        ]
        assert (pos_data.SizeT, pos_data.SizeZ) == (3, 2)
        assert (pos_data.PhysicalSizeX, pos_data.PhysicalSizeZ) == (0.1, 0.5)
        assert pos_data.TimeIncrement == 7.5
        assert pos_data.chNames == ['GFP', 'mCherry']


def test_layout1_imagej_h5(tmp_path):
    src = tmp_path / ODD_FOLDER / 'src'
    dst = tmp_path / ODD_FOLDER / 'exp'
    files = synthetic.make_one_per_pos_set(str(src), filenames=(
        'pos 1.tif', 'pos 2.tif'
    ))
    result = convert_folder(
        src, dst, sources.RawLayout.ONE_FILE_PER_POS, files, to_h5=True
    )
    for index, written in enumerate(result.written):
        number = index + 1
        # bioio-tifffile names ImageJ channels 'Channel:0:<c>'
        assert os.path.basename(written.files[-1]) == (
            f'pos {number}_s{number}_Channel_0_1.h5'
        )
        pos_data = load_position(written, 'Channel_0_1')
        assert origins(pos_data) == [
            (index, t, 1, z) for t in range(3) for z in range(2)
        ]
        # BioIO files without pixel sizes: the default 1.0
        assert pos_data.PhysicalSizeX == 1.0
        assert pos_data.SizeT == 3


def test_layout2_zero_padded_ome_tif(tmp_path):
    src = tmp_path / ODD_FOLDER / 'src'
    dst = tmp_path / ODD_FOLDER / 'exp'
    files = synthetic.make_one_per_channel_set(
        str(src), basename='exp_', positions=(1, 2),
        channels=('GFP', 'mNeon-Green'), pad=2, ext='.ome.tif'
    )
    result = convert_folder(
        src, dst, sources.RawLayout.ONE_FILE_PER_CHANNEL, sorted(files)
    )
    assert [w.output_number for w in result.written] == [1, 2]
    for written in result.written:
        assert os.path.basename(written.files[2]) == (
            f'exp__s{written.output_number}_GFP.tif'
        )
        pos_data = load_position(written, 'mNeon_Green')
        assert origins(pos_data) == [
            (written.output_number, t, 1, z)
            for t in range(3) for z in range(2)
        ]
        assert pos_data.chNames == ['GFP', 'mNeon_Green']


@conditions.subprocess_heavy
def test_child_reads_the_same_metadata(tmp_path):
    src = tmp_path / ODD_FOLDER
    src.mkdir()
    synthetic.make_ome_tif(str(src / 'a.ome.tif'), series=2)
    synthetic.make_imagej_tif(str(src / 'b.tif'), T=4)
    synthetic.make_one_per_channel_set(
        str(src), basename='c_', positions=(1,), channels=('GFP',),
        pad=2, ext='.ome.tif'
    )
    paths = [
        str(src / name) for name in ('a.ome.tif', 'b.tif', 'c_01_GFP.ome.tif')
    ]
    in_child = backends.SubprocessBackend().read_metadata(paths)
    in_process = backends.InProcessBackend().read_metadata(paths)
    assert in_child == in_process
    assert [raw.size_s for raw in in_child] == [2, 1, 1]
    assert in_child[0].scene_names == ('Scene #1', 'Scene #2')
