"""`acdc-data convert` end to end with real bioio + bioio-tifffile through
the real worker children.

A synthetic ImageJ TIFF (T=4, C=2, Z=2) converted with
`--time-range 2:3`: the written frames are the 2nd and 3rd of the file, in
order (decoded pixel origins), metadata.csv has SizeT 2, and the
`--dry-run` JSON plan equals the executed positions (apart from the written
files). `--no-lazy-load` keeps zarr out of the required reader packages.
"""
from __future__ import annotations

import json
import os

import tifffile

from cellacdc import data_cli
from cellacdc.acdc_datastruct import writers

from . import conditions, synthetic

pytestmark = [conditions.requires_bioio, conditions.subprocess_heavy]


def test_time_range_through_the_real_runner(tmp_path, capsys):
    src, dst = tmp_path / 'raw files', tmp_path / 'exp'
    src.mkdir()
    synthetic.make_imagej_tif(str(src / 'pos1.tif'), T=4, C=2, Z=2)
    argv = [
        'convert', '-i', str(src), '-o', str(dst), '--layout', 'one-per-pos',
        '--time-range', '2:3', '--no-lazy-load', '--format', 'json',
        '--no-log-file',
    ]
    assert data_cli.main(argv + ['--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not dst.exists()
    assert data_cli.main(argv) == 0
    result = json.loads(capsys.readouterr().out)

    def without_files(positions):
        return [
            {key: value for key, value in position.items() if key != 'files'}
            for position in positions
        ]

    assert without_files(result['positions']) == without_files(
        plan['positions']
    )
    position = result['positions'][0]
    assert position['time_range'] == [2, 3]
    images = position['images_path']
    tifs = sorted(name for name in os.listdir(images) if name.endswith('.tif'))
    assert len(tifs) == 2
    for c, name in enumerate(tifs):
        array = tifffile.imread(os.path.join(images, name))
        planes = array.reshape(-1, *array.shape[-2:])
        origins = [synthetic.decode_pixel(plane[0, 0]) for plane in planes]
        assert origins == [(0, 1, c, 0), (0, 1, c, 1), (0, 2, c, 0),
                           (0, 2, c, 1)]
    values = writers.read_metadata_csv(
        os.path.join(images, 'pos1_s1_metadata.csv')
    )
    assert values['SizeT'] == '2'
