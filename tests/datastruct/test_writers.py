"""acdc_datastruct.writers: metadata.csv byte parity with the legacy
saveData code, metadataXML, Images folders, raw-file moves and other
files."""
from __future__ import annotations

import logging
import os
import shutil
import traceback
from types import SimpleNamespace

import pandas as pd
import pytest

from cellacdc.acdc_datastruct import errors, planning, writers


def _legacy_metadata_csv(self, metadata_csv_path, basename):
    """Inline oracle: dataStruct.bioFormatsWorker.saveData (main,
    dataStruct.py:815-876), verbatim except that `self` is a namespace and
    the unused exception names are dropped (flake8 F841)."""
    savedSizeT = (
        self.timeRangeToSave[1] - self.timeRangeToSave[0] + 1
    )
    df = pd.DataFrame({
        'LensNA': self.LensNA,
        'SizeT': savedSizeT,
        'SizeZ': self.SizeZ,
        'TimeIncrement': self.TimeIncrement,
        'PhysicalSizeZ': self.PhysicalSizeZ,
        'PhysicalSizeY': self.PhysicalSizeY,
        'PhysicalSizeX': self.PhysicalSizeX,
        'basename': basename
    }, index=['values']).T
    df.index.name = 'Description'

    ch_metadata = [
        chName for c, chName in enumerate(self.chNames)
        if self.saveChannels[c]
    ]
    description = [
        f'channel_{c}_name' for c in range(self.SizeC)
        if self.saveChannels[c]
    ]
    ch_metadata.extend([
        wavelen for c, wavelen in enumerate(self.emWavelens)
        if self.saveChannels[c]
    ])
    description.extend([
        f'channel_{c}_emWavelen' for c in range(self.SizeC)
        if self.saveChannels[c]
    ])

    df_channelNames = pd.DataFrame({
        'Description': description,
        'values': ch_metadata
    }).set_index('Description')

    df = pd.concat([df, df_channelNames])

    if os.path.exists(metadata_csv_path):
        # Keep channel names already existing and not saved now
        existing_df = pd.read_csv(metadata_csv_path).set_index('Description')
        for c, chName in enumerate(self.chNames):
            if self.saveChannels[c]:
                continue
            chName_idx = f'channel_{c}_name'
            chWavelen_idx = f'channel_{c}_emWavelen'
            try:
                existing_chName = existing_df.at[chName_idx, 'values']
                df.at[chName_idx, 'values'] = existing_chName
            except Exception:
                traceback.print_exc()
                pass

            try:
                existing_chWavelen = existing_df.at[chWavelen_idx, 'values']
                df.at[chWavelen_idx, 'values'] = existing_chWavelen
            except Exception:
                traceback.print_exc()
                pass

    df.to_csv(metadata_csv_path)
    return df


CASES = {
    'all_saved': dict(
        LensNA=1.4, timeRangeToSave=(1, 10), SizeZ=5, TimeIncrement=0.5,
        PhysicalSizeZ=0.3, PhysicalSizeY=0.1, PhysicalSizeX=0.1,
        chNames=['GFP', 'mCherry'], saveChannels=[True, True],
        emWavelens=[510.0, 610.0], SizeC=2,
    ),
    'subset_and_odd_floats': dict(
        LensNA=0.75, timeRangeToSave=(2, 4), SizeZ=1, TimeIncrement=1 / 3,
        PhysicalSizeZ=1.0, PhysicalSizeY=0.0645, PhysicalSizeX=1e-7,
        chNames=['phase', 'GFP', 'RFP'], saveChannels=[False, True, False],
        emWavelens=[0.0, 500.0, 1e3], SizeC=3,
    ),
    'unicode_one_channel': dict(
        LensNA=1.4, timeRangeToSave=(1, 1), SizeZ=12, TimeIncrement=60.0,
        PhysicalSizeZ=2.0, PhysicalSizeY=0.2, PhysicalSizeX=0.2,
        chNames=['µNeon'], saveChannels=[True], emWavelens=[500.0], SizeC=1,
    ),
}


def _spec(case, basename):
    s = SimpleNamespace(**case)
    return planning.MetadataCsvSpec(
        lens_na=s.LensNA,
        saved_size_t=s.timeRangeToSave[1] - s.timeRangeToSave[0] + 1,
        size_z=s.SizeZ, time_increment_s=s.TimeIncrement,
        physical_size_z_um=s.PhysicalSizeZ, physical_size_y_um=s.PhysicalSizeY,
        physical_size_x_um=s.PhysicalSizeX, basename=basename,
        channel_names=tuple(s.chNames), save_channels=tuple(s.saveChannels),
        em_wavelens=tuple(s.emWavelens),
    )


def _read_bytes(path):
    with open(path, 'rb') as file:
        return file.read()


@pytest.mark.parametrize('name', list(CASES))
def test_metadata_csv_equals_legacy(tmp_path, name):
    case = CASES[name]
    legacy_path = str(tmp_path / 'legacy.csv')
    new_path = str(tmp_path / 'new.csv')
    legacy_df = _legacy_metadata_csv(
        SimpleNamespace(**case), legacy_path, 'exp_s01_'
    )
    spec = _spec(case, 'exp_s01_')
    pd.testing.assert_frame_equal(writers.build_metadata_df(spec), legacy_df)
    writers.write_metadata_csv(new_path, spec)
    assert _read_bytes(new_path) == _read_bytes(legacy_path)


EXISTING_CSV = (
    'Description,values\n'
    'LensNA,1.2\nSizeT,3\nSizeZ,1\nTimeIncrement,2.0\n'
    'PhysicalSizeZ,1.0\nPhysicalSizeY,0.5\nPhysicalSizeX,0.5\n'
    'basename,old_s01_\n'
    'channel_0_name,phase\nchannel_2_name,RFP\n'
    'channel_0_emWavelen,0.0\nchannel_2_emWavelen,612.5\n'
    'extra_row,kept?\n'
)


@pytest.mark.parametrize('existing', [
    EXISTING_CSV,
    # channel_2_emWavelen missing: skipped (legacy printed a traceback)
    EXISTING_CSV.replace('channel_2_emWavelen,612.5\n', ''),
    'Description,values\nLensNA,1.0\n',
])
def test_metadata_csv_add_merge_equals_legacy(tmp_path, existing, caplog):
    """Rows of channels not saved now are copied from the existing file."""
    case = CASES['subset_and_odd_floats']
    legacy_path = tmp_path / 'legacy.csv'
    new_path = tmp_path / 'new.csv'
    legacy_path.write_text(existing, encoding='utf-8')
    new_path.write_text(existing, encoding='utf-8')
    _legacy_metadata_csv(SimpleNamespace(**case), str(legacy_path),
                         'exp_s01_')
    with caplog.at_level(logging.DEBUG, logger=writers.__name__):
        writers.write_metadata_csv(str(new_path), _spec(case, 'exp_s01_'))
    assert _read_bytes(new_path) == _read_bytes(legacy_path)
    values = writers.read_metadata_csv(str(new_path))
    if existing == EXISTING_CSV:
        assert values['channel_0_name'] == 'phase'
        assert values['channel_2_emWavelen'] == '612.5'
        assert not caplog.records
    else:
        assert any('not copied' in r.getMessage() for r in caplog.records)
    assert values['channel_1_name'] == 'GFP'
    assert values['basename'] == 'exp_s01_'


def test_read_metadata_csv(tmp_path):
    path = str(tmp_path / 'm.csv')
    writers.write_metadata_csv(path, _spec(CASES['all_saved'], 'b_s1_'))
    assert writers.read_metadata_csv(path) == {
        'LensNA': '1.4', 'SizeT': '10', 'SizeZ': '5', 'TimeIncrement': '0.5',
        'PhysicalSizeZ': '0.3', 'PhysicalSizeY': '0.1',
        'PhysicalSizeX': '0.1', 'basename': 'b_s1_',
        'channel_0_name': 'GFP', 'channel_1_name': 'mCherry',
        'channel_0_emWavelen': '510.0', 'channel_1_emWavelen': '610.0',
    }


@pytest.mark.parametrize('text', ['None', '<OME>µm 中</OME>\n', ''])
def test_write_metadata_xml_equals_legacy(tmp_path, text):
    legacy = tmp_path / 'legacy.txt'
    with open(legacy, 'w', encoding="utf-8") as txt:
        txt.write(str(text))
    new = tmp_path / 'new.txt'
    writers.write_metadata_xml(str(new), text)
    assert _read_bytes(new) == _read_bytes(legacy)


def test_prepare_images_folder(tmp_path):
    images = tmp_path / 'exp' / 'Position_1' / 'Images'
    writers.prepare_images_folder(str(images), clear=True)
    assert images.is_dir()
    (images / 'segm.npz').write_text('x')
    (images.parent / 'notes.txt').write_text('keep')
    writers.prepare_images_folder(str(images), clear=False)
    assert (images / 'segm.npz').exists()
    writers.prepare_images_folder(str(images), clear=True)
    assert images.is_dir() and list(images.iterdir()) == []
    assert (images.parent / 'notes.txt').exists()


def _touch(folder, *names):
    os.makedirs(folder, exist_ok=True)
    for name in names:
        with open(os.path.join(folder, name), 'w') as file:
            file.write(name)


def _mutations(tmp_path, files, *, renames=None, companion=()):
    src = str(tmp_path / 'src')
    dst = str(tmp_path / 'exp')
    _touch(src, *files)
    for folder in companion:
        _touch(os.path.join(src, folder), 'inner.txt')
    return src, dst, planning.plan_source_mutations(
        src, files, dst_dir=dst, move_raw=True, renames=renames
    )


def test_execute_source_mutations(tmp_path):
    src, dst, m = _mutations(
        tmp_path, ['a b.vsi', 'c.tif'], renames={'a b.vsi': 'a_b.vsi'},
        companion=['_a_b_']
    )
    logged = []
    moved = writers.execute_source_mutations(
        m, log=lambda text, level='INFO': logged.append((text, level))
    )
    raw_dir = os.path.join(dst, 'raw_microscopy_files')
    assert moved == [
        (os.path.join(src, 'a_b.vsi'), os.path.join(raw_dir, 'a_b.vsi')),
        (os.path.join(src, 'c.tif'), os.path.join(raw_dir, 'c.tif')),
    ]
    assert sorted(os.listdir(raw_dir)) == ['_a_b_', 'a_b.vsi', 'c.tif']
    assert os.listdir(os.path.join(raw_dir, '_a_b_')) == ['inner.txt']
    assert os.listdir(src) == []
    assert logged == [('Renaming "a b.vsi" to "a_b.vsi"...', 'INFO')]
    for filename, path in m.final_paths.items():
        assert os.path.exists(path), filename


@pytest.mark.parametrize('exception, permission', [
    (PermissionError('locked'), True),
    (OSError('disk full'), False),
    (shutil.Error('boom'), False),
])
def test_execute_source_mutations_failure(tmp_path, monkeypatch, exception,
                                          permission):
    src, dst, m = _mutations(tmp_path, ['a.tif', 'b.tif', 'c.tif'])
    real_move = shutil.move

    def failing_move(source, target, *args, **kwargs):
        if os.path.basename(source) == 'b.tif':
            raise exception
        return real_move(source, target, *args, **kwargs)

    monkeypatch.setattr(shutil, 'move', failing_move)
    with pytest.raises(errors.MoveRawFilesError) as info:
        writers.execute_source_mutations(m, log=lambda *a, **k: None)
    err = info.value
    assert err.permission is permission
    assert err.failed == os.path.join(src, 'b.tif')
    raw_dir = os.path.join(dst, 'raw_microscopy_files')
    assert err.moved == [
        (os.path.join(src, 'a.tif'), os.path.join(raw_dir, 'a.tif'))
    ]
    assert 'Traceback' in err.details
    # already moved files stay moved (GUI parity)
    assert os.listdir(raw_dir) == ['a.tif']
    assert sorted(os.listdir(src)) == ['b.tif', 'c.tif']


def test_execute_source_mutations_rename_failure(tmp_path, monkeypatch):
    _, _, m = _mutations(tmp_path, ['a b.tif'], renames={'a b.tif': 'a.tif'})

    def failing_rename(old, new):
        raise PermissionError('locked')

    monkeypatch.setattr(os, 'rename', failing_rename)
    with pytest.raises(errors.MoveRawFilesError) as info:
        writers.execute_source_mutations(m, log=lambda *a, **k: None)
    assert info.value.permission is True
    assert info.value.moved == []


def test_execute_other_files(tmp_path):
    src = str(tmp_path / 'src')
    images = str(tmp_path / 'Images')
    _touch(src, 'n1.txt', 'n2.txt')
    os.makedirs(images)
    actions = [
        planning.OtherFileAction(
            os.path.join(src, 'n1.txt'), os.path.join(images, 'x_n1.txt'),
            'copy'
        ),
        planning.OtherFileAction(
            os.path.join(src, 'n2.txt'), os.path.join(images, 'x_n2.txt'),
            'move'
        ),
        planning.OtherFileAction(
            os.path.join(src, 'missing.txt'), os.path.join(images, 'm.txt'),
            'copy'
        ),
    ]
    logged = []
    done = writers.execute_other_files(
        actions, log=lambda text, level='INFO': logged.append((text, level))
    )
    assert done == [actions[0].dst, actions[1].dst]
    assert sorted(os.listdir(images)) == ['x_n1.txt', 'x_n2.txt']
    assert sorted(os.listdir(src)) == ['n1.txt']
    assert len(logged) == 1 and logged[0][1] == 'WARNING'
    assert 'missing.txt' in logged[0][0]
