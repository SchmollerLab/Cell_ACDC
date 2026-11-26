"""Metadata cascade parity: the new path
``extract_from_metadata_object`` / ``read_raw_metadata`` (child side) ->
``RawFileMetadata`` (JSON transport) -> ``resolve_metadata`` ->
``confirm_signal_args`` + warning lines gives exactly what the legacy
``bioFormatsWorker.readMetadata`` emitted on main (``golden/cascade.json``):
the 17 ``confirmMetadata`` arguments and every log line.

The only normalisation is for unnamed channels: main passed ``None`` in
``chNames`` for a channel whose Name is None, the new path passes ``''``
(the dialog's line edit shows the same empty text). No entry needs a GV-02
claim: the fake objects are evaluated directly and the ``.ome.tif`` files go
through ``load.OMEXML`` on both paths, so the TimeIncrement text round trip
fixed by GV-02 is not involved.
"""
from __future__ import annotations

import pytest

from cellacdc.acdc_bioio_bioformats import raw_metadata
from cellacdc.acdc_datastruct import metadata, sources

from . import fake_omexml, scenarios

CONFIRM_NAMES = (
    'filename', 'LensNA', 'SizeT', 'SizeZ', 'SizeC', 'SizeS',
    'TimeIncrement', 'TimeIncrementUnit', 'PhysicalSizeX', 'PhysicalSizeY',
    'PhysicalSizeZ', 'PhysicalSizeUnit', 'chNames', 'emWavelens',
    'ImageName', 'rawFilePath', 'sampleImgData',
)
GOLDEN = scenarios.load_golden('cascade')
LAYOUTS = {
    'layout1': sources.RawLayout.ONE_FILE_PER_POS,
    'layout2': sources.RawLayout.ONE_FILE_PER_CHANNEL,
}
FAKE_CASES = [
    (layout_key, name)
    for layout_key in LAYOUTS for name in GOLDEN['fake'][layout_key]
]
OME_CASES = list(GOLDEN['ome_tif']['layout1'])


def _one_per_channel_index() -> sources.OnePerChannelIndex:
    """The index the legacy capture used for layout 2."""
    files = [
        f'ASY015_{position}_{channel}.tif'
        for position in range(1, GOLDEN['layout2_size_s'] + 1)
        for channel in GOLDEN['layout2_channels']
    ]
    return sources.parse_one_per_channel_filenames(files)


def _normalised(confirm: dict) -> dict:
    """Golden confirm args with None channel names replaced by ''."""
    confirm = dict(confirm)
    confirm['chNames'] = [
        '' if name is None else name for name in confirm['chNames']
    ]
    return confirm


def _new_path(raw_dict: dict, layout: sources.RawLayout,
              golden_confirm: dict) -> tuple[dict, list[str]]:
    """Confirm args (by name) and log lines of the new path."""
    raw = metadata.RawFileMetadata.from_dict(raw_dict)
    opc = (
        _one_per_channel_index()
        if layout == sources.RawLayout.ONE_FILE_PER_CHANNEL else None
    )
    resolved, warnings = metadata.resolve_metadata(
        raw, layout=layout, one_per_channel=opc
    )
    args = metadata.confirm_signal_args(
        resolved, filename=golden_confirm['filename'],
        raw_path=golden_confirm['rawFilePath'], sample=None
    )
    log = ['Reading OME metadata...']
    for warning in warnings:
        log.extend(warning.gui_lines())
    return dict(zip(CONFIRM_NAMES, args)), log


def _assert_entry(entry: dict, confirm: dict, log: list[str]) -> None:
    assert entry['criticalError'] is None
    assert entry['confirm'] is not None
    assert confirm == _normalised(entry['confirm'])
    assert log == entry['log']


@pytest.mark.parametrize('layout_key, name', FAKE_CASES)
def test_fake_cascade_matches_main(layout_key, name):
    """70 FakeOMEXML grid entries x layouts 1 and 2 (legacy readMetadata on
    the same objects, captured on main)."""
    entry = GOLDEN['fake'][layout_key][name]
    golden_confirm = entry['confirm']
    fake = fake_omexml.FakeOMEXML(fake_omexml.cascade_grid()[name])
    raw_dict = raw_metadata.extract_from_metadata_object(
        fake, path=golden_confirm['rawFilePath'], source='bioio',
        reader='fake', metadata_xml='xml'
    )
    confirm, log = _new_path(raw_dict, LAYOUTS[layout_key], golden_confirm)
    _assert_entry(entry, confirm, log)


@pytest.mark.parametrize('name', OME_CASES)
def test_ome_tif_cascade_matches_main(name, tmp_path):
    """Real `.ome.tif` variants read by `read_raw_metadata` (load.OMEXML,
    as main); the raw path is set to the golden's placeholder path so the
    warning lines compare as text."""
    entry = GOLDEN['ome_tif']['layout1'][name]
    golden_confirm = entry['confirm']
    filename = scenarios.make_ome_cascade_file(str(tmp_path), name)
    assert golden_confirm['filename'] == filename
    raw_dict = raw_metadata.read_raw_metadata(str(tmp_path / filename))
    raw_dict['path'] = golden_confirm['rawFilePath']
    confirm, log = _new_path(
        raw_dict, sources.RawLayout.ONE_FILE_PER_POS, golden_confirm
    )
    _assert_entry(entry, confirm, log)


def test_normalisation_is_needed_only_for_none_names():
    """The None -> '' normalisation is needed only for 'channels:name_none'
    of layout 1, the only golden entry with a None channel name."""
    with_none = sorted(
        f'{layout_key}/{name}'
        for layout_key, name in FAKE_CASES
        if None in GOLDEN['fake'][layout_key][name]['confirm']['chNames']
    )
    assert with_none == ['layout1/channels:name_none']
    assert not any(
        None in entry['confirm']['chNames']
        for entry in GOLDEN['ome_tif']['layout1'].values()
    )


def test_every_golden_entry_is_compared():
    assert len(FAKE_CASES) == 2 * len(fake_omexml.cascade_grid())
    assert set(OME_CASES) == set(scenarios.OME_CASCADE_VARIANTS)
