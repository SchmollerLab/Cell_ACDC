"""Module 0 dialogs, real widgets offscreen.

`apps.QDialogMetadataXML` delegates to the shared core: unit conversion
(GV-03), channel-name validation after sanitizing and ignoring case (GV-04),
checks of the time range, the position selection, the saved channels and the
one-file-per-channel SizeC in `ok_cb` (GV-05, GV-05b, GV-05c; a hidden time
range is never checked) and `metadata.settings_from_dialog_values`; the
"Include image name" checkbox works for any number of channels (GV-14).
`apps.DataStructureSetupDialogue` lists and checks the source files with
`sources` (GV-15: upper-case extensions, the "not supported" details list
the file names), numbers Positions with `planning` and renames invalid
file names with `sources.plan_invalid_name_renames` and `writers` (GV-21:
main's new names; nothing is renamed when a new name is taken).

Message boxes are replaced by a non-blocking recorder (one GV-21 test
shows the real box without blocking, to measure it).
"""
from __future__ import annotations

import os

import pytest

from . import conditions

pytestmark = conditions.requires_qt


@pytest.fixture(scope='module')
def qapp():
    return conditions.qapp()


class MessageBoxRecorder:
    """Stand-in for `widgets.myMessageBox`: records every call (method,
    title, text, keyword arguments) in `calls`, never blocks. `click` is
    the index in `buttonsTexts` of the button "clicked" (None = Cancel)."""
    calls: list[dict] = []
    click: int | None = None

    def __init__(self, *args, **kwargs) -> None:
        self.cancel = True
        self.clickedButton = None

    def _record(self, method, parent, title, text, **kwargs):
        type(self).calls.append(
            {'method': method, 'title': title, 'text': text, **kwargs}
        )
        # The real box returns one button per entry (the entry itself when
        # it is already a button): the entries stand in for them
        buttons = list(kwargs.get('buttonsTexts', ()))
        if type(self).click is not None and buttons:
            self.cancel = False
            self.clickedButton = buttons[type(self).click]
        return buttons

    def critical(self, *args, **kwargs):
        return self._record('critical', *args, **kwargs)

    def warning(self, *args, **kwargs):
        return self._record('warning', *args, **kwargs)

    def information(self, *args, **kwargs):
        return self._record('information', *args, **kwargs)

    def question(self, *args, **kwargs):
        return self._record('question', *args, **kwargs)


@pytest.fixture
def boxes(monkeypatch):
    from cellacdc import widgets

    monkeypatch.setattr(MessageBoxRecorder, 'calls', [])
    monkeypatch.setattr(MessageBoxRecorder, 'click', None)
    monkeypatch.setattr(widgets, 'myMessageBox', MessageBoxRecorder)
    return MessageBoxRecorder


@pytest.fixture
def open_dialogs():
    dialogs = []
    yield dialogs
    for dlg in dialogs:
        dlg.hide()
        dlg.deleteLater()


@pytest.fixture
def make_dialog(qapp, boxes, open_dialogs):
    """Metadata dialog with the kwargs of `askConfirmMetadata`, shown
    offscreen (as the user sees it)."""
    from cellacdc import apps

    def make(**kwargs):
        values = dict(
            title='Metadata for pos1.tif', rawFilename='pos1.tif',
            LensNA=1.4, SizeT=5, SizeZ=1, SizeC=2, SizeS=1,
            TimeIncrement=1.0, TimeIncrementUnit='s', PhysicalSizeX=1.0,
            PhysicalSizeY=1.0, PhysicalSizeZ=1.0, PhysicalSizeUnit='μm',
            ImageName='', chNames=['GFP', 'mCherry'],
            emWavelens=[510.0, 610.0], parent=None, rawDataStruct=1,
            sampleImgData=None, rawFilePath='pos1.tif',
        )
        values.update(kwargs)
        dlg = apps.QDialogMetadataXML(**values)
        open_dialogs.append(dlg)
        dlg.show()
        return dlg
    return make


def press_ok(dlg) -> bool:
    """Press Ok; True when the dialog accepted the values and closed."""
    dlg.ok_cb(None)
    return not dlg.cancel


def titles(boxes) -> list[str]:
    return [call['title'] for call in boxes.calls]


# ------------------------------------------------------------------ GV-03
@pytest.mark.parametrize('unit, factor', [
    ('nm', 1e-3), ('μm', 1.0), ('mm', 1e3), ('cm', 1e4),
])
def test_physical_sizes_are_converted_to_um(make_dialog, unit, factor):
    dlg = make_dialog(SizeZ=3, PhysicalSizeX=0.5, PhysicalSizeY=0.25,
                      PhysicalSizeZ=2.0, PhysicalSizeUnit=unit)
    assert dlg.PhysicalSizeUnit_CB.currentText() == unit
    assert press_ok(dlg)
    assert dlg.PhysicalSizeX == pytest.approx(0.5 * factor)
    assert dlg.PhysicalSizeY == pytest.approx(0.25 * factor)
    assert dlg.PhysicalSizeZ == pytest.approx(2.0 * factor)


@pytest.mark.parametrize('label, value, expected', [
    # The arithmetic of main's convertUnits, bit for bit
    ('ms', 250.0, 250.0 / 1000), ('seconds', 7.5, 7.5),
    ('minutes', 7.5, 7.5 * 60), ('hours', 0.3, 0.3 * 3600),
])
def test_time_increment_is_converted_to_seconds(
        make_dialog, label, value, expected):
    dlg = make_dialog(TimeIncrement=value)
    dlg.TimeIncrementUnit_CB.setCurrentText(label)
    assert press_ok(dlg)
    assert dlg.TimeIncrement == expected


# ------------------------------------------------------------------ GV-04
@pytest.mark.parametrize('names, saved, valid', [
    (('GFP', 'mCherry'), (True, True), True),
    (('GFP', 'GFP2'), (True, True), True),
    (('GFP!', 'GFP?'), (True, True), False),       # both become 'GFP'
    (('GFP', 'gfp'), (True, True), False),         # case only
    (('!!', 'mCherry'), (True, True), False),      # empty after sanitizing
    (('GFP', 'gfp'), (True, False), True),         # duplicate not saved
    (('', 'mCherry'), (False, True), True),        # empty, not saved
    (('a b', 'a_b'), (True, True), False),
])
def test_channel_names_are_checked_as_saved(
        make_dialog, boxes, names, saved, valid):
    from cellacdc.acdc_datastruct import naming

    dlg = make_dialog()
    for c, (name, save) in enumerate(zip(names, saved)):
        dlg.chNames_QLEs[c].setText(name)
        dlg.saveChannels_QCBs[c].setChecked(save)
    assert dlg.checkChNames() is valid
    # Same rule as the core (command line and backstop)
    sanitized = [naming.sanitize_name(name) for name in names]
    assert (naming.validate_channel_names(sanitized, saved) == []) is valid
    assert press_ok(dlg) is valid
    if not valid:
        assert titles(boxes) == ['Invalid channel names']


def test_single_channel_name_empty_after_sanitizing(make_dialog, boxes):
    dlg = make_dialog(SizeC=1, chNames=['GFP'], emWavelens=[510.0])
    dlg.chNames_QLEs[0].setText('?!')
    assert not press_ok(dlg)
    assert titles(boxes) == ['Invalid channel names']


def test_saved_names_are_sanitized(make_dialog):
    dlg = make_dialog()
    dlg.chNames_QLEs[0].setText('GFP 1!')
    assert press_ok(dlg)
    assert dlg.chNames == ['GFP_1', 'mCherry']


# ------------------------------------------------------------------ GV-05
@pytest.mark.parametrize('time_range, valid', [
    ((1, 5), True), ((2, 4), True), ((3, 3), True),
    ((0, 3), False), ((2, 6), False), ((4, 2), False),
])
def test_time_range_is_checked_while_visible(
        make_dialog, boxes, time_range, valid):
    dlg = make_dialog(SizeT=5)
    assert dlg.timeRangeToSaveWidget.isVisible()
    dlg.timeRangeToSaveWidget.setRange(*time_range)
    assert press_ok(dlg) is valid
    if valid:
        assert tuple(dlg.timeRangeToSave) == time_range
    else:
        assert titles(boxes) == ['Invalid time range']
        assert 'frame' in boxes.calls[0]['text']


def test_hidden_stale_time_range_is_accepted(make_dialog, boxes):
    """SizeT edited to 1 hides the range widget, which keeps (1, 5):
    main accepted it and wrote what the stale range says."""
    from cellacdc.acdc_datastruct import metadata

    dlg = make_dialog(SizeT=5)
    dlg.SizeT_SB.setValue(1)
    assert not dlg.timeRangeToSaveWidget.isVisible()
    assert press_ok(dlg), boxes.calls
    assert dlg.SizeT == 1
    assert tuple(dlg.timeRangeToSave) == (1, 5)
    settings, _ = metadata.settings_from_dialog_values(dlg, file_size_t=5)
    assert settings.time_range == (1, 5)
    assert settings.errors() == []


def test_size_t_edited_below_the_file_keeps_an_explicit_range(make_dialog):
    """S18a: SizeT 10 -> 5 resets the range to (1, 5), an explicit range
    of the 10-frame file."""
    from cellacdc.acdc_datastruct import metadata

    dlg = make_dialog(SizeT=10)
    dlg.SizeT_SB.setValue(5)
    assert press_ok(dlg)
    settings, _ = metadata.settings_from_dialog_values(dlg, file_size_t=10)
    assert settings.time_range == (1, 5)
    assert settings.saved_size_t(10) == 5


def _select(dlg, labels) -> None:
    list_box = dlg.posSelector.listW.listBox
    list_box.clearSelection()
    for i in range(list_box.count()):
        item = list_box.item(i)
        item.setSelected(item.text() in labels)


@pytest.mark.parametrize('labels, expected', [
    ((), None),
    (('All positions',), None),           # selects no Position_k (main)
    (('Position_2',), frozenset({2})),
    (('Position_1', 'Position_3'), frozenset({1, 3})),
])
def test_position_selection_must_not_be_empty(
        make_dialog, boxes, labels, expected):
    from cellacdc.acdc_datastruct import metadata

    dlg = make_dialog(rawDataStruct=0, SizeS=3)
    _select(dlg, labels)
    accepted = press_ok(dlg)
    assert accepted is (expected is not None)
    if not accepted:
        assert titles(boxes) == ['No position selected']
        return
    settings, _ = metadata.settings_from_dialog_values(dlg, file_size_t=5)
    assert settings.selected_positions == expected


def test_default_position_selection_is_every_position(make_dialog):
    from cellacdc.acdc_datastruct import metadata

    dlg = make_dialog(rawDataStruct=0, SizeS=3)
    assert press_ok(dlg)
    settings, _ = metadata.settings_from_dialog_values(dlg, file_size_t=5)
    assert settings.selected_positions == frozenset({1, 2, 3})


# ----------------------------------------------------------------- GV-05b
def test_at_least_one_channel_must_be_saved(make_dialog, boxes):
    dlg = make_dialog()
    for checkbox in dlg.saveChannels_QCBs:
        checkbox.setChecked(False)
    assert not press_ok(dlg)
    assert titles(boxes) == ['No channel selected']
    dlg.saveChannels_QCBs[1].setChecked(True)
    assert press_ok(dlg)
    assert dlg.saveChannels == [False, True]


# ----------------------------------------------------------------- GV-05c
@pytest.mark.parametrize('size_c, valid', [(1, True), (2, True), (3, False)])
def test_one_file_per_channel_size_c_up_to_the_channels_found(
        make_dialog, boxes, size_c, valid):
    dlg = make_dialog(
        title='Metadata for ASY015_', rawFilename='ASY015_',
        rawDataStruct=2, SizeS=2, chNames=['GFP', 'mNeon-Green'],
        emWavelens=[500.0, 500.0],
    )
    dlg.SizeC_SB.setValue(size_c)
    assert press_ok(dlg) is valid
    if valid:
        assert dlg.SizeC == size_c
    else:
        assert titles(boxes) == ['Invalid number of channels']
        assert '2 channels found' in boxes.calls[0]['text']


def test_size_c_is_not_limited_for_other_layouts(make_dialog):
    dlg = make_dialog()
    dlg.SizeC_SB.setValue(3)
    assert press_ok(dlg)
    assert dlg.SizeC == 3


# ------------------------------------------------------------------ GV-14
@pytest.mark.parametrize('size_c', [1, 2, 3])
def test_include_image_name_checkbox(make_dialog, size_c):
    """The visible "Include image name" checkbox decides, whatever the
    number of channels (main read the hidden dummy checkbox of the last
    channel row when SizeC > 1)."""
    dlg = make_dialog(
        ImageName='Scene #1', SizeC=size_c,
        chNames=['GFP', 'mCherry', 'Cy5'][:size_c],
        emWavelens=[510.0, 610.0, 670.0][:size_c],
    )
    checkbox = dlg.addImageName_QCB
    assert checkbox.text() == 'Include image name'
    assert checkbox.isVisible()
    checkbox.setChecked(True)
    # every file-name preview shows the sanitized image name
    assert [label.toolTip() for label in dlg.filename_QLabels] == [
        f'pos1_Scene__1_{name}.tif'
        for name in ['GFP', 'mCherry', 'Cy5'][:size_c]
    ]
    assert press_ok(dlg)
    assert dlg.addImageName is True


def test_no_image_name_no_checkbox(make_dialog):
    dlg = make_dialog(ImageName='')
    assert not dlg.addImageName_QCB.isVisible()
    assert press_ok(dlg)
    assert dlg.addImageName is False


# ------------------------------------------- settings_from_dialog_values
@pytest.mark.parametrize('button, apply_to_next', [
    ('ok', 'ask'), ('trust', 'trust'), ('reuse', 'reuse'),
])
def test_settings_from_dialog_values(make_dialog, button, apply_to_next):
    from cellacdc.acdc_datastruct import metadata

    dlg = make_dialog(SizeZ=2, PhysicalSizeUnit='nm', PhysicalSizeX=100.0,
                      PhysicalSizeY=100.0, PhysicalSizeZ=500.0)
    dlg.TimeIncrementUnit_CB.setCurrentText('minutes')
    dlg.TimeIncrement_DSB.setValue(2.0)
    dlg.chNames_QLEs[0].setText('GFP!')
    dlg.saveChannels_QCBs[1].setChecked(False)
    dlg.to_h5_radiobutton.setChecked(True)
    if button == 'ok':
        dlg.ok_cb(None)
    else:
        widget = dlg.trustButton if button == 'trust' else dlg.overWriteButton
        widget.click()
    assert not dlg.cancel

    settings, mode = metadata.settings_from_dialog_values(
        dlg, file_size_t=5
    )
    assert mode == apply_to_next
    assert settings == metadata.ConversionSettings(
        to_h5=True, lens_na=1.4, size_t=5, size_z=2, size_c=2, size_s=1,
        time_increment_s=120.0, physical_size_x_um=0.1,
        physical_size_y_um=0.1, physical_size_z_um=0.5,
        channel_names=('GFP', 'mCherry'), save_channels=(True, False),
        em_wavelens=(510.0, 610.0), time_range=None,
        selected_positions=None, add_image_name=False,
    )


@pytest.mark.parametrize('file_size_t, time_range, expected', [
    (5, (1, 5), None),              # every frame of the file
    (5, (2, 4), (2, 4)),
    (None, (1, 5), (1, 5)),         # file SizeT unknown: explicit
    (8, (1, 5), (1, 5)),            # SizeT edited: explicit
])
def test_time_range_none_only_for_every_frame_of_the_file(
        make_dialog, file_size_t, time_range, expected):
    from cellacdc.acdc_datastruct import metadata

    dlg = make_dialog(SizeT=5)
    dlg.timeRangeToSaveWidget.setRange(*time_range)
    assert press_ok(dlg)
    settings, _ = metadata.settings_from_dialog_values(
        dlg, file_size_t=file_size_t
    )
    assert settings.time_range == expected


# ------------------------------------------------- setup dialog (GV-15)
@pytest.fixture
def setup_dialog(qapp, boxes, open_dialogs, monkeypatch):
    from cellacdc import apps, myutils

    # Never touch the user's recent paths
    monkeypatch.setattr(myutils, 'addToRecentPaths', lambda *a, **k: None)
    dlg = apps.DataStructureSetupDialogue(logger_func=lambda *a: None)
    open_dialogs.append(dlg)
    return dlg


def _touch(folder, *names) -> None:
    os.makedirs(folder, exist_ok=True)
    for name in names:
        with open(os.path.join(folder, name), 'wb') as file:
            file.write(b'x')


def test_unsupported_files_are_listed_by_name(setup_dialog, boxes):
    files = ['a.TIF', 'b.notanimage', 'c.ome.tif', 'd.alsonotanimage']
    assert setup_dialog.checkFilesExtensions(files) is False
    assert [(c['method'], c['title']) for c in boxes.calls] == [
        ('warning', 'File not supported')
    ]
    assert boxes.calls[0]['detailsText'] == 'b.notanimage<br>d.alsonotanimage'


def test_upper_case_extensions_are_supported(setup_dialog, boxes):
    assert setup_dialog.checkFilesExtensions(['a.TIF', 'b.Tif', 'c.ome.tif'])
    assert boxes.calls == []


def test_source_files_are_listed_like_the_core(setup_dialog, tmp_path):
    src = str(tmp_path / 'src')
    _touch(src, 'pos10.tif', 'pos2.tif', 'pos1.tif', '.hidden.tif')
    os.makedirs(os.path.join(src, 'subfolder'))
    setup_dialog.howRawDataStructCombobox.setCurrentIndex(1)
    setup_dialog.srcFolderPathSelected(src)
    assert setup_dialog.actionsFilesSourceFolder['files'] == [
        'pos1.tif', 'pos2.tif', 'pos10.tif'
    ]


@pytest.mark.parametrize('click, expected', [
    (None, []),                     # "Let me rename files myself"
    (1, ['a_b_.tif', 'ok.tif']),    # replace invalid characters with "_"
])
def test_invalid_file_names(setup_dialog, boxes, tmp_path, click, expected):
    src = str(tmp_path / 'src')
    _touch(src, 'a(b).tif', 'ok.tif')
    boxes.click = click
    result = setup_dialog.checkFileNames(['a(b).tif', 'ok.tif'], src)
    assert [c['title'] for c in boxes.calls] == ['Invalid filename']
    assert '<code>a(b).tif</code>' in boxes.calls[0]['text']
    assert result == expected
    assert sorted(os.listdir(src)) == sorted(expected or ['a(b).tif',
                                                          'ok.tif'])


def test_valid_file_names_ask_nothing(setup_dialog, boxes):
    files = ['pos 1.tif', 'pos-2.ome.tif']
    assert setup_dialog.checkFileNames(files, 'unused') == files
    assert boxes.calls == []


# ------------------------------------------------------------------ GV-21
def _write_named(folder, *names) -> None:
    """Files holding their own name (to follow them through renames)."""
    os.makedirs(folder, exist_ok=True)
    for name in names:
        with open(os.path.join(folder, name), 'w', encoding='utf-8') as file:
            file.write(name)


def _named_contents(folder) -> dict[str, str]:
    contents = {}
    for name in os.listdir(folder):
        with open(os.path.join(folder, name), encoding='utf-8') as file:
            contents[name] = file.read()
    return contents


@pytest.mark.parametrize('click, replacement', [(1, '_'), (2, '-')])
def test_rename_buttons_rename_like_main(
        setup_dialog, boxes, tmp_path, monkeypatch, click, replacement):
    """Without a collision the rename buttons rename as on main: same new
    names, same os.rename calls in the same order, one log line first."""
    from .test_sources import legacy_rename_files_replace_invalid_chars

    names = ['a b.tif', 'c!d.tif', 'ok.tif', 'x (1).tif', 'é+ü.tif']
    src, oracle = str(tmp_path / 'src'), str(tmp_path / 'oracle')
    _write_named(src, *names)
    _write_named(oracle, *names)
    calls = []
    real_rename = os.rename

    def rename(old, new):
        calls.append((os.path.basename(old), os.path.basename(new)))
        real_rename(old, new)

    monkeypatch.setattr(os, 'rename', rename)
    expected = legacy_rename_files_replace_invalid_chars(
        names, oracle, replacement_char=replacement
    )
    main_calls = list(calls)
    calls.clear()
    logs = []
    setup_dialog.logger_func = logs.append
    boxes.click = click
    assert setup_dialog.checkFileNames(names, src) == expected
    assert titles(boxes) == ['Invalid filename']
    assert calls == main_calls
    assert _named_contents(src) == _named_contents(oracle)
    assert logs == [
        'Renaming files to replace invalid characters with '
        f'"{replacement}"...'
    ]


def test_rename_collision_renames_nothing(setup_dialog, boxes, tmp_path):
    """'a b.tif' -> 'a_b.tif' would replace the raw file a_b.tif (main:
    FileExistsError on Windows, after renaming the files listed before it,
    if any; a_b.tif silently replaced on Linux/macOS): a message box lists
    the collisions in its details, nothing is renamed and the dialog is
    left as after "Let me rename files myself"."""
    src = str(tmp_path / 'src')
    _write_named(src, 'a b.tif', 'a_b.tif', 'x+y.tif')
    before = _named_contents(src)
    setup_dialog.howRawDataStructCombobox.setCurrentIndex(1)
    setup_dialog.srcFolderPathSelected(src)     # "Let me rename files myself"
    cancelled = dict(setup_dialog.actionsFilesSourceFolder)
    assert titles(boxes) == ['Invalid filename']
    boxes.calls.clear()
    logs = []
    setup_dialog.logger_func = logs.append

    boxes.click = 1                             # replace with "_"
    setup_dialog.srcFolderPathSelected(src)
    assert titles(boxes) == ['Invalid filename', 'Files not renamed']
    box = boxes.calls[1]
    assert box['method'] == 'warning'
    assert 'No file was renamed' in box['text']
    assert 'a b.tif' not in box['text']
    assert ('a b.tif -&gt; a_b.tif: a_b.tif already exists<br>'
            in box['detailsText'])
    assert box['path_to_browse'] == src
    assert _named_contents(src) == before
    assert setup_dialog.actionsFilesSourceFolder == cancelled
    assert logs == []


def test_rename_collision_box_lists_every_pair(setup_dialog, boxes, tmp_path):
    """The details list every colliding pair, with the words of
    `acdc-data convert --rename-invalid`, also for new names that differ
    from another name only in case (the files need not exist)."""
    names = ['A_B.czi', 'a+b.czi', 'a_b.czi']
    assert setup_dialog.renameInvalidFilenames(names, str(tmp_path), '_') == []
    assert titles(boxes) == ['Files not renamed']
    assert boxes.calls[0]['detailsText'].split('<br>')[1:] == [
        '  - A_B.czi and a+b.czi would be named A_B.czi and a_b.czi, which '
        'differ only in case',
        '  - a+b.czi and a_b.czi would both be named a_b.czi',
    ]


def test_rename_collision_box_fits_long_names(
        qapp, open_dialogs, monkeypatch, tmp_path):
    """The real "Files not renamed" box is as wide for 70-character file
    names as for short ones: the collisions are in its details (a text box
    that wraps and scrolls), not in its text."""
    from cellacdc import apps, myutils, widgets

    monkeypatch.setattr(myutils, 'addToRecentPaths', lambda *a, **k: None)
    shown = []

    def show_without_blocking(self):
        self.show(block=False)
        shown.append(self)
        open_dialogs.append(self)

    monkeypatch.setattr(widgets.myMessageBox, 'exec_', show_without_blocking)
    dlg = apps.DataStructureSetupDialogue(logger_func=lambda *a: None)
    open_dialogs.append(dlg)
    long_stem = (
        '20240512_experiment_strainXYZ_condition-A_60x_oil_timelapse_pos+12'
    )
    widths = []
    for i, stem in enumerate(('p+1', long_stem)):
        names = [f'{stem}.czi', f'{stem.replace("+", "_")}.czi']
        src = str(tmp_path / str(i))
        _write_named(src, *names)
        assert dlg.renameInvalidFilenames(names, src, '_') == []
        assert [box.windowTitle() for box in shown] == ['Files not renamed']
        widths.append(shown.pop().sizeHint().width())
        assert sorted(os.listdir(src)) == sorted(names)
    assert len(long_stem) + len('.czi') == 70
    assert widths[1] == widths[0]


def _make_position(dst, n, *, images=True) -> None:
    folder = os.path.join(dst, f'Position_{n}', 'Images')
    os.makedirs(folder)
    if images:
        _touch(folder, 'x.tif')


def test_missing_destination_has_no_existing_positions(
        setup_dialog, boxes, tmp_path):
    setup_dialog.dstFolderPathSelected(str(tmp_path / 'does_not_exist'))
    assert boxes.calls == []
    assert setup_dialog.actionsPosFoldersExisting['start_pos_n'] == 1


@pytest.mark.parametrize('click, flags, start_pos_n', [
    (None, (False, False, False), 1),     # Cancel: write into them (add)
    (1, (True, False, False), 1),         # Overwrite
    (2, (False, True, False), 1),         # Add image files
    (3, (False, False, True), 4),         # Create new after Position_3
])
def test_existing_positions(
        setup_dialog, boxes, tmp_path, click, flags, start_pos_n):
    dst = str(tmp_path / 'dst')
    _make_position(dst, 1)
    _make_position(dst, 3)
    _make_position(dst, 7, images=False)      # not a valid Position
    boxes.click = click
    setup_dialog.dstFolderPathControl.le.setText(dst)  # no signal
    setup_dialog.dstFolderPathSelected(dst)
    assert [c['title'] for c in boxes.calls] == [
        'Warning: existing Position folders detected!'
    ]
    actions = setup_dialog.actionsPosFoldersExisting
    assert (
        actions['overwrite'], actions['add_files'], actions['create_new']
    ) == flags
    assert actions['start_pos_n'] == start_pos_n
