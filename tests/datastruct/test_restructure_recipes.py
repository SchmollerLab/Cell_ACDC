"""Re-structure recipes and metadata: the recipe JSON the "Re-structure
image files" dialog saves is what `acdc-data restructure --recipe`
consumes, so saving, loading and turning a recipe into core options must
match the dialog byte for byte.

Legacy oracles below are inline copies of the dialog code on main
(apps.py:23523-23551 `metadataDataFrame`, 23962-23964 `saveRecipe`,
24067-24069 `saveLastParameters`, 24033-24034 `loadRecipe`, 23988-23999
`selectRecipeFilepath`).
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from cellacdc.acdc_datastruct import errors, restructure, units

from . import conditions, manifest, synthetic

_APP = None

RECIPE_PER_FILE = {
    'channels': {'mode': 'per_file', 'count': 2, 'names': [
        {'name': 'GFP', 'filename_identifier': 'GFP'},
        {'name': 'mCh', 'filename_identifier': 'mCh'}]},
    'timepoints': {'mode': 'per_file', 'filename_marker': '_t'},
    'z_slices': {'mode': 'in_file', 'axis': '0'},
}
METADATA = {
    'PhysicalSizeX': 0.1, 'PhysicalSizeY': 0.2, 'PhysicalSizeZ': 0.5,
    'TimeIncrement': 5.0, 'TimeIncrementUnit': 'minutes',
}


# ------------------------------------------------------------ legacy oracles
def legacy_save(path, dimensionOptions, metadata):
    """apps.py saveRecipe / saveLastParameters on main."""
    recipe = {**dimensionOptions, 'metadata': metadata}
    with open(path, 'w') as f:
        json.dump(recipe, f, indent=2)


def legacy_load(path):
    """apps.py loadRecipe / loadLastParameters on main."""
    with open(path, 'r') as f:
        savedOptions = json.load(f)
    return savedOptions


def legacy_metadata_df(metadata, channelNames):
    """apps.py metadataDataFrame on main (channel names already stripped)."""
    TimeIncrementUnit = metadata['TimeIncrementUnit']
    toSecondsFactor = {
        'ms': 1e-3, 'seconds': 1.0, 'minutes': 60.0, 'hours': 3600.0
    }[TimeIncrementUnit]
    TimeIncrement = metadata['TimeIncrement']*toSecondsFactor

    values = {
        'TimeIncrement': TimeIncrement,
        'PhysicalSizeZ': metadata['PhysicalSizeZ'],
        'PhysicalSizeY': metadata['PhysicalSizeY'],
        'PhysicalSizeX': metadata['PhysicalSizeX'],
    }
    for c, name in enumerate(channelNames):
        values[f'channel_{c}_name'] = name

    df = pd.DataFrame({
        'Description': list(values.keys()),
        'values': list(values.values())
    }).set_index('Description')
    return df


def build_df(metadata, channel_names):
    return restructure.build_metadata_df(
        physical_size_x=metadata['PhysicalSizeX'],
        physical_size_y=metadata['PhysicalSizeY'],
        physical_size_z=metadata['PhysicalSizeZ'],
        time_increment=metadata['TimeIncrement'],
        time_increment_unit=metadata['TimeIncrementUnit'],
        channel_names=channel_names,
    )


def read_bytes(path):
    with open(path, 'rb') as file:
        return file.read()


def legacy_recipe_files(folder):
    """apps.py selectRecipeFilepath on main: (display name, file name) of
    the listed recipes, in the listed order."""
    from cellacdc import myutils

    recipes = []
    for file in myutils.listdir(folder):
        if not file.startswith('datastruct_restructure_recipe'):
            continue
        if not file.lower().endswith('.json'):
            continue
        parts = file.split('datastruct_restructure_recipe_', 1)
        if len(parts) != 2:
            continue
        recipes.append((parts[1], file))
    return recipes


RECIPE_FOLDER_FILES = (
    'datastruct_restructure_recipe_10.json',
    'datastruct_restructure_recipe_2.json',
    'datastruct_restructure_recipe_abc.json',
    'datastruct_restructure_recipe_Mine.json',
    'datastruct_restructure_recipe_x-y.JSON',
    'datastruct_restructure_recipe_x.json',
    'datastruct_restructure_recipe.json',
    'datastruct_restructure_recipe_notes.txt',
    'other.json',
    '_last_setup.json',
)
"""A recipes folder: recipes with mixed case, numbers and '-', and files
that are not recipes."""


# ------------------------------------------------------------ save and load
@pytest.mark.parametrize('recipe', [
    RECIPE_PER_FILE,
    {'channels': {'mode': 'not_present', 'count': 1,
                  'names': [{'name': 'Channel_1'}]},
     'timepoints': {'mode': 'in_file', 'axis': '1'},
     'z_slices': {'mode': 'not_present'}},
    {'channels': {'mode': 'in_file', 'axis': 0, 'count': 2,
                  'names': [{'name': 'ph'}, {'name': 'fl'}]},
     'timepoints': {'mode': 'per_file', 'filename_marker': '_µt'},
     'z_slices': {'mode': 'not_present'}},
], ids=['per_file', 'in_file_t', 'non_ascii_marker'])
def test_save_recipe_is_byte_identical_to_the_dialog(tmp_path, recipe):
    legacy = tmp_path / 'legacy.json'
    new = tmp_path / 'sub' / 'new.json'
    legacy_save(str(legacy), recipe, METADATA)
    restructure.save_recipe(str(new), recipe, METADATA)
    assert read_bytes(new) == read_bytes(legacy)
    assert restructure.load_recipe(str(new)) == legacy_load(str(legacy))
    assert restructure.load_recipe(str(new)) == {**recipe,
                                                 'metadata': METADATA}


def test_load_recipe_returns_the_json_unchanged(tmp_path):
    recipe = {'channels': {'mode': 'in_file', 'axis_index': 1,
                           'names': ['a', 'b']},
              'timepoints': {'mode': 'in_file', 'axis': '0'},
              'extra': [1, 2]}
    path = tmp_path / 'r.json'
    path.write_text(json.dumps(recipe), encoding='utf-8')
    assert restructure.load_recipe(str(path)) == recipe


@pytest.mark.parametrize('content, match', [
    (None, 'Cannot read the recipe'),
    ('{not json', 'Cannot read the recipe'),
    ('[1, 2]', 'does not contain a JSON object'),
])
def test_load_recipe_errors(tmp_path, content, match):
    path = tmp_path / 'r.json'
    if content is not None:
        path.write_text(content, encoding='utf-8')
    with pytest.raises(errors.RestructureOptionsError, match=match):
        restructure.load_recipe(str(path))


# ------------------------------------------------------- resolve_recipe_spec
@pytest.fixture
def recipes_dir(tmp_path, monkeypatch):
    import cellacdc

    folder = tmp_path / 'recipes'
    folder.mkdir()
    monkeypatch.setattr(cellacdc, 'restruct_files_recipes_path', str(folder))
    return folder


def test_recipes_folder_reads_the_package_setting(recipes_dir):
    assert restructure.recipes_folder() == str(recipes_dir)


def test_resolve_recipe_spec(recipes_dir, tmp_path, monkeypatch):
    for name in ('mine', '10', '2'):
        restructure.save_recipe(
            str(recipes_dir / f'{restructure.RECIPE_PREFIX}{name}.json'),
            RECIPE_PER_FILE, METADATA,
        )
    last = recipes_dir / restructure.LAST_SETUP_FILENAME
    with pytest.raises(errors.RestructureOptionsError, match='No last'):
        restructure.resolve_recipe_spec('last')
    restructure.save_recipe(str(last), RECIPE_PER_FILE, METADATA)
    assert restructure.resolve_recipe_spec('last') == str(last)
    expected = str(recipes_dir / f'{restructure.RECIPE_PREFIX}mine.json')
    assert restructure.resolve_recipe_spec('mine') == expected
    assert restructure.resolve_recipe_spec('mine.json') == expected

    other = tmp_path / 'other.json'
    other.write_text('{}', encoding='utf-8')
    assert restructure.resolve_recipe_spec(str(other)) == str(other)
    monkeypatch.chdir(tmp_path)
    assert restructure.resolve_recipe_spec('other.json') == str(other)

    with pytest.raises(errors.RestructureOptionsError) as info:
        restructure.resolve_recipe_spec('nope')
    assert 'saved recipes: 2, 10, mine' in str(info.value)
    custom = tmp_path / 'custom'
    with pytest.raises(errors.RestructureOptionsError,
                       match='saved recipes: none'):
        restructure.resolve_recipe_spec('mine', folder=str(custom))


def test_recipe_filenames_like_the_dialog(tmp_path):
    """recipe_filenames lists the files of main's "Load recipe" dialog in
    its order (natsorted file names); resolve_recipe_spec names them in
    that order and finds the last setup with last_setup_path."""
    for name in RECIPE_FOLDER_FILES:
        (tmp_path / name).write_text('{}', encoding='utf-8')
    filenames = restructure.recipe_filenames(str(tmp_path))
    assert filenames == [
        file for _, file in legacy_recipe_files(str(tmp_path))
    ]
    assert filenames[:4] == [
        'datastruct_restructure_recipe_2.json',
        'datastruct_restructure_recipe_10.json',
        'datastruct_restructure_recipe_Mine.json',
        'datastruct_restructure_recipe_abc.json',
    ]
    with pytest.raises(errors.RestructureOptionsError) as info:
        restructure.resolve_recipe_spec('nope', folder=str(tmp_path))
    assert 'saved recipes: 2, 10, Mine, abc, x-y, x;' in str(info.value)
    assert restructure.recipe_filenames(str(tmp_path / 'missing')) == []

    last = str(tmp_path / restructure.LAST_SETUP_FILENAME)
    assert restructure.last_setup_path(str(tmp_path)) == last
    assert restructure.resolve_recipe_spec('last', str(tmp_path)) == last
    with pytest.raises(errors.RestructureOptionsError, match='No last'):
        restructure.last_setup_path(str(tmp_path / 'missing'))


# ------------------------------------------------------------- metadata_df
_UNIT_GRID = [
    (unit, value)
    for unit in ('ms', 'seconds', 'minutes', 'hours')
    for value in (0.0, 1.0, 1 / 3, 0.1, 7.5, 1e-7, 123456.789)
]


@pytest.mark.parametrize('unit, value', _UNIT_GRID)
def test_build_metadata_df_matches_the_dialog_code(unit, value):
    metadata = {'PhysicalSizeX': 0.1, 'PhysicalSizeY': 1 / 3,
                'PhysicalSizeZ': 2.0, 'TimeIncrement': value,
                'TimeIncrementUnit': unit}
    names = ['GFP', 'mCh-2']
    new = build_df(metadata, names)
    legacy = legacy_metadata_df(metadata, names)
    pd.testing.assert_frame_equal(new, legacy, check_exact=True)
    assert new.to_csv() == legacy.to_csv()


def test_build_metadata_df_rejects_unknown_units():
    metadata = dict(METADATA, TimeIncrementUnit='s')
    with pytest.raises(errors.RestructureOptionsError, match="'s'"):
        build_df(metadata, ['GFP'])


def test_time_unit_tables():
    assert restructure.TIME_UNIT_FACTORS == {
        'ms': 1e-3, 'seconds': 1.0, 'minutes': 60.0, 'hours': 3600.0
    }
    assert tuple(restructure.TIME_UNIT_FACTORS) == units.GUI_TIME_UNIT_LABELS
    assert tuple(restructure.CLI_TIME_UNIT_TO_LABEL) == units.CLI_TIME_UNITS
    assert set(restructure.CLI_TIME_UNIT_TO_LABEL.values()) == set(
        restructure.TIME_UNIT_FACTORS
    )


# --------------------------------------------------------- options_from_recipe
def test_options_from_recipe(tmp_path):
    recipe = dict(RECIPE_PER_FILE, metadata=METADATA)
    options = restructure.options_from_recipe(
        recipe, source_folder='S', destination_folder='D',
        basename='exp', files=('a.tif',), on_existing='create-new',
        metadata={'PhysicalSizeX': 0.3, 'TimeIncrement': None},
    )
    assert options['source_folder'] == 'S'
    assert options['destination_folder'] == 'D'
    assert options['overwrite_destination'] is False
    assert options['create_new_position_destination'] is True
    assert options['basename'] == 'exp'
    assert options['files'] == ['a.tif']
    assert options['dimensions'] == RECIPE_PER_FILE
    assert options['dimensions'] is not RECIPE_PER_FILE
    assert options['dimensions']['channels'] is not \
        RECIPE_PER_FILE['channels']
    assert options['metadata'] == dict(METADATA, PhysicalSizeX=0.3)
    expected_df = legacy_metadata_df(options['metadata'], ['GFP', 'mCh'])
    pd.testing.assert_frame_equal(options['metadata_df'], expected_df,
                                  check_exact=True)


def test_options_from_recipe_defaults():
    options = restructure.options_from_recipe(
        {}, source_folder='S', destination_folder='D'
    )
    assert options['dimensions'] == {
        key: {'mode': 'not_present'} for key in restructure.DIMENSION_KEYS
    }
    assert options['overwrite_destination'] is False
    assert options['create_new_position_destination'] is False
    assert options['basename'] is None and options['files'] is None
    assert options['metadata'] == restructure.DEFAULT_METADATA
    assert list(options['metadata_df'].index) == [
        'TimeIncrement', 'PhysicalSizeZ', 'PhysicalSizeY', 'PhysicalSizeX',
        'channel_0_name',
    ]
    assert options['metadata_df'].loc['channel_0_name', 'values'] == \
        'Channel_1'
    overwrite = restructure.options_from_recipe(
        {}, source_folder='S', destination_folder='D',
        on_existing='overwrite',
    )
    assert overwrite['overwrite_destination'] is True
    assert overwrite['create_new_position_destination'] is False


@pytest.mark.parametrize('recipe, kwargs, match', [
    ([], {}, 'JSON object'),
    ({'channels': 'x'}, {}, "'channels' must be"),
    ({}, {'on_existing': 'add'}, 'on_existing'),
    ({}, {'metadata': {'Pixel': 1}}, 'Unknown metadata keys'),
    ({'metadata': {'TimeIncrement': 'fast'}}, {}, 'must be a number'),
    ({'metadata': []}, {}, "'metadata' entry"),
    ({'metadata': {'TimeIncrementUnit': 'min'}}, {}, "'min'"),
])
def test_options_from_recipe_errors(recipe, kwargs, match):
    with pytest.raises(errors.RestructureOptionsError, match=match):
        restructure.options_from_recipe(
            recipe, source_folder='S', destination_folder='D', **kwargs
        )


def _run_core(options):
    return restructure.create_datastruct_from_restruct_tiff_dialogue_options(
        options
    )


@pytest.mark.parametrize('channels', [
    {'mode': 'in_file', 'axis': '0'},
    {'mode': 'in_file', 'axis': 0},
    {'mode': 'in_file', 'axis_index': '0'},
    {'mode': 'in_file', 'axis_index': 0, 'axis': '3'},
], ids=['axis_str', 'axis_int', 'axis_index_str', 'axis_index_wins'])
def test_recipe_axes_as_int_or_str(tmp_path, channels):
    """'axis' or 'axis_index', int or str: same output."""
    src = tmp_path / 'src'
    src.mkdir()
    synthetic.make_loose_tiff_set(str(src), pattern='well.tif',
                                  file_axes='CTYX', C=2, T=3)
    outputs = []
    for index, dims in enumerate(({'mode': 'in_file', 'axis': '0'},
                                  channels)):
        recipe = {
            'channels': dict(dims, names=[{'name': 'ph'}, {'name': 'fl'}]),
            'timepoints': {'mode': 'in_file', 'axis': '1'},
        }
        path = tmp_path / f'recipe{index}.json'
        restructure.save_recipe(str(path), recipe, METADATA)
        dst = tmp_path / f'dst{index}'
        options = restructure.options_from_recipe(
            restructure.load_recipe(str(path)), source_folder=str(src),
            destination_folder=str(dst),
        )
        _run_core(options)
        outputs.append(manifest.build_manifest(str(dst), include_env=False))
    assert manifest.diff_manifests(outputs[0], outputs[1]) == []
    assert 'Position_1/Images/fl.tif' in outputs[0]


# ------------------------------------------------------ dialog (Qt, offscreen)
@pytest.fixture
def dialog_factory(tmp_path, monkeypatch):
    """Offscreen `ReStructTiffDataStructureSetupDialogue` instances with the
    recipes folder in tmp_path and no recent-paths side effects."""
    global _APP
    _APP = conditions.qapp()
    from cellacdc import apps

    folder = tmp_path / 'recipes'
    monkeypatch.setattr(apps, 'restruct_files_recipes_path', str(folder))
    monkeypatch.setattr(apps.myutils, 'addToRecentPaths',
                        lambda *args, **kwargs: None)
    created = []

    def make():
        dlg = apps.ReStructTiffDataStructureSetupDialogue()
        created.append(dlg)
        return dlg

    make.folder = folder
    make.apps = apps
    yield make
    for dlg in created:
        dlg.close()
        dlg.deleteLater()


def _configure(dlg):
    """Per-file channels GFP/mCh, T per file '_t', Z in file axis 0 and the
    metadata of METADATA, set through the widgets."""
    controls = dlg.dimensionControls
    controls['channels']['buttons']['per_file'].setChecked(True)
    dlg.numberChannelsSpinBox.setValue(2)
    for (nameLE, tokenLE), name in zip(dlg.channelEntries, ('GFP', 'mCh')):
        nameLE.setText(name)
        tokenLE.setText(name)
    controls['timepoints']['buttons']['per_file'].setChecked(True)
    controls['timepoints']['detailLE'].setText('_t')
    controls['z_slices']['buttons']['in_file'].setChecked(True)
    controls['z_slices']['detailLE'].setText('0')
    dlg.PhysicalSizeX_DSB.setValue(METADATA['PhysicalSizeX'])
    dlg.PhysicalSizeY_DSB.setValue(METADATA['PhysicalSizeY'])
    dlg.PhysicalSizeZ_DSB.setValue(METADATA['PhysicalSizeZ'])
    dlg.TimeIncrement_DSB.setValue(METADATA['TimeIncrement'])
    dlg.TimeIncrementUnit_CB.setCurrentText(METADATA['TimeIncrementUnit'])


class _FakeFilenameDialog:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cancel = False
        self.filename = f'{kwargs["basename"]}_mine{kwargs["ext"]}'

    def exec_(self):
        pass


@conditions.requires_qt
def test_dialog_saves_and_loads_recipes(dialog_factory, monkeypatch):
    apps = dialog_factory.apps
    monkeypatch.setattr(apps, 'filenameDialog', _FakeFilenameDialog)
    dlg = dialog_factory()
    _configure(dlg)
    assert dlg.dimensionOptions() == RECIPE_PER_FILE
    assert dlg.metadata() == METADATA

    dlg.saveRecipe()
    saved = dialog_factory.folder / 'datastruct_restructure_recipe_mine.json'
    oracle = dialog_factory.folder / 'oracle.json'
    legacy_save(str(oracle), dlg.dimensionOptions(), dlg.metadata())
    assert read_bytes(saved) == read_bytes(oracle)

    dlg.saveLastParameters(dlg.dimensionOptions())
    last = dialog_factory.folder / restructure.LAST_SETUP_FILENAME
    assert read_bytes(last) == read_bytes(oracle)

    loaded = dialog_factory()
    monkeypatch.setattr(loaded, 'selectRecipeFilepath', lambda: str(saved))
    loaded.loadRecipe()
    assert loaded.dimensionOptions() == RECIPE_PER_FILE
    assert loaded.metadata() == METADATA

    last_loaded = dialog_factory()
    last_loaded.loadLastParameters()
    assert last_loaded.dimensionOptions() == RECIPE_PER_FILE
    assert last_loaded.metadata() == METADATA


@conditions.requires_qt
def test_dialog_loads_int_axis_index(dialog_factory, tmp_path, monkeypatch):
    path = tmp_path / 'int_axes.json'
    restructure.save_recipe(str(path), {
        'channels': {'mode': 'in_file', 'axis_index': 1,
                     'names': [{'name': 'a'}, {'name': 'b'}]},
        'timepoints': {'mode': 'in_file', 'axis': 0},
    }, METADATA)
    dlg = dialog_factory()
    monkeypatch.setattr(dlg, 'selectRecipeFilepath', lambda: str(path))
    dlg.loadRecipe()
    options = dlg.dimensionOptions()
    assert options['channels']['axis'] == '1'
    assert options['timepoints'] == {'mode': 'in_file', 'axis': '0'}
    assert options['z_slices'] == {'mode': 'not_present'}
    assert [c['name'] for c in options['channels']['names']] == ['a', 'b']


class _FakeTreeDialog:
    """Stand-in for `QTreeDialog`: records the listed items, selects
    `select` (None = Cancel)."""
    select = None
    items = None

    def __init__(self, items, **kwargs):
        type(self).items = list(items)
        self.cancel = self.select is None
        self.selectedText = self.select

    def exec_(self):
        pass


@conditions.requires_qt
def test_dialog_recipe_files_use_the_core(dialog_factory, monkeypatch):
    """GV-R6: "Load last parameters" loads `_last_setup.json`, the file of
    `acdc-data restructure --recipe last` (`restructure.last_setup_path`;
    main loaded the first natsorted file starting with '_last_setup', e.g.
    '_last_setup - Copy.json'); "Load recipe" lists
    `restructure.recipe_filenames` exactly as main listed the recipes."""
    import datetime

    apps = dialog_factory.apps
    folder = dialog_factory.folder
    monkeypatch.setattr(apps, 'QTreeDialog', _FakeTreeDialog)
    monkeypatch.setattr(_FakeTreeDialog, 'items', None)
    dlg = dialog_factory()
    warnings = []
    monkeypatch.setattr(dlg, 'warnNoRecipesAvailable',
                        lambda: warnings.append('no recipes'))
    initial = dlg.dimensionOptions()

    # No recipes folder
    dlg.loadLastParameters()
    assert dlg.selectRecipeFilepath() is None
    assert warnings == ['no recipes'] * 2

    # Only a copy of the last setup: no last setup (main loaded the copy)
    folder.mkdir()
    copied = {**RECIPE_PER_FILE, 'z_slices': {'mode': 'not_present'}}
    restructure.save_recipe(
        str(folder / '_last_setup - Copy.json'), copied, METADATA
    )
    dlg.loadLastParameters()
    assert warnings == ['no recipes'] * 3
    assert dlg.dimensionOptions() == initial
    assert dlg.selectRecipeFilepath() is None
    assert warnings == ['no recipes'] * 4

    restructure.save_recipe(
        str(folder / restructure.LAST_SETUP_FILENAME), RECIPE_PER_FILE,
        METADATA
    )
    dlg.loadLastParameters()
    assert warnings == ['no recipes'] * 4
    assert dlg.dimensionOptions() == RECIPE_PER_FILE
    assert dlg.metadata() == METADATA

    for name in RECIPE_FOLDER_FILES:
        if not (folder / name).exists():
            (folder / name).write_text('{}', encoding='utf-8')
    legacy = legacy_recipe_files(str(folder))
    monkeypatch.setattr(_FakeTreeDialog, 'select', 'Mine.json')
    path = dlg.selectRecipeFilepath()
    assert path == str(folder / 'datastruct_restructure_recipe_Mine.json')
    assert _FakeTreeDialog.items == [
        (display, datetime.datetime.fromtimestamp(
            os.path.getmtime(folder / file)).strftime(r'%Y/%m/%d %H:%M'))
        for display, file in legacy
    ]
    assert [display for display, _ in legacy][:4] == [
        '2.json', '10.json', 'Mine.json', 'abc.json'
    ]
    monkeypatch.setattr(_FakeTreeDialog, 'select', None)
    assert dlg.selectRecipeFilepath() is None
    assert warnings == ['no recipes'] * 4


@conditions.requires_qt
@pytest.mark.parametrize('unit', ['ms', 'seconds', 'minutes', 'hours'])
def test_dialog_metadata_df(dialog_factory, unit):
    dlg = dialog_factory()
    _configure(dlg)
    dlg.TimeIncrement_DSB.setValue(1 / 3)
    dlg.TimeIncrementUnit_CB.setCurrentText(unit)
    assert dlg.metadata()['TimeIncrementUnit'] == unit
    expected = legacy_metadata_df(dlg.metadata(), ['GFP', 'mCh'])
    pd.testing.assert_frame_equal(dlg.metadataDataFrame(), expected,
                                  check_exact=True)


@conditions.requires_qt
def test_dialog_time_unit_labels(dialog_factory):
    dlg = dialog_factory()
    combo = dlg.TimeIncrementUnit_CB
    labels = tuple(combo.itemText(i) for i in range(combo.count()))
    assert labels == tuple(restructure.TIME_UNIT_FACTORS)
    assert combo.currentText() == \
        restructure.DEFAULT_METADATA['TimeIncrementUnit']
    defaults = dlg.metadata()
    assert defaults == restructure.DEFAULT_METADATA


@conditions.requires_qt
def test_last_setup_recipe_reproduces_the_dialog_run(dialog_factory,
                                                     tmp_path, monkeypatch):
    """What `acdc-data restructure --recipe last` does with the file the
    dialog's Ok wrote gives the same output as the dialog's own options."""
    import cellacdc

    src = tmp_path / 'src'
    src.mkdir()
    files = synthetic.make_loose_tiff_set(
        str(src), pattern='{pos}_t{t}_{ch}.tif', positions=('p1', 'p2'),
        channels=('GFP', 'mCh'), timepoints=(1, 2), file_axes='ZYX', Z=2,
    )
    dst_gui, dst_cli = tmp_path / 'gui', tmp_path / 'cli'
    dst_gui.mkdir()
    dlg = dialog_factory()
    _configure(dlg)
    dlg.baseNameLE.setText('exp')
    dlg.sourceFolderPathControl.le.setText(str(src))
    dlg.dstFolderPathControl.le.setText(str(dst_gui))
    dlg.dstFolderPathSelected(str(dst_gui))
    dlg.selectedOptions['files'] = sorted(files)
    dlg.ok_cb()
    assert dlg.cancel is False
    _run_core(dlg.selectedOptions)

    monkeypatch.setattr(cellacdc, 'restruct_files_recipes_path',
                        str(dialog_factory.folder))
    recipe = restructure.load_recipe(
        restructure.resolve_recipe_spec('last')
    )
    options = restructure.options_from_recipe(
        recipe, source_folder=str(src), destination_folder=str(dst_cli),
        basename='exp', files=sorted(files),
    )
    pd.testing.assert_frame_equal(options['metadata_df'],
                                  dlg.selectedOptions['metadata_df'],
                                  check_exact=True)
    _run_core(options)
    gui = manifest.build_manifest(str(dst_gui), include_env=False)
    cli = manifest.build_manifest(str(dst_cli), include_env=False)
    assert manifest.diff_manifests(gui, cli) == []
    csv_rows = gui['Position_1/Images/exp_metadata.csv']['rows']
    assert ['TimeIncrement', '300.0'] in csv_rows
    assert os.path.isfile(dst_cli / 'Position_2' / 'Images' / 'exp_mCh.tif')
