"""Registry of the trained promptable models saved from the GUI
(`cellacdc.promptable_models.saved_models`)."""
import os
import types

import pytest

import cellacdc
from cellacdc.promptable_models import saved_models

MODEL_NAME = 'ConvPaint'
EXT = '.pkl'

# Captured at import (collection) time, before the conftest redirects them
# to a temporary settings folder for each test (`_isolate_user_settings`)
REAL_SETTINGS_FOLDERPATH = cellacdc.settings_folderpath
REAL_SAVED_MODELS_PATH = cellacdc.saved_promptable_models_path


@pytest.fixture
def models_folder(monkeypatch, tmp_path):
    root = tmp_path / 'settings' / 'saved_promptable_models'
    monkeypatch.setattr(saved_models, 'saved_promptable_models_path', str(root))
    return root / MODEL_NAME


def _save(models_folder, name, with_yml=True):
    models_folder.mkdir(parents=True, exist_ok=True)
    (models_folder / f'{name}.pkl').write_text('model')
    if with_yml:
        (models_folder / f'{name}.yml').write_text('params')
    return models_folder / f'{name}.pkl'


def _external_model(tmp_path, filename='napari model.pkl'):
    folder = tmp_path / 'napari_models'
    folder.mkdir(exist_ok=True)
    filepath = folder / filename
    filepath.write_text('external model')
    return filepath


def test_settings_folder_path(_isolate_user_settings):
    assert REAL_SAVED_MODELS_PATH == os.path.join(
        REAL_SETTINGS_FOLDERPATH, 'saved_promptable_models'
    )
    # Tests never touch the user's saved models
    assert saved_models.get_folderpath(MODEL_NAME) == os.path.join(
        _isolate_user_settings, 'saved_promptable_models', MODEL_NAME
    )


def test_list_saved_models_empty(models_folder):
    assert saved_models.list_saved_models(MODEL_NAME, EXT) == []
    assert not models_folder.exists()


def test_list_saved_and_linked_models(models_folder, tmp_path):
    _save(models_folder, 'b_model')
    _save(models_folder, 'A_model', with_yml=False)
    # Not models: other extensions, names with dots, sub-folders
    (models_folder / 'notes.txt').write_text('x')
    (models_folder / 'a.b.pkl').write_text('x')
    (models_folder / 'folder.pkl').mkdir()
    external = _external_model(tmp_path)
    linked = saved_models.add_link(MODEL_NAME, str(external), EXT)

    models = saved_models.list_saved_models(MODEL_NAME, 'pkl')
    assert [model.name for model in models] == [
        'A_model', 'b_model', 'napari_model'
    ]
    a_model, b_model, napari_model = models
    assert not a_model.is_link and a_model.exists
    assert a_model.filepath == str(models_folder / 'A_model.pkl')
    assert napari_model == linked
    assert napari_model.is_link and napari_model.exists
    assert napari_model.filepath == str(external)
    assert saved_models.get_saved_model_names(MODEL_NAME, EXT) == [
        'A_model', 'b_model', 'napari_model'
    ]
    assert saved_models.get_saved_model(MODEL_NAME, 'b_model', EXT) == b_model
    assert saved_models.get_saved_model(MODEL_NAME, 'missing', EXT) is None

    # The external file is not copied, the link stores its path
    link_file = models_folder / 'napari_model.link.txt'
    assert link_file.read_text(encoding='utf-8').strip() == str(external)
    assert sorted(os.listdir(models_folder)) == sorted([
        'A_model.pkl', 'b_model.pkl', 'b_model.yml', 'notes.txt', 'a.b.pkl',
        'folder.pkl', 'napari_model.link.txt',
    ])

    # Moved/deleted linked files are listed (to be removed by the user)
    external.unlink()
    napari_model, = [
        model for model in saved_models.list_saved_models(MODEL_NAME, EXT)
        if model.is_link
    ]
    assert not napari_model.exists


def test_saved_model_has_priority_over_link_with_same_name(
        models_folder, tmp_path
    ):
    external = _external_model(tmp_path, 'my_model.pkl')
    saved_models.add_link(MODEL_NAME, str(external), EXT)
    _save(models_folder, 'my_model')
    model, = saved_models.list_saved_models(MODEL_NAME, EXT)
    assert not model.is_link
    assert model.filepath == str(models_folder / 'my_model.pkl')


def test_empty_or_unreadable_link_files_are_ignored(models_folder):
    models_folder.mkdir(parents=True)
    (models_folder / 'empty.link.txt').write_text('\n  \n')
    assert saved_models.list_saved_models(MODEL_NAME, EXT) == []


def test_add_link_names(models_folder, tmp_path):
    first = _external_model(tmp_path, 'my model.v2.pkl')
    linked = saved_models.add_link(MODEL_NAME, str(first), EXT)
    assert linked.name == 'my_model_v2'

    # Same file again (also via a relative path) --> existing entry
    cwd = os.getcwd()
    os.chdir(first.parent)
    try:
        again = saved_models.add_link(MODEL_NAME, first.name, EXT)
    finally:
        os.chdir(cwd)
    assert again == linked
    assert len(saved_models.list_saved_models(MODEL_NAME, EXT)) == 1

    # Another file with the same name --> unique name
    other_folder = tmp_path / 'other'
    other_folder.mkdir()
    second = other_folder / 'my model.v2.pkl'
    second.write_text('other model')
    linked2 = saved_models.add_link(MODEL_NAME, str(second), EXT)
    assert linked2.name == 'my_model_v2_2'

    # A saved model with the same name --> unique name too
    _save(models_folder, 'trained')
    third = _external_model(tmp_path, 'trained.pkl')
    assert saved_models.add_link(MODEL_NAME, str(third), EXT).name == (
        'trained_2'
    )

    # A model already saved in the folder is not linked
    saved = saved_models.add_link(
        MODEL_NAME, str(models_folder / 'trained.pkl'), EXT
    )
    assert saved.name == 'trained' and not saved.is_link

    # Explicit name
    fourth = _external_model(tmp_path, 'x.pkl')
    assert saved_models.add_link(
        MODEL_NAME, str(fourth), EXT, name='custom name'
    ).name == 'custom_name'


def test_add_link_missing_file(models_folder, tmp_path):
    with pytest.raises(FileNotFoundError):
        saved_models.add_link(MODEL_NAME, str(tmp_path / 'missing.pkl'), EXT)
    assert not models_folder.exists()


def test_remove_link_keeps_the_linked_file(models_folder, tmp_path):
    external = _external_model(tmp_path, 'linked.pkl')
    saved_models.add_link(MODEL_NAME, str(external), EXT)
    removed = saved_models.remove_saved_model(MODEL_NAME, 'linked')
    assert removed == [str(models_folder / 'linked.link.txt')]
    assert external.read_text() == 'external model'
    assert saved_models.list_saved_models(MODEL_NAME, EXT) == []


def test_remove_saved_model_deletes_only_its_files(models_folder):
    _save(models_folder, 'model')
    _save(models_folder, 'model_2')
    removed = saved_models.remove_saved_model(MODEL_NAME, 'model')
    assert sorted(removed) == sorted([
        str(models_folder / 'model.pkl'), str(models_folder / 'model.yml')
    ])
    assert saved_models.get_saved_model_names(MODEL_NAME, EXT) == ['model_2']
    assert (models_folder / 'model_2.yml').exists()
    assert saved_models.remove_saved_model(MODEL_NAME, 'missing') == []


def _save_staged(filepath, content='new'):
    """Like ConvPaint's `save`: `<name>.pkl` and `<name>.yml`."""
    with open(filepath, 'w') as file:
        file.write(content)
    with open(filepath[:-4] + '.yml', 'w') as file:
        file.write(f'{content} params')


def _staging_folders(models_folder):
    if not models_folder.exists():
        return []
    return [
        p for p in models_folder.iterdir()
        if p.name.startswith(saved_models.STAGING_FOLDER_PREFIX)
    ]


def _model_files(models_folder):
    return sorted(p.name for p in models_folder.iterdir() if p.is_file())


def test_save_new_model(models_folder):
    staged = saved_models.prepare_save(MODEL_NAME, 'new_model', 'pkl')
    staging_folder, = _staging_folders(models_folder)
    assert staged == str(staging_folder / 'new_model.pkl')
    # Staged files are not listed
    _save_staged(staged)
    assert saved_models.list_saved_models(MODEL_NAME, EXT) == []

    saved = saved_models.finish_save(MODEL_NAME, staged)
    assert saved == saved_models.SavedModel(
        'new_model', str(models_folder / 'new_model.pkl'), False
    )
    assert (models_folder / 'new_model.pkl').read_text() == 'new'
    assert (models_folder / 'new_model.yml').read_text() == 'new params'
    assert _staging_folders(models_folder) == []
    assert saved_models.get_saved_model_names(MODEL_NAME, EXT) == [
        'new_model'
    ]

    for name in ('with space', 'dot.name', '', '../escape'):
        with pytest.raises(ValueError):
            saved_models.prepare_save(MODEL_NAME, name, EXT)


def test_concurrent_saves_use_separate_staging_folders(models_folder):
    staged_a = saved_models.prepare_save(MODEL_NAME, 'a', EXT)
    staged_b = saved_models.prepare_save(MODEL_NAME, 'b', EXT)
    assert os.path.dirname(staged_a) != os.path.dirname(staged_b)
    _save_staged(staged_a, 'a')
    _save_staged(staged_b, 'b')
    saved_models.finish_save(MODEL_NAME, staged_b)
    saved_models.finish_save(MODEL_NAME, staged_a)
    assert saved_models.get_saved_model_names(MODEL_NAME, EXT) == ['a', 'b']
    assert (models_folder / 'a.pkl').read_text() == 'a'
    assert _staging_folders(models_folder) == []


def test_save_replaces_link_and_keeps_linked_file(models_folder, tmp_path):
    external = _external_model(tmp_path, 'linked.pkl')
    saved_models.add_link(MODEL_NAME, str(external), EXT)

    staged = saved_models.prepare_save(MODEL_NAME, 'linked', EXT)
    # The link is kept until the model is saved
    assert (models_folder / 'linked.link.txt').exists()
    _save_staged(staged)
    saved_models.finish_save(MODEL_NAME, staged)

    model, = saved_models.list_saved_models(MODEL_NAME, EXT)
    assert model.name == 'linked' and not model.is_link
    assert not (models_folder / 'linked.link.txt').exists()
    assert external.read_text() == 'external model'


def test_save_overwrites_model_with_same_name_ignoring_case(models_folder):
    _save(models_folder, 'mine')
    (models_folder / 'mine_2.pkl').write_text('other')
    staged = saved_models.prepare_save(MODEL_NAME, 'Mine', EXT)
    _save_staged(staged)
    saved_models.finish_save(MODEL_NAME, staged)

    assert saved_models.get_saved_model_names(MODEL_NAME, EXT) == [
        'Mine', 'mine_2'
    ]
    assert _model_files(models_folder) == [
        'Mine.pkl', 'Mine.yml', 'mine_2.pkl'
    ]
    assert (models_folder / 'Mine.pkl').read_text() == 'new'


def test_save_does_not_touch_other_files_with_the_same_prefix(
        models_folder, tmp_path
    ):
    models_folder.mkdir(parents=True)
    # Manual backups and a napari model copied into the folder (names with
    # dots are not models)
    for filename in ('cells.old.pkl', 'cells.pkl.bak', 'cells.v2.pkl'):
        (models_folder / filename).write_text(filename)
    linked = saved_models.add_link(
        MODEL_NAME, str(models_folder / 'cells.v2.pkl'), EXT
    )
    assert linked.name == 'cells_v2'

    staged = saved_models.prepare_save(MODEL_NAME, 'cells', EXT)
    _save_staged(staged)
    saved_models.finish_save(MODEL_NAME, staged)
    assert _model_files(models_folder) == [
        'cells.old.pkl', 'cells.pkl', 'cells.pkl.bak', 'cells.v2.pkl',
        'cells.yml', 'cells_v2.link.txt',
    ]
    for filename in ('cells.old.pkl', 'cells.pkl.bak', 'cells.v2.pkl'):
        assert (models_folder / filename).read_text() == filename

    # Removing the saved model 'cells' does not remove them either
    saved_models.remove_saved_model(MODEL_NAME, 'cells')
    assert _model_files(models_folder) == [
        'cells.old.pkl', 'cells.pkl.bak', 'cells.v2.pkl', 'cells_v2.link.txt',
    ]


def test_upper_case_extension(models_folder, tmp_path):
    models_folder.mkdir(parents=True)
    (models_folder / 'Cells.PKL').write_text('upper')
    model, = saved_models.list_saved_models(MODEL_NAME, EXT)
    assert model.name == 'Cells' and not model.is_link
    # Browsing it finds the existing entry (no link is added)
    assert saved_models.find_by_filepath(
        MODEL_NAME, str(models_folder / 'Cells.PKL'), EXT
    ) == model
    assert saved_models.add_link(
        MODEL_NAME, str(models_folder / 'Cells.PKL'), EXT
    ) == model
    assert _model_files(models_folder) == ['Cells.PKL']
    # Saving 'cells' replaces it (the overwrite prompt lists 'Cells')
    assert saved_models.get_saved_model_names(MODEL_NAME, EXT) == ['Cells']
    staged = saved_models.prepare_save(MODEL_NAME, 'cells', EXT)
    _save_staged(staged)
    saved_models.finish_save(MODEL_NAME, staged)
    assert _model_files(models_folder) == ['cells.pkl', 'cells.yml']


def test_remove_link_to_file_in_models_folder_keeps_the_file(
        models_folder, tmp_path
    ):
    models_folder.mkdir(parents=True)
    target = models_folder / 'napari.v1.pkl'
    target.write_text('napari')
    saved_models.add_link(MODEL_NAME, str(target), EXT, name='napari')
    removed = saved_models.remove_saved_model(MODEL_NAME, 'napari')
    assert removed == [str(models_folder / 'napari.link.txt')]
    assert target.read_text() == 'napari'


def test_failed_save_keeps_existing_model(models_folder):
    _save(models_folder, 'kept')
    # The model failed before writing its file
    staged = saved_models.prepare_save(MODEL_NAME, 'kept', EXT)
    with pytest.raises(FileNotFoundError):
        saved_models.finish_save(MODEL_NAME, staged)
    assert (models_folder / 'kept.pkl').read_text() == 'model'
    assert _staging_folders(models_folder) == []

    # The model wrote some files and then failed
    staged = saved_models.prepare_save(MODEL_NAME, 'kept', EXT)
    _save_staged(staged)
    saved_models.cancel_save(staged)
    assert (models_folder / 'kept.pkl').read_text() == 'model'
    assert (models_folder / 'kept.yml').read_text() == 'params'
    assert _staging_folders(models_folder) == []
    # Not a staging folder --> nothing removed
    saved_models.cancel_save(str(models_folder / 'kept.pkl'))
    assert (models_folder / 'kept.pkl').exists()


@pytest.mark.parametrize('existing', ['saved', 'link'])
def test_finish_save_restores_existing_model_on_errors(
        models_folder, monkeypatch, tmp_path, existing
    ):
    if existing == 'saved':
        _save(models_folder, 'kept')
    else:
        external = _external_model(tmp_path, 'kept.pkl')
        saved_models.add_link(MODEL_NAME, str(external), EXT)
    before = {
        name: (models_folder / name).read_text()
        for name in _model_files(models_folder)
    }
    staged = saved_models.prepare_save(MODEL_NAME, 'kept', EXT)
    _save_staged(staged)

    replace = os.replace

    def failing_replace(src, dst):
        # Moving the second new file (.yml) into the models folder fails,
        # after the .pkl was moved in
        is_staged = (
            saved_models.STAGING_FOLDER_PREFIX in src
            and '.previous' not in src
        )
        if is_staged and src.endswith('.yml'):
            raise PermissionError('file in use')
        return replace(src, dst)

    monkeypatch.setattr(saved_models.os, 'replace', failing_replace)
    # .yml after .pkl
    monkeypatch.setattr(
        saved_models.os, 'listdir',
        lambda path, _listdir=os.listdir: sorted(_listdir(path))
    )
    with pytest.raises(PermissionError):
        saved_models.finish_save(MODEL_NAME, staged)
    monkeypatch.undo()
    after = {
        name: (models_folder / name).read_text()
        for name in _model_files(models_folder)
    }
    assert after == before
    assert _staging_folders(models_folder) == []


def test_finish_save_keeps_backup_if_restore_fails(
        models_folder, monkeypatch
    ):
    _save(models_folder, 'kept')
    staged = saved_models.prepare_save(MODEL_NAME, 'kept', EXT)
    _save_staged(staged)

    def failing_replace(src, dst):
        if '.previous' in src or saved_models.STAGING_FOLDER_PREFIX in src:
            if '.previous' not in dst:
                raise PermissionError('file in use')
        return os_replace(src, dst)

    os_replace = os.replace
    monkeypatch.setattr(saved_models.os, 'replace', failing_replace)
    with pytest.raises(RuntimeError, match='could not be fully restored'):
        saved_models.finish_save(MODEL_NAME, staged)
    monkeypatch.undo()
    staging_folder, = _staging_folders(models_folder)
    backup = staging_folder / '.previous'
    assert sorted(p.name for p in backup.iterdir()) == ['kept.pkl', 'kept.yml']
    assert (backup / 'kept.pkl').read_text() == 'model'


def test_names_are_case_insensitive(models_folder, tmp_path):
    _save(models_folder, 'model')
    assert saved_models.get_unique_name(MODEL_NAME, 'Model', EXT) == (
        'Model_2'
    )
    assert saved_models.get_unique_name(MODEL_NAME, 'other', EXT) == 'other'
    assert saved_models.get_saved_model(MODEL_NAME, 'MODEL', EXT).name == (
        'model'
    )

    # Browsing `Model.pkl` does not overwrite the entry `model`
    external = _external_model(tmp_path, 'Model.pkl')
    linked = saved_models.add_link(MODEL_NAME, str(external), EXT)
    assert linked.name == 'Model_2'
    assert saved_models.get_saved_model_names(MODEL_NAME, EXT) == [
        'model', 'Model_2'
    ]
    assert (models_folder / 'model.pkl').read_text() == 'model'


def test_find_by_filepath_and_link_name(models_folder, tmp_path):
    saved = _save(models_folder, 'saved')
    external = _external_model(tmp_path, 'Napari Model.PKL')
    assert saved_models.find_by_filepath(MODEL_NAME, str(external), EXT) is (
        None
    )
    assert saved_models.get_link_name(str(external), EXT) == 'Napari_Model'
    linked = saved_models.add_link(MODEL_NAME, str(external), EXT)
    assert linked.name == 'Napari_Model'
    assert saved_models.find_by_filepath(
        MODEL_NAME, str(external), EXT
    ) == linked
    assert saved_models.find_by_filepath(
        MODEL_NAME, str(saved), EXT
    ).name == 'saved'


def test_sanitize_name():
    assert saved_models.sanitize_name(' my model.v1 ') == 'my_model_v1'
    assert saved_models.sanitize_name('ok-name_1') == 'ok-name_1'
    assert saved_models.sanitize_name('') == 'model'


def test_get_saved_model_ext():
    class Saveable:
        saved_model_ext = 'pkl'

        def save_model(self, filepath):
            pass

        def load_model(self, filepath):
            pass

    assert saved_models.get_saved_model_ext(Saveable()) == '.pkl'

    no_ext = Saveable()
    no_ext.saved_model_ext = None
    assert saved_models.get_saved_model_ext(no_ext) is None

    only_ext = types.SimpleNamespace(saved_model_ext='.pkl')
    assert saved_models.get_saved_model_ext(only_ext) is None
    assert saved_models.get_saved_model_ext(object()) is None
