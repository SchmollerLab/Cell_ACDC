"""Registry of the trained promptable models saved from the GUI.

Each promptable model that can be saved (e.g., ConvPaint) has its own folder
`<settings folder>/saved_promptable_models/<model_name>`. A saved model
called `<name>` is either:

* a model file `<name><ext>` saved in that folder (plus any other file the
  model writes next to it, e.g. ConvPaint also writes `<name>.yml`), or
* a link to a model file stored somewhere else (e.g., loaded from a custom
  location): a text file `<name>.link.txt` containing the absolute path of
  the model file. Text files are used instead of symbolic links because
  creating symbolic links on Windows requires administrator rights.

Names contain only letters, numbers, underscores and dashes (see
`sanitize_name`) and are compared case-insensitively (`My_model` and
`my_model` are the same model), because the file systems of Windows and
macOS are case-insensitive.

The files of a saved model `<name>` are the `<name>.<ext>` files with a
single extension (e.g., `<name>.pkl` and `<name>.yml`). Other files in the
folder (e.g., `<name>.old.pkl` or `<name>.pkl.bak`) are never listed,
overwritten, or deleted.

Saving a model is done in three steps, so that an existing model with the
same name is replaced only if the new model was saved successfully:

1. `prepare_save` returns a path in a new (hidden) staging sub-folder.
2. The model saves its files there.
3. `finish_save` replaces the existing model (or link) with the same name by
   the staged files (or `cancel_save` removes them if saving failed).

This module does not import Qt, so that it can be tested and used anywhere.
"""
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from .. import saved_promptable_models_path

LINK_SUFFIX = '.link.txt'
# Prefix of the sub-folders of the models folder where models are saved
# before replacing the existing ones (folders are never listed)
STAGING_FOLDER_PREFIX = '.saving-'

@dataclass(frozen=True)
class SavedModel:
    name: str
    # Path of the model file (the link target for linked models)
    filepath: str
    # True if the model file is stored somewhere else (link file)
    is_link: bool

    @property
    def exists(self) -> bool:
        return os.path.isfile(self.filepath)

def _normalize_ext(ext: str) -> str:
    if not ext:
        raise ValueError('The model file extension cannot be empty.')
    return ext if ext.startswith('.') else f'.{ext}'

def _same_name(name1: str, name2: str) -> bool:
    return name1.casefold() == name2.casefold()

def get_folderpath(model_name: str, create: bool = False) -> str:
    folderpath = os.path.join(saved_promptable_models_path, model_name)
    if create:
        os.makedirs(folderpath, exist_ok=True)
    return folderpath

def sanitize_name(name: str) -> str:
    """Replace characters other than letters, numbers, underscores, and
    dashes with underscores (e.g., spaces and dots)."""
    name = re.sub(r'[^A-Za-z0-9_\-]', '_', name.strip())
    return name or 'model'

def _validate_name(name: str):
    if sanitize_name(name) != name:
        raise ValueError(
            f'Invalid model name "{name}". Use only letters, numbers, '
            'underscores, and dashes.'
        )

def _name_of_file(filename: str) -> str:
    # Names never contain dots --> everything before the first dot
    return filename.split('.', 1)[0]

def _is_model_file(filename: str, name: str, ignore_case=False) -> bool:
    """True if `filename` is a file of the saved model `name`, i.e.
    `<name>.<ext>` with a single extension (e.g., `name.pkl`, `name.yml`,
    but not `name.old.pkl` or the link file `name.link.txt`)."""
    stem, sep, suffix = filename.partition('.')
    if not sep or not suffix or '.' in suffix:
        return False
    if ignore_case:
        return _same_name(stem, name)
    return stem == name

def _is_link_file(filename: str, name: str, ignore_case=False) -> bool:
    if not filename.casefold().endswith(LINK_SUFFIX):
        return False
    stem = filename[:-len(LINK_SUFFIX)]
    if ignore_case:
        return _same_name(stem, name)
    return stem == name

def get_model_filepath(model_name: str, name: str, ext: str) -> str:
    """Path of the file of a model saved (not linked) in the models folder."""
    return os.path.join(
        get_folderpath(model_name), f'{name}{_normalize_ext(ext)}'
    )

def get_link_filepath(model_name: str, name: str) -> str:
    return os.path.join(get_folderpath(model_name), f'{name}{LINK_SUFFIX}')

def read_link(link_filepath: str) -> str:
    with open(link_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                return line
    return ''

def list_saved_models(model_name: str, ext: str) -> List[SavedModel]:
    """Saved and linked models of `model_name`, sorted by name. The
    extension `ext` is matched ignoring the case (e.g., `.PKL`).

    If a name is both saved and linked (should not happen), the saved model
    file is used.
    """
    ext = _normalize_ext(ext)
    folderpath = get_folderpath(model_name)
    if not os.path.isdir(folderpath):
        return []

    models = {}
    for filename in os.listdir(folderpath):
        filepath = os.path.join(folderpath, filename)
        if not os.path.isfile(filepath):
            continue

        if filename.endswith(LINK_SUFFIX):
            name = filename[:-len(LINK_SUFFIX)]
            if name in models or '.' in name or not name:
                continue
            try:
                target = read_link(filepath)
            except Exception:
                continue
            if not target:
                continue
            models[name] = SavedModel(name, target, True)
        elif filename.casefold().endswith(ext.casefold()):
            name = filename[:-len(ext)]
            if not name or '.' in name:
                continue
            models[name] = SavedModel(name, filepath, False)

    return [models[name] for name in sorted(models, key=str.casefold)]

def get_saved_model(
        model_name: str, name: str, ext: str
    ) -> Optional[SavedModel]:
    """Saved (or linked) model called `name`, case-insensitive, or None."""
    saved_models = list_saved_models(model_name, ext)
    for saved_model in saved_models:
        if saved_model.name == name:
            return saved_model
    for saved_model in saved_models:
        if _same_name(saved_model.name, name):
            return saved_model
    return None

def get_saved_model_names(model_name: str, ext: str) -> List[str]:
    return [model.name for model in list_saved_models(model_name, ext)]

def _same_path(path1: str, path2: str) -> bool:
    try:
        if os.path.exists(path1) and os.path.exists(path2):
            return os.path.samefile(path1, path2)
    except OSError:
        pass
    return (
        os.path.normcase(os.path.abspath(path1))
        == os.path.normcase(os.path.abspath(path2))
    )

def find_by_filepath(
        model_name: str, filepath: str, ext: str
    ) -> Optional[SavedModel]:
    """Saved or linked model whose file is `filepath`, or None."""
    for saved_model in list_saved_models(model_name, ext):
        if _same_path(saved_model.filepath, filepath):
            return saved_model
    return None

def get_unique_name(model_name: str, name: str, ext: str) -> str:
    """`name` (sanitized), with a number appended if a model with the same
    name (case-insensitive) already exists (e.g., `my_model_2`)."""
    name = sanitize_name(name)
    existing = {
        existing_name.casefold()
        for existing_name in get_saved_model_names(model_name, ext)
    }
    unique_name = name
    i = 2
    while unique_name.casefold() in existing:
        unique_name = f'{name}_{i}'
        i += 1
    return unique_name

def get_link_name(filepath: str, ext: str) -> str:
    """Default (sanitized) name of a linked model file: its filename without
    the extension."""
    ext = _normalize_ext(ext)
    filename = os.path.basename(filepath)
    if filename.casefold().endswith(ext.casefold()):
        name = filename[:-len(ext)]
    else:
        name = os.path.splitext(filename)[0]
    return sanitize_name(name)

def remove_saved_model(model_name: str, name: str) -> List[str]:
    """Remove the saved model `name` from the list.

    For a linked model only the link file is removed (the linked model file
    is never deleted, also if it is in the models folder). For a model saved
    in the models folder, all its files (e.g., ConvPaint's `<name>.pkl` and
    `<name>.yml`, see `_is_model_file`) are deleted.

    Returns the list of removed files.
    """
    folderpath = get_folderpath(model_name)
    if not os.path.isdir(folderpath):
        return []

    filenames = [
        filename for filename in os.listdir(folderpath)
        if os.path.isfile(os.path.join(folderpath, filename))
    ]
    link_filenames = [f for f in filenames if _is_link_file(f, name)]
    model_filenames = [f for f in filenames if _is_model_file(f, name)]
    if link_filenames and not model_filenames:
        to_remove = link_filenames
    else:
        to_remove = model_filenames + link_filenames

    removed = []
    for filename in to_remove:
        filepath = os.path.join(folderpath, filename)
        os.remove(filepath)
        removed.append(filepath)
    return removed

def cancel_save(staged_filepath: str):
    """Remove the staging folder of `staged_filepath` (returned by
    `prepare_save`), e.g. because saving failed."""
    staging_folderpath = os.path.dirname(staged_filepath)
    if os.path.basename(staging_folderpath).startswith(STAGING_FOLDER_PREFIX):
        shutil.rmtree(staging_folderpath, ignore_errors=True)

def prepare_save(model_name: str, name: str, ext: str) -> str:
    """Return the path where the model `name` must save its file, in a new
    empty staging folder (one per save, so that saves running at the same
    time do not interfere). Call `finish_save` after saving the model, or
    `cancel_save` if saving failed.
    """
    _validate_name(name)
    folderpath = get_folderpath(model_name, create=True)
    staging_folderpath = tempfile.mkdtemp(
        prefix=STAGING_FOLDER_PREFIX, dir=folderpath
    )
    return os.path.join(staging_folderpath, f'{name}{_normalize_ext(ext)}')

def finish_save(model_name: str, staged_filepath: str) -> SavedModel:
    """Move the model files staged by `prepare_save` into the models folder.

    The caller must have confirmed overwriting an existing model with the
    same name (case-insensitive): its files (or its link file) are replaced.
    A linked model file is never modified. If moving the files fails, the
    files of the existing model are restored.
    """
    filename = os.path.basename(staged_filepath)
    name = _name_of_file(filename)
    _validate_name(name)
    folderpath = get_folderpath(model_name)
    staging_folderpath = os.path.dirname(staged_filepath)
    try:
        staged = [
            f for f in os.listdir(staging_folderpath)
            if _is_model_file(f, name) and os.path.isfile(
                os.path.join(staging_folderpath, f)
            )
        ]
    except FileNotFoundError:
        staged = []
    if filename not in staged:
        cancel_save(staged_filepath)
        raise FileNotFoundError(
            f'The model "{name}" was not saved (file "{filename}" not '
            'found).'
        )

    backup_folderpath = os.path.join(staging_folderpath, '.previous')
    os.makedirs(backup_folderpath, exist_ok=True)
    backed_up = []
    moved = []
    restored = True
    try:
        # Move the files of the existing model (or its link) away first: on
        # Windows/macOS `Name.pkl` would otherwise overwrite `name.pkl` but
        # keep the old name
        for f in os.listdir(folderpath):
            if not os.path.isfile(os.path.join(folderpath, f)):
                continue
            is_existing = (
                _is_model_file(f, name, ignore_case=True)
                or _is_link_file(f, name, ignore_case=True)
            )
            if not is_existing:
                continue
            os.replace(
                os.path.join(folderpath, f),
                os.path.join(backup_folderpath, f)
            )
            backed_up.append(f)
        for f in staged:
            os.replace(
                os.path.join(staging_folderpath, f),
                os.path.join(folderpath, f)
            )
            moved.append(f)
    except Exception as err:
        # Remove the new files already moved in, then restore the old ones
        for f in moved:
            try:
                os.remove(os.path.join(folderpath, f))
            except Exception:
                restored = False
        for f in backed_up:
            try:
                os.replace(
                    os.path.join(backup_folderpath, f),
                    os.path.join(folderpath, f)
                )
            except Exception:
                restored = False
        if not restored:
            # Keep the staging folder with the backup of the old files
            raise RuntimeError(
                f'Saving the model "{name}" failed and the previous files '
                'could not be fully restored. They are in the folder '
                f'"{backup_folderpath}". Error: {err}'
            ) from err
        raise
    finally:
        if restored:
            cancel_save(staged_filepath)
    return SavedModel(name, os.path.join(folderpath, filename), False)

def add_link(
        model_name: str, target_filepath: str, ext: str,
        name: Optional[str] = None
    ) -> SavedModel:
    """Add the model file `target_filepath` to the list of saved models,
    without copying it (a link file with its path is created).

    The name defaults to the filename of `target_filepath` without
    extension. If the file is already in the list (saved or linked), the
    existing entry is returned. If the name is already used by another
    model (case-insensitive), a number is appended to it (e.g.,
    `my_model_2`).
    """
    ext = _normalize_ext(ext)
    target_filepath = os.path.abspath(target_filepath)
    if not os.path.isfile(target_filepath):
        raise FileNotFoundError(
            f'The model file "{target_filepath}" does not exist.'
        )

    existing = find_by_filepath(model_name, target_filepath, ext)
    if existing is not None:
        return existing

    if name is None:
        name = get_link_name(target_filepath, ext)
    unique_name = get_unique_name(model_name, name, ext)

    get_folderpath(model_name, create=True)
    link_filepath = get_link_filepath(model_name, unique_name)
    with open(link_filepath, 'w', encoding='utf-8') as file:
        file.write(f'{target_filepath}\n')
    return SavedModel(unique_name, target_filepath, True)

def get_saved_model_ext(model) -> Optional[str]:
    """Extension of the model files that `model` can save and load, or None
    if the model cannot be saved.

    `model` is either a promptable model instance (in-process) or a
    `service.client.ModelClient` (isolated environment).
    """
    ext = getattr(model, 'saved_model_ext', None)
    if not ext:
        return None
    has_methods = (
        callable(getattr(model, 'save_model', None))
        and callable(getattr(model, 'load_model', None))
    )
    if not has_methods:
        return None
    return _normalize_ext(ext)
