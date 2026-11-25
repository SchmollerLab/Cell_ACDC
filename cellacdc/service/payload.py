"""Encode the arguments of model calls (init, segment, prompts) for HTTP.

Every argument is sent in one binary attachment (a multipart file part),
never in form fields: the worker's web framework (Starlette) limits form
fields to 1 MiB, but not files.

Attachment format, also used for the worker's responses that contain nested
values (`encode_value_body`): an 8-byte big-endian length, that many bytes
of UTF-8 JSON, then an optional `.npz` archive with the binary data that the
JSON refers to. In the model arguments (`encode`):
- numpy arrays are stored in the `.npz` archive;
- pandas DataFrames are embedded in the JSON (`orient='split'`);
- arguments that are paths of existing files are uploaded (see
  `is_file_path`): the file's bytes are stored in the `.npz` archive and
  the worker receives a `ModelFile`, never the client's path. The worker
  writes it to a temporary file. Directories are not uploaded.

The `.npz` keys are generated (`array_0`, `file_1`, ...), never argument
names, so no argument can overwrite another one.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import io
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

_ARRAY = '__acdc_array__'
_DATAFRAME = '__acdc_dataframe__'
_FILE = '__acdc_file__'
_DICT = '__acdc_dict__'
_TUPLE = '__acdc_tuple__'


@dataclass(frozen=True)
class ModelFile:
    name: str
    data: bytes

    def __post_init__(self):
        if (
            not self.name or self.name in ('.', '..')
            or any(char in self.name for char in ('/', '\\', ':', '\x00'))
        ):
            raise ValueError(f'Invalid uploaded model filename: {self.name!r}')


def _is_dataframe(value) -> bool:
    return type(value).__name__ == 'DataFrame' and hasattr(value, 'to_json')


def is_file_path(value) -> bool:
    """True if `value` is the path of an existing file that is uploaded.

    Path-like objects, absolute paths and relative paths with a folder
    (e.g. `models/weights.pth`) count; bare words such as `'None'`,
    `'point'` or `'cyto3'` never do, even if a file with that name exists in
    the working directory, because they are common argument values.
    """
    if isinstance(value, os.PathLike):
        try:
            value = os.fspath(value)
        except TypeError:
            return False
        if not isinstance(value, str):
            return False
    elif not isinstance(value, str):
        return False
    elif not (
        os.path.isabs(value) or '/' in value or os.sep in value
    ):
        return False
    try:
        return bool(value) and os.path.isfile(value)
    except (TypeError, ValueError, OSError):
        return False


def _npz_bytes(arrays: Dict[str, np.ndarray]) -> bytes:
    if not arrays:
        return b''
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _pack(text: str, data: Optional[bytes]) -> bytes:
    """Pack JSON `text` and the optional `.npz` `data` into one attachment."""
    text_bytes = text.encode('utf-8')
    return len(text_bytes).to_bytes(8, 'big') + text_bytes + (data or b'')


def _unpack(body: bytes) -> Tuple[str, bytes]:
    """Inverse of `_pack`: return the JSON text and the `.npz` bytes."""
    if len(body) < 8:
        raise ValueError('Truncated payload metadata length')
    text_end = 8 + int.from_bytes(body[:8], 'big')
    if text_end > len(body):
        raise ValueError('Truncated payload metadata')
    return body[8:text_end].decode('utf-8'), body[text_end:]


def encode(kwargs: Dict[str, Any], upload_files: bool = True) -> bytes:
    """Encode the keyword arguments of a model call as one attachment.

    Arguments that are paths of existing files (on the client, see
    `is_file_path`) are uploaded as `ModelFile`, unless `upload_files` is
    False (e.g. the path where a local worker saves a model, which may
    already exist); other path-like objects are sent as strings. Other
    values must be JSON serializable, numpy arrays or pandas DataFrames.
    """
    plain, arrays = {}, {}
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            name = f'array_{len(arrays)}'
            arrays[name] = value
            plain[key] = {_ARRAY: name}
        elif _is_dataframe(value):
            plain[key] = {_DATAFRAME: value.to_json(orient='split')}
        elif upload_files and is_file_path(value):
            path = os.fspath(value)
            with open(path, 'rb') as stream:
                content = stream.read()
            name = f'file_{len(arrays)}'
            arrays[name] = np.frombuffer(content, dtype=np.uint8)
            plain[key] = {_FILE: name, 'name': os.path.basename(path)}
        elif isinstance(value, os.PathLike):
            # A folder or a missing file: send the path as text
            plain[key] = os.fspath(value)
        else:
            plain[key] = value
    return _pack(json.dumps(plain), _npz_bytes(arrays))


def _marker(value) -> Optional[str]:
    """The marker of a value encoded by `encode` (exact shape), or None."""
    if not isinstance(value, dict):
        return None
    if set(value) == {_ARRAY} and isinstance(value[_ARRAY], str):
        return _ARRAY
    if set(value) == {_DATAFRAME} and isinstance(value[_DATAFRAME], str):
        return _DATAFRAME
    if (
        set(value) == {_FILE, 'name'} and isinstance(value[_FILE], str)
        and isinstance(value['name'], str)
    ):
        return _FILE
    return None


def decode(body: bytes) -> Dict[str, Any]:
    """Inverse of `encode`. Uploaded files are returned as `ModelFile`.

    Raises ValueError if `body` is not a valid attachment.
    """
    import zipfile

    text, data = _unpack(body)
    kwargs = json.loads(text)
    if not isinstance(kwargs, dict):
        raise ValueError(
            f'Model arguments must be a JSON object, not {type(kwargs).__name__}'
        )
    try:
        with ExitStack() as stack:
            arrays = (
                stack.enter_context(np.load(io.BytesIO(data))) if data else {}
            )
            for key, value in kwargs.items():
                marker = _marker(value)
                if marker == _FILE:
                    kwargs[key] = ModelFile(
                        value['name'], arrays[value[_FILE]].tobytes()
                    )
                elif marker == _ARRAY:
                    kwargs[key] = arrays[value[_ARRAY]]
                elif marker == _DATAFRAME:
                    import pandas as pd
                    kwargs[key] = pd.read_json(
                        io.StringIO(value[_DATAFRAME]), orient='split'
                    )
    except (KeyError, OSError, EOFError, zipfile.BadZipFile) as err:
        raise ValueError(
            f'Invalid or truncated model arguments attachment: {err!r}'
        ) from err
    return kwargs


def encode_value(value: Any) -> Tuple[str, Optional[bytes]]:
    """Safely encode a nested value containing arrays or tensor-like values.

    Returns the JSON text and the optional `.npz` bytes (also the format of
    SAM embeddings saved to disk, see `cellacdc.sam_embeddings`).
    """
    arrays = {}

    def encode_item(item):
        if isinstance(item, np.ndarray):
            key = f'array_{len(arrays)}'
            arrays[key] = item
            return {_ARRAY: key}
        if (
            hasattr(item, 'detach') and hasattr(item, 'cpu')
            and hasattr(item, 'numpy')
        ):
            key = f'array_{len(arrays)}'
            arrays[key] = item.detach().cpu().numpy()
            return {_ARRAY: key}
        if isinstance(item, dict):
            return {_DICT: [
                [encode_item(key), encode_item(val)]
                for key, val in item.items()
            ]}
        if isinstance(item, tuple):
            return {_TUPLE: [encode_item(val) for val in item]}
        if isinstance(item, list):
            return [encode_item(val) for val in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    encoded = json.dumps(encode_item(value))
    return encoded, (_npz_bytes(arrays) or None)


def decode_value(value_json: str, data: Optional[bytes] = None) -> Any:
    """Decode a value produced by :func:`encode_value` without pickle.

    Raises ValueError if the `.npz` data is invalid or incomplete.
    """
    import zipfile

    try:
        return _decode_value(value_json, data)
    except (KeyError, OSError, EOFError, zipfile.BadZipFile) as err:
        raise ValueError(f'Invalid or truncated payload: {err!r}') from err


def _decode_value(value_json: str, data: Optional[bytes]) -> Any:
    encoded = json.loads(value_json)
    with ExitStack() as stack:
        arrays = stack.enter_context(np.load(io.BytesIO(data))) if data else {}

        def decode_item(item):
            if isinstance(item, list):
                return [decode_item(val) for val in item]
            if not isinstance(item, dict):
                return item
            if _ARRAY in item:
                return arrays[item[_ARRAY]]
            if _DICT in item:
                return {
                    decode_item(key): decode_item(val)
                    for key, val in item[_DICT]
                }
            if _TUPLE in item:
                return tuple(decode_item(val) for val in item[_TUPLE])
            return {key: decode_item(val) for key, val in item.items()}

        return decode_item(encoded)


def encode_value_body(value: Any) -> bytes:
    """Encode a nested value (see `encode_value`) as one attachment/body."""
    return _pack(*encode_value(value))


def decode_value_body(body: bytes) -> Any:
    """Unpack a body produced by :func:`encode_value_body` without pickle."""
    return decode_value(*_unpack(body))
