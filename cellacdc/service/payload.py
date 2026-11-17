"""Encode model init kwargs that contain data (arrays, tables, files) for HTTP.

JSON values are sent as-is. numpy arrays go into an ``.npz`` attachment and
pandas DataFrames are embedded as JSON (``orient='split'``). Selected model
files are sent as bytes in the attachment, never as client-side paths.
The worker keeps its own temporary copies for the lifetime of the model.
Directory uploads are not supported.

Embedding responses use a single binary body: an 8-byte big-endian metadata
length, that many bytes of UTF-8 JSON, then the optional ``.npz`` attachment.
Metadata stays out of HTTP headers, whose line lengths are limited.

The JSON text is sent as a form field, which the worker (Starlette) limits to
``MAX_FORM_FIELD_SIZE`` bytes. Larger JSON text (e.g. a table of point prompts
with tens of thousands of rows) goes into the ``.npz`` attachment instead, and
the form field contains only a placeholder.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import io
import json
import os
from typing import Any, Collection, Dict, Optional, Tuple

import numpy as np

_ARRAY = '__acdc_array__'
_DATAFRAME = '__acdc_dataframe__'
_FILE = '__acdc_file__'
_DICT = '__acdc_dict__'
_TUPLE = '__acdc_tuple__'
_JSON_IN_ATTACHMENT = '__acdc_json_in_attachment__'
_JSON_KEY = '__acdc_json__'

# Starlette (the worker's web framework) rejects form fields larger than 1 MiB
# ("Part exceeded maximum size of 1024KB.")
MAX_FORM_FIELD_SIZE = 1024 * 1024


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


def needs_payload(kwargs: Dict[str, Any]) -> bool:
    return any(
        isinstance(v, np.ndarray) or _is_dataframe(v) for v in kwargs.values()
    )


def _move_large_json(
        text: str, arrays: Dict[str, np.ndarray], max_size: Optional[int]
    ) -> str:
    """Return `text`, or a placeholder if it is larger than `max_size` bytes.

    The text is then stored in `arrays` (sent in the `.npz` attachment). The
    placeholder is a JSON list, which older workers cannot decode as kwargs:
    they fail with an error instead of silently ignoring the kwargs.
    """
    if max_size is None:
        return text
    encoded = text.encode('utf-8')
    if len(encoded) <= max_size:
        return text
    arrays[_JSON_KEY] = np.frombuffer(encoded, dtype=np.uint8)
    return json.dumps([_JSON_IN_ATTACHMENT, _JSON_KEY])


def _load_json(text: str, arrays) -> Any:
    """Inverse of `_move_large_json`."""
    value = json.loads(text)
    if (
        isinstance(value, list) and len(value) == 2
        and value[0] == _JSON_IN_ATTACHMENT
        and isinstance(value[1], str) and value[1] in arrays
    ):
        value = json.loads(arrays[value[1]].tobytes().decode('utf-8'))
    return value


def encode(
        kwargs: Dict[str, Any], file_params: Collection[str] = (),
        max_text_size: Optional[int] = MAX_FORM_FIELD_SIZE,
    ) -> Tuple[str, Optional[bytes]]:
    """Encode `kwargs` as JSON text plus an optional `.npz` attachment.

    JSON text larger than `max_text_size` bytes is moved to the attachment
    (None: never move it).
    """
    plain, arrays = {}, {}
    for key, value in kwargs.items():
        if key in file_params and value not in (None, '', 'None'):
            path = os.fspath(value)
            with open(path, 'rb') as stream:
                arrays[key] = np.frombuffer(stream.read(), dtype=np.uint8)
            plain[key] = {_FILE: key, 'name': os.path.basename(path)}
        elif isinstance(value, np.ndarray):
            arrays[key] = value
            plain[key] = {_ARRAY: key}
        elif _is_dataframe(value):
            plain[key] = {_DATAFRAME: value.to_json(orient='split')}
        else:
            plain[key] = value
    text = _move_large_json(json.dumps(plain), arrays, max_text_size)
    data = None
    if arrays:
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        data = buf.getvalue()
    return text, data


def decode(kwargs_json: str, data: Optional[bytes] = None) -> Dict[str, Any]:
    with ExitStack() as stack:
        arrays = stack.enter_context(np.load(io.BytesIO(data))) if data else {}
        kwargs = _load_json(kwargs_json, arrays) if kwargs_json else {}
        for key, value in kwargs.items():
            if isinstance(value, dict) and _FILE in value:
                kwargs[key] = ModelFile(
                    value['name'], arrays[value[_FILE]].tobytes()
                )
            elif isinstance(value, dict) and _ARRAY in value:
                kwargs[key] = arrays[value[_ARRAY]]
            elif isinstance(value, dict) and _DATAFRAME in value:
                import pandas as pd
                kwargs[key] = pd.read_json(
                    io.StringIO(value[_DATAFRAME]), orient='split'
                )
    return kwargs


def encode_value(
        value: Any, max_text_size: Optional[int] = None,
    ) -> Tuple[str, Optional[bytes]]:
    """Safely encode a nested value containing arrays or tensor-like values.

    JSON text larger than `max_text_size` bytes is moved to the attachment
    (pass `MAX_FORM_FIELD_SIZE` when sending the text as a form field).
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

    encoded = _move_large_json(
        json.dumps(encode_item(value)), arrays, max_text_size
    )
    data = None
    if arrays:
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        data = buf.getvalue()
    return encoded, data


def decode_value(value_json: str, data: Optional[bytes] = None) -> Any:
    """Decode a value produced by :func:`encode_value` without pickle."""
    with ExitStack() as stack:
        arrays = stack.enter_context(np.load(io.BytesIO(data))) if data else {}
        encoded = _load_json(value_json, arrays)

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
    """Pack nested-value metadata and arrays into one HTTP response body."""
    metadata, data = encode_value(value)
    metadata_bytes = metadata.encode('utf-8')
    return (
        len(metadata_bytes).to_bytes(8, 'big')
        + metadata_bytes + (data or b'')
    )


def decode_value_body(body: bytes) -> Any:
    """Unpack a body produced by :func:`encode_value_body` without pickle."""
    if len(body) < 8:
        raise ValueError('Truncated payload metadata length')
    metadata_end = 8 + int.from_bytes(body[:8], 'big')
    if metadata_end > len(body):
        raise ValueError('Truncated payload metadata')
    return decode_value(
        body[8:metadata_end].decode('utf-8'), body[metadata_end:]
    )
