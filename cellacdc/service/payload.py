"""Encode model init kwargs that contain data (arrays, tables, files) for HTTP.

JSON values are sent as-is. numpy arrays go into an ``.npz`` attachment and
pandas DataFrames are embedded as JSON (``orient='split'``). Selected model
files are sent as bytes in the attachment, never as client-side paths.
The worker keeps its own temporary copies for the lifetime of the model.
Directory uploads are not supported.
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


def encode(
        kwargs: Dict[str, Any], file_params: Collection[str] = (),
    ) -> Tuple[str, Optional[bytes]]:
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
    data = None
    if arrays:
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        data = buf.getvalue()
    return json.dumps(plain), data


def decode(kwargs_json: str, data: Optional[bytes] = None) -> Dict[str, Any]:
    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    with ExitStack() as stack:
        arrays = stack.enter_context(np.load(io.BytesIO(data))) if data else {}
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
