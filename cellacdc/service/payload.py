"""Encode model init kwargs that contain data (arrays, tables) for HTTP.

JSON values are sent as-is. numpy arrays go into an ``.npz`` attachment and
pandas DataFrames are embedded as JSON (``orient='split'``). File paths are
never sent: the client loads the data, so a remote worker works the same.
"""

import io
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np

_ARRAY = '__acdc_array__'
_DATAFRAME = '__acdc_dataframe__'


def _is_dataframe(value) -> bool:
    return type(value).__name__ == 'DataFrame' and hasattr(value, 'to_json')


def needs_payload(kwargs: Dict[str, Any]) -> bool:
    return any(
        isinstance(v, np.ndarray) or _is_dataframe(v) for v in kwargs.values()
    )


def encode(kwargs: Dict[str, Any]) -> Tuple[str, Optional[bytes]]:
    plain, arrays = {}, {}
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
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
    arrays = np.load(io.BytesIO(data)) if data else {}
    for key, value in kwargs.items():
        if isinstance(value, dict) and _ARRAY in value:
            kwargs[key] = arrays[value[_ARRAY]]
        elif isinstance(value, dict) and _DATAFRAME in value:
            import pandas as pd
            kwargs[key] = pd.read_json(
                io.StringIO(value[_DATAFRAME]), orient='split'
            )
    return kwargs
