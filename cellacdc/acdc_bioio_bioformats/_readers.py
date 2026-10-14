from pathlib import Path
from typing import Any, Tuple

import dask.array as da
import h5py
import numpy as np
import xarray as xr
from fsspec.spec import AbstractFileSystem

from bioio_base.reader import Reader
from bioio_base.types import PhysicalPixelSizes
from imaris_ims_file_reader.ims import ims

class _ImsArray:
    """Array-like view of one IMS resolution level for dask."""

    def __init__(self, source: Any, level: int, shape: Tuple[int, ...]):
        self.source = source
        self.level = level
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = np.dtype(source.dtype)
        source_chunks = tuple(int(c) for c in source.chunks)
        self.chunks = tuple(min(c, n) for c, n in zip(source_chunks, shape))

    def __getitem__(self, key: Any) -> np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (5 - len(key))
        # The IMS reader accepts a leading resolution-level index.
        return np.asarray(self.source[(self.level, *key)])


class ImsReader(Reader):
    NAME = "bioio-ims"

    def __init__(self, image: str | Path, **kwargs: Any):
        super().__init__(image, **kwargs)
        self._source = ims(str(image), ResolutionLevelLock=0, squeeze_output=False)
        self._current_resolution_level = 0
        self._array = None

    @staticmethod
    def _is_supported_image(
        fs: AbstractFileSystem, path: str, **kwargs: Any
    ) -> bool:
        # The underlying IMS package opens local HDF5 files.
        protocol = fs.protocol
        if isinstance(protocol, (tuple, list)):
            local = any(p in ("file", "local") for p in protocol)
        else:
            local = protocol in ("file", "local")

        return (
            local
            and path.lower().endswith(".ims")
            and h5py.is_hdf5(path)
        )

    @property
    def scenes(self) -> Tuple[str, ...]:
        # The IMS reader exposes one image series.
        return ("0",)

    @property
    def resolution_levels(self) -> Tuple[int, ...]:
        return tuple(range(int(self._source.ResolutionLevels)))

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        # imaris_ims_file_reader reports spacing in Z, Y, X order.
        z, y, x = self._source.resolution
        return PhysicalPixelSizes(Z=float(z), Y=float(y), X=float(x))

    @property
    def metadata(self) -> Any:
        return self._source.metaData

    def set_resolution_level(self, resolution_level: int) -> None:
        super().set_resolution_level(resolution_level)
        self._source.change_resolution_lock(resolution_level)
        self._array = None

    def _get_array(self) -> _ImsArray:
        level = self.current_resolution_level
        raw_shape = tuple(
            int(n) for n in self._source.metaData[(level, 0, 0, "shape")]
        )

        # IMS metadata can include leading singleton axes; bioio expects TCZYX.
        zyx = raw_shape[-3:]
        shape = (
            int(self._source.TimePoints),
            int(self._source.Channels),
            *zyx,
        )
        return _ImsArray(self._source, level, shape)

    def _read_delayed(self) -> xr.DataArray:
        self._array = self._get_array()
        data = da.from_array(
            self._array,
            chunks=self._array.chunks,
            asarray=False,
            fancy=False,
        )
        coords = {"C": [f"Channel {i}" for i in range(self._source.Channels)]}
        return xr.DataArray(data, dims=tuple("TCZYX"), coords=coords)

    def _read_immediate(self) -> xr.DataArray:
        array = self._get_array()
        data = array[tuple(slice(None) for _ in range(5))]
        coords = {"C": [f"Channel {i}" for i in range(self._source.Channels)]}
        return xr.DataArray(data, dims=tuple("TCZYX"), coords=coords)