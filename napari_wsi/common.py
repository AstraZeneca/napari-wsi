import re
from abc import ABC
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import zarr
from numpy.typing import DTypeLike
from typing_extensions import Self
from upath import UPath
from zarr.abc.metadata import Metadata
from zarr.core.common import JSON
from zarr.storage import MemoryStore

from .backends.common import WSIReaderBackend
from .color_transform import ColorSpace, ColorTransform

if TYPE_CHECKING:
    import napari
    from zarr.core.array import Array


@dataclass(frozen=True)
class PyramidLevel(Metadata):
    factor: int
    shape: tuple[int, int]
    chunks: tuple[int, int]


class PyramidLevels:
    def __init__(self, num_channels: int, dtype: DTypeLike) -> None:
        if num_channels <= 0:
            raise ValueError("Invalid number of channels.")
        self._num_channels = num_channels
        self._dtype = np.dtype(dtype)
        self._levels: list[PyramidLevel] = []

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    def __iter__(self) -> Iterator[PyramidLevel]:
        yield from self._levels

    def __iadd__(self, level: PyramidLevel) -> Self:
        if len(self._levels) > 0:
            top_level = self._levels[-1]
            if level.factor <= top_level.factor:
                raise ValueError("The downsample factor must increase.")
            if np.any(np.array(level.shape) >= np.array(top_level.shape)):
                raise ValueError("The shape must decrease.")
        self._levels.append(level)
        return self


class PyramidStore(MemoryStore, ABC):
    def __init__(self, name: str, levels: PyramidLevels) -> None:
        super().__init__()
        self._name = name
        self._pyramid: list[tuple[int, Array]] = []
        for i, level in enumerate(levels):
            array = zarr.create_array(
                self,
                name=str(i),
                shape=(*level.shape, levels.num_channels),
                dtype=levels.dtype,
                chunks=(*level.chunks, levels.num_channels),
                shards=None,
                filters=None,
                compressors=None,
                serializer="auto",
                fill_value=0,
            )
            self._pyramid.append((level.factor, array))
        self._read_only = True

    @property
    def name(self) -> str:
        return self._name

    @cached_property
    def metadata(self) -> dict[str, JSON]:
        return {}

    @staticmethod
    def _parse_key(key: str) -> tuple[int, int, int] | None:
        assert isinstance(key, str)
        match = re.fullmatch(r"(\d*)\/c\/(\d*)\/(\d*)\/(\d*)", key)
        if match is not None:
            i, u, v, w = map(int, match.groups())
            # We expect the entire channel dimension to be one chunk.
            assert w == 0
            return i, u, v
        return None

    def to_layer_data_tuples(
        self, *, rgb: bool = True, **kwargs
    ) -> list["napari.types.LayerDataTuple"]:
        """Convert to a napari layer data tuple.

        The multi-scale image data itself will be represented by a list of dask arrays.

        Args:
            rgb: If `False`, the image data will be converted to channels-first format.
                If `True`, 3- and 4-channel data will be left in channels-last format.
            kwargs: All additional keyword arguments are used as layer parameters.

        Returns:
            A one-element list containing a napari layer data tuple of type `image`.

        """
        pyramid_data: list[da.Array] = []
        for _, array in self._pyramid:
            level_data = da.from_zarr(array, chunks=array.chunks)
            _, _, num_channels = level_data.shape
            if (not rgb) or (num_channels not in [3, 4]):
                # Move the channel dimension up, to enable the slider view in napari.
                level_data = da.moveaxis(level_data, -1, 0)
            pyramid_data.append(level_data)

        return [
            (
                pyramid_data,
                {
                    "name": self.name,
                    "metadata": self.metadata,
                    "rgb": None,
                    **kwargs,
                },
                "image",
            ),
        ]


class WSIStore(PyramidStore, ABC):
    """A base class for reading multi-scale whole-slide images."""

    def __init__(self, path: UPath | None, levels: PyramidLevels) -> None:
        self._path = path
        super().__init__(name=path.stem if path is not None else "Image", levels=levels)

    @property
    def path(self) -> UPath | None:
        return self._path

    @property
    def resolution(self) -> tuple[float, float] | None:
        return None

    @property
    def units(self) -> str | None:
        return None

    @property
    def spatial_transform(self) -> np.ndarray:
        return np.identity(3)

    @property
    def color_transform(self) -> ColorTransform:
        return ColorTransform()

    @cached_property
    def metadata(self) -> dict[str, JSON]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "resolution": self.resolution,
            "color_space": str(self.color_transform.color_space),
        }

    @cached_property
    def label_image(self) -> np.ndarray | None:
        return None

    def to_transformed_layer_data_tuples(
        self, **kwargs
    ) -> list["napari.types.LayerDataTuple"]:
        """Add a spatial transform to all layer data tuples.

        This adds both 'affine' and 'units' to the layer parameters.
        """
        items = self.to_layer_data_tuples(**kwargs)
        for item in items:
            _, layer_params, _ = item
            layer_params["affine"] = self.spatial_transform
            layer_params["units"] = self.units
        return items

    def to_viewer(
        self,
        viewer: "napari.viewer.Viewer",
        *,
        spatial_transform: bool = False,
        **kwargs,
    ) -> list["napari.layers.Layer"]:
        """Add all available layer data to the napari viewer.

        Args:
            viewer: The napari viewer.
            spatial_transform: If `True` and a spatial transform is available, all
                layers are display in the corresponding transfored coordinate system.
            kwargs: All additional keword arguments are passed to the
                `to_layer_data_tuples` method.

        Returns:
            A list of layers added to the viewer.

        """
        layers = []
        for item in (
            self.to_transformed_layer_data_tuples(**kwargs)
            if spatial_transform
            else self.to_layer_data_tuples(**kwargs)
        ):
            layer_data, layer_params, layer_type = item
            add_layer = getattr(viewer, f"add_{layer_type}")
            layers.append(add_layer(layer_data, **layer_params))
        if spatial_transform:
            viewer.scale_bar.unit = self.units
        return layers


def open_store(
    path: str | Path | UPath,
    backend: WSIReaderBackend,
    color_space: str | ColorSpace = ColorSpace.RAW,
) -> WSIStore:
    if backend == WSIReaderBackend.OPENSLIDE:
        from .backends.openslide import OpenSlideStore

        return OpenSlideStore(path, color_space=color_space)
    if backend == WSIReaderBackend.RASTERIO:
        from .backends.rasterio import RasterioStore

        return RasterioStore(path)
    if backend == WSIReaderBackend.WSIDICOM:
        from .backends.wsidicom import WSIDicomStore

        return WSIDicomStore(path, color_space=color_space)
    raise ValueError(f"Invalid backend: {backend}")
