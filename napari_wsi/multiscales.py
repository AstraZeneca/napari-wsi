from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import zarr.storage
from napari.types import LayerDataTuple
from numpy.typing import DTypeLike

from .util import DEFAULT_COLORMAP, Size, as_layer_data_tuple, create_colormaps


@dataclass(frozen=True)
class LevelInfo:
    level: int
    factor: Union[int, float]
    size: Size


class BaseStore(zarr.storage.BaseStore, ABC):
    def __init__(
        self,
        name: str,
        num_channels: int,
        tile_size: Size,
        scale: Tuple[float, float],
        dtype: DTypeLike,
        store: zarr.storage.StoreLike,
    ):
        if not name:
            raise ValueError("Invalid name for store.")
        if num_channels <= 0:
            raise ValueError("Invalid number of channels for store.")
        if tile_size.width <= 0 or tile_size.height <= 0:
            raise ValueError("Invalid tile size for store.")
        if len(scale) != 2 or scale[0] <= 0 or scale[1] <= 0:
            raise ValueError("Invalid scale for store.")

        self._name = name
        self._num_channels = num_channels
        self._tile_size = tile_size
        self._scale = scale
        self._dtype = dtype

        self._store = store

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def tile_size(self) -> Size:
        return self._tile_size

    @property
    def scale(self) -> Tuple[float, float]:
        return self._scale

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return sum(1 for _ in self)

    def __contains__(self, key: str):
        return key in self._store

    def __eq__(self, other):
        raise NotImplementedError

    def __setitem__(self, key, val):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def keys(self):
        return self._store.keys()


class MultiScalesStore(BaseStore, ABC):
    def __init__(
        self,
        name: str,
        num_channels: int,
        tile_size: Size,
        scale: Tuple[float, float],
        dtype: DTypeLike,
        level_info: Sequence[LevelInfo],
    ):
        if len(level_info) == 0:
            raise ValueError("Invalid level information for store.")

        store: zarr.storage.StoreLike = {}
        zarr.storage.init_group(store)
        store[zarr.storage.attrs_key] = zarr.util.json_dumps(
            {
                "multiscales": [
                    {
                        "name": name,
                        "datasets": [{"path": str(item.level)} for item in level_info],
                        "version": "0.1",
                    }
                ]
            }
        )

        channels = (num_channels,) if num_channels > 1 else ()
        for item in level_info:
            zarr.storage.init_array(
                store=store,
                path=str(item.level),
                shape=((item.size.height, item.size.width) + channels),
                chunks=((tile_size.height, tile_size.width) + channels),
                dtype=dtype,
                compressor=None,
            )

        super().__init__(
            name=name,
            num_channels=num_channels,
            tile_size=tile_size,
            scale=scale,
            dtype=dtype,
            store=store,
        )

        self._level_info = level_info

    @property
    def level_info(self) -> Sequence[LevelInfo]:
        return self._level_info


def read_pyramid(store: zarr.storage.StoreLike) -> List[da.Array]:
    """Read an image pyramid from a zarr store.

    Args:
        store: The zarr store to read from.

    Returns:
        An image pyramid as a list of dask arrays.
    """
    # Load the image data from the zarr store, either from a group or an array.
    group = zarr.open(store, "r")
    if isinstance(group, zarr.hierarchy.Group):
        try:
            pyramid = [
                da.from_zarr(store, component=dataset["path"])
                for dataset in group.attrs["multiscales"][0]["datasets"]
            ]
        except KeyError as err:
            raise RuntimeError("Failed to read multiscales from zarr store.") from err
    elif isinstance(group, zarr.hierarchy.Array):
        pyramid = [da.from_zarr(group)]
    else:
        raise RuntimeError("Failed to read from zarr store.")

    if len(pyramid) == 0:
        raise RuntimeError("Failed to read image pyramid.")

    return pyramid


def read_multiscales_data(
    store: zarr.storage.StoreLike,
    *,
    name: str,
    metadata: Dict[str, Any],
    split_rgb: bool = False,
) -> List[LayerDataTuple]:
    """Read (multiscale) image data from a zarr store.

    If the image data has three channels, it is assumed to be an RGB image and a single layer data
    tuple is returned, unless split_rgb is True. If the image data has any other number of
    channels, one layer data tuple per channel is returned.

    Args:
        store: The zarr store to read from.
        name: The base name of the created layer(s).
        metadata: The metadata of the created layer(s).
        split_rgb: If True, a separate layer will be created for each RGB channel.

    Returns:
        A list of layer data tuples.
    """
    # Read the store data.
    pyramid = read_pyramid(store)
    assert len(pyramid) > 0

    # Ensure that the data is three-dimensional.
    pyramid = [da.atleast_3d(pyr_level) for pyr_level in pyramid]

    # Check that the data is valid image data.
    shape_per_lev = [pyr_level.shape for pyr_level in pyramid]
    if any(len(shape) != 3 for shape in shape_per_lev):
        raise RuntimeError("Expected only 3D data in pyramid.")

    # Convert to channels-last format.
    channel_axes = set(np.argmin(shape) for shape in shape_per_lev)
    if len(channel_axes) != 1:
        raise RuntimeError("Different channel axes in pyramid.")
    channel_axis = channel_axes.pop()
    if len(set(shape[channel_axis] for shape in shape_per_lev)) != 1:
        raise RuntimeError("Different number of channels in pyramid.")
    if channel_axis != 2:
        pyramid = [da.moveaxis(pyr_level, channel_axis, -1) for pyr_level in pyramid]
    _, _, num_channels = pyramid[0].shape

    # Remove the channel axis for a one-channel image.
    pyramid = [da.squeeze(pyr_level) for pyr_level in pyramid]

    # Create the layer data for display.
    multiscale = len(pyramid) > 1
    # pylint: disable-next=consider-using-ternary
    multichannel = (num_channels == 3 and split_rgb) or (num_channels not in [1, 3])
    layer_data, layer_names = [], []
    if multichannel:
        for i in range(num_channels):
            layer_data.append([pyr_level[:, :, i] for pyr_level in pyramid])
            layer_names.append(f"{name}_{i}")
    else:
        layer_data.append(pyramid)
        layer_names.append(name)
    if not multiscale:
        layer_data = [pyramid[0] for pyramid in layer_data]

    # Set the layer display properties.
    if multichannel:
        colormaps, blending = list(create_colormaps(layer_names)), "additive"
    else:
        colormaps, blending = [DEFAULT_COLORMAP] * len(layer_data), "translucent"

    return [
        as_layer_data_tuple(
            layer_data=data,
            layer_params=dict(
                name=name,
                multiscale=multiscale,
                metadata=metadata,
                colormap=colormap,
                blending=blending,
            ),
            layer_type="image",
        )
        for data, name, colormap in zip(layer_data, layer_names, colormaps)
    ]
