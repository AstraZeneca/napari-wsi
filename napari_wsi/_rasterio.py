from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import rasterio
from napari.types import LayerDataTuple
from rasterio import DatasetReader
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

from .multiscales import LevelInfo, MultiScalesStore, read_multiscales_data
from .util import Size, catch_warnings


class RasterioStore(MultiScalesStore):
    def __init__(self, name: str, handle: DatasetReader):
        if handle.count == 0:
            raise ValueError("The image must have at least one channel.")

        factors = handle.overviews(1)
        for band in range(2, handle.count + 1):
            if handle.overviews(band) != factors:
                raise ValueError("All channels must have the same number of overviews.")

        factors = [1] + factors  # The base level has a factor of unity.
        num_levels = len(factors)

        tile_size_per_channel = handle.block_shapes
        if len(set(tile_size_per_channel)) != 1:
            raise ValueError("All channels must have the same tile size.")
        tile_size = tile_size_per_channel[0]

        dtype_per_channel = handle.dtypes
        if len(set(dtype_per_channel)) != 1:
            raise ValueError("All channels must have the same data type.")
        dtype = dtype_per_channel[0]

        def _get_level_size(level) -> Size:
            if level == 0:
                return Size(height=handle.height, width=handle.width)
            with rasterio.open(handle.name, overview_level=(level - 1)) as overview:
                return Size(height=overview.height, width=overview.width)

        level_info = [
            LevelInfo(level=level, factor=factors[level], size=_get_level_size(level))
            for level in range(num_levels)
        ]

        super().__init__(
            name=name,
            num_channels=handle.count,
            tile_size=Size(height=tile_size[0], width=tile_size[1]),
            scale=(1.0, 1.0),
            dtype=np.dtype(dtype),
            level_info=level_info,
        )

        self._handle = handle

    @property
    def handle(self) -> DatasetReader:
        return self._handle

    def __getitem__(self, key: str):
        if key in self._store:
            return self._store[key]

        try:
            level_str, chunk_key = key.split("/")
            chunk_pos = chunk_key.split(".")
            x, y = int(chunk_pos[1]), int(chunk_pos[0])
            level = int(level_str)
            factor = self.level_info[level].factor
            tile = np.transpose(
                self.handle.read(
                    indexes=list(range(1, self.num_channels + 1)),
                    window=Window(
                        int(x * factor * self.tile_size.width),
                        int(y * factor * self.tile_size.height),
                        int(factor * self.tile_size.width),
                        int(factor * self.tile_size.height),
                    ),
                    out_shape=(
                        self.num_channels,
                        self.tile_size.height,
                        self.tile_size.width,
                    ),
                    boundless=True,
                    fill_value=0,
                ),
                axes=[1, 2, 0],
            )
        except Exception as err:
            raise KeyError(key) from err

        return np.array(tile).tobytes()

    def close(self):
        self.handle.close()


def _read_metadata(handle: DatasetReader) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}

    # Set some basic image metadata.
    metadata["file_path"] = handle.name
    metadata["size"] = (handle.height, handle.width)

    return metadata


def read_rasterio(
    path: Union[str, Path], *, split_rgb: bool = False
) -> List[LayerDataTuple]:
    """Read an image using rasterio.

    Args:
        path: The path to the image file.
        split_rgb: If True, a separate layer will be created for each RGB channel.

    Returns:
        A list of layer data tuples.
    """
    with catch_warnings(category=NotGeoreferencedWarning):
        handle = rasterio.open(path)

    store = RasterioStore(name=Path(path).stem, handle=handle)

    return read_multiscales_data(
        store=store,
        name=store.name,
        metadata=_read_metadata(handle),
        split_rgb=split_rgb,
    )
