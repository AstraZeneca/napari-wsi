from functools import cached_property
from pathlib import Path
from warnings import catch_warnings

import numpy as np
from upath import UPath
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.common import JSON

try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    from rasterio.windows import Window
except ImportError as err:
    raise ImportError("Please install the optional 'rasterio' dependency.") from err

from ..common import PyramidLevel, PyramidLevels, WSIStore


class RasterioStore(WSIStore):
    """A class for reading whole-slide images using the `rasterio` backend."""

    def __init__(self, path: str | Path | UPath) -> None:
        """Initialize a `RasterioStore`.

        Args:
            path: The path to the input image file.

        """
        with catch_warnings(category=NotGeoreferencedWarning, action="ignore"):
            path = UPath(path)
            self._handle = rasterio.open(str(path))
            num_channels = self._handle.count

            factors_per_channel = [
                (1, *self._handle.overviews(channel))
                for channel in range(1, num_channels + 1)
            ]
            if len(set(factors_per_channel)) != 1:
                raise ValueError("Different number of overviews per channel.")
            factors = factors_per_channel[0]

            chunks_per_channel = self._handle.block_shapes
            if len(set(chunks_per_channel)) != 1:
                raise ValueError("Different tile sizes per channel.")
            chunks = chunks_per_channel[0]

            dtype_per_channel = self._handle.dtypes
            if len(set(dtype_per_channel)) != 1:
                raise ValueError("Different data types per channel.")
            dtype = dtype_per_channel[0]

            levels = PyramidLevels(num_channels=num_channels, dtype=dtype)
            for channel, factor in enumerate(factors):
                if channel == 0:
                    shape = self._handle.height, self._handle.width
                else:
                    with rasterio.open(path, overview_level=channel - 1) as overview:
                        shape = overview.height, overview.width
                levels += PyramidLevel(factor=factor, shape=shape, chunks=chunks)

        super().__init__(path=path, levels=levels)

    def __repr__(self) -> str:
        return f"RasterioStore({self.name})"

    @property
    def spatial_transform(self) -> np.ndarray:
        transform = self._handle.transform
        return np.array(
            [
                [transform.e, transform.d, transform.f],
                [transform.b, transform.a, transform.c],
                [0.0, 0.0, 1.0],
            ]
        )

    @cached_property
    def metadata(self) -> dict[str, JSON]:
        # We're not providing any backend-specific metadata right now.
        return {**super().metadata, "backend": {}}

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        assert self._is_open
        assert byte_range is None
        if (match := self._parse_key(key)) is not None:
            i, u, v = match
            factor, data = self._pyramid[i]
            _, _, num_channels = data.shape
            tile_height, tile_width, _ = data.chunks

            tile = np.transpose(
                self._handle.read(
                    window=Window(
                        factor * v * tile_width,
                        factor * u * tile_height,
                        factor * tile_width,
                        factor * tile_height,
                    ),
                    out_shape=(num_channels, tile_height, tile_width),
                    boundless=True,
                    fill_value=data.fill_value,
                ),
                axes=[1, 2, 0],
            )

            return prototype.buffer.from_bytes(tile.tobytes())
        return None

    def close(self) -> None:
        self._handle.close()
        return super().close()
