from contextlib import suppress
from functools import cached_property
from pathlib import Path

import numpy as np
from upath import UPath
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.common import JSON

try:
    from openslide import (
        PROPERTY_NAME_MPP_X,
        PROPERTY_NAME_MPP_Y,
        OpenSlide,
    )
except ImportError as err:
    raise ImportError("Please install the optional 'openslide' dependency.") from err

from ..color_transform import ColorSpace, ColorTransform
from ..common import PyramidLevel, PyramidLevels, WSIStore


def _get_shape(handle: OpenSlide, level: int) -> tuple[int, int]:
    width, height = handle.level_dimensions[level]
    return height, width


def _get_chunks(handle: OpenSlide, level: int) -> tuple[int, int]:
    with suppress(KeyError):
        tile_width = int(handle.properties[f"openslide.level[{level}].tile-width"])
        tile_height = int(handle.properties[f"openslide.level[{level}].tile-height"])
        return tile_height, tile_width
    return (512, 512)


class OpenSlideStore(WSIStore):
    """A class for reading whole-slide images using the `openslide` backend."""

    def __init__(
        self,
        path: str | Path | UPath,
        *,
        color_space: str | ColorSpace = ColorSpace.RAW,
    ) -> None:
        """Initialize an `OpenSlideStore`.

        Args:
            path: The path to the input image file.
            color_space: The target color space.

        """
        if not isinstance(color_space, ColorSpace):
            color_space = ColorSpace(color_space)

        path = UPath(path)
        self._handle = OpenSlide(path)

        # We need to read some data to determine the number of channels and dtype.
        sample_image = self._handle.read_region(location=(0, 0), level=0, size=(1, 1))

        levels = PyramidLevels(
            num_channels=len(sample_image.getbands()),
            dtype=np.asarray(sample_image).dtype,
        )
        for i in range(self._handle.level_count):
            levels += PyramidLevel(
                factor=round(self._handle.level_downsamples[i]),
                shape=_get_shape(self._handle, level=i),
                chunks=_get_chunks(self._handle, level=i),
            )

        self._color_transform = ColorTransform(
            profile=self._handle.color_profile,
            mode=sample_image.mode,
            color_space=color_space,
        )

        super().__init__(path=path, levels=levels)

    def __repr__(self) -> str:
        return f"OpenSlideStore({self.name})"

    @property
    def resolution(self) -> tuple[float, float] | None:
        with suppress(KeyError):
            mpp_x = float(self._handle.properties[PROPERTY_NAME_MPP_X])
            mpp_y = float(self._handle.properties[PROPERTY_NAME_MPP_Y])
            return mpp_y, mpp_x
        return None

    @property
    def units(self) -> str:
        return "micrometer"

    @property
    def spatial_transform(self) -> np.ndarray:
        matrix = np.identity(3)
        if self.resolution is None:
            return matrix
        matrix[[0, 1], [0, 1]] = self.resolution
        return matrix

    @property
    def color_transform(self) -> ColorTransform:
        return self._color_transform

    @cached_property
    def metadata(self) -> dict[str, JSON]:
        backend_metadata = dict(self._handle.properties)
        return {**super().metadata, "backend": backend_metadata}

    @cached_property
    def label_image(self) -> np.ndarray | None:
        if "label" in self._handle.associated_images:
            return np.asarray(self._handle.associated_images["label"])
        return None

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
            tile_height, tile_width, _ = data.chunks

            tile = np.asarray(
                self.color_transform(
                    self._handle.read_region(
                        location=(factor * v * tile_width, factor * u * tile_height),
                        level=i,
                        # We need to account for tiles that extend beyond the level
                        # dimensions, since openslide automatically returns zeros.
                        size=(tile_width, tile_height),
                    )
                )
            )

            return prototype.buffer.from_bytes(tile.tobytes())
        return None

    def close(self) -> None:
        self._handle.close()
        return super().close()
