import warnings
from contextlib import suppress
from functools import cached_property
from math import log2
from pathlib import Path

import numpy as np
from PIL.Image import Image
from upath import UPath
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.common import JSON

try:
    from wsidicom import WsiDicom
    from wsidicom.errors import WsiDicomNotFoundError
    from wsidicom.metadata import OpticalPath, WsiMetadata
    from wsidicom.metadata.schema.json import WsiMetadataJsonSchema
except ImportError as err:
    raise ImportError("Please install the optional 'wsidicom' dependency.") from err

from ..color_transform import ColorSpace, ColorTransform
from ..common import PyramidLevel, PyramidLevels, WSIStore


def _get_optical_path(
    metadata: WsiMetadata, identifier: str | None = None
) -> OpticalPath:
    optical_paths = metadata.optical_paths
    if len(optical_paths) == 0:
        raise ValueError("No optical paths in metadata.")

    if identifier is not None:
        for path in optical_paths:
            if path.identifier == identifier:
                return path
        warnings.warn("No optical path with given identifier.")

    return optical_paths[0]


class WSIDicomStore(WSIStore):
    """A class for reading whole-slide images using the `wsidicom` backend."""

    def __init__(
        self,
        path: str | Path | UPath,
        pyramid: int = 0,
        optical_path: str | None = None,
        color_space: ColorSpace = ColorSpace.RAW,
    ) -> None:
        """Initialize a `WSIDicomStore`.

        Args:
            path: The path to the input image directory, or a URL.
            pyramid: An index to select one of multiple image pyramids.
            optical_path: An identifier to select one of multiple optical paths.
            color_space: The target color space.
        """
        self._handle = WsiDicom.open(path)
        self._handle.set_selected_pyramid(pyramid)

        self._optical_path = _get_optical_path(self._handle.metadata, optical_path)

        # We need to read some data to determine the number of channels and dtype.
        sample_image = self._read_region(location=(0, 0), level=0, size=(1, 1))

        levels = PyramidLevels(
            num_channels=len(sample_image.getbands()),
            dtype=np.asarray(sample_image).dtype,
        )
        for level in self._handle.pyramid.levels:
            levels += PyramidLevel(
                # The scale factor is always a power of two.
                factor=2**level.level,
                shape=(level.size.height, level.size.width),
                chunks=(level.tile_size.height, level.tile_size.width),
            )

        super().__init__(
            path=path,
            levels=levels,
            resolution=(self._handle.mpp.height, self._handle.mpp.width),
            color_transform=ColorTransform(
                profile=self._optical_path.icc_profile,
                mode=sample_image.mode,
                color_space=color_space,
            ),
        )

    def _read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image:
        return self._handle.read_region(
            location=location,
            level=level,
            size=size,
            path=self._optical_path.identifier,
        )

    @cached_property
    def typed_metadata(self) -> WsiMetadata:
        return self._handle.metadata

    @cached_property
    def metadata(self) -> dict[str, JSON]:
        slide_metadata = WsiMetadataJsonSchema().dump(self.typed_metadata)
        return slide_metadata | super().metadata

    @cached_property
    def label_image(self) -> np.ndarray | None:
        with suppress(WsiDicomNotFoundError):
            return np.asarray(self._handle.read_label())
        return None

    @cached_property
    def overview_image(self) -> np.ndarray | None:
        with suppress(WsiDicomNotFoundError):
            return np.asarray(self._handle.read_overview())
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
            height, width, num_channels = data.shape
            tile_height, tile_width, _ = data.chunks

            # We need to account for tiles that extend beyond the level dimensions.
            x0, y0 = v * tile_width, u * tile_height
            x1, y1 = min(x0 + tile_width, width), min(y0 + tile_height, height)
            clipped_tile_width, clipped_tile_height = x1 - x0, y1 - y0

            tile = np.full(
                (tile_height, tile_width, num_channels),
                fill_value=data.fill_value,
                dtype=data.dtype,
            )
            tile[:clipped_tile_height, :clipped_tile_width] = np.asarray(
                self.color_transform(
                    self._read_region(
                        location=(x0, y0),
                        level=int(log2(factor)),
                        size=(clipped_tile_width, clipped_tile_height),
                    )
                )
            )

            return prototype.buffer.from_bytes(tile.tobytes())
        return None

    def close(self) -> None:
        self._handle.close()
        return super().close()
