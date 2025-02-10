from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING

from napari_wsi.backends.common import WSIReaderBackend

from .color_transform import ColorSpace
from .common import open_store

if TYPE_CHECKING:
    import napari


def _wsi_reader(
    path: str, backend: WSIReaderBackend
) -> list["napari.types.LayerDataTuple"]:
    store = open_store(path=path, backend=backend, color_space=ColorSpace.sRGB)
    return store.to_layer_data_tuples()


def wsi_reader_openslide(path: str | Sequence[str]) -> Callable | None:
    if not isinstance(path, str):
        return None

    return partial(_wsi_reader, backend=WSIReaderBackend.OPENSLIDE)


def wsi_reader_rasterio(path: str | Sequence[str]) -> Callable | None:
    if not isinstance(path, str):
        return None

    return partial(_wsi_reader, backend=WSIReaderBackend.RASTERIO)
