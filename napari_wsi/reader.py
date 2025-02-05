from collections.abc import Callable, Sequence

from napari_wsi.backends.common import WSIReaderBackend

from .color_transform import ColorSpace
from .common import open_store


def _wsi_reader(
    path: str | Sequence[str], backend: WSIReaderBackend
) -> Callable | None:
    if not isinstance(path, str):
        return None

    store = open_store(path=path, backend=backend, color_space=ColorSpace.sRGB)
    return store.to_layer_data_tuples


def wsi_reader_openslide(path: str | Sequence[str]) -> Callable | None:
    return _wsi_reader(path, backend=WSIReaderBackend.OPENSLIDE)


def wsi_reader_rasterio(path: str | Sequence[str]) -> Callable | None:
    return _wsi_reader(path, backend=WSIReaderBackend.RASTERIO)
