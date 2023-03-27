from enum import Enum
from pathlib import Path
from typing import List

from magicgui import magic_factory
from napari.types import LayerDataTuple

from . import read_rasterio, read_tifffile


class WSIReaderBackend(Enum):
    tifffile = "tifffile"
    rasterio = "rasterio"


DEFAULT_BACKEND = WSIReaderBackend.tifffile


@magic_factory(
    call_button="Load",
    path={"label": "Path", "tooltip": "The file path."},
    backend={"label": "Backend", "tooltip": "The backend for reading WSI data."},
    series={"label": "Series", "tooltip": "The image series to read."},
    split_rgb={"label": "Split RGB", "tooltip": "If set, split 3-channel images."},
)
def get_wsi_reader_widget(
    path: Path,
    backend: WSIReaderBackend = DEFAULT_BACKEND,
    series: int = 0,
    split_rgb: bool = False,
) -> List[LayerDataTuple]:
    if not path.is_file():
        raise FileNotFoundError(path)

    if backend == WSIReaderBackend.tifffile:
        return read_tifffile(path, series=series, split_rgb=split_rgb)
    if backend == WSIReaderBackend.rasterio:
        return read_rasterio(path, split_rgb=split_rgb)

    raise ValueError(f"Invalid backend {backend}.")
