from collections.abc import Callable, Sequence

from tifffile import TiffFile
from tifffile.tifffile import TiffPage

from . import read_rasterio, read_tifffile


def get_wsi_reader(path: str | Sequence[str]) -> Callable | None:
    if not isinstance(path, str):
        return None

    handle = TiffFile(path)

    page = handle.series[0].pages[0]
    assert isinstance(page, TiffPage)
    if "GDAL_METADATA" in page.tags:
        return read_rasterio

    return read_tifffile
