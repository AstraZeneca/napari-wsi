from typing import Callable, Optional, Sequence, Union

from tifffile import TiffFile

from . import read_rasterio, read_tifffile


def get_wsi_reader(path: Union[str, Sequence[str]]) -> Optional[Callable]:
    if not isinstance(path, str):
        return None

    handle = TiffFile(path)

    page = handle.series[0].pages[0]
    if "GDAL_METADATA" in page.tags:
        return read_rasterio

    return read_tifffile
