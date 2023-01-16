from importlib.metadata import version

from ._rasterio import read_rasterio
from ._tifffile import read_tifffile

__all__ = ["read_rasterio", "read_tifffile"]
__version__ = version(__package__)
