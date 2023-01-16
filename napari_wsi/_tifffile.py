from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from napari.types import LayerDataTuple
from tifffile import TiffFile
from tifffile.tifffile import svs_description_metadata

from .multiscales import read_multiscales_data
from .util import get_isotropic_resolution


def _read_metadata(handle: TiffFile) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}

    tags = handle.pages[0].tags

    # Set some basic image metadata.
    metadata["file_path"] = handle.filehandle.path
    metadata["size"] = (tags["ImageLength"].value, tags["ImageWidth"].value)

    # Collect additional metadata.
    if handle.is_svs:
        svs_metadata = svs_description_metadata(handle.pages[0].description)
        if "MPP" in svs_metadata:
            metadata["resolution"] = float(svs_metadata["MPP"]) * 1e-6
    else:
        if all(key in tags for key in ["XResolution", "YResolution", "ResolutionUnit"]):
            x_resolution = np.divide(*tags["XResolution"].value[::-1])
            y_resolution = np.divide(*tags["YResolution"].value[::-1])
            resolution = get_isotropic_resolution(x_resolution, y_resolution)
            resolution_unit = tags["ResolutionUnit"].value
            if resolution_unit.value == 2:
                assert resolution_unit.name == "INCH"
                metadata["resolution"] = resolution * 2.54e-2
            elif resolution_unit.value == 3:
                assert resolution_unit.name == "CENTIMETER"
                metadata["resolution"] = resolution * 1e-2

    return metadata


def read_tifffile(
    path: Union[str, Path], *, split_rgb: bool = False
) -> List[LayerDataTuple]:
    """Read an image using tifffile.

    Args:
        path: The path to the image file.
        split_rgb: If True, a separate layer will be created for each RGB channel.

    Returns:
        A list of layer data tuples.
    """
    handle = TiffFile(path)

    return read_multiscales_data(
        store=handle.aszarr(),
        name=Path(handle.filename).stem,
        metadata=_read_metadata(handle),
        split_rgb=split_rgb,
    )
