import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, Generator, Optional, Sequence, Tuple, Type, Union, cast

import matplotlib
import numpy as np
from napari.types import LayerDataTuple
from napari.utils import Colormap
from napari.utils.colormaps import AVAILABLE_COLORMAPS

DEFAULT_COLORMAP = AVAILABLE_COLORMAPS["gray"]

ColorType = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Size:
    height: int
    width: int


def as_layer_data_tuple(
    layer_data: Any, layer_params: Optional[Dict[str, Any]], layer_type: str
) -> LayerDataTuple:
    """Create a napari layer data tuple from the given arguments."""

    return cast(LayerDataTuple, (layer_data, layer_params or {}, layer_type))


@contextmanager
def catch_warnings(category: Type[Warning]):
    with warnings.catch_warnings(record=True) as caught_warnings:
        try:
            yield
        finally:
            for warning in caught_warnings:
                if warning.category == category:
                    continue
                warnings.warn(message=warning.message, category=warning.category)


def create_colormap(color: Union[str, int, Sequence[int]], name: str) -> Colormap:
    """Create a napari alpha-blending color map for the given color."""

    color_end = np.array(parse_color(color)) / 255
    color_start = color_end.copy()
    color_start[-1] = 0
    return Colormap([color_start, color_end], name=name)


def create_colormaps(
    layer_names: Sequence[str], cmap: str = "tab10"
) -> Generator[Colormap, None, None]:
    """Create napari color maps for the given layers."""

    if not len(layer_names) == len(set(layer_names)):
        raise RuntimeError("The layer names are not unique.")

    colors = (255 * np.array(matplotlib.colormaps[cmap].colors)).astype(np.uint8)
    iter_colors = cycle(colors)
    for layer_name in layer_names:
        # pylint: disable-next=stop-iteration-return
        yield create_colormap(next(iter_colors), layer_name)


def get_isotropic_resolution(resolution_x: float, resolution_y: float) -> float:
    """Compute the geometric mean of the given resolution values."""

    if np.isclose(resolution_x, resolution_y):
        return resolution_x

    warnings.warn("The resolution of the slide is different along the X and Y axes.")

    return np.sqrt(np.abs(resolution_x * resolution_y))


def parse_color(color: Union[str, int, Sequence[int]]) -> ColorType:
    """Parse the given input as a color, returning an RGBA tuple."""

    if isinstance(color, int):
        return _parse_int_color(color)

    if isinstance(color, str) and re.match(r"^0x[0-9a-fA-F]{6}$", color):
        return _parse_hex_color(color)

    if isinstance(color, str):
        return _parse_str_color(color)

    if len(color) == 3:
        return tuple(color) + (255,)  # type: ignore

    assert len(color) == 4
    return tuple(color)  # type: ignore


def _parse_int_color(color: int) -> ColorType:
    rgba_color = np.array([color], dtype=np.uint32).view(np.uint8)
    assert rgba_color.shape == (4,)
    return tuple(rgba_color)  # type: ignore


def _parse_hex_color(color: str) -> ColorType:
    return _parse_int_color(int(color, 16) + 0xFF000000)


def _parse_str_color(color: str) -> ColorType:
    mpl_color = matplotlib.colors.to_rgba(color)
    return tuple(int(255 * value) for value in mpl_color)  # type: ignore
