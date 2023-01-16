from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest

from napari_wsi.util import get_isotropic_resolution, parse_color


@pytest.mark.parametrize(
    "x, y",
    [(1.0, 1.0), (1.0, 2.0)],
    ids=["isotropic", "anisotropic"],
)
def test_get_isotropic_resolution(x: float, y: float):
    with nullcontext() if x == y else pytest.warns(UserWarning):  # type: ignore
        assert np.isclose(get_isotropic_resolution(x, y), np.sqrt(np.abs(x * y)))


@pytest.mark.parametrize(
    "value",
    ["0x0000FF", 4278190335, "red", (255, 0, 0), (255, 0, 0, 255)],
    ids=["hex", "int", "str", "rgb", "rgba"],
)
def test_parse_color(value: Any):
    color = parse_color(value)
    assert color == (255, 0, 0, 255)
