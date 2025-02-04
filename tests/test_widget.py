from collections.abc import Callable

import numpy as np
import pytest
from napari.layers import Image
from pytest import FixtureRequest

from napari_wsi.widget import WSIReaderWidget

from .conftest import (
    DEFAULT_TEST_CASES,
    Case,
)


@pytest.mark.parametrize("case", DEFAULT_TEST_CASES, ids=lambda case: case.id)
def test_wsi_reader_widget(
    case: Case, request: FixtureRequest, make_napari_viewer: Callable
):
    """Test that the reader widget works for the given sample data."""

    viewer = make_napari_viewer()
    path = case.path(request)

    widget = WSIReaderWidget(viewer)
    widget._backend_edit.value = case.backend
    widget._path_edit.value = path
    widget._color_space_edit.value = case.color_space

    assert len(viewer.layers) == 0
    widget._on_load_button_clicked()
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert isinstance(layer, Image)

    assert layer.name
    assert layer.visible
    assert layer.ndim == 3 if layer.multiscale else 2
    assert np.allclose(layer.scale, 1)
    assert layer.multiscale == case.multiscale
    assert layer.rgb == case.rgb

    assert layer.metadata["path"] == str(path)
    assert layer.metadata["resolution"] == case.resolution
    assert layer.metadata["color_space"] == str(case.color_space)

    widget.close()
