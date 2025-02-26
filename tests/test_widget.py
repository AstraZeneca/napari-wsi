from collections.abc import Callable

import numpy as np
import pytest
from napari.layers import Image

from napari_wsi.widget import WSIReaderWidget

from .conftest import DEFAULT_TEST_CASES, Case


class TestWSIReaderWidget:
    @pytest.mark.parametrize("case", DEFAULT_TEST_CASES, ids=lambda case: case.id)
    def test_read_file(
        self, case: Case, request: pytest.FixtureRequest, make_napari_viewer: Callable
    ) -> None:
        """Test that the reader widget works for the given sample data."""
        viewer = make_napari_viewer()
        path = case.path(request)

        widget = WSIReaderWidget(viewer)
        widget._backend_edit.value = case.backend
        widget._path_edit.value = path
        widget._color_space_edit.value = case.target_color_space

        assert len(viewer.layers) == 0
        widget._on_load_button_clicked()
        assert len(viewer.layers) == 1
        layer = viewer.layers[0]
        assert isinstance(layer, Image)

        assert layer.name
        assert layer.visible
        assert layer.ndim == 3 if layer.multiscale else 2
        assert np.allclose(layer.scale, 1)
        assert layer.multiscale == case.expected_multiscale
        assert layer.rgb == case.expected_rgb

        assert layer.metadata["path"] == str(path)
        assert layer.metadata["resolution"] == case.expected_resolution
        assert layer.metadata["color_space"] == str(case.expected_color_space)
        assert "backend" in layer.metadata

        widget.close()
