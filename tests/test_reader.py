from collections.abc import Callable

import pytest
from napari.layers import Image

from napari_wsi.backends.common import WSIReaderBackend
from napari_wsi.reader import wsi_reader_openslide, wsi_reader_rasterio

from .conftest import DEFAULT_TEST_CASES, TEST_DATA_GTIF, TEST_DATA_SVS, Case


class TestWSIReader:
    @pytest.mark.parametrize("case", DEFAULT_TEST_CASES, ids=lambda case: case.id)
    def test_read_single_file(
        self, case: Case, request: pytest.FixtureRequest, make_napari_viewer: Callable
    ) -> None:
        """Test that the napari reader works for the given sample data."""
        if case.backend == WSIReaderBackend.OPENSLIDE:
            get_wsi_reader = wsi_reader_openslide
        elif case.backend == WSIReaderBackend.RASTERIO:
            get_wsi_reader = wsi_reader_rasterio
        else:
            return

        viewer = make_napari_viewer()
        path = case.path(request)

        reader = get_wsi_reader(str(path))
        assert callable(reader)

        items = reader(str(path))
        assert len(items) == 1
        layer_data, layer_params, layer_type = items[0]
        add_layer = getattr(viewer, f"add_{layer_type}")
        layer = add_layer(layer_data, **layer_params)
        assert isinstance(layer, Image)

    @pytest.mark.parametrize(
        "get_wsi_reader",
        [wsi_reader_openslide, wsi_reader_rasterio],
        ids=["openslide", "rasterio"],
    )
    def test_read_multiple_files(self, get_wsi_reader: Callable) -> None:
        """Test that the napari readers only accept single file paths."""
        reader = get_wsi_reader([str(TEST_DATA_SVS), str(TEST_DATA_GTIF)])
        assert reader is None
