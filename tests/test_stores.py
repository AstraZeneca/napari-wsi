from collections.abc import Callable

import pytest
from napari.layers import Image
from pytest import FixtureRequest

from napari_wsi.common import open_store

from .conftest import DEFAULT_TEST_CASES, Case


@pytest.mark.parametrize("spatial_transform", [True, False], ids=["pixel", "slide"])
@pytest.mark.parametrize("case", DEFAULT_TEST_CASES, ids=lambda case: case.id)
def test_stores(
    case: Case,
    spatial_transform: bool,
    request: FixtureRequest,
    make_napari_viewer: Callable,
):
    """Test that the backend classes work for the given sample data."""

    viewer = make_napari_viewer()
    path = case.path(request)

    store = open_store(
        path=path, backend=case.backend, color_space=case.target_color_space
    )
    (layer,) = store.to_viewer(viewer, spatial_transform=spatial_transform)
    assert isinstance(layer, Image)
