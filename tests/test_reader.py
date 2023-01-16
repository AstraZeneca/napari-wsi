from typing import Callable

import pytest
from pytest import FixtureRequest

from napari_wsi.reader import get_wsi_reader

from .conftest import (
    DEFAULT_TEST_CASES,
    TEST_DATA_GTIF,
    TEST_DATA_SVS,
    TestCase,
    check_image_layers,
    from_layer_data_tuple,
)


@pytest.mark.parametrize("case", DEFAULT_TEST_CASES, ids=(lambda case: case.id))
def test_wsi_reader(
    case: TestCase, request: FixtureRequest, make_napari_viewer: Callable
):
    """Test that the napari reader works for the given sample data."""

    path = case.get_path(request)

    reader = get_wsi_reader(str(path))
    assert callable(reader)
    assert reader == case.expected_reader

    result = reader(str(path), split_rgb=case.split_rgb)

    layers = list(map(from_layer_data_tuple, result))
    assert len(layers) == case.expected_num_layers

    assert check_image_layers(
        layers=layers,
        base_name=path.stem,
        multiscale=case.expected_multiscale,
        rgb=case.expected_rgb,
        viewer=make_napari_viewer(),
    )


def test_wsi_reader_pass():
    """Test that the napari reader only accepts single file paths."""

    reader = get_wsi_reader([str(TEST_DATA_SVS), str(TEST_DATA_GTIF)])
    assert reader is None
