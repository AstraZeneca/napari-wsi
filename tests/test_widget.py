from typing import Callable

import pytest
from pytest import FixtureRequest

from napari_wsi.widget import get_wsi_reader_widget

from .conftest import (
    DEFAULT_TEST_CASES,
    TestCase,
    check_image_layers,
    from_layer_data_tuple,
    get_backend,
)


@pytest.mark.parametrize("case", DEFAULT_TEST_CASES, ids=(lambda case: case.id))
def test_wsi_reader_widget(
    case: TestCase, request: FixtureRequest, make_napari_viewer: Callable
):
    """Test that the reader widget works for the given sample data."""

    path = case.get_path(request)

    # pylint: disable-next=no-value-for-parameter
    widget = get_wsi_reader_widget()

    result = widget(
        path=path,
        backend=get_backend(case.expected_reader),
        split_rgb=case.split_rgb,
    )
    assert result is not None

    layers = list(map(from_layer_data_tuple, result))
    assert len(layers) == case.expected_num_layers

    assert check_image_layers(
        layers=layers,
        base_name=path.stem,
        multiscale=case.expected_multiscale,
        rgb=case.expected_rgb,
        viewer=make_napari_viewer(),
    )
