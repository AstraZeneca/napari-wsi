import itertools
import json
from collections.abc import Callable, Sequence
from unittest.mock import patch

import pytest
from napari.layers import Image, Points, Shapes
from wsidicom.graphical_annotations import (
    AnnotationCategoryCode,
    AnnotationGroup,
    AnnotationTypeCode,
    Geometry,
)

from napari_wsi.backends.wsidicom import WSIDicomStore

from .conftest import TEST_DATA_DCM, TEST_DATA_PATH

TEST_DATA_ANN = TEST_DATA_PATH / "annotations.json"


class MockAnnotations:
    def __init__(self, geometry_type: str) -> None:
        super().__init__()
        with TEST_DATA_ANN.open("r") as geojson_file:
            geojson_data = json.load(geojson_file)
            features = geojson_data["features"]
            self._group = AnnotationGroup.from_geometries(
                geometries=list(
                    itertools.chain(
                        *[
                            Geometry.from_geojson(feature["geometry"])
                            for feature in features
                            if feature["geometry"]["type"] == geometry_type
                        ]
                    )
                ),
                label="Nuclei",
                category_code=AnnotationCategoryCode("Tissue"),
                type_code=AnnotationTypeCode("Nucleus"),
            )

    @property
    def coordinate_type(self) -> str:
        return "image"

    @property
    def groups(self) -> Sequence[AnnotationGroup]:
        return [self._group]


@pytest.mark.parametrize("spatial_transform", [True, False], ids=["pixel", "slide"])
class TestAnnotations:
    @pytest.mark.parametrize("geometry_type", ["Polygon", "LineString"])
    def test_shape_annotations(
        self, geometry_type: str, spatial_transform: bool, make_napari_viewer: Callable
    ) -> None:
        viewer = make_napari_viewer()
        store = WSIDicomStore(TEST_DATA_DCM)
        with patch.object(
            store._handle, "_annotations", [MockAnnotations(geometry_type)]
        ):
            layers = store.to_viewer(
                viewer,
                layer_type=("image", "shapes"),
                spatial_transform=spatial_transform,
            )
        assert len(layers) == 2
        assert isinstance(layers[0], Image)
        assert isinstance(layers[1], Shapes)
        shapes = layers[1]
        assert len(shapes.data) > 0

    def test_point_annotations(
        self, spatial_transform: bool, make_napari_viewer: Callable
    ) -> None:
        viewer = make_napari_viewer()
        store = WSIDicomStore(TEST_DATA_DCM)
        with patch.object(store._handle, "_annotations", [MockAnnotations("Point")]):
            layers = store.to_viewer(
                viewer,
                layer_type=("image", "points"),
                spatial_transform=spatial_transform,
            )
        assert len(layers) == 2
        assert isinstance(layers[0], Image)
        assert isinstance(layers[1], Points)
        points = layers[1]
        assert len(points.data) > 0
