import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from functools import cached_property
from math import log2
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from PIL.Image import Image
from upath import UPath
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.common import JSON

try:
    import pandas as pd
    from colorspacious import cspace_converter
    from shapely import LineString as ShapelyPolyline
    from shapely import Polygon as ShapelyPolygon
    from wsidicom import WsiDicom, WsiDicomWebClient
    from wsidicom.errors import WsiDicomNotFoundError
    from wsidicom.graphical_annotations import (
        Annotation,
        AnnotationGroup,
        LabColor,
        Point,
        Polygon,
        Polyline,
    )
    from wsidicom.metadata import OpticalPath, WsiMetadata
    from wsidicom.metadata.schema.json import WsiMetadataJsonSchema
except ImportError as err:
    raise ImportError("Please install the optional 'wsidicom' dependency.") from err

from ..color_transform import ColorSpace, ColorTransform
from ..common import PyramidLevel, PyramidLevels, WSIStore

if TYPE_CHECKING:
    import napari
    from shapely.geometry.base import BaseGeometry as ShapelyGeometry


@dataclass(frozen=True)
class AnnotationData:
    data: np.ndarray
    shape_type: Literal["polygon", "path", "point"]


LAB_TO_RGB_CONVERTER = cspace_converter("CIELab", "sRGB1")


def _lab_to_rgb(color: LabColor | None) -> np.ndarray:
    if color is None:
        return np.ones(3)
    return np.clip(LAB_TO_RGB_CONVERTER([color.l, color.a, color.b]), 0, 1)


def _get_group_features(group: AnnotationGroup) -> dict[str, str | float]:
    return {
        "category": group.category_code.meaning,
        "type": group.type_code.meaning,
    }


def _get_measurement_features(annotation: Annotation) -> dict[str, str | float]:
    measurements = {}
    for measurement in annotation.measurements:
        name = measurement.code.meaning
        measurements[name] = measurement.value
        measurements[f"{name}_unit"] = measurement.unit.value
    return measurements


def _validate_annotation(
    annotation: Annotation, tol: float = 0.0
) -> AnnotationData | None:
    coords = np.array(annotation.geometry.to_coords())[:, ::-1]

    # We expect DICOM annotations to be valid geometries without
    # holes, but let's check to avoid errors on layer creation.
    if isinstance(annotation.geometry, Point):
        assert len(coords) == 1
        return AnnotationData(coords[0], shape_type="point")
    if isinstance(annotation.geometry, Polygon):
        shape: ShapelyGeometry = ShapelyPolygon(coords)
        shape_type: Literal["polygon", "path"] = "polygon"
    elif isinstance(annotation.geometry, Polyline):
        shape = ShapelyPolyline(coords)
        shape_type = "path"
    else:
        raise TypeError("Unsupported geometry type.")
    if not shape.is_valid:
        return None
    if len(getattr(shape, "interiors", [])) > 0:
        return None
    if tol > 0:
        shape = shape.simplify(tol)
    return AnnotationData(coords, shape_type=shape_type)


def _get_shape_annotations(
    groups: Iterable[AnnotationGroup], tol: float = 0.0
) -> Iterator["napari.types.LayerDataTuple"]:
    for group in groups:
        if group.geometry_type == Point:
            continue
        if group.geometry_type not in {Polygon, Polyline}:
            warnings.warn(f"Skipping unsupported geometry type: {group.geometry_type}.")
            continue

        group_color = _lab_to_rgb(group.color)
        group_features = _get_group_features(group)

        data: dict[str, Any] = defaultdict(list)
        for annotation in group:
            validated_annotation = _validate_annotation(annotation, tol=tol)
            if validated_annotation is None:
                continue
            assert validated_annotation.shape_type in {"polygon", "path"}
            data["data"].append(validated_annotation.data)
            data["shape_type"].append(validated_annotation.shape_type)
            data["face_color"].append(group_color)
            measurement_features = _get_measurement_features(annotation)
            data["features"].append(measurement_features | group_features)

        if "data" in data:
            shapes_data = data.pop("data")
            if len(shapes_data) != len(group):
                warnings.warn(f"Skipping invalid geometries in group: {group.label}.")
            features = pd.DataFrame(data.pop("features"))
            yield (
                shapes_data,
                {
                    "name": group.label,
                    **data,
                    "features": features,
                },
                "shapes",
            )


def _get_point_annotations(
    groups: Iterable[AnnotationGroup],
) -> Iterator["napari.types.LayerDataTuple"]:
    for group in groups:
        if group.geometry_type != Point:
            continue

        group_color = _lab_to_rgb(group.color)
        group_features = _get_group_features(group)

        data: dict[str, Any] = defaultdict(list)
        for annotation in group:
            validated_annotation = _validate_annotation(annotation)
            assert validated_annotation is not None
            assert validated_annotation.shape_type == "point"
            data["data"].append(validated_annotation.data)
            data["face_color"].append(group_color)
            measurement_features = _get_measurement_features(annotation)
            data["features"].append(measurement_features | group_features)

        if "data" in data:
            points_data = data.pop("data")
            features = pd.DataFrame(data.pop("features"))
            yield (
                points_data,
                {
                    "name": group.label,
                    **data,
                    "features": features,
                },
                "points",
            )


def _get_optical_path(
    metadata: WsiMetadata, identifier: str | None = None
) -> OpticalPath:
    optical_paths = metadata.optical_paths
    if len(optical_paths) == 0:
        raise ValueError("No optical paths in metadata.")

    if identifier is not None:
        for path in optical_paths:
            if path.identifier == identifier:
                return path
        warnings.warn("No optical path with given identifier.")

    return optical_paths[0]


class WSIDicomStore(WSIStore):
    """A class for reading whole-slide images using the `wsidicom` backend."""

    def __init__(
        self,
        path: str | Path | UPath | None = None,
        *,
        client: WsiDicomWebClient | None = None,
        pyramid: int = 0,
        optical_path: str | None = None,
        color_space: str | ColorSpace = ColorSpace.RAW,
        **kwargs,
    ) -> None:
        """Initialize a `WSIDicomStore`.

        This requires either a `path` or a DICOMWeb `client`. All additional keyword
        arguments are passed to `WsiDicom.open` or `WsiDicom.open_web`, respectively.

        Args:
            path: A path to the input image directory, or a URL.
            client: A previously initialized DICOMWeb client. A `study_uid` and
                `series_uids` must be provided as additional keyword arguments.
            pyramid: An index to select one of multiple image pyramids.
            optical_path: An identifier to select one of multiple optical paths.
            color_space: The target color space.
            kwargs: All additional keyword arguments are passed to the `WsiDicom.open`
                or `WsiDicom.open_web` method.

        """
        if not isinstance(color_space, ColorSpace):
            color_space = ColorSpace(color_space)

        if path is not None and client is None:
            path = UPath(path)
            self._handle = WsiDicom.open(path, **kwargs)
        elif client is not None and path is None:
            self._handle = WsiDicom.open_web(client, **kwargs)
        else:
            raise ValueError("Need either 'path' or 'client'.")
        self._handle.set_selected_pyramid(pyramid)

        self._optical_path = _get_optical_path(self._handle.metadata, optical_path)

        # We need to read some data to determine the number of channels and dtype.
        sample_image = self._read_region(location=(0, 0), level=0, size=(1, 1))

        levels = PyramidLevels(
            num_channels=len(sample_image.getbands()),
            dtype=np.asarray(sample_image).dtype,
        )
        for level in self._handle.pyramid.levels:
            levels += PyramidLevel(
                # The scale factor is always a power of two.
                factor=2**level.level,
                shape=(level.size.height, level.size.width),
                chunks=(level.tile_size.height, level.tile_size.width),
            )

        self._color_transform = ColorTransform(
            profile=self._optical_path.icc_profile,
            mode=sample_image.mode,
            color_space=color_space,
        )

        super().__init__(path=path, levels=levels)

    def __repr__(self) -> str:
        return f"WSIDicomStore({self.name})"

    def _read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image:
        return self._handle.read_region(
            location=location,
            level=level,
            size=size,
            path=self._optical_path.identifier,
        )

    @property
    def resolution(self) -> tuple[float, float]:
        return (self._handle.mpp.height, self._handle.mpp.width)

    @property
    def units(self) -> str:
        return "micrometer"

    @property
    def spatial_transform(self) -> np.ndarray:
        matrix = np.identity(3)
        ics = self._handle.metadata.image.image_coordinate_system
        if ics is None:
            return matrix
        scale_y, scale_x = self.resolution  # mu/px
        offset_y, offset_x = ics.origin.y * 1000, ics.origin.x * 1000  # mu
        orientation = ics.orientation.values  # noqa: PD011
        matrix[0] = (orientation[1] * scale_y, orientation[4] * scale_x, offset_y)
        matrix[1] = (orientation[0] * scale_y, orientation[3] * scale_x, offset_x)
        return matrix

    @property
    def color_transform(self) -> ColorTransform:
        return self._color_transform

    @cached_property
    def typed_metadata(self) -> WsiMetadata:
        return self._handle.metadata

    @cached_property
    def metadata(self) -> dict[str, JSON]:
        backend_metadata = WsiMetadataJsonSchema().dump(self.typed_metadata)
        return {**super().metadata, "backend": backend_metadata}

    @cached_property
    def label_image(self) -> np.ndarray | None:
        with suppress(WsiDicomNotFoundError):
            return np.asarray(self._handle.read_label())
        return None

    @cached_property
    def overview_image(self) -> np.ndarray | None:
        with suppress(WsiDicomNotFoundError):
            return np.asarray(self._handle.read_overview())
        return None

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        assert self._is_open
        assert byte_range is None
        if (match := self._parse_key(key)) is not None:
            i, u, v = match
            factor, data = self._pyramid[i]
            height, width, num_channels = data.shape
            tile_height, tile_width, _ = data.chunks

            # We need to account for tiles that extend beyond the level dimensions.
            x0, y0 = v * tile_width, u * tile_height
            x1, y1 = min(x0 + tile_width, width), min(y0 + tile_height, height)
            clipped_tile_width, clipped_tile_height = x1 - x0, y1 - y0

            tile = np.full(
                (tile_height, tile_width, num_channels),
                fill_value=data.fill_value,
                dtype=data.dtype,
            )
            tile[:clipped_tile_height, :clipped_tile_width] = np.asarray(
                self.color_transform(
                    self._read_region(
                        location=(x0, y0),
                        level=int(log2(factor)),
                        size=(clipped_tile_width, clipped_tile_height),
                    )
                )
            )

            return prototype.buffer.from_bytes(tile.tobytes())
        return None

    def to_layer_data_tuples(
        self,
        *,
        layer_type: str | tuple[str, ...] = "image",
        tol: float = 0.0,
        **kwargs,
    ) -> list["napari.types.LayerDataTuple"]:
        """Convert to a napari layer data tuple.

        All additional keword arguments are passed to the `to_layer_data_tuples` method
        of the base class.

        Args:
            layer_type: The type of the layer data tuples to return. This can be one or
                any subset of ("image", "shapes", "points").
            tol: A tolerance value used to simplify all shape annotations. If this is
                not greater zero, no simplification is performed.
            kwargs: All additional keword arguments are passed to the
                `to_layer_data_tuples` method of the base class.

        Returns:
            A list containing a napari layer data tuple of the given types.

        """
        if isinstance(layer_type, str):
            layer_type = (layer_type,)

        items = []
        if "image" in layer_type:
            items.extend(super().to_layer_data_tuples(**kwargs))
        for annotations in self._handle.annotations:
            if annotations.coordinate_type != "image":
                continue
            if "shapes" in layer_type:
                items.extend(list(_get_shape_annotations(annotations.groups, tol=tol)))
            if "points" in layer_type:
                items.extend(list(_get_point_annotations(annotations.groups)))
        return items

    def close(self) -> None:
        self._handle.close()
        return super().close()
