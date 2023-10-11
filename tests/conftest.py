from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import rasterio
import skimage.data
import skimage.transform
import skimage.util
import tifffile
from attr import dataclass
from napari.layers import Image, Labels, Layer, Points, Shapes
from napari.types import LayerDataTuple
from napari.viewer import Viewer
from pytest import FixtureRequest, TempPathFactory
from rasterio.enums import Resampling
from tifffile import COMPRESSION

from napari_wsi import read_rasterio, read_tifffile
from napari_wsi.widget import DEFAULT_BACKEND, WSIReaderBackend

TEST_DATA_PATH = Path("tests") / "data"

# https://github.com/rasterio/rasterio/blob/72bdbc05b4670043fb81413e6ee718f06617b86c/tests/data/RGB.byte.tif
# https://creativecommons.org/publicdomain/zero/1.0/
TEST_DATA_GTIF = TEST_DATA_PATH / "RGB.byte.tif"

# https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs
# https://creativecommons.org/publicdomain/zero/1.0/
TEST_DATA_SVS = TEST_DATA_PATH / "CMU-1-Small-Region.svs"


@dataclass(frozen=True)
# pylint: disable-next=too-many-instance-attributes
class TestCase:
    id: str
    expected_reader: Callable
    expected_num_layers: int
    expected_multiscale: bool
    expected_rgb: bool
    file_path: Path | None = None
    file_fixture: str | None = None
    split_rgb: bool = False

    def get_path(self, request: FixtureRequest) -> Path:
        path = self.file_path or request.getfixturevalue(self.file_fixture)  # type: ignore
        assert isinstance(path, Path) and path.is_file()
        return path


DEFAULT_TEST_CASES = [
    TestCase(
        id="svs-default",
        file_path=TEST_DATA_SVS,
        expected_reader=read_tifffile,
        expected_num_layers=1,
        expected_multiscale=False,
        expected_rgb=True,
    ),
    TestCase(
        id="gtiff-default",
        file_path=TEST_DATA_GTIF,
        expected_reader=read_rasterio,
        expected_num_layers=1,
        expected_multiscale=False,
        expected_rgb=True,
    ),
    TestCase(
        id="svs-split-rgb",
        file_path=TEST_DATA_SVS,
        split_rgb=True,
        expected_reader=read_tifffile,
        expected_num_layers=3,
        expected_multiscale=False,
        expected_rgb=False,
    ),
    TestCase(
        id="gtiff-split-rgb",
        file_path=TEST_DATA_GTIF,
        split_rgb=True,
        expected_reader=read_rasterio,
        expected_num_layers=3,
        expected_multiscale=False,
        expected_rgb=False,
    ),
    TestCase(
        id="ome-pyramid",
        file_fixture="tmp_data_ome",
        expected_reader=read_tifffile,
        expected_num_layers=1,
        expected_multiscale=True,
        expected_rgb=False,
    ),
    TestCase(
        id="gtiff-pyramid",
        file_fixture="tmp_data_gtiff",
        expected_reader=read_rasterio,
        expected_num_layers=1,
        expected_multiscale=True,
        expected_rgb=False,
    ),
]


def check_image_layers(
    layers: list[Layer],
    base_name: str | None = None,
    viewer: Viewer | None = None,
    **kwargs,
) -> bool:
    if len(layers) == 1:
        layer = layers[0]
        if viewer is not None:
            layer = viewer.add_layer(layer)
        return check_image_layer(layer, name=base_name, **kwargs)

    for i, layer in enumerate(layers):
        if viewer is not None:
            layer = viewer.add_layer(layer)
        if not check_image_layer(layer, name=f"{base_name}_{i}", **kwargs):
            return False

    return True


# pylint: disable-next=too-many-return-statements
def check_image_layer(
    layer: Layer,
    name: str | None = None,
    multiscale: bool | None = None,
    rgb: bool | None = None,
) -> bool:
    if not isinstance(layer, Image):
        return False
    if layer.ndim != 2:
        return False
    if not np.allclose(layer.scale, 1):
        return False

    if name is not None and layer.name != name:
        return False
    if multiscale is not None and layer.multiscale != multiscale:
        return False
    if rgb is not None and layer.rgb != rgb:
        return False

    return True


def from_layer_data_tuple(data: LayerDataTuple) -> Image | Labels | Shapes | Points:
    layer_data, layer_params, layer_type = data

    if layer_type == "image":
        return Image(layer_data, **layer_params)
    if layer_type == "labels":
        return Labels(layer_data, **layer_params)
    if layer_type == "shapes":
        return Shapes(layer_data, **layer_params)
    if layer_type == "points":
        return Points(layer_data, **layer_params)

    raise RuntimeError("Unexpected layer type.")


def get_backend(reader: Callable) -> WSIReaderBackend:
    if reader is read_tifffile:
        return WSIReaderBackend.tifffile
    if reader is read_rasterio:
        return WSIReaderBackend.rasterio
    return DEFAULT_BACKEND


@pytest.fixture(scope="session")
def pyramid() -> list[np.ndarray]:
    return list(
        map(
            skimage.util.img_as_ubyte,
            skimage.transform.pyramid_gaussian(
                image=skimage.data.binary_blobs(2048),
                max_layer=2,
            ),
        )
    )


@pytest.fixture(scope="session")
def tmp_data_ome(
    tmp_path_factory: TempPathFactory,
    pyramid: list[np.ndarray],  # pylint: disable=redefined-outer-name
) -> Path:
    path = tmp_path_factory.mktemp("data") / "image.ome.tif"

    with tifffile.TiffWriter(path, bigtiff=True) as handle:
        options = dict(tile=(256, 256), dtype=np.uint8, compression=COMPRESSION.JPEG)

        assert len(pyramid) > 1
        handle.write(pyramid[0], subifds=len(pyramid) - 1, **options)  # type: ignore
        for pyr_level in pyramid[1:]:
            handle.write(pyr_level, **options)  # type: ignore

    return path


@pytest.fixture(scope="session")
def tmp_data_gtiff(
    tmp_path_factory: TempPathFactory,
    pyramid: list[np.ndarray],  # pylint: disable=redefined-outer-name
) -> Path:
    path = tmp_path_factory.mktemp("data") / "image.tif"

    profile = {
        "dtype": "uint8",
        "driver": "GTiff",
        "bigtiff": "yes",
        "compress": "lzw",
        "predictor": "2",
        "photometric": "MINISBLACK",
        "tiled": "yes",
        "blockxsize": 256,
        "blockysize": 256,
        "width": pyramid[0].shape[1],
        "height": pyramid[0].shape[0],
        "count": 1,
    }

    with rasterio.open(path, "w", **profile) as handle:
        handle.write(pyramid[0], 1)
        handle.build_overviews([8, 16], Resampling.average)
        handle.update_tags(ns="rio_overview", resampling="average")

    return path
