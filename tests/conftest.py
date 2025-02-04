from pathlib import Path
from warnings import catch_warnings

import pytest
import rasterio
import skimage.data
import skimage.transform
import skimage.util
from attr import dataclass
from pytest import FixtureRequest, TempPathFactory
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

from napari_wsi.color_transform import ColorSpace
from napari_wsi.widget import WSIReaderBackend

TEST_DATA_PATH = Path(__file__).parent.resolve() / "data"

# https://github.com/rasterio/rasterio/blob/72bdbc05b4670043fb81413e6ee718f06617b86c/tests/data/RGB.byte.tif
# https://creativecommons.org/publicdomain/zero/1.0/
TEST_DATA_GTIF = TEST_DATA_PATH / "RGB.byte.tif"

# https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs
# https://creativecommons.org/publicdomain/zero/1.0/
TEST_DATA_SVS = TEST_DATA_PATH / "CMU-1-Small-Region.svs"

# This is the SVS test image, converted to DICOM via wsidicomizer:
# https://github.com/imi-bigpicture/wsidicomizer
TEST_DATA_DCM = TEST_DATA_PATH / "CMU-1-Small-Region"


@dataclass(frozen=True)
class Case:
    id: str
    backend: WSIReaderBackend
    multiscale: bool
    rgb: bool
    resolution: tuple[float, float] | None
    color_space: ColorSpace
    file_path: Path | None = None
    file_fixture: str | None = None

    def path(self, request: FixtureRequest) -> Path:
        if self.file_path is not None:
            path = self.file_path
        elif self.file_fixture is not None:
            path = request.getfixturevalue(self.file_fixture)
        else:
            raise ValueError("Need either 'file_path' or 'file_fixture'.")
        assert isinstance(path, Path) and path.exists()
        return path.resolve()


DEFAULT_TEST_CASES = [
    Case(
        id="svs-default",
        file_path=TEST_DATA_SVS,
        backend=WSIReaderBackend.OPENSLIDE,
        resolution=(0.499, 0.499),
        color_space=ColorSpace.RAW,
        multiscale=False,
        rgb=True,
    ),
    Case(
        id="svs-icc",
        file_path=TEST_DATA_SVS,
        backend=WSIReaderBackend.OPENSLIDE,
        resolution=(0.499, 0.499),
        color_space=ColorSpace.sRGB,
        multiscale=False,
        rgb=True,
    ),
    Case(
        id="gtiff-default",
        file_path=TEST_DATA_GTIF,
        backend=WSIReaderBackend.RASTERIO,
        resolution=None,
        color_space=ColorSpace.RAW,
        multiscale=False,
        rgb=True,
    ),
    Case(
        id="gtiff-pyramid",
        file_fixture="dummy_gtiff",
        backend=WSIReaderBackend.RASTERIO,
        resolution=None,
        color_space=ColorSpace.RAW,
        multiscale=True,
        rgb=False,
    ),
    Case(
        id="dcm-default",
        file_path=TEST_DATA_DCM,
        backend=WSIReaderBackend.WSIDICOM,
        resolution=(0.499, 0.499),
        color_space=ColorSpace.RAW,
        multiscale=False,
        rgb=True,
    ),
]


@pytest.fixture(scope="session")
def dummy_gtiff(tmp_path_factory: TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "image.tiff"

    data = skimage.util.img_as_ubyte(skimage.data.binary_blobs(2048))
    height, width = data.shape

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
        "width": width,
        "height": height,
        "count": 1,
    }

    with (
        catch_warnings(category=NotGeoreferencedWarning, action="ignore"),
        rasterio.open(path, mode="w", **profile) as handle,
    ):
        handle.write(data, 1)
        handle.build_overviews([8, 16], Resampling.average)
        handle.update_tags(ns="rio_overview", resampling="average")

    return path
