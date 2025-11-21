import numpy as np
import pytest
from numpy.typing import DTypeLike

from napari_wsi.common import PyramidLevel, PyramidLevels


@pytest.mark.parametrize(("dtype", "chunks"), [(np.uint8, (32, 32))])
class TestPyramid:
    def test_pyramid_construction(
        self, dtype: DTypeLike, chunks: tuple[int, int]
    ) -> None:
        """Test some consistency checks in `PyramidLevels`."""
        with pytest.raises(ValueError, match="channels"):
            PyramidLevels(num_channels=0, dtype=dtype)

        levels = PyramidLevels(num_channels=3, dtype=dtype)
        levels += PyramidLevel(factor=1, shape=(256, 256), chunks=chunks)
        with pytest.raises(ValueError, match="factor"):
            levels += PyramidLevel(factor=1, shape=(128, 128), chunks=chunks)
        with pytest.raises(ValueError, match="shape"):
            levels += PyramidLevel(factor=2, shape=(256, 256), chunks=chunks)
