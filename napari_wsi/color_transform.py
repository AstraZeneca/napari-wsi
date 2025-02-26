from enum import StrEnum
from io import BytesIO

from PIL.Image import Image
from PIL.ImageCms import (
    ImageCmsProfile,
    ImageCmsTransform,
    Intent,
    applyTransform,
    buildTransform,
    createProfile,
    getDefaultIntent,
    getOpenProfile,
)


class ColorSpace(StrEnum):
    RAW = "RAW"
    sRGB = "sRGB"  # noqa: N815


class ColorTransform:
    """A class for applying a color transform based on an ICC profile."""

    def __init__(
        self,
        profile: ImageCmsProfile | bytes | None = None,
        mode: str = "RGB",
        color_space: ColorSpace = ColorSpace.RAW,
    ) -> None:
        """Initialize a `ColorTransform`.

        Args:
            profile: The input ICC profile.
            mode: The image mode, specifying the pixel format (usually 'RGB' or 'RGBA').
            color_space: The target color space.

        """
        self._transform = None
        self._color_space = color_space
        if profile is not None and color_space != ColorSpace.RAW:
            if not isinstance(profile, ImageCmsProfile):
                profile = getOpenProfile(BytesIO(profile))
            self._transform = buildTransform(
                profile,
                createProfile(str(color_space)),  # type: ignore[arg-type]
                mode,
                mode,
                Intent(getDefaultIntent(profile)),
            )

    @property
    def transform(self) -> ImageCmsTransform | None:
        return self._transform

    @property
    def color_space(self) -> ColorSpace:
        if self.transform is None:
            return ColorSpace.RAW
        return self._color_space

    def __call__(self, image: Image) -> Image:
        if self.transform is not None:
            transformed_image = applyTransform(image, self.transform)
            # None is only returned if the transform is applied in-place.
            assert transformed_image is not None
            return transformed_image
        return image
