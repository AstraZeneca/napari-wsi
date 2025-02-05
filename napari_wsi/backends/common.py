from enum import StrEnum


class WSIReaderBackend(StrEnum):
    OPENSLIDE = "openslide"
    RASTERIO = "rasterio"
    WSIDICOM = "wsidicom"
