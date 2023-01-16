# Description

This [napari] plugin provides a reader for various whole slide image formats.

By default, any of the following formats is read using the [tifffile] library.
If the image file contains a tag `GDAL_METADATA`, the [rasterio] library is used
instead.

- .bif
- .ndpi
- .qptiff
- .scn
- .svs
- .tif
- .tiff

# Quickstart

From the terminal:

```bash
napari CMU-1.svs
```

From python:

```python
import napari

viewer = napari.Viewer()
viewer.open("CMU-1.svs")
```

[napari]: https://github.com/napari/napari
[rasterio]: https://github.com/rasterio/rasterio
[tifffile]: https://github.com/cgohlke/tifffile
