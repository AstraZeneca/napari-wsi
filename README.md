# napari-wsi

[![PyPI](https://img.shields.io/pypi/v/napari-wsi.svg?color=green)](https://pypi.org/project/napari-wsi)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-wsi)](https://napari-hub.org/plugins/napari-wsi)
[![Tests](https://github.com/AstraZeneca/napari-wsi/actions/workflows/main.yml/badge.svg)](https://github.com/AstraZeneca/napari-wsi/actions)
![Maturity Level-1](https://img.shields.io/badge/Maturity%20Level-ML--1-yellow)

A plugin to read whole-slide images within [napari].

---

## Installation

You can install `napari-wsi` via [pip]:

```bash
pip install napari-wsi[all]
```

This automatically installs all optional backends, as a shortcut for:

```bash
pip install napari-wsi[openslide,rasterio,wsidicom]
```

In addition, to be able to read images using the `openslide` backend, it is
required to install the OpenSlide library itself, for example by installing the
[openslide-bin] python package (also via [pip]).

# Description

This [napari] plugin provides a widget for reading various whole-slide image
formats using a common [zarr] store inteface, based on the libraries
[openslide], [rasterio], and [wsidicom].

# Quickstart

After installation, open the `Plugins` menu in the viewer and select
`WSI Reader` to open the widget. Then select a `Backend` to use, select a `Path`
to open, and click `Load`.

![The napari viewer displaying a sample image.](./resources/sample_data.jpg)

If `sRGB` is selected in the `Color Space` menu and an ICC profile is attached
to the given image, a transformation to this color space will be applied when
the image data is read. Otherwise, the raw RGB image data will be displayed.

This plugin can also be used to open image files via drag and drop into the
viewer window. The file suffixes '.bif', '.ndpi', '.scn', '.svs' are registered
with the `openslide` backend, while the suffixes '.tif' and '.tiff' are
registered with the `rasterio` backend. These files can also be opened directly
from the command line or from a python script:

```bash
napari CMU-1.svs
```

```python
from napari.viewer import Viewer

viewer = Viewer()
viewer.open("CMU-1.svs", plugin="napari-wsi")
```

[napari]: https://github.com/napari/napari
[openslide]: https://github.com/openslide/openslide-python
[openslide-bin]: https://pypi.org/project/openslide-bin/
[pip]: https://github.com/pypa/pip
[rasterio]: https://github.com/rasterio/rasterio
[wsidicom]: https://github.com/imi-bigpicture/wsidicom
[zarr]: https://github.com/zarr-developers/zarr-python
