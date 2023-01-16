# Contribution Guide

Thank you for your interest in improving this project. This project is
open-source under the Apache 2.0 License and welcomes contributions in the form
of bug reports and feature requests submitted using the issue tracker, as well
as pull requests.

## Reporting a bug

When filing an issue, please make sure to answer these questions:

- Which operating system and python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case and the steps to
reproduce the issue. If possible, please include a minimal, reproducible
example.

## Setting up a development environment

This project uses [poetry] packaging and dependency management.

```bash
poetry install
poetry run pip install pyqt5
poetry run napari
```

Note: `pyqt5` is not a direct dependency of this plugin, but is necessary to run
napari (see the [napari installation guide]).

## Running checks and tests

```bash
poetry run invoke check
poetry run invoke test
```

[napari installation guide]:
  https://napari.org/stable/tutorials/fundamentals/installation.html#choosing-a-different-qt-backend
[poetry]: https://python-poetry.org/
