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

This project uses [uv] for packaging and dependency management. To install the
full set of dependencies needed for developement, run:

```bash
uv sync --all-extras --dev
```

To run a code formatter, a linter and all unit tests, run:

```bash
uv run invoke fix
uv run invoke check
uv run invoke test
```

[uv]: https://github.com/astral-sh/uv
