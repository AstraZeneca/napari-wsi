from invoke import task
from invoke.context import Context


@task
def fix(ctx: Context) -> None:
    ctx.run("ruff format .")
    ctx.run("ruff check . --fix")


@task
def check(ctx: Context) -> None:
    ctx.run("ruff format . --check")
    ctx.run("ruff check .")
    ctx.run("mypy .")


@task
def test(ctx: Context) -> None:
    ctx.run("pytest --verbose --cov=napari_wsi tests")
