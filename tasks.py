from invoke import task


@task
def fix(ctx):
    ctx.run("ruff format .")
    ctx.run("ruff check . --fix")


@task
def check(ctx):
    ctx.run("ruff format . --check")
    ctx.run("ruff check .")
    ctx.run("mypy .")


@task
def test(ctx):
    ctx.run("pytest --verbose --cov=napari_wsi tests")
