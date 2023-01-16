import invoke


@invoke.task
def format(ctx):
    print("## Run black")
    ctx.run("black .")
    print("## Run isort")
    ctx.run("isort .")


@invoke.task
def check(ctx):
    print("## Check formatting")
    ctx.run("black --check .")
    print("## Check imports")
    ctx.run("isort . --check-only")
    print("## Check static typing")
    ctx.run("mypy . --check-untyped-defs")
    print("## Linting code")
    ctx.run("pylint napari_wsi")
    ctx.run("pylint tests")


@invoke.task
def test(ctx):
    print("## Run tests")
    ctx.run("pytest --verbose --cov=napari_wsi tests")
