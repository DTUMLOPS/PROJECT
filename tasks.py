import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "ehr_classification"
PYTHON_VERSION = "3.12"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/evaluate.py", echo=True, pty=not WINDOWS)


@task
def infer(ctx: Context) -> None:
    """Run model inference."""
    ctx.run(f"python src/{PROJECT_NAME}/inference.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train_ehr:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t evaluate_ehr:latest . -f dockerfiles/evaluate.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task(docker_build)
def docker_train(ctx: Context) -> None:
    """Run training in Docker container."""
    ctx.run("docker run --rm train_ehr:latest", echo=True, pty=not WINDOWS)


@task(docker_build)
def docker_evaluate(ctx: Context) -> None:
    """Run evaluation in Docker container."""
    ctx.run("docker run --rm evaluate_ehr:latest", echo=True, pty=not WINDOWS)
