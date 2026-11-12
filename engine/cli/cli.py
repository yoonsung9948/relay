from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from pydantic import ValidationError

from engine.bootstrap import build_app
from engine.config import load_config

app = typer.Typer()


@app.callback()
def main():
    """Run the inference server."""


@app.command()
def serve(
    path: Annotated[str, typer.Option("--path", envvar="RELAY_CONFIG_PATH")] = "docker/engine.yaml",
):
    path = str(Path(path).resolve())
    try:
        config = load_config(path)

    except FileNotFoundError:
        typer.echo(
            f"Config file not found: {path}",
            err=True,
        )
        raise typer.Exit(code=1)

    except ValidationError as exc:
        typer.echo(
            f"Invalid config values:\n{exc}",
            err=True,
        )
        raise typer.Exit(code=1)

    uvicorn.run(build_app(config, config_path=path), host=config.serve.host, port=config.serve.port)


if __name__ == "__main__":
    app()
