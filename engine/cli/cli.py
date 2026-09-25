import typer
import uvicorn

from engine.bootstrap import build_app
from engine.config import Config, ServeConfig, ModelConfig, SchedulerConfig
from engine.config import load_config
from pydantic import ValidationError
app = typer.Typer()


@app.callback()
def main():
    """Run the inference server."""


@app.command()
def serve(
    path: str = "config.yaml"
):

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

    uvicorn.run(build_app(config), host=config.serve.host, port=config.serve.port)


if __name__ == "__main__":
    app()
