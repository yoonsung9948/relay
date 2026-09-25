import typer
import uvicorn

from engine.bootstrap import build_app
from engine.config import Config, ServeConfig

cli = typer.Typer()


@cli.callback()
def main():
    """Run the inference server."""


@cli.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
):
    config = Config(serve=ServeConfig(host=host, port=port))
    uvicorn.run(build_app(config), host=config.serve.host, port=config.serve.port)


if __name__ == "__main__":
    cli()
