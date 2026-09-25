import yaml
from engine.config.config import Config


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return Config.model_validate(raw)