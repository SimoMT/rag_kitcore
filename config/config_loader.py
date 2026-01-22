from pathlib import Path
import yaml

from .schema import AppConfig


def load_config(path: str = "config.yaml") -> AppConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    return AppConfig(**data)
