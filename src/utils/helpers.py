from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_data_dir(subdir: str = "") -> Path:
    data_path = get_project_root() / "data"
    if subdir:
        data_path = data_path / subdir
    return data_path


def get_config_dir(config_file: str = "") -> Path:
    config_path = get_project_root() / "configs"
    if config_file:
        config_path = config_path / config_file
    return config_path


def get_datasets_dir(subdir: str = "") -> Path:
    if subdir:
        datasets_path = datasets_path / subdir
    return datasets_path
