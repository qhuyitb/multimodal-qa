import yaml
from pathlib import Path
from typing import Dict, Any

_config_cache = None

def get_config() -> Dict[str, Any]:
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            _config_cache = yaml.safe_load(f) or {}
    else:
        _config_cache = {
            "paths": {"vector_db": "data/vector_db"},
            "translation": {"enabled": True}
        }
    
    return _config_cache
