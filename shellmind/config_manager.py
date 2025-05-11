import yaml
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "shellmind"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

class ConfigManager:
    DEFAULT_CONFIG = {
        "ai_provider": "openai",
        "api_key": None,
        "base_url": None,
        "ai_model": "gpt-4-turbo",
        "temperature": 0.7,
        "max_tokens": 1000,
        "execution_mode": "confirm",  # 'confirm' or 'auto'
        "command_color": "blue"
    }
    def __init__(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()

    def _load_config(self):
        if not CONFIG_FILE.exists():
            self._save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG
        try:
            with open(CONFIG_FILE, "r") as f:
                config = yaml.safe_load(f)
                if config is None:
                    config = {}
                updated = False
                for key, value in self.DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                        updated = True
                if updated:
                    self._save_config(config)
                return config
        except Exception as e:
            print(f"Error loading config file: {e}. Using default configuration and attempting to save it.")
            self._save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG

    def _save_config(self, config_data):
        try:
            with open(CONFIG_FILE, "w") as f:
                yaml.dump(config_data, f, indent=4)
            if CONFIG_FILE.exists():
                os.chmod(CONFIG_FILE, 0o600)
        except Exception as e:
            print(f"Error saving config file: {e}")

    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def set(self, key, value):
        if key in self.DEFAULT_CONFIG:
            self.config[key] = value
            self._save_config(self.config)
        else:
            raise ValueError(
                f"Key {key} is not a recognized configuration option. "
                f"Supported keys are: {', '.join(self.DEFAULT_CONFIG.keys())}"
            )