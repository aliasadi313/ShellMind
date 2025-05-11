import pytest
import yaml
import os
from pathlib import Path
from shellmind.config_manager import ConfigManager

@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Fixture that creates a temporary config environment"""
    # Create config directory structure
    config_dir = tmp_path / ".config" / "shellmind"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.yaml"
    
    # Mock home directory and config paths
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path)
    monkeypatch.setattr('shellmind.config_manager.CONFIG_DIR', config_dir)
    monkeypatch.setattr('shellmind.config_manager.CONFIG_FILE', config_file)
    
    yield config_file

def test_config_manager_initialization(tmp_config):
    """Test that ConfigManager initializes properly"""
    if tmp_config.exists():
        tmp_config.unlink()
    
    cm = ConfigManager()
    assert cm is not None
    assert tmp_config.exists()
    
    # Verify default config was created
    with open(tmp_config, 'r') as f:
        config_data = yaml.safe_load(f)
        assert config_data == ConfigManager.DEFAULT_CONFIG

def test_config_set_get(tmp_config):
    """Test setting and getting config values"""
    cm = ConfigManager()
    
    # Test setting and getting a value
    test_key = "temperature"
    test_value = 0.5
    cm.set(test_key, test_value)
    assert cm.get(test_key) == test_value
    
    # Verify it was saved to file
    with open(tmp_config, 'r') as f:
        config_data = yaml.safe_load(f)
        assert config_data[test_key] == test_value