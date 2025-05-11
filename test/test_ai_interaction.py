import pytest
from unittest.mock import patch, MagicMock, Mock
from shellmind.ai_interaction import AIInteraction, MockAI

def test_mock_ai_provider():
    """Test the mock AI provider directly"""
    mock_ai = MockAI()
    result = mock_ai.get_response("test query")
    assert "mock response to: test query" in result

@pytest.fixture
def mock_config():
    """Fixture to mock the config manager"""
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "ai_provider": "openai",
        "ai_model": "gpt-3.5-turbo",
        "temperature": "0.7",
        "max_tokens": "1000"
    }.get(key, default)
    return config

@pytest.fixture
def mock_os_adapter():
    """Fixture to mock the OS adapter"""
    adapter = MagicMock()
    adapter.get_os_details.return_value = {"name": "TestOS"}
    return adapter

def test_ai_interaction_mock_provider(mock_config, mock_os_adapter):
    """Test with mock provider"""
    mock_config.get.side_effect = lambda key, default=None: {
        "ai_provider": "mock",
    }.get(key, default)
    
    ai = AIInteraction(config_manager=mock_config, os_adapter=mock_os_adapter)
    result = ai.get_command("test query")
    assert "mock response to: test query" in result

def test_ai_interaction_openai_provider(mock_config, mock_os_adapter):
    """Test with OpenAI provider (mocked)"""
    # Configure the mock config to return necessary values including API key
    mock_config.get.side_effect = lambda key, default=None: {
        "ai_provider": "openai",
        "api_key": "test-api-key",
        "base_url": None,
        "ai_model": "gpt-3.5-turbo",
        "temperature": "0.7",
        "max_tokens": "1000"
    }.get(key, default)
    
    # Create a mock OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "openai response"
    mock_client.chat.completions.create.return_value = mock_response
    
    with patch('openai.OpenAI', return_value=mock_client):
        ai = AIInteraction(config_manager=mock_config, os_adapter=mock_os_adapter)
        result = ai.get_command("test query")
        assert result == "openai response"
        # Verify the OpenAI client was called with expected parameters
        mock_client.chat.completions.create.assert_called_once()