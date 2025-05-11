import os
from .config_manager import ConfigManager
from .os_adapter import OSAdapter
from .exceptions import AIInteractionError, ConfigError
import openai
from typing import Optional

class MockAI:
    """Mock AI provider for testing"""
    def get_response(self, query: str) -> str:
        """Return a mock command response"""
        return f"echo 'mock response to: {query}'"

class AIInteraction:
    def __init__(self, config_manager=None, os_adapter=None):
        self.config_manager = config_manager or ConfigManager()
        self.os_adapter = os_adapter or OSAdapter()
        self.client = None
        self._configure_client()
        
    def _configure_client(self):
        self.provider = self.config_manager.get("ai_provider", "openai")
        self.api_key = self.config_manager.get("api_key")
        self.base_url = self.config_manager.get("base_url")
        self.model = self.config_manager.get("ai_model", "gpt-3.5-turbo")
        self.temperature = float(self.config_manager.get("temperature", 0.7))
        self.max_tokens = int(self.config_manager.get("max_tokens", 1000))

        if self.provider == "mock":
            self.client = MockAI()
        else:
            if not self.api_key:
                raise ConfigError("API key not set in configuration")
            try:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url if self.base_url else "https://api.openai.com/v1"
                )
            except Exception as e:
                raise AIInteractionError(f"Failed to initialize AI client: {str(e)}")

    def _get_base_prompt_for_model(self) -> str:
        os_details = self.os_adapter.get_os_details()
        return (
            f"You are ShellMind, an AI assistant. The user is on {os_details['name']}. "
            "Your goal is to translate their natural language queries into a single, "
            "executable shell command. Do not provide any explanations, only the command itself. "
            "If you cannot determine a command, respond with 'Error: Unable to determine command.'"
        )

    def get_command(self, user_query: str) -> str:
        if not self.client:
            return "Error: AI client not initialized"

        if self.provider == "mock":
            return self.client.get_response(user_query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_base_prompt_for_model()},
                    {"role": "user", "content": user_query}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            command = response.choices[0].message.content.strip()

            if not command or command.startswith("Error:"):
                return "Error: AI failed to generate a valid command"
            return command

        except openai.APIConnectionError as e:
            err_msg = f"API Connection Error: {e}. Check your network and the API base URL if you set a custom one"
            raise AIInteractionError(err_msg) from e
        except openai.AuthenticationError as e:
            err_msg = f"Authentication Error: {e}. Check your API key"
            raise AIInteractionError(err_msg) from e
        except openai.RateLimitError as e:
            err_msg = f"Rate Limit Exceeded: {e}. Please wait and try again later, or check your plan"
            raise AIInteractionError(err_msg) from e
        except Exception as e:
            raise AIInteractionError(f"Failed to get command from AI: {str(e)}") from e