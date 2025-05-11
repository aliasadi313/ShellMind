import os
from .config_manager import ConfigManager
from .os_adapter import OSAdapter
from .exceptions import AIInteractionError, ConfigError
import openai


class AIInteraction:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.os_adapter = OSAdapter()
        self.client = None
        self._configure_client()

    def _configure_client(self):
        self.api_key = self.config_manager.get("api_key")
        self.base_url = self.config_manager.get("base_url")
        if not self.api_key:
            print("Warning: API key not set in configuration.")
        
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else "https://platform.openai.com/v1"
            )
        except TypeError as te:
            raise

    def _get_base_prompt_for_model(self):
        os_details = self.os_adapter.get_os_details()
        prompt = f"You are ShellMind, an AI assistant. The user is on {os_details['name']}. Your goal is to translate their natural language queries into a single, executable shell command. Do not provide any explanations, only the command itself. If you cannot determine a command, respond with 'Error: Unable to determine command.'"
        return prompt

    def get_command(self, user_query: str) -> str:

        if not self.client:
            return "Error: AI client not initialized. Check API key and base URL configuration."
        
        system_prompt = self._get_base_prompt_for_model()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        try:
            model_name = self.config_manager.get("ai_model")
            temperature = float(self.config_manager.get("temperature"))
            max_tokens = int(self.config_manager.get("max_tokens"))

            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            command = response.choices[0].message.content.strip()

            if not command or command.startswith("Error:"):
                return "Error: AI failed to generate a valid command."
            return command

        except openai.APIConnectionError as e:
            err_msg = f"AI API Connection Error: {e}. Check your network and the API base URL if you set a custom one."
            print(err_msg)
            raise AIInteractionError(err_msg) from e
        except openai.AuthenticationError as e:
            err_msg = f"AI API Authentication Error: {e}. Check your API key."
            print(err_msg)
            raise AIInteractionError(err_msg) from e
        except openai.RateLimitError as e:
            err_msg = f"AI API Rate Limit Exceeded: {e}. Please wait and try again later, or check your plan."
            print(err_msg)
            raise AIInteractionError(err_msg) from e
        except Exception as e:
            raise AIInteractionError(f"Failed to get command from AI: {str(e)}") from e