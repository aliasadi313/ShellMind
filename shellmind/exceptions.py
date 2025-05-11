# ShellMind Custom Exceptions

class ShellMindError(Exception):
    """Base exception class for ShellMind application errors."""
    pass

class ConfigError(ShellMindError):
    """Exception raised for errors in the configuration."""
    pass

class AIInteractionError(ShellMindError):
    """Exception raised for errors during interaction with the AI model."""
    pass

class CommandGenerationError(ShellMindError):
    """Exception raised when the AI fails to generate a usable command."""
    pass

class CommandExecutionError(ShellMindError):
    """Exception raised for errors during command execution."""
    def __init__(self, message, original_command=None, stdout=None, stderr=None, exit_code=None):
        super().__init__(message)
        self.original_command = original_command
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code

    def __str__(self):
        msg = super().__str__()
        if self.original_command:
            msg += f"\nOriginal Command: {self.original_command}"
        if self.stderr:
            msg += f"\nSTDERR: {self.stderr.strip()}"
        return msg

class OSAdapterError(ShellMindError):
    """Exception raised for errors within the OS Adapter."""
    pass

class UserCancellation(ShellMindError):
    """Exception raised when the user cancels an operation."""
    pass

