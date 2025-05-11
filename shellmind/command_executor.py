import subprocess
import shutil
from .config_manager import ConfigManager
from .exceptions import CommandExecutionError

class CommandExecutor:
    def __init__(self):
        self.config_manager = ConfigManager()

    def _get_confirmation(self, command: str) -> bool:
        """Asks the user for confirmation before executing a command."""
        command_color = self.config_manager.get("command_color")
        colored_command = f"\033[1;34m{command}\033[0m"
        if command_color == "green":
            colored_command = f"\033[1;32m{command}\033[0m"
        elif command_color == "yellow":
            colored_command = f"\033[1;33m{command}\033[0m"
        
        try:
            response = input(f"ShellMind proposes to execute: {colored_command}\nExecute? (y/n/explain): ").lower()
            if response == "y":
                return True
            elif response == "explain":
                print("Explain functionality is not fully implemented in this version. Command not executed.")
                return False
            else:
                print("Command execution cancelled by user.")
                return False
        except KeyboardInterrupt:
            print("\nCommand execution cancelled by user (Ctrl+C).")
            return False

    def execute_command(self, command: str, ask_confirm: bool = True) -> tuple[str, str, int]:
        """
        Executes the given shell command.
        Returns a tuple: (stdout, stderr, exit_code).
        """
        execution_mode = self.config_manager.get("execution_mode")
        
        if ask_confirm and execution_mode == "confirm":
            if not self._get_confirmation(command):
                return "", "Execution cancelled by user.", -1

        if not command.strip().startswith("sudo") and any(kw in command for kw in ["rm -rf /", "mkfs"]):
            if execution_mode == "confirm":
                # print(f"Warning: The command appears potentially dangerous.")
                if not self._get_confirmation(f"DANGEROUS COMMAND: {command}"):
                    return "", "Execution of potentially dangerous command cancelled by user.", -2
            else:
                 raise CommandExecutionError(f"Refused to automatically execute potentially dangerous command: {command}. Use confirm mode or ensure AI generates safer commands.")

        try:
            use_shell = False
            if any(char in command for char in ["|", ";", "&&", "||", ">", "<", "`", "$", "(", ")"]):
                use_shell = True

            if use_shell:
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            else:
                cmd_parts = shutil.split(command)
                process = subprocess.Popen(cmd_parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            stdout, stderr = process.communicate()
            exit_code = process.returncode
            return stdout, stderr, exit_code
        except FileNotFoundError as e:
            err_msg = f"Command not found: {e.filename}"
            raise CommandExecutionError(err_msg, original_command=command) from e
        except Exception as e:
            raise CommandExecutionError(f"Error executing command: {str(e)}", original_command=command) from e
