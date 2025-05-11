#!/usr/bin/env python3
import argparse
import sys

try:
    from shellmind.config_manager import ConfigManager
    from shellmind.os_adapter import OSAdapter
    from shellmind.input_processor import InputProcessor
    from shellmind.ai_interaction import AIInteraction
    from shellmind.command_executor import CommandExecutor
    from shellmind.exceptions import ShellMindError, AIInteractionError, CommandExecutionError, ConfigError
except ImportError as e:
    missing_module = str(e).split()[-1].replace("'", "")
    if missing_module == "shellmind":
        print("Error: The shellmind package is not installed. Please install it first.")
    elif missing_module == "shellmind.config_manager":
        print("Error: ConfigManager module not found in shellmind package.")
    elif missing_module == "shellmind.os_adapter":
        print("Error: OSAdapter module not found in shellmind package.")
    elif missing_module == "shellmind.input_processor":
        print("Error: InputProcessor module not found in shellmind package.")
    elif missing_module == "shellmind.ai_interaction":
        print("Error: AIInteraction module not found in shellmind package.")
    elif missing_module == "shellmind.command_executor":
        print("Error: CommandExecutor module not found in shellmind package.")
    elif missing_module == "shellmind.exceptions":
        print("Error: Exceptions module not found in shellmind package.")
    else:
        print(f"Error: Failed to import required module - {missing_module}")
    print("Please ensure all components of the shellmind package are properly installed.")
    exit(1)

def handle_query(args):
    query = args.query_string
    if not query:
        print("Error: No query provided. Use -h for help.")
        return

    try:
        print(f"ShellMind processing: '{query}'...")       
        config_manager = ConfigManager()
        
        input_processor = InputProcessor()
        ai_interaction = AIInteraction()
        command_executor = CommandExecutor()

        processed_query = input_processor.process_query(query)
        print(f"Sending to AI: '{processed_query}'")        
        generated_command = ai_interaction.get_command(processed_query)

        if generated_command.startswith("Error:"):
            print(f"ShellMind Error: {generated_command}")
            return

        print(f"AI suggested command: {generated_command}")
        stdout, stderr, exit_code = command_executor.execute_command(generated_command)

        if exit_code == 0:
            print("\nCommand executed successfully.")
            if stdout:
                print("--- STDOUT ---")
                print(stdout.strip())
        else:
            print(f"\nCommand execution failed with exit code: {exit_code}")
        
        if stderr:
            print("--- STDERR ---")
            print(stderr.strip())

    except Exception as e:
        print(f"An unexpected ShellMind error occurred: {e}")

def handle_config(args):
    config_manager = ConfigManager()
    
    if args.action == "set" and args.key and args.value is not None:
        value_to_set = args.value
        if args.key == "execution_mode":
            if args.value.lower() in ["true", "confirm"]:
                value_to_set = "confirm"
            elif args.value.lower() in ["false", "auto"]:
                value_to_set = "auto"
        elif args.key == "temperature":
            try:
                value_to_set = float(args.value)
                if not 0 <= value_to_set <= 2:
                    print("Error: Temperature must be between 0 and 2")
                    return
            except ValueError:
                print("Error: Invalid value for temperature. Must be a number.")
                return
        elif args.key == "max_tokens":
            try:
                value_to_set = int(args.value)
                if value_to_set <= 0:
                    print("Error: max_tokens must be a positive integer")
                    return
            except ValueError:
                print("Error: Invalid value for max_tokens. Must be an integer.")
                return

        if args.key not in config_manager.DEFAULT_CONFIG:
            print(f"Error: Unknown configuration key. Supported keys are: {', '.join(config_manager.DEFAULT_CONFIG.keys())}")
            return

        config_manager.set(args.key, value_to_set)
        print(f"Configuration updated: {args.key} = {value_to_set}")
    elif args.action == "get" and args.key:
        value = config_manager.get(args.key)
        if value is not None:
            print(f"{args.key}: {value}")
        else:
            print(f"Error: Configuration key '{args.key}' not found")
    elif args.action == "list":
        print("Current ShellMind Configuration:")
        for key, value in config_manager.config.items():
            print(f"  {key}: {value}")
    else:
        print("Invalid config command. Use `shellmind config set <key> <value>`, `shellmind config get <key>`, or `shellmind config list`.")


def main():
    parser = argparse.ArgumentParser(
        description="ShellMind: AI-Powered Command-Line Assistant.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(
        dest="command_group",
        title="Available commands",
        required=True,
        help="Run `shellmind <command> -h` for more help."
    )

    query_parser = subparsers.add_parser("query", help="Ask ShellMind to perform a task.")
    query_parser.add_argument("query_string", type=str, help="The natural language query for ShellMind.")
    query_parser.set_defaults(func=handle_query)

    config_parser = subparsers.add_parser("config", help="Manage ShellMind configuration.")
    config_subparsers = config_parser.add_subparsers(
        dest="action",
        title="Config actions",
        required=True,
        help="Configuration management commands"
    )

    config_set_parser = config_subparsers.add_parser("set", help="Set a configuration value.")
    config_set_parser.add_argument("key", help="Configuration key to set")
    config_set_parser.add_argument("value", help="Value to set")
    config_set_parser.set_defaults(func=handle_config)

    config_get_parser = config_subparsers.add_parser("get", help="Get a configuration value.")
    config_get_parser.add_argument("key", help="Configuration key to get")
    config_get_parser.set_defaults(func=handle_config)

    config_list_parser = config_subparsers.add_parser("list", help="List all configurations.")
    config_list_parser.set_defaults(func=handle_config)

    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()