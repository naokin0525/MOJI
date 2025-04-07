# src/utils/config.py
"""
Configuration loading utility for the SVG Handwriting Generation project.

Provides functions to load settings from YAML files and merge them with
command-line arguments.
"""

import yaml
import os
import logging
from argparse import Namespace # Used for type hinting

# Import custom exception
try:
    from .error_handling import ConfigError
except ImportError:
    # Define a minimal fallback if error_handling isn't available yet
    class ConfigError(Exception):
        pass

logger = logging.getLogger(__name__) # Use module-specific logger

def load_config(config_path: str) -> dict:
    """
    Loads configuration settings from a YAML file.

    Args:
        config_path (str): The full path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration settings. Returns an
              empty dictionary if the file is not found or cannot be parsed,
              and logs a warning.

    Raises:
        ConfigError: If there is an error reading or parsing the YAML file
                     (other than FileNotFoundError).
    """
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found at: {config_path}. Proceeding without it.")
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        if config_data is None: # Handle empty YAML file case
            logger.warning(f"Configuration file {config_path} is empty.")
            return {}
        logger.info(f"Successfully loaded configuration from: {config_path}")
        return config_data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}", exc_info=True)
        raise ConfigError(f"Invalid YAML format in {config_path}: {e}") from e
    except IOError as e:
        logger.error(f"Error reading configuration file {config_path}: {e}", exc_info=True)
        raise ConfigError(f"Could not read file {config_path}: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred loading configuration from {config_path}: {e}", exc_info=True)
        raise ConfigError(f"Unexpected error loading {config_path}: {e}") from e


def get_config_value(key: str, args: Namespace, config: dict, default=None):
    """
    Retrieves a configuration value based on priority: CLI args > config file > default.

    Args:
        key (str): The configuration key to retrieve (e.g., 'learning_rate_g').
        args (argparse.Namespace): The namespace object containing parsed command-line arguments.
        config (dict): The dictionary loaded from the configuration file.
        default (any, optional): The default value to return if the key is not found
                                 in args or config. Defaults to None.

    Returns:
        any: The retrieved configuration value.
    """
    # 1. Check CLI arguments (if the argument exists and was actually provided, i.e., not None)
    #    We check hasattr because some args might be defined in the parser but not relevant/provided for every run.
    if hasattr(args, key) and getattr(args, key) is not None:
        # logger.debug(f"Using value for '{key}' from command line: {getattr(args, key)}")
        return getattr(args, key)

    # 2. Check configuration file
    if key in config:
        # logger.debug(f"Using value for '{key}' from config file: {config[key]}")
        return config[key]

    # 3. Return default value
    # logger.debug(f"Using default value for '{key}': {default}")
    return default

# Example usage (can be uncommented for direct testing)
# if __name__ == "__main__":
#     # Create dummy args and config
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--learning_rate", type=float, default=None)
#     parser.add_argument("--batch_size", type=int, default=None)
#     parser.add_argument("--unused_arg", type=str, default=None)
#     # Simulate providing only learning_rate via CLI
#     args = parser.parse_args(["--learning_rate", "0.001"])

#     # Create dummy config file content
#     dummy_config_content = """
# batch_size: 64
# epochs: 50
# learning_rate: 0.0005 # This should be overridden by CLI arg
# """
#     dummy_config_path = "dummy_config.yaml"
#     with open(dummy_config_path, "w") as f:
#         f.write(dummy_config_content)

#     # Set up basic logging for testing this module
#     logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')

#     # Test loading config
#     loaded_config = load_config(dummy_config_path)
#     print(f"Loaded config: {loaded_config}")

#     # Test get_config_value
#     lr = get_config_value('learning_rate', args, loaded_config, default=0.01)
#     bs = get_config_value('batch_size', args, loaded_config, default=32)
#     ep = get_config_value('epochs', args, loaded_config, default=100)
#     ua = get_config_value('unused_arg', args, loaded_config, default="default_unused")
#     missing = get_config_value('missing_key', args, loaded_config, default="fallback")

#     print(f"Learning Rate (CLI override): {lr}") # Expected: 0.001
#     print(f"Batch Size (from config): {bs}")     # Expected: 64
#     print(f"Epochs (from config): {ep}")         # Expected: 50
#     print(f"Unused Arg (CLI default None, not in config -> default): {ua}") # Expected: default_unused
#     print(f"Missing Key (fallback default): {missing}") # Expected: fallback

#     # Clean up dummy file
#     os.remove(dummy_config_path)