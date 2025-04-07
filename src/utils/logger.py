# src/utils/logger.py
"""
Logging setup utility for the SVG Handwriting Generation project.

Provides a standardized way to configure logging for console and file output.
"""

import logging
import os
import sys

# Define a mapping from string log levels to logging constants
LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Keep track if logging has been configured to avoid issues with basicConfig
_logging_configured = False

def setup_logging(level='INFO', log_file=None, log_format=DEFAULT_LOG_FORMAT, date_format=DEFAULT_DATE_FORMAT):
    """
    Configures the root logger for the application.

    Sets up logging to stream to console (stderr) and optionally to a file.
    Uses basicConfig for simplicity, ensuring it's only called effectively once.

    Args:
        level (str): The desired logging level (e.g., 'DEBUG', 'INFO', 'WARNING').
                     Defaults to 'INFO'. Case-insensitive.
        log_file (str, optional): Path to the file where logs should be saved.
                                  If None, logging will only go to the console.
                                  Defaults to None.
        log_format (str, optional): The format string for log messages.
                                    Defaults to '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'.
        date_format (str, optional): The format string for the timestamp in logs.
                                     Defaults to '%Y-%m-%d %H:%M:%S'.

    Returns:
        None
    """
    global _logging_configured

    # Prevent re-configuring if already done, basicConfig doesn't handle this well if called again
    # Although basicConfig itself is idempotent if root logger already has handlers,
    # this prevents potential confusion or unwanted side effects if called with different params later.
    if _logging_configured:
        # logging.warning("Logging setup already configured. Skipping reconfiguration.")
        # Optional: could update level if needed without full reconfig
        current_level_str = logging.getLevelName(logging.getLogger().getEffectiveLevel())
        requested_level_str = level.upper()
        if current_level_str != requested_level_str:
            log_level_int = LOG_LEVEL_MAP.get(requested_level_str, LOG_LEVEL_MAP[DEFAULT_LOG_LEVEL])
            logging.getLogger().setLevel(log_level_int)
            logging.info(f"Updated logging level to: {requested_level_str}")
        return

    # Validate and get the logging level integer
    level_upper = level.upper()
    log_level_int = LOG_LEVEL_MAP.get(level_upper)

    if log_level_int is None:
        print(f"Warning: Invalid log level '{level}'. Defaulting to '{DEFAULT_LOG_LEVEL}'.", file=sys.stderr)
        log_level_int = LOG_LEVEL_MAP[DEFAULT_LOG_LEVEL]
        level_upper = DEFAULT_LOG_LEVEL # Use the defaulted level name for consistency

    # Prepare arguments for basicConfig
    config_kwargs = {
        'level': log_level_int,
        'format': log_format,
        'datefmt': date_format,
        'handlers': [] # Start with an empty list of handlers
    }

    # Configure console handler (always add)
    console_handler = logging.StreamHandler(sys.stderr) # Log to stderr by default
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    config_kwargs['handlers'].append(console_handler)
    # print(f"Debug: Adding Console Handler with level {level_upper}", file=sys.stderr) # For debugging setup

    # Configure file handler if path is provided
    if log_file:
        try:
            # Ensure the directory for the log file exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8') # Append mode
            file_handler.setFormatter(logging.Formatter(log_format, date_format))
            config_kwargs['handlers'].append(file_handler)
            # print(f"Debug: Adding File Handler for {log_file}", file=sys.stderr) # For debugging setup
        except OSError as e:
            # Use print as logging might not be fully configured yet
            print(f"Error: Could not create or open log file '{log_file}': {e}. Logging to file disabled.", file=sys.stderr)
        except Exception as e:
            print(f"Error: Unexpected issue setting up log file handler for '{log_file}': {e}. Logging to file disabled.", file=sys.stderr)


    # Apply the configuration using basicConfig
    # Force=True allows overwriting previous basicConfig settings (useful if run in interactive sessions or tests)
    # However, using our _logging_configured flag is safer to avoid conflicts within the app's lifecycle.
    # Let's stick to the flag and avoid force=True for now.
    # If handlers list is provided to basicConfig, it uses those instead of default StreamHandler.
    try:
        logging.basicConfig(**config_kwargs)
        _logging_configured = True
        logging.info(f"Logging configured. Level: {level_upper}. Outputting to console." + (f" Log file: {log_file}" if log_file and any(isinstance(h, logging.FileHandler) for h in config_kwargs['handlers']) else ""))
    except Exception as e:
         print(f"CRITICAL: Failed to configure logging: {e}", file=sys.stderr)
         # Fallback to very basic print logging if setup fails catastrophically
         logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
         logging.critical(f"Logging setup failed: {e}")


# Example usage (can be run directly for testing)
if __name__ == "__main__":
    print("Testing logger setup...")

    # Test 1: Basic INFO level to console
    print("\n--- Test 1: INFO to Console ---")
    _logging_configured = False # Reset for test
    setup_logging(level='INFO')
    logging.debug("This debug message should NOT appear.")
    logging.info("This info message should appear.")
    logging.warning("This warning message should appear.")
    logging.error("This error message should appear.")

    # Test 2: DEBUG level to console and file
    print("\n--- Test 2: DEBUG to Console and File (test_log.log) ---")
    _logging_configured = False # Reset for test
    test_log_file = "test_log.log"
    if os.path.exists(test_log_file):
        os.remove(test_log_file)
    setup_logging(level='DEBUG', log_file=test_log_file)
    logging.debug("This debug message SHOULD appear (console and file).")
    logging.info("This info message SHOULD appear (console and file).")
    # Verify file content
    if os.path.exists(test_log_file):
        print(f"Log file '{test_log_file}' created. Check its contents.")
        # Cleanup: remove test log file
        # os.remove(test_log_file)
    else:
        print(f"Error: Log file '{test_log_file}' was not created.")


    # Test 3: Invalid level
    print("\n--- Test 3: Invalid Level ---")
    _logging_configured = False # Reset for test
    setup_logging(level='INVALID_LEVEL')
    logging.info("This info message should appear (default level).") # Should default to INFO

    # Test 4: Reconfiguration attempt (should update level)
    print("\n--- Test 4: Reconfiguration Attempt ---")
    # _logging_configured should be True now from previous test
    print(f"Logging configured status: {_logging_configured}")
    setup_logging(level='WARNING') # Should just update level without full reconfig
    logging.info("This info message should NOT appear (level is WARNING).")
    logging.warning("This warning message SHOULD appear.")

    print("\nLogger testing complete.")