# src/utils/error_handling.py
"""
Custom exception classes for the SVG Handwriting Generation project.

Provides a hierarchy of specific exceptions for better error management and
reporting throughout the application.
"""

# --- Base Exception ---
class HandwritingError(Exception):
    """Base class for all custom exceptions in this project."""
    def __init__(self, message="An error occurred in the handwriting application."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"


# --- Specific Exception Categories ---

class ConfigError(HandwritingError):
    """Raised for errors related to loading or parsing configuration files."""
    def __init__(self, message="Configuration error."):
        super().__init__(message)

class DataError(HandwritingError):
    """Raised for errors related to dataset loading, parsing, or processing."""
    def __init__(self, message="Data handling error."):
        super().__init__(message)

class APIError(HandwritingError):
    """Raised for errors related to interacting with external APIs (e.g., GlyphWiki)."""
    def __init__(self, message="External API error."):
        super().__init__(message)

class ModelError(HandwritingError):
    """Base class for errors related to the deep learning model."""
    def __init__(self, message="Model error."):
        super().__init__(message)

class ModelLoadError(ModelError):
    """Raised specifically when loading a model checkpoint fails."""
    def __init__(self, message="Failed to load model checkpoint."):
        super().__init__(message)

class TrainingError(HandwritingError):
    """Raised for errors encountered during the model training process."""
    def __init__(self, message="Training process error."):
        super().__init__(message)

class GenerationError(HandwritingError):
    """Raised for errors encountered during the handwriting generation process."""
    def __init__(self, message="Handwriting generation error."):
        super().__init__(message)

class ConversionError(HandwritingError):
    """Raised for errors during format conversion (SVG to raster, model to font)."""
    def __init__(self, message="Format conversion error."):
        super().__init__(message)


# Example usage:
#
# try:
#     # Some operation that might fail
#     result = risky_data_loading_function(path)
# except FileNotFoundError as e:
#     raise DataError(f"Dataset file not found: {path}") from e
# except Exception as e:
#     raise HandwritingError(f"An unexpected error occurred: {e}") from e