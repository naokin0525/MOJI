# src/data/glyphwiki_api.py
"""
Utility for interacting with the GlyphWiki API.

Provides functions to fetch structural information about Kanji characters.
Requires the 'requests' library.
"""

import requests
import logging
import time

# Import custom exception
try:
    from ..utils.error_handling import APIError
except ImportError:
    # Define a minimal fallback if error_handling isn't available yet
    class APIError(Exception):
        pass

logger = logging.getLogger(__name__)

# --- Configuration ---
GLYPHWIKI_API_BASE_URL = "https://glyphwiki.org/api/glyph"
DEFAULT_TIMEOUT = 10 # seconds for API requests
RETRY_COUNT = 3
RETRY_DELAY = 1 # seconds

# --- API Interaction ---
def fetch_glyph_data(character: str) -> dict | None:
    """
    Fetches structural data for a given character (primarily Kanji) from GlyphWiki.

    Args:
        character (str): The single character to query (e.g., "猫").

    Returns:
        dict | None: A dictionary containing the data returned by the API
                     (structure depends on GlyphWiki API response), or None if
                     the character is not found, the API request fails, or the
                     input is not suitable (e.g., multi-character string).
                     Returns None immediately for non-single characters.

    Raises:
        APIError: If the API returns an unexpected status code or network issues occur after retries.
    """
    if not isinstance(character, str) or len(character) != 1:
        logger.warning(f"GlyphWiki query skipped: Input '{character}' is not a single character.")
        return None

    # Construct API URL (using 'name' parameter as per typical GlyphWiki usage)
    # Note: API might require URL encoding for some characters, `requests` handles this.
    params = {'name': f'u{ord(character):x}'} # Use uXXXX unicode name convention often used by GW

    for attempt in range(RETRY_COUNT):
        try:
            response = requests.get(GLYPHWIKI_API_BASE_URL, params=params, timeout=DEFAULT_TIMEOUT)

            # Check response status
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Basic check if data seems valid (e.g., contains expected keys)
                    # The exact structure depends on the API version and glyph data.
                    # We might just return the whole JSON for the dataset class to process.
                    if data: # Simplest check: non-empty response
                         logger.debug(f"Successfully fetched GlyphWiki data for '{character}' (u{ord(character):x}).")
                         return data
                    else:
                         logger.warning(f"GlyphWiki API returned empty data for '{character}' (u{ord(character):x}).")
                         return None # Treat empty data as 'not found' or 'no info'
                except requests.exceptions.JSONDecodeError:
                    logger.error(f"Failed to decode JSON response from GlyphWiki for '{character}'. Response text: {response.text[:200]}...")
                    # Treat as failure for this attempt, maybe retry
                    if attempt == RETRY_COUNT - 1:
                         raise APIError(f"Failed to decode GlyphWiki JSON response for {character} after retries.")
                except Exception as e: # Catch other unexpected errors during processing
                     logger.error(f"Unexpected error processing GlyphWiki response for '{character}': {e}", exc_info=True)
                     if attempt == RETRY_COUNT - 1:
                          raise APIError(f"Unexpected error processing GlyphWiki response for {character} after retries.") from e


            elif response.status_code == 404:
                logger.debug(f"Character '{character}' (u{ord(character):x}) not found on GlyphWiki (404).")
                return None # Character not found is not an error, just no data

            else:
                # Handle other HTTP errors (e.g., 5xx Server Error, 4xx Client Error)
                logger.warning(f"GlyphWiki API request for '{character}' failed with status code {response.status_code}. Attempt {attempt + 1}/{RETRY_COUNT}.")
                # If it's the last attempt, raise an error
                if attempt == RETRY_COUNT - 1:
                     raise APIError(f"GlyphWiki API request failed for {character} with status {response.status_code} after {RETRY_COUNT} attempts.")

        except requests.exceptions.Timeout:
            logger.warning(f"GlyphWiki API request timed out for '{character}'. Attempt {attempt + 1}/{RETRY_COUNT}.")
            if attempt == RETRY_COUNT - 1:
                raise APIError(f"GlyphWiki API request timed out for {character} after {RETRY_COUNT} attempts.")

        except requests.exceptions.RequestException as e:
            # Handle other network errors (DNS failure, connection error, etc.)
            logger.warning(f"Network error during GlyphWiki API request for '{character}': {e}. Attempt {attempt + 1}/{RETRY_COUNT}.")
            if attempt == RETRY_COUNT - 1:
                 raise APIError(f"Network error connecting to GlyphWiki API for {character} after {RETRY_COUNT} attempts: {e}") from e

        # Wait before retrying
        if attempt < RETRY_COUNT - 1:
             time.sleep(RETRY_DELAY)

    # Should not be reached if retries are exhausted (errors are raised), but as fallback:
    return None


# --- Example Usage ---
# if __name__ == "__main__":
#     # Set up basic logging for testing this module
#     logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - [%(name)s] - %(message)s')

#     test_chars = ["猫", "字", "A", "あ", "NotFoundChar"] # Mix of Kanji, non-Kanji, non-existent

#     print("--- Testing GlyphWiki API Fetch ---")
#     for char in test_chars:
#         print(f"\nQuerying for character: '{char}' (u{ord(char):x})")
#         try:
#             data = fetch_glyph_data(char)
#             if data:
#                 print(f"  Success! Received data (showing keys): {list(data.keys())}")
#                 # Example: print(json.dumps(data, indent=2, ensure_ascii=False))
#             elif data is None:
#                  print("  No data returned (possibly not found on GlyphWiki or non-single char).")
#         except APIError as e:
#             print(f"  Caught expected APIError: {e}")
#         except Exception as e:
#             print(f"  Caught unexpected error: {e}")