"""
GlyphWiki API Utility

This script provides a function to interact with the GlyphWiki API
for analyzing the structure of Kanji characters. It helps the model
understand that complex characters are composed of simpler radicals.
"""

import requests
import logging

from src.config import GLYPHWIKI_API_URL

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_kanji_components(kanji_character: str):
    """
    Queries the GlyphWiki API to get the component parts of a Kanji character.

    Args:
        kanji_character (str): A single Kanji character.

    Returns:
        list[str] or None: A list of component characters if the query is successful,
                           otherwise None.
    """
    if len(kanji_character) != 1:
        logging.warning("get_kanji_components expects a single character.")
        return None

    try:
        # The API uses the character's name in the URL, e.g., u8c4a for '豊'
        char_code = f"u{ord(kanji_character):x}"
        params = {"name": char_code}

        response = requests.get(GLYPHWIKI_API_URL, params=params, timeout=5)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        # The response is a string like "u8c4a:u8c4a-g,u8c4a-k,u8c4a-t"
        # We are interested in parsing this to find relationships
        # A more complex parser would be needed for a full implementation.
        # For this example, we just confirm connectivity.

        # A full implementation would parse response.text to find component glyphs.
        # For example, if it returns "u5c71:u4e00,u4e28", it means '山' is composed of '一' and '丨'.
        # This is a placeholder for that logic.
        if response.ok and ":" in response.text:
            parts = response.text.strip().split(":")[1].split(",")
            # Convert unicode strings back to characters
            components = [chr(int(p[1:], 16)) for p in parts if p.startswith("u")]
            logging.info(
                f"Components for '{kanji_character}' ({char_code}): {components}"
            )
            return components
        return None

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for character '{kanji_character}': {e}")
        return None
