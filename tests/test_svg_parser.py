"""
Unit Tests for the SVG Parser utility.

This test suite verifies the functionality of src.utils.svg_parser.
It checks for correct parsing of valid SVG files and robust handling of
errors, such as missing files or malformed content.
"""

import unittest
import os
import tempfile
import numpy as np

# Add project root to the Python path to allow importing from 'src'
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.svg_parser import parse_svg_file


class TestSVGParser(unittest.TestCase):

    def setUp(self):
        """Set up a temporary SVG file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.svg_file_path = os.path.join(self.temp_dir.name, "test.svg")

        # A simple SVG with one path
        svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
            <path d="M10 10 L90 90" stroke="black"/>
        </svg>"""

        with open(self.svg_file_path, "w") as f:
            f.write(svg_content)

    def tearDown(self):
        """Clean up the temporary directory and file."""
        self.temp_dir.cleanup()

    def test_parse_valid_svg(self):
        """Test that a valid SVG file is parsed correctly."""
        stroke_data = parse_svg_file(self.svg_file_path)

        # Check that the output is a numpy array
        self.assertIsInstance(stroke_data, np.ndarray)

        # Check that the array is not empty
        self.assertGreater(stroke_data.shape[0], 0)

        # Check that the array has the correct number of features (5)
        self.assertEqual(stroke_data.shape[1], 5)

        # Check that the end-of-stroke flag is set correctly on the last point
        self.assertEqual(stroke_data[-1, 4], 1.0)

    def test_parse_nonexistent_file(self):
        """Test that parsing a non-existent file returns None and logs an error."""
        nonexistent_path = os.path.join(self.temp_dir.name, "no_such_file.svg")

        with self.assertLogs("src.utils.svg_parser", level="ERROR") as cm:
            stroke_data = parse_svg_file(nonexistent_path)
            self.assertIsNone(stroke_data)
            self.assertTrue(any("Failed to parse" in log for log in cm.output))

    def test_parse_svg_with_no_paths(self):
        """Test that an SVG with no <path> elements returns None."""
        empty_svg_path = os.path.join(self.temp_dir.name, "empty.svg")
        empty_svg_content = """<svg xmlns="http://www.w3.org/2000/svg"></svg>"""

        with open(empty_svg_path, "w") as f:
            f.write(empty_svg_content)

        stroke_data = parse_svg_file(empty_svg_path)
        self.assertIsNone(stroke_data)


if __name__ == "__main__":
    unittest.main()
