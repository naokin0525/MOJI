# src/conversion/to_raster.py
"""
Utility to convert SVG files or data to raster image formats (PNG, JPG).

Uses the 'cairosvg' library for rendering and 'Pillow' for JPG conversion.
Requires external dependencies: `pip install cairosvg Pillow`
May also require Cairo graphics library installed on the system (e.g., via package manager).
"""

import cairosvg
from PIL import Image
import io
import os
import logging

# Import custom exception
try:
    # Use relative import within the same package level
    from ..utils.error_handling import ConversionError
except ImportError:
    # Fallback if structure isn't fully set up during development
    try:
        from src.utils.error_handling import ConversionError
    except ImportError:
        class ConversionError(Exception):
            pass

logger = logging.getLogger(__name__)

def convert_svg_to_raster(svg_input: str | bytes,
                          output_filepath: str,
                          output_format: str = 'png',
                          dpi: int = 150,
                          background_color: str | None = 'white') -> None:
    """
    Converts SVG input (file path, string, or bytes) to PNG or JPG format.

    Args:
        svg_input (str | bytes): Path to the input SVG file, or SVG content as string/bytes.
        output_filepath (str): Path to save the output raster image.
        output_format (str): Desired output format ('png' or 'jpg'). Case-insensitive.
        dpi (int): Dots per inch, controls the output resolution.
        background_color (str | None): CSS-style background color (e.g., 'white', '#FF0000').
                                       Needed for formats without transparency like JPG.
                                       If None, background is transparent (for PNG).

    Raises:
        ConversionError: If conversion fails due to library errors, invalid input,
                         or unsupported formats.
        FileNotFoundError: If svg_input is a path and the file doesn't exist.
    """
    output_format = output_format.lower()
    if output_format not in ['png', 'jpg', 'jpeg']:
        raise ConversionError(f"Unsupported output format: {output_format}. Use 'png' or 'jpg'.")

    svg_data = None
    input_desc = "" # For logging

    # --- Read SVG Input ---
    try:
        if isinstance(svg_input, str) and os.path.exists(svg_input):
            input_desc = f"file '{svg_input}'"
            with open(svg_input, 'rb') as f: # Read as bytes
                svg_data = f.read()
        elif isinstance(svg_input, str):
            input_desc = "SVG string"
            svg_data = svg_input.encode('utf-8') # Convert string to bytes
        elif isinstance(svg_input, bytes):
            input_desc = "SVG bytes"
            svg_data = svg_input
        else:
            raise ConversionError("Invalid svg_input type. Expected file path (str), SVG content (str), or bytes.")

        if not svg_data:
             raise ConversionError("SVG input data is empty.")

    except FileNotFoundError as e:
         logger.error(f"Input SVG file not found: {svg_input}")
         raise e # Re-raise FileNotFoundError
    except Exception as e:
         logger.error(f"Error reading SVG input '{svg_input}': {e}", exc_info=True)
         raise ConversionError(f"Failed to read SVG input: {e}") from e

    logger.info(f"Converting SVG ({input_desc}) to {output_format.upper()} at {output_filepath} (DPI: {dpi})")

    # --- Perform Conversion ---
    try:
        if output_format == 'png':
            cairosvg.svg2png(
                bytestring=svg_data,
                write_to=output_filepath,
                dpi=dpi,
                background_color=background_color,
                # parent_width=, parent_height= # Can specify explicit dimensions if needed
            )
            logger.info(f"Successfully saved PNG to {output_filepath}")

        elif output_format in ['jpg', 'jpeg']:
            # CairoSVG doesn't directly output JPG well in all versions/setups.
            # Render to PNG in memory, then use Pillow to convert and save as JPG.
            png_bytes = cairosvg.svg2png(
                bytestring=svg_data,
                write_to=None, # Render to memory
                dpi=dpi,
                background_color=background_color if background_color else 'white' # JPG needs background
            )

            if not png_bytes:
                 raise ConversionError("CairoSVG rendering to PNG bytes failed.")

            with Image.open(io.BytesIO(png_bytes)) as img:
                # Ensure image is in RGB mode for JPG saving
                if img.mode == 'RGBA' or img.mode == 'P':
                    # Create a new image with the specified background color if needed
                    # Pillow requires RGB for background color fill typically
                    bg_color_rgb = 'white' # Default fallback
                    if background_color:
                        try:
                            from PIL import ImageColor
                            bg_color_rgb = ImageColor.getrgb(background_color)
                        except ValueError:
                            logger.warning(f"Invalid background color '{background_color}'. Defaulting to white.")


                    background = Image.new('RGB', img.size, bg_color_rgb)
                    # Paste the RGBA image onto the RGB background using alpha channel as mask
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
                        final_img = background
                    else: # Handle indexed color 'P' - convert first
                         final_img = img.convert('RGB')
                elif img.mode == 'RGB':
                     final_img = img
                else: # Grayscale 'L', etc.
                    final_img = img.convert('RGB')

                # Save as JPEG
                final_img.save(output_filepath, 'JPEG', quality=95) # Adjust quality as needed
            logger.info(f"Successfully saved JPG to {output_filepath}")

    except ImportError as e:
         logger.error(f"Missing required library for conversion: {e}. Please install cairosvg and Pillow.")
         raise ConversionError(f"Missing dependency: {e}. Ensure cairosvg and Pillow are installed.") from e
    except Exception as e:
        # Catch potential errors from cairosvg or Pillow
        logger.error(f"Error during SVG to {output_format.upper()} conversion: {e}", exc_info=True)
        # Clean up potentially partially created file?
        if os.path.exists(output_filepath):
             try: os.remove(output_filepath)
             except OSError: pass
        raise ConversionError(f"Conversion failed: {e}") from e

# --- Example Usage ---
# if __name__ == "__main__":
#     # Create a dummy SVG file for testing
#     dummy_svg = '<svg height="50" width="150"><text x="10" y="35" font-size="30" fill="blue">Test SVG</text></svg>'
#     svg_filename = "test_conversion.svg"
#     with open(svg_filename, "w") as f:
#         f.write(dummy_svg)

#     # Set up basic logging
#     logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

#     output_dir = "conversion_output"
#     os.makedirs(output_dir, exist_ok=True)
#     png_output = os.path.join(output_dir, "output.png")
#     jpg_output = os.path.join(output_dir, "output.jpg")

#     print(f"\n--- Testing SVG to PNG ---")
#     try:
#         convert_svg_to_raster(svg_filename, png_output, format='png', dpi=100, background_color=None)
#     except Exception as e:
#         print(f"  PNG Conversion failed: {e}")

#     print(f"\n--- Testing SVG to JPG ---")
#     try:
#         convert_svg_to_raster(svg_filename, jpg_output, format='jpg', dpi=100, background_color='lightgray')
#     except Exception as e:
#         print(f"  JPG Conversion failed: {e}")

#     # Clean up
#     # os.remove(svg_filename)
#     # import shutil
#     # shutil.rmtree(output_dir)
#     # print("\nCleaned up test files.")