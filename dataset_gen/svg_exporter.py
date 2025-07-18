"""
Handles the conversion of drawn strokes into an SVG file format.
"""
from PyQt5.QtCore import QPoint

def strokes_to_svg(strokes: list[list[QPoint]], width: int, height: int) -> str:
    """
    Converts a list of strokes (each a list of QPoints) into an SVG string.

    Args:
        strokes (list[list[QPoint]]): The drawing data.
        width (int): The width of the SVG canvas.
        height (int): The height of the SVG canvas.

    Returns:
        str: A string containing the complete SVG file content.
    """
    svg_paths = []
    for stroke in strokes:
        if not stroke:
            continue
        
        # Start the path with a "Move To" command
        start_point = stroke[0]
        path_d = f"M {start_point.x()} {start_point.y()}"
        
        # Continue the path with "Line To" commands
        for point in stroke[1:]:
            path_d += f" L {point.x()} {point.y()}"
            
        # Define the SVG path element with styling
        svg_paths.append(
            f'<path d="{path_d}" stroke="black" stroke-width="2" fill="none" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    # Combine all paths into a single SVG document
    svg_content = (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
        + "\n".join(svg_paths)
        + "\n</svg>"
    )
    
    return svg_content