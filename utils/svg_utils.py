import torch
from svgpathtools import Path, Line
from cairosvg import svg2png
from PIL import Image
import numpy as np

def sequence_to_svg(sequence, stroke_width=1.2):
    x, y = 0, 0
    path_str = ""
    for dx, dy, pressure, pen_up in sequence:
        x_new = x + dx
        y_new = y + dy
        width = stroke_width * pressure
        if not pen_up:
            path_str += f"M{x} {y} L{x_new} {y_new} "
        x, y = x_new, y_new
    svg = f'<svg><path d="{path_str}" stroke="black" stroke-width="{width}" fill="none"/></svg>'
    return svg

def sequence_to_image(sequence):
    svg = sequence_to_svg(sequence)
    svg2png(bytestring=svg, write_to='temp.png', output_width=64, output_height=64)
    img = Image.open('temp.png').convert('L')
    return torch.tensor(np.array(img) / 255.0, dtype=torch.float32).unsqueeze(0)

def arrange_svgs(svgs, writing_direction='left-to-right', spacing=10):
    total_width = sum(len(svg) * 10 for svg in svgs) + spacing * (len(svgs) - 1)
    paths = []
    x_offset = 0
    for svg in (svgs if writing_direction == 'left-to-right' else reversed(svgs)):
        paths.append(f'<path d="{svg.split("d=")[1].split(" stroke")[0]}" stroke="black" stroke-width="1.2" transform="translate({x_offset},0)"/>')
        x_offset += 10 + spacing
    return f'<svg width="{total_width}" height="64">{"".join(paths)}</svg>'