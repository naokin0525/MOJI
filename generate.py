import argparse
import torch
from models.vae_gan import HandwritingModel
from utils.svg_utils import sequence_to_svg, arrange_svgs
from utils.font_utils import svgs_to_font

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SVG handwriting")
    parser.add_argument("program_path", type=str, help="Program directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument("--output", type=str, default="output/output.svg", help="Output SVG file")
    parser.add_argument("--random_variation", type=bool, default=True, help="Add random variation")
    parser.add_argument("--stroke_width", type=float, default=1.2, help="Stroke width")
    parser.add_argument("--style", type=str, default="casual", choices=["casual", "formal", "cursive"], help="Handwriting style")
    parser.add_argument("--convert_to", type=str, choices=["png", "jpg", "otf", "ttf"], help="Convert output to format")
    return parser.parse_args()

def generate_handwriting(model, text, char_to_id, device, args):
    model.eval()
    svgs = []
    chars = []
    for char in text:
        if char not in char_to_id:
            continue
        char_id = torch.tensor([char_to_id[char]], device=device)
        z = torch.randn(1, 64, device=device) if args.random_variation else torch.zeros(1, 64, device=device)
        sequence = model.decode(model.embedding(char_id), z, sequence_length=50)
        svg = sequence_to_svg(sequence[0], stroke_width=args.stroke_width)
        svgs.append(svg)
        chars.append(char)
    
    direction = 'right-to-left' if any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in text) else 'left-to-right'
    final_svg = arrange_svgs(svgs, writing_direction=direction)
    
    with open(args.output, 'w') as f:
        f.write(final_svg)
    
    if args.convert_to in ['png', 'jpg']:
        from cairosvg import svg2png, svg2jpg
        (svg2png if args.convert_to == 'png' else svg2jpg)(url=args.output, write_to=args.output.replace('.svg', f'.{args.convert_to}'))
    elif args.convert_to in ['otf', 'ttf']:
        svgs_to_font(svgs, chars, args.output.replace('.svg', ''), font_format=args.convert_to)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load char_to_id mapping (assumed saved with model)
    char_to_id = torch.load(f"{args.model_path}/char_to_id.pt")  # Save this during training
    model = HandwritingModel(char_vocab_size=len(char_to_id), embedding_dim=50, hidden_dim=256, latent_dim=64).to(device)
    model.load_state_dict(torch.load(f"{args.model_path}/model.pt"))
    
    generate_handwriting(model, args.text, char_to_id, device, args)

if __name__ == "__main__":
    main()