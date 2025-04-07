# AI SVG Handwriting Generator

## Overview

This project provides a complete software package for training deep learning models and generating realistic handwritten character datasets and text output in SVG format. It leverages AI, specifically a VAE-GAN architecture combined with sequence models (RNN/Transformer), to learn and replicate nuanced handwriting patterns, including stroke order, pressure variations (simulated), and speed variations (simulated).

The system supports multiple writing systems, including Latin, Cyrillic, and Japanese (Kanji, Hiragana, Katakana), using data formats like individual SVG files per character or the custom `.moj` format.

## Features

* **Deep Learning Model:** Utilizes a hybrid VAE-GAN architecture for high-quality, realistic generation.
* **Sequence Modeling:** Employs RNNs (LSTM/GRU) or Transformers to capture the sequential nature of handwriting strokes.
* **Multi-Script Support:** Designed to train on and generate handwriting for various scripts (Latin, Cyrillic, Hiragana, Katakana, Kanji). Learns script-specific quirks.
* **Rich Stroke Data Learning:** Aims to capture stroke pressure, speed, order, and natural variations during training (requires suitable dataset format like `.moj` or simulation).
* **Realistic Generation:** Generates handwriting with simulated jitter, pressure, and speed variations. Supports different styles (e.g., 'casual', 'formal').
* **SVG Output:** Primarily generates output in scalable SVG format, preserving stroke information.
* **Raster Output:** Optionally converts generated SVGs to PNG or JPG formats.
* **Font Conversion Tools:** Includes utilities (`fontTools`-based) to compile generated character SVGs into installable OpenType (`.ttf`) font files.
* **Dataset Flexibility:** Accepts datasets composed of individual `.svg` files (named by character) or the `.moj` JSON format.
* **GlyphWiki Integration:** Optionally uses the GlyphWiki API during training for Kanji structure analysis (requires internet).
* **Transfer Learning:** Supports fine-tuning pre-trained models on personalized handwriting samples (via checkpoint loading).
* **Command-Line Interface:** Full functionality accessible via CLI scripts (`learn.py`, `generate.py`).
* **Optional GUI:** Provides a basic graphical interface (`src/ui/gui.py`) for easier generation using Tkinter.

## Project Structure

```
svg_handwriting_generator/
├── config/
│   └── default_config.yaml   # Default configuration parameters
├── data/                     # Placeholder for datasets (not included)
├── models/                   # Placeholder for saved trained models
├── output/                   # Default output directory for generated files
├── scripts/
│   ├── learn.py              # Main training script
│   └── generate.py           # Main generation script
├── src/
│   ├── data/                 # Data loading, parsing (SVG, MOJ), preprocessing, GlyphWiki API
│   ├── models/               # VAE-GAN model definitions (Encoder, Decoder, Discriminator, Sequence Models)
│   ├── training/             # Training loop logic, loss functions, optimizers
│   ├── generation/           # Generation orchestration, style control, stroke simulation
│   ├── conversion/           # SVG -> Raster (PNG/JPG) and SVG -> Font (TTF) utilities
│   ├── ui/                   # Optional Tkinter GUI application
│   └── utils/                # Shared utilities (logging, config, error handling)
├── README.md                 # This documentation file
└── requirements.txt          # Python package dependencies
```

## Installation

**1. Prerequisites:**

* **Python:** Version 3.8 or higher recommended.
* **Git:** To clone the repository.
* **Cairo Graphics Library (Optional but Recommended):** Required by `cairosvg` for SVG to PNG/JPG conversion and GUI preview. Installation varies by OS:
    * **Debian/Ubuntu:** `sudo apt-get update && sudo apt-get install libcairo2-dev pkg-config python3-dev`
    * **macOS (using Homebrew):** `brew install cairo pkg-config`
    * **Windows:** Download pre-compiled binaries or use a package manager like MSYS2/MinGW to install Cairo and pkg-config. Ensure the necessary DLLs are in the system PATH. This can sometimes be challenging.

**2. Clone Repository:**

```bash
git clone [https://github.com/your-username/ai-svg-handwriting-generator.git](https://www.google.com/search?q=https://github.com/your-username/ai-svg-handwriting-generator.git) # Replace with actual URL
cd ai-svg-handwriting-generator
```

**3. Set up Virtual Environment (Recommended):**

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

**4. Install Dependencies:**

```bash
pip install -r requirements.txt
```

## Dataset Preparation

* Place your handwriting datasets in a designated directory (e.g., create a `data/` directory or use any path).
* **SVG Format:** Create subdirectories within your dataset path, one for each character set if desired. Place individual SVG files named by the character they represent (e.g., `a.svg`, `b.svg`, `猫.svg`, `uniFF21.svg`). The system uses the filename (without extension) as the character label.
* **MOJ Format:** Place `.moj` files (assumed JSON format, see `src/data/moj_parser.py` for expected structure) in the dataset directory. The parser attempts to read `"label"` or `"character"` keys for labels, falling back to the filename. The format should contain stroke data (either point lists with x, y, pressure, time, or raw SVG path strings).
* **GlyphWiki:** If using `--use_glyphwiki` during training, ensure an internet connection is available.

## Usage

### 1. Training (`learn.py`)

Use the `learn.py` script to train the VAE-GAN model on your dataset.

**Example:**

```bash
python scripts/learn.py C:\program\path \
    --dataset_path "C:\path\to\your\datasets\svg_latin" \
    --model_path "C:\path\to\save\models" \
    --model_file_name "latin_handwriting_model" \
    --config_file "config/default_config.yaml" \
    --epochs 150 \
    --batch_size 64 \
    --learning_rate_g 0.0001 \
    --vae_weight 1.0 \
    --gan_weight 0.3 \
    --kl_weight 0.05 \
    --sequence_model_type transformer \
    --device cuda
```

*(Note: The first `C:\program\path` argument in the example seems potentially redundant depending on script execution context, primary paths are `--dataset_path` and `--model_path`)*

**Key Arguments:**

* `--dataset_path`: Path to the training data directory.
* `--model_path`: Directory to save model checkpoints and the final model.
* `--model_file_name`: Base name for saved model files (e.g., `mymodel`). Checkpoints will be named like `mymodel_epoch_10.pth`.
* `--config_file`: Path to YAML configuration file (defaults override CLI args).
* `--epochs`, `--batch_size`, `--learning_rate_g`, `--learning_rate_d`: Training hyperparameters.
* `--vae_weight`, `--gan_weight`, `--kl_weight`: Weights for balancing loss components.
* `--sequence_model_type`: 'rnn' or 'transformer'.
* `--device`: 'cpu', 'cuda', or 'auto'.
* `--checkpoint_path`: Path to a checkpoint to resume training or fine-tune.
* `--use_glyphwiki`: Flag to enable GlyphWiki API calls for Kanji.
* *(See `learn.py --help` for all options)*

### 2. Generation (`generate.py`)

Use the `generate.py` script to generate handwriting from text using a trained model.

**Example:**

```bash
python scripts/generate.py \
    --model_path "C:\path\to\save\models\latin_handwriting_model_best.pth" \
    --text "Hello World! This is generated handwriting." \
    --output_path "output_generated" \
    --style "casual" \
    --random_variation \
    --stroke_width 1.1 \
    --output_format png
```

**Key Arguments:**

* `--model_path`: Path to the trained model file (`.pth`).
* `--text`: The input text string.
* `--output_path`: Directory to save the generated SVG/PNG/JPG file(s).
* `--style`: Desired style preset name (e.g., 'default', 'casual', 'formal', 'cursive'). See `src/generation/style_control.py`.
* `--random_variation`: Flag to enable random noise scaling during generation for variability.
* `--stroke_width`: Base width for strokes in the output SVG.
* `--output_format`: 'svg', 'png', or 'jpg'.
* `--num_samples`: Generate multiple variations of the same text.
* `--seed`: Integer for reproducible randomness.
* *(See `generate.py --help` for all options)*

### 3. Graphical User Interface (GUI)

An optional Tkinter-based GUI is provided for interactive generation.

**Launch:**

```bash
python src/ui/gui.py
```

**Features:**

* Browse and select model files.
* Enter text input.
* Choose styles and output formats from dropdowns.
* Adjust parameters like stroke width and variation.
* Specify output directory.
* Click "Generate" to start the process (runs in background).
* View a preview of the generated output.
* See status messages.

*(Note: Requires `Pillow` and `cairosvg` with its system dependencies for preview and PNG/JPG output).*

### 4. Font Conversion (`to_font.py`)

The `src/conversion/to_font.py` module provides a function `convert_svgs_to_font` that can be used programmatically or via a separate script (not provided) to compile a set of character SVGs into a `.ttf` font file.

**Example (Programmatic):**

```python
from src.conversion import convert_svgs_to_font

svg_map = {
    'A': 'path/to/A.svg',
    'B': 'path/to/B.svg',
    # ... map all desired characters to their SVG files
}
output_font = "GeneratedFont.ttf"

try:
    convert_svgs_to_font(svg_map, output_font, font_name="MyHandwriting")
except Exception as e:
    print(f"Font conversion failed: {e}")

```

*(Note: This tool provides basic TTF generation using `fontTools`. Creating high-quality fonts with proper metrics and advanced features is complex and may require refinement or manual post-processing in font editing software.)*

## Configuration

Default parameters for training and generation can be found in `config/default_config.yaml`. These values are used if not overridden by command-line arguments when running `learn.py` or `generate.py`.

## Technology Stack

* **Language:** Python 3
* **Deep Learning:** PyTorch
* **Configuration:** PyYAML
* **SVG Rasterization:** cairosvg (+ system Cairo library)
* **Image Handling:** Pillow (PIL fork)
* **Font Generation:** fontTools (+ cu2qu)
* **API Requests:** requests
* **GUI (Optional):** Tkinter (built-in)
* **Numerical:** NumPy
* **Simplification (Optional):** RDP

## Limitations & Future Work

* **Font Generation:** The current `to_font.py` implementation is basic. It needs significant enhancements for robust SVG path conversion (especially complex curves), automatic calculation of typographic metrics (side bearings, advance widths, kerning), and support for OpenType features.
* **Layout:** The `generate.py` script currently overlays characters at the origin in the final SVG. Proper text layout requires calculating glyph bounds and positioning characters sequentially with appropriate spacing/kerning.
* **Style Control:** Style is primarily controlled via simulation parameters. More advanced style control could involve conditioning the model on style embeddings or interpolating in the latent space.
* **Cursive/Connected Script:** Generating connected scripts requires additional logic to identify connection points and modify stroke generation accordingly.
* **Data Augmentation:** Currently not implemented but could improve model robustness.
* **Conditional Generation:** The generator currently samples `z` unconditionally. Conditioning generation on the specific input character (beyond just iterating through text) could improve consistency.