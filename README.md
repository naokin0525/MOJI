# Handwriting Generation System

Welcome to the Handwriting Generation System! This project offers a powerful and flexible solution for learning and generating SVG-based handwriting using deep learning techniques. It comprises two core components: a **learning program** (`learn.py`) that trains a model on handwritten character datasets, and a **generation program** (`generate.py` and `gui.py`) that creates realistic SVG handwriting from text input. The system supports multiple scripts—such as Latin, Cyrillic, Kanji, Hiragana, and Katakana—and is designed to simulate natural handwriting features like stroke dynamics, styles, and variations.

## Project Overview

This system leverages a hybrid Variational Autoencoder-Generative Adversarial Network (VAE-GAN) model combined with Recurrent Neural Networks (RNNs) to learn and generate handwriting. It can capture intricate details like stroke pressure, speed, and order, and even adapt to personalized handwriting through transfer learning. Whether you’re generating casual notes or formal scripts, this project provides the tools to create high-quality SVG outputs, convertible to formats like PNG, JPG, or even fonts (OTF/TTF).

---

## Project Structure

```
handwriting_project/
├── learn.py             # Script to train the handwriting model
├── generate.py          # Command-line script to generate handwriting
├── gui.py              # GUI application for handwriting generation
├── models/              # Directory for model definitions
│   ├── __init__.py
│   ├── vae_gan.py      # VAE-GAN model implementation
│   └── discriminator.py # Discriminator model for GAN
├── utils/              # Utility scripts
│   ├── __init__.py
│   ├── svg_utils.py    # SVG parsing and generation functions
│   ├── data_utils.py   # Dataset loading and preprocessing
│   └── font_utils.py   # Tools for font conversion
├── datasets/           # Dataset handling
│   ├── __init__.py
│   └── moj_dataset.py  # .moj dataset processing
├── output/             # Directory for generated SVG and converted files
└── README.md           # Project documentation (this file)
```

---

## Dataset Format

The system uses a custom `.moj` format for its dataset, where each file is a JSON object representing a single handwritten character sample.

### Example `.moj` File

```json
{
  "character": "あ",
  "script": "Hiragana",
  "svg": "<svg><path d='M10 20 L30 40' stroke-width='1'/></svg>",
  "strokes": [
    {
      "points": [
        {"x": 10, "y": 20, "pressure": 0.5, "time": 0.1},
        {"x": 30, "y": 40, "pressure": 0.7, "time": 0.2}
      ],
      "order": 1
    }
  ],
  "metadata": {
    "writing_direction": "left-to-right",
    "connected": false
  }
}
```

This format captures both the visual representation (SVG) and the dynamic aspects of handwriting (stroke data).

---

## Learning Program (`learn.py`)

The `learn.py` script trains a VAE-GAN hybrid model with an RNN component to learn handwriting patterns from `.moj` datasets. It models stroke dynamics—pressure, speed, and order—and supports transfer learning for custom handwriting styles.

### Usage

```bash
python learn.py <program_path> --dataset_path <dataset_path> --model_file_name <model_name> --model_path <model_path> --epochs <epochs> --vae_weight <vae_weight> --gan_weight <gan_weight>
```

### Arguments
- `<program_path>`: Path to the program directory.
- `--dataset_path`: Path to the directory containing `.moj` files.
- `--model_file_name` (optional): Name of the saved model file (default: `model.pt`).
- `--model_path`: Directory where the trained model will be saved.
- `--epochs` (optional): Number of training epochs (default: `100`).
- `--vae_weight` (optional): Weight for the VAE loss term (default: `0.5`).
- `--gan_weight` (optional): Weight for the GAN loss term (default: `0.5`).

### Example

```bash
python learn.py C:\handwriting_project --dataset_path "C:\datasets\moj" --model_file_name "handwriting_model.pt" --model_path "C:\models" --epochs 150 --vae_weight 0.6 --gan_weight 0.4
```

---

## Generation Program

### Command-Line Generation (`generate.py`)

```bash
python generate.py <program_path> --model_path <model_path> --text <text> --output <output> --random_variation <true/false> --stroke_width <width> --style <style> --convert_to <format>
```

### GUI Generation (`gui.py`)

Launch the GUI with:

```bash
python gui.py
```

---

## Key Features

- **Multi-Script Support**: Works with Latin, Cyrillic, Kanji, Hiragana, Katakana, and more via Unicode.
- **Stroke Dynamics**: Simulates pressure, speed, and stroke order for realism.
- **Styles**: Choose from casual, formal, or cursive; extendable via latent space tweaks.
- **Personalization**: Fine-tune with user-specific data using transfer learning.
- **Output Flexibility**: Export as SVG, PNG, JPG, or fonts (OTF/TTF).
- **Natural Variations**: Adds jitter and thickness variations for authentic handwriting.

---

## Requirements

To run this project, install the following Python libraries:

### `requirements.txt`
```
torch
torchvision
svgpathtools
cairosvg
Pillow
fontforge
tkinter
```

---

## Installation

Clone the repository:

```bash
git clone <repository_url>
cd handwriting_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare your `.moj` dataset and trained model (or train one using `learn.py`).

Start generating handwriting with `generate.py` or `gui.py`!

---

## Additional Notes

- **Right-to-Left Scripts**: Automatically detected and supported via metadata.
- **Accents & Symbols**: Handled through Unicode character support.
- **Extensibility**: Add new styles or scripts by expanding the dataset and fine-tuning the model.

This project combines cutting-edge AI with practical usability, making it ideal for researchers, hobbyists, and developers interested in handwriting synthesis. Enjoy creating your own digital handwriting!