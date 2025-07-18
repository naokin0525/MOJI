# AI-Powered Handwriting Synthesis

This project provides a complete software package for training deep learning models to generate realistic, human-like handwriting in SVG format. It supports multiple writing systems, including Latin, Kanji, Hiragana, and Katakana, and can be adapted to individual handwriting styles through transfer learning.

## Features

- **Multi-Script Support**: Natively handles Latin, Kanji, and Kana scripts.
- **Realistic Generation**: Simulates stroke order, pressure, speed, and natural variations using a hybrid VAE-GAN architecture.
- **Stroke-Based Learning**: The model learns from stroke data, not just rasterized images, preserving the dynamics of handwriting.
- **Style Transfer**: Fine-tune the model on a small sample of a user's handwriting to capture and replicate their personal style.
- **Flexible Output**: Generates handwriting in SVG, PNG, and JPG formats.
- **Font Generation**: Converts handwriting models into usable OpenType (`.otf`) and TrueType (`.ttf`) fonts.
- **Kanji Analysis**: Integrates with the GlyphWiki API to decompose Kanji characters into their constituent parts, improving learning and generation for complex characters.

## System Architecture

The system is built around a hybrid deep learning model that combines a Variational Autoencoder (VAE) with a Generative Adversarial Network (GAN).

- **Variational Autoencoder (VAE)**: The VAE is used to learn a compressed, latent representation of each character's stroke data. This ensures that the generated handwriting is structurally sound.
- **Recurrent Neural Network (RNN)**: An RNN (specifically, an LSTM or GRU) is used within the VAE to process the sequential nature of handwriting strokes. It learns the order, timing, and flow of writing.
- **Generative Adversarial Network (GAN)**: The GAN is used to refine the output of the VAE, making it appear more realistic and less like a "computer-generated" image. The discriminator learns to distinguish between real and generated handwriting, pushing the generator to create higher-fidelity samples.

### Mathematical Stroke Simulation

To achieve natural variations, we model stroke properties mathematically. A single stroke is a sequence of points $(x_i, y_i, p_i)$, where `p` is pressure. We introduce jitter using a Gaussian noise model:

- **Position Jitter**:
  - $x'_i = x_i + \mathcal{N}(0, \sigma_{xy}^2)$
  - $y'_i = y_i + \mathcal{N}(0, \sigma_{xy}^2)$
- **Pressure Variation**:
  - $p'_i = p_i \cdot (1 + \mathcal{N}(0, \sigma_{p}^2))$

The parameters $\sigma_{xy}$ and $\sigma_{p}$ can be adjusted by the user to control the level of "messiness" or style. Strokes are rendered as smooth Bézier curves connecting these points.

## Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/handwriting-synthesis.git
    cd handwriting-synthesis
    ```

2.  **Create a virtual environment and install dependencies:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

### 1. Training a New Model

Place your SVG dataset files in the `data/raw/{script_name}/` directory. The filename should be the character it represents (e.g., `a.svg`, `b.svg`, `字.svg`).

**Command:**

```sh
python src/learn.py --dataset_path "data/raw" --model_file_name "handwriting_model.pth" --model_path "models" --epochs 100 --vae_weight 0.5 --gan_weight 0.5
```

- `--dataset_path`: Path to the root directory of the raw SVG datasets.
- `--model_file_name`: The name for the output model file.
- `--model_path`: Directory to save the trained model.
- `--epochs`: Number of training epochs.
- `--vae_weight`, `--gan_weight`: Weights for the VAE and GAN loss components.

### 2. Generating Handwriting

**Command:**

```sh
python src/generate.py --model_path "models/handwriting_model.pth" --text "Hello World" --output_path "output/svg/hello.svg" --random_variation 0.1 --stroke_width 1.2
```

- `--model_path`: Path to the trained `.pth` model file.
- `--text`: The text string to generate.
- `--output_path`: The file path for the generated SVG.
- `--random_variation`: A float controlling the amount of randomness/jitter.
- `--stroke_width`: The base width of the strokes in the SVG.
