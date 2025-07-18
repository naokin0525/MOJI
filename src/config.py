"""
Configuration file for the AI-Powered Handwriting Synthesis project.

This file centralizes all static configuration variables, such as file paths,
model hyperparameters, and data processing settings.
"""

import torch

# ----------------------------------------------------------------------------
# General Settings
# ----------------------------------------------------------------------------
# Determine the available device for PyTorch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# File and Directory Paths
# ----------------------------------------------------------------------------
# Note: These are relative paths from the project root.
# The command-line arguments will override these defaults.
DATA_DIR = "data"
RAW_DATA_PATH = f"{DATA_DIR}/raw"
PROCESSED_DATA_PATH = f"{DATA_DIR}/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "output"
SVG_OUTPUT_DIR = f"{OUTPUT_DIR}/svg"
PNG_OUTPUT_DIR = f"{OUTPUT_DIR}/png"
FONT_OUTPUT_DIR = f"{OUTPUT_DIR}/fonts"

# ----------------------------------------------------------------------------
# Data Processing and Dataset Configuration
# ----------------------------------------------------------------------------
# The custom dataset format extension
DATASET_FILE_EXTENSION = ".moj"

# SVG parsing settings
# Max number of points to sample per stroke. Strokes with more points will be downsampled.
MAX_POINTS_PER_STROKE = 100
# A small constant to avoid division by zero
EPSILON = 1e-6

# GlyphWiki API settings for Kanji analysis
GLYPHWIKI_API_URL = "https://glyphwiki.org/api/relation"

# ----------------------------------------------------------------------------
# Model Hyperparameters
# ----------------------------------------------------------------------------
# Shared parameters
INPUT_DIM = (
    5  # Each point is represented by (Δx, Δy, pressure, pen_down, end_of_stroke)
)
LATENT_DIM = 128  # Size of the latent space vector

# RNN (LSTM/GRU) settings
RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.2

# VAE-GAN component weights
# These can be adjusted via command-line args in learn.py
VAE_LOSS_WEIGHT = 0.5  # Weight for VAE reconstruction loss
GAN_LOSS_WEIGHT = 0.5  # Weight for GAN adversarial loss
KLD_WEIGHT = 0.1  # Weight for Kullback-Leibler divergence in VAE loss

# ----------------------------------------------------------------------------
# Training Configuration
# ----------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE_GENERATOR = 1e-4
LEARNING_RATE_DISCRIMINATOR = 1e-4
BETA1 = 0.5  # Adam optimizer parameter
BETA2 = 0.999  # Adam optimizer parameter

# Logging and saving frequency
LOG_INTERVAL = 10  # Log training status every 10 batches
SAVE_MODEL_EPOCH_INTERVAL = 5  # Save a model checkpoint every 5 epochs

# ----------------------------------------------------------------------------
# Generation Configuration
# ----------------------------------------------------------------------------
# Default values for the generation script
DEFAULT_STROKE_WIDTH = 1.0
DEFAULT_RANDOM_VARIATION = 0.05  # Controls jitter and style variation
DEFAULT_LINE_HEIGHT = 100  # Vertical distance between lines of text
DEFAULT_CHAR_SPACING = 10  # Horizontal distance between characters

# Image conversion settings
PNG_SCALE_FACTOR = 10  # Scale factor for rendering SVG to PNG for higher quality

# Font generation settings
FONT_FAMILY_NAME = "AIHandwriting"
FONT_STYLE_NAME = "Regular"
FONT_UNITS_PER_EM = 1024
