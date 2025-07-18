"""
Dataset Loader Utility

This script defines the PyTorch Dataset for loading handwriting data.
It processes a directory of raw SVG files, converts them into a numerical format
using svg_parser, and saves the result in a compressed .moj file for fast loading.
"""
import os
import logging
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np

from src.utils.svg_parser import parse_svg_file
from src.config import PROCESSED_DATA_PATH, DATASET_FILE_EXTENSION

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_or_load_dataset(dataset_path: str, force_recreate=False):
    """
    Loads a pre-processed .moj dataset file or creates it from raw SVGs if it doesn't exist.

    Args:
        dataset_path (str): Path to the root directory of raw SVG files (e.g., 'data/raw').
        force_recreate (bool): If True, forces regeneration of the dataset file.

    Returns:
        list: A list of data samples, where each sample is a dictionary
              {'label': str, 'strokes': np.ndarray}.
    """
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    processed_file_path = os.path.join(PROCESSED_DATA_PATH, f"{dataset_name}{DATASET_FILE_EXTENSION}")

    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    if not force_recreate and os.path.exists(processed_file_path):
        logging.info(f"Loading pre-processed dataset from: {processed_file_path}")
        with open(processed_file_path, 'rb') as f:
            return pickle.load(f)

    logging.info("Creating new dataset from raw SVG files...")
    dataset = []
    
    # Walk through subdirectories (e.g., latin, kanji)
    for root, _, files in os.walk(dataset_path):
        for filename in tqdm(files, desc=f"Processing SVGs in {os.path.basename(root)}"):
            if filename.endswith(".svg"):
                file_path = os.path.join(root, filename)
                char_label = os.path.splitext(filename)[0]

                stroke_data = parse_svg_file(file_path)
                if stroke_data is not None:
                    dataset.append({"label": char_label, "strokes": stroke_data})

    if not dataset:
        raise ValueError("No valid SVG files found. Dataset creation failed.")

    logging.info(f"Saving processed dataset to: {processed_file_path}")
    with open(processed_file_path, 'wb') as f:
        pickle.dump(dataset, f)

    return dataset


class HandwritingDataset(Dataset):
    """PyTorch Dataset for handwriting data."""
    def __init__(self, data):
        """
        Args:
            data (list): A list of data samples from create_or_load_dataset.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Convert numpy array to torch tensor for use in the model
        return {
            "label": sample["label"],
            "strokes": torch.from_numpy(sample["strokes"]).float()
        }


def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch to the same length.
    """
    labels = [item['label'] for item in batch]
    strokes = [item['strokes'] for item in batch]
    lengths = torch.tensor([len(s) for s in strokes])

    # Pad stroke sequences
    padded_strokes = pad_sequence(strokes, batch_first=True, padding_value=0.0)

    return {
        'labels': labels,
        'strokes': padded_strokes,
        'lengths': lengths
    }