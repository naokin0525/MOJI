# src/data/dataset.py
"""
PyTorch Dataset and DataLoader definitions for handwriting data.

Handles loading from .svg or .moj files, preprocessing, optional GlyphWiki
integration, sequence conversion, and batching.
"""

import os
import glob
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# Import local data handling modules and utils
try:
    from .svg_utils import parse_svg_file, normalize_strokes, strokes_to_sequence_tensor, simplify_strokes
    from .moj_parser import parse_moj_file
    from .glyphwiki_api import fetch_glyph_data
    from ..utils.error_handling import DataError, APIError
except ImportError:
    # Define minimal fallbacks if modules aren't available yet
     class DataError(Exception): pass
     class APIError(Exception): pass
     def parse_svg_file(fp, bp): return []
     def normalize_strokes(s, ts, ka, m): return s
     def strokes_to_sequence_tensor(s, msl, n, ns, ipt): return None
     def simplify_strokes(s, t): return s
     def parse_moj_file(fp, bp): return None, None
     def fetch_glyph_data(c): return None

logger = logging.getLogger(__name__)

class HandwritingDataset(Dataset):
    """
    PyTorch Dataset for loading handwriting data from SVG or MOJ files.

    Args:
        dataset_path (str): Path to the directory containing data files.
        data_format (str): Type of data files ('svg' or 'moj').
        max_seq_len (int): Maximum length of stroke sequences. Longer sequences
                           will be truncated, shorter ones padded.
        normalization_size (int): Target size for stroke normalization (e.g., 256).
        use_glyphwiki (bool): Whether to fetch and use GlyphWiki data (for Kanji).
        glyphwiki_cache (dict): A dictionary to cache GlyphWiki results (optional).
        simplify_tolerance (float | None): Tolerance for RDP stroke simplification.
                                           Set to None or 0 to disable.
        bezier_points (int): Number of points to sample for Bezier curves in SVGs.
        include_pressure_time (bool): If True, expects/uses pressure&time data
                                      (primarily from .moj files). Changes tensor dim.
        data_augmentation (dict | None): Configuration for data augmentation (e.g.,
                                        {'rotate': 5.0, 'scale': 0.1}). NYI.
    """
    def __init__(self,
                 dataset_path: str,
                 data_format: str = 'svg',
                 max_seq_len: int = 512,
                 normalization_size: int = 256,
                 use_glyphwiki: bool = False,
                 glyphwiki_cache: dict | None = None,
                 simplify_tolerance: float | None = None,
                 bezier_points: int = 10,
                 include_pressure_time: bool = False,
                 data_augmentation: dict | None = None): # Not Yet Implemented

        super().__init__()

        if not os.path.isdir(dataset_path):
            raise DataError(f"Dataset directory not found or is not a directory: {dataset_path}")

        self.dataset_path = dataset_path
        self.data_format = data_format.lower()
        self.max_seq_len = max_seq_len
        self.normalization_size = normalization_size
        self.use_glyphwiki = use_glyphwiki
        self.glyphwiki_cache = glyphwiki_cache if glyphwiki_cache is not None else {}
        self.simplify_tolerance = simplify_tolerance
        self.bezier_points = bezier_points
        self.include_pressure_time = include_pressure_time # Determines if loaded points have p/t
        self._points_have_pressure_time = False # Internal flag set based on first loaded sample
        self.data_augmentation = data_augmentation # NYI

        self.file_paths = []
        self.labels = []

        # --- Input dimension consistency check ---
        # If using pressure/time, but format is SVG (which lacks it), log warning.
        if self.include_pressure_time and self.data_format == 'svg':
             logger.warning("`include_pressure_time` is True, but `data_format` is 'svg'. "
                           "SVG files lack pressure/time; default values will be used.")
             # Proceed, but defaults (0.5 pressure, 0.0 time) will be used in tensor conversion.

        # Find all data files
        search_pattern = os.path.join(self.dataset_path, f'*.{self.data_format}')
        self.file_paths = sorted(glob.glob(search_pattern))

        if not self.file_paths:
            raise DataError(f"No .{self.data_format} files found in {self.dataset_path}")

        # Extract labels from filenames (can be overridden by parser if available)
        # Assumes filename is the character itself (e.g., 'a.svg', '猫.svg', 'U+732B.svg')
        for fp in self.file_paths:
            base_name = os.path.basename(fp)
            label, _ = os.path.splitext(base_name)
            # TODO: Handle potential URL encoding in filenames if needed
            # label = urllib.parse.unquote(label)
            self.labels.append(label)

        logger.info(f"Initialized HandwritingDataset: Found {len(self.file_paths)} samples in '{dataset_path}'.")
        logger.info(f"Dataset params: format={self.data_format}, max_len={max_seq_len}, norm_size={self.normalization_size}, use_glyphwiki={self.use_glyphwiki}, simplify={self.simplify_tolerance}, pressure_time={self.include_pressure_time}")

        # Check format of first sample to set internal flag
        self._check_first_sample_format()

    def _check_first_sample_format(self):
        """Loads the first sample to determine if points include pressure/time."""
        if not self.file_paths: return
        try:
            _, strokes = self._load_raw_strokes(0)
            if strokes and strokes[0] and len(strokes[0][0]) >= 4:
                self._points_have_pressure_time = True
                logger.info("Detected pressure/time data in the first data sample.")
                if not self.include_pressure_time:
                     logger.warning("Dataset seems to contain pressure/time, but `include_pressure_time` is False. This data will be ignored.")
            else:
                 self._points_have_pressure_time = False
                 if self.include_pressure_time:
                      logger.warning("Dataset does not seem to contain pressure/time, but `include_pressure_time` is True. Default values (p=0.5, t=0.0) will be used.")

        except Exception as e:
            logger.error(f"Failed to check format of first sample: {e}", exc_info=True)
            # Assume no pressure/time if check fails
            self._points_have_pressure_time = False

    def _load_raw_strokes(self, idx: int) -> tuple[str | None, list[list[tuple]] | None]:
        """Loads raw stroke data for a given index using appropriate parser."""
        file_path = self.file_paths[idx]
        label = self.labels[idx] # Use filename label as default

        try:
            if self.data_format == 'moj':
                parsed_label, strokes = parse_moj_file(file_path, self.bezier_points)
                # Use label from file if parsing was successful and provided one
                if parsed_label is not None: label = parsed_label
                return label, strokes
            elif self.data_format == 'svg':
                # SVG parsing returns list of strokes; points are (x, y)
                strokes = parse_svg_file(file_path, self.bezier_points)
                # Convert (x,y) to (x,y,p,t) with defaults if needed later
                return label, strokes
            else:
                raise DataError(f"Unsupported data format: {self.data_format}")
        except DataError as e:
            logger.error(f"Failed to load or parse data file {file_path}: {e}")
            return label, None # Return label but None for strokes on error
        except Exception as e:
            logger.error(f"Unexpected error loading index {idx} ({file_path}): {e}", exc_info=True)
            return label, None


    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, torch.Tensor | None]:
        """
        Fetches, preprocesses, and returns a single data sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, str, torch.Tensor | None]:
                - sequence_tensor: Preprocessed stroke data as a tensor (max_seq_len, feature_dim).
                                   Returns zeros tensor if loading/processing fails.
                - label: The character label string.
                - glyph_tensor: Optional tensor representing GlyphWiki data (if enabled and successful).
                                Returns None otherwise. Format TBD based on model needs.
        """
        if idx < 0 or idx >= len(self.file_paths):
            raise IndexError(f"Index {idx} out of bounds for dataset with size {len(self.file_paths)}")

        label, raw_strokes = self._load_raw_strokes(idx)

        # --- Handle Loading Failures ---
        if raw_strokes is None:
             logger.error(f"Failed to load strokes for index {idx}, file {self.file_paths[idx]}. Returning dummy data.")
             # Return zero tensor and the label
             feature_dim = 7 if self.include_pressure_time else 5
             dummy_tensor = torch.zeros((self.max_seq_len, feature_dim), dtype=torch.float32)
             # Ensure end-of-sequence is marked on dummy data for safety
             if self.max_seq_len > 0:
                 dummy_tensor[0, -1] = 1.0 # Mark EOS on first step
             return dummy_tensor, label or f"ErrorIdx_{idx}", None


        # --- Preprocessing ---
        # 1. Data Augmentation (NYI)
        # if self.data_augmentation:
        #     raw_strokes = self._augment_strokes(raw_strokes, self.data_augmentation)

        # 2. Simplification (Optional)
        if self.simplify_tolerance is not None and self.simplify_tolerance > 0:
            simplified_strokes = simplify_strokes(raw_strokes, tolerance=self.simplify_tolerance)
            if not simplified_strokes and raw_strokes: # Don't discard data if simplification fails badly
                logger.warning(f"Simplification removed all points for index {idx}. Using original strokes.")
            elif simplified_strokes:
                raw_strokes = simplified_strokes


        # 3. Normalization & Conversion to Sequence Tensor
        # Pass the flag indicating if pressure/time should *actually* be used based on detection/config
        should_include_pt = self.include_pressure_time and self._points_have_pressure_time

        sequence_tensor = strokes_to_sequence_tensor(
            strokes=raw_strokes,
            max_seq_len=self.max_seq_len,
            normalize=True, # Normalization usually desired
            normalization_size=self.normalization_size,
            include_pressure_time=should_include_pt
        )

        # Handle conversion failure
        if sequence_tensor is None:
            logger.error(f"Failed to convert strokes to tensor for index {idx}, file {self.file_paths[idx]}. Returning dummy data.")
            feature_dim = 7 if should_include_pt else 5
            sequence_tensor = torch.zeros((self.max_seq_len, feature_dim), dtype=torch.float32)
            if self.max_seq_len > 0:
                sequence_tensor[0, -1] = 1.0 # Mark EOS


        # --- GlyphWiki Data (Optional) ---
        glyph_tensor = None
        if self.use_glyphwiki and label and len(label) == 1: # Only for single characters
            # Check cache first
            if label in self.glyphwiki_cache:
                 glyph_data = self.glyphwiki_cache[label]
                 logger.debug(f"Using cached GlyphWiki data for '{label}'.")
            else:
                 try:
                     glyph_data = fetch_glyph_data(label)
                     self.glyphwiki_cache[label] = glyph_data # Store result (even if None) in cache
                 except APIError as e:
                     logger.error(f"GlyphWiki API error for character '{label}': {e}")
                     glyph_data = None
                 except Exception as e: # Catch unexpected errors during fetch
                      logger.error(f"Unexpected error fetching GlyphWiki data for '{label}': {e}", exc_info=True)
                      glyph_data = None

            # Convert glyph_data (dict) to a tensor suitable for model conditioning
            # This conversion is highly dependent on the model architecture!
            # Placeholder: Could be an embedding index, a multi-hot vector of components, etc.
            # For now, just indicate success/failure with a simple tensor.
            if glyph_data is not None:
                 # Example: Simple indicator tensor [1.0] if data exists, [0.0] otherwise
                 # A real implementation would extract features and create a meaningful tensor.
                 # Needs definition based on how model conditions on this info.
                 glyph_tensor = torch.tensor([1.0], dtype=torch.float32) # Placeholder
            else:
                 glyph_tensor = torch.tensor([0.0], dtype=torch.float32) # Placeholder


        return sequence_tensor, label, glyph_tensor


def create_dataloader(dataset_path: str,
                      data_format: str,
                      max_seq_len: int,
                      normalization_size: int,
                      use_glyphwiki: bool,
                      include_pressure_time: bool,
                      batch_size: int,
                      num_workers: int,
                      shuffle: bool = True,
                      simplify_tolerance: float | None = None,
                      bezier_points: int = 10,
                      # Add data_augmentation_config=None later
                      ) -> DataLoader:
    """
    Factory function to create a DataLoader for the HandwritingDataset.

    Args:
        dataset_path (str): Path to the dataset directory.
        data_format (str): 'svg' or 'moj'.
        max_seq_len (int): Max sequence length for tensors.
        normalization_size (int): Size for normalization box.
        use_glyphwiki (bool): Enable GlyphWiki fetching.
        include_pressure_time (bool): Whether model expects pressure/time features.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        shuffle (bool): Whether to shuffle the data each epoch.
        simplify_tolerance (float | None): Stroke simplification tolerance.
        bezier_points (int): Bezier sampling points for SVG parsing.

    Returns:
        DataLoader: Configured PyTorch DataLoader instance.

    Raises:
        DataError: If dataset initialization fails.
    """
    try:
        dataset = HandwritingDataset(
            dataset_path=dataset_path,
            data_format=data_format,
            max_seq_len=max_seq_len,
            normalization_size=normalization_size,
            use_glyphwiki=use_glyphwiki,
            glyphwiki_cache={}, # Provide a shared cache for the dataset instance
            simplify_tolerance=simplify_tolerance,
            bezier_points=bezier_points,
            include_pressure_time=include_pressure_time,
            # data_augmentation=data_augmentation_config, # Add later
        )

        # Simple default collate function works if __getitem__ returns tensors of consistent shape
        # (which it should due to padding/truncation)
        # If variable lengths were returned, a custom collate_fn using pack_padded_sequence might be needed.
        collate_fn = None

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False, # Helps speed up CPU->GPU transfer
            persistent_workers=True if num_workers > 0 else False, # Avoid worker startup overhead
        )
        logger.info(f"DataLoader created with batch_size={batch_size}, num_workers={num_workers}, shuffle={shuffle}")
        return dataloader

    except DataError as e:
        logger.error(f"Failed to create dataset/dataloader: {e}")
        raise e # Re-raise the specific error
    except Exception as e:
        logger.error(f"An unexpected error occurred during DataLoader creation: {e}", exc_info=True)
        # Wrap unexpected errors in DataError for consistent handling upstream
        raise DataError(f"Unexpected error creating DataLoader: {e}") from e


# --- Example Usage ---
# if __name__ == "__main__":
#     # Assumes a directory named 'sample_data/' exists with either 'a.svg', 'b.svg' or 'a.moj', 'b.moj' etc.
#     # You would need to create dummy data files first.
#     sample_dir = "sample_data"
#     data_fmt = "svg" # or "moj"

#     if not os.path.exists(sample_dir):
#         os.makedirs(sample_dir)
#         # Create dummy SVG file
#         dummy_svg_content = '<svg><path d="M10 10 L 90 90"/></svg>'
#         with open(os.path.join(sample_dir, f"test.{data_fmt}"), "w") as f:
#              f.write(dummy_svg_content) # Replace with moj content if testing moj
#         print(f"Created dummy data directory '{sample_dir}' with a test file.")


#     if os.path.exists(sample_dir) and os.listdir(sample_dir):
#         # Set up basic logging for testing this module
#         logging.basicConfig(level=logging.INFO, format='%(levelname)s - [%(name)s] - %(message)s')

#         print(f"\n--- Testing HandwritingDataset & DataLoader (format: {data_fmt}) ---")
#         try:
#             dataloader = create_dataloader(
#                 dataset_path=sample_dir,
#                 data_format=data_fmt,
#                 max_seq_len=100,
#                 normalization_size=64,
#                 use_glyphwiki=False, # Set True to test API calls if relevant chars exist
#                 include_pressure_time=False, # Set True if testing moj with p/t
#                 batch_size=2,
#                 num_workers=0, # Set > 0 to test multiprocessing
#                 shuffle=False,
#                 simplify_tolerance=1.0
#             )

#             print(f"Dataset size: {len(dataloader.dataset)}")

#             # Fetch one batch
#             print("\nFetching one batch...")
#             batch = next(iter(dataloader))
#             sequence_tensors, labels, glyph_tensors = batch

#             print(f"Batch sequence tensor shape: {sequence_tensors.shape}") # Should be (batch_size, max_seq_len, feature_dim)
#             print(f"Batch labels: {labels}")
#             print(f"Batch glyph tensors: {glyph_tensors}") # Will be None or placeholder tensor

#             # Test __getitem__ directly for one sample
#             print("\nFetching sample 0 directly:")
#             seq_tensor_0, label_0, glyph_tensor_0 = dataloader.dataset[0]
#             print(f"Sample 0 sequence tensor shape: {seq_tensor_0.shape}")
#             print(f"Sample 0 label: {label_0}")
#             print(f"Sample 0 glyph tensor: {glyph_tensor_0}")

#         except DataError as e:
#             print(f"Caught expected DataError during testing: {e}")
#         except Exception as e:
#             print(f"Caught unexpected error during testing: {e}", exc_info=True)

#         # Clean up dummy dir?
#         # import shutil
#         # shutil.rmtree(sample_dir)
#         # print(f"Removed dummy data directory '{sample_dir}'.")
#     else:
#         print(f"Sample data directory '{sample_dir}' not found or empty. Skipping dataset tests.")