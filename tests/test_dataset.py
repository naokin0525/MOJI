"""
Unit Tests for the Dataset Loader utility.

This test suite verifies the functionality of src.utils.dataset_loader.
It ensures that:
1. A dataset is correctly created from a directory of raw SVG files.
2. The processed dataset is saved to a .moj file.
3. The HandwritingDataset and DataLoader correctly load and batch the data.
"""
import unittest
import os
import shutil
import tempfile
import torch
from torch.utils.data import DataLoader

# Add project root to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.dataset_loader import create_or_load_dataset, HandwritingDataset, collate_fn
from src.config import BATCH_SIZE

class TestDatasetLoader(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory structure for a mock dataset."""
        self.test_dir = tempfile.mkdtemp()
        self.raw_path = os.path.join(self.test_dir, "raw")
        self.processed_path_parent = os.path.join(self.test_dir, "processed")
        
        # Override the default config path for testing purposes
        from src.utils import dataset_loader
        dataset_loader.PROCESSED_DATA_PATH = self.processed_path_parent
        
        # Create mock SVG files
        os.makedirs(os.path.join(self.raw_path, "latin"))
        svg_content = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M10 10 L 90 90"/></svg>'
        
        with open(os.path.join(self.raw_path, "latin", "a.svg"), "w") as f:
            f.write(svg_content)
        with open(os.path.join(self.raw_path, "latin", "b.svg"), "w") as f:
            f.write(svg_content)

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_create_and_load_dataset(self):
        """Test creating a .moj file and loading data."""
        # 1. Create the dataset from raw files
        dataset_list = create_or_load_dataset(self.raw_path, force_recreate=True)
        
        # Check that the returned data is a list of dicts
        self.assertIsInstance(dataset_list, list)
        self.assertEqual(len(dataset_list), 2)
        self.assertIn("label", dataset_list[0])
        self.assertIn("strokes", dataset_list[0])
        self.assertEqual(dataset_list[0]["label"], "a")

        # 2. Check if the .moj file was created
        dataset_name = os.path.basename(os.path.normpath(self.raw_path))
        expected_moj_path = os.path.join(self.processed_path_parent, f"{dataset_name}.moj")
        self.assertTrue(os.path.exists(expected_moj_path))

        # 3. Test loading from the existing .moj file
        loaded_dataset_list = create_or_load_dataset(self.raw_path, force_recreate=False)
        self.assertEqual(len(loaded_dataset_list), 2)
        self.assertEqual(loaded_dataset_list[1]["label"], "b")

    def test_dataloader_and_collation(self):
        """Test the HandwritingDataset and collate_fn with a DataLoader."""
        raw_data = create_or_load_dataset(self.raw_path, force_recreate=True)
        dataset = HandwritingDataset(raw_data)
        self.assertEqual(len(dataset), 2)

        data_loader = DataLoader(
            dataset,
            batch_size=2, # Use a batch size of 2 to get all data
            collate_fn=collate_fn
        )
        
        batch = next(iter(data_loader))
        
        # Check the batch structure and types
        self.assertIn("labels", batch)
        self.assertIn("strokes", batch)
        self.assertIn("lengths", batch)
        
        self.assertIsInstance(batch['labels'], list)
        self.assertIsInstance(batch['strokes'], torch.Tensor)
        self.assertIsInstance(batch['lengths'], torch.Tensor)
        
        # Check the shapes and content
        self.assertEqual(batch['strokes'].shape[0], 2) # batch size
        self.assertEqual(batch['strokes'].shape[2], 5) # num features
        self.assertEqual(batch['lengths'].shape[0], 2)

if __name__ == '__main__':
    unittest.main()