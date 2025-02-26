import os
import json
import torch
from svgpathtools import parse_path
from torch.utils.data import Dataset

class MojDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = []
        self.char_to_id = {}
        self.id_to_char = {}
        for file in os.listdir(dataset_path):
            if file.endswith('.moj'):
                with open(os.path.join(dataset_path, file), 'r') as f:
                    sample = json.load(f)
                char_id = self.char_to_id.setdefault(sample['character'], len(self.char_to_id))
                self.id_to_char[char_id] = sample['character']
                sequence = self.svg_to_sequence(sample['svg'], sample['strokes'])
                self.data.append((char_id, sequence))
        self.vocab_size = len(self.char_to_id)

    def svg_to_sequence(self, svg, strokes):
        # Convert strokes to sequence of (dx, dy, pressure, pen_up)
        sequence = []
        for stroke in sorted(strokes, key=lambda s: s['order']):
            points = stroke['points']
            for i in range(len(points)):
                x, y = points[i]['x'], points[i]['y']
                pressure = points[i]['pressure']
                dx = x - (points[i-1]['x'] if i > 0 else x)
                dy = y - (points[i-1]['y'] if i > 0 else y)
                pen_up = 1 if i == len(points) - 1 else 0
                sequence.append([dx, dy, pressure, pen_up])
        return torch.tensor(sequence, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        char_id, sequence = self.data[idx]
        return {'char': char_id, 'sequence': sequence}