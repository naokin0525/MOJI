import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch to the same length.
    
    Args:
        batch (list): A list of dictionaries, each containing 'char' (character ID)
                      and 'sequence' (tensor of stroke data).
    
    Returns:
        dict: A dictionary with 'char' (tensor of character IDs) and 'sequence'
              (padded tensor of stroke sequences).
    """
    chars = [item['char'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return {'char': torch.tensor(chars), 'sequence': sequences_padded}