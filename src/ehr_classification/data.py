from pathlib import Path
import pdb
import typer
from torch.utils.data import Dataset
import numpy as np

# Define the path to the data directory
DATA_PATH = "../../data/physion_data"

def load_data(data_path: Path, splits: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the dataset by combining data from multiple splits."""
    train_data = np.array([])
    val_data = np.array([])
    test_data = np.array([])
    
    # Iterate over each split and load the corresponding data
    for split in splits:
        train_data_split, val_data_split, test_data_split = load_split_data(split, base_path=data_path)
        
        # Concatenate the data from the current split with the previously loaded data
        train_data = np.concatenate([train_data, train_data_split])
        val_data = np.concatenate([val_data, val_data_split])
        test_data = np.concatenate([test_data, test_data_split])
    
    return train_data, val_data, test_data

def load_split_data(split_number, base_path='P12data'):
    """Load data for a specific split."""
    split_path = f'{base_path}/split_{split_number}'

    # Load the training, validation, and test data for the given split
    train_data = np.load(f'{split_path}/train_physionet2012_{split_number}.npy', allow_pickle=True)
    val_data = np.load(f'{split_path}/validation_physionet2012_{split_number}.npy', allow_pickle=True)
    test_data = np.load(f'{split_path}/test_physionet2012_{split_number}.npy', allow_pickle=True)
    
    return train_data, val_data, test_data
