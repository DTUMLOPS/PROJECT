"""
data.py

This file provides data handling functionalities for the machine learning pipeline.
It includes a custom PyTorch Dataset for handling time-series data and a PyTorch Lightning 
DataModule for streamlined data loading and management across training, validation, and testing phases.
"""

import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class PhysionetDataset(Dataset):
    """
    Custom dataset for handling PhysioNet data.

    Attributes:
        data (List[dict]): List of data records with keys `ts_values`, `static`, `labels`, and `ts_times`.
    """

    def __init__(self, data: List[dict]):
        self.data = data

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        """
        Retrieve a single record from the dataset.

        Args:
            idx (int): Index of the record to retrieve.

        Returns:
            Tuple[Tensor, Tensor, Tensor, int]: A tuple containing:
                - Time-series values as a Tensor.
                - Static features as a Tensor.
                - Labels as a Tensor.
                - Length of the time-series sequence.
        """
        record = self.data[idx]
        return (
            torch.tensor(record["ts_values"], dtype=torch.float32),
            torch.tensor(record["static"], dtype=torch.float32),
            torch.tensor(record["labels"], dtype=torch.long),
            len(record["ts_times"]),
        )


class PhysionetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling PhysioNet data loading.

    Attributes:
        data_dir (Path): Directory containing the dataset.
        split_number (int): Index of the dataset split to use.
        batch_size (int): Batch size for data loaders.
    """

    def __init__(self, data_dir: str, split_number: int = 1, batch_size: int = 32):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split_number = split_number
        self.batch_size = batch_size
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage=None):
        """ Load data from processed directory based on the stage (train/val/test). """
        split_dir = self.data_dir / f"split_{self.split_number}"

        if stage == "fit" or stage is None:
            self.train_data = np.load(split_dir / f"train_physionet2012_{self.split_number}.npy", allow_pickle=True)
            self.val_data = np.load(split_dir / f"validation_physionet2012_{self.split_number}.npy", allow_pickle=True)

        if stage == "test" or stage is None:
            self.test_data = np.load(split_dir / f"test_physionet2012_{self.split_number}.npy", allow_pickle=True)

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for the training data."""        
        return DataLoader(
            PhysionetDataset(self.train_data), 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.__collate_fn
        )

    def val_dataloader(self)-> DataLoader:
        """Return the DataLoader for the validation data."""
        return DataLoader(
            PhysionetDataset(self.val_data), 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.__collate_fn
        )

    def test_dataloader(self)-> DataLoader:
        """Return the DataLoader for the test data."""
        return DataLoader(
            PhysionetDataset(self.test_data), 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.__collate_fn
        )

    @staticmethod
    def __collate_fn(batch: List[Tuple[Tensor, Tensor, Tensor, int]]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Collate function to batch and pad variable-length time-series sequences.

        Args:
            batch (List[Tuple[Tensor, Tensor, Tensor, int]]): List of data records.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: A tuple containing:
                - Padded time-series values.
                - Stacked static features.
                - Stacked labels.
                - Tensor of sequence lengths.
        """
        ts_values, static, labels, lengths = zip(*batch)
        lengths_tensor = torch.tensor(lengths)
        ts_values_padded = pad_sequence(ts_values, batch_first=True, padding_value=0.0)
        static_tensor = torch.stack(static)
        labels_tensor = torch.stack(labels)

        return ts_values_padded, static_tensor, labels_tensor, lengths_tensor