import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class PhysionetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        return (
            torch.tensor(record['ts_values'], dtype=torch.float32),
            torch.tensor(record['static'], dtype=torch.float32),
            torch.tensor(record['labels'], dtype=torch.long),
            len(record['ts_times'])
        )


class PhysionetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, split_number: int = 1, batch_size: int = 32):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split_number = split_number
        self.batch_size = batch_size
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage=None):
        """Load data from processed directory"""
        split_dir = self.data_dir / f"split_{self.split_number}"

        if stage == 'fit' or stage is None:
            self.train_data = np.load(split_dir / f"train_physionet2012_{self.split_number}.npy",
                                      allow_pickle=True)
            self.val_data = np.load(split_dir / f"validation_physionet2012_{self.split_number}.npy",
                                    allow_pickle=True)

        if stage == 'test' or stage is None:
            self.test_data = np.load(split_dir / f"test_physionet2012_{self.split_number}.npy",
                                     allow_pickle=True)

    def train_dataloader(self):
        return DataLoader(
            PhysionetDataset(self.train_data),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.__collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            PhysionetDataset(self.val_data),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.__collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            PhysionetDataset(self.test_data),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.__collate_fn
        )

    @staticmethod
    def __collate_fn(batch):
        ts_values, static, labels, lengths = zip(*batch)
        lengths = torch.tensor(lengths)
        ts_values_padded = pad_sequence(ts_values, batch_first=True)
        static = torch.stack(static)
        labels = torch.stack(labels)

        return ts_values_padded, static, labels, lengths