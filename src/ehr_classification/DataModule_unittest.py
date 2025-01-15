import pytest
import torch
from torch.utils.data import DataLoader
from ehr_classification.data import PhysionetDataModule

@pytest.fixture
def mock_data():
    return [
        {'ts_values': [[1.0] * 37, [2.0] * 37, [3.0] * 37], 'static': [0.5, 1.5], 'labels': [1], 'ts_times': [0, 1, 2]},
        {'ts_values': [[4.0] * 37, [5.0] * 37], 'static': [2.5, 3.5], 'labels': [0], 'ts_times': [0, 1]},
    ]

@pytest.fixture
def datamodule(mock_data):
    datamodule = PhysionetDataModule(data_dir="/tmp", split_number=1, batch_size=2)
    datamodule.train_data = mock_data
    return datamodule

def test_train_dataloader_instance(datamodule):
    dataloader = datamodule.train_dataloader()
    assert isinstance(dataloader, DataLoader)  # Ensure it is a DataLoader instance

def test_train_dataloader_batch_size(datamodule):
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    ts_values, static, labels, lengths = batch
    assert ts_values.shape[0] == 2  # Ensure batch size is 2

def test_train_dataloader_ts_values_shape(datamodule):
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    ts_values, _, _, _ = batch
    assert ts_values.shape == (2, 3, 37)  # 2 samples, 3 timestamps, 37 features per timestamp

def test_train_dataloader_static_shape(datamodule):
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    _, static, _, _ = batch
    assert static.shape == (2, 2)  # 2 samples, 2 static features each

def test_train_dataloader_labels_shape(datamodule):
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    _, _, labels, _ = batch
    assert labels.shape == (2, 1)  # 2 samples, 1 label each

def test_train_dataloader_lengths_shape(datamodule):
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    _, _, _, lengths = batch
    assert lengths.shape == (2,)  # 2 samples, 1 length value each

def test_train_dataloader_ts_values_first_sample(datamodule):
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    ts_values, _, _, _ = batch
    assert torch.equal(ts_values[0, 0], torch.tensor([1.0] * 37, dtype=torch.float32))  # First timestamp of the first sample

def test_train_dataloader_ts_values_second_sample(datamodule):
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    ts_values, _, _, _ = batch
    assert torch.equal(ts_values[1, 0], torch.tensor([4.0] * 37, dtype=torch.float32))  # First timestamp of the second sample

def test_train_dataloader_ts_values_padding(datamodule):
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    ts_values, _, _, _ = batch
    # Check if padding has been applied: second sample should have 2 timestamps, padded to 3
    assert torch.equal(ts_values[1, 2], torch.tensor([0.0] * 37, dtype=torch.float32))  # Padding should be all zeros
