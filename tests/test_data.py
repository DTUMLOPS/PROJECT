import pytest
import torch
from torch.utils.data import DataLoader

from ehr_classification.data import PhysionetDataModule
from ehr_classification.data import PhysionetDataset


@pytest.fixture
def mock_data():
    return [
        {"ts_values": [[1.0] * 37, [2.0] * 37, [3.0] * 37], "static": [0.5, 1.5], "labels": [1], "ts_times": [0, 1, 2]},
        {"ts_values": [[4.0] * 37, [5.0] * 37], "static": [2.5, 3.5], "labels": [0], "ts_times": [0, 1]},
    ]


@pytest.fixture(scope="function", autouse=True)
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


def test_train_dataloader_ts_values_padding(datamodule):
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    ts_values, _, _, lengths = batch

    # Check the actual sequence lengths match expected
    assert lengths[0] == 3  # First sequence has 3 timestamps
    assert lengths[1] == 2  # Second sequence has 2 timestamps

    # Check padding of second sequence
    assert torch.equal(ts_values[1, 2], torch.zeros(37, dtype=torch.float32))


@pytest.fixture
def dataset(mock_data):
    return PhysionetDataset(mock_data)


def test_len(dataset):
    assert len(dataset) == 2  # Check if dataset length matches mock data length


def test_first_item_ts_values_shape(dataset):
    ts_values, _, _, _ = dataset[0]
    assert ts_values.shape == (3, 37)  # 3 timestamps, 37 features each


def test_first_item_static_shape(dataset):
    _, static, _, _ = dataset[0]
    assert static.shape == (2,)  # 2 static features


def test_first_item_labels_shape(dataset):
    _, _, labels, _ = dataset[0]
    assert labels.shape == (1,)  # 1 label


def test_first_item_length(dataset):
    _, _, _, length = dataset[0]
    assert length == 3  # Length of ts_times for the first item
