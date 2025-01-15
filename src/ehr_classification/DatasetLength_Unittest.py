import pytest
import torch
from ehr_classification.data import PhysionetDataset

@pytest.fixture
def mock_data():
    return [
        {'ts_values': [[1.0] * 37, [2.0] * 37, [3.0] * 37], 'static': [0.5, 1.5], 'labels': [1], 'ts_times': [0, 1, 2]},
        {'ts_values': [[4.0] * 37, [5.0] * 37], 'static': [2.5, 3.5], 'labels': [0], 'ts_times': [0, 1]},
    ]

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
