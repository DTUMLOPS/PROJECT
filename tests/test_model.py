import pytest
import torch
import torch.nn as nn
from ehr_classification.model import (
    DSSMLightning,
    StaticEncoder,
    StateTransition,
    Classifier,
)  # Adjust the import path as needed


@pytest.fixture
def model_params():
    """Fixture to set up common model parameters."""
    return {
        "input_size": 10,
        "static_input_size": 5,
        "hidden_size": 16,
        "num_classes": 2,
        "num_layers": 2,
        "dropout_rate": 0.2,
        "bidirectional": True,
        "learning_rate": 0.001,
    }


@pytest.fixture
def data(model_params):
    """Fixture to create sample input data for testing."""
    batch_size = 4
    seq_len = 6
    temporal_data = torch.rand(batch_size, seq_len, model_params["input_size"])  # Random temporal input data
    static_data = torch.rand(batch_size, model_params["static_input_size"])  # Random static input data
    seq_lengths = torch.randint(1, seq_len + 1, (batch_size,))  # Random sequence lengths
    labels = torch.randint(0, model_params["num_classes"], (batch_size,))  # Random labels
    return temporal_data, static_data, seq_lengths, labels


@pytest.fixture
def model(model_params):
    """Fixture to create the DSSM model instance."""
    return DSSMLightning(**model_params)


def test_forward_pass(model, data):
    """
    Test the forward pass of the DSSM model.
    Ensures that the model's output has the correct shape based on the input batch size and number of classes.
    """
    temporal_data, static_data, seq_lengths, _ = data
    outputs = model(temporal_data, static_data, seq_lengths)
    assert outputs.shape == (temporal_data.size(0), model.hparams.num_classes)


def test_configure_optimizers(model):
    """
    Test the optimizer configuration of the DSSM model.
    Ensures the model returns an Adam optimizer.
    """
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)


# def test_temporal_encoder(model_params, data):
#     """
#     Test the TemporalEncoder submodule.
#     Verifies that the output shape matches the expected LSTM output size, considering the hidden size and bidirectionality.
#     """
#     encoder = TemporalEncoder(
#         input_size=model_params["input_size"],
#         hidden_size=model_params["hidden_size"],
#         num_layers=model_params["num_layers"],
#         dropout_rate=model_params["dropout_rate"],
#         bidirectional=model_params["bidirectional"],
#     )
#     temporal_data, _, seq_lengths, _ = data
#     output = encoder(temporal_data, seq_lengths)
#     assert output.shape == (temporal_data.size(0), temporal_data.size(1), model_params["hidden_size"] * (2 if model_params["bidirectional"] else 1))


def test_static_encoder(model_params, data):
    """
    Test the StaticEncoder submodule.
    Ensures the encoder processes static data correctly and produces the expected output shape.
    """
    encoder = StaticEncoder(
        input_size=model_params["static_input_size"],
        hidden_size=model_params["hidden_size"],
        dropout_rate=model_params["dropout_rate"],
    )
    _, static_data, _, _ = data
    output = encoder(static_data)
    assert output.shape == (static_data.size(0), model_params["hidden_size"])


def test_state_transition(model_params, data):
    """
    Test the StateTransition submodule.
    Checks that the state transition layer processes input data and produces the correct output shape.
    """
    state_transition = StateTransition(
        hidden_size=model_params["hidden_size"],
        dropout_rate=model_params["dropout_rate"],
    )
    input_data = torch.rand(data[0].size(0), model_params["hidden_size"])
    output = state_transition(input_data)
    assert output.shape == (input_data.size(0), model_params["hidden_size"])


def test_classifier(model_params, data):
    """
    Test the Classifier submodule.
    Ensures the classifier produces the correct output shape given the combined input size and number of classes.
    """
    classifier = Classifier(
        input_size=model_params["hidden_size"] * 3,  # Combined size for the classifier input
        hidden_size=model_params["hidden_size"],
        num_classes=model_params["num_classes"],
        dropout_rate=model_params["dropout_rate"],
    )
    input_data = torch.rand(data[0].size(0), model_params["hidden_size"] * 3)
    output = classifier(input_data)
    assert output.shape == (input_data.size(0), model_params["num_classes"])


def test_training_step(model, data):
    """
    Test the training step of the DSSM model.
    Verifies that the training step computes a valid loss tensor.
    """
    temporal_data, static_data, seq_lengths, labels = data
    batch = (temporal_data, static_data, labels, seq_lengths)
    loss = model.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)


def test_validation_step(model, data):
    """
    Test the validation step of the DSSM model.
    Ensures that the validation step computes and returns a valid loss tensor.
    """
    temporal_data, static_data, seq_lengths, labels = data
    batch = (temporal_data, static_data, labels, seq_lengths)
    loss = model.validation_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)


def test_test_step(model, data):
    """
    Test the test step of the DSSM model.
    Confirms that the test step outputs a dictionary containing test metrics like loss, accuracy, AUROC, and AUPRC.
    """
    temporal_data, static_data, seq_lengths, labels = data
    batch = (temporal_data, static_data, labels, seq_lengths)
    result = model.test_step(batch, batch_idx=0)
    assert "test_loss" in result
    assert "test_acc" in result
    assert "test_auroc" in result
    assert "test_auprc" in result


# def test_temporal_encoder_with_zero_length_sequences(model_params):
#     """
#     Test the TemporalEncoder with zero-length sequences to ensure it handles them gracefully.
#     """
#     encoder = TemporalEncoder(
#         input_size=model_params["input_size"],
#         hidden_size=model_params["hidden_size"],
#         num_layers=model_params["num_layers"],
#         dropout_rate=model_params["dropout_rate"],
#         bidirectional=model_params["bidirectional"],
#     )
#     batch_size = 4
#     seq_len = 6
#     temporal_data = torch.rand(batch_size, seq_len, model_params["input_size"])
#     seq_lengths = torch.zeros(batch_size, dtype=torch.int64)  # Zero-length sequences
#     output = encoder(temporal_data, seq_lengths)
#     assert output.shape == (batch_size, seq_len, model_params["hidden_size"] * (2 if model_params["bidirectional"] else 1))


def test_static_encoder_with_large_inputs(model_params):
    """
    Test the StaticEncoder with larger-than-expected inputs to check for robustness.
    """
    encoder = StaticEncoder(
        input_size=model_params["static_input_size"],
        hidden_size=model_params["hidden_size"],
        dropout_rate=model_params["dropout_rate"],
    )
    batch_size = 4
    large_static_data = torch.rand(batch_size, model_params["static_input_size"] * 2)  # Larger input size
    with pytest.raises(RuntimeError):
        encoder(large_static_data)


def test_classifier_with_incorrect_input_size(model_params):
    """
    Test the Classifier with an incorrect input size to ensure it raises an error.
    """
    classifier = Classifier(
        input_size=model_params["hidden_size"] * 3,
        hidden_size=model_params["hidden_size"],
        num_classes=model_params["num_classes"],
        dropout_rate=model_params["dropout_rate"],
    )
    incorrect_input_data = torch.rand(4, model_params["hidden_size"] * 2)  # Mismatched input size
    with pytest.raises(RuntimeError):
        classifier(incorrect_input_data)


def test_state_transition_with_empty_input(model_params):
    """
    Test the StateTransition module with an empty tensor to check for robustness.
    """
    state_transition = StateTransition(
        hidden_size=model_params["hidden_size"],
        dropout_rate=model_params["dropout_rate"],
    )
    empty_input = torch.empty(0, model_params["hidden_size"])
    output = state_transition(empty_input)
    assert output.shape == (0, model_params["hidden_size"])


def test_forward_pass_with_mismatched_static_and_temporal_data(model, data):
    """
    Test the forward pass with mismatched temporal and static data dimensions to check for graceful failure.
    """
    temporal_data, _, seq_lengths, _ = data
    mismatched_static_data = torch.rand(
        temporal_data.size(0) + 1, model.hparams.static_input_size
    )  # Mismatched batch size
    with pytest.raises(RuntimeError):
        model(temporal_data, mismatched_static_data, seq_lengths)


def test_loss_function_with_missing_class_weights(model_params):
    """
    Test the loss function in DSSMLightning when no class weights are provided.
    """
    model_params["class_weights"] = None
    model = DSSMLightning(**model_params)
    criterion = model.criterion
    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.weight is None


# def test_training_step_with_empty_batch(model):
#     """
#     Test the training step with an empty batch to check for robustness.
#     """
#     empty_batch = (torch.empty(0, 6, model.hparams.input_size),  # Empty temporal data
#                    torch.empty(0, model.hparams.static_input_size),  # Empty static data
#                    torch.empty(0, dtype=torch.int64),  # Empty labels
#                    torch.empty(0, dtype=torch.int64))  # Empty sequence lengths
#     with pytest.raises(IndexError):
#         model.training_step(empty_batch, batch_idx=0)
