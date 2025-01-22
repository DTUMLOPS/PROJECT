"""
This file defines the neural network components and the main model architecture
used for classification tasks in the DSSM model.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from typing import Optional, Union
import wandb


class DSSMLightning(pl.LightningModule):
    """
    DSSM implementation using PyTorch Lightning.
    This model is designed for classification tasks using both temporal and static input data.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        static_input_size: int,
        num_classes: int,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        bidirectional: bool = True,
        learning_rate: float = 0.001,
        class_weights: Optional[Union[list, torch.Tensor]] = None,
    ) -> None:
        """
        Initialize the model.

        Args:
            input_size (int): Number of features in the temporal input.
            hidden_size (int): Number of hidden units in the encoder.
            static_input_size (int): Number of features in the static input.
            num_classes (int): Number of target classes.
            num_layers (int, optional): Number of RNN layers. Defaults to 2.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
            bidirectional (bool, optional): Whether to use bidirectional RNN. Defaults to True.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            class_weights (Optional[Union[list, torch.Tensor]], optional): Class weights for loss function. Defaults to None.
        """
        super().__init__()

        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.learning_rate = learning_rate

        # Model components
        self.temporal_encoder = TemporalEncoder(input_size, hidden_size, num_layers, dropout_rate, bidirectional)
        self.static_encoder = StaticEncoder(static_input_size, hidden_size, dropout_rate)
        self.state_transition = StateTransition(hidden_size * self.num_directions, dropout_rate)
        self.classifier = Classifier(hidden_size * (self.num_directions + 1), hidden_size, num_classes, dropout_rate)

        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights) if class_weights else None)
        self.train_accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.val_accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.test_accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.auroc = torchmetrics.AUROC(task="binary", num_classes=2)
        self.auprc = torchmetrics.AveragePrecision(task="binary", num_classes=2)

        # save hyper-parameters to self.hparamsm auto-logged by wandb
        self.save_hyperparameters()

    def forward(
        self, temporal_data: torch.Tensor, static_data: torch.Tensor, seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the DSSM model.

        Args:
            temporal_data (torch.Tensor): Temporal input data of shape [batch_size, seq_len, input_size].
            static_data (torch.Tensor): Static input data of shape [batch_size, static_input_size].
            seq_lengths (torch.Tensor): Sequence lengths for the temporal data.

        Returns:
            torch.Tensor: Output logits of shape [batch_size, num_classes].
        """
        batch_size = temporal_data.size(0)

        temporal_repr = self.temporal_encoder(temporal_data, seq_lengths)
        static_repr = self.static_encoder(static_data)

        temporal_repr = temporal_repr[torch.arange(batch_size), seq_lengths - 1]
        state = self.state_transition(temporal_repr)
        combined = torch.cat([state, static_repr], dim=1)

        return self.classifier(combined)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _shared_step(self, batch, batch_idx):
        temporal_data, static_data, labels, seq_lengths = batch
        outputs = self(temporal_data, static_data, seq_lengths)  # Shape: [batch_size, 2]
        loss = self.criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)  # Shape: [batch_size, 2]
        preds = torch.argmax(outputs, dim=1)  # Shape: [batch_size]
        return loss, probs, preds, labels

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step logic.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        loss, probs, preds, labels = self._shared_step(batch, batch_idx)
        self.train_accuracy(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Validation step logic.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        loss, probs, preds, labels = self._shared_step(batch, batch_idx)
        self.val_accuracy(preds, labels)
        self.auroc(probs[:, 1], labels)  # AUROC expects probabilities of positive class
        self.auprc(probs[:, 1], labels)  # AUPRC expects probabilities of positive class

        self.log("val_loss", loss)
        self.log("val_acc", self.val_accuracy, on_epoch=True)
        self.log("val_auroc", self.auroc, on_epoch=True)
        self.log("val_auprc", self.auprc, on_epoch=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int):
        """
        Test step logic.
        """
        loss, probs, preds, labels = self._shared_step(batch, batch_idx)

        self.test_accuracy(preds, labels)
        self.auroc(probs[:, 1], labels)
        self.auprc(probs[:, 1], labels)

        # Log all metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test_auroc", self.auroc, on_step=False, on_epoch=True)
        self.log("test_auprc", self.auprc, on_step=False, on_epoch=True)

        return {"test_loss": loss, "test_acc": self.test_accuracy, "test_auroc": self.auroc, "test_auprc": self.auprc}


class TemporalEncoder(nn.Module):
    """
    Temporal Encoder that uses LSTM followed by a Multihead Attention mechanism.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        bidirectional: bool,
    ) -> None:
        """
        Initialize the TemporalEncoder.

        Args:
            input_size (int): Number of input features for each timestep.
            hidden_size (int): Number of hidden units in the LSTM.
            num_layers (int): Number of LSTM layers.
            dropout_rate (float): Dropout rate for regularization.
            bidirectional (bool): Whether the LSTM is bidirectional.
        """
        super(TemporalEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * (2 if bidirectional else 1), num_heads=4, dropout=dropout_rate
        )

    def forward(self, temporal_data: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TemporalEncoder.

        Args:
            temporal_data (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_size].
            seq_lengths (torch.Tensor): Sequence lengths for each sample in the batch.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        # Pack sequence for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            temporal_data, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Process with LSTM
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Apply attention
        attention_mask = torch.arange(lstm_output.size(1))[None, :].to(lstm_output.device) >= seq_lengths[:, None].to(
            lstm_output.device
        )
        attention_output = self._apply_attention(lstm_output, attention_mask)

        return attention_output

    @staticmethod
    def _create_attention_mask(tensor, seq_lengths):
        return torch.arange(tensor.size(1))[None, :] >= seq_lengths[:, None]

    def _apply_attention(self, lstm_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply Multihead Attention to LSTM outputs.
        """
        # Prepare attention inputs
        query = lstm_output.permute(1, 0, 2)
        key = value = query

        # Apply attention
        attn_output, _ = self.attention(query, key, value, key_padding_mask=mask)
        return attn_output.permute(1, 0, 2)


class StaticEncoder(nn.Module):
    """
    Static Encoder using a feed-forward neural network.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float) -> None:
        """
        Initialize the StaticEncoder.
        """
        super(StaticEncoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, static_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for StaticEncoder.
        """
        return self.network(static_data)


class StateTransition(nn.Module):
    """
    State Transition network that transforms encoded temporal data.
    """

    def __init__(self, hidden_size: int, dropout_rate: float) -> None:
        """
        Initialize the StateTransition
        """
        super(StateTransition, self).__init__()

        self.network = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for StateTransition.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, hidden_size].

        Returns:
            torch.Tensor: Transformed tensor of shape [batch_size, hidden_size].
        """
        return self.network(x)


class Classifier(nn.Module):
    """
    Classifier network for generating predictions.
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout_rate: float) -> None:
        """
        Initialize the Classifier.
        """
        super(Classifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x):
        """
        Forward pass for Classifier.
        """
        return self.network(x)

# Save the model as an artifact at the end of training
class SaveModelAsArtifact(pl.LightningModule):
    def on_train_end(self):
        # Save the checkpoint
        artifact_path = self.trainer.checkpoint_callback.best_model_path
        wandb.save(artifact_path)
        
        # Log the artifact to W&B
        artifact = wandb.Artifact(
            name="ehr_classification",
            type="model",
        )
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)