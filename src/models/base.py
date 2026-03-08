"""Base classes for encoder models."""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all encoders.

    All encoders must implement:
    - forward(...): returns (batch, output_dim) embedding tensor
    - output_dim: final embedding dimension

    Subclasses may accept additional positional/keyword arguments in forward
    (e.g. TabMixer accepts an optional x_categorical tensor).
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize base encoder.

        Args:
            input_dim: Input feature dimension
            output_dim: Output embedding dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through encoder.

        Subclasses may accept additional inputs (e.g. categorical indices).

        Returns:
            Output tensor of shape (batch, output_dim)
        """
        pass
