"""Temporal Convolutional Network (TCN) module."""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Causal 1D convolution (no future info leakage)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Causal padding: (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,  # We'll handle padding manually
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with causal padding."""
        # Pad left side only (causal)
        x_padded = torch.nn.functional.pad(x, (self.padding, 0))
        out = self.conv(x_padded)
        # Crop to maintain sequence length (causal: output[t] only depends on input[:t+1])
        if out.shape[-1] != x.shape[-1]:
            out = out[..., : x.shape[-1]]
        return out


class TCNBlock(nn.Module):
    """Single TCN residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Crop output to match residual length (causal conv maintains length)
        if out.shape[-1] != residual.shape[-1]:
            out = out[..., : residual.shape[-1]]

        return self.relu_out(out + residual)


class TemporalConvNet(nn.Module):
    """Full TCN with dilated convolutions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = output_dim if i == num_layers - 1 else hidden_dim
            dilation = 2**i  # Exponential dilation: 1, 2, 4, 8...

            layers.append(
                TCNBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        out = self.network(x)
        out = out.transpose(1, 2)  # (batch, seq_len, output_dim)
        return out
