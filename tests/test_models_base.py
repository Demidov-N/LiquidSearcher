import pytest
import torch
from src.models.base import BaseEncoder


def test_base_encoder_is_abstract():
    """Test that BaseEncoder cannot be instantiated directly."""
    with pytest.raises(TypeError):
        encoder = BaseEncoder(input_dim=10, output_dim=128)


def test_base_encoder_subclass_must_implement_forward():
    """Test that subclasses must implement forward method."""

    class DummyEncoder(BaseEncoder):
        def __init__(self):
            super().__init__(input_dim=10, output_dim=128)

    with pytest.raises(TypeError):
        encoder = DummyEncoder()
