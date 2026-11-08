import numpy as np
import pytest
import torch

from gewvol.evaluation.attention import lag_importance, market_spillover
from gewvol.models.lstm import LSTMModel
from gewvol.models.transformer import TransformerModel


@pytest.mark.parametrize("features", [15, 6])
def test_lstm_forward_shape(features):
    out = LSTMModel(features)(torch.randn(8, 60, features))
    assert out.shape == (8,)


@pytest.mark.parametrize("features", [15, 6])
def test_transformer_forward_shapes(features):
    torch.manual_seed(0)
    model = TransformerModel(input_size=features, num_markets=3, d_model=32, n_heads=4,
                             num_layers=2, dim_ff=64).eval()
    pred, attn = model(torch.randn(4, 20, features))
    assert pred.shape == (4,)
    assert len(attn) == 2
    assert attn[0].shape == (4, 60, 60)  # T * M tokens
    assert torch.allclose(attn[0].sum(-1), torch.ones(4, 60), atol=1e-5)


def test_transformer_rejects_bad_market_split():
    with pytest.raises(ValueError):
        TransformerModel(input_size=7, num_markets=3)


def test_attention_summaries():
    T, M = 10, 3
    a = np.random.default_rng(0).random((T * M, T * M))
    a /= a.sum(axis=1, keepdims=True)
    spill = market_spillover(a, M)
    assert spill.shape == (M, M)
    assert np.allclose(spill.sum(axis=1), 1.0, atol=1e-6)
    assert lag_importance(a, M).shape == (T,)
