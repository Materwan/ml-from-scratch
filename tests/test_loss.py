"""Tests for Model.calc_loss (binary cross-entropy averaged over the batch)."""

import numpy as np
import pytest

from ml_from_scratch.model import Model


@pytest.fixture
def model():
    np.random.seed(0)
    return Model(2, [3, 1], 0.1)


def test_loss_matches_manual_binary_cross_entropy(model):
    y_true = np.array([[0.0, 1.0]])
    y_pred = np.array([[0.2, 0.6]])
    expected = -(np.log(0.8) + np.log(0.6)) / 2
    assert model.calc_loss(y_pred, y_true) == pytest.approx(expected)


def test_loss_is_mean_over_batch(model):
    y_true = np.array([[0.0, 1.0, 1.0, 0.0]])
    y_pred = np.array([[0.1, 0.9, 0.4, 0.3]])
    per_example = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    assert model.calc_loss(y_pred, y_true) == pytest.approx(per_example.mean())


def test_loss_is_near_zero_for_perfect_predictions(model):
    y_true = np.array([[0.0, 1.0, 1.0, 0.0]])
    assert model.calc_loss(y_true.copy(), y_true) < 1e-6


def test_loss_is_finite_when_predictions_saturate(model):
    # Regression: log(0) produced -inf, then nan.
    y_true = np.array([[0.0, 1.0]])
    y_pred = np.array([[0.0, 1.0]])
    loss = model.calc_loss(y_pred, y_true)
    assert np.isfinite(loss)


def test_loss_is_chance_level_at_half(model):
    y_true = np.array([[0.0, 1.0, 1.0, 0.0]])
    y_pred = np.full_like(y_true, 0.5)
    assert model.calc_loss(y_pred, y_true) == pytest.approx(np.log(2))


def test_loss_raises_on_shape_mismatch(model):
    with pytest.raises(ValueError):
        model.calc_loss(np.array([[0.5, 0.5]]), np.array([[0.0, 1.0, 1.0]]))
