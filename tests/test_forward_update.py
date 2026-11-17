"""Tests for Model.forward (reference values) and Model.update."""

import numpy as np
import pytest

from ml_from_scratch.model import Model

X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=float)


def reference_forward(weights, biases, X):
    """Forward pass written directly in NumPy, used as the reference."""
    a = X
    for W, b in zip(weights[:-1], biases[:-1]):
        a = np.maximum(W @ a + b, 0.0)
    W, b = weights[-1], biases[-1]
    return 1 / (1 + np.exp(-(W @ a + b)))


@pytest.mark.parametrize("layers", [[3, 1], [4, 3, 1], [5, 4, 3, 1]])
def test_forward_matches_reference(layers):
    np.random.seed(1)
    model = Model(2, layers, 0.1)
    expected = reference_forward(model.weights, model.bias, X)
    np.testing.assert_allclose(model.forward(X), expected, rtol=1e-10)


@pytest.mark.parametrize("layers", [[3, 1], [4, 3, 1]])
def test_forward_output_shape_and_range(layers):
    np.random.seed(2)
    model = Model(2, layers, 0.1)
    y = model.forward(X)
    assert y.shape == (1, X.shape[1])
    assert np.all(y > 0) and np.all(y < 1)


def test_forward_raises_on_bad_input_size():
    model = Model(2, [3, 1], 0.1)
    with pytest.raises(ValueError):
        model.forward(np.zeros((3, 4)))


def test_update_applies_gradient_descent_step():
    np.random.seed(3)
    model = Model(2, [3, 1], 0.5)
    W_before = [W.copy() for W in model.weights]
    b_before = [b.copy() for b in model.bias]
    dW = [np.full_like(W, 2.0) for W in model.weights]
    db = [np.full_like(b, 4.0) for b in model.bias]

    model.update(dW, db)

    for i in range(model.nb_layers):
        np.testing.assert_allclose(model.weights[i], W_before[i] - 0.5 * 2.0)
        np.testing.assert_allclose(model.bias[i], b_before[i] - 0.5 * 4.0)


def test_update_with_zero_gradient_changes_nothing():
    np.random.seed(4)
    model = Model(2, [4, 3, 1], 0.1)
    W_before = [W.copy() for W in model.weights]
    zeros_dW = [np.zeros_like(W) for W in model.weights]
    zeros_db = [np.zeros_like(b) for b in model.bias]
    model.update(zeros_dW, zeros_db)
    for i in range(model.nb_layers):
        np.testing.assert_array_equal(model.weights[i], W_before[i])


def test_update_with_zero_learning_rate_changes_nothing():
    np.random.seed(5)
    model = Model(2, [3, 1], 0.0)
    W_before = [W.copy() for W in model.weights]
    dW = [np.ones_like(W) for W in model.weights]
    db = [np.ones_like(b) for b in model.bias]
    model.update(dW, db)
    for i in range(model.nb_layers):
        np.testing.assert_array_equal(model.weights[i], W_before[i])
