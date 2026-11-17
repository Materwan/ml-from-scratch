"""Tests for the activation functions (ReLu, dReLu, sigmoid) and their vectorized versions."""

import numpy as np
import pytest

from ml_from_scratch.model import Model, ReLu, dReLu, sigmoid


@pytest.mark.parametrize("x, expected", [(-2.0, 0), (-1e-9, 0), (0.0, 0), (3.5, 3.5)])
def test_relu_scalar(x, expected):
    assert ReLu(x) == expected


@pytest.mark.parametrize("x, expected", [(-2.0, 0), (-1e-9, 0), (0.0, 1), (3.5, 1)])
def test_drelu_scalar(x, expected):
    assert dReLu(x) == expected


def test_sigmoid_zero_is_one_half():
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_is_symmetric():
    x = np.array([-3.0, -0.5, 0.7, 4.0])
    np.testing.assert_allclose(sigmoid(x) + sigmoid(-x), 1.0)


def test_sigmoid_values_in_open_unit_interval():
    x = np.linspace(-10, 10, 101)
    y = sigmoid(x)
    assert np.all(y > 0) and np.all(y < 1)


def test_model_relu_returns_float_even_if_first_value_is_negative():
    # Regression: np.vectorize without otypes truncated activations to integers
    # when the first element was negative.
    model = Model(2, [3, 1], 0.1)
    out = model.relu(np.array([[-0.5, 1.25], [0.3, 2.75]]))
    np.testing.assert_allclose(out, [[0.0, 1.25], [0.3, 2.75]])
    assert np.issubdtype(out.dtype, np.floating)


def test_model_drelu_matches_scalar_version():
    model = Model(2, [3, 1], 0.1)
    z = np.array([[-1.0, 0.0, 2.0], [0.5, -0.1, 3.0]])
    expected = np.vectorize(dReLu)(z)
    np.testing.assert_array_equal(model.drelu(z), expected)
