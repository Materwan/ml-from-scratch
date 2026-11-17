"""Checks Model.backward gradients against finite differences.

The numerical gradient is computed on calc_loss (the batch-averaged loss),
and serves as the reference for the dW and db returned by backward.
"""

import numpy as np
import pytest

from ml_from_scratch.model import Model

X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=float)
Y = np.array([[0, 1, 1, 0]], dtype=float)
H = 1e-6
ARCHITECTURES = [[3, 1], [4, 3, 1], [5, 4, 3, 1]]


def numerical_grads(model, X, Y):
    """Gradients of calc_loss for every parameter, using central differences."""
    dW_num, db_num = [], []
    for params, out in ((model.weights, dW_num), (model.bias, db_num)):
        for layer in params:
            grad = np.zeros_like(layer)
            for idx in np.ndindex(layer.shape):
                orig = layer[idx]
                layer[idx] = orig + H
                loss_plus = model.calc_loss(model.forward(X), Y)
                layer[idx] = orig - H
                loss_minus = model.calc_loss(model.forward(X), Y)
                layer[idx] = orig
                grad[idx] = (loss_plus - loss_minus) / (2 * H)
            out.append(grad)
    return dW_num, db_num


def analytic_grads(model, X, Y):
    y_pred = model.forward(X)
    return model.backward(y_pred, Y)


@pytest.fixture(params=ARCHITECTURES, ids=lambda a: "-".join(map(str, a)))
def model(request):
    np.random.seed(0)
    return Model(2, request.param, 0.1)


def test_gradient_shapes_match_parameters(model):
    dW, db = analytic_grads(model, X, Y)
    for i in range(model.nb_layers):
        assert dW[i].shape == model.weights[i].shape
        assert db[i].shape == model.bias[i].shape


def test_dW_matches_numerical_gradient(model):
    dW, _ = analytic_grads(model, X, Y)
    dW_num, _ = numerical_grads(model, X, Y)
    for i in range(model.nb_layers):
        np.testing.assert_allclose(dW[i], dW_num[i], rtol=1e-4, atol=1e-6,
                                   err_msg=f"dW of layer {i}")


def test_db_matches_numerical_gradient(model):
    _, db = analytic_grads(model, X, Y)
    _, db_num = numerical_grads(model, X, Y)
    for i in range(model.nb_layers):
        np.testing.assert_allclose(db[i], db_num[i], rtol=1e-4, atol=1e-6,
                                   err_msg=f"db of layer {i}")


def test_backward_raises_on_shape_mismatch():
    np.random.seed(0)
    model = Model(2, [3, 1], 0.1)
    y_pred = model.forward(X)
    with pytest.raises(ValueError):
        model.backward(y_pred, Y[:, :2])
