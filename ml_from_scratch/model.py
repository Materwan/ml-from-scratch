import numpy as np
from typing import List, Optional, Tuple


def ReLu(x: float) -> float:
    return 0 if x < 0 else x


def dReLu(x: float) -> float:
    return 0 if x < 0 else 1


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


class Model:

    def __init__(self, input_size: int, layer_sizes: List[int], learning_rate: float):
        if any([type(layer_sizes[i]) != int for i in range(len(layer_sizes))]):
            raise ValueError("Layer sizes must be int.")

        self.input_size = self.input_size = input_size

        self.lr = learning_rate

        self.nb_layers = len(layer_sizes)

        # print(layer_sizes[0], input_size)
        self.weights: List[np.ndarray] = [
            np.random.randn(layer_sizes[0], input_size) * np.sqrt(2 / layer_sizes[0])
        ]
        # print(input_size, 1)
        self.bias: List[np.ndarray] = [
            np.random.randn(layer_sizes[0], 1) * np.sqrt(2 / layer_sizes[0])
        ]

        self.outputs: List[np.ndarray] = [np.ndarray(0)] * self.nb_layers
        self.activations: List[np.ndarray] = [np.ndarray(0)] * self.nb_layers
        self.last_in: np.ndarray = np.ndarray(0)

        self.relu = np.vectorize(ReLu, otypes=[float])
        self.drelu = np.vectorize(dReLu, otypes=[float])
        self.sigmoid = np.vectorize(sigmoid, otypes=[float])

        for i in range(1, len(layer_sizes)):

            # print(layer_sizes[i], layer_sizes[i - 1])
            self.weights.append(
                np.random.randn(layer_sizes[i], layer_sizes[i - 1])
                * np.sqrt(2 / layer_sizes[i - 1])
            )
            # print(layer_sizes[i - 1], 1)
            self.bias.append(
                np.random.randn(layer_sizes[i], 1) * np.sqrt(2 / layer_sizes[i - 1])
            )

    def forward(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] != self.input_size:
            raise ValueError(
                f"Bad input : expected shape : ({self.input_size}, None), got {X.shape}."
            )

        self.last_in = X

        i = 1

        self.outputs[0] = np.matmul(self.weights[0], X) + self.bias[0]
        self.activations[0] = self.relu(self.outputs[0])

        while i < self.nb_layers - 1:

            self.outputs[i] = (
                np.matmul(self.weights[i], self.activations[i - 1]) + self.bias[i]
            )
            self.activations[i] = self.relu(self.outputs[i])

            i += 1

        self.outputs[i] = (
            np.matmul(self.weights[i], self.activations[i - 1]) + self.bias[i]
        )
        self.activations[i] = self.sigmoid(self.outputs[i])

        return self.activations[i]

    def calc_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        if y_pred.shape != y_true.shape:
            raise ValueError(
                "Shape not equal: y_pred shape must be equal to y_true shape,"
                f"but got : y_pred shape : {y_pred.shape} != y_true shape : {y_true.shape}."
            )

        eps = 1e-7
        p = np.clip(y_pred, eps, 1 - eps)
        m = y_true.shape[1]

        return -(1 / m) * np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def backward(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                "Shape not equal: y_pred shape must be equal to y_true shape,"
                f"but got : y_pred shape : {y_pred.shape} != y_true shape : {y_true.shape}."
            )

        dZ: List[np.ndarray] = [np.ndarray(0)] * self.nb_layers
        dW: List[np.ndarray] = [np.ndarray(0)] * self.nb_layers
        db: List[np.ndarray] = [np.ndarray(0)] * self.nb_layers

        i = self.nb_layers - 1

        dZ[i] = (y_pred - y_true) / y_true.shape[1]
        dW[i] = np.matmul(dZ[i], np.transpose(self.activations[i - 1]))
        db[i] = dZ[i].sum(axis=1, keepdims=True)

        i -= 1

        while i > 0:

            dZ[i] = np.multiply(
                np.matmul(np.transpose(self.weights[i + 1]), dZ[i + 1]),
                self.drelu(self.outputs[i]),
            )
            dW[i] = np.matmul(dZ[i], np.transpose(self.activations[i - 1]))
            db[i] = dZ[i].sum(axis=1, keepdims=True)

            i -= 1

        dZ[i] = np.multiply(
            np.matmul(np.transpose(self.weights[i + 1]), dZ[i + 1]),
            self.drelu(self.outputs[i]),
        )
        dW[i] = np.matmul(dZ[i], np.transpose(self.last_in))
        db[i] = dZ[i].sum(axis=1, keepdims=True)

        return dW, db

    def update(self, dW: List[np.ndarray], db: List[np.ndarray]):

        for i in range(self.nb_layers):

            self.weights[i] = self.weights[i] - dW[i] * self.lr
            self.bias[i] = self.bias[i] - db[i] * self.lr
