import numpy as np
from .model import Model

if __name__ == "__main__":

    np.random.seed(14)

    model = Model(2, [3, 1], 0.1)

    X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    y = np.array([[0, 1, 1, 0]])

    for i in range(5000):

        y_pred = model.forward(X)
        print(model.calc_loss(y_pred, y))
        dW, db = model.backward(y_pred, y)
        model.update(dW, db)

    y_pred = model.forward(X)
    print(y_pred.shape, y_pred)
    print(model.calc_loss(y_pred, y))
