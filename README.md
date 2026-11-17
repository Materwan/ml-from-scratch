# ml-from-scratch

A dense neural network written by hand in NumPy: forward pass, backpropagation and gradient descent, no framework. The point is not performance but watching every gradient go by.

**Status**: work in progress (started 17/11/2026, phase 0 of the 2026-2027 ML plan). What exists today: a fully connected MLP, ReLU in the hidden layer, sigmoid at the output, binary cross-entropy and full-batch gradient descent. The demo learns XOR.

## Why this repo

Write the algorithm before calling the library. A neural network fits in about a hundred lines of NumPy; once you have written the backward pass by hand and checked it numerically, PyTorch's `nn.Linear` has no secrets left.

Phase goal: a clean, tested v0.1 before 20 December 2026.

## What works

- He initialisation (normal, standard deviation `sqrt(2 / n_in)`) for the ReLU layers;
- forward pass for any number of layers (`layer_sizes`);
- mean binary cross-entropy, with `eps = 1e-7` clipping to avoid `log(0)`;
- full backpropagation: output `dZ = ŷ - y`, then layer by layer upwards;
- SGD update of weights and biases;
- XOR learned with 2 inputs, 3 hidden units and 1 output.

## Requirements and install

Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Only dependency: NumPy.

## Run the demo

```bash
uv run python -m ml_from_scratch.main
```

`main.py` builds `Model(2, [3, 1], lr=0.1)`, seeds at 14, trains 5000 full-batch iterations on the four XOR points, prints the loss at every iteration, then the final predictions.

```python
X = [[0, 1, 0, 1],
     [0, 0, 1, 1]]
y = [[0, 1, 1, 0]]
```

## How it works

**Forward pass.** For each layer:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \qquad a^{(l)} = \mathrm{ReLU}(z^{(l)})$$

and at the output $\hat{y} = \sigma(z^{(L)})$. Intermediate activations are cached: the backward pass needs them.

**Loss.** Binary cross-entropy, averaged over the batch:

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right]$$

**Backpropagation.** Sigmoid and cross-entropy collapse nicely: at the output, `dZ = ŷ - y`. Then, going back up:

$$dZ^{(l)} = \left( W^{(l+1)\top} dZ^{(l+1)} \right) \odot \mathrm{ReLU}'(z^{(l)})$$
$$dW^{(l)} = dZ^{(l)} a^{(l-1)\top}, \qquad db^{(l)} = \sum_i dZ^{(l)}_i$$

**Update.**

$$\theta \leftarrow \theta - \eta \, d\theta$$

## Layout

| File | Role |
|---|---|
| `ml_from_scratch/model.py` | `Model`: initialisation, `forward`, `calc_loss`, `backward`, `update` |
| `ml_from_scratch/main.py` | XOR demo: data, training loop, output |
| `pyproject.toml` | project metadata, NumPy dependency |

## Known limitations

- **No gradient check yet**: the gradients are derived by hand, not verified numerically.
- **Gradient scale**: the loss is averaged over the batch (`1/m`) but `dZ` is not, so the gradients are sums. With a batch of 4 and `lr = 0.1` it works; as soon as the batch size changes, the effective step size changes with it. Fix before adding mini-batches.
- Sigmoid + binary cross-entropy only: no multi-class softmax, no regression.
- `np.vectorize` for ReLU, its derivative and the sigmoid: readable, but slow next to vectorised NumPy operations.
- No weight saving, no validation split, no external configuration (architecture, hyperparameters and seed live in the code).
- The loss is printed at every iteration, which drowns the output on a long run.

## Next steps

- [ ] Numerical gradient check on every parameter;
- [ ] unit tests (shapes, loss, XOR convergence) and a green `uv run pytest`;
- [ ] `train.py` driven by a config file (architecture, lr, batch, epochs, seed) with metrics logged to CSV;
- [ ] vectorised activations, softmax + cross-entropy, mini-batches with shuffling;
- [ ] comparison with an equivalent PyTorch implementation (same init, same seed);
- [ ] MNIST with a dense network from scratch, with a target accuracy written down before the run;
- [ ] `v0.1` tag.

## Context

This repo is the "from scratch" building block of the 2026-2027 ML plan: the next phases move on to vision (CNNs), MLOps, then a full end-to-end capstone. Every experiment gets a hypothesis written before, a one-line result after, and a seed.
