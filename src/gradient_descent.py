"""Linear regression trained with gradient descent, using only NumPy.

Model:  y_hat = w * x + b
Loss:   L(w, b) = (1/n) * sum((w*x + b - y)^2)        (mean squared error)

Gradients (derived by hand with the chain rule, see README):
    dL/dw = (2/n) * sum((w*x + b - y) * x)
    dL/db = (2/n) * sum( w*x + b - y )
"""

import numpy as np


def make_data(n=200, w=3.0, b=2.0, noise=1.0, seed=0):
    """Generate noisy samples from the line y = w*x + b."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    y = w * x + b + rng.normal(0, noise, n)
    return x, y


def mse(w, b, x, y):
    """Mean squared error of the line (w, b) on data (x, y)."""
    return np.mean((w * x + b - y) ** 2)


def gradients(w, b, x, y):
    """Analytic gradient of the MSE with respect to w and b."""
    err = w * x + b - y
    return 2 * np.mean(err * x), 2 * np.mean(err)


def gradient_descent(x, y, lr=0.05, steps=200, w0=-4.0, b0=-4.0):
    """Run batch gradient descent.

    Returns an array of shape (steps + 1, 3) with columns (w, b, loss),
    so you can plot the whole optimization path.
    """
    w, b = w0, b0
    path = [(w, b, mse(w, b, x, y))]
    for _ in range(steps):
        dw, db = gradients(w, b, x, y)
        w -= lr * dw
        b -= lr * db
        path.append((w, b, mse(w, b, x, y)))
    return np.array(path)


def closed_form(x, y):
    """Exact least-squares solution, used to check gradient descent."""
    X = np.column_stack([x, np.ones_like(x)])
    w, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return w, b
