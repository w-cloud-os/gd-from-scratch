import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gradient_descent import closed_form, gradient_descent, gradients, make_data, mse


def test_gradient_matches_numerical_estimate():
    """Check the hand-derived gradient against finite differences."""
    x, y = make_data()
    w, b, eps = 1.3, -0.7, 1e-6
    dw, db = gradients(w, b, x, y)
    num_dw = (mse(w + eps, b, x, y) - mse(w - eps, b, x, y)) / (2 * eps)
    num_db = (mse(w, b + eps, x, y) - mse(w, b - eps, x, y)) / (2 * eps)
    assert np.isclose(dw, num_dw, atol=1e-5)
    assert np.isclose(db, num_db, atol=1e-5)


def test_converges_to_closed_form():
    x, y = make_data()
    path = gradient_descent(x, y, lr=0.05, steps=500)
    w_cf, b_cf = closed_form(x, y)
    assert np.isclose(path[-1, 0], w_cf, atol=1e-3)
    assert np.isclose(path[-1, 1], b_cf, atol=1e-3)


def test_loss_decreases():
    x, y = make_data()
    path = gradient_descent(x, y, lr=0.05, steps=100)
    assert path[-1, 2] < path[0, 2]
