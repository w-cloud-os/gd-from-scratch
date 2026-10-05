"""Run gradient descent on synthetic data and save plots to figures/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gradient_descent import closed_form, gradient_descent, make_data, mse

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def main():
    FIG_DIR.mkdir(exist_ok=True)
    x, y = make_data()
    path = gradient_descent(x, y, lr=0.05, steps=200)
    w_gd, b_gd, _ = path[-1]
    w_cf, b_cf = closed_form(x, y)
    print(f"Gradient descent: w={w_gd:.4f}, b={b_gd:.4f}")
    print(f"Closed form:      w={w_cf:.4f}, b={b_cf:.4f}")

    # 1) Fitted line
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=10, alpha=0.6, label="data")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, w_gd * xs + b_gd, "r", label="gradient descent fit")
    ax.set(xlabel="x", ylabel="y", title="Linear fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fit.png", dpi=150)
    plt.close(fig)

    # 2) Loss curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(path[:, 2])
    ax.set(xlabel="step", ylabel="MSE", title="Loss per step", yscale="log")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "loss_curve.png", dpi=150)
    plt.close(fig)

    # 3) Loss surface contours with the descent path
    ws = np.linspace(-5, 6, 120)
    bs = np.linspace(-5, 6, 120)
    W, B = np.meshgrid(ws, bs)
    Z = np.array([[mse(w, b, x, y) for w in ws] for b in bs])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contour(W, B, Z, levels=30, cmap="viridis")
    ax.plot(path[:, 0], path[:, 1], "r.-", markersize=3, label="descent path")
    ax.plot(w_cf, b_cf, "k*", markersize=12, label="closed-form optimum")
    ax.set(xlabel="w", ylabel="b", title="Loss surface and descent path")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "loss_surface.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
