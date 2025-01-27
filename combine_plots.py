#!/usr/bin/env python3
"""
Combine-Plot script:
Loads training loss logs for MLP and KAN from .npy files
and plots them on the same figure for direct comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_combined_curves(
    path_mlp: str,
    path_kan: str,
    label_mlp: str = "MLP",
    label_kan: str = "KAN",
    title: str = "Train Loss Comparison",
    y_label: str = "Loss",
    save_path: str = "combined_loss_plot.png"
):
    """
    Loads two .npy arrays containing training losses
    and plots them side by side.
    """
    # 1) Load the arrays
    mlp_losses = np.load(path_mlp)
    kan_losses = np.load(path_kan)

    # 2) Figure
    plt.figure()
    plt.plot(mlp_losses, label=label_mlp, color="blue")
    plt.plot(kan_losses, label=label_kan, color="orange")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.legend()
    # 3) Save + show
    plt.savefig(save_path)
    print(f"Saved combined plot => {save_path}")
    plt.show()

def main():
    # Example usage:
    # A) For Jane Street Weighted MSE (QKAN vs MLP):
    # We'll assume the npy files from test_jane_street.py
    path_qkan = "./models_janestreet/qkan_train_losses.npy"
    path_mlp  = "./models_janestreet/mlp_train_losses.npy"
    if os.path.exists(path_qkan) and os.path.exists(path_mlp):
        plot_combined_curves(
            path_mlp=path_mlp,
            path_kan=path_qkan,
            label_mlp="MLP Weighted MSE",
            label_kan="QKAN Weighted MSE",
            title="Jane Street Weighted MSE: MLP vs QKAN",
            save_path="./models_janestreet/combined_loss_janestreet.png"
        )

    # B) For Covertype classification (CE losses):
    path_mlp_cov = "models_tabular_data/mlp_covertype_losses.npy"
    path_qkan_cov = "models_tabular_data/qkan_covertype_losses.npy"
    if os.path.exists(path_mlp_cov) and os.path.exists(path_qkan_cov):
        plot_combined_curves(
            path_mlp=path_mlp_cov,
            path_kan=path_qkan_cov,
            label_mlp="MLP CE Loss",
            label_kan="QKAN CE Loss",
            title="Covertype Classification: MLP vs QKAN",
            save_path="models_tabular_data/combined_loss_covertype.png"
        )

    # C) For House Sales regression (MSE losses):
    path_mlp_house = "models_tabular_data/mlp_house_sales_losses.npy"
    path_qkan_house = "models_tabular_data/qkan_house_sales_losses.npy"
    if os.path.exists(path_mlp_house) and os.path.exists(path_qkan_house):
        plot_combined_curves(
            path_mlp=path_mlp_house,
            path_kan=path_qkan_house,
            label_mlp="MLP MSE",
            label_kan="QKAN MSE",
            title="House Sales Regression: MLP vs QKAN",
            save_path="models_tabular_data/combined_loss_house_sales.png"
        )

if __name__ == "__main__":
    main()