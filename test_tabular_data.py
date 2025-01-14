#!/usr/bin/env python3
"""
Example script demonstrating how to:
1) Load two datasets (one classification, one regression) from the HuggingFace
   "inria-soda/tabular-benchmark" repository.
2) Perform a quick EDA (print shapes, columns, etc.).
3) Compare:
   - QKAN (Kolmogorov–Arnold Network) with QUBO-based degree selection
   - A Tiny MLP (PyTorch)
   - RandomForest (Scikit-Learn)
   - XGBoost

We also attempt to keep the KAN and MLP at a similar number of trainable parameters,
to have a fairer comparison. This involves adjusting KAN's hidden layers or number
of neurons to roughly match the MLP's parameter count.

NOTE:
- If you see issues with the "inria-soda/tabular-benchmark" load, ensure you have
  `pip install datasets`.
- If you see issues with QUBO, ensure you have `pip install pyqubo neal`.

Author: Your Name
Date: 2023-xx-yy
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from typing import Tuple
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb

############################################
# Import your KAN code from a separate file
############################################
# e.g.: from my_kan_file import FixedKANConfig, FixedKAN
# Here, we inline minimal references to user code placeholders:
from KAN_w_cumulative_polynomials import FixedKANConfig, FixedKAN


#############################################################################
# 1) Helper: count_parameters for KAN and MLP to roughly match param counts
#############################################################################
def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def approximate_kan_shape_for_param_count(
        input_dim: int,
        target_dim: int,
        desired_params: int,
        max_degree: int = 3
    ) -> list:
    """
    Try a naive approach to guess a KAN architecture (network_shape) that yields
    param_count ~ desired_params, using a single hidden layer for demonstration.

    NOTE: This is quite heuristic. We will do a small search for hidden_dim
    to see if we can approximate the param_count.
    """
    # We'll do a single hidden layer: [input_dim -> hidden_dim -> target_dim].
    # Then we compute param_count for different hidden_dim until we get close to desired_params.
    # max_degree determines the overhead from polynomial expansions. But let's keep it constant.

    def param_count_kan_shape(h: int) -> int:
        """
        Build a mock KAN with shape [input_dim, h, target_dim] and count parameters.
        We'll skip QUBO param overhead for simplicity (that is minor).
        """
        config = FixedKANConfig(
            network_shape=[input_dim, h, target_dim],
            max_degree=max_degree,
            complexity_weight=0.0
        )
        model = FixedKAN(config)
        return count_parameters(model)

    # We'll do a simple linear search or exponential search for hidden dim
    best_h = 1
    best_diff = float("inf")
    for hdim in range(1, 501, 5):  # step by 5 up to 500
        pc = param_count_kan_shape(hdim)
        diff = abs(pc - desired_params)
        if diff < best_diff:
            best_diff = diff
            best_h = hdim
        if pc > desired_params * 2:
            # If it's too large, break early
            break

    return [input_dim, best_h, target_dim]


def approximate_mlp_for_param_count(
        input_dim: int,
        target_dim: int,
        desired_params: int,
        hidden_init: int = 64
    ) -> nn.Module:
    """
    Build a small MLP that tries to match the desired_params count. We'll
    vary the hidden layer size or add a second layer if needed.
    """
    class TinyMLP(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
        def forward(self, x):
            return self.model(x)

    # simple approach: we vary hidden_dim
    best_h = hidden_init
    best_diff = float("inf")
    best_model = None

    for hdim in range(1, 501, 5):
        mlp_temp = TinyMLP(input_dim, hdim, target_dim)
        pc = count_parameters(mlp_temp)
        diff = abs(pc - desired_params)
        if diff < best_diff:
            best_diff = diff
            best_model = mlp_temp
            best_h = hdim
        if pc > desired_params * 2:
            # break early
            break
    # best_model should be the approximate param count model
    return best_model


##################################################################
# 2) Let's define a classification experiment with "covertype"
#    (numerical-only classification)
##################################################################
def load_covertype_data() -> Tuple[np.ndarray, np.ndarray]:
    dataset = load_dataset(
        "inria-soda/tabular-benchmark",
        data_files="clf_num/covertype.csv",
        split="train"
    )
    df = pd.DataFrame(dataset)
    print(f"[covertype] shape={df.shape}, columns={df.columns[:5]} ...")
    # By default, we assume there's a 'target' col
    label_col = "target" if "target" in df.columns else df.columns[-1]

    y = df[label_col].values
    X = df.drop(columns=[label_col]).values
    print("[covertype] Class distribution:", pd.Series(y).value_counts())
    return X, y


def experiment_covertype_classification():
    print("\n=== Classification on covertype (numeric) ===")
    X, y = load_covertype_data()

    # Basic train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2.1) Let's define the MLP with ~ certain param count
    # We'll pick an arbitrary param_count for reference. e.g. 10k
    desired_params = 10000

    # Build MLP
    mlp_model = approximate_mlp_for_param_count(input_dim=X_train.shape[1], target_dim=len(np.unique(y)),
                                                desired_params=desired_params, hidden_init=64)
    mlp_params = count_parameters(mlp_model)
    print(f"[MLP] param_count ~ {mlp_params}")

    # Train MLP quickly
    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
    loss_ce = nn.CrossEntropyLoss()

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    print("Training MLP for 10 epochs ...")
    for epoch in range(100):
        mlp_model.train()
        mlp_optimizer.zero_grad()
        logits = mlp_model(X_tr_t)
        loss = loss_ce(logits, y_tr_t)
        loss.backward()
        mlp_optimizer.step()
        if (epoch+1) % 2 == 0:
            mlp_model.eval()
            with torch.no_grad():
                val_preds = mlp_model(X_val_t)
                val_loss = loss_ce(val_preds, y_val_t).item()
            print(f"MLP epoch {epoch+1}, train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

    # Evaluate MLP
    mlp_model.eval()
    with torch.no_grad():
        y_pred_mlp = mlp_model(X_val_t).argmax(dim=1).numpy()
    acc_mlp = accuracy_score(y_val, y_pred_mlp)
    print(f"[MLP] covertype ACC={acc_mlp:.4f}")

    # 2.2) Build QKAN with ~ desired_params
    # We'll do a single hidden layer => [input_dim, hidden_dim, num_classes].
    # We'll guess max_degree=3 for polynomials
    kan_shape = approximate_kan_shape_for_param_count(
        input_dim=X_train.shape[1],
        target_dim=len(np.unique(y)),
        desired_params=desired_params,
        max_degree=3
    )
    print(f"[QKAN] approx shape = {kan_shape}")

    config = FixedKANConfig(
        network_shape=kan_shape,
        max_degree=3,
        complexity_weight=0.0,
        trainable_coefficients=False,
        skip_qubo_for_hidden=False,
        default_hidden_degree=2
    )
    qkan = FixedKAN(config)
    print(f"[QKAN] param_count ~ {count_parameters(qkan)}")

    # We'll do QUBO with one-hot training => we pass one-hot for final layer
    # Then cross-entropy training
    from torch.nn.functional import one_hot
    y_train_onehot = one_hot(y_tr_t, num_classes=len(np.unique(y))).float()

    print("Running QKAN cross-entropy training with QUBO-based degree selection ...")
    qkan.train_model_cross_entropy(
        x_data=X_tr_t,
        y_data_int=y_tr_t,
        y_data_onehot=y_train_onehot,
        num_epochs=100,
        lr=1e-3,
        complexity_weight=0.0,
        do_qubo=True
    )

    # Evaluate QKAN
    qkan.eval()
    with torch.no_grad():
        logits_qkan = qkan(X_val_t)
    y_pred_qkan = logits_qkan.argmax(dim=1).numpy()
    acc_qkan = accuracy_score(y_val, y_pred_qkan)
    print(f"[QKAN] covertype ACC={acc_qkan:.4f}")

    # 2.3) RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    acc_rf = accuracy_score(y_val, y_pred_rf)
    print(f"[RF] covertype ACC={acc_rf:.4f}")

    # 2.4) XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_val)
    acc_xgb = accuracy_score(y_val, y_pred_xgb)
    print(f"[XGB] covertype ACC={acc_xgb:.4f}")


##################################################################
# 3) A regression experiment with "house_sales" (numeric-only)
##################################################################
def load_house_sales_data() -> Tuple[np.ndarray, np.ndarray]:
    dataset = load_dataset(
        "inria-soda/tabular-benchmark",
        data_files="reg_num/house_sales.csv",
        split="train"
    )
    df = pd.DataFrame(dataset)
    print(f"[house_sales] shape={df.shape}, columns={df.columns[:5]} ...")
    label_col = "target" if "target" in df.columns else df.columns[-1]
    y = df[label_col].values
    X = df.drop(columns=[label_col]).values
    print("[house_sales] Y mean, std:", y.mean(), y.std())
    return X, y


def experiment_house_sales_regression():
    print("\n=== Regression on house_sales (numeric) ===")
    X, y = load_house_sales_data()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3.1) MLP with ~ desired params
    desired_params = 10000

    mlp_model = approximate_mlp_for_param_count(input_dim=X_train.shape[1], target_dim=1, desired_params=desired_params,
                                                hidden_init=64)
    mlp_params = count_parameters(mlp_model)
    print(f"[MLP] param_count ~ {mlp_params}")

    # Train MLP (MSE)
    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
    loss_mse = nn.MSELoss()

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

    print("Training MLP for 10 epochs (Regression) ...")
    for epoch in range(1000):
        mlp_model.train()
        mlp_optimizer.zero_grad()
        preds = mlp_model(X_tr_t)
        mse = loss_mse(preds, y_tr_t)
        mse.backward()
        mlp_optimizer.step()
        if (epoch+1) % 2 == 0:
            mlp_model.eval()
            with torch.no_grad():
                val_preds = mlp_model(X_val_t)
                val_loss = loss_mse(val_preds, y_val_t).item()
            print(f"MLP epoch {epoch+1}, train_mse={mse.item():.4f}, val_mse={val_loss:.4f}")

    # Evaluate MLP
    mlp_model.eval()
    with torch.no_grad():
        y_pred_mlp = mlp_model(X_val_t).squeeze(-1).numpy()
    mse_mlp = mean_squared_error(y_val, y_pred_mlp)
    print(f"[MLP] house_sales MSE={mse_mlp:.2f}")

    # 3.2) QKAN
    kan_shape = approximate_kan_shape_for_param_count(
        input_dim=X_train.shape[1],
        target_dim=1,
        desired_params=desired_params,
        max_degree=3
    )
    print(f"[QKAN] approx shape = {kan_shape}")
    config = FixedKANConfig(
        network_shape=kan_shape,
        max_degree=3,
        complexity_weight=0.0,
        trainable_coefficients=False,
        skip_qubo_for_hidden=False,
        default_hidden_degree=2
    )
    qkan = FixedKAN(config)
    print(f"[QKAN] param_count ~ {count_parameters(qkan)}")

    print("Running QKAN regression (MSE) with QUBO-based optimize ...")
    qkan.train_model(
        x_data=X_tr_t,
        y_data=y_tr_t,
        num_epochs=10,
        lr=1e-3,
        complexity_weight=0.0,
        do_qubo=True
    )

    # Evaluate QKAN
    qkan.eval()
    with torch.no_grad():
        preds_qkan = qkan(X_val_t).squeeze(-1).numpy()
    mse_qkan = mean_squared_error(y_val, preds_qkan)
    print(f"[QKAN] house_sales MSE={mse_qkan:.2f}")

    # 3.3) RandomForest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    mse_rf = mean_squared_error(y_val, y_pred_rf)
    print(f"[RF] house_sales MSE={mse_rf:.2f}")

    # 3.4) XGBoost
    xgb_reg = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_reg.fit(X_train, y_train)
    y_pred_xgb = xgb_reg.predict(X_val)
    mse_xgb = mean_squared_error(y_val, y_pred_xgb)
    print(f"[XGB] house_sales MSE={mse_xgb:.2f}")


##################################################################
# Main
##################################################################
def main():
    print("===== BEGIN TABULAR BENCHMARK EXPERIMENTS =====")
    # 1) Classification test on covertype
    experiment_covertype_classification()
    # 2) Regression test on house_sales
    #experiment_house_sales_regression()
    print("===== END =====")


if __name__ == "__main__":
    main()