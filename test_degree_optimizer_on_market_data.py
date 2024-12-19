import time
from typing import Tuple

import polars as pl
import numpy as np
import torch
from torch import nn
from torch.nn.functional as F
from DegreeOptimizer import DegreeOptimizer

def validate_market_data_schema(lf: pl.LazyFrame) -> None:
    """
    Validate schema for market data.
    Ensures features are float64 and date_id is datetime.
    """
    schema = lf.schema

    # Check date column
    assert "date_id" in schema, "Missing date_id column"

    # Check feature columns exist and are float64
    for i in range(79):  # Features 0-78
        col_name = f"feature_{i:02d}"
        assert col_name in schema, f"Missing {col_name}"
        assert schema[col_name] == pl.Float64, f"{col_name} should be float64"

    # Check response column
    assert "resp" in schema, "Missing response column"
    assert schema["resp"] == pl.Float64, "Response should be float64"

def normalize_to_chebyshev_domain(lf: pl.LazyFrame, feature_cols: list) -> pl.LazyFrame:
    """
    Normalize features to [-1,1] range for Chebyshev polynomials using lazy evaluation.
    Uses robust scaling to handle outliers in financial data.
    """
    # First calculate quantiles and std for each feature
    stats = lf.select([
        *[pl.col(col).quantile(0.05).alias(f"{col}_q05") for col in feature_cols],
        *[pl.col(col).quantile(0.95).alias(f"{col}_q95") for col in feature_cols],
        *[pl.col(col).std().alias(f"{col}_std") for col in feature_cols],
        pl.col('resp').quantile(.95).alias("resp_q95"),
        pl.col('resp').quantile(.05).alias("resp_q05"),
        pl.col('resp').std().alias("resp_std"),
        pl.col('weight').quantile(.95).alias("weight_q95"),
        pl.col('weight').quantile(.05).alias("weight_q05"),
        pl.col('weight').std().alias("weight_std"),
    ]).collect()

    # Create normalization expressions
    normalized_features_and_resp = []
    for col in feature_cols + ['resp','weight']:
        q05 = stats.get_column(f"{col}_q05")[0]
        q95 = stats.get_column(f"{col}_q95")[0]
        std = stats.get_column(f"{col}_std")[0]

        # Calculate center and scale with zero protection
        center = (q95 + q05) / 2
        scale = (q95 - q05) / 2 if abs(q95 - q05) > 1e-10 else std if std > 1e-10 else 1.0

        normalized_features_and_resp.append(
            pl.when(pl.col(col) > q95)
            .then(1.0)
            .when(pl.col(col) < q05)
            .then(-1.0)
            .otherwise((pl.col(col) - center) / scale)
            .alias(f"{col}_normalized")
        )

    # First add normalized features
    temp_lf = lf.select([pl.col("date_id"),*normalized_features_and_resp])
    # Then select required columns with proper names
    return temp_lf

def get_simple_split(timestamps: pl.DataFrame,
                     weights: np.ndarray,
                     train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a simple train-validation split based on timestamps.

    Args:
        timestamps: DataFrame containing date_id timestamps
        weights: Sample weights array
        train_ratio: Ratio of data to use for training (default 0.8)

    Returns:
        Tuple of (train_mask, val_mask, train_weights, val_weights)
    """
    unique_timestamps = timestamps.get_column('date_id').unique().sort()
    split_idx = int(len(unique_timestamps) * train_ratio)

    train_times = unique_timestamps[:split_idx]
    val_times = unique_timestamps[split_idx:]

    train_mask = timestamps.get_column('date_id').is_in(train_times).to_numpy()
    val_mask = timestamps.get_column('date_id').is_in(val_times).to_numpy()

    train_weights = weights[train_mask]
    val_weights = weights[val_mask]

    return train_mask, val_mask, train_weights, val_weights

def test_degree_optimizer_on_market_data():
    """Test DegreeOptimizer on Jane Street market data with proper lazy evaluation"""
    print("Starting market data test...")

    # 1. Load data using lazy evaluation
    lf = pl.scan_parquet("~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/").fill_null(3)

    # 2. Validate schema
    #print("Validating schema...")
    #validate_market_data_schema(lf)

    # 3. Select initial features and target
    feature_cols = [f"feature_{i:02d}" for i in range(79)]  # Start with 5 features
    target_col = "resp"

    # 4. Create initial lazy query
    query = (lf
             .select([
        pl.col("date_id"),
        pl.col("responder_6").alias("resp"),
        pl.col('weight'),
        *[pl.col(f) for f in feature_cols]
    ])
             .tail(100000)
             .sort("date_id"))

    # 5. Normalize features for Chebyshev polynomials
    print("\nNormalizing features to [-1,1]...")
    normalized_lf = normalize_to_chebyshev_domain(query, feature_cols)

    # 5. Create train-val split
    train_mask, val_mask, train_weights, val_weights = get_simple_split(
        timestamps=normalized_lf.select('date_id').collect(),
        weights=normalized_lf.select('weight_normalized').collect().to_numpy(),
        train_ratio=0.8
    )

    # 6. Initialize optimizer
    optimizer = DegreeOptimizer(
        network_shape=[79, 1],  # 79 features -> 1 output
        max_degree=3,
        complexity_weight=0.1,
        significance_threshold=0.05
    )
    query_params = {
        'n_rows': 100000,
        'columns': ['date_id_normalized', 'resp_normalized', 'weight_normalized'] + [f'feature_{i:02d}_normalized' for i in range(79)],
        'sort_by':'date_id',
    }
    optimizer.save_state('optimizer_state.npy', query_params=query_params)

    train_data = normalized_lf.filter(train_mask).select([
        pl.col(f'{col}_normalized') for col in feature_cols
    ]).collect()

    train_target = normalized_lf.filter(train_mask).select('resp_normalized').collect()

    # 8. Run optimization
    optimal_degrees = optimizer.optimize_layer(
        layer_idx=0,
        x_data=train_data,
        y_data=train_target.to_numpy(),
        weights=train_weights,
        num_reads=1000
    )

    # 9. Evaluate on validation set
    val_data = normalized_lf.filter(val_mask).select([
        pl.col(f"{col}_normalized") for col in feature_cols
    ]).collect()

    val_target = normalized_lf.filter(val_mask).select('resp_normalized').collect()

    scores = optimizer.evaluate_degree(
        x_data=val_data,
        y_data=val_target.to_numpy(),
        weights=val_weights
    )

    print("\nResults:")
    print("Optimal polynomial degrees found:")
    for i, degrees in enumerate(optimal_degrees):
        print(f"Output node {i}:")
        for j, degree in enumerate(degrees):
            print(f"  Feature {j} -> degree {degree}")

    print("\nValidation Performance:")
    print(f"R² score: {scores[0]:.4f}")


def compare_models(x_train, y_train, x_val, y_val, weights_train=None, weights_val=None):
    results = {}

    #1. QKAN (DegreeOptimizer) baseline
    optimizer = DegreeOptimizer(
        network_shape=[79, 1], max_degree=5,
    )

    start_time = time.time()
    optimal_degrees = optimizer.optimize_layer(
        layer_idx=0,
        x_data=x_train,
        y_data=y_train,
        weights=weights_train,
        num_reads=1000
    )


    val_scores = optimizer.evaluate_degree(x_val, y_val, weights=weights_val)
    qkan_time = time.time() - start_time
    results['QKAN'] = {
        'mse':val_scores[-1],
        'train_time':qkan_time,

    }

    #2. Simple MLP baseline
    mlp = nn.Sequential(
        nn.Linear(79, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    start_time = time.time()
    train_mlp(mlp, x_train, y_train, weights_train)
    mlp_time = time.time() - start_time

    mlp.eval()
    with torch.no_grad():
        val_pred = mlp(torch.FloatTensor(x_val.to_numpy()))
        mlp_mse = F.mse_loss(val_pred, torch.FloatTensor(y_val.to_numpy())).item()

if __name__ == "__main__":
    test_degree_optimizer_on_market_data()