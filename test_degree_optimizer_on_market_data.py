from typing import Tuple

import polars as pl
import numpy as np
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
    # Take first 100k samples sorted by date
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

if __name__ == "__main__":
    test_degree_optimizer_on_market_data()