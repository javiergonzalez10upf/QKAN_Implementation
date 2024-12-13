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
    ]).collect()

    # Create normalization expressions
    normalized_features_and_resp = []
    for col in feature_cols + ['resp']:
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
        pl.col("responder_6").alias("resp"),  # Use correct column name
        *[pl.col(f) for f in feature_cols]
    ])
             .limit(100000)
             .sort("date_id"))

    # 5. Normalize features for Chebyshev polynomials
    print("\nNormalizing features to [-1,1]...")
    normalized_lf = normalize_to_chebyshev_domain(query, feature_cols)

    # # 5. Verify normalization (using lazy operations)
    # norm_stats = normalized_lf.select([
    #     *[pl.col(col).min().alias(f"{col}_min") for col in feature_cols],
    #     *[pl.col(col).max().alias(f"{col}_max") for col in feature_cols],
    #     *[pl.col(col).mean().alias(f"{col}_mean") for col in feature_cols]
    # ]).collect()
    #
    # for col in feature_cols:
    #     print(f"\n{col} after normalization:")
    #     print(f"  Min: {norm_stats.get_column(f'{col}_min')[0]:.4f}")
    #     print(f"  Max: {norm_stats.get_column(f'{col}_max')[0]:.4f}")
    #     print(f"  Mean: {norm_stats.get_column(f'{col}_mean')[0]:.4f}")

    # 7. Initialize optimizer
    optimizer = DegreeOptimizer(
        network_shape=[5, 1],  # 5 inputs -> 1 output
        max_degree=3,
        complexity_weight=0.1,
        significance_threshold=0.05
    )

    print("\nStarting optimization...")
    # 8. Get final data for optimization
    # Only collect at the last moment when needed
    final_df = normalized_lf.collect()
    print(final_df.describe())
    optimal_degrees = optimizer.optimize_layer(
        layer_idx=0,
        x_data=final_df,
        y_data=final_df.get_column('resp_normalized').to_numpy(),
        num_reads=1000
    )

    print("\nResults:")
    print("Optimal polynomial degrees found:")
    for i, degrees in enumerate(optimal_degrees):
        print(f"Output node {i}:")
        for j, degree in enumerate(degrees):
            print(f"  Feature {j} -> degree {degree}")

    # 9. Performance Analysis
    print("\nModel Performance:")
    scores = optimizer.evaluate_expressiveness(
        x_data=final_df,
        y_data=final_df.get_column(target_col).to_numpy()
    )

    print("\nR² scores for different polynomial degrees:")
    for degree, score in enumerate(scores):
        print(f"Degree {degree}: {score:.4f}")

    is_definitive, best_degree = optimizer.is_degree_definitive(scores)
    print(f"\nFound definitive best degree: {is_definitive}")
    print(f"Best degree identified: {best_degree}")

if __name__ == "__main__":
    test_degree_optimizer_on_market_data()