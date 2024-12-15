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
        pl.col('weight')
        *[pl.col(f) for f in feature_cols]
    ])
             .limit(1000000)
             .sort("date_id"))

    # 5. Normalize features for Chebyshev polynomials
    print("\nNormalizing features to [-1,1]...")
    normalized_lf = normalize_to_chebyshev_domain(query, feature_cols)

    dates = normalized_lf.select('date_id').collect().unique()
    n_dates = len(dates)

    train_dates = dates[:int(0.6 * n_dates)]
    val_dates = dates[int(0.6 * n_dates):int(0.8 * n_dates)]
    test_dates = dates[int(0.8 * n_dates):]

    train_mask = dates.filter(train_dates)
    val_mask = dates.filter(val_dates)
    test_mask = dates.filter(test_dates)

    train_data = normalized_lf.filter(train_mask).select([
        pl.col(f'{col}_normalized') for col in feature_cols
    ]).collect()

    val_data = normalized_lf.filter(val_mask).select([
        pl.col(f"{col}_normalized") for col in feature_cols
    ]).collect()

    test_data = normalized_lf.filter(test_mask).select([
        pl.col(f"{col}_normalized") for col in feature_cols
    ]).collect()


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
    feature_data = normalized_lf.select([pl.col(f"{col}_normalized") for col in feature_cols]).collect()
    target_data = normalized_lf.select('resp_normalized').collect()
    print(feature_data.describe())
    optimal_degrees = optimizer.optimize_layer(
        layer_idx=0,
        x_data=feature_data,
        y_data=target_data.to_numpy(),
        time_data=normalized_lf.select('date_id').collect(),
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