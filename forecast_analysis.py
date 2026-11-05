from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sales_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# --------------------------------------------------
# 1. Load and validate sales data
# --------------------------------------------------

def load_sales_data():
    """Load and validate monthly sales data."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"Date", "Sales"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain Date and Sales columns."
        )

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    df["Sales"] = pd.to_numeric(
        df["Sales"], errors="coerce"
    )

    # Remove invalid records
    df = df.dropna(subset=["Date", "Sales"])

    # Sort chronologically
    df = df.sort_values("Date")

    if df.empty:
        raise ValueError(
            "The dataset contains no valid sales records."
        )

    if df["Date"].duplicated().any():
        raise ValueError(
            "Duplicate dates found in the dataset."
        )

    # Convert dates to the first day of each month
    df["Date"] = (
        df["Date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    if df["Date"].duplicated().any():
        raise ValueError(
            "More than one record exists for a month."
        )

    # Set Date as the index and enforce monthly frequency
    df = df.set_index("Date").asfreq("MS")

    if df["Sales"].isna().any():
        raise ValueError(
            "Missing monthly sales values found."
        )

    if len(df) < 4:
        raise ValueError(
            "At least four monthly records are required."
        )

    return df


# --------------------------------------------------
# 2. Calculate monthly sales growth
# --------------------------------------------------

def calculate_monthly_growth():
    """Calculate monthly sales changes and growth rates."""

    df = load_sales_data()

    growth_data = df.copy()

    growth_data["Previous_Month_Sales"] = (
        growth_data["Sales"].shift(1)
    )

    growth_data["Sales_Change"] = (
        growth_data["Sales"]
        - growth_data["Previous_Month_Sales"]
    )

    growth_data["Growth_Percentage"] = (
        growth_data["Sales"]
        .pct_change()
        .mul(100)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    growth_data.to_csv(
        OUTPUT_DIR / "monthly_sales_growth.csv",
        index_label="Date"
    )

    return growth_data


# --------------------------------------------------
# 3. Plot monthly sales trend
# --------------------------------------------------

def plot_sales_trend():
    """Create and save the monthly sales trend chart."""

    df = load_sales_data()

    plt.figure(figsize=(12, 6))

    plt.plot(
        df.index,
        df["Sales"],
        marker="o",
        linewidth=2,
        label="Monthly Sales"
    )

    plt.title(
        "Monthly Sales Trend",
        fontsize=16
    )

    plt.xlabel("Month")
    plt.ylabel("Sales")

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.legend()
    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_DIR / "sales_trend.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Sales trend chart saved.")


# --------------------------------------------------
# 4. Plot monthly sales growth
# --------------------------------------------------

def plot_monthly_growth():
    """Create and save the monthly sales growth chart."""

    growth_data = calculate_monthly_growth()

    plot_data = growth_data.dropna(
        subset=["Sales_Change"]
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        plot_data.index,
        plot_data["Sales_Change"],
        width=20,
        label="Monthly Sales Change"
    )

    plt.title(
        "Month-over-Month Sales Change",
        fontsize=16
    )

    plt.xlabel("Month")
    plt.ylabel("Change in Sales")

    plt.axhline(
        y=0,
        linewidth=1
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "monthly_sales_growth.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Monthly sales growth chart saved.")


# --------------------------------------------------
# 5. Forecast sales and evaluate models
# --------------------------------------------------

def forecast_sales(horizon=3):
    """
    Evaluate a naive baseline and Exponential Smoothing.
    Then forecast future sales using all available data.
    """

    if horizon < 1:
        raise ValueError(
            "Forecast horizon must be at least 1."
        )

    df = load_sales_data()

    sales = df["Sales"]

    # Use the last two months for validation
    validation_size = 2

    if len(sales) <= validation_size:
        raise ValueError(
            "Not enough observations for validation."
        )

    train = sales.iloc[:-validation_size]
    actual = sales.iloc[-validation_size:]

    # Naive baseline predicts the last training value
    baseline_predictions = pd.Series(
        train.iloc[-1],
        index=actual.index,
        name="Baseline_Predicted"
    )

    # Train Exponential Smoothing on training data only
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=None,
        initialization_method="estimated"
    ).fit(optimized=True)

    predictions = model.forecast(
        validation_size
    )

    predictions.name = "Forecast_Predicted"

    # Calculate model evaluation metrics
    baseline_mae = mean_absolute_error(
        actual,
        baseline_predictions
    )

    forecast_mae = mean_absolute_error(
        actual,
        predictions
    )

    baseline_rmse = mean_squared_error(
        actual,
        baseline_predictions
    ) ** 0.5

    forecast_rmse = mean_squared_error(
        actual,
        predictions
    ) ** 0.5

    metrics = pd.DataFrame({
        "Model": [
            "Naive Baseline",
            "Exponential Smoothing"
        ],
        "MAE": [
            baseline_mae,
            forecast_mae
        ],
        "RMSE": [
            baseline_rmse,
            forecast_rmse
        ]
    })

    # Combine actual and predicted validation values
    validation_results = pd.DataFrame({
        "Actual_Sales": actual,
        "Baseline_Predicted": baseline_predictions,
        "Forecast_Predicted": predictions
    })

    # Fit the final model using all available sales data
    final_model = ExponentialSmoothing(
        sales,
        trend="add",
        seasonal=None,
        initialization_method="estimated"
    ).fit(optimized=True)

    future_forecast = final_model.forecast(
        horizon
    )

    future_forecast.name = "Forecast_Sales"

    # Save results
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    validation_results.to_csv(
        OUTPUT_DIR / "validation_forecasts.csv",
        index_label="Date"
    )

    metrics.to_csv(
        OUTPUT_DIR / "model_metrics.csv",
        index=False
    )

    future_forecast.to_frame().to_csv(
        OUTPUT_DIR / "three_month_forecast.csv",
        index_label="Date"
    )

    return (
        validation_results,
        metrics,
        future_forecast
    )


# --------------------------------------------------
# 6. Plot actual sales vs predicted sales
# --------------------------------------------------

def plot_forecast_comparison(validation_results):
    """Compare actual sales with validation predictions."""

    plt.figure(figsize=(12, 6))

    plt.plot(
        validation_results.index,
        validation_results["Actual_Sales"],
        marker="o",
        linewidth=2,
        label="Actual Sales"
    )

    plt.plot(
        validation_results.index,
        validation_results["Baseline_Predicted"],
        marker="s",
        linestyle="--",
        label="Naive Baseline"
    )

    plt.plot(
        validation_results.index,
        validation_results["Forecast_Predicted"],
        marker="^",
        linestyle="--",
        label="Exponential Smoothing"
    )

    plt.title(
        "Actual vs Predicted Sales",
        fontsize=16
    )

    plt.xlabel("Month")
    plt.ylabel("Sales")

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "forecast_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Forecast comparison chart saved.")


# --------------------------------------------------
# 7. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    print("\nLoading and validating sales data...")

    sales_data = load_sales_data()

    print("\nDataset preview:")
    print(sales_data.head())

    print("\nDataset information:")
    print(f"Total monthly records: {len(sales_data)}")
    print(f"First month: {sales_data.index.min():%Y-%m}")
    print(f"Last month: {sales_data.index.max():%Y-%m}")

    print("\nCalculating monthly sales growth...")

    growth_data = calculate_monthly_growth()

    print(
        growth_data[
            [
                "Sales",
                "Previous_Month_Sales",
                "Sales_Change",
                "Growth_Percentage"
            ]
        ].tail()
    )

    print("\nGenerating visualizations...")

    plot_sales_trend()
    plot_monthly_growth()

    print("\nEvaluating forecasting models...")

    (
        validation_results,
        metrics,
        future_forecast
    ) = forecast_sales(horizon=3)

    plot_forecast_comparison(
        validation_results
    )

    print("\nValidation Results:")
    print(validation_results.round(2))

    print("\nModel Evaluation:")
    print(
        metrics.round(2).to_string(index=False)
    )

    print("\nForecast for the Next Three Months:")
    print(future_forecast.round(2))

    print("\nAll output files have been generated.")
    print(f"Output folder: {OUTPUT_DIR}")

