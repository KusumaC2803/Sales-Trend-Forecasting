
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.forecast_analysis import (
    load_sales_data,
    forecast_sales
)


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Forecasting Dashboard")

st.markdown(
    """
    Analyze historical monthly sales, track growth,
    compare forecasting models, and explore future sales.
    """
)

PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

@st.cache_data
def get_dashboard_data():
    sales_df = load_sales_data()

    validation, metrics, future_forecast = forecast_sales(
        horizon=3
    )

    return sales_df, validation, metrics, future_forecast


try:
    sales_df, validation, metrics, future_forecast = (
        get_dashboard_data()
    )

except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


# --------------------------------------------------
# 3. Dataset information
# --------------------------------------------------

st.caption(
    f"Data period: {sales_df.index.min():%b %Y} – "
    f"{sales_df.index.max():%b %Y} | "
    f"Monthly observations: {len(sales_df)}"
)


# --------------------------------------------------
# 4. Key performance indicators
# --------------------------------------------------

total_sales = sales_df["Sales"].sum()
average_sales = sales_df["Sales"].mean()
highest_sales = sales_df["Sales"].max()
latest_sales = sales_df["Sales"].iloc[-1]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Sales",
    f"{total_sales:,.0f}"
)

col2.metric(
    "Average Monthly Sales",
    f"{average_sales:,.2f}"
)

col3.metric(
    "Highest Monthly Sales",
    f"{highest_sales:,.0f}"
)

col4.metric(
    "Latest Monthly Sales",
    f"{latest_sales:,.0f}"
)


st.divider()


# --------------------------------------------------
# 5. Historical monthly sales trend
# --------------------------------------------------

st.subheader("1. Monthly Sales Trend")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    sales_df.index,
    sales_df["Sales"],
    marker="o",
    linewidth=2,
    label="Actual Sales"
)

ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.set_title("Historical Monthly Sales")

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend()

fig.autofmt_xdate()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 6. Monthly growth analysis
# --------------------------------------------------

st.subheader("2. Month-over-Month Growth")

growth_df = sales_df.copy()

growth_df["Sales Change"] = growth_df["Sales"].diff()

growth_df["Growth Percentage"] = (
    growth_df["Sales"].pct_change() * 100
)

# Display growth KPIs
growth_col1, growth_col2 = st.columns(2)

average_monthly_change = growth_df["Sales Change"].mean()

average_monthly_growth = (
    growth_df["Growth Percentage"].mean()
)

growth_col1.metric(
    "Average Monthly Sales Change",
    f"{average_monthly_change:.2f}"
)

growth_col2.metric(
    "Average Monthly Growth",
    f"{average_monthly_growth:.2f}%"
)


# Bar chart with a zero baseline
fig, ax = plt.subplots(figsize=(12, 5))

colors = [
    "tab:blue" if value >= 0 else "tab:red"
    for value in growth_df["Sales Change"].fillna(0)
]

ax.bar(
    growth_df.index,
    growth_df["Sales Change"].fillna(0),
    width=20,
    color=colors
)

ax.axhline(
    y=0,
    linewidth=1,
    linestyle="--"
)

ax.set_xlabel("Month")
ax.set_ylabel("Change in Sales")
ax.set_title("Monthly Increase or Decrease in Sales")

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.autofmt_xdate()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


with st.expander("View monthly growth data"):
    display_growth = growth_df.reset_index()

    display_growth["Date"] = pd.to_datetime(
        display_growth["Date"]
    ).dt.strftime("%Y-%m-%d")

    st.dataframe(
        display_growth.round(2),
        use_container_width=True,
        hide_index=True
    )


st.divider()


# --------------------------------------------------
# 7. Forecasting model comparison
# --------------------------------------------------

st.subheader("3. Forecasting Model Comparison")

st.caption(
    "Models are evaluated on the final two months of the "
    "historical dataset. Lower MAE and RMSE indicate "
    "smaller prediction errors on this validation period."
)

st.dataframe(
    metrics.round(2),
    use_container_width=True,
    hide_index=True
)

best_model = metrics.loc[metrics["MAE"].idxmin()]

st.info(
    f"Lowest validation MAE: {best_model['Model']} "
    f"({best_model['MAE']:.2f})"
)


# --------------------------------------------------
# 8. Actual versus predicted sales
# --------------------------------------------------

st.subheader("4. Actual vs. Predicted Sales")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    validation.index,
    validation["Actual_Sales"],
    marker="o",
    linewidth=2,
    label="Actual Sales"
)

ax.plot(
    validation.index,
    validation["Baseline_Predicted"],
    marker="x",
    linestyle="--",
    label="Naive Baseline"
)

ax.plot(
    validation.index,
    validation["Forecast_Predicted"],
    marker="s",
    linestyle="--",
    label="Exponential Smoothing"
)

ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.set_title("Validation Forecast Comparison")

ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend()

fig.autofmt_xdate()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


st.divider()


# --------------------------------------------------
# 9. Future sales forecast
# --------------------------------------------------

st.subheader("5. Future Sales Forecast")

forecast_df = future_forecast.rename(
    "Forecast Sales"
).to_frame()

forecast_col1, forecast_col2 = st.columns([2, 1])

with forecast_col1:

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        sales_df.index,
        sales_df["Sales"],
        marker="o",
        linewidth=2,
        label="Historical Sales"
    )

    ax.plot(
        future_forecast.index,
        future_forecast.values,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Forecast Sales"
    )

    # Connect the last actual observation to the first forecast
    ax.plot(
        [sales_df.index[-1], future_forecast.index[0]],
        [sales_df["Sales"].iloc[-1], future_forecast.iloc[0]],
        linestyle="--",
        linewidth=2
    )

    ax.axvline(
        sales_df.index[-1],
        linestyle=":",
        linewidth=1
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.set_title("Historical Sales and Three-Month Forecast")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


with forecast_col2:

    st.markdown("#### Forecast Values")

    st.dataframe(
        forecast_df.round(2),
        use_container_width=True
    )


st.divider()


# --------------------------------------------------
# 10. Business insights
# --------------------------------------------------

st.subheader("6. Key Business Insights")

first_sales = sales_df["Sales"].iloc[0]
last_sales = sales_df["Sales"].iloc[-1]

if first_sales != 0:
    overall_change = (
        (last_sales - first_sales) / abs(first_sales)
    ) * 100
else:
    overall_change = None


insight_col1, insight_col2 = st.columns(2)

with insight_col1:

    st.markdown("#### Historical Performance")

    st.write(
        f"- Sales changed from **{first_sales:.2f}** "
        f"to **{last_sales:.2f}** over the observed period."
    )

    if overall_change is not None:
        st.write(
            f"- The overall change between the first and "
            f"last observations was **{overall_change:.2f}%**."
        )

    st.write(
        f"- The highest monthly sales value was "
        f"**{highest_sales:.2f}**."
    )

    st.write(
        f"- Average monthly sales were "
        f"**{average_sales:.2f}**."
    )


with insight_col2:

    st.markdown("#### Forecast Performance")

    st.write(
        f"- The model with the lowest validation MAE was "
        f"**{best_model['Model']}**."
    )

    st.write(
        f"- Its MAE on the two-month validation period was "
        f"**{best_model['MAE']:.2f}**."
    )

    st.write(
        f"- The forecast covers the next "
        f"**{len(future_forecast)} months**."
    )


st.warning(
    """
    **Dataset limitations:** This project uses 24 monthly
    sample observations. The series has a highly regular
    upward trend and does not provide enough historical
    data to assess annual seasonality reliably. Model
    evaluation uses only two validation observations.
    Forecasts are educational estimates, not guaranteed
    business outcomes.
    """
)


# --------------------------------------------------
# 11. Download results
# --------------------------------------------------

st.subheader("7. Download Results")

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    st.download_button(
        label="Download Validation Results",
        data=validation.to_csv(index_label="Date"),
        file_name="validation_forecasts.csv",
        mime="text/csv"
    )

with download_col2:
    st.download_button(
        label="Download Model Metrics",
        data=metrics.to_csv(index=False),
        file_name="model_metrics.csv",
        mime="text/csv"
    )

with download_col3:
    st.download_button(
        label="Download Future Forecast",
        data=forecast_df.to_csv(index_label="Date"),
        file_name="three_month_forecast.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 12. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Sales Forecasting Dashboard | "
    "Python • Pandas • Matplotlib • "
    "Statsmodels • Streamlit"
)

