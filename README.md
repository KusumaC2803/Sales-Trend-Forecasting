# 📊 Sales Forecasting Dashboard

## Project Overview

This project analyzes monthly sales data, identifies trends, evaluates forecasting models, and predicts future sales using Python and statistical forecasting techniques.

An interactive dashboard built with Streamlit presents historical sales, monthly growth, model evaluation results, and future forecasts.

**Note:** This project uses a small sample dataset containing 24 monthly observations from January 2024 to December 2025. The dataset is intended for learning and demonstration, not for representing a real company's performance.

## Objectives

* Analyze historical monthly sales.
* Calculate key sales performance indicators.
* Examine month-over-month sales changes.
* Build and evaluate sales forecasting models.
* Compare forecasting results against a baseline.
* Forecast sales for the next three months.
* Present findings through an interactive dashboard.

## Technologies Used

* Python
* Pandas
* Matplotlib
* Statsmodels
* Scikit-learn
* Streamlit

## Dataset

The dataset contains two columns:

| Column | Description                         |
| ------ | ----------------------------------- |
| Date   | Monthly observation date            |
| Sales  | Sales value recorded for that month |

The dataset includes 24 observations covering January 2024 to December 2025.

## Project Structure

```text
Task-9-Data-Analyst-main/
├── data/
│   └── sales_data.csv
├── notebooks/
│   └── sales_forecast.ipynb
├── src/
│   └── forecast_analysis.py
├── outputs/
│   ├── model_metrics.csv
│   ├── validation_forecasts.csv
│   └── three_month_forecast.csv
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Analysis Performed

### 1. Exploratory Data Analysis

* Calculated total and average sales.
* Identified the highest and latest monthly sales values.
* Visualized historical sales trends.
* Analyzed month-over-month sales changes.

### 2. Forecasting Models

Two approaches are compared:

* **Naive Baseline:** Uses the last training observation as the prediction for each validation month.
* **Exponential Smoothing:** Uses an additive trend model to estimate future sales.

### 3. Model Evaluation

The models are evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

The last two observations are reserved for chronological validation.

### 4. Future Forecasting

The selected forecasting approach is refitted using all available observations to estimate sales for the next three months.

## Sample Results

The current sample dataset produces the following results:

| Metric                    |  Value |
| ------------------------- | -----: |
| Total Sales               |  4,153 |
| Average Monthly Sales     | 173.04 |
| Highest Monthly Sales     |    230 |
| Latest Monthly Sales      |    230 |
| Baseline MAE              |   7.50 |
| Exponential Smoothing MAE |   0.49 |

These values are based on the current dataset and implementation. The low validation error should be interpreted cautiously because only two observations are used for validation and the sample series has a highly regular upward trend.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Task-9-Data-Analyst-main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the forecasting analysis

```bash
python -m src.forecast_analysis
```

### 5. Launch the Streamlit dashboard

```bash
python -m streamlit run app.py
```

## Limitations

* The dataset contains only 24 monthly observations.
* The sales series has a highly regular upward trend.
* The available history is insufficient to assess annual seasonality reliably.
* Model evaluation uses only two validation observations.
* Forecasts are illustrative and should not be treated as reliable business predictions without additional data and testing.

## Future Improvements

* Evaluate models using rolling or expanding time-series validation.
* Test additional forecasting approaches.
* Use a larger, more realistic sales dataset.
* Add interactive date filters and downloadable charts.
* Deploy the Streamlit dashboard online.

## Author

**Kusuma C**

Computer Science and Engineering Graduate

GitHub: [KusumaC2803](https://github.com/KusumaC2803)

LinkedIn: [Kusuma C](https://linkedin.com/in/kusumac28)
