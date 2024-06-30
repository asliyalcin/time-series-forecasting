# Time Series Forecasting with ARIMA and SARIMA Models

This repository contains Python code for time series forecasting using ARIMA and SARIMA models, focusing on retail sales data analysis.

## Overview

### Features

- **Data Loading and Preprocessing:**
  - Loads sales data (`data.csv`) and store information (`store.csv`).
  - Adds features like year, month, day, and sales per customer.

- **Exploratory Data Analysis:**
  - Empirical Cumulative Distribution Function (ECDF) plots for sales, customers, and sale per customer.
  - Weekly sales trends visualization for different stores.

- **Data Cleaning:**
  - Removes rows with closed stores (`Open != 0`) and zero sales (`Sales != 0`).
  - Imputes missing values in `CompetitionDistance` with the median and other columns with zeros.

- **Merging Datasets:**
  - Combines sales and store datasets (`df_merged`) based on store ID.

### Analysis

- **Stationarity Test:**
  - Function `test_stationarity` checks stationarity using rolling statistics and Augmented Dickey-Fuller (ADF) test.

- **Time Series Decomposition:**
  - `plot_timeseries` decomposes sales data into trend and seasonality using seasonal decomposition.

- **Autocorrelation and Partial Autocorrelation Analysis:**
  - ACF and PACF plots with `auto_corr` function.

### Modeling

- **ARIMA and SARIMA Models:**
  - Applies models to weekly sales data (`df_arima` and `df_sarima`) using `auto_arima` and `statsmodels`.

- **Model Evaluation:**
  - RMSE, MAE, and MAPE for model performance evaluation.

### Visualization

- **Forecast Visualization:**
  - Observed vs. forecasted sales plots using ARIMA and SARIMA models.

## Requirements

- Python 3.x
- Required Libraries: pandas, numpy, matplotlib, seaborn, plotly, statsmodels, pmdarima, scikit-learn

## Usage

To run the project:

```bash
python time_series_forecasting.py
