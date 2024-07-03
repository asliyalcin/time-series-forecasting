# Time Series Forecasting with ARIMA and SARIMA Models

This repository contains Python code for time series forecasting using ARIMA and SARIMA models, focusing on retail sales data analysis.
 - With this project, we will find answers to the following questions:
 - Can we predict sales in businesses?
 - Which methods allow predicting the future from past data?
 - What kind of procedure should be followed in case of missing and erroneous data?
 - Are sales affected by seasonality (seasons, months, days of the week, etc.)?
 - How do we measure the success of our prediction?

You can visit my medium page for more explanation:
(https://medium.com/@asliyalcnn/time-series-forecasting-with-ar%C4%B1ma-and-sar%C4%B1ma-f1a8f4ab5a39)

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
- ## References
- (https://mlpills.dev/time-series/how-to-train-a-sarima-model-step-by-step)  access date: 26.12.2023
- (https://people.duke.edu/~rnau/411arim.htm#:~:text=A%20nonseasonal%20ARIMA%20model%20is,errors%20in%20the%20prediction%20equation)
- Wanjuki T.M. & Wagala A. et. al,(2022), “Evaluating the Predictive Ability of Seasonal Autoregressive Integrated Moving Average (SARIMA) Models When Applied to Food and Beverages Price Index in Kenya”, EJ-MATH, European Journal of Mathematics and Statistics, 2736-5484, 29-31
- Wanjuki T.M. & Wagala A. et. al,(2022), “Evaluating the Predictive Ability of Seasonal Autoregressive Integrated Moving Average (SARIMA) Models When Applied to Food and Beverages Price Index in Kenya”, EJ-MATH, European Journal of Mathematics and Statistics, 2736-5484, 29-31
- (https://www.statisticshowto.com/adf-augmented-dickey-fuller-test)  access date: 24.12.2023 
- (https://www.analyticsvidhya.com/blog/2021/06/statistical-tests-to-check-stationarity-in-time-series-part-1/)  access date: 24.12.2023
- (https://permetrics.readthedocs.io/en/latest/pages/regression/RMSE.html) access date: 26.12.2023
- (https://permetrics.readthedocs.io/en/latest/pages/regression/MAE.html) access date: 26.12.2023
- https://docs.oracle.com/en/cloud/saas/planning-budgeting-cloud/pfusu/insights_metrics_MAPE.html#GUID-C33B0F01-83E9-468B-B96C-413A12882334) access date: 26.12.2023


## Usage

To run the project:

```bash
python time_series_forecasting.py

