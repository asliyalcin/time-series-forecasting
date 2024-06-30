Time Series Forecasting with ARIMA and SARIMA Models
This repository contains Python code for time series forecasting using ARIMA and SARIMA models. The project focuses on sales data analysis and forecasting for retail stores.

Overview
The project performs the following steps:

Data Loading and Preprocessing:

Loads sales data (data.csv) and store information (store.csv).
Adds additional features like year, month, day, and sales per customer.
Exploratory Data Analysis:

Generates Empirical Cumulative Distribution Function (ECDF) plots for sales, customers, and sale per customer.
Visualizes weekly sales trends for different stores using matplotlib and seaborn.
Data Cleaning:

Handles missing data by removing rows where stores were closed (Open != 0) and zero sales (Sales != 0).
Imputes missing values in CompetitionDistance with the median and other columns with zeros.
Merging Datasets:

Combines the sales and store datasets (df_merged) based on the store ID.
Stationarity Test:

Defines a function test_stationarity to check stationarity using rolling statistics and the Augmented Dickey-Fuller (ADF) test.
Time Series Decomposition:

Implements plot_timeseries to decompose sales data into trend and seasonality components using seasonal decomposition.
Autocorrelation and Partial Autocorrelation Analysis:

Computes autocorrelation function (ACF) and partial autocorrelation function (PACF) plots using auto_corr.
ARIMA and SARIMA Modeling:

Applies ARIMA and SARIMA models to weekly sales data (df_arima and df_sarima) using auto_arima and statsmodels.
Model Evaluation:

Evaluates model performance using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).
Visualization of Forecasts:

Plots observed vs. forecasted sales using both ARIMA and SARIMA models.
Requirements
Python 3.x
Required Libraries: pandas, numpy, matplotlib, seaborn, plotly, statsmodels, pmdarima, scikit-learn
