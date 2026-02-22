# GARCH-Volatility-Prediction
This project implement volatility prediction on an asset using GARCH(1,1) model and a time series analysis of the actual volatility.

Volatility is a critical metric in risk management and option pricing. Standard deviation calculations are backward-looking; the GARCH model improves upon this by accounting for volatility clustering—the tendency for high-volatility periods to be followed by high-volatility periods.

Key Features

    Automated Data Retrieval: Fetches historical price data using yfinance.

    Rolling Window Forecast: Implements a 1-step ahead walk-forward validation to simulate real-world trading conditions.

    Statistical Visualization: Generates a comprehensive dashboard including:

    Actual vs. Predicted Volatility.

    Squared Returns plot to visualize clustering.

    ACF (Autocorrelation Function) of squared returns to validate GARCH suitability.

Analysis Breakdown

Based on the generated output (see AAPL_vol_pred.png):

Volatility Clusters: The "Squared Returns" plot clearly shows spikes where high variance persists for a duration, confirming that a constant variance model would be insufficient.

Autocorrelation: The ACF plot shows significant positive correlations in squared returns across multiple lags. This statistical "memory" justifies the use of a GARCH framework over a simple moving average.

Model Performance: The GARCH(1,1) prediction (orange dashed line) tracks the realized 12-day rolling volatility closely, capturing the rapid expansion and contraction of risk during market stress events.

Prerequisites

    pip install yfinance pandas numpy matplotlib arch statsmodels scikit-learn
