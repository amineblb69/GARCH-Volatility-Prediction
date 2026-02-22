import yfinance as yf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error

TICKER =  'AAPL'
START_DATE = '2015-01-01'
END_DATE = '2025-01-01'

def garch_vol(
        startd0 : str = START_DATE, 
        endd : str = END_DATE, 
        ticker=TICKER, 
        min_window: int = 60, 
        window :int = 12,
        rolling_window_size : int = 252, 
        interv : str = '1d'): 

    """ Equity Volalility Prediction using GARCH(1,1) Model + Volatility Analysis """
     
    
    wrmp_start = pd.to_datetime(startd0) - pd.DateOffset(days=min_window)
    dates = pd.date_range(startd0,endd, freq='B')


    #Downloading Warmup data
    stock = yf.download(
        ticker, start=wrmp_start, end=endd,
        interval=interv, progress=False, auto_adjust=True
    )['Close']

    if isinstance(stock, pd.DataFrame):
        stock = stock[ticker]
    stock = stock.dropna()

    #Computing Returns & Vol 
    stock_returns = stock.pct_change().dropna() * 100
    squared_returns = np.square(stock_returns)
    stock_vol = stock_returns.rolling(window=window).std()

    #GARCH Vol forecasting 1-step ahead
    forecast_values = {}
    for current_date in dates:

        historical_data = stock_returns[stock_returns.index < current_date]
        
        
        if len(historical_data) < rolling_window_size:
            continue

        train = historical_data.iloc[-rolling_window_size:]
        train = stock_returns[stock_returns.index < current_date]   
        
        if len(train) < min_window:
            forecast_values[current_date] = np.nan
            continue

        try:
            model = arch_model(train, mean='Constant', vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')

            forecast_var = fitted_model.forecast(horizon=1).variance.dropna()
            forecast_values[current_date] = np.sqrt(float(forecast_var.values[-1][0])) 
        
        except Exception as e:

            print(f"GARCH failed at {current_date}: {e}")
            forecast_values[current_date] = np.nan

    results = pd.Series(forecast_values, name='GARCH')

    
    verif_df = pd.concat([stock_vol,results],axis=1,join='inner')
    verif_df.columns = ['actual', 'preds']
    mse = mean_squared_error(verif_df['actual'],verif_df['preds'])

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2) 

    #Comparison Vol vs Prediction
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(stock_vol.loc[startd0:endd], label=f'Actual {window} Vol {ticker}')
    ax_top.plot(results.loc[startd0:endd], label='GARCH(1,1) Prediction', color='orange', linestyle='--')
    ax_top.set_title(f'Volatility Analysis: {ticker}')
    ax_top.legend()

    #Squared Returns
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bl.plot(squared_returns)
    ax_bl.set_title('Squared Returns (Volatility Clusters)')

    #Auto Correlation Function
    ax_br = fig.add_subplot(gs[1, 1])
    plot_acf(squared_returns.dropna(), ax=ax_br, lags=40)
    ax_br.set_title('Autocorrelation of Squared Returns')

    plt.tight_layout()
    plt.show()

    print(27*'=')
    print(f'Mean Squared Error : {mse}')
    print(27*'=')

    return results, stock_vol

garch, actual = garch_vol()

