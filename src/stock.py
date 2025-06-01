import pandas as pd
import numpy as np
import ta  # pip install ta
import matplotlib.pyplot as plt
import os

# -----------------------
# 1. Load Stock Data
# -----------------------
def load_stock_data(ticker, data_path='../data/yfinance_data/yfinance_data/'):
    file_path = os.path.join(data_path, f"{ticker}.csv")
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in {ticker} data")
    
    df = df.dropna()
    df = df.sort_index()
    return df

# -----------------------
# 2. Technical Indicators
# -----------------------
def calculate_ta_indicators(df):
    indicators = {}

    # Simple and Exponential Moving Averages
    indicators['SMA_20'] = df['Close'].rolling(window=20).mean()
    indicators['SMA_50'] = df['Close'].rolling(window=50).mean()
    indicators['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI
    rsi = ta.momentum.RSIIndicator(close=df['Close'], window=14)
    indicators['RSI_14'] = rsi.rsi()

    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    indicators['MACD'] = macd.macd()
    indicators['MACD_signal'] = macd.macd_signal()
    indicators['MACD_hist'] = macd.macd_diff()

    # ATR
    atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    indicators['ATR_14'] = atr.average_true_range()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    indicators['BB_upper'] = bb.bollinger_hband()
    indicators['BB_middle'] = bb.bollinger_mavg()
    indicators['BB_lower'] = bb.bollinger_lband()

    return indicators

# -----------------------
# 3. Financial Metrics
# -----------------------
def calculate_financial_metrics(df):
    metrics = {}

    # Daily Returns
    metrics['Daily_Return'] = df['Close'].pct_change()
    
    # Cumulative Returns
    metrics['Cumulative_Return'] = (1 + metrics['Daily_Return']).cumprod()

    # Rolling Volatility
    metrics['Volatility_20D'] = metrics['Daily_Return'].rolling(window=20).std()

    # Volume Moving Average
    metrics['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

    # Sharpe Ratio (annualized, risk-free rate = 0)
    sharpe_ratio = (metrics['Daily_Return'].mean() / metrics['Daily_Return'].std()) * np.sqrt(252)
    metrics['Sharpe_Ratio'] = sharpe_ratio

    return metrics

# -----------------------
# 4. Visualization
# -----------------------
def visualize_analysis(df, indicators, metrics, ticker):
    plt.figure(figsize=(15, 10))

    # Subplots
    ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((4, 2), (2, 0))
    ax3 = plt.subplot2grid((4, 2), (2, 1))
    ax4 = plt.subplot2grid((4, 2), (3, 0), colspan=2)

    # Price + Moving Averages + Bollinger
    ax1.plot(df['Close'], label='Close', color='black')
    ax1.plot(indicators['SMA_20'], label='SMA 20', color='blue')
    ax1.plot(indicators['SMA_50'], label='SMA 50', color='green')
    ax1.fill_between(df.index, indicators['BB_lower'], indicators['BB_upper'], color='gray', alpha=0.1, label='Bollinger Bands')
    ax1.set_title(f"{ticker} Price with Moving Averages & Bollinger Bands")
    ax1.legend()

    # RSI
    ax2.plot(indicators['RSI_14'], label='RSI 14', color='purple')
    ax2.axhline(70, linestyle='--', color='red')
    ax2.axhline(30, linestyle='--', color='green')
    ax2.set_ylim(0, 100)
    ax2.set_title("RSI")

    # MACD
    ax3.plot(indicators['MACD'], label='MACD', color='blue')
    ax3.plot(indicators['MACD_signal'], label='Signal', color='orange')
    ax3.bar(df.index, indicators['MACD_hist'], color=np.where(indicators['MACD_hist'] > 0, 'green', 'red'), alpha=0.3)
    ax3.set_title("MACD")
    ax3.legend()

    # Volume and Volatility
    ax4a = ax4.twinx()
    ax4.plot(metrics['Volatility_20D'], label='Volatility 20D', color='red')
    ax4a.bar(df.index, df['Volume'], label='Volume', alpha=0.3)
    ax4.set_title("Volatility and Volume")
    ax4.legend(loc='upper left')
    ax4a.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# -----------------------
# 5. Complete Workflow
# -----------------------
def perform_complete_analysis(ticker):
    try:
        df = load_stock_data(ticker)
        print(f"Successfully loaded {len(df)} days of data for {ticker}")

        indicators = calculate_ta_indicators(df)
        print("Calculated technical indicators:", list(indicators.keys()))

        metrics = calculate_financial_metrics(df)
        print("Calculated financial metrics:", list(metrics.keys()))

        visualize_analysis(df, indicators, metrics, ticker)

        return {
            'status': 'success',
            'data': df,
            'indicators': indicators,
            'metrics': metrics
        }

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}

# -----------------------
# 6. Main Execution
# -----------------------
if __name__ == "__main__":
    results = perform_complete_analysis('TSLA_historical_data')  # Make sure CSV is named 'AAPL_historical_data.csv'

    if results['status'] == 'success':
        print("\nSample data:")
        print(results['data'].head())

        print("\nLatest RSI value:", round(results['indicators']['RSI_14'].dropna()[-1], 2))
        print("Latest Sharpe Ratio:", round(results['metrics']['Sharpe_Ratio'], 2))
