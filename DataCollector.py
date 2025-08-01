import requests
import pandas as pd
import time
import numpy as np

def fetch_binance_ohlcv(symbol='ETHUSDT', interval='1h', limit=301, end_time=None):
    """
    Fetch OHLCV data ending at the given timestamp (end_time),
    and going back 'limit' candles (default 301 for 300 previous + 1 current).
    """
    base_url = 'https://api.binance.com/api/v3/klines'

    if isinstance(end_time, pd.Timestamp):
        end_time = int(end_time.timestamp() * 1000)
    elif isinstance(end_time, str):
        end_time = int(pd.to_datetime(end_time).timestamp() * 1000)

    params = {
        'symbol': symbol,
        'interval': interval,
        'endTime': end_time,
        'limit': limit
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"❌ Error fetching data: {response.status_code} {response.text}")

    data = response.json()

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']] = df[
        ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
    ].apply(pd.to_numeric, errors='coerce')

    return df

def calculate_technical_indicators(df):
    # RSI
    def calculate_rsi(df, window):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df['RSI14'] = calculate_rsi(df, 14)
    df['RSI30'] = calculate_rsi(df, 30)
    df['RSI200'] = calculate_rsi(df, 200)

    # EMAs
    df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']

    # Momentum
    df['MOM10'] = df['close'].diff(10)
    df['MOM30'] = df['close'].diff(30)

    # PROC
    df['PROC9'] = ((df['close'] - df['close'].shift(9)) / df['close'].shift(9)) * 100

    # Stochastic
    df['Low14'] = df['low'].rolling(window=14).min()
    df['High14'] = df['high'].rolling(window=14).max()
    df['%K'] = 100 * ((df['close'] - df['Low14']) / (df['High14'] - df['Low14']))
    df['%K10'] = df['%K'].rolling(window=10).mean()
    df['%K30'] = df['%K'].rolling(window=30).mean()
    df['%K200'] = df['%K'].rolling(window=200).mean()

    # SMA
    #df['SMA_10'] = df['close'].rolling(window=10).mean()
    #df['SMA_20'] = df['close'].rolling(window=20).mean()
    #df['SMA_30'] = df['close'].rolling(window=30).mean()

    # Bollinger Bands
    #sma_20 = df['close'].rolling(window=20).mean()
    #std_20 = df['close'].rolling(window=20).std()
    #df['BB_upper'] = sma_20 + 2 * std_20
    #df['BB_middle'] = sma_20
    #df['BB_lower'] = sma_20 - 2 * std_20

    # ROC
    #df['ROC_14'] = 100 * (df['close'] - df['close'].shift(14)) / df['close'].shift(14)
    #df['ROC_30'] = 100 * (df['close'] - df['close'].shift(30)) / df['close'].shift(30)

    return df

def create_feature_df(timestamp):
    """
    Given a timestamp (str or pd.Timestamp), returns a DataFrame with:
    - 100 previous + 1 current OHLCV row (total 101)
    - Technical indicators added
    """
    df = fetch_binance_ohlcv(end_time=timestamp)
    df = calculate_technical_indicators(df)
    print(df.tail(1))
    print(df.shape)
    return df

df1= create_feature_df("2023-10-01 01:00:00")
print("df1 tail: ", df1.tail(1))
print("df1 shape: ", df1.shape)