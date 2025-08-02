import pandas as pd
import ta

def compute_indicators(df):
    df = df.copy()

    df['EMA10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['EMA30'] = ta.trend.ema_indicator(df['close'], window=30)
    df['EMA200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['EMA12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['RSI14'] = ta.momentum.rsi(df['close'], window=14)
    df['RSI30'] = ta.momentum.rsi(df['close'], window=30)
    df['RSI200'] = ta.momentum.rsi(df['close'], window=200)
    df['MOM10'] = ta.momentum.roc(df['close'], window=10)
    df['MOM30'] = ta.momentum.roc(df['close'], window=30)
    df['MACD'] = ta.trend.macd_diff(df['close'])
    df['PROC9'] = ta.momentum.roc(df['close'], window=9)
    df['%K'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    df['%K10'] = df['%K'].rolling(window=10).mean()
    df['%K30'] = df['%K'].rolling(window=30).mean()
    df['%K200'] = df['%K'].rolling(window=200).mean()
    df['EMA_Open_10'] = ta.trend.ema_indicator(df['open'], window=10)

    df = df.dropna().reset_index(drop=True)

    return df
