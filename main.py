import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_loader import load_models_and_stats
from utils import compute_indicators
from typing import Optional
from fastapi.responses import JSONResponse

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and stats
btc_models, btc_stats = load_models_and_stats("btc")
eth_models, eth_stats = load_models_and_stats("eth")

# Binance symbols
BINANCE_SYMBOLS = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT"
}

# Features used in training
features_ohlc = ['volume', 'quote_asset_volume', 'RSI14', 'RSI30', 'RSI200', 'MOM10', 'MOM30',
                 'MACD', 'PROC9', 'EMA10', 'EMA30', 'EMA200', '%K10', '%K30', '%K200']
features_volume = ['quote_asset_volume', 'MOM10', 'MOM30', 'EMA12', 'RSI14', 'RSI30',
                   'RSI200', 'EMA26', 'MACD']

class PredictionResponse(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float

def fetch_binance_data(symbol: str, interval: str = "1h", limit: int = 500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch data from Binance")
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_base_volume", "taker_quote_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float,
        "volume": float, "quote_asset_volume": float
    })
    return df

# @app.get("/predict", response_model=list[PredictionResponse])
# def predict(asset: str = Query(...), timestamp: Optional[str] = Query(None)):
#     if asset not in BINANCE_SYMBOLS:
#         return [{"error": "Invalid asset"}]

#     symbol = BINANCE_SYMBOLS[asset]
#     models = btc_models if asset == "btc" else eth_models
#     stats = btc_stats if asset == "btc" else eth_stats

#     df = fetch_binance_data(symbol, interval="1h", limit=500)
#     df = compute_indicators(df)

#     # If timestamp is given, find closest matching row
#     if timestamp:
#         try:
#             ts = pd.to_datetime(timestamp)
#             # df["diff"] = (df["timestamp"] - ts).abs()
#             # df = df.sort_values("diff").head(1).drop(columns=["diff"])
#             df["diff"] = (df["timestamp"] - ts).abs()
#             df = df.sort_values("diff")

# # Get index of the closest row
#             closest_index = df.index[0]

# # Get 4 rows: closest + next 3 hours
#             df = df.loc[closest_index:closest_index + 3].reset_index(drop=True)

# # Remove "diff" column if it exists
#             if "diff" in df.columns:
#              df = df.drop(columns=["diff"])

#         except Exception as e:
#             print(f"❌ Error parsing timestamp: {e}")
#             return [{"date": timestamp, "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0}]
#     else:
#         # df = df.tail(4)  # Default to last 4 hours
#          return JSONResponse(content={"Error": "You have not selected any date."}, status_code=400)
@app.get("/predict", response_model=list[PredictionResponse])
def predict(asset: str = Query(...), timestamp: Optional[str] = Query(None)):
    if asset not in BINANCE_SYMBOLS:
        return [{"error": "Invalid asset"}]

    symbol = BINANCE_SYMBOLS[asset]
    models = btc_models if asset == "btc" else eth_models
    stats = btc_stats if asset == "btc" else eth_stats

    df = fetch_binance_data(symbol, interval="1h", limit=500)
    df = compute_indicators(df)

    if timestamp:
        try:
            ts = pd.to_datetime(timestamp)
            df["diff"] = (df["timestamp"] - ts).abs()
            row = df.sort_values("diff").head(1).drop(columns=["diff"]).iloc[0]

            X_ohlc = np.array(row[features_ohlc], dtype=float).reshape(1, -1)
            X_volume = np.array(row[features_volume], dtype=float).reshape(1, -1)

            def unscale(pred, col): return pred * stats[col]["std"] + stats[col]["mean"]

            pred_open = unscale(models["open"].predict(X_ohlc)[0], "open")
            pred_high = unscale(models["high"].predict(X_ohlc)[0], "high")
            pred_low = unscale(models["low"].predict(X_ohlc)[0], "low")
            pred_close = unscale(models["close"].predict(X_ohlc)[0], "close")
            pred_volume = unscale(models["volume"].predict(X_volume)[0], "volume")

            result = [{
                "date": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "open": pred_open,
                "high": pred_high,
                "low": pred_low,
                "close": pred_close,
                "volume": pred_volume
            }]
            return result

        except Exception as e:
            print(f"❌ Error parsing timestamp or predicting: {e}")
            return [{"date": timestamp, "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0}]
    else:
        return JSONResponse(content={"Error": "You have not selected any date."}, status_code=400)


    results = []

    for i in range(len(df)):
        row = df.iloc[i]
        try:
            X_ohlc = np.array(row[features_ohlc], dtype=float).reshape(1, -1)
            X_volume = np.array(row[features_volume], dtype=float).reshape(1, -1)

            def unscale(pred, col): return pred * stats[col]["std"] + stats[col]["mean"]

            pred_open = unscale(models["open"].predict(X_ohlc)[0], "open")
            pred_high = unscale(models["high"].predict(X_ohlc)[0], "high")
            pred_low = unscale(models["low"].predict(X_ohlc)[0], "low")
            pred_close = unscale(models["close"].predict(X_ohlc)[0], "close")
            pred_volume = unscale(models["volume"].predict(X_volume)[0], "volume")

            results.append({
                "date": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "open": pred_open,
                "high": pred_high,
                "low": pred_low,
                "close": pred_close,
                "volume": pred_volume
            })
        except Exception as e:
            print(f"❌ Prediction failed for row {i}: {e}")

    return results
@app.get("/")
def root():
    return {"message": "Use /predict?asset=btc&timestamp=... to get OHLCV predictions."}


