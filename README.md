# RLTradingStrategist  

**Reinforcement Learning-Based Algorithmic Trading in Cryptocurrency Markets**  

This project explores the use of **Machine Learning** and **Reinforcement Learning (RL)** to design and deploy a cryptocurrency trading platform. It integrates **time series forecasting** with **autonomous trading strategies**, targeting BTC/USDT and ETH/USDT pairs.  

The platform is built with **FastAPI** as the backend and features a simple **web interface** to provide:  
1. **OHLCV Prediction** – Forecasts Open, High, Low, Close, and Volume values using **XGBoost**.  
2. **RL Strategy Advisor** – Suggests optimal trading actions (**Buy / Sell / Hold**) using RL agents trained in custom trading environments.  

---

##  Features  

- **Data-Driven Forecasting**  
  - Hourly BTC/ETH data (2020–2023) collected from Binance (via CCXT/yfinance).  
  - Technical indicators: RSI, EMA/SMA, MACD, VWAP, Aroon, Bollinger Bands, ADX, ATR.  
  - Feature engineering with rolling stats, normalization, and volume-based features.  

- **Supervised Learning (XGBoost)**  
  - Separate models for predicting OHLCV components.  
  - Time-aware validation (80/20 split).  
  - Metrics: MAE, RMSE, accuracy.  

- **Reinforcement Learning Strategy Advisor**  
  - Custom **Gymnasium** trading environments for BTC & ETH.  
  - Algorithms: **A2C, PPO, Recurrent PPO (RPPO)** with **Stable-Baselines3**.  
  - Reward function integrates profit, volatility, drawdown, and trade penalties.  
  - Hyperparameter optimization via **Optuna**.  

- **Backtesting & Evaluation**  
  - ETH strategy achieved strong Sharpe (1.6), Sortino (6.9), Calmar (6.5).  
  - BTC strategy showed conservative performance with +28.59% return and solid risk-adjusted metrics.  
  - RL agents significantly outperformed Buy & Hold benchmarks (BTC CAGR ≈ 2674%, ETH CAGR ≈ 600%).  

- **Deployment**  
  - Backend: **FastAPI** for REST API endpoints.  
  - Frontend: Web interface for predictions & strategy recommendations.  
  - Demo available [here](https://drive.google.com/file/d/1flgTMzg5V5RG8PklFU77W94gdCAQGdjy/view).  

---

## Tech Stack  

- **Languages**: Python  
- **Libraries/Frameworks**:  
  - Machine Learning: `xgboost`, `scikit-learn`  
  - Reinforcement Learning: `stable-baselines3`, `optuna`, `gymnasium`  
  - Backtesting: `backtesting.py`  
  - API & Deployment: `FastAPI`, `uvicorn`  
  - Data: `ccxt`, `yfinance`, `pandas`, `numpy`, `matplotlib`  

---

## Installation  

Clone the repository:  
```bash
git clone https://github.com/Vaish-Ventra/RLTradingStrategist.git
cd RLTradingStrategist
---

```

### Access API endpoints

* `POST /predict` → Returns OHLCV predictions (BTC/ETH).
* `POST /strategy` → Returns RL-based trading decision.

---

##  Results (Highlights)

* **BTC RL Agent**:

  * Total Return: **+754%**
  * Sharpe: **6.59**, Calmar: **171.5**
  * Max Drawdown: **-15.6%**

* **ETH RL Agent**:

  * Total Return: **+251%**
  * Sharpe: **4.23**, Calmar: **32.7**
  * Max Drawdown: **-9.7%**

---

## Contributors

* **Vaishnavi Ventrapragada** (Team Lead)
* **Devanshi Mahto** – [GitHub](https://github.com/Devanshi-Mahto)
* **Priyanshi Mahto** – [GitHub](https://github.com/priyanshi-mahto)
* **Shambhavi S Vijay** – [GitHub](https://github.com/Shambhavi951)

---

