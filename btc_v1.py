# Import required libraries and classes
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, AroonIndicator

# Load data from CSV file into a pandas Dataframe
df = pd.read_csv(r'C:\Users\Shambhavi S Vijay\OneDrive\Desktop\proj\predicted.csv')

# Renaming columns as backtesting.py requires specific column names to work accurately
df.rename(columns={
    'timestamp': 'datetime',
    'open_predicted': 'Open',
    'high_predicted': 'High',
    'low_predicted': 'Low',
    'close_predicted': 'Close',
    'Predicted_Volume': 'Volume'
}, inplace=True)

# Modifying the DataFrame so that it can be used without errors in Backtesting.py
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True) # Converting datetime column to datetime data type
df.set_index('datetime', inplace=True) # Setting datetime column as index
df.dropna(subset=['Close'], inplace=True)# Removing any columns without a close price to eliminate errors

# Strategy Class
class Strat_1(Strategy):

    # Initializing values of variables to those best suited for the strategy(Modify this!!)
    rsi_up = 75 
    rsi_down = 30
    ema_fastest_period = 9
    ema_fast_period = 21
    ema_slow_period = 50
    volume_sma_fast_period = 5
    volume_sma_slow_period = 50
    aroon_period = 25
    long_hold_limit = 576
    short_hold_limit = 8
    trailing_pct = 0.02  # 2% trailing stoploss

    # _init_ method to create and store indicators and initialize some member variables (runs once)
    def init(self):
        close = self.data.Close
        volume = self.data.Volume
        high = self.data.High
        low = self.data.Low
        
        # Indicator functions
        self.rsi = self.I(lambda x: RSIIndicator(pd.Series(x), window=14).rsi().rolling(3).mean(), close) # Rolling mean is taken to smooth our RSI values and                                                                                                              reduce noise
        self.ema_fastest = self.I(lambda x: EMAIndicator(pd.Series(x), window=self.ema_fastest_period).ema_indicator(), close)
        self.ema_fast = self.I(lambda x: EMAIndicator(pd.Series(x), window=self.ema_fast_period).ema_indicator(), close)
        self.ema_slow = self.I(lambda x: EMAIndicator(pd.Series(x), window=self.ema_slow_period).ema_indicator(), close)
        self.volume_sma_fast = self.I(lambda x: SMAIndicator(pd.Series(x), window=self.volume_sma_fast_period).sma_indicator(), volume)
        self.volume_sma_slow = self.I(lambda x: SMAIndicator(pd.Series(x), window=self.volume_sma_slow_period).sma_indicator(), volume)
        self.aroon_up = self.I(lambda h, l: AroonIndicator(high=pd.Series(h), low=pd.Series(l), window=self.aroon_period).aroon_up(), high, low)
        self.aroon_down = self.I(lambda h, l: AroonIndicator(high=pd.Series(h), low=pd.Series(l), window=self.aroon_period).aroon_down(), high, low)
        self.vwap = self.I(lambda h, l, c, v: ((h + l + c) / 3 * v).cumsum() / v.cumsum(), high, low, close, volume)
        
        # Initializing trade tracking variables to execute trailing stop loss and exits
        self.entry_time = None
        self.max_price = None
        self.min_price = None
        self.trailing_stop_price = None
        
    # next method that runs every candle by checking conditions and updating position
    def next(self):
        i = len(self.data) - 1
        price = self.data.Close[i]

        # Conditions to take long position
        cond_long_1 = self.rsi[i] > self.rsi_up
        cond_long_2 = self.volume_sma_fast[i] > self.volume_sma_slow[i]
        cond_long_3 = self.ema_fastest[i] > self.ema_fast[i] > self.ema_slow[i]
        cond_long_4 = self.aroon_up[i] < self.aroon_down[i]
        long_entry = sum([cond_long_1, cond_long_2, cond_long_3, cond_long_4]) >= 3

        # Conditions to take short position
        cond_short_1 = self.ema_fastest[i] < self.ema_fast[i] < self.ema_slow[i]
        cond_short_2 = crossover(self.vwap, self.ema_fastest)
        cond_short_3 = self.rsi[i] < self.rsi_down
        cond_short_4 = self.aroon_down[i] < self.aroon_up[i]
        short_entry = sum([cond_short_1, cond_short_2, cond_short_3, cond_short_4]) >= 3

        # Entry Logic
        if not self.position:
            if long_entry:
                self.buy()
                self.entry_time = self.data.index[i]
                self.entry_price = price
                self.max_price = price
                self.trailing_stop_price = price * (1 - self.trailing_pct)
            elif short_entry:
                self.sell()
                self.entry_time = self.data.index[i]
                self.entry_price = price
                self.min_price = price
                self.trailing_stop_price = price * (1 + self.trailing_pct)

        # Stoploss adjustment
        if self.position:
            if self.position.is_long:
                self.max_price = max(self.max_price, price)
                self.trailing_stop_price = self.max_price * (1 - self.trailing_pct)

                long_exit = (
                    self.rsi[i] < self.rsi_down or
                    (self.entry_time is not None and self.data.index[i] - self.entry_time >= pd.Timedelta(hours=self.long_hold_limit)) or
                    price < self.trailing_stop_price
                )

                if long_exit:
                    self.position.close()
                    self.entry_time = None
                    self.entry_price = None
                    self.max_price = None
                    self.trailing_stop_price = None

            elif self.position.is_short:
                self.min_price = min(self.min_price, price)
                self.trailing_stop_price = self.min_price * (1 + self.trailing_pct)

                short_exit = (
                    self.vwap[i] > self.ema_fastest[i] or
                    (self.entry_time is not None and self.data.index[i] - self.entry_time >= pd.Timedelta(hours=self.short_hold_limit)) or
                    price > self.trailing_stop_price
                )

                if short_exit:
                    self.position.close()
                    self.entry_time = None
                    self.entry_price = None
                    self.min_price = None
                    self.trailing_stop_price = None


# Manual optimization loop
param_sets = [
    {"rsi_up": 70, "ema_fastest_period": 5, "ema_fast_period": 15, "ema_slow_period": 45},
    {"rsi_up": 75, "ema_fastest_period": 9, "ema_fast_period": 21, "ema_slow_period": 50},
    {"rsi_up": 80, "ema_fastest_period": 12, "ema_fast_period": 30, "ema_slow_period": 60},
]

best = None
best_equity = -np.inf

for params in param_sets:
    for key, value in params.items():
        setattr(Strat_1, key, value)

    bt = Backtest(df, Strat_1, cash=100_000, commission=0.0015, exclusive_orders=True)
    stats = bt.run()
    print(f"Tested params: {params}, Equity: {stats['Equity Final [$]']}")

    if stats["Equity Final [$]"] > best_equity:
        best_equity = stats["Equity Final [$]"]
        best = stats

# Showing best result
print("\n Performance Metrics:")
print(best)
bt.plot()