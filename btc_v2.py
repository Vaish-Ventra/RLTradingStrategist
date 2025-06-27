# Import required libraries and classes
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, ADXIndicator

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
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)  # Converting datetime column to datetime data type
df.set_index('datetime', inplace=True)  # Setting datetime column as index
df.dropna(subset=['Close'], inplace=True)  # Removing any rows without a close price to eliminate errors

# Strategy Class
class Strat_1(Strategy):

    # Initializing values of variables to those best suited for the strategy
    bb_window = 20
    bb_std = 2.0
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    adx_period = 14
    adx_threshold = 25
    atr_period = 14
    atr_mult = 2.0
    long_hold_limit = 576
    short_hold_limit = 576

    # _init_ method to create and store indicators and initialize some member variables (runs once)
    def init(self):
        
        # Casting NumPy array to pandas Series for ta-lib compatibility
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # Ensure integer window values for rolling functions
        self.bb_window   = int(self.bb_window)
        self.macd_fast   = int(self.macd_fast)
        self.macd_slow   = int(self.macd_slow)
        self.macd_signal = int(self.macd_signal)
        self.adx_period  = int(self.adx_period)
        self.atr_period  = int(self.atr_period)

        # Indicator functions
        bb = BollingerBands(close, window=self.bb_window, window_dev=self.bb_std)
        self.bb_lower = self.I(lambda: bb.bollinger_lband())
        self.bb_upper = self.I(lambda: bb.bollinger_hband())

        macd_ind = MACD(close, window_fast=self.macd_fast,
                        window_slow=self.macd_slow, window_sign=self.macd_signal)
        self.macd_diff = self.I(lambda: macd_ind.macd_diff())

        adx_ind = ADXIndicator(high, low, close, window=self.adx_period)
        self.adx = self.I(lambda: adx_ind.adx())

        atr_ind = AverageTrueRange(high, low, close, window=self.atr_period)
        self.atr = self.I(lambda: atr_ind.average_true_range())

        # Initializing trade tracking variables to execute trailing stop loss and exits
        self.entry_time = None
        self.max_price = None
        self.min_price = None
        self.trailing_stop_price = None

    # next method that runs every candle by checking conditions and updating position
    def next(self):
        i = len(self.data) - 1
        price = self.data.Close[i]
        atr_now = self.atr[i]

        # Conditions to take long position
        cond_long_1 = crossover(self.data.Close, self.bb_lower)
        cond_long_2 = self.macd_diff[i] > 0
        cond_long_3 = self.adx[i] > self.adx_threshold
        long_entry = cond_long_1 and cond_long_2 and cond_long_3

        # Conditions to take short position
        cond_short_1 = crossover(self.bb_upper, self.data.Close)
        cond_short_2 = self.macd_diff[i] < 0
        cond_short_3 = self.adx[i] > self.adx_threshold
        short_entry = cond_short_1 and cond_short_2 and cond_short_3

        # Entry Logic
        if not self.position:
            if long_entry:
                self.buy()
                self.entry_time = self.data.index[i]
                self.max_price = price
                self.trailing_stop_price = price - self.atr_mult * atr_now
            elif short_entry:
                self.sell()
                self.entry_time = self.data.index[i]
                self.min_price = price
                self.trailing_stop_price = price + self.atr_mult * atr_now

        # Stoploss adjustment
        if self.position:
            if self.position.is_long:
                self.max_price = max(self.max_price, price)
                self.trailing_stop_price = self.max_price - self.atr_mult * atr_now

                long_exit = (
                    price < self.trailing_stop_price or
                    (self.entry_time is not None and self.data.index[i] - self.entry_time >= pd.Timedelta(hours=self.long_hold_limit))
                )

                if long_exit:
                    self.position.close()
                    self.entry_time = None
                    self.max_price = None
                    self.trailing_stop_price = None

            elif self.position.is_short:
                self.min_price = min(self.min_price, price)
                self.trailing_stop_price = self.min_price + self.atr_mult * atr_now

                short_exit = (
                    price > self.trailing_stop_price or
                    (self.entry_time is not None and self.data.index[i] - self.entry_time >= pd.Timedelta(hours=self.short_hold_limit))
                )

                if short_exit:
                    self.position.close()
                    self.entry_time = None
                    self.min_price = None
                    self.trailing_stop_price = None

# Backtest instance
bt = Backtest(df, Strat_1,
              cash=100_000,
              commission=0.0015,
              exclusive_orders=True,
              trade_on_close=True)

# Using bt.optimize to obtain the best parameter combination
stats = bt.optimize(
    # Parameter ranges
    bb_window      = range(10, 31, 10),  # 10, 20, 30
    bb_std         = [1.5, 2.0, 2.5],
    macd_fast      = range(8, 17, 4),    # 8, 12, 16
    macd_slow      = [21, 26, 34],
    macd_signal    = [5, 9],
    adx_period     = [14, 20, 28],
    adx_threshold  = [20, 25, 30],
    atr_mult       = [1.5, 2.0, 2.5],
    constraint     = lambda p: p.macd_fast < p.macd_slow,
    maximize       = 'Equity Final [$]',
    method         = 'grid',
    return_heatmap = False
)

# Showing only best performance metrics
print("\nPerformance metrics:")
print(stats)

# Plotting the best run and opening in browser
bt.plot()
