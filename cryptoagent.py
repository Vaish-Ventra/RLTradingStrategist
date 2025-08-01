# main.py
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from datetime import datetime, timezone
import uvicorn
from stable_baselines3 import PPO
import gymnasium as gym
from enum import Enum
from DataCollector import create_feature_df
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://10.15.2.129:8080"   # Add your frontend origin here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this to the exact frontend origin for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Actions(Enum):
    Hold = 0
    Sell = -1
    Buy = 1

class Positions(Enum):
    Short = -1
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

class CryptoEnv3(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 3}

    def __init__(self, df, window_size, frame_bound, initial_amount=100000, render_mode=None, market_condition= None):
        super().__init__()
        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.initial_amount = initial_amount
        self.render_mode = render_mode
        self.trade_fee_percent = 0.001  # 0.1%
        self.market_condition = market_condition

        self._init_data()
        self._init_spaces()

        self.action_map = {
          0: Actions.Hold.value,
          1: Actions.Sell.value,
          2: Actions.Buy.value,
         }
        self.action_space = gym.spaces.Discrete(len(self.action_map))


    def _init_data(self):
        prices = self.df["close"].values[
            self.frame_bound[0] - self.window_size : self.frame_bound[1]
        ].astype(np.float32)

        # Prompt-based simulation
        if self.market_condition == "high_volatility":
            noise = np.random.normal(0, 0.05, size=prices.shape)
            prices = prices * (1 + noise)

        elif self.market_condition == "bull_market":
            trend = np.linspace(1, 1.2, num=len(prices))  # 20% upward drift
            prices = prices * trend

        elif self.market_condition == "bear_market":
            trend = np.linspace(1, 0.8, num=len(prices))  # 20% downward drift
            prices = prices * trend

        elif self.market_condition == "sideways":
            mean_price = np.mean(prices)
            prices = mean_price + np.random.normal(0, 0.002 * mean_price, size=prices.shape)  # ~0.2% std dev

        self.prices = prices

        self.signal_features = self.df.iloc[
            self.frame_bound[0] - self.window_size : self.frame_bound[1], 1:
        ].values.astype(np.float32)

        self.shape = (self.window_size, self.signal_features.shape[1])
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1

    def _init_spaces(self):
    # original shape for signal features
        self.feature_dim = self.signal_features.shape[1]
    # +2 for balance and holdings
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size * self.feature_dim + 2,),
            dtype=np.float32,
        )

        self.position_map = {
          0: Positions.Short.value,
          1: Positions.Long.value,
         }
        self.position_space = gym.spaces.Discrete(len(self.position_map))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = [None] * self._start_tick + [self._position]

        # Capital tracking
        self.balance = self.initial_amount
        self.holdings = 0
        self.net_worth = self.initial_amount
        self.prev_net_worth = self.net_worth
        self._total_profit = self.initial_amount
        self._total_reward = 0.0

        self.return_history = []
        self.net_worth_history = [self.net_worth]
        self.prev_action = Actions.Hold.value

        self._truncated = False
        return self._get_observation(), self._get_info()

    def _get_observation(self):
        signal_obs = self.signal_features[
            self._current_tick - self.window_size + 1 : self._current_tick + 1
        ].flatten()

        norm_balance = self.balance / self.initial_amount
        norm_holdings = self.holdings  # you can normalize this if needed

        return np.concatenate([signal_obs, [norm_balance, norm_holdings]]).astype(np.float32)

    def get_observation_from_tick(self, tick, balance, holdings):
        self._current_tick = tick
        self.balance = balance
        self.holdings = holdings
        return self._get_observation()

    def step(self, action_index):
        action = self.action_map[action_index]
        self._current_tick += 1
        self._truncated = self._current_tick >= self._end_tick

        reward = self._calculate_reward(action)
        self._total_reward += reward

        self._update_profit(action)
        self._handle_position_change()
        self._position_history.append(self._position)

        return (
            self._get_observation(),
            reward,
            False,  # No terminal condition except truncated
            self._truncated,
            self._get_info(),
        )

    #def _handle_position_change(self, action):
    #    if (
    #        (action == Actions.Buy.value and self._position == Positions.Short)
    #        or (action == Actions.Sell.value and self._position == Positions.Long)
    #    ):
    #        self._position = self._position.opposite()
    #        self._last_trade_tick = self._current_tick

    def _handle_position_change(self):
        if self.holdings > 0:
            self._position = Positions.Long
        elif self.holdings <= 0:
            self._position = Positions.Short

    def _calculate_reward(self, action):
        current_price = self.prices[self._current_tick]

        # Execute trades
       # if self.net_worth<= 0.1 * self.initial_amount:
       #     self.balance = self.initial_amount
       #     self.holdings = 0
        if action == Actions.Buy.value and self.balance >= current_price:
            risk_per_trade=0.05*self.balance
            pos_size= risk_per_trade/ 0.05   #stop loss= 10%-->0.1
            to_buy = min(pos_size // current_price, self.balance // current_price)
            self.holdings += to_buy
            self.balance -= current_price *(1 + self.trade_fee_percent) * to_buy
        elif action == Actions.Sell.value and self.holdings > 0:
            risk_per_trade=0.05*self.balance
            pos_size= risk_per_trade/ 0.05   #stop loss= 10%-->0.1
            to_sell= (pos_size // current_price)
            self.holdings -= to_sell
            self.balance += current_price *(1 + self.trade_fee_percent) *(to_sell)

        # Update net worth
        self.net_worth = self.balance + self.holdings * current_price
        self.net_worth_history.append(self.net_worth)

        # Profit reward normalized to initial amount
        profit_reward = (self.net_worth - self.prev_net_worth) / self.initial_amount
        self.prev_net_worth = self.net_worth

        # Daily return for volatility
        if len(self.net_worth_history) > 1:
            daily_return = (
                self.net_worth_history[-1] / self.net_worth_history[-2] - 1
            )
            self.return_history.append(daily_return)

        volatility_penalty = (
            np.std(self.return_history[-self.window_size :])
            if len(self.return_history) >= self.window_size
            else 0
        )

        # Drawdown penalty
        peak = max(self.net_worth_history) if self.net_worth_history else self.initial_amount
        drawdown = (self.net_worth - peak) / (peak + 1e-8)
        drawdown_penalty = abs(drawdown) if drawdown < 0 else 0

        # Trade penalty
        # trade_penalty = 1.0 if action != self.prev_action else 0
        trade_penalty = 0.001 * current_price  # proportional to cost
        self.prev_action = action

        # Final reward
        # reward = (
        #     + 100 * profit_reward
        #     - 0.3 * volatility_penalty
        #     - 0.5 * drawdown_penalty
        #     - 0.05 * trade_penalty
        # )

        reward = (
        + 300 * profit_reward
        - 0.1 * volatility_penalty
        - 0.1 * drawdown_penalty
        - 0.0005 * trade_penalty
    )
        reward = np.clip(reward, -30, 30)


        return reward

    def _update_profit(self, action):
        if action == Actions.Hold.value:
            return

        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        shares = (self._total_profit * (1 - self.trade_fee_percent)) / last_trade_price

        if self._position == Positions.Long:
            self._total_profit = shares * current_price * (1 - self.trade_fee_percent)
        elif self._position == Positions.Short:
            price_diff = last_trade_price - current_price
            self._total_profit = shares * (last_trade_price + price_diff) * (1 - self.trade_fee_percent)

    def _get_info(self):
        return {
            "total_reward": self._total_reward,
            "total_profit": self._total_profit,
            "net_worth": self.net_worth,
            "balance": self.balance,
            "holdings": self.holdings,
            "initial_amount": self.initial_amount,
            "current_step": self._current_tick,
            "position": self._position.name,
        }

    def render(self):
        print(f"Step: {self._current_tick} | Net Worth: ₹{self.net_worth:.2f} | Balance: ₹{self.balance:.2f} | Holdings: {self.holdings} | Position: {self._position.name}")

    def close(self):
        pass

# --- 3) UTIL to find nearest tick in your CSV ---
def find_nearest_tick(df: pd.DataFrame, dt: datetime) -> int:
    df_ts = pd.to_datetime(df['timestamp'])
    idx = (df_ts - dt).abs().idxmin()
    return int(idx)

# --- 4) LOAD DATA & MODEL ON STARTUP ---
# adjust paths as needed
#btc_df = pd.read_csv("BTC_DATA_1h_2024.csv")
#btc_df.dropna(inplace=True)
#btc_df.reset_index(drop=True, inplace=True)


# your saved PPO model
ppo_model = PPO.load("FinalModelLowRisk300.zip")

# --- 5) FASTAPI SETUP ---

@app.get("/get-action")
async def get_action(
    datetime_str: str = Query(..., description="ISO-8601 timestamp, e.g. 2022-01-01T12:00:00Z"),
    balance: float = Query(..., description="Account balance at the given time"),
    holdings: float = Query(..., description="BTC holdings at the given time")
):
    print(f"[DEBUG] Received datetime_str: {datetime_str}")
    print(f"[DEBUG] Balance: {balance}, Holdings: {holdings}")

    # parse & normalize to UTC
    try:
        dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00")).astimezone(timezone.utc)
        print(f"[DEBUG] Parsed datetime: {dt}")
    except ValueError:
        print("[ERROR] Invalid datetime format")
        return {"error": "Invalid datetime. Use ISO format like 2022-01-01T12:00:00Z"}
    
    df= create_feature_df(datetime_str)
    print("[DEBUG] DataFrame created", df.tail(1))
    print(f"[DEBUG] Data shape: {df.shape}")
    
    if df.empty:
        print("[ERROR] No data available for the requested datetime.")
        return {"error": "No data available for the requested datetime."}

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # map to nearest index in CSV
    idx = find_nearest_tick(df, dt)
    print(f"[DEBUG] Nearest tick index: {idx}")
    # build a fresh env slice around that tick
    env = CryptoEnv3(
        df=df,
        window_size=30,
        frame_bound=(30, len(df)),
        initial_amount=balance
    )
    env.reset()

    # set env to that tick
    obs = env.reset()[0]
    obs = env.get_observation_from_tick(idx, balance, holdings)
    print(f"[DEBUG] Observation shape: {obs.shape}")

    # get SB3 action index
    action_idx, _ = ppo_model.predict(obs, deterministic=True)
    print(f"[DEBUG] Predicted action index: {action_idx}")

    # change the action index to a string
    action_str = Actions(action_idx).name

    return {
        "requested_datetime": dt.isoformat(),
        "nearest_index": idx,
        "balance": balance,
        "holdings": holdings,
        "action": action_str
    }

def main():
    uvicorn.run("cryptoagent:app", host="10.15.5.156", port=8002, reload= True)

if __name__ == "__main__":
    main()
