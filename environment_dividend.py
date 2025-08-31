import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import IntEnum

class Action(IntEnum):
    """Trading actions for each asset"""
    SELL = 0    # Sell shares
    HOLD = 1    # Hold current position  
    BUY = 2     # Buy shares

class MultiAssetTradingEnv(gym.Env):
    """
    Multi-asset trading environment for reinforcement learning.
    
    The agent can trade multiple assets simultaneously, with the ability to:
    - Buy, hold, or sell each asset independently
    - Maintain cash positions
    - Use OHLCV data for decision making
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],  # {asset_name: OHLCV DataFrame}
        initial_balance: float = 100000.0,
        lookback_window: int = 20,
        max_position_per_asset: float = 0.3,  # Max 30% of portfolio per asset
        normalize_observations: bool = True,
        buy_fraction: float = 0.2,  # Use 20% of available cash per buy action
        sell_fraction: float = 0.5  # Sell 50% of holdings per
    ):
        super().__init__()
        
        # Environment parameters
        self.data = data
        self.asset_names = list(data.keys())
        self.n_assets = len(self.asset_names)
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.max_position_per_asset = max_position_per_asset
        self.normalize_observations = normalize_observations
        self.buy_fraction = buy_fraction
        self.sell_fraction = sell_fraction
        
        # Validate data consistency
        self._validate_data()
        
        # Environment state
        self.current_step = 0
        self.max_steps = len(next(iter(data.values()))) - lookback_window - 1
        
        # Portfolio state
        self.cash = initial_balance
        self.shares = {asset: 0.0 for asset in self.asset_names}  # Shares owned
        self.portfolio_history = []
        
        # Define action space: one action per asset (sell, hold, buy)
        self.action_space = spaces.MultiDiscrete([3] * self.n_assets)
        
        # Define observation space
        self.observation_space = self._create_observation_space()
        
    def _validate_data(self):
        """Validate that all assets have consistent data"""
        if not self.data:
            raise ValueError("No data provided")
            
        # Check that all DataFrames have required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividend']
        for asset, df in self.data.items():
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Asset {asset} missing columns: {missing_cols}")
        
        # Check data length consistency
        lengths = [len(df) for df in self.data.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All assets must have the same number of data points")
            
    def _create_observation_space(self):
        """Create the observation space"""
        # Observation includes:
        # 1. OHLCV data for lookback window for each asset (5 * lookback_window * n_assets)
        # 2. Current portfolio allocation (n_assets + 1 for cash)
        # 3. Technical indicators (optional, can be extended)
        
        market_data_dim = 5 * self.lookback_window * self.n_assets  # OHLCV
        portfolio_dim = self.n_assets + 1  # shares + cash ratio
        technical_dim = self.n_assets * 3  # RSI, SMA_ratio, volatility per asset
        
        total_dim = market_data_dim + portfolio_dim + technical_dim
        
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current closing prices for all assets"""
        current_idx = self.lookback_window + self.current_step
        prices = {}
        for asset in self.asset_names:
            price = self.data[asset].iloc[current_idx]['Close']
            # Ensure we get a scalar value, not a Series
            if hasattr(price, 'item'):
                price = price.item()
            prices[asset] = float(price)
        return prices
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        current_idx = self.lookback_window + self.current_step
        obs_parts = []
        
        # 1. Market data (OHLCV for lookback window)
        for asset in self.asset_names:
            asset_data = self.data[asset].iloc[current_idx-self.lookback_window:current_idx]
            ohlcv = asset_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
            
            if self.normalize_observations:
                # Normalize OHLC by current close, volume by its own mean
                close_prices = ohlcv[:, 3]  # Close prices
                current_close = close_prices[-1]
                ohlcv[:, :4] = ohlcv[:, :4] / current_close  # Normalize OHLC
                ohlcv[:, 4] = ohlcv[:, 4] / np.mean(ohlcv[:, 4])  # Normalize volume
            
            obs_parts.append(ohlcv.flatten())
        
        # 2. Portfolio allocation
        current_prices = self._get_current_prices()
        total_value = self._calculate_portfolio_value(current_prices)
        
        # Cash ratio
        cash_ratio = self.cash / total_value if total_value > 0 else 1.0
        obs_parts.append([cash_ratio])
        
        # Asset allocation ratios
        for asset in self.asset_names:
            asset_value = self.shares[asset] * current_prices[asset]
            asset_ratio = asset_value / total_value if total_value > 0 else 0.0
            obs_parts.append([asset_ratio])
        
        # 3. Technical indicators
        for asset in self.asset_names:
            asset_data = self.data[asset].iloc[current_idx-self.lookback_window:current_idx]
            
            # RSI (simplified)
            close_prices = asset_data['Close'].values.flatten()  # Ensure 1D array
            rsi = self._calculate_rsi(close_prices)
            
            # SMA ratio (current price vs 10-day average)
            sma_10 = np.mean(close_prices[-10:]) if len(close_prices) >= 10 else close_prices[-1]
            sma_ratio = close_prices[-1] / sma_10 if sma_10 > 0 else 1.0
            
            # Volatility (standard deviation of returns)
            if len(close_prices) > 1:
                returns = np.diff(close_prices) / close_prices[:-1]
                volatility = np.std(returns) if len(returns) > 0 else 0.0
            else:
                volatility = 0.0
            
            obs_parts.extend([[rsi], [sma_ratio], [volatility]])
        
        return np.concatenate(obs_parts).astype(np.float32)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        total_value = float(self.cash)
        for asset in self.asset_names:
            asset_value = self.shares[asset] * current_prices[asset]
            # Ensure we're working with scalar values
            if hasattr(asset_value, 'item'):
                asset_value = asset_value.item()
            total_value += float(asset_value)
        return total_value
    
    def _execute_trades(self, actions: np.ndarray, current_prices: Dict[str, float]) -> None:
        """Execute trades based on actions"""
        total_portfolio_value = self._calculate_portfolio_value(current_prices)
        
        for i, asset in enumerate(self.asset_names):
            action = actions[i]
            current_price = current_prices[asset]
            current_shares = self.shares[asset]
            
            if action == Action.BUY and self.cash > 0:
                # Calculate maximum buyable amount (considering position limits)
                max_position_value = total_portfolio_value * self.max_position_per_asset
                current_position_value = current_shares * current_price
                available_position_value = max_position_value - current_position_value
                
                # Use portion of available cash (e.g., 20% per buy signal)
                buy_amount = min(
                    self.cash * self.buy_fraction,  # Use 20% of available cash
                    available_position_value,
                    self.cash
                )
                
                if buy_amount > current_price:  # Ensure we can buy at least one share worth
                    shares_to_buy = buy_amount / current_price
                    
                    self.shares[asset] += shares_to_buy
                    self.cash -= buy_amount
            
            elif action == Action.SELL and current_shares > 0:
                # Sell portion of holdings (e.g., 50% per sell signal)
                shares_to_sell = current_shares * self.sell_fraction
                sell_value = shares_to_sell * current_price
                
                self.shares[asset] -= shares_to_sell
                self.cash += sell_value
            
            # Action.HOLD requires no action
    
    def _calculate_reward(self, previous_value: float, current_value: float) -> float:
        """Calculate reward based on portfolio performance"""
        # Portfolio return
        portfolio_return = (current_value - previous_value) / previous_value if previous_value > 0 else 0
        
        # Additional penalties/rewards
        cash_penalty = 0.0
        if self.cash / current_value > 0.95:  # Penalize holding too much cash
            cash_penalty = -0.001
        
        # Risk-adjusted return (simplified Sharpe-like ratio)
        reward = portfolio_return + cash_penalty
        
        return reward
    
    # Credit dividends to cash
    def _apply_dividends(self, current_idx: int):
        for asset in self.asset_names:
            df = self.data[asset]
            if 'Dividend' in df.columns:
                dividend = df.iloc[current_idx]['Dividend']
                if dividend > 0:
                    self.cash += self.shares[asset] * dividend

    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Get current prices
        current_prices = self._get_current_prices()
        previous_value = self._calculate_portfolio_value(current_prices)
        
        # Execute trades
        self._execute_trades(action, current_prices)
        # Inside step()
        self._apply_dividends(self.lookback_window + self.current_step)
        
        # Move to next time step
        self.current_step += 1
        
        # Calculate new portfolio value
        new_prices = self._get_current_prices() if self.current_step < self.max_steps else current_prices
        current_value = self._calculate_portfolio_value(new_prices)
        
        # Calculate reward
        reward = self._calculate_reward(previous_value, current_value)
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Store portfolio history
        self.portfolio_history.append({
            'step': self.current_step,
            'portfolio_value': current_value,
            'cash': self.cash,
            'shares': self.shares.copy(),
            'prices': new_prices.copy() if self.current_step < self.max_steps else current_prices,
            'action': action.copy()
        })
        
        # Get next observation
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape[0])
        
        # Info dictionary
        info = {
            'portfolio_value': current_value,
            'cash': self.cash,
            'portfolio_return': (current_value - previous_value) / previous_value if previous_value > 0 else 0
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset environment state
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares = {asset: 0.0 for asset in self.asset_names}
        self.portfolio_history = []
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            'portfolio_value': self.initial_balance,
            'cash': self.cash
        }
        
        return obs, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment (optional)"""
        if mode == 'human':
            current_prices = self._get_current_prices()
            portfolio_value = self._calculate_portfolio_value(current_prices)
            
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${portfolio_value:,.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            for asset in self.asset_names:
                print(f"{asset}: {self.shares[asset]:.4f} shares @ ${current_prices[asset]:.2f}")
            print("-" * 50)
    
    def get_portfolio_history(self) -> List[Dict]:
        """Get the complete portfolio history"""
        return self.portfolio_history
