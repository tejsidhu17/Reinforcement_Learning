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

class MultiAssetTradingEnvForex(gym.Env):
    """
    Multi-asset trading environment for reinforcement learning with 10x leverage.
    
    The agent can trade multiple assets simultaneously, with the ability to:
    - Buy, hold, or sell each asset independently
    - Maintain cash positions
    - Use OHLCV data for decision making
    - Apply 10x leverage to all trades
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],  # {asset_name: OHLCV DataFrame}
        initial_balance: float = 100000.0,
        lookback_window: int = 20,
        max_position_per_asset: float = 0.3,  # Max 30% of portfolio per asset
        normalize_observations: bool = True,
        buy_fraction: float = 0.5,  # Use 50% of available cash per buy action
        sell_fraction: float = 0.5,  # Sell 50% of holdings per sell action
        leverage: float = 10.0,  # 10x leverage
        margin_requirement: float = 0.1  # 10% margin requirement (1/leverage)
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
        self.leverage = leverage
        self.margin_requirement = margin_requirement
        
        # Validate data consistency
        self._validate_data()
        
        # Environment state
        self.current_step = 0
        self.max_steps = len(next(iter(data.values()))) - lookback_window - 1
        
        # Portfolio state
        self.cash = initial_balance
        self.shares = {asset: 0.0 for asset in self.asset_names}  # Leveraged shares owned
        self.entry_prices = {asset: 0.0 for asset in self.asset_names}  # Track entry prices for P&L calculation
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
        required_columns = ['Open', 'High', 'Low', 'Close']
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
        # 1. OHLCV data for lookback window for each asset (4 * lookback_window * n_assets)
        # 2. Current portfolio allocation (n_assets + 1 for cash)
        # 3. Technical indicators (3 per asset)
        # 4. Leverage metrics (unrealized P&L, margin utilization)
        
        market_data_dim = 4 * self.lookback_window * self.n_assets  # OHLC
        portfolio_dim = self.n_assets + 1  # shares + cash ratio
        technical_dim = self.n_assets * 3  # RSI, SMA_ratio, volatility per asset
        leverage_dim = self.n_assets + 2  # unrealized P&L per asset + margin utilization + available buying power
        
        total_dim = market_data_dim + portfolio_dim + technical_dim + leverage_dim
        
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
    
    def _calculate_unrealized_pnl(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate unrealized P&L for each asset position"""
        unrealized_pnl = {}
        for asset in self.asset_names:
            if self.shares[asset] != 0 and self.entry_prices[asset] != 0:
                # Leveraged P&L calculation
                price_change = current_prices[asset] - self.entry_prices[asset]
                pnl = self.shares[asset] * price_change  # Already leveraged since shares are leveraged
                unrealized_pnl[asset] = pnl
            else:
                unrealized_pnl[asset] = 0.0
        return unrealized_pnl
    
    def _calculate_margin_used(self, current_prices: Dict[str, float]) -> float:
        """Calculate total margin used across all positions"""
        margin_used = 0.0
        for asset in self.asset_names:
            if abs(self.shares[asset]) > 0:
                # Margin requirement is based on notional value
                notional_value = abs(self.shares[asset]) * current_prices[asset]
                margin_used += notional_value * self.margin_requirement
        return margin_used
    
    def _calculate_available_buying_power(self, current_prices: Dict[str, float]) -> float:
        """Calculate available buying power considering leverage and margin"""
        margin_used = self._calculate_margin_used(current_prices)
        unrealized_pnl = sum(self._calculate_unrealized_pnl(current_prices).values())
        
        # Available equity = cash + unrealized P&L
        available_equity = self.cash + unrealized_pnl
        
        # Available margin = available equity - margin used
        available_margin = available_equity - margin_used
        
        # Buying power = available margin * leverage
        buying_power = max(0, available_margin * self.leverage)
        
        return buying_power
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        current_idx = self.lookback_window + self.current_step
        obs_parts = []
        
        # 1. Market data (OHLC for lookback window)
        for asset in self.asset_names:
            asset_data = self.data[asset].iloc[current_idx-self.lookback_window:current_idx]
            ohlc = asset_data[['Open', 'High', 'Low', 'Close']].values
            
            if self.normalize_observations:
                # Normalize OHLC by current close
                close_prices = ohlc[:, 3]  # Close prices
                current_close = close_prices[-1]
                ohlc[:, :4] = ohlc[:, :4] / current_close  # Normalize OHLC
            
            obs_parts.append(ohlc.flatten())
        
        # 2. Portfolio allocation
        current_prices = self._get_current_prices()
        total_value = self._calculate_portfolio_value(current_prices)
        
        # Cash ratio
        cash_ratio = self.cash / total_value if total_value > 0 else 1.0
        obs_parts.append([cash_ratio])
        
        # Asset allocation ratios (based on notional value of leveraged positions)
        for asset in self.asset_names:
            notional_value = abs(self.shares[asset]) * current_prices[asset]
            asset_ratio = notional_value / (total_value * self.leverage) if total_value > 0 else 0.0
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
        
        # 4. Leverage metrics
        unrealized_pnl = self._calculate_unrealized_pnl(current_prices)
        for asset in self.asset_names:
            # Normalize unrealized P&L by initial balance
            normalized_pnl = unrealized_pnl[asset] / self.initial_balance
            obs_parts.append([normalized_pnl])
        
        # Margin utilization
        margin_used = self._calculate_margin_used(current_prices)
        margin_utilization = margin_used / total_value if total_value > 0 else 0.0
        obs_parts.append([margin_utilization])
        
        # Available buying power ratio
        buying_power = self._calculate_available_buying_power(current_prices)
        buying_power_ratio = buying_power / (total_value * self.leverage) if total_value > 0 else 1.0
        obs_parts.append([buying_power_ratio])
        
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
        """Calculate total portfolio value including unrealized P&L from leveraged positions"""
        # Start with cash
        total_value = float(self.cash)
        
        # Add unrealized P&L from all positions
        unrealized_pnl = self._calculate_unrealized_pnl(current_prices)
        total_value += sum(unrealized_pnl.values())
        
        return total_value
    
    def _execute_trades(self, actions: np.ndarray, current_prices: Dict[str, float]) -> None:
        """Execute leveraged trades based on actions"""
        buying_power = self._calculate_available_buying_power(current_prices)
        total_portfolio_value = self._calculate_portfolio_value(current_prices)
        
        for i, asset in enumerate(self.asset_names):
            action = actions[i]
            current_price = current_prices[asset]
            current_shares = self.shares[asset]
            
            if action == Action.BUY:
                # Calculate maximum buyable amount with leverage
                max_position_value = total_portfolio_value * self.max_position_per_asset * self.leverage
                current_position_value = abs(current_shares) * current_price
                available_position_value = max_position_value - current_position_value
                
                # Use portion of available buying power
                buy_amount = min(
                    buying_power * self.buy_fraction,
                    available_position_value,
                    buying_power
                )
                
                if buy_amount > current_price:  # Ensure we can buy at least one unit
                    leveraged_shares_to_buy = buy_amount / current_price
                    margin_required = buy_amount * self.margin_requirement
                    
                    if self.cash >= margin_required:
                        # Update position
                        if current_shares <= 0:  # New long position or closing short
                            if current_shares < 0:  # Closing short position first
                                # Realize P&L from closing short position
                                short_pnl = current_shares * (self.entry_prices[asset] - current_price)
                                self.cash += short_pnl
                                leveraged_shares_to_buy += abs(current_shares)  # Add shares to close short
                            
                            self.shares[asset] = leveraged_shares_to_buy
                            self.entry_prices[asset] = current_price
                        else:  # Adding to existing long position
                            # Weighted average entry price
                            total_shares = current_shares + leveraged_shares_to_buy
                            self.entry_prices[asset] = (
                                (current_shares * self.entry_prices[asset]) + 
                                (leveraged_shares_to_buy * current_price)
                            ) / total_shares
                            self.shares[asset] = total_shares
                        
                        # Deduct margin requirement from cash
                        self.cash -= margin_required
            
            elif action == Action.SELL and (current_shares > 0 or buying_power > 0):
                if current_shares > 0:
                    # Sell portion of long holdings
                    shares_to_sell = current_shares * self.sell_fraction
                    
                    # Realize P&L
                    realized_pnl = shares_to_sell * (current_price - self.entry_prices[asset])
                    self.cash += realized_pnl
                    
                    # Return margin
                    margin_returned = shares_to_sell * current_price * self.margin_requirement
                    self.cash += margin_returned
                    
                    # Update position
                    self.shares[asset] -= shares_to_sell
                    
                    if self.shares[asset] <= 0:
                        self.shares[asset] = 0
                        self.entry_prices[asset] = 0
                
                else:
                    # Open short position
                    max_short_value = total_portfolio_value * self.max_position_per_asset * self.leverage
                    short_amount = min(
                        buying_power * self.buy_fraction,
                        max_short_value,
                        buying_power
                    )
                    
                    if short_amount > current_price:
                        leveraged_shares_to_short = short_amount / current_price
                        margin_required = short_amount * self.margin_requirement
                        
                        if self.cash >= margin_required:
                            self.shares[asset] = -leveraged_shares_to_short  # Negative for short
                            self.entry_prices[asset] = current_price
                            self.cash -= margin_required
            
            # Action.HOLD requires no action
    
    def _calculate_reward(self, previous_value: float, current_value: float) -> float:
        """Calculate reward based on leveraged portfolio performance"""
        # Portfolio return (amplified by leverage effect)
        portfolio_return = ((current_value - previous_value) / previous_value) if previous_value > 0 else 0
        
        # Scale reward to account for leverage amplification
        leveraged_reward = portfolio_return * self.leverage
        
        # Additional penalties/rewards
        current_prices = self._get_current_prices()
        margin_used = self._calculate_margin_used(current_prices)
        
        # Penalize high margin utilization (risk management)
        margin_penalty = 0.0
        margin_ratio = margin_used / current_value if current_value > 0 else 0
        if margin_ratio > 0.8:  # Penalize if using >80% of equity as margin
            margin_penalty = -0.01 * (margin_ratio - 0.8)
        
        # Penalize holding too much cash (opportunity cost with leverage)
        cash_penalty = 0.0
        cash_ratio = self.cash / current_value if current_value > 0 else 1.0
        if cash_ratio > 0.5:  # Penalize holding >50% cash when leverage is available
            cash_penalty = -0.001 * (cash_ratio - 0.5)
        
        reward = leveraged_reward + margin_penalty + cash_penalty
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Get current prices
        current_prices = self._get_current_prices()
        previous_value = self._calculate_portfolio_value(current_prices)
        
        # Execute trades
        self._execute_trades(action, current_prices)
        
        # Move to next time step
        self.current_step += 1
        
        # Calculate new portfolio value
        new_prices = self._get_current_prices() if self.current_step < self.max_steps else current_prices
        current_value = self._calculate_portfolio_value(new_prices)
        
        # Check for margin call (portfolio value drops too low)
        margin_used = self._calculate_margin_used(new_prices)
        if current_value < margin_used * 1.2:  # Margin call if equity < 120% of margin used
            # Force close all positions (simplified margin call handling)
            for asset in self.asset_names:
                if self.shares[asset] != 0:
                    # Realize P&L
                    if self.entry_prices[asset] != 0:
                        realized_pnl = self.shares[asset] * (new_prices[asset] - self.entry_prices[asset])
                        self.cash += realized_pnl
                    
                    # Return/pay margin
                    margin_returned = abs(self.shares[asset]) * new_prices[asset] * self.margin_requirement
                    self.cash += margin_returned
                    
                    self.shares[asset] = 0
                    self.entry_prices[asset] = 0
            
            current_value = self.cash  # Recalculate after force close
        
        # Calculate reward
        reward = self._calculate_reward(previous_value, current_value)
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps or current_value <= 0
        truncated = False
        
        # Store portfolio history
        self.portfolio_history.append({
            'step': self.current_step,
            'portfolio_value': current_value,
            'cash': self.cash,
            'shares': self.shares.copy(),
            'entry_prices': self.entry_prices.copy(),
            'prices': new_prices.copy() if self.current_step < self.max_steps else current_prices,
            'action': action.copy(),
            'unrealized_pnl': self._calculate_unrealized_pnl(new_prices),
            'margin_used': self._calculate_margin_used(new_prices),
            'buying_power': self._calculate_available_buying_power(new_prices)
        })
        
        # Get next observation
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape[0])
        
        # Info dictionary
        info = {
            'portfolio_value': current_value,
            'cash': self.cash,
            'portfolio_return': ((current_value - previous_value) / previous_value) if previous_value > 0 else 0,
            'leverage_ratio': margin_used / current_value if current_value > 0 else 0,
            'unrealized_pnl': sum(self._calculate_unrealized_pnl(new_prices).values()),
            'margin_used': margin_used,
            'buying_power': self._calculate_available_buying_power(new_prices)
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset environment state
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares = {asset: 0.0 for asset in self.asset_names}
        self.entry_prices = {asset: 0.0 for asset in self.asset_names}
        self.portfolio_history = []
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            'portfolio_value': self.initial_balance,
            'cash': self.cash,
            'leverage_ratio': 0.0,
            'margin_used': 0.0,
            'buying_power': self.initial_balance * self.leverage
        }
        
        return obs, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment (optional)"""
        if mode == 'human':
            current_prices = self._get_current_prices()
            portfolio_value = self._calculate_portfolio_value(current_prices)
            unrealized_pnl = self._calculate_unrealized_pnl(current_prices)
            margin_used = self._calculate_margin_used(current_prices)
            buying_power = self._calculate_available_buying_power(current_prices)
            
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${portfolio_value:,.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Margin Used: ${margin_used:,.2f}")
            print(f"Buying Power: ${buying_power:,.2f}")
            print(f"Leverage Ratio: {margin_used/portfolio_value:.2f}x" if portfolio_value > 0 else "N/A")
            
            for asset in self.asset_names:
                position_value = abs(self.shares[asset]) * current_prices[asset]
                position_type = "LONG" if self.shares[asset] > 0 else "SHORT" if self.shares[asset] < 0 else "NONE"
                print(f"{asset}: {self.shares[asset]:.4f} shares ({position_type}) @ ${current_prices[asset]:.2f} | "
                      f"Value: ${position_value:,.2f} | P&L: ${unrealized_pnl[asset]:,.2f}")
            print("-" * 70)
    
    def get_portfolio_history(self) -> List[Dict]:
        """Get the complete portfolio history"""
        return self.portfolio_history