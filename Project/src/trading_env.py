import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    """Custom Gym environment for algorithmic trading"""
    
    def __init__(self, data, initial_balance=100000):
        """
        Args:
            data: Preprocessed DataFrame (normalized OHLCV + indicators)
            initial_balance: Starting capital ($)
        """
        super(TradingEnv, self).__init__()
        
        # Data parameters
        self.data = data.copy()
        self.n_step = 0
        self.max_steps = len(data) - 1
        
        # Trading parameters
        self.initial_balance = float(initial_balance)
        self.current_balance = float(initial_balance)
        self.position = 0  # -1=short, 0=neutral, 1=long
        self.shares_held = 0.0
        self.transaction_cost = 0.001  # 0.1% transaction cost
        self.max_position_size = 0.2  # Maximum 20% of balance per position
        self.max_leverage = 2.0  # Maximum 2x leverage
        
        # Risk management
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        self.entry_price = 0.0
        
        # Track performance
        self.max_portfolio_value = float(initial_balance)
        self.min_portfolio_value = float(initial_balance)
        self.initial_portfolio_value = float(initial_balance)
        
        # Observation space: OHLCV window + position (paper Section 3.4.1)
        self.window_size = 10  # τ=10 days (paper's value)
        n_features = len(data.columns)  # OHLCV + indicators
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_features * self.window_size + 1,)  # Features*window + position
        )
        
        # Action space: 3 choices (paper Section 3.4.2)
        self.action_space = spaces.Discrete(3)  # 0=Long, 1=Short, 2=Hold
        
        # For logging and debugging
        self.trade_history = []

    def reset(self):
        """Reset environment to initial state"""
        self.n_step = self.window_size  # Start after initial window
        self.current_balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.position = 0
        self.entry_price = 0.0
        self.max_portfolio_value = float(self.initial_balance)
        self.min_portfolio_value = float(self.initial_balance)
        self.trade_history = []
        return self._get_observation()

    def _get_observation(self):
        """Get current observation (feature window + position)"""
        # Verify we have enough data for the window
        if self.n_step < self.window_size:
            # We should never get here if coded correctly, but just in case
            print(f"Warning: Insufficient data for window at step {self.n_step}", flush=True)
            # Return zeros with correct shape
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
            
        try:
            # Get the window of data
            window_data = self.data.iloc[self.n_step - self.window_size:self.n_step].values
            
            # Ensure window data is valid
            if window_data.shape[0] != self.window_size:
                print(f"Warning: Invalid window shape {window_data.shape}", flush=True)
                # Return zeros as fallback
                return np.zeros(self.observation_space.shape[0], dtype=np.float32)
            
            # Flatten the window and append position
            obs = np.append(window_data.flatten(), self.position)
            return obs.astype(np.float32)
        except Exception as e:
            print(f"Error in _get_observation: {e}", flush=True)
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def _check_stop_loss_take_profit(self, current_price):
        """Check if stop loss or take profit has been hit"""
        if self.shares_held == 0 or self.entry_price == 0:
            return False
            
        if self.shares_held > 0:  # Long position
            loss_pct = (self.entry_price - current_price) / self.entry_price
            profit_pct = (current_price - self.entry_price) / self.entry_price
        else:  # Short position
            loss_pct = (current_price - self.entry_price) / self.entry_price
            profit_pct = (self.entry_price - current_price) / self.entry_price
            
        return loss_pct >= self.stop_loss_pct or profit_pct >= self.take_profit_pct

    def _close_position(self, current_price):
        """Close current position"""
        if self.shares_held > 0:
            sell_value = self.shares_held * current_price
            self.current_balance += sell_value * (1 - self.transaction_cost)
        elif self.shares_held < 0:
            buy_value = abs(self.shares_held) * current_price
            self.current_balance -= buy_value * (1 + self.transaction_cost)
            
        self.shares_held = 0.0
        self.position = 0
        self.entry_price = 0.0

    def _execute_trade(self, action):
        """Execute trade based on action"""
        try:
            current_price = float(self.data.iloc[self.n_step]['Close'])
            
            # Safety check - ensure price is valid
            if current_price <= 0:
                print(f"Warning: Invalid price {current_price}, skipping trade", flush=True)
                return
            
            prev_position = self.position
            prev_shares = self.shares_held
            
            # Check stop loss / take profit
            if self._check_stop_loss_take_profit(current_price):
                self._close_position(current_price)
                return
            
            # Calculate position size (Kelly Criterion inspired)
            volatility = float(self.data.iloc[self.n_step]['BBB_20_2.0'])  # Using Bollinger Band width as volatility
            position_size = min(
                self.max_position_size,
                1 / (volatility * 10) if volatility > 0 else self.max_position_size
            )
            
            # Calculate maximum shares based on position size and leverage
            max_position_value = self.current_balance * position_size * self.max_leverage
            max_shares = max_position_value / (current_price * (1 + self.transaction_cost))
            
            if action == 0:  # Long
                if self.shares_held < 0:  # Close short first
                    self._close_position(current_price)
                
                if self.shares_held == 0:
                    shares_to_buy = max_shares * 0.95  # Use 95% of max to account for price movements
                    if shares_to_buy >= 0.01:
                        self.shares_held = shares_to_buy
                        cost = self.shares_held * current_price * (1 + self.transaction_cost)
                        self.current_balance -= cost
                        self.entry_price = current_price
                    
            elif action == 1:  # Short
                if self.shares_held > 0:  # Close long first
                    self._close_position(current_price)
                
                if self.shares_held == 0:
                    shares_to_short = max_shares * 0.95
                    if shares_to_short >= 0.01:
                        self.shares_held = -shares_to_short
                        sell_value = abs(self.shares_held) * current_price
                        cost = sell_value * self.transaction_cost
                        self.current_balance += (sell_value - cost)
                        self.entry_price = current_price
            
            # action == 2 is Hold, do nothing
            
            # Update position state (-1/0/1)
            self.position = np.sign(self.shares_held)
            
            # Log position change
            if prev_position != self.position:
                print(f"Position changed from {prev_position} to {self.position} at step {self.n_step}", flush=True)
                
        except Exception as e:
            print(f"Error in _execute_trade: {e}", flush=True)
            # Keep previous position to be safe
            self.shares_held = prev_shares
            self.position = prev_position

    def step(self, action):
        """Execute one time step"""
        if self.n_step >= self.max_steps:
            # Return terminal state if we're already done
            return self._get_observation(), 0.0, True, {
                "portfolio_value": self.current_balance,
                "balance": self.current_balance,
                "shares": 0.0,
                "position": 0,
                "max_value": self.max_portfolio_value,
                "min_value": self.min_portfolio_value,
                "return": (self.current_balance / self.initial_portfolio_value - 1) * 100
            }
        
        try:
            # Store the portfolio value before trade
            prev_price = float(self.data.iloc[self.n_step]['Close'])
            prev_value = max(0.01, self.current_balance + self.shares_held * prev_price)
            
            # Execute trade
            self._execute_trade(action)
            
            # Move to the next step
            self.n_step += 1
            done = self.n_step >= self.max_steps
            
            # Close positions at end of episode
            if done and self.shares_held != 0:
                self._close_position(float(self.data.iloc[self.n_step-1]['Close']))
            
            # Calculate new portfolio value
            if done and self.shares_held != 0:
                # Close all positions at the end
                current_price = float(self.data.iloc[self.n_step-1]['Close'])
                if self.shares_held > 0:
                    self.current_balance += self.shares_held * current_price * (1 - self.transaction_cost)
                else:  # Short position
                    self.current_balance += abs(self.shares_held) * current_price * (1 - self.transaction_cost)
                self.shares_held = 0.0
                self.position = 0
            
            current_price = float(self.data.iloc[min(self.n_step, self.max_steps-1)]['Close'])
            current_value = max(0.01, self.current_balance + self.shares_held * current_price)
            
            # Update max/min portfolio values
            self.max_portfolio_value = max(self.max_portfolio_value, current_value)
            self.min_portfolio_value = min(self.min_portfolio_value, current_value)
            
            # Calculate reward using log returns for better numerical stability
            returns = np.log(current_value / prev_value)
            volatility = float(self.data.iloc[self.n_step-1]['BBB_20_2.0'])
            reward = returns / (volatility + 1e-6)  # Add small constant to avoid division by zero
            reward = np.clip(reward, -1, 1)  # Clip reward for stability
            
            # Get next observation
            next_obs = self._get_observation()
            
            # Create info dictionary
            info = {
                "portfolio_value": current_value,
                "balance": self.current_balance,
                "shares": self.shares_held,
                "position": self.position,
                "max_value": self.max_portfolio_value,
                "min_value": self.min_portfolio_value,
                "return": (current_value / self.initial_portfolio_value - 1) * 100
            }
            
            return next_obs, reward, done, info
                
        except Exception as e:
            print(f"Error in step: {e}", flush=True)
            # Return safe terminal state if error
            return self._get_observation(), 0.0, True, {
                "portfolio_value": self.current_balance,
                "balance": self.current_balance,
                "shares": 0.0,
                "position": 0,
                "max_value": self.max_portfolio_value,
                "min_value": self.min_portfolio_value,
                "return": (self.current_balance / self.initial_portfolio_value - 1) * 100
            }

    def render(self, mode='human'):
        """Optional rendering"""
        current_price = float(self.data.iloc[min(self.n_step, self.max_steps-1)]['Close'])
        portfolio_value = self.current_balance + self.shares_held * current_price
        position_value = self.shares_held * current_price
        
        position_type = "NEUTRAL"
        if self.position > 0:
            position_type = "LONG"
        elif self.position < 0:
            position_type = "SHORT"
            
        print(f"\nStep: {self.n_step}/{self.max_steps}", flush=True)
        print(f"Date: {self.data.index[min(self.n_step, self.max_steps-1)]}", flush=True)
        print(f"Price: ${current_price:.2f}", flush=True)
        print(f"Balance: ${self.current_balance:.2f}", flush=True)
        print(f"Shares: {abs(self.shares_held):.4f} ({position_type})", flush=True)
        print(f"Position Value: ${abs(position_value):.2f}", flush=True)
        print(f"Portfolio Value: ${portfolio_value:.2f}", flush=True)
        print(f"Return: {(portfolio_value/self.initial_portfolio_value - 1)*100:.2f}%", flush=True)
        print("-" * 50, flush=True)