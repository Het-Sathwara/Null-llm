# Deep Q-Learning Trading Bot

A Deep Q-Network (DQN) based trading bot that learns to trade AAPL stock using reinforcement learning.

## Project Structure

```
Project/
├── data/
│   ├── raw/             # Raw stock data with indicators
│   └── processed/       # Normalized training/testing data
├── src/
│   ├── model.py         # DQN agent implementation
│   ├── trading_env.py   # Custom trading environment
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation metrics and visualization
│   ├── data_preprocessing.py # Data preparation script
│   └── evaluation_dashboard.py # Streamlit dashboard
└── requirements.txt     # Project dependencies
```

## Features

- Custom OpenAI Gym trading environment
- Deep Q-Network with experience replay and target network
- Risk management with stop-loss and take-profit
- Position sizing based on volatility
- Technical indicators (SMA, RSI, MACD, Bollinger Bands)
- Interactive Streamlit dashboard for performance monitoring

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Het-Sathwara/Null-llm
cd Null-llm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare the data:
```bash
python Project/src/data_preprocessing.py
```

2. Train the model:
```bash
python Project/src/train.py
```

3. Launch the evaluation dashboard:
```bash
streamlit run Project/src/evaluation_dashboard.py
```

## Dashboard Features

The Streamlit dashboard provides real-time monitoring and analysis:

```python
# Example dashboard code
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Trading Bot Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar controls
st.sidebar.title("Trading Controls")
ticker = st.sidebar.selectbox("Select Asset", ["AAPL", "GOOGL", "MSFT"])
initial_balance = st.sidebar.number_input("Initial Balance", 10000, 1000000, 100000)
risk_level = st.sidebar.slider("Risk Level", 0.0, 1.0, 0.5)

# Main dashboard
st.title("Trading Bot Performance")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio Value", "$120,500", "+20.5%")
col2.metric("Current Position", "2.5 AAPL", "+0.5")
col3.metric("Last Trade", "BUY @ $150.25", "")
col4.metric("P&L", "+$2,500", "+2.5%")

# Candlestick chart
st.subheader("Price Chart")
# Add your plotly candlestick chart here

# Trade history
st.subheader("Trade History")
trade_history = pd.DataFrame({
    "Timestamp": ["2024-03-23 10:30:00", "2024-03-23 11:15:00"],
    "Action": ["BUY", "SELL"],
    "Price": [150.25, 152.50],
    "Quantity": [10, 10],
    "P&L": [0, 225]
})
st.dataframe(trade_history)
```

## Training Parameters

- Episodes: 100
- Initial Balance: $100,000
- Batch Size: 32
- Buffer Size: 50,000
- Learning Rate: 0.0001
- Gamma (Discount Factor): 0.95
- Epsilon Decay: 0.99

## Risk Management

- Maximum Position Size: 20% of balance
- Maximum Leverage: 2x
- Stop Loss: 2%
- Take Profit: 4%
- Transaction Cost: 0.1%

## Model Architecture

- Input: 151 features (15 indicators × 10 time steps + position)
- Hidden Layer 1: 128 units with BatchNorm and LeakyReLU
- Hidden Layer 2: 64 units with BatchNorm and LeakyReLU
- Output: 3 units (Long, Short, Hold)

## Dashboard Customization

You can customize the dashboard by modifying `evaluation_dashboard.py`:

1. Update metrics:
```python
# Add custom metrics
st.sidebar.markdown("### Performance Metrics")
st.sidebar.metric("Sharpe Ratio", "1.85", "+0.15")
st.sidebar.metric("Max Drawdown", "-12.5%", "")
```

2. Add technical indicators:
```python
# Plot technical indicators
def plot_indicators(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
    return fig

st.plotly_chart(plot_indicators(data))
```

3. Configure auto-trading:
```python
# Auto-trading settings
auto_trade = st.sidebar.checkbox("Enable Auto-Trading")
update_interval = st.sidebar.number_input("Update Interval (seconds)", 1, 60, 5)
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
