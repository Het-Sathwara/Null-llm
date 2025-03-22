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
│   └── train.py         # Training script
├── data_preprocessing.py # Data preparation script
└── requirements.txt     # Project dependencies
```

## Features

- Custom OpenAI Gym trading environment
- Deep Q-Network with experience replay and target network
- Risk management with stop-loss and take-profit
- Position sizing based on volatility
- Technical indicators (SMA, RSI, MACD, Bollinger Bands)

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare the data:
```bash
python Project/data_preprocessing.py
```

2. Train the model:
```bash
python Project/src/train.py
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

## Results

Training results, including portfolio value, returns, and training metrics, are saved in:
- `Project/models/`: Model checkpoints and weights
- `Project/plots/`: Training progress plots
- `Project/logs/`: Training logs 