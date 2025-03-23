# NULL-v1: Trading AI with Deep Reinforcement Learning

NULL-v1 is an advanced trading system that leverages Deep Q-Network (DQN) reinforcement learning to develop profitable trading strategies for financial markets.

## Project Overview

This project implements a sophisticated trading agent that learns optimal trading strategies through interaction with a custom trading environment. The system uses a Deep Q-Network architecture with experience replay to learn complex patterns in market data.

## Key Features

- **Deep Reinforcement Learning**: Uses a DQN architecture to make trading decisions
- **Custom Trading Environment**: Simulates realistic market conditions with transaction costs
- **Interactive Dashboard**: Visualize performance metrics and trading decisions
- **Risk Management**: Implements stop-loss, position sizing, and portfolio balancing
- **Technical Indicators**: Leverages common financial indicators (SMA, RSI, MACD, etc.)
- **Performance Analytics**: Calculates Sharpe ratio, drawdown, returns, and other metrics

## Project Structure

```
Null-llm/
├── Project/
│   ├── data/
│   │   ├── raw/             # Raw market data
│   │   └── processed/       # Preprocessed and normalized data
│   ├── models/              # Saved model weights and hyperparameters
│   ├── src/
│   │   ├── model.py         # DQN agent implementation
│   │   ├── trading_env.py   # Custom trading environment
│   │   ├── train.py         # Training script
│   │   ├── evaluate.py      # Evaluation tools
│   │   ├── data_preprocessing.py # Data preparation utilities
│   │   └── evaluation_dashboard.py # Streamlit dashboard
│   └── dashboard.py         # Simplified dashboard launcher
└── requirements.txt         # Project dependencies
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Het-Sathwara/Null-llm.git
cd Null-llm
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Environment setup**:
Make sure you have Python 3.8+ and TensorFlow 2.4+ installed.

## Usage

### Training the Model

Run the training script to train a new model:

```bash
python Project/src/train.py
```

You can configure training parameters by editing the training script or providing command-line arguments.

### Evaluating Performance

Evaluate the trained model on test data:

```bash
python Project/src/evaluate.py
```

### Running the Dashboard

Start the interactive dashboard to visualize results:

```bash
python Project/dashboard.py
```

Or use the more detailed evaluation dashboard:

```bash
streamlit run Project/src/evaluation_dashboard.py
```

## Model Architecture

- **State Space**: Market data features including price, volume, and technical indicators
- **Action Space**: Three possible actions (Long, Short, Hold)
- **Neural Network**: 
  - Input Layer: State size (varies based on features)
  - Hidden Layer 1: 128 neurons with BatchNorm and LeakyReLU
  - Hidden Layer 2: 64 neurons with BatchNorm and LeakyReLU
  - Output Layer: Action size (3)
- **Experience Replay**: Buffer size of 10,000 with prioritized sampling
- **Training Algorithm**: Double DQN with target network updates

## Performance Metrics

The model is evaluated using the following metrics:
- Sharpe Ratio (risk-adjusted return)
- Maximum Drawdown
- Annualized Return
- Total Return
- Win Rate
- Trade Distribution

## Customization

You can customize the trading agent by modifying:
- The features used in the state representation
- The neural network architecture
- The reward function
- Risk management parameters
