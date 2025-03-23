import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trading_env import TradingEnv
from model import TDQNAgent
import os
from io import BytesIO
import sys

def evaluate_model(agent, test_data, initial_balance=100000):
    """
    Evaluate TDQN agent on test data
    Returns:
        results_df (DataFrame): Detailed trade-by-trade results
        metrics (dict): Key performance metrics
    """
    env = TradingEnv(test_data, initial_balance=initial_balance)
    state = env.reset()
    portfolio_values = []
    actions_history = []
    done = False

    while not done:
        action = agent.act(state)  # Use trained policy (epsilon should be 0)
        next_state, reward, done, info = env.step(action)
        
        # Record step information
        portfolio_values.append(info['portfolio_value'])
        actions_history.append(action)
        state = next_state

    # Calculate metrics
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    max_drawdown = (pd.Series(portfolio_values).cummax() - pd.Series(portfolio_values)).max()
    annualized_return = (portfolio_values[-1] / portfolio_values[0]) ** (252/len(portfolio_values)) - 1

    # Create results DataFrame
    results_df = pd.DataFrame({
        'Date': test_data.index[:len(portfolio_values)],
        'Portfolio_Value': portfolio_values,
        'Market_Price': test_data['Close'].values[:len(portfolio_values)],
        'Action': actions_history
    }).set_index('Date')

    metrics = {
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'annualized_return': annualized_return,
        'total_return': (portfolio_values[-1] - initial_balance) / initial_balance
    }

    return results_df, metrics

def visualize_performance(results_df, metrics, save_path=None, return_image=False):
    """
    Generate detailed performance visualizations
    
    Args:
        results_df: DataFrame with trading results
        metrics: Dictionary of performance metrics
        save_path: Path to save the figure (optional)
        return_image: If True, return the figure as bytes (for Streamlit)
    
    Returns:
        bytes if return_image is True, otherwise None
    """
    plt.figure(figsize=(15, 10))
    
    # Portfolio vs Market
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(results_df['Market_Price'], label='Market Price', alpha=0.7)
    ax1.plot(results_df['Portfolio_Value'], label='Portfolio Value', color='orange')
    ax1.set_title('Portfolio Value vs Market Performance')
    ax1.legend()
    
    # Drawdown
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    maxs = results_df['Portfolio_Value'].cummax()
    drawdown = (maxs - results_df['Portfolio_Value']) / maxs
    ax2.fill_between(results_df.index, drawdown, color='red', alpha=0.3)
    ax2.set_title('Portfolio Drawdown')
    
    # Actions
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    actions = results_df['Action'].replace({0: 'Long', 1: 'Short', 2: 'Hold'})
    action_counts = actions.value_counts()
    ax3.pie(action_counts, labels=action_counts.index, autopct='%1.1f%%')
    ax3.set_title('Trade Distribution')
    
    # Metrics
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    metric_text = "\n".join([f"{k.replace('_', ' ').title()}: {v:.2f}" 
                           if isinstance(v, float) else f"{k.replace('_', ' ').title()}: {v}"
                           for k, v in metrics.items()])
    ax4.text(0.5, 0.5, metric_text, ha='center', va='center', fontsize=12)
    ax4.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if return_image:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return buf.getvalue()
    
    plt.show()
    return None

if __name__ == "__main__":
    # Get the project directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    # Load trained model
    agent = TDQNAgent(state_size=15, action_size=3)  # Match training config
    agent.load(os.path.join(project_dir, "models/best_model.weights.h5"))
    
    # Load test data
    test_data_path = os.path.join(project_dir, "data/processed/test_scaled.csv")
    if not os.path.exists(test_data_path):
        print(f"Error: Test data not found at {test_data_path}")
        sys.exit(1)
        
    test_data = pd.read_csv(test_data_path, index_col='Date', parse_dates=True)
    
    # Evaluate
    results_df, metrics = evaluate_model(agent, test_data)
    
    # Visualize
    plot_path = os.path.join(project_dir, "plots")
    os.makedirs(plot_path, exist_ok=True)
    
    visualize_performance(
        results_df, 
        metrics,
        save_path=os.path.join(plot_path, "test_performance.png")
    )
    
    print("\nPerformance Metrics:")
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title()}: {v:.2f}")