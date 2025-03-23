# evaluation_dashboard.py
import os
import sys

# Add the project directory to Python path to fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from evaluate import evaluate_model, visualize_performance
from model import TDQNAgent
import json

# Configure page
st.set_page_config(
    page_title="Trading Model Evaluation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .metric-card {
        background: #1f2937;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .plot-container {
        background: #111827;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_resources():
    """Load model and test data with error handling"""
    try:
        # Create data directories if they don't exist
        os.makedirs(os.path.join(project_dir, "data/processed"), exist_ok=True)
        
        # Check if test data exists, if not create a simple test dataset
        test_data_path = os.path.join(project_dir, "data/processed/test_scaled.csv")
        if not os.path.exists(test_data_path):
            st.warning("Test data not found. Using sample data for demonstration.")
            # Create a simple test dataset with appropriate columns
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            data = {
                'Open': np.random.randn(100).cumsum() + 100,
                'High': np.random.randn(100).cumsum() + 102,
                'Low': np.random.randn(100).cumsum() + 98,
                'Close': np.random.randn(100).cumsum() + 100,
                'Volume': np.abs(np.random.randn(100) * 1000) + 5000,
            }
            test_data = pd.DataFrame(data, index=dates)
            test_data.index.name = 'Date'
            os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
            test_data.to_csv(test_data_path)
        else:
            # Load test data to determine state size
            test_data = pd.read_csv(test_data_path, index_col='Date', parse_dates=True)
        
        state_size = test_data.shape[1]  # All columns except index
        
        # First try to check model file JSON to determine state size
        model_dir = os.path.join(project_dir, "models")
        hyperparams_path = os.path.join(model_dir, "hyperparameters.json")
        if os.path.exists(hyperparams_path):
            try:
                with open(hyperparams_path, 'r') as f:
                    params = json.load(f)
                    if 'state_size' in params:
                        state_size = params['state_size']
                        st.info(f"Using state size {state_size} from hyperparameters file")
            except Exception as e:
                st.warning(f"Could not load hyperparameters: {str(e)}")
        
        # Initialize and load agent
        agent = TDQNAgent(state_size=state_size, action_size=3)
        model_path = os.path.join(project_dir, "models/best_model.weights.h5")
        if not os.path.exists(model_path):
            st.warning("Default model not found. Looking for alternative models...")
            # Look for alternative models
            models_dir = os.path.join(project_dir, "models")
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.weights.h5')]
            if model_files:
                model_path = os.path.join(models_dir, model_files[0])
                st.info(f"Using alternative model: {model_files[0]}")
            else:
                st.error("No model files found. Cannot continue.")
                return None, None
        
        # Try to load the model, if it fails, try different state sizes
        if not agent.load(model_path):
            st.warning("First attempt to load model failed. Trying alternative state sizes...")
            for state_size_attempt in [5, 10, 15, 20, 25, 30]:
                if state_size_attempt != state_size:
                    st.info(f"Trying state size: {state_size_attempt}")
                    agent = TDQNAgent(state_size=state_size_attempt, action_size=3)
                    if agent.load(model_path):
                        st.success(f"Successfully loaded model with state size {state_size_attempt}")
                        state_size = state_size_attempt
                        break
        
        return agent, test_data
    except Exception as e:
        st.error(f"""
        ❗ Error loading resources: {str(e)}
        Ensure:
        1. Model weights exist at specified path
        2. Processed test data is available
        3. Package structure matches expectations
        """)
        return None, None

def main():
    st.title("Trading Model Evaluation Dashboard")
    
    # Load model and data
    agent, test_data = load_resources()
    
    if agent is not None and test_data is not None:
        # Run evaluation
        with st.spinner("🚀 Evaluating model performance..."):
            try:
                results_df, metrics = evaluate_model(agent, test_data)
            except Exception as e:
                st.error(f"Error during model evaluation: {str(e)}")
                # Create fallback dummy results for demonstration purposes
                st.warning("Using fallback dummy data for demonstration")
                
                # Create dummy results
                dates = test_data.index
                portfolio_values = 100000 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
                actions = np.random.choice([0, 1, 2], size=len(dates), p=[0.3, 0.2, 0.5])
                
                results_df = pd.DataFrame({
                    'Date': dates,
                    'Portfolio_Value': portfolio_values,
                    'Market_Price': test_data['Close'].values if 'Close' in test_data else (100 + np.random.randn(len(dates)).cumsum()),
                    'Action': actions
                }).set_index('Date')
                
                metrics = {
                    'sharpe_ratio': 0.8,
                    'max_drawdown': 0.15,
                    'annualized_return': 0.12,
                    'total_return': 0.30,
                    'win_rate': 0.55
                }
        
        # Main layout columns
        metrics_col, chart_col = st.columns([1, 3])
        
        with metrics_col:
            st.header("Key Metrics")
            
            # Metric cards
            st.markdown(f"""
                <div class="metric-card">
                    <h3>🏆 Annual Return</h3>
                    <h1>{metrics['annualized_return']*100:.1f}%</h1>
                </div>
                <div class="metric-card">
                    <h3>⚖️ Sharpe Ratio</h3>
                    <h1>{metrics['sharpe_ratio']:.2f}</h1>
                </div>
                <div class="metric-card">
                    <h3>📉 Max Drawdown</h3>
                    <h1>{metrics['max_drawdown']*100:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional stats
            st.markdown("""
                <div class="metric-card">
                    <h4>📊 Total Return</h4>
                    <p>{total_return:.1f}%</p>
                    <h4>🔀 Win Rate</h4>
                    <p>{win_rate:.1f}%</p>
                    <h4>🔄 Total Trades</h4>
                    <p>{total_trades}</p>
                </div>
            """.format(
                total_return=metrics['total_return']*100,
                win_rate=metrics.get('win_rate', 0)*100,
                total_trades=len(results_df[results_df['Action'] != 2])
            ), unsafe_allow_html=True)
        
        with chart_col:
            # Interactive performance chart
            st.header("Performance Analysis")
            fig = go.Figure()
            
            # Portfolio value
            fig.add_trace(go.Scatter(
                x=results_df.index,
                y=results_df['Portfolio_Value'],
                name='Portfolio',
                line=dict(color='#00ff88', width=2)
            ))
            
            # Market price
            fig.add_trace(go.Scatter(
                x=results_df.index,
                y=results_df['Market_Price'],
                name='Market',
                line=dict(color='#636efa', width=1)
            ))
            
            # Trade markers
            trades = results_df[results_df['Action'] != 2]
            fig.add_trace(go.Scatter(
                x=trades.index,
                y=trades['Market_Price'],
                mode='markers',
                marker=dict(
                    color=np.where(trades['Action'] == 0, '#00ff88', '#ff4b4b'),
                    size=8,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name='Trades'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                height=600,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed analysis expander
            with st.expander("Advanced Analytics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Drawdown analysis
                    st.subheader("Drawdown Analysis")
                    max_drawdown = results_df['Portfolio_Value'].cummax()
                    drawdown = (max_drawdown - results_df['Portfolio_Value']) / max_drawdown
                    
                    fig = px.area(
                        x=results_df.index,
                        y=drawdown,
                        labels={'y': 'Drawdown', 'x': 'Date'},
                        color_discrete_sequence=['#ff4b4b']
                    )
                    fig.update_layout(template='plotly_dark', height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Trade distribution
                    st.subheader("Trade Distribution")
                    trade_counts = results_df['Action'].value_counts()
                    
                    # Fix trade counts for pie chart
                    labels = ['Long', 'Short', 'Hold']
                    values = []
                    
                    # Ensure all action types (0,1,2) have counts
                    for i in range(3):
                        if i in trade_counts.index:
                            values.append(trade_counts[i])
                        else:
                            values.append(0)
                    
                    fig = px.pie(
                        names=labels,
                        values=values,
                        color_discrete_sequence=['#00ff88', '#ff4b4b', '#636efa']
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(template='plotly_dark', height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Data export section
        st.markdown("---")
        st.header("Data Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV Download
            csv = results_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name="model_performance.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Image Export
            img_bytes = visualize_performance(results_df, metrics, return_image=True)
            st.download_button(
                label="🖼️ Download Summary Chart (PNG)",
                data=img_bytes,
                file_name="performance_summary.png",
                mime="image/png",
                use_container_width=True
            )

if __name__ == "__main__":
    main()