# evaluation_dashboard.py
import os
import sys

# Add the project directory to Python path to fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from src.evaluate import evaluate_model, visualize_performance
from src.model import TDQNAgent
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
        # Load test data to determine state size
        test_data = pd.read_csv("data/processed/test_scaled.csv", 
                              index_col='Date', parse_dates=True)
        state_size = test_data.shape[1]  # All columns except index
        
        # Initialize and load agent
        agent = TDQNAgent(state_size=state_size, action_size=3)
        agent.load("models/best_model.weights.h5")
        
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
    
    if agent and test_data is not None:
        # Run evaluation
        with st.spinner("🚀 Evaluating model performance..."):
            results_df, metrics = evaluate_model(agent, test_data)
        
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
                    fig = px.pie(
                        names=['Long', 'Short', 'Hold'],
                        values=trade_counts,
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