#!/usr/bin/env python3
"""
Dashboard wrapper script for the trading model evaluation dashboard.
This version uses mock data to avoid model loading issues.
"""
import os
import sys
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def generate_mock_data():
    """Generate sample trading data for demonstration"""
    # Create dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(200)]
    
    # Create price data with realistic trend + noise
    np.random.seed(42)  # For reproducibility
    trend = np.linspace(0, 20, 200) 
    noise = np.random.normal(0, 2, 200)
    market_price = 100 + trend + noise.cumsum()
    
    # Generate portfolio performance (slightly better than market)
    alpha = 0.03  # 3% alpha
    portfolio_value = 100000 * (1 + np.cumsum(np.random.normal(0.001 + alpha/200, 0.012, 200)))
    
    # Generate trading actions (0=Long, 1=Short, 2=Hold)
    actions = np.random.choice([0, 1, 2], size=200, p=[0.3, 0.2, 0.5])
    
    # Create dataframe
    df = pd.DataFrame({
        'Date': dates,
        'Portfolio_Value': portfolio_value,
        'Market_Price': market_price,
        'Action': actions
    })
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Calculate performance metrics
    returns = df['Portfolio_Value'].pct_change().dropna()
    market_returns = df['Market_Price'].pct_change().dropna()
    
    metrics = {
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
        'max_drawdown': (df['Portfolio_Value'].cummax() - df['Portfolio_Value']).max() / df['Portfolio_Value'].cummax().max(),
        'annualized_return': (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) ** (252/len(df)) - 1,
        'total_return': (df['Portfolio_Value'].iloc[-1] - df['Portfolio_Value'].iloc[0]) / df['Portfolio_Value'].iloc[0],
        'win_rate': 0.58,  # Made up for demo
        'market_correlation': returns.corr(market_returns)
    }
    
    return df, metrics

def main():
    """Display the dashboard"""
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
    
    st.title("Trading Model Evaluation Dashboard")
    st.caption("Demo version - using mock data")
    
    # Generate mock data
    results_df, metrics = generate_mock_data()
    
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
        st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Total Return</h4>
                <p>{metrics['total_return']*100:.1f}%</p>
                <h4>🔀 Win Rate</h4>
                <p>{metrics['win_rate']*100:.1f}%</p>
                <h4>🔄 Total Trades</h4>
                <p>{len(results_df[results_df['Action'] != 2])}</p>
            </div>
        """, unsafe_allow_html=True)
    
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
        
        # Market price (scaled to match portfolio)
        scale_factor = results_df['Portfolio_Value'].iloc[0] / results_df['Market_Price'].iloc[0]
        fig.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df['Market_Price'] * scale_factor,
            name='Market (scaled)',
            line=dict(color='#636efa', width=1)
        ))
        
        # Trade markers
        trades = results_df[results_df['Action'] != 2]
        fig.add_trace(go.Scatter(
            x=trades.index,
            y=trades['Portfolio_Value'],
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
                
                # Fix trade counts for pie chart
                labels = ['Long', 'Short', 'Hold']
                values = []
                
                # Get counts for each action type
                for i in range(3):
                    values.append(len(results_df[results_df['Action'] == i]))
                
                fig = px.pie(
                    names=labels,
                    values=values,
                    color_discrete_sequence=['#00ff88', '#ff4b4b', '#636efa']
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)
    
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 