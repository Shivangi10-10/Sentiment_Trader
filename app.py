import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
from sentiment_engine import SentimentEngine
from data_sources import DataSources
from trading_agent import TradingAgent
from utils import Utils
from backtesting_engine import BacktestingEngine

# Page configuration
st.set_page_config(
    page_title="Sentiment Sage - AI Trading Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = pd.DataFrame()
if 'trading_history' not in st.session_state:
    st.session_state.trading_history = pd.DataFrame()
if 'portfolio_balance' not in st.session_state:
    st.session_state.portfolio_balance = {"APT": 0.0, "USDT": 1000.0}  # Starting with 1000 USDT
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# Initialize components
@st.cache_resource
def initialize_components():
    sentiment_engine = SentimentEngine()
    data_sources = DataSources()
    trading_agent = TradingAgent()
    backtesting_engine = BacktestingEngine()
    return sentiment_engine, data_sources, trading_agent, backtesting_engine

sentiment_engine, data_sources, trading_agent, backtesting_engine = initialize_components()

# Sidebar
st.sidebar.title("🧠 Sentiment Sage")
st.sidebar.markdown("AI-Powered Crypto Trading Agent")

# Trading settings
st.sidebar.subheader("Trading Settings")
buy_threshold = st.sidebar.slider("Buy Threshold", 0.0, 1.0, 0.7, 0.05)
sell_threshold = st.sidebar.slider("Sell Threshold", 0.0, 1.0, 0.3, 0.05)
trading_enabled = st.sidebar.checkbox("Enable Auto Trading", value=True)

# Advanced features
st.sidebar.subheader("Advanced Features")
risk_level = st.sidebar.selectbox("Risk Level", ["Conservative", "Moderate", "Aggressive"], index=1)
stop_loss_enabled = st.sidebar.checkbox("Enable Stop Loss", value=True)
stop_loss_percentage = st.sidebar.slider("Stop Loss %", 1.0, 10.0, 5.0, 0.5)

# Multi-token support
st.sidebar.subheader("Multi-Token Portfolio")
tokens_to_track = st.sidebar.multiselect(
    "Tokens to Track", 
    ["APT", "BTC", "ETH", "USDT"], 
    default=["APT"]
)

# Sentiment filtering
st.sidebar.subheader("Sentiment Filtering")
fud_detection = st.sidebar.checkbox("Enable FUD Detection", value=True)
min_sentiment_confidence = st.sidebar.slider("Min Sentiment Confidence", 0.1, 0.9, 0.3, 0.1)

# Data refresh controls
st.sidebar.subheader("Data Controls")
if st.sidebar.button("Refresh Data Now"):
    with st.spinner("Fetching latest data..."):
        # Fetch latest news and social media data
        news_data = data_sources.fetch_crypto_news()
        social_data = data_sources.fetch_social_media_data()
        
        # Analyze sentiment with FUD detection
        all_data = news_data + social_data
        sentiment_scores = []
        
        for item in all_data:
            # Analyze sentiment
            score = sentiment_engine.analyze_sentiment(item['text'])
            
            # FUD detection if enabled
            fud_result = None
            if fud_detection:
                fud_result = sentiment_engine.detect_fud(item['text'])
                # Adjust sentiment if suspicious content detected
                if fud_result['is_suspicious']:
                    score = max(0.2, score * 0.7)  # Reduce sentiment for suspicious content
            
            # Only include if sentiment confidence is above minimum threshold
            if abs(score - 0.5) >= (min_sentiment_confidence - 0.5):
                sentiment_scores.append({
                    'timestamp': item['timestamp'],
                    'text': item['text'][:100] + "..." if len(item['text']) > 100 else item['text'],
                    'source': item['source'],
                    'sentiment_score': score,
                    'token': item.get('token', 'APT'),
                    'fud_detected': fud_result['is_suspicious'] if fud_result else False,
                    'manipulation_score': fud_result['manipulation_score'] if fud_result else 0.0
                })
        
        # Update session state
        new_data = pd.DataFrame(sentiment_scores)
        if not new_data.empty:
            st.session_state.sentiment_data = pd.concat([st.session_state.sentiment_data, new_data], ignore_index=True)
            st.session_state.sentiment_data = st.session_state.sentiment_data.tail(1000)  # Keep last 1000 records
            st.session_state.last_update = datetime.now()
            
            # Execute trading logic if enabled
            if trading_enabled and not new_data.empty:
                latest_sentiment = trading_agent.calculate_rolling_sentiment(st.session_state.sentiment_data)
                trade_decision = trading_agent.make_trading_decision(
                    latest_sentiment, buy_threshold, sell_threshold
                )
                
                if trade_decision['action'] != 'HOLD':
                    # Execute trade
                    trade_result = trading_agent.execute_trade(
                        trade_decision, st.session_state.portfolio_balance
                    )
                    
                    if trade_result['success']:
                        # Update portfolio
                        st.session_state.portfolio_balance = trade_result['new_balance']
                        
                        # Add to trading history
                        trade_record = {
                            'timestamp': datetime.now(),
                            'action': trade_decision['action'],
                            'token': trade_decision['token'],
                            'amount': trade_result['amount'],
                            'price': trade_result['price'],
                            'sentiment_score': latest_sentiment['APT'],
                            'reason': trade_decision['reason']
                        }
                        
                        new_trade = pd.DataFrame([trade_record])
                        st.session_state.trading_history = pd.concat([st.session_state.trading_history, new_trade], ignore_index=True)
                        
                        st.success(f"✅ {trade_decision['action']} executed: {trade_result['amount']:.4f} {trade_decision['token']}")
    
    st.rerun()

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
if auto_refresh:
    time.sleep(30)
    st.rerun()

# Main content
st.title("🧠 Sentiment Sage - AI Trading Agent")
st.markdown("**Real-time sentiment analysis and automated trading for Aptos ecosystem**")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Live Trading", "Backtesting", "Advanced Analytics"])

with tab1:
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Trading Status", "🟢 Active" if trading_enabled else "🔴 Disabled")
    
    with col2:
        if st.session_state.last_update:
            time_diff = datetime.now() - st.session_state.last_update
            st.metric("Last Update", f"{time_diff.seconds//60}m ago")
        else:
            st.metric("Last Update", "Never")
    
    with col3:
        st.metric("Data Points", len(st.session_state.sentiment_data))
    
    with col4:
        st.metric("Total Trades", len(st.session_state.trading_history))

# Portfolio overview
st.subheader("💰 Portfolio Balance")
portfolio_cols = st.columns(3)

with portfolio_cols[0]:
    st.metric("APT Balance", f"{st.session_state.portfolio_balance['APT']:.4f}")

with portfolio_cols[1]:
    st.metric("USDT Balance", f"{st.session_state.portfolio_balance['USDT']:.2f}")

with portfolio_cols[2]:
    # Calculate total portfolio value (assuming APT price)
    apt_price = 8.50  # Mock current APT price
    total_value = (st.session_state.portfolio_balance['APT'] * apt_price + 
                   st.session_state.portfolio_balance['USDT'])
    st.metric("Total Value (USDT)", f"{total_value:.2f}")

# Sentiment analysis section
st.subheader("📊 Real-time Sentiment Analysis")

if not st.session_state.sentiment_data.empty:
    # Calculate rolling sentiment
    rolling_sentiment = trading_agent.calculate_rolling_sentiment(st.session_state.sentiment_data)
    
    # Current sentiment display
    sentiment_display_cols = st.columns(2)
    
    with sentiment_display_cols[0]:
        apt_sentiment = rolling_sentiment.get('APT', 0.5)
        sentiment_color = "🟢" if apt_sentiment > 0.6 else "🟡" if apt_sentiment > 0.4 else "🔴"
        st.metric("APT Sentiment", f"{sentiment_color} {apt_sentiment:.3f}")
        
        # Sentiment progress bar
        st.progress(apt_sentiment, text=f"APT Sentiment: {apt_sentiment:.3f}")
        
        # Color-coded sentiment display
        if apt_sentiment > 0.7:
            st.success(f"Strong Positive Sentiment: {apt_sentiment:.3f}")
        elif apt_sentiment > 0.5:
            st.info(f"Positive Sentiment: {apt_sentiment:.3f}")
        elif apt_sentiment > 0.3:
            st.warning(f"Negative Sentiment: {apt_sentiment:.3f}")
        else:
            st.error(f"Strong Negative Sentiment: {apt_sentiment:.3f}")
    
    with sentiment_display_cols[1]:
        usdt_sentiment = rolling_sentiment.get('USDT', 0.5)
        sentiment_color = "🟢" if usdt_sentiment > 0.6 else "🟡" if usdt_sentiment > 0.4 else "🔴"
        st.metric("USDT Sentiment", f"{sentiment_color} {usdt_sentiment:.3f}")
        
        # Trading signal indicator
        latest_sentiment_score = rolling_sentiment.get('APT', 0.5)
        if latest_sentiment_score > buy_threshold:
            signal = "🔥 STRONG BUY"
            signal_color = "green"
        elif latest_sentiment_score < sell_threshold:
            signal = "❄️ STRONG SELL"
            signal_color = "red"
        else:
            signal = "⚖️ HOLD"
            signal_color = "orange"
        
        st.markdown(f"### Current Signal: <span style='color:{signal_color}'>{signal}</span>", unsafe_allow_html=True)
        st.markdown(f"**Buy Threshold:** {buy_threshold}")
        st.markdown(f"**Sell Threshold:** {sell_threshold}")
        st.markdown(f"**Current Score:** {latest_sentiment_score:.3f}")
    
    # Sentiment trend chart
    st.subheader("📈 Sentiment Trends")
    
    # Prepare data for plotting
    df_plot = st.session_state.sentiment_data.copy()
    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
    df_plot = df_plot.sort_values('timestamp')
    
    # Show recent sentiment data in table format
    if len(df_plot) > 0:
        # Calculate rolling average for each token
        for token in df_plot['token'].unique():
            token_data = df_plot[df_plot['token'] == token].copy()
            if len(token_data) > 1:
                token_data = token_data.sort_values('timestamp')
                # Use pandas rolling average
                token_data['rolling_sentiment'] = token_data['sentiment_score'].rolling(window=5, min_periods=1).mean()
                
                st.write(f"**{token} Recent Sentiment Trend:**")
                
                # Show last 10 data points
                recent_data = token_data.tail(10)[['timestamp', 'sentiment_score', 'rolling_sentiment']]
                recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%H:%M:%S')
                recent_data['sentiment_score'] = recent_data['sentiment_score'].round(3)
                recent_data['rolling_sentiment'] = recent_data['rolling_sentiment'].round(3)
                
                st.dataframe(recent_data, use_container_width=True)
                
                # Simple line chart using Streamlit's built-in charting
                chart_data = token_data.set_index('timestamp')[['sentiment_score', 'rolling_sentiment']]
                st.line_chart(chart_data)
    else:
        st.info("No sentiment trend data available yet.")

else:
    st.info("No sentiment data available. Click 'Refresh Data Now' to fetch the latest information.")

# Trading history
st.subheader("📋 Trading History")

if not st.session_state.trading_history.empty:
    # Display recent trades
    trades_df = st.session_state.trading_history.copy()
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df = trades_df.sort_values('timestamp', ascending=False)
    
    # Format for display
    display_trades = trades_df[['timestamp', 'action', 'token', 'amount', 'price', 'sentiment_score', 'reason']].copy()
    display_trades['timestamp'] = display_trades['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_trades['amount'] = display_trades['amount'].round(4)
    display_trades['price'] = display_trades['price'].round(2)
    display_trades['sentiment_score'] = display_trades['sentiment_score'].round(3)
    
    st.dataframe(display_trades, use_container_width=True)
    
    # Trading performance chart
    if len(trades_df) > 1:
        st.subheader("📊 Portfolio Performance")
        
        # Calculate cumulative performance
        trades_df['cumulative_value'] = 0.0
        running_balance = {"APT": 0.0, "USDT": 1000.0}
        
        for idx, trade in trades_df.iterrows():
            if trade['action'] == 'BUY':
                running_balance['USDT'] -= float(trade['amount']) * float(trade['price'])
                running_balance['APT'] += float(trade['amount'])
            elif trade['action'] == 'SELL':
                running_balance['USDT'] += float(trade['amount']) * float(trade['price'])
                running_balance['APT'] -= float(trade['amount'])
            
            # Calculate total value
            total_value = running_balance['APT'] * float(trade['price']) + running_balance['USDT']
            trades_df.at[idx, 'cumulative_value'] = total_value
        
        # Use Streamlit's built-in line chart
        performance_data = trades_df.sort_values('timestamp').set_index('timestamp')[['cumulative_value']]
        st.line_chart(performance_data)

else:
    st.info("No trading history available. Trading decisions will appear here once auto-trading is enabled.")

# Recent sentiment data
st.subheader("📰 Recent Sentiment Data")

if not st.session_state.sentiment_data.empty:
    recent_data = st.session_state.sentiment_data.tail(10).copy()
    recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp']).dt.strftime('%H:%M:%S')
    recent_data['sentiment_score'] = recent_data['sentiment_score'].round(3)
    
    st.dataframe(
        recent_data[['timestamp', 'source', 'text', 'sentiment_score', 'token']], 
        use_container_width=True
    )
else:
    st.info("No recent sentiment data available.")

# Footer
st.markdown("---")
st.markdown("**Sentiment Sage** - Built for Aptos Thunderdome Hackathon | Powered by HuggingFace Transformers")
