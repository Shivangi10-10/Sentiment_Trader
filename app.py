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
    page_title="Sentiment Sage - Trading Agent",
    page_icon="📈",
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

# Enhanced Professional Sidebar
with st.sidebar:
    # Sidebar Header with Professional Styling
    st.markdown("""
    <style>
        .sidebar-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            text-align: center;
            color: white;
        }
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #667eea;
        }
        .sidebar-metric {
            background: white;
            padding: 0.8rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active { background-color: #48bb78; }
        .status-inactive { background-color: #e53e3e; }
    </style>
    
    <div class="sidebar-header">
        <h2 style="margin: 0; font-size: 1.4rem;">📈 Sentiment Sage</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">Advanced Trading Control</p>
        <p style="margin: 0.3rem 0 0 0; opacity: 0.8; font-size: 0.8rem;">Made by Shivangi Suyash</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Trading Status Overview
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**📊 System Status**")
    
    # Data source status
    data_status = data_sources.get_data_sources_status()
    api_status = "🟢 Connected" if data_status.get('cryptopanic', False) else "🔴 Disconnected"
    st.markdown(f"**API Status:** {api_status}")
    st.markdown(f"**Data Points:** {len(st.session_state.sentiment_data)}")
    st.markdown(f"**Active Trades:** {len(st.session_state.trading_history)}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Trading Parameters Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**🎯 Trading Parameters**")
    buy_threshold = st.slider("Buy Signal Threshold", 0.0, 1.0, 0.65, 0.05, 
                             help="Sentiment score threshold for buy signals (higher = more conservative)")
    sell_threshold = st.slider("Sell Signal Threshold", 0.0, 1.0, 0.35, 0.05,
                              help="Sentiment score threshold for sell signals (lower = more conservative)")
    
    # Visual threshold indicator
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #e53e3e 0%, #ed8936 {sell_threshold*100}%, #ecc94b {(sell_threshold+buy_threshold)*50}%, #68d391 {buy_threshold*100}%, #38a169 100%); 
                height: 10px; border-radius: 5px; margin: 10px 0;"></div>
    <div style="font-size: 0.8rem; color: #666;">
        Sell Zone ← Neutral Zone → Buy Zone
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk Management Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**🛡️ Risk Management**")
    risk_level = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"],
                             help="Conservative: Small positions, Moderate: Balanced, Aggressive: Large positions")
    
    risk_colors = {"Conservative": "#48bb78", "Moderate": "#ed8936", "Aggressive": "#e53e3e"}
    st.markdown(f'<div style="color: {risk_colors[risk_level]}; font-weight: bold;">Current: {risk_level}</div>', unsafe_allow_html=True)
    
    stop_loss_enabled = st.checkbox("Enable Stop Loss Protection", value=True,
                                   help="Automatically sell positions to limit losses")
    
    if stop_loss_enabled:
        stop_loss_percentage = st.slider("Stop Loss Percentage", 1.0, 20.0, 5.0, 0.5,
                                        help="Maximum loss percentage before auto-sell")
        st.markdown(f'<div style="color: #e53e3e; font-size: 0.9rem;">Max Loss: -{stop_loss_percentage}%</div>', unsafe_allow_html=True)
    else:
        stop_loss_percentage = 0.0
    st.markdown('</div>', unsafe_allow_html=True)
    
    # System Settings Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**🔄 System Settings**")
    auto_refresh = st.checkbox("Auto-Refresh Data", value=False,
                              help="Automatically fetch new data every 30 seconds")
    
    if auto_refresh:
        st.markdown('<div style="color: #48bb78; font-size: 0.9rem;">⏱️ Refreshing every 30 seconds</div>', unsafe_allow_html=True)
    
    # Manual refresh button
    if st.button("🔄 Refresh Data Now", type="secondary", use_container_width=True):
        st.session_state.last_update = datetime.now()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Trading Control Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**🚀 Trading Control**")
    trading_enabled = st.toggle("Enable Automated Trading", value=False,
                               help="Activate sentiment-based automated trading")
    
    status_class = "status-active" if trading_enabled else "status-inactive"
    status_text = "ACTIVE" if trading_enabled else "INACTIVE"
    st.markdown(f'<div><span class="status-indicator {status_class}"></span>Status: {status_text}</div>', unsafe_allow_html=True)
    
    if trading_enabled:
        st.markdown('<div style="color: #48bb78; font-size: 0.9rem;">🤖 Agent is monitoring markets</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color: #e53e3e; font-size: 0.9rem;">⏸️ Trading is paused</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Actions Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**⚡ Quick Actions**")
    
    if st.button("⚠️ Emergency Stop", type="primary", use_container_width=True):
        st.session_state.trading_enabled = False
        st.warning("Trading has been stopped!")
    st.markdown('</div>', unsafe_allow_html=True)

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

# Advanced UI with modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><polygon fill="%23ffffff08" points="0,1000 1000,0 1000,1000"/></svg>');
        pointer-events: none;
    }
    
    .hero-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        color: #f8f9fa;
        font-size: 1.4rem;
        font-weight: 300;
        margin-bottom: 2rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    
    .badge {
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .status-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .status-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #667eea, #764ba2);
    }
    
    .portfolio-card {
        background: linear-gradient(145deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .portfolio-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .sentiment-gauge {
        background: linear-gradient(145deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .gauge-container {
        position: relative;
        width: 200px;
        height: 100px;
        margin: 0 auto 1rem;
    }
    
    .gauge-bg {
        width: 100%;
        height: 100%;
        border-radius: 100px 100px 0 0;
        background: linear-gradient(90deg, #e53e3e 0%, #ed8936 25%, #ecc94b 50%, #68d391 75%, #38a169 100%);
        position: relative;
    }
    
    .gauge-needle {
        position: absolute;
        bottom: 0;
        left: 50%;
        width: 4px;
        height: 80px;
        background: #2d3748;
        border-radius: 2px;
        transform-origin: bottom center;
        transition: transform 0.5s ease;
    }
    
    .chart-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    .info-tooltip {
        background: linear-gradient(145deg, #4299e1 0%, #3182ce 100%);
        color: white;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        border-left: 4px solid #2b6cb0;
    }
    
    .trading-signal {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .signal-buy {
        background: linear-gradient(145deg, #48bb78 0%, #38a169 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);
    }
    
    .signal-sell {
        background: linear-gradient(145deg, #f56565 0%, #e53e3e 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(245, 101, 101, 0.4);
    }
    
    .signal-hold {
        background: linear-gradient(145deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.4);
    }
    
    .tab-container {
        margin: 2rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f7fafc;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        color: #4a5568;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
</style>

<div class="hero-section">
    <h1 class="hero-title">📈 Sentiment Sage</h1>
    <p class="hero-subtitle">Advanced Cryptocurrency Trading Agent by Shivangi Suyash</p>
    <div class="hero-badges">
        <span class="badge">🔴 Live Data</span>
        <span class="badge">📊 Real-time Analysis</span>
        <span class="badge">⚡ Automated Trading</span>
        <span class="badge">🎯 Smart Decisions</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Live Trading", "Backtesting", "Advanced Analytics"])

with tab1:
    # Advanced status dashboard with cards
    st.markdown('<div class="status-card">', unsafe_allow_html=True)
    st.markdown("### 📊 System Dashboard")
    
    # Create modern metric cards
    status_cols = st.columns(4)
    
    with status_cols[0]:
        status_value = "🟢 ACTIVE" if trading_enabled else "🔴 INACTIVE"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Trading Status</div>
            <div class="metric-value" style="color: {'#38a169' if trading_enabled else '#e53e3e'}">{status_value}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="info-tooltip">💡 Automated trading engine status. When active, AI makes buy/sell decisions based on sentiment analysis.</div>', unsafe_allow_html=True)
    
    with status_cols[1]:
        if st.session_state.last_update:
            time_diff = datetime.now() - st.session_state.last_update
            update_text = f"{time_diff.seconds//60}m ago"
        else:
            update_text = "Never"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Last Data Refresh</div>
            <div class="metric-value">{update_text}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="info-tooltip">⏱️ Time elapsed since last cryptocurrency news data was fetched from authentic sources.</div>', unsafe_allow_html=True)
    
    with status_cols[2]:
        data_count = len(st.session_state.sentiment_data)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Analyzed Articles</div>
            <div class="metric-value">{data_count}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="info-tooltip">📰 Total cryptocurrency news articles and social posts processed for sentiment.</div>', unsafe_allow_html=True)
    
    with status_cols[3]:
        trade_count = len(st.session_state.trading_history)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Executed Trades</div>
            <div class="metric-value">{trade_count}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="info-tooltip">⚡ Number of automated buy/sell transactions executed by the AI agent.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Premium Portfolio Card
    st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
    st.markdown("### 💰 Portfolio Overview")
    st.markdown("**Live portfolio valuation powered by real market data**")
    
    # Get live market data
    market_data = data_sources.fetch_market_data()
    apt_price = market_data.get('APT', {}).get('price', 8.50)
    total_value = (st.session_state.portfolio_balance['APT'] * apt_price + 
                   st.session_state.portfolio_balance['USDT'])
    
    portfolio_cols = st.columns(3)
    
    with portfolio_cols[0]:
        st.markdown(f"""
        <div style="text-align: center; color: white; position: relative; z-index: 2;">
            <h3 style="margin: 0; font-size: 1.2rem;">APT Holdings</h3>
            <div style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;">
                {st.session_state.portfolio_balance['APT']:.4f}
            </div>
            <div style="font-size: 0.9rem; opacity: 0.8;">
                ≈ ${st.session_state.portfolio_balance['APT'] * apt_price:.2f} USD
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with portfolio_cols[1]:
        st.markdown(f"""
        <div style="text-align: center; color: white; position: relative; z-index: 2;">
            <h3 style="margin: 0; font-size: 1.2rem;">USDT Balance</h3>
            <div style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;">
                {st.session_state.portfolio_balance['USDT']:.2f}
            </div>
            <div style="font-size: 0.9rem; opacity: 0.8;">
                Available for Trading
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with portfolio_cols[2]:
        price_change = market_data.get('APT', {}).get('change_24h', 0)
        change_color = "#48bb78" if price_change >= 0 else "#f56565"
        change_icon = "📈" if price_change >= 0 else "📉"
        
        st.markdown(f"""
        <div style="text-align: center; color: white; position: relative; z-index: 2;">
            <h3 style="margin: 0; font-size: 1.2rem;">Total Value</h3>
            <div style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;">
                ${total_value:.2f}
            </div>
            <div style="font-size: 0.9rem; opacity: 0.8; color: {change_color};">
                {change_icon} {price_change:+.2f}% (24h)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Live market info tooltip
    st.markdown('<div class="info-tooltip">📊 Portfolio values updated with live market data from CoinGecko API. APT holdings converted to USD using current market price.</div>', unsafe_allow_html=True)

    # Advanced Sentiment Analysis Dashboard
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown("### 📊 AI Sentiment Analysis Engine")
    st.markdown("**Real-time cryptocurrency sentiment powered by advanced natural language processing**")

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

with tab2:
    # Backtesting Section
    st.markdown("### 🔄 Strategy Backtesting")
    st.markdown("**Test different trading strategies against historical sentiment data**")
    st.info("ℹ️ Backtesting allows you to see how well your trading strategy would have performed in the past using real sentiment data.")
    
    if len(st.session_state.sentiment_data) > 10:
        backtest_cols = st.columns(2)
        
        with backtest_cols[0]:
            st.markdown("#### Strategy Parameters")
            bt_buy_threshold = st.slider("Backtest Buy Threshold", 0.0, 1.0, 0.65, 0.05, key="bt_buy")
            bt_sell_threshold = st.slider("Backtest Sell Threshold", 0.0, 1.0, 0.35, 0.05, key="bt_sell")
            bt_risk_level = st.selectbox("Backtest Risk Level", ["Conservative", "Moderate", "Aggressive"], key="bt_risk")
            
        with backtest_cols[1]:
            st.markdown("#### Backtest Settings")
            initial_usdt = st.number_input("Initial USDT", min_value=100.0, max_value=10000.0, value=1000.0, key="bt_usdt")
            initial_apt = st.number_input("Initial APT", min_value=0.0, max_value=100.0, value=0.0, key="bt_apt")
            
        if st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest simulation..."):
                initial_balance = {"USDT": initial_usdt, "APT": initial_apt}
                backtest_result = backtesting_engine.run_backtest(
                    st.session_state.sentiment_data,
                    initial_balance,
                    bt_buy_threshold,
                    bt_sell_threshold,
                    bt_risk_level
                )
                
                if backtest_result['success']:
                    st.success("Backtest completed successfully!")
                    
                    # Display performance metrics
                    perf = backtest_result['performance']
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.metric("Total Return", f"{perf.get('total_return_pct', 0):.2f}%")
                    with metric_cols[1]:
                        st.metric("Win Rate", f"{perf.get('win_rate_pct', 0):.1f}%")
                    with metric_cols[2]:
                        st.metric("Total Trades", perf.get('total_trades', 0))
                    with metric_cols[3]:
                        st.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}")
                    
                    # Portfolio value chart
                    if backtest_result['portfolio_values']:
                        portfolio_df = pd.DataFrame(backtest_result['portfolio_values'])
                        st.line_chart(portfolio_df.set_index('timestamp')['portfolio_value'])
                else:
                    st.error("Backtest failed. Please ensure you have sufficient sentiment data.")
    else:
        st.warning("Not enough sentiment data for backtesting. Please refresh data to collect more historical information.")

with tab3:
    # Advanced Analytics Section
    st.markdown("### 📈 Advanced Analytics")
    st.markdown("**Deep insights into sentiment patterns and trading performance**")
    
    if not st.session_state.sentiment_data.empty:
        analytics_cols = st.columns(2)
        
        with analytics_cols[0]:
            st.markdown("#### Sentiment Distribution")
            st.info("ℹ️ Shows how sentiment scores are distributed across all analyzed content.")
            
            # Sentiment distribution chart
            sentiment_scores = st.session_state.sentiment_data['sentiment_score']
            
            # Create sentiment ranges
            ranges = {
                'Very Negative (0.0-0.2)': len(sentiment_scores[(sentiment_scores >= 0.0) & (sentiment_scores < 0.2)]),
                'Negative (0.2-0.4)': len(sentiment_scores[(sentiment_scores >= 0.2) & (sentiment_scores < 0.4)]),
                'Neutral (0.4-0.6)': len(sentiment_scores[(sentiment_scores >= 0.4) & (sentiment_scores < 0.6)]),
                'Positive (0.6-0.8)': len(sentiment_scores[(sentiment_scores >= 0.6) & (sentiment_scores < 0.8)]),
                'Very Positive (0.8-1.0)': len(sentiment_scores[(sentiment_scores >= 0.8) & (sentiment_scores <= 1.0)])
            }
            
            # Create DataFrame for chart
            chart_data = pd.DataFrame(list(ranges.items()), columns=['Sentiment Range', 'Count'])
            st.bar_chart(chart_data.set_index('Sentiment Range'))
            
            # FUD Detection Stats
            if 'fud_detected' in st.session_state.sentiment_data.columns:
                fud_count = st.session_state.sentiment_data['fud_detected'].sum()
                total_count = len(st.session_state.sentiment_data)
                st.metric("FUD Content Detected", f"{fud_count}/{total_count} ({fud_count/total_count*100:.1f}%)")
                st.info("ℹ️ FUD (Fear, Uncertainty, Doubt) detection helps filter out manipulative content that could mislead the trading algorithm.")
        
        with analytics_cols[1]:
            st.markdown("#### Sentiment Trends by Source")
            st.info("ℹ️ Compare sentiment across different news sources and social media platforms.")
            
            # Sentiment by source
            source_sentiment = st.session_state.sentiment_data.groupby('source')['sentiment_score'].mean().sort_values(ascending=False)
            st.bar_chart(source_sentiment)
            
            # Token mention analysis
            if 'token' in st.session_state.sentiment_data.columns:
                token_sentiment = st.session_state.sentiment_data.groupby('token')['sentiment_score'].mean()
                st.markdown("#### Average Sentiment by Token")
                for token, sentiment in token_sentiment.items():
                    sentiment_color = "🟢" if sentiment > 0.6 else "🟡" if sentiment > 0.4 else "🔴"
                    st.metric(f"{token} Sentiment", f"{sentiment_color} {sentiment:.3f}")
        
        # Strategy Performance Analysis
        if not st.session_state.trading_history.empty:
            st.markdown("#### Trading Strategy Analysis")
            st.info("ℹ️ Analysis of actual trades made by the AI agent, including timing and profitability.")
            
            trades_df = st.session_state.trading_history.copy()
            
            perf_cols = st.columns(3)
            with perf_cols[0]:
                buy_trades = len(trades_df[trades_df['action'] == 'BUY'])
                sell_trades = len(trades_df[trades_df['action'] == 'SELL'])
                st.metric("Buy/Sell Ratio", f"{buy_trades}:{sell_trades}")
            
            with perf_cols[1]:
                avg_sentiment = trades_df['sentiment_score'].mean()
                st.metric("Avg Trade Sentiment", f"{avg_sentiment:.3f}")
            
            with perf_cols[2]:
                if len(trades_df) > 1:
                    trade_frequency = len(trades_df) / max(1, (datetime.now() - trades_df['timestamp'].min()).days)
                    st.metric("Trades per Day", f"{trade_frequency:.1f}")
    else:
        st.warning("No sentiment data available for analysis. Please refresh data first.")

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;'>
    <h4>📈 Sentiment Sage Trading Agent</h4>
    <p><strong>Created by Shivangi Suyash for Aptos Thunderdome Hackathon</strong></p>
    <p>Real-time cryptocurrency sentiment analysis powered by authentic news data from CryptoPanic API</p>
    <p>Features: Sentiment analysis • FUD detection • Multi-token portfolio • Risk management • Backtesting</p>
</div>
""", unsafe_allow_html=True)
