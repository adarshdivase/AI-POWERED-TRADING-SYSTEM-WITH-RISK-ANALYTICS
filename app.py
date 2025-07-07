# Essential imports
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import asyncio
import streamlit as st
import time
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.stream import Stream

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- Technical Analysis (No changes needed) ---
class TechnicalAnalysis:
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

# --- Live Trading Logic ---
class LiveTradingSystem:
    def __init__(self, symbols: List[str], api: REST):
        self.symbols = symbols
        self.api = api
        self.technical_analysis = TechnicalAnalysis()

    def initialize_data(self):
        """Fetch initial historical data to build indicators."""
        if 'data' not in st.session_state:
            st.session_state.data = {}
        
        for symbol in self.symbols:
            try:
                # Fetch last 200 hours of data to have enough for indicators
                bars = self.api.get_bars(symbol, TimeFrame.Hour, start=(datetime.now() - pd.Timedelta(hours=200)).isoformat()).df
                bars = bars[bars.exchange.isin(['CRYPTO', 'NASDAQ', 'NYSE'])] # Filter for primary exchanges
                bars.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
                st.session_state.data[symbol] = self.technical_analysis.add_indicators(bars)
                logger.info(f"Initialized data for {symbol} with {len(bars)} bars.")
            except Exception as e:
                logger.error(f"Failed to initialize data for {symbol}: {e}")
                st.session_state.data[symbol] = pd.DataFrame()


    def update_data(self, bar):
        """Append a new bar of data and update indicators."""
        symbol = bar.symbol
        if symbol in st.session_state.data:
            new_row = pd.DataFrame([{
                'Open': bar.open, 'High': bar.high, 'Low': bar.low,
                'Close': bar.close, 'Volume': bar.volume
            }], index=[pd.to_datetime(bar.timestamp)])
            
            # Use concat instead of append
            st.session_state.data[symbol] = pd.concat([st.session_state.data[symbol], new_row])
            # Recalculate indicators for the whole series
            st.session_state.data[symbol] = self.technical_analysis.add_indicators(st.session_state.data[symbol])

    def generate_signal(self, symbol: str) -> int:
        """Generate a trading signal from the latest data point."""
        if symbol not in st.session_state.data or len(st.session_state.data[symbol]) < 2:
            return 0
        
        latest_data = st.session_state.data[symbol].iloc[-1]
        
        # Simple Crossover and RSI Strategy
        if latest_data['MACD'] > latest_data['MACD_Signal'] and latest_data['RSI'] < 70:
            return 1  # Buy signal
        elif latest_data['MACD'] < latest_data['MACD_Signal'] and latest_data['RSI'] > 30:
            return -1 # Sell signal
        return 0 # Hold


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📈 Live AI-Powered Trading System")

# --- Connect to Alpaca ---
try:
    API_KEY = st.secrets["API_KEY"]
    SECRET_KEY = st.secrets["SECRET_KEY"]
    api = REST(API_KEY, SECRET_KEY, base_url='https://paper-api.alpaca.markets')
    st.success("Connected to Alpaca Paper Trading API.")
except Exception as e:
    st.error(f"Failed to connect to Alpaca. Please check your `secrets.toml` file. Error: {e}")
    st.stop()


# --- Initialize Session State ---
if 'trading_system' not in st.session_state:
    st.session_state.trading_system = LiveTradingSystem(symbols=['BTC/USD', 'ETH/USD', 'AAPL'], api=api)
    st.session_state.trading_system.initialize_data()
    st.session_state.log = []
    st.session_state.run = False

# --- UI Controls ---
st.sidebar.header("⚙️ Controls")
if st.sidebar.button("▶️ Start Trading"):
    st.session_state.run = True
if st.sidebar.button("⏹️ Stop Trading"):
    st.session_state.run = False

# --- Main Dashboard ---
account = api.get_account()
st.header(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")

# Layout for metrics
col1, col2, col3 = st.columns(3)
col1.metric("Cash", f"${float(account.cash):,.2f}")
col2.metric("Buying Power", f"${float(account.buying_power):,.2f}")
col3.metric("Today's P&L", f"${float(account.equity) - float(account.last_equity):,.2f}")

st.divider()

# Placeholders for live charts and data
chart_placeholder = st.empty()
log_placeholder = st.empty()
positions_placeholder = st.empty()


async def trade_callback(bar):
    """This function is called for every new trade bar."""
    symbol = bar.symbol
    st.session_state.trading_system.update_data(bar)
    signal = st.session_state.trading_system.generate_signal(symbol)
    
    # Get current position
    try:
        position = api.get_position(symbol)
        position_qty = int(position.qty)
    except:
        position_qty = 0

    log_msg = f"{datetime.now().strftime('%H:%M:%S')} | {symbol} | Price: {bar.close:.2f} | RSI: {st.session_state.data[symbol].iloc[-1]['RSI']:.2f} | Signal: {signal}"

    # --- Trade Execution Logic ---
    if signal == 1 and position_qty == 0: # Buy signal and no position
        try:
            api.submit_order(symbol=symbol, qty=1, side='buy', type='market', time_in_force='day')
            log_msg += " | ACTION: BUY order placed."
        except Exception as e:
            log_msg += f" | ACTION: BUY failed: {e}"
    elif signal == -1 and position_qty > 0: # Sell signal and have a position
        try:
            api.submit_order(symbol=symbol, qty=position_qty, side='sell', type='market', time_in_force='day')
            log_msg += " | ACTION: SELL order placed."
        except Exception as e:
            log_msg += f" | ACTION: SELL failed: {e}"
    else:
        log_msg += " | ACTION: HOLD."
        
    st.session_state.log.insert(0, log_msg)
    if len(st.session_state.log) > 20: # Keep log size manageable
        st.session_state.log.pop()


# --- Main Loop ---
async def main_loop():
    # Subscribe to minute bars
    stream = Stream(API_KEY, SECRET_KEY, base_url='https://paper-api.alpaca.markets', data_feed='iex') # 'iex' for stocks, 'us_crypto' for crypto
    crypto_stream = Stream(API_KEY, SECRET_KEY, base_url='https://paper-api.alpaca.markets', data_feed='us_crypto')

    # Subscribe to trade updates
    for symbol in st.session_state.trading_system.symbols:
        if "/" in symbol: # Crypto
            crypto_stream.subscribe_bars(trade_callback, symbol)
        else: # Stock
            stream.subscribe_bars(trade_callback, symbol)
    
    # Start the streams in the background
    stream.run()
    crypto_stream.run()

    while st.session_state.get('run', False):
        # Update UI components
        with chart_placeholder.container():
            symbol_to_show = st.selectbox("Select asset to view:", st.session_state.trading_system.symbols)
            if symbol_to_show and not st.session_state.data[symbol_to_show].empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=st.session_state.data[symbol_to_show].index,
                    open=st.session_state.data[symbol_to_show]['Open'],
                    high=st.session_state.data[symbol_to_show]['High'],
                    low=st.session_state.data[symbol_to_show]['Low'],
                    close=st.session_state.data[symbol_to_show]['Close'],
                    name='Candles'
                ))
                fig.add_trace(go.Scatter(x=st.session_state.data[symbol_to_show].index, y=st.session_state.data[symbol_to_show]['SMA_20'], mode='lines', name='SMA 20'))
                fig.update_layout(title=f"Live Chart for {symbol_to_show}", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

        with log_placeholder.container():
            st.subheader("📜 Trading Log")
            st.text_area("", value="\n".join(st.session_state.log), height=300)

        with positions_placeholder.container():
            st.subheader("💼 Open Positions")
            positions = api.list_positions()
            if positions:
                positions_df = pd.DataFrame(
                    [{'symbol': p.symbol, 'qty': p.qty, 'market_value': p.market_value, 'unrealized_pl': p.unrealized_pl} for p in positions]
                )
                st.dataframe(positions_df, use_container_width=True)
            else:
                st.info("No open positions.")
        
        # Loop delay
        time.sleep(5) # Refresh UI every 5 seconds
    
    # Stop the streams when the loop exits
    stream.stop()
    crypto_stream.stop()
    logger.info("Trading streams stopped.")


if st.session_state.get('run', False):
    try:
        asyncio.run(main_loop())
    except Exception as e:
        st.error(f"An error occurred in the main loop: {e}")
        st.session_state.run = False
else:
    st.info("System is stopped. Click 'Start Trading' in the sidebar to begin.")
