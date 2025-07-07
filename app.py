# Essential imports
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import logging
from typing import List
from datetime import datetime, timedelta
import asyncio
import streamlit as st
import time

# NEW: Imports from the modern alpaca-py library
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetBarsRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.common.timeframe import TimeFrame
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.data.historical import StockHistoricalDataClient

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- Technical Analysis (No changes needed) ---
class TechnicalAnalysis:
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            # Ensure 'Close' column is numeric
            df['Close'] = pd.to_numeric(df['Close'])
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            if loss.eq(0).all():
                rs = np.inf
            else:
                rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'].fillna(100, inplace=True) # Fill RSI NaNs where loss is zero
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

# --- Live Trading Logic ---
class LiveTradingSystem:
    # NEW: Using the new TradingClient
    def __init__(self, symbols: List[str], trading_client: TradingClient):
        self.symbols = symbols
        self.trading_client = trading_client
        self.historical_client = StockHistoricalDataClient(st.secrets["API_KEY"], st.secrets["SECRET_KEY"])
        self.technical_analysis = TechnicalAnalysis()

    def initialize_data(self):
        if 'data' not in st.session_state:
            st.session_state.data = {}
        
        for symbol in self.symbols:
            try:
                # NEW: Using the new GetBarsRequest format
                request_params = GetBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame.Hour,
                    start=datetime.now() - timedelta(hours=200)
                )
                bars = self.historical_client.get_stock_bars(request_params).df
                bars.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
                st.session_state.data[symbol] = self.technical_analysis.add_indicators(bars)
                logger.info(f"Initialized data for {symbol} with {len(bars)} bars.")
            except Exception as e:
                logger.error(f"Failed to initialize data for {symbol}: {e}")
                st.session_state.data[symbol] = pd.DataFrame()

    def update_data(self, bar):
        symbol = bar.symbol
        if symbol in st.session_state.data:
            new_row = pd.DataFrame([{
                'Open': bar.open, 'High': bar.high, 'Low': bar.low,
                'Close': bar.close, 'Volume': bar.volume
            }], index=[pd.to_datetime(bar.timestamp)])
            st.session_state.data[symbol] = pd.concat([st.session_state.data[symbol], new_row])
            st.session_state.data[symbol] = self.technical_analysis.add_indicators(st.session_state.data[symbol])

    def generate_signal(self, symbol: str) -> int:
        if symbol not in st.session_state.data or len(st.session_state.data[symbol]) < 2:
            return 0
        latest_data = st.session_state.data[symbol].iloc[-1]
        if latest_data['MACD'] > latest_data['MACD_Signal'] and latest_data['RSI'] < 70:
            return 1
        elif latest_data['MACD'] < latest_data['MACD_Signal'] and latest_data['RSI'] > 30:
            return -1
        return 0

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📈 Live AI-Powered Trading System (Upgraded)")

# --- Connect to Alpaca ---
try:
    # NEW: Initializing the new TradingClient
    trading_client = TradingClient(st.secrets["API_KEY"], st.secrets["SECRET_KEY"], paper=True)
    st.success("Connected to Alpaca Paper Trading API.")
except Exception as e:
    st.error(f"Failed to connect to Alpaca. Check `secrets.toml`. Error: {e}")
    st.stop()

# --- Initialize Session State ---
if 'trading_system' not in st.session_state:
    st.session_state.trading_system = LiveTradingSystem(symbols=['AAPL', 'GOOGL', 'MSFT'], api=trading_client)
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
account = trading_client.get_account()
st.header(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")

col1, col2, col3 = st.columns(3)
col1.metric("Cash", f"${float(account.cash):,.2f}")
col2.metric("Buying Power", f"${float(account.buying_power):,.2f}")
col3.metric("Today's P&L", f"${float(account.equity) - float(account.last_equity):,.2f}")
st.divider()

chart_placeholder = st.empty()
log_placeholder = st.empty()
positions_placeholder = st.empty()

# --- Async Functions for Live Data ---
async def bar_callback(bar):
    symbol = bar.symbol
    st.session_state.trading_system.update_data(bar)
    signal = st.session_state.trading_system.generate_signal(symbol)
    
    try:
        position = trading_client.get_open_position(symbol)
        position_qty = int(position.qty)
    except:
        position_qty = 0

    log_msg = f"{datetime.now().strftime('%H:%M:%S')} | {symbol} | Price: {bar.close:.2f} | RSI: {st.session_state.data[symbol].iloc[-1]['RSI']:.2f} | Signal: {signal}"

    # NEW: Using the new MarketOrderRequest format
    if signal == 1 and position_qty == 0:
        order_data = MarketOrderRequest(symbol=symbol, qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        try:
            trading_client.submit_order(order_data=order_data)
            log_msg += " | ACTION: BUY order placed."
        except Exception as e:
            log_msg += f" | ACTION: BUY failed: {e}"
    elif signal == -1 and position_qty > 0:
        order_data = MarketOrderRequest(symbol=symbol, qty=position_qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        try:
            trading_client.submit_order(order_data=order_data)
            log_msg += " | ACTION: SELL order placed."
        except Exception as e:
            log_msg += f" | ACTION: SELL failed: {e}"
    else:
        log_msg += " | ACTION: HOLD."
        
    st.session_state.log.insert(0, log_msg)
    if len(st.session_state.log) > 20:
        st.session_state.log.pop()

# --- Main Loop ---
async def main_loop():
    # NEW: Using the new DataStream classes
    stock_stream = StockDataStream(st.secrets["API_KEY"], st.secrets["SECRET_KEY"])
    
    # Subscribe to symbols
    for symbol in st.session_state.trading_system.symbols:
        stock_stream.subscribe_bars(bar_callback, symbol)
    
    # Run the stream in a separate thread
    stock_stream.run()

    while st.session_state.get('run', False):
        with chart_placeholder.container():
            symbol_to_show = st.selectbox("Select asset to view:", st.session_state.trading_system.symbols)
            if symbol_to_show and symbol_to_show in st.session_state.data and not st.session_state.data[symbol_to_show].empty:
                fig = go.Figure()
                df = st.session_state.data[symbol_to_show]
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candles'))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'))
                fig.update_layout(title=f"Live Chart for {symbol_to_show}", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

        with log_placeholder.container():
            st.subheader("📜 Trading Log")
            st.text_area("", value="\n".join(st.session_state.log), height=300)

        with positions_placeholder.container():
            st.subheader("💼 Open Positions")
            positions = trading_client.get_all_positions()
            if positions:
                positions_df = pd.DataFrame([{'symbol': p.symbol, 'qty': p.qty, 'market_value': p.market_value, 'unrealized_pl': p.unrealized_pl} for p in positions])
                st.dataframe(positions_df, use_container_width=True)
            else:
                st.info("No open positions.")
        
        await asyncio.sleep(5)
    
    stock_stream.stop()
    logger.info("Trading streams stopped.")


if st.session_state.get('run', False):
    try:
        asyncio.run(main_loop())
    except Exception as e:
        st.error(f"An error occurred in the main loop: {e}")
        st.session_state.run = False
else:
    st.info("System is stopped. Click 'Start Trading' in the sidebar to begin.")

