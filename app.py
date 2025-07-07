# Essential imports
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yfinance as yf
# import tensorflow as tf  <- THIS LINE IS REMOVED
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dataclasses import dataclass
import asyncio
import base64
from io import BytesIO
import networkx as nx
from PIL import Image
import streamlit as st

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 100,
        target_horizon: int = 10
    ):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.target_horizon = target_horizon

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.target_horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length: idx + self.sequence_length + self.target_horizon]
        return x, y

@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    entry_price: float
    position_size: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    side: str = 'long'
    pnl: float = 0.0

class TechnicalAnalysis:
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            # Basic indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df

    @staticmethod
    def generate_signals(df: pd.DataFrame) -> pd.Series:
        try:
            signals = pd.Series(index=df.index, data=0)
            # Combined signals
            signals[(df['RSI'] < 30) & (df['Close'] < df['BB_Lower'])] = 1  # Strong buy
            signals[(df['RSI'] > 70) & (df['Close'] > df['BB_Upper'])] = -1 # Strong sell
            signals[(df['MACD'] > df['MACD_Signal']) & (df['Close'] > df['SMA_20'])] += 0.5
            signals[(df['MACD'] < df['MACD_Signal']) & (df['Close'] < df['SMA_20'])] -= 0.5
            return signals
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return pd.Series(index=df.index, data=0)

class RiskManager:
    def __init__(
        self,
        confidence_level: float = 0.95,
        max_position_size: float = 0.2,
        max_drawdown_limit: float = 0.15
    ):
        self.confidence_level = confidence_level
        self.max_position_size = max_position_size
        self.max_drawdown_limit = max_drawdown_limit

    def calculate_metrics(self, returns: pd.Series) -> Dict:
        try:
            var = self._calculate_var(returns)
            return {
                'var': var,
                'cvar': self._calculate_cvar(returns, var),
                'sharpe': self._calculate_sharpe(returns),
                'max_drawdown': self._calculate_max_drawdown(returns)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def _calculate_var(self, returns: pd.Series) -> float:
        return np.percentile(returns, (1 - self.confidence_level) * 100)

    def _calculate_cvar(self, returns: pd.Series, var: float) -> float:
        return returns[returns <= var].mean()

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        return returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns.min()

class EnhancedRiskManager(RiskManager):
    def advanced_risk_metrics(self, returns: pd.Series) -> Dict:
        try:
            base_metrics = super().calculate_metrics(returns)
            enhanced_metrics = {
                'sortino_ratio': self._calculate_sortino_ratio(returns),
                'downside_risk': self._calculate_downside_risk(returns),
                'calmar_ratio': self._calculate_calmar_ratio(returns),
                'omega_ratio': self._calculate_omega_ratio(returns)
            }
            return {**base_metrics, **enhanced_metrics}
        except Exception as e:
            logger.error(f"Error calculating enhanced metrics: {str(e)}")
            return {}

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        downside_risk = self._calculate_downside_risk(returns)
        return (returns.mean() * 252) / (downside_risk * np.sqrt(252)) if downside_risk != 0 else 0

    def _calculate_downside_risk(self, returns: pd.Series) -> float:
        downside_returns = returns[returns < 0]
        return np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        max_dd = self._calculate_max_drawdown(returns)
        return (returns.mean() * 252) / abs(max_dd) if max_dd != 0 else 0

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        positive_returns = returns[returns > threshold].sum()
        negative_returns = abs(returns[returns < threshold].sum())
        return positive_returns / negative_returns if negative_returns != 0 else float('inf')

class AdvancedVisualizationEngine:
    @staticmethod
    def create_correlation_heatmap(returns_data: pd.DataFrame, title: str = 'Asset Correlation Heatmap') -> go.Figure:
        try:
            # Ensure data is properly aligned
            correlation_matrix = returns_data.corr()
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=np.round(correlation_matrix, 3),
                texttemplate='%{text}',
                textfont={"size": 12},
                hoverongaps=False,
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            fig.update_layout(
                title={
                    'text': f'{title}<br><sup>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</sup>',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                width=900,
                height=800,
                xaxis_title="Assets",
                yaxis_title="Assets",
                xaxis={'side': 'bottom'},
                yaxis={'autorange': 'reversed'},
                template='plotly_white'
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            return None

    @staticmethod
    def create_performance_dashboard(results: Dict) -> go.Figure:
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Cumulative Returns',
                    'Risk Metrics',
                    'Daily Returns Distribution',
                    'Rolling Volatility (20-day)'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            colors = px.colors.qualitative.Set3
            for i, (symbol, data) in enumerate(results['market_data'].items()):
                returns = data['Close'].pct_change().dropna()
                cumulative_returns = (1 + returns).cumprod()

                # Cumulative Returns
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=cumulative_returns,
                        name=f"{symbol}",
                        mode='lines',
                        line=dict(color=colors[i % len(colors)])
                    ),
                    row=1, col=1
                )

                # Risk Metrics
                metrics = results['risk_metrics'][symbol]
                fig.add_trace(
                    go.Bar(
                        x=['Sharpe', 'Sortino', 'Calmar'],
                        y=[
                            metrics['sharpe'],
                            metrics['sortino_ratio'],
                            metrics.get('calmar_ratio', 0)
                        ],
                        name=f"{symbol}",
                        marker_color=colors[i % len(colors)]
                    ),
                    row=1, col=2
                )

                # Returns Distribution
                fig.add_trace(
                    go.Histogram(
                        x=returns,
                        name=f"{symbol}",
                        nbinsx=50,
                        histnorm='probability',
                        marker_color=colors[i % len(colors)]
                    ),
                    row=2, col=1
                )

                # Rolling Volatility
                rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
                fig.add_trace(
                    go.Scatter(
                        x=data.index[20:],
                        y=rolling_vol[20:],
                        name=f"{symbol}",
                        mode='lines',
                        line=dict(color=colors[i % len(colors)])
                    ),
                    row=2, col=2
                )

            fig.update_layout(
                height=1000,
                width=1200,
                showlegend=True,
                title_text="Trading Performance Dashboard",
                template='plotly_white'
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {str(e)}")
            return None

class AdvancedTradingSystem:
    def __init__(
        self,
        symbols: List[str] = ['BTC-USD', 'ETH-USD', 'AAPL', 'GOOGL'],
        initial_capital: float = 100000.0,
        risk_params: Dict = None
    ):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_params = risk_params or {
            'confidence_level': 0.95,
            'max_position_size': 0.2,
            'max_drawdown_limit': 0.15
        }
        self.visualization_engine = AdvancedVisualizationEngine()
        self.risk_manager = EnhancedRiskManager(**self.risk_params)
        self.technical_analysis = TechnicalAnalysis()
        self.results = {}
        logger.info(f"Trading system initialized with symbols: {', '.join(symbols)}")

    async def run(self, train: bool = True) -> Dict:
        """Execute trading system analysis."""
        try:
            logger.info("Fetching market data...")
            market_data = await self._fetch_data()
            # Process data and calculate metrics
            processed_data = {}
            returns_data = {}
            risk_metrics = {}
            for symbol, df in market_data.items():
                # Add technical indicators
                processed_data[symbol] = self.technical_analysis.add_indicators(df.copy())
                # Calculate returns
                returns = df['Close'].pct_change().dropna()
                returns_data[symbol] = returns
                # Calculate risk metrics
                risk_metrics[symbol] = self.risk_manager.advanced_risk_metrics(returns)

            self.results = {
                'market_data': market_data,
                'processed_data': processed_data,
                'returns_data': returns_data,
                'risk_metrics': risk_metrics
            }
            logger.info("Analysis completed successfully")
            return self.results
        except Exception as e:
            logger.error(f"Error running analysis: {str(e)}")
            return {}

    async def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch market data for all symbols."""
        data = {}
        for symbol in self.symbols:
            try:
                logger.info(f"Fetching data for {symbol}...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='60d', interval='1h')
                if not df.empty:
                    data[symbol] = df
                    logger.info(f"Successfully fetched data for {symbol}")
                else:
                    logger.warning(f"No data received for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return data

    def generate_comprehensive_report(self, results: Dict) -> Dict:
        """Generate comprehensive trading report with visualizations."""
        try:
            # Create returns dataframe for correlation analysis
            returns_df = pd.DataFrame({
                symbol: data['Close'].pct_change().dropna()
                for symbol, data in results['market_data'].items()
            })
            # Log correlation matrix for verification
            correlation_matrix = returns_df.corr()
            logger.info("\nCorrelation Matrix:")
            logger.info(f"\n{correlation_matrix}")

            # Generate correlation heatmap
            correlation_heatmap = self.visualization_engine.create_correlation_heatmap(
                returns_df,
                title='Asset Correlation Heatmap'
            )

            # Generate performance dashboard
            performance_dashboard = self.visualization_engine.create_performance_dashboard(
                results
            )

            # Generate text report
            text_report = self._generate_text_report(results)
            return {
                'text_report': text_report,
                'correlation_heatmap': correlation_heatmap,
                'performance_dashboard': performance_dashboard
            }
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {'text_report': f"Error generating report: {str(e)}"}

    def _generate_text_report(self, results: Dict) -> str:
        """Generate detailed text report."""
        try:
            report = [
                "# Trading System Performance Report",
                f"## Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "\n### Risk Analysis Summary"
            ]
            for symbol in self.symbols:
                if symbol in results['risk_metrics']:
                    metrics = results['risk_metrics'][symbol]
                    report.extend([
                        f"\n#### {symbol}",
                        f"- Sharpe Ratio: {metrics['sharpe']:.3f}",
                        f"- Sortino Ratio: {metrics['sortino_ratio']:.3f}",
                        f"- Maximum Drawdown: {metrics['max_drawdown']:.2%}",
                        f"- Value at Risk (95%): {metrics['var']:.2%}",
                        f"- Conditional VaR: {metrics['cvar']:.2%}",
                        f"- Calmar Ratio: {metrics['calmar_ratio']:.3f}",
                        f"- Omega Ratio: {metrics['omega_ratio']:.3f}"
                    ])
            return "\n".join(report)
        except Exception as e:
            logger.error(f"Error generating text report: {str(e)}")
            return f"Error generating text report: {str(e)}"


# This decorator is key: it prevents Streamlit from re-fetching data and
# re-calculating everything every time a widget is changed.
@st.cache_data
def run_analysis(symbols: List[str]):
    """
    This function takes the selected symbols, runs the full trading system analysis,
    and returns the generated report.
    """
    try:
        # We need to run the async functions within Streamlit's synchronous flow.
        # asyncio.run() is the simplest way to do this for a self-contained operation.
        logger.info(f"Starting analysis for: {', '.join(symbols)}")
        trading_system = AdvancedTradingSystem(symbols=symbols)
        
        # Run the async data fetching and processing
        results = asyncio.run(trading_system.run(train=True))
        
        if not results or not results.get('market_data'):
            st.error("Failed to retrieve or process data. Check the logs for errors.")
            return None
            
        # Generate the final report dictionary (text + plots)
        report = trading_system.generate_comprehensive_report(results)
        logger.info("Successfully generated comprehensive report.")
        return report
        
    except Exception as e:
        logger.error(f"An error occurred in run_analysis: {e}")
        st.error(f"An error occurred during analysis: {e}")
        return None


# --- Streamlit User Interface ---

st.set_page_config(layout="wide")

st.title("📈 AI-POWERED TRADING SYSTEM WITH RISK ANALYTICS")

st.markdown("""
Welcome to the interactive trading system dashboard. 
Use the sidebar to select the assets you want to analyze and click 'Run Analysis' to generate a comprehensive report.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Analysis Configuration")

# Let the user select multiple symbols from a predefined list.
available_symbols = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', # Crypto
    'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA',   # Tech Stocks
    'SPY', 'QQQ'                              # ETFs
]
selected_symbols = st.sidebar.multiselect(
    "Select assets for analysis:",
    options=available_symbols,
    default=['BTC-USD', 'ETH-USD', 'AAPL', 'GOOGL'] # Default selection
)

# Button to trigger the analysis
run_button = st.sidebar.button("🚀 Run Analysis")


# --- Main Panel for Displaying Results ---
if run_button and selected_symbols:
    # Show a spinner while the analysis is running
    with st.spinner(f"Performing analysis for **{', '.join(selected_symbols)}**... This may take a moment."):
        report_data = run_analysis(selected_symbols)

    if report_data:
        st.success("Analysis complete!")
        st.divider()

        # Display Text Report
        st.header("📝 Risk Analysis Report")
        st.markdown(report_data['text_report'])
        st.divider()

        # Display Visualizations
        # Using columns for a side-by-side layout
        col1, col2 = st.columns(2)

        with col1:
            st.header("🔗 Asset Correlation Heatmap")
            if report_data.get('correlation_heatmap'):
                st.plotly_chart(report_data['correlation_heatmap'], use_container_width=True)
            else:
                st.warning("Could not generate the correlation heatmap.")

        with col2:
            st.header("📊 Performance Dashboard")
            if report_data.get('performance_dashboard'):
                st.plotly_chart(report_data['performance_dashboard'], use_container_width=True)
            else:
                st.warning("Could not generate the performance dashboard.")
                
elif run_button and not selected_symbols:
    st.warning("Please select at least one asset to analyze.")

else:
    st.info("Select your desired assets in the sidebar and click 'Run Analysis' to begin.")
