# ==============================================================================
# ALL YOUR EXISTING CLASSES GO HERE:
# TimeSeriesDataset, Trade, TechnicalAnalysis, RiskManager, EnhancedRiskManager,
# AdvancedVisualizationEngine, AdvancedTradingSystem
# ...
# ==============================================================================


# NEW: A function to run the analysis, with caching for performance.
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
        
        if not results:
            st.error("Failed to retrieve or process data. Check the logs.")
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

st.title("📈 AI-Powered Trading System with Risk Analytics")

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
        col1, col2 = st.columns(2)

        with col1:
            st.header("🔗 Asset Correlation Heatmap")
            if report_data.get('correlation_heatmap'):
                st.plotly_chart(report_data['correlation_heatmap'], use_container_width=True)

        with col2:
            st.header("📊 Performance Dashboard")
            if report_data.get('performance_dashboard'):
                st.plotly_chart(report_data['performance_dashboard'], use_container_width=True)
                
elif run_button and not selected_symbols:
    st.warning("Please select at least one asset to analyze.")

else:
    st.info("Select your desired assets in the sidebar and click 'Run Analysis' to begin.")
