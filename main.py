from .trading_analyzer import TradingAnalyzer  # Note the dot (.) for relative import
from .bybit_api import BybitAPI
from .config import Config
from .logger_setup import setup_logger

# Add a module docstring
"""
Main module for running the trading analysis.
"""

config = Config("config.json")  # Pass the config file name


def main():
    #  """
    # Main function to run the trading analysis.
    # """
    colorama.init()  # Initialize colorama for colored output
    logger = setup_logger(__name__)  # Set up the logger

    api = BybitAPI(config, logger)
    analyzer = TradingAnalyzer(config, logger)  # Pass config here as well
    output = TradingOutput(logger, config)  # Pass both logger and config

    while True:
        # Your trading analysis logic here
        current_price = api.fetch_current_price()  # Example: Fetch current price
        if current_price is None:
            logger.warning("Failed to fetch current price. Skipping analysis.")
            time.sleep(60)
            continue  # skip to the next loop

        analysis_results = analyzer.analyze(current_price)
        if analysis_results is None:
            logger.warning("Analysis results are empty. Skipping display.")
            time.sleep(60)
            continue  # skip to the next loop

        output.display_analysis(analysis_results)

        # Sleep for a specific duration, e.g., 60 seconds
        time.sleep(60)


if __name__ == "__main__":
    main()
