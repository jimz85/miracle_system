# autoresearch package - moved from kronos_autoresearch/src
from autoresearch.strategy_config import StrategyConfig, BacktestResult, Trade, Direction
from autoresearch.backtest_engine import run_single_backtest, run_walkforward
from autoresearch.data_loader import COINS, compute_indicators, load_timeframe_data
