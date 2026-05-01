"""
Data Loader - 从本地CSV加载K线数据，支持重采样到1H/4H
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import argparse


# 默认数据目录
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
LOCAL_DATA_DIR = Path("/Users/jimingzhang/Desktop/crypto_data_Pre5m")

# 可用币种
DEFAULT_COINS = ["BTC", "ETH", "ADA", "DOGE", "AVAX", "DOT", "BCH", "ETC", "ALGO", "CRO", "BNT", "CVC", "GAS"]
COINS = DEFAULT_COINS  # Alias for backwards compatibility
# v1.1: 添加BCH/ETC/ALGO/CRO等8年+数据币种，寻找Sharpe>1.0的新盈利币种

# 缓存目录
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# 目标时间框架
TIMEFRAMES = {
    "1H": "1h",
    "4H": "4h",
    "1D": "1d",
}


def find_data_file(coin: str, data_dir: Path, timeframe: str = None) -> Optional[Path]:
    """查找币种对应的CSV文件，支持两种命名格式：
    1. klines_{coin}_{timeframe}.csv (kronos数据管道格式)
    2. {coin}_USDT*.csv (原格式)

    优先匹配请求的timeframe（如1H匹配klines_DOGE_1H.csv），
    若无则返回最早的文件（更多历史数据对walk-forward更有价值）。
    """
    if timeframe:
        # 精确匹配timeframe：klines_{coin}_{timeframe}.csv
        pattern_exact = f"klines_{coin}_{timeframe}.csv"
        files_exact = list(data_dir.glob(pattern_exact))
        if files_exact:
            return files_exact[0]

    # 优先: klines_{coin}_*.csv (任意timeframe)
    pattern1 = f"klines_{coin}_*.csv"
    files = list(data_dir.glob(pattern1))
    if files:
        # 返回最新修改的（通常是最完整的数据文件）
        return sorted(files, key=lambda p: p.stat().st_mtime)[-1]
    # 备用: {coin}_USDT*.csv
    pattern2 = f"{coin}_USDT*.csv"
    files = list(data_dir.glob(pattern2))
    if files:
        return files[0]
    return None


def _parse_datetime(val) -> pd.Timestamp:
    """解析时间：支持datetime字符串和Unix毫秒时间戳"""
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, (int, float)):
        # Unix毫秒时间戳
        return pd.to_datetime(val, unit="ms")
    s = str(val).strip()
    if not s:
        return pd.NaT
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT


def load_raw_5m(coin: str, data_dir: Path = None, timeframe: str = None) -> Optional[pd.DataFrame]:
    """加载原始数据（可能是5m/1h/4h），自动检测列格式"""
    if data_dir is None:
        data_dir = LOCAL_DATA_DIR if LOCAL_DATA_DIR.exists() else DEFAULT_DATA_DIR

    csv_path = find_data_file(coin, data_dir, timeframe=timeframe)
    if csv_path is None or not csv_path.exists():
        print(f"[WARN] Data file not found for {coin} in {data_dir}")
        return None

    # 自动检测格式：读第一行判断
    raw = pd.read_csv(csv_path, nrows=2, header=0)
    first_val = str(raw.iloc[0, 0]).strip()
    n_cols = len(raw.columns)

    if first_val.replace(".", "").replace("-", "").isdigit() and len(first_val) > 12:
        # Unix毫秒时间戳格式 (DOGE, ADA等) - OKX导出格式
        # CSV columns: timestamp, open, high, low, close, vol, volCcy, volCcyQuote, confirm, datetime_utc, volume
        # Use usecols to grab first 7 + datetime_utc column (index 9) + volume (index 10)
        if n_cols >= 11:
            df = pd.read_csv(csv_path, usecols=[0, 1, 2, 3, 4, 5, 9, 10],
                             names=["ts_ms", "open", "high", "low", "close", "vol", "datetime_utc", "volume"],
                             header=0)
        else:
            # Fallback for files with fewer columns (CVC, GAS, etc have 10 cols)
            # Column 5 = vol (trade count), use as volume proxy for indicators
            df = pd.read_csv(csv_path, usecols=[0, 1, 2, 3, 4, 5, 9],
                             names=["ts_ms", "open", "high", "low", "close", "volume", "datetime_utc"],
                             header=0)
        # Convert timestamp → datetime
        df["datetime_utc"] = pd.to_datetime(df["ts_ms"], unit="ms")
    else:
        # datetime字符串格式 (BTC, ETH等)
        df = pd.read_csv(csv_path, usecols=range(7),
                         names=["datetime_utc", "datetime_utc2", "open", "high", "low", "close", "volume"],
                         header=0)
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")

    df = df.dropna(subset=["datetime_utc"])
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    # 数值列
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"].fillna(0)

    # 过滤异常时间范围
    df = df[df["datetime_utc"] > "2018-01-01"]
    df = df[df["datetime_utc"] < "2026-04-25"]

    return df[["datetime_utc", "open", "high", "low", "close", "volume"]]


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    重采样K线数据到更高时间框架
    timeframe: '1h', '4h', '1d'
    """
    freq_map = {"1h": "1h", "4h": "4h", "1d": "1d"}  # pandas 2.x 统一用小写
    freq = freq_map.get(timeframe, "1h")

    df = df.set_index("datetime_utc")

    ohlcv = pd.DataFrame()
    ohlcv["open"] = df["open"].resample(freq).first()
    ohlcv["high"] = df["high"].resample(freq).max()
    ohlcv["low"] = df["low"].resample(freq).min()
    ohlcv["close"] = df["close"].resample(freq).last()
    ohlcv["volume"] = df["volume"].resample(freq).sum()

    # 删除NaN行（没有交易的周期）
    ohlcv = ohlcv.dropna()
    ohlcv = ohlcv.reset_index()
    ohlcv.columns = ["datetime_utc", "open", "high", "low", "close", "volume"]

    return ohlcv


def load_timeframe_data(
    coin: str, timeframe: str = "1h", data_dir: Path = None
) -> Optional[pd.DataFrame]:
    """加载指定时间框架的数据（带Parquet缓存）"""
    cache_path = CACHE_DIR / f"{coin}_{timeframe}.parquet"

    # 优先读缓存
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if len(df) > 100:
                return df
        except Exception:
            pass

    # 从CSV加载
    df_5m = load_raw_5m(coin, data_dir, timeframe=timeframe)
    if df_5m is None:
        return None

    if timeframe == "5m":
        result = df_5m
    else:
        result = resample_ohlcv(df_5m, timeframe)

    # 保存缓存
    if result is not None and len(result) > 100:
        try:
            result.to_parquet(cache_path, index=False)
        except Exception:
            pass

    return result


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算技术指标
    - RSI(14)
    - ATR(14)
    - ADX(14)
    - EMA(9, 21, 50)
    - MACD(12, 26, 9)
    - Bollinger Bands(20, 2)
    - Volume SMA(20) + Volume Ratio
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # ATR(14) - True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # ADX(14) - Average Directional Index
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    atr = df["atr_14"]
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx_14"] = dx.rolling(14).mean()

    # +DI and -DI
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # EMA
    df["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()

    # MACD(12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands(20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bollinger_upper"] = sma20 + 2 * std20
    df["bollinger_middle"] = sma20
    df["bollinger_lower"] = sma20 - 2 * std20
    df["bollinger_position"] = ((close - df["bollinger_lower"]) / 
                                 (df["bollinger_upper"] - df["bollinger_lower"]) * 100).clip(0, 100)

    # Volume
    df["volume_sma_20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_sma_20"].replace(0, np.nan)

    # Return (for IC calculation)
    df["return"] = close.pct_change()

    # Hour of day (for time-of-day filtering)
    if "datetime_utc" in df.columns:
        dt = pd.to_datetime(df["datetime_utc"])
        df["hour"] = dt.dt.hour
    elif "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        df["hour"] = dt.dt.hour
    else:
        df["hour"] = -1  # Unknown hour, bypass filter

    # ATR Percentile (avoid high-vol regimes) — rolling rank of current ATR vs recent ATRs
    atr_series = df["atr_14"]
    for period in [20, 50]:
        pct = atr_series.rolling(period).apply(
            lambda x: (x < x[-1]).sum() / len(x) * 100, raw=True
        )
        df[f"atr_pct_{period}"] = pct

    # Momentum Score: normalized combination of RSI, MACD histogram, Bollinger position
    # RSI normalized: 0-100 already
    rsi_n = df["rsi_14"] / 100.0  # 0-1
    # MACD histogram normalized: divide by ATR to get scale-invariant score
    macd_n = (df["macd_histogram"] / atr_series.replace(0, np.nan)).clip(-1, 1)  # -1 to 1
    # Bollinger position: 0-100 → 0-1
    bb_n = df["bollinger_position"] / 100.0  # 0-1
    # Combined score: LONG momentum (higher = more bullish)
    # For LONG entry: want RSI low (oversold), MACD positive, BB low
    df["momentum_score"] = ((1 - rsi_n) * 0.4 + (macd_n + 1) * 0.3 + (1 - bb_n) * 0.3).clip(0, 1) * 100

    return df.reset_index(drop=True)


def walkforward_split(
    df: pd.DataFrame,
    n_windows: int = 8,
    train_ratio: float = 0.7,
    mode: str = "expanding",
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Walk-Forward分割：返回 (train_df, test_df) 列表

    mode:
      - "rolling": 等宽窗口滑动（适合长历史数据）
      - "expanding": 从最新数据往前切expanding窗口（适合聚焦近期）
      - "rolling_recent": 从数据末尾往前，取最近N个窗口
    """
    if mode == "expanding":
        # Expanding窗口：每个窗口都从数据开头训练，测试集逐步往前推
        test_size = len(df) // n_windows
        splits = []
        for i in range(n_windows - 1):
            test_end = len(df) - i * test_size
            test_start = max(0, test_end - test_size)
            train_end = test_start

            train_df = df.iloc[:train_end].copy() if train_end > 100 else None
            test_df = df.iloc[test_start:test_end].copy()

            if train_df is not None and len(train_df) > 100 and len(test_df) > 20:
                splits.append((train_df, test_df))
        splits.reverse()  # 从最早窗口到最新窗口
        return splits

    elif mode == "rolling_recent":
        # 只取最近 n_windows 个窗口
        window_size = len(df) // n_windows
        splits = []
        for i in range(n_windows - 1):
            train_end = int(window_size * (i + 1) * train_ratio)
            test_start = train_end
            test_end = min(test_start + window_size, len(df))
            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            if len(train_df) > 100 and len(test_df) > 20:
                splits.append((train_df, test_df))
        return splits

    else:
        # 默认rolling窗口
        window_size = len(df) // n_windows
        splits = []
        for i in range(n_windows - 1):
            train_end = int(window_size * (i + 1) * train_ratio)
            test_start = train_end
            test_end = min(test_start + window_size, len(df))
            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            if len(train_df) > 100 and len(test_df) > 20:
                splits.append((train_df, test_df))
        return splits


def get_data_summary() -> Dict:
    """返回数据概览"""
    summary = {}
    for coin in COINS:
        df = load_raw_5m(coin)
        if df is not None:
            summary[coin] = {
                "rows_5m": len(df),
                "start": str(df["datetime_utc"].min()),
                "end": str(df["datetime_utc"].max()),
            }
    return summary


if __name__ == "__main__":
    # 打印数据概览
    print("=== Local Data Summary ===")
    summary = get_data_summary()
    for coin, info in summary.items():
        print(f"{coin}: {info['rows_5m']:,} rows | {info['start'][:10]} ~ {info['end'][:10]}")

    # 测试加载
    print("\n=== Loading BTC 1H with indicators ===")
    df = load_timeframe_data("BTC", "1h")
    if df is not None:
        print(f"Rows: {len(df)}")
        df = compute_indicators(df)
        print(df[["datetime_utc", "close", "rsi_14", "adx_14", "macd_histogram", "volume_ratio"]].tail(5))
