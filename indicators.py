# indicators.py — модуль индикаторов и вспомогательных функций
import pandas as pd
import numpy as np

# === ATR Volatility Bands ===

def atr_bands(df: pd.DataFrame, atr_period: int = 14, ma_period: int = 20, mult: float = 2.0):
    """Возвращает basis, upper, lower ATR‑канала."""
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low']  - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    basis = df['Close'].ewm(span=ma_period, adjust=False).mean()
    upper = basis + mult * atr
    lower = basis - mult * atr
    return basis, upper, lower

# === Bollinger Band Stops (упрощённая логика направления) ===

def bb_stops(df: pd.DataFrame, length: int = 20, mult: float = 1.0):
    basis = df['Close'].ewm(span=length, adjust=False).mean()
    dev = mult * df['Close'].rolling(length).std()
    upper = basis + dev
    lower = basis - dev

    direction = np.full(len(df), np.nan, dtype=object)
    stop_line = np.full(len(df), np.nan)

    for i in range(1, len(df)):
        if df['Close'].iloc[i] > upper.iloc[i]:
            direction[i] = 'up'
        elif df['Close'].iloc[i] < lower.iloc[i]:
            direction[i] = 'down'
        else:
            direction[i] = direction[i-1] if pd.notna(direction[i-1]) else 'none'
        stop_line[i] = lower.iloc[i] if direction[i] == 'up' else upper.iloc[i] if direction[i] == 'down' else np.nan
    return pd.Series(direction, index=df.index), pd.Series(stop_line, index=df.index)

# === ADX Histogram с цветовой логикой ===

def adx_histogram(df: pd.DataFrame, period: int = 14):
    up_move   = df['High'].diff()
    down_move = df['Low'].diff().abs()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low']  - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    plus_di  = 100 * pd.Series(plus_dm).rolling(period).sum() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).sum() / atr
    dx       = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx      = dx.rolling(period).mean()

    colors = []
    last_color = None
    for i in range(len(df)):
        if i < period:
            colors.append(None)
            continue
        if plus_di.iloc[i] > minus_di.iloc[i]:
            last_color = 'blue'
        elif minus_di.iloc[i] > plus_di.iloc[i]:
            last_color = 'red'
        colors.append(last_color)
    return pd.Series(adx, index=df.index), pd.Series(colors, index=df.index)

# === Агрегация 1‑минутных данных в 15‑минутный таймфрейм ===

def resample_to_15min(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Принимает DF с индексом‑датой и колонками Open, High, Low, Close, Volume; возвращает 15‑минутный DF."""
    df = df_1min.copy()

    # Гарантируем DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    agg = {
        'Open':   'first',
        'High':   'max',
        'Low':    'min',
        'Close':  'last',
        'Volume': 'sum',
    }
    df_15 = df.resample('15T').agg(agg)
    df_15.dropna(inplace=True)
    df_15.reset_index(inplace=True)
    return df_15

# === Генерация сигналов стратегии ===

def generate_signals(df_15min: pd.DataFrame, atr_touch_pct: float = 0.05):
    basis, up_atr, low_atr = atr_bands(df_15min)
    bb_dir, _ = bb_stops(df_15min)
    _, adx_col = adx_histogram(df_15min)

    channel = up_atr - low_atr
    close = df_15min['Close']

    long_sig = (
        (abs(close - low_atr) / channel < atr_touch_pct) &
        (bb_dir == 'up') & (adx_col == 'blue')
    )
    short_sig = (
        (abs(close - up_atr) / channel < atr_touch_pct) &
        (bb_dir == 'down') & (adx_col == 'red')
    )
    return long_sig, short_sig
