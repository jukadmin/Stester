# indicators.py — модуль индикаторов и вспомогательных функций
import pandas as pd
import numpy as np

# === Определение точности цен (десятичных знаков) ===
def detect_price_precision(df: pd.DataFrame) -> int:
    """Определяет максимальное количество знаков после запятой по Open, High, Low, Close."""
    sample_values = df[['Open', 'High', 'Low', 'Close']].iloc[0]
    max_decimals = max(
        [len(str(val).split('.')[-1]) if '.' in str(val) else 0 for val in sample_values]
    )
    return max_decimals

# === ATR Volatility Bands ===
def atr_bands(df: pd.DataFrame, atr_period: int = 14, ma_period: int = 20, mult: float = 2.0):
    precision = detect_price_precision(df)
    # tr = pd.concat([
    #     df['High'] - df['Low'],
    #     (df['High'] - df['Close'].shift()).abs(),
    #     (df['Low']  - df['Close'].shift()).abs()
    # ], axis=1).max(axis=1)
    h = df['High'].to_numpy()
    l = df['Low'].to_numpy()
    c = df['Close'].to_numpy()
    c_prev = np.roll(c, 1).astype("float64")
    c_prev[0] = np.nan

    tr_np = np.nanmax([
        h - l,
        np.abs(h - c_prev),
        np.abs(l - c_prev)
    ], axis=0)
    tr = pd.Series(tr_np, index=df.index)

    atr = tr.rolling(atr_period).mean()

    basis = df['Close'].ewm(span=ma_period, adjust=False).mean()
    upper = basis + mult * atr
    lower = basis - mult * atr

    return basis.round(precision), upper.round(precision), lower.round(precision)

# === Bollinger Band Stops (упрощённая логика направления) ===
def bb_stops(df: pd.DataFrame, bb_length, bb_mult):
    precision = detect_price_precision(df)
    basis = df['Close'].ewm(span=bb_length, adjust=False).mean()
    mult = bb_mult / 100
    dev = mult * df['Close'].rolling(bb_length).std()
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

    return pd.Series(direction, index=df.index), pd.Series(stop_line, index=df.index).round(precision)

   
# === ADX Histogram с цветовой логикой ===
def adx_histogram(df: pd.DataFrame, period, NINT):
    # #print("df15 indic", df)
    # precision = detect_price_precision(df)
    # up_move   = df['High'].diff().abs()
    # #print("upmove cmd :", df['High'].diff())
    # #print("upmove cmd abs :", df['High'].diff().abs())
    # #print("upmove str :", up_move)
    # down_move = df['Low'].diff().abs()
    # plus_dm = up_move.where((up_move > down_move) & (up_move > 0), other=0) # type: ignore
    # minus_dm = down_move.where((down_move > up_move) & (down_move > 0), other=0) # type: ignore

    # tr = pd.concat([
    #     df['High'] - df['Low'],
    #     (df['High'] - df['Close'].shift()).abs(),
    #     (df['Low']  - df['Close'].shift()).abs()
    # ], axis=1).max(axis=1)
    # atr = tr.rolling(period).mean()

    # plus_di  = 100 * pd.Series(plus_dm).rolling(period).sum() / atr
    # minus_di = 100 * pd.Series(minus_dm).rolling(period).sum() / atr
    # dx       = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    # adx      = dx.rolling(period).mean()

    # colors = []
    # last_color = None
    # for i in range(len(df)):
    #     if i < period:
    #         colors.append(None)
    #         continue
    #     if plus_di.iloc[i] > minus_di.iloc[i]:
    #         last_color = 'blue'
    #     elif minus_di.iloc[i] > plus_di.iloc[i]:
    #         last_color = 'red'
    #     colors.append(last_color)


    
    precision = detect_price_precision(df)      # 4 у ADA
    scale = 10 ** precision                      # 10000

    high_i = (df["High"]  * scale).astype("int64")
    low_i  = (df["Low"]   * scale).astype("int64")
    close_i= (df["Close"] * scale).astype("int64")

    up_move   = (high_i.diff()).abs()
    down_move = (low_i.diff()).abs()

    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), other=0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), other=0)

    # True Range — NumPy быстрее
    h = high_i.to_numpy()
    l = low_i.to_numpy()
    c = close_i.to_numpy()
    c_prev = np.roll(c, 1).astype("float64")  # <--- это фикс
    c_prev[0] = np.nan
    tr_np = np.nanmax([h - l, np.abs(h - c_prev), np.abs(l - c_prev)], axis=0)
    tr = pd.Series(tr_np, index=df.index)
    atr = tr.rolling(period).mean()

    plus_di  = 100 * plus_dm.rolling(period).sum() / atr
    minus_di = 100 * minus_dm.rolling(period).sum() / atr
    dx       = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx      = dx.rolling(period).mean()

    # Цветовая логика без цикла
    color_series = pd.Series("black", index=df.index, dtype=object)

    for i in range(period, len(df)):
        if pd.isna(plus_di.iloc[i]) or pd.isna(minus_di.iloc[i]):
            continue
        if plus_di.iloc[i] > minus_di.iloc[i]:
            color_series.iloc[i] = "blue"
        elif minus_di.iloc[i] > plus_di.iloc[i]:
            color_series.iloc[i] = "red"
        else:
            color_series.iloc[i] = color_series.iloc[i - 1]

    adx_series = adx.round(precision)

    # 8️⃣ Диагностические логи (ldbg ≥ 2)
    # if ldbg >= 4:
    #     logger.info(
    #         f"[ADX] up_move={up_move.iloc[-1]} down_move={down_move.iloc[-1]}  plus_dm={plus_dm.iloc[-1]} minus_dm={minus_dm.iloc[-1]} "
    #         f"tr={tr.iloc[-1]}  atr={atr.iloc[-1]} plus_di={plus_di.iloc[-1]:.2f}  minus_di={minus_di.iloc[-1]:.2f} "
    #         f"[ADX] dx={dx.iloc[-1]:.2f}  adx={adx.iloc[-1]:.2f}"
    #     )

    # 9️⃣ Возвращаем:
    #     • adx – снова float (делим на 1, т.к. плюс_di / minus_di уже в float)
    #     • округляем по detect_price_precision
    #adx_series   = adx.round(precision)
    #color_series = pd.Series(colors, index=df.index)

    #print(f"📊 IterN: {NINT} Colors: {color_series.value_counts(dropna=False).to_dict()} | 📈 ADX>20: {(adx_series > 20).sum()} | 📉 ADX≤20: {(adx_series <= 20).sum()}")

    return adx_series, color_series
    
    #print(f"type: {type(adx)}, dtype: {adx.dtype}, \n ADX: {adx.round(precision).tail(10)}, \n Colors: {pd.Series(colors, index=df.index).tail(10)}  \n ") 
    #return pd.Series(adx, index=df.index).round(precision), pd.Series(colors, index=df.index)

# === Агрегация 1‑минутных данных в 15‑минутный таймфрейм ===
def resample_to_15min(df_1min: pd.DataFrame) -> pd.DataFrame:
    # Принимает DF с индексом‑датой и колонками Open, High, Low, Close, Volume; возвращает 15‑минутный DF
    df = df_1min.copy()
    
    # Гарантируем DatetimeIndex
    #if not isinstance(df.index, pd.DatetimeIndex): # было раньше как проверка. 
    #    df.index = pd.to_datetime(df.index)
    df.index = pd.to_datetime(df.index)  # без условия, всегда преобразуем

    agg = {
        'Open':   'first',
        'High':   'max',
        'Low':    'min',
        'Close':  'last',
        'Volume': 'sum',
    }
    df_15 = df.resample('15min').agg(agg) # type: ignore
    df_15.dropna(inplace=True)
    #df_15.reset_index(inplace=True)
    #print("Загружено баров (15мин):", len(df_15))  # Контрольная точка
    return df_15

# === Генерация сигналов стратегии ===
def generate_signals(df_15min: pd.DataFrame, adx_period, atr_touch_pct, bb_length, bb_mult, lookback_bars, adx_min, NINT: int):
    basis, up_atr, low_atr = atr_bands(df_15min)
    bb_dir, bb_stop = bb_stops(df_15min, bb_length, bb_mult)
    adx, adx_col = adx_histogram(df_15min, adx_period, NINT)

    channel = up_atr - low_atr
    close = df_15min['Close']
    open_ = df_15min['Open']


    # Создаём флаги касания ATR по Close и Open за lookback_bars
    # long_touch = pd.Series(False, index=df_15min.index)
    # short_touch = pd.Series(False, index=df_15min.index)
    # #print("long_touch" , long_touch) 
    # for i in range(lookback_bars, len(df_15min)):
    #     recent_open = open_.iloc[i-lookback_bars:i+1]
    #     recent_close = close.iloc[i-lookback_bars:i+1]
    #     recent_channel = channel.iloc[i]
    #     recent_lower = low_atr.iloc[i]
    #     recent_upper = up_atr.iloc[i]

    #     atr_touch_pct_new = atr_touch_pct / 100
    #     if ((abs(recent_open - recent_lower) / recent_channel < atr_touch_pct_new) | 
    #         (abs(recent_close - recent_lower) / recent_channel < atr_touch_pct_new)).any():
    #         long_touch.iloc[i] = True

    #     if ((abs(recent_open - recent_upper) / recent_channel < atr_touch_pct_new) | 
    #         (abs(recent_close - recent_upper) / recent_channel < atr_touch_pct_new)).any():
    #         short_touch.iloc[i] = True


    # # === Векторизация касания ATR по Open/Close за lookback_bars ===
    pct = atr_touch_pct / 100
    long_touch  = pd.Series(False, index=df_15min.index)
    short_touch = pd.Series(False, index=df_15min.index)

    for j in range(lookback_bars + 1):          # j = 0 … lookback_bars
        open_shift  = open_.shift(j)
        close_shift = close.shift(j)

        long_touch  |= (
            (abs(open_shift  - low_atr) / channel < pct) |
            (abs(close_shift - low_atr) / channel < pct)
        )

        short_touch |= (
            (abs(open_shift  - up_atr) / channel < pct) |
            (abs(close_shift - up_atr) / channel < pct)
        )

    long_sig = long_touch & (bb_dir == 'up') & (adx_col == 'blue') & (adx > adx_min)
    short_sig = short_touch & (bb_dir == 'down') & (adx_col == 'red') & (adx > adx_min)

    adx_sum_min = (adx < adx_min).sum()
    adx_sum_max = (adx > adx_min).sum()
    #print(f"📊 IterN: {NINT} | 📈 ADX>{adx_min}: {adx_sum_max} | 📉 ADX≤{adx_min}: {adx_sum_min}")
    print(f"📊 IterN: {NINT} [GS]  S-l-15min :{long_sig.sum()} S-s-15min :{short_sig.sum()} | adx={adx.iloc[-1]} adx_col={adx_col.iloc[-1]} | \
  adx_p={adx_period} | 📈 ADX>{adx_min}: {adx_sum_max} | 📉 ADX≤{adx_min}: {adx_sum_min} | bb_l={bb_length} bb_mult={bb_mult}  \n" )
       #    f"BB:{bb_dir.tail(20).to_string(index=False).replace('\n', ' | ')}  \n" ) 
    #print(f"type: {type(long_touch)}, dtype: {long_touch.dtype}, индекс совпадает: {long_touch.index.equals(df_15min.index)} \n ")
    #print(f"первые 10 значений: {long_touch.head(10)} \n " ) 
    #print(f"adx_per={adx_period} atr_touch_pct={atr_touch_pct} bb_length={bb_length} bb_mult={(bb_mult / 100) } lookback_bars={lookback_bars} \n ")
    # print(f"IterN: {NINT} Количество long 15min :{long_sig.sum()} and short 15min :{short_sig.sum()} \n ")

    return long_sig, short_sig


# === Растяжение сигналов с 15м на 1м индекс ===
def stretch_signals_to_minute(df_15min, df_1min, long_sig, short_sig):
    if not isinstance(df_15min.index, pd.DatetimeIndex):
        df_15min.index = pd.to_datetime(df_15min.index)

    if not isinstance(df_1min.index, pd.DatetimeIndex):
        df_1min.index = pd.to_datetime(df_1min.index)
    
    # """Реиндексирует сигналы с 15м таймфрейма на минутный."""
    #print(f"Тип всех колонок long_sig in indic : {long_sig.dtypes}")
    #print(f"chek df15 ", isinstance(df_15min.index, pd.DatetimeIndex))
    #print(f"check df 1min", isinstance(df_1min.index, pd.DatetimeIndex))
    #print(f"check df long", isinstance(long_sig.index, pd.DatetimeIndex))
    #print(f"Тип индекса long_sig in indic : {type(long_sig.index)}")
    long_signal_min = long_sig.reindex(df_1min.index, method='ffill').fillna(False)
    short_signal_min = short_sig.reindex(df_1min.index, method='ffill').fillna(False)
    return long_signal_min, short_signal_min


# === Выгрузка значений индикаторов в CSV ===
# def export_indicators_to_csv(df_15min: pd.DataFrame, adx_period, bb_length, bb_mult, output_file: str = 'indicators_export.csv'):
#     basis, upper, lower = atr_bands(df_15min)
#     bb_dir, bb_line = bb_stops(df_15min,  bb_length, bb_mult)
#     adx, adx_color = adx_histogram(df_15min, adx_period)

#     out = pd.DataFrame({
#         'Datetime': df_15min.index,
#         'ATR_Basis': basis,
#         'ATR_Upper': upper,
#         'ATR_Lower': lower,
#         'BB_Direction': bb_dir,
#         'BB_StopLine': bb_line,
#         'ADX_Value': adx,
#         'ADX_Color': adx_color
#     })
#     out.to_csv(output_file, index=False)
#     print(f"Экспорт индикаторов выполнен: {output_file}")
