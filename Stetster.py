from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import logging, time
import warnings
from indicators import resample_to_15min, generate_signals, stretch_signals_to_minute
from indicators import detect_price_precision # export_indicators_to_csv, 
from xlsxwriter import Workbook
import numpy as np

start_time = time.time()
opti = True
NINT = 0

class MyStrategy(Strategy ):
    # Параметры для оптимизации (при желании)
    stop_loss_pct = 1
    #stop_loss_pct    = 0.01   # 1 % — начальный стоп‑лосс
    risk_pct    = 48   # 25 % — размер депо на сделку
    margin_int    = 1   # 1  — коэфициент маржи
    trail_start_pct = 1
    #trail_start_pct  = 0.01   # старт трейлинга (от входа в плюс)
    trail_step_pct = 1  # !!! лишний 0. !!! 1 - понимается как 0.1 %
    #trail_step_pct   = 0.001  # «шаг» собственного трейлинга (0.1 %)
    adx_period = 12  # 14  — Период расчета ADX
    atr_touch_pct = 18  # 5 %  — ATR процент касания.
    bb_length = 10  # 20  Длина Боллинджера (optim 10-30 )
    bb_mult = 10 #  1 Множитель BB *100 (optim 0.5 – 2.5 (шаг 0.25 )
    lookback_bars = 18  #  количество баров ATR

    def init(self):
        super().init()
        global NINT
        NINT = NINT + 1
        
        #self.stop_loss_pct = stop_loss_pct
        """Инициализируем индикаторы и сигналы"""
        df_1min   = self.data.df                      # минутные данные
        df_15min  = resample_to_15min(df_1min)        # 15‑минутные бары

        # Сигналы long/short на 15‑мин
        long_15, short_15 = generate_signals(df_15min, self.adx_period, self.atr_touch_pct, self.bb_length, self.bb_mult, self.lookback_bars, NINT=NINT)
        # print(f" IterNum= {NINT}  adx_period={self.adx_period} atr_touch_pct={self.atr_touch_pct} lookback_bars={self.lookback_bars} \n")
        # Растягиваем сигналы на минутный индекс
        self.long_signal_min, self.short_signal_min = stretch_signals_to_minute(
            df_15min, df_1min, long_15, short_15
        )
        self.precision = detect_price_precision(df)

        #export_indicators_to_csv(df_15min, self.adx_period)
        # для ведения трейлинга
        self.last_stop = None      # текущий stop‑loss (обновляется трейлингом)

    def next(self):
        super().next()
        """Вызывается библиотекой на каждом новом минутном баре"""
        i            = len(self.data) - 1
        price        = float(self.data.Close[i])
        current_time = self.data.index[i]
        # Сигналы на текущую минуту
        long_signal  = self.long_signal_min.get(current_time, False)
        short_signal = self.short_signal_min.get(current_time, False)
        #print(f"Текущая минута: i= {i} время {self.data.index[-1]}, время new {current_time} price {price} long: {long_signal}, short: {short_signal}")
        # print( )
        # === Вход в позицию ===
        if not self.position:
            entry_value = self.equity * (self.risk_pct / 100) * self.margin_int
            size = int(round(entry_value / price))  # округляем до целого
            self.trail_last = 0

            if long_signal:
                sl = price * (1 - (self.stop_loss_pct / 1000))
                sl = round(sl, self.precision)
                self.buy(size=size)
                self.last_stop = sl
                self.entry_price = price
                #print(f"Long: Price={price} Position={self.entry_price} SL={sl}")
                msg = f"Long: Price={price} Position={self.entry_price} SL={sl}"; trail_log.append(msg)

            elif short_signal:
                sl = price * (1 + (self.stop_loss_pct / 1000))
                sl = round(sl, self.precision)
                self.sell(size=size)
                self.last_stop = sl
                self.entry_price = price
                #print(f"Short: Price={price} Position={self.entry_price} SL={sl}")
                msg = f"Short: Price={price} Position={self.entry_price} SL={sl}"; trail_log.append(msg)

        # === Кастомный трейлинг ===
        elif self.position.is_long:
            profit_pct = (price - self.entry_price) / self.entry_price
            profit_pct = round(profit_pct, self.precision)
            if profit_pct > (self.trail_start_pct / 100) :  # включение трейлинга
                new_sl = price - price * (self.trail_start_pct / 100 )
                new_sl = round(new_sl, self.precision)
                new_sl_pct = (( new_sl - self.last_stop ) / self.last_stop ) * 100  # type: ignore
                new_sl_pct = round(new_sl_pct, self.precision)
                new_step_pct = self.trail_step_pct / 10
                if new_sl_pct > new_step_pct:
                    #self.position.update_sl(new_sl) # type: ignore
                    #print(f"Last_stop={self.last_stop} new_sl={new_sl} profit_pct={round((profit_pct * 100), 2)}%  new_sl_pct={round(new_sl_pct , 2)}% ")
                    msg = f"Last_stop={self.last_stop} new_sl={new_sl} profit_pct={round((profit_pct * 100), 2)}%  new_sl_pct={round(new_sl_pct , 2)}% "; trail_log.append(msg)
                    self.last_stop = new_sl
            if price <= self.last_stop: # type: ignore
                #print(f"Position {self.position}")
                self.position.close()
                #print(f"Position {self.position}")
                    

        elif self.position.is_short:
            profit_pct = (self.entry_price - price) / self.entry_price
            profit_pct = round(profit_pct, self.precision)
            if profit_pct > (self.trail_start_pct / 100 ) :  # включение трейлинга
                new_sl = price + price * (self.trail_start_pct / 100 )
                new_sl = round(new_sl, self.precision)
                new_sl_pct = (( self.last_stop - new_sl ) / self.last_stop ) * 100  # type: ignore
                new_sl_pct = round(new_sl_pct, self.precision)
                new_step_pct = self.trail_step_pct / 10
                if new_sl_pct > new_step_pct:
                    #self.position.update_sl(new_sl) # type: ignore
                    #print(f"Last_stop={self.last_stop} new_sl={new_sl} profit_pct={round((profit_pct * 100), 2)}%  new_sl_pct={round(new_sl_pct , 2)}% ")
                    msg = f"Last_stop={self.last_stop} new_sl={new_sl} profit_pct={round((profit_pct * 100), 2)}%  new_sl_pct={round(new_sl_pct , 2)}% "; trail_log.append(msg)
                    self.last_stop = new_sl
            if price >= self.last_stop: # type: ignore
                #print(f"Position {self.position}")
                self.position.close()
                #print(f"Position {self.position}")




# Настройка логгера
logging.basicConfig(
    filename='main.log',
    filemode='w',  # 'a' — чтобы добавлять, 'w' — чтобы перезаписывать
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Буфер для накопления логов трейлинга ===
trail_log = []

# Перехват предупреждений в лог
logging.captureWarnings(True)


# Загрузка данных из файла Hystory.csv
csv_file = 'Hystory.csv'
df = pd.read_csv(csv_file, parse_dates=['Date'])
df.set_index(['Date'], inplace=True)
#df = df[(df.index >= '2025-05-01') & (df.index < '2025-05-05')]
df = df.rename(columns=lambda x: x.capitalize())  # Убедимся, что заголовки: Open, High, Low, Close, Volume
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
ldf = len(df)


if opti == False:
    bt = Backtest(df, MyStrategy, cash=200, commission=0.0)
    stats = bt.run()
    # ==== Формируем DataFrame сделок ====
    tradesx = stats._trades.copy()
    if not tradesx.empty:
        # 1) Направление сделки
        tradesx['Direction'] = tradesx['Size'].apply(lambda x: 'long' if x > 0 else 'short')
        # 2) Приведение времени и длительности к человекочитаемому виду
        tradesx['Duration']  = tradesx['Duration'].astype(str)
        tradesx['EntryTime'] = tradesx['EntryTime'].dt.strftime('%Y-%m-%d %H:%M')
        tradesx['ExitTime']  = tradesx['ExitTime'].dt.strftime('%Y-%m-%d %H:%M')
        # 3) Порядок колонок
        cols = [
            'EntryTime', 'EntryBar', 'EntryPrice', 'Direction', 'Size',
            'SL', 'TP', 'ExitTime', 'ExitBar', 'ExitPrice',
            'Duration', 'PnL', 'ReturnPct', 'Tag'
        ]
        tradesx = tradesx[cols]
        print(tradesx)
        # ==== Экспорт в Excel с шириной колонок EntryTime и ExitTime = 16 сим ====
        with pd.ExcelWriter('Trades.xlsx', engine='xlsxwriter') as writer:
            tradesx.to_excel(writer, sheet_name='Trades', index=False)
            # Доступ к листу для настройки ширины столбцов
            worksheet = writer.sheets['Trades']
            entry_col = tradesx.columns.get_loc('EntryTime')  # индекс колонки
            exit_col  = tradesx.columns.get_loc('ExitTime')
            duration_col  = tradesx.columns.get_loc('Duration')
            # Ширина = 16 символов
            worksheet.set_column(entry_col, entry_col, 16)
            worksheet.set_column(exit_col,  exit_col,  16)
            worksheet.set_column(duration_col,  duration_col,  16)
        print("История сделок экспортирована в Trades.xlsx")
    else:
        print("❗ Сделок не было — экспорт не выполнен.")
    print(stats[['Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]', 'Equity Peak [$]',
             'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration', 'Avg. Drawdown Duration', '# Trades',
             'Win Rate [%]', 'Best Trade [%]', 'Return [%]',
             'Worst Trade [%]', 'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration', 'Profit Factor',
             'Expectancy [%]']])
    bt.plot(resample='15min', open_browser=False) # type: ignore

if opti == True:
    bt = Backtest(df, MyStrategy, cash=1000000, commission=0.0)
    #heatmap(bt, p='stop_loss_pct' ) # values='Return [%]
    stats = bt.optimize(adx_period=range(10, 16, 2), 
                        atr_touch_pct=range(14, 20, 2), stop_loss_pct=range(4, 8, 1), lookback_bars=range(16, 22, 2),
                        maximize='Equity Final [$]', 
                        return_heatmap=False) # max_tries=200,  random_state=0, constraint=lambda p: p.stop_loss_pct < 0.02,
                        #  trail_start_pct=range(1, 2, 1),  
                        # trail_step_pct=range(1, 2, 1),  
                        #  risk_pct=range(40, 50, 2), bb_length=range(10, 16, 2),  bb_mult=range(10, 50, 10),
    print(stats)
    new_st = stats._strategy  # type: ignore
    #new_st = new_st.to_string()
    print(new_st)
    print(f"Iterations= {NINT}")
    #print(stats['_strategy'])
    #with open("Stat.log", "w", encoding="utf-8") as sp:
    #    sp.write(str(stats))
    #res = pd.DataFrame(stats.to_dict())
    #res.to_excel("Stat.xlsx", index=False)
    #stats.to_frame().to_excel("Stat.xlsx")

# Вывод результатов
# === Сохраняем накопленные логи в файл ===
with open("Trailing.log", "w", encoding="utf-8") as fp:
    fp.write("\n".join(trail_log))
print("Log saved to Trailing.log")
print("Длинна загруженных данных", ldf)
elapsed = time.time() - start_time

hours, rem = divmod(elapsed, 3600)
minutes, seconds = divmod(rem, 60)

if hours >= 1:
    print(f"⏱ Время выполнения: {int(hours)}ч {int(minutes)}м {seconds:.2f}с")
elif minutes >= 1:
    print(f"⏱ Время выполнения: {int(minutes)}м {seconds:.2f}с")
else:
    print(f"⏱ Время выполнения: {seconds:.2f} секунд")


# Пример: только ключевые метрики

# Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]',
#        'Equity Peak [$]', 'Commissions [$]', 'Return [%]',
#        'Buy & Hold Return [%]', 'Return (Ann.) [%]', 'Volatility (Ann.) [%]',
#        'CAGR [%]', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
#        'Alpha [%]', 'Beta', 'Max. Drawdown [%]', 'Avg. Drawdown [%]',
#        'Max. Drawdown Duration', 'Avg. Drawdown Duration', '# Trades',
#        'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]',
#        'Max. Trade Duration', 'Avg. Trade Duration', 'Profit Factor',
#        'Expectancy [%]', 'SQN', 'Kelly Criterion', '_strategy',
#        '_equity_curve', '_trades


