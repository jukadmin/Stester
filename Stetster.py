from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import logging
import warnings
from indicators import resample_to_15min, generate_signals, stretch_signals_to_minute
from indicators import export_indicators_to_csv, detect_price_precision
from xlsxwriter import Workbook
import math


class MyStrategy(Strategy):
    # Параметры для оптимизации (при желании)
    stop_loss_pct    = 0.01   # 1 % — начальный стоп‑лосс
    risk_pct    = 0.25   # 25 % — размер депо на сделку
    margin_int    = 1   # 1  — коэфициент маржи
    trail_start_pct  = 0.01   # старт трейлинга (от входа в плюс)
    trail_step_pct   = 0.001  # «шаг» собственного трейлинга (0.1 %)

    def init(self):
        """Инициализируем индикаторы и сигналы"""
        df_1min   = self.data.df                      # минутные данные
        df_15min  = resample_to_15min(df_1min)        # 15‑минутные бары

        # Сигналы long/short на 15‑мин
        long_15, short_15 = generate_signals(df_15min)

        # Растягиваем сигналы на минутный индекс
        self.long_signal_min, self.short_signal_min = stretch_signals_to_minute(
            df_15min, df_1min, long_15, short_15
        )
        self.precision = detect_price_precision(df)

        # для ведения трейлинга
        self.last_stop = None      # текущий stop‑loss (обновляется трейлингом)

    def next(self):
        """Вызывается библиотекой на каждом новом минутном баре"""
        i            = len(self.data) - 1
        price        = float(self.data.Close[i])
        current_time = self.data.index[i]
        # Сигналы на текущую минуту
        long_signal  = self.long_signal_min.get(current_time, False)
        short_signal = self.short_signal_min.get(current_time, False)
        #print(f"Текущая минута: i= {i} время {self.data.index[-1]}, время new {current_time} price {price} long: {long_signal}, short: {short_signal}")

        # === Вход в позицию ===
        if not self.position:
            entry_value = self.equity * self.risk_pct * self.margin_int
            size = int(round(entry_value / price))  # округляем до целого

            if long_signal:
                sl = price * (1 - self.stop_loss_pct)
                sl = round(sl, self.precision)
                self.buy(size=size, sl=sl)
                self.last_stop = sl

            elif short_signal:
                sl = price * (1 + self.stop_loss_pct)
                sl = round(sl, self.precision)
                self.sell(size=size, sl=sl)
                self.last_stop = sl

        # === Кастомный трейлинг ===
        # elif self.position.is_long:
        #     profit_pct = (price - self.position.entry_price) / self.position.entry_price # type: ignore
        #     if profit_pct > self.trail_start_pct:
        #         new_sl = price - price * self.trail_step_pct
        #         if new_sl > self.last_stop:
        #             self.position.update_sl(new_sl) # type: ignore
        #             self.last_stop = new_sl

        # elif self.position.is_short:
        #     profit_pct = (self.position.entry_price - price) / self.position.entry_price # type: ignore
        #     if profit_pct > self.trail_start_pct:
        #         new_sl = price + price * self.trail_step_pct
        #         if new_sl < self.last_stop:
        #             self.position.update_sl(new_sl) # type: ignore
        #             self.last_stop = new_sl


# Настройка логгера
logging.basicConfig(
    filename='main.log',
    filemode='w',  # 'a' — чтобы добавлять, 'w' — чтобы перезаписывать
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Перехват предупреждений в лог
logging.captureWarnings(True)


# Загрузка данных из файла Hystory.csv
csv_file = 'Hystory.csv'
df = pd.read_csv(csv_file, parse_dates=['Date'])
#print(f"Тип индекса df - 1: {type(df.index)}")
#print("df upload - 1 ", df)
df.set_index(['Date'], inplace=True)
#print(f"Тип индекса df - 2: {type(df.index)}")
#print("df upload - 2 ", df)
df = df[(df.index >= '2025-05-01') & (df.index < '2025-05-05')]
#print("df upload - 3 ", df)
df = df.rename(columns=lambda x: x.capitalize())  # Убедимся, что заголовки: Open, High, Low, Close, Volume
#print("df upload - 4 ", df)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
#print("df upload - 5 ", df)
ldf = len(df)


# Запуск бэктеста
#bt = Backtest(df, SmaCross, cash=50_000, commission=0.002)
#stats = bt.run()

bt = Backtest(df, MyStrategy, cash=10000, commission=0.0)
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

# Вывод результатов
#print(stats.keys())
#print(stats)
print("Длинна загруженных данных", ldf)
# Пример: только ключевые метрики
print(stats[['Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]', 'Equity Peak [$]',
             'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration', 'Avg. Drawdown Duration', '# Trades',
             'Win Rate [%]', 'Best Trade [%]',
             'Worst Trade [%]', 'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration', 'Profit Factor',
             'Expectancy [%]']])
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


bt.plot(resample='15min', open_browser=False) # type: ignore