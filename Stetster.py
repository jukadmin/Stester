from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import logging
import warnings
from indicators import resample_to_15min, generate_signals, stretch_signals_to_minute
from indicators import export_indicators_to_csv



class MyStrategy(Strategy):
    stop_loss_pct = 0.01   # начальный стоп 1%
    trail_start_pct = 0.01 # начать трейлинг после 1%
    trail_step_pct = 0.001 # шаг трейлинга 0.1%

    def init(self):
        df_15min = resample_to_15min(self.data.df)
        export_indicators_to_csv(df_15min)
        self.long_signal, self.short_signal = generate_signals(df_15min)
        #print(f"Тип индекса df: {type(self.data.df.index)}")
        #print(f"Тип индекса df_15min: {type(df_15min.index)}")
        self.long_signal_min, self.short_signal_min = stretch_signals_to_minute(df_15min, self.data.df, self.long_signal, self.short_signal)
        #print("init short сигнал true .str() :", self.short_signal_min[self.short_signal_min].to_string())
        print("init short сигнал true .sum() :", self.short_signal_min[self.short_signal_min].sum())
        self.last_entry_price = None
        self.trailing_stop = None

    def next(self):
        i = len(self.data.Close) - 1
        # print("i =" ,i)
        current_time = self.data.index[-1]  # индекс текущего бара
        long = self.long_signal_min.get(current_time, False)
        short = self.short_signal_min.get(current_time, False)
        
        # if short == True :
        #     print(f"short = {short}"  )
        #     print(f"Текущая минута: {self.data.index[-1]}, long: {long}, short: {short}")
        # if long == True :
        #     print(f"long = {long}"  )
        #     print(f"Текущая минута: {self.data.index[-1]}, long: {long}, short: {short}")

        if self.position:
            price_now = self.data.Close[i]
            if self.position.is_long:
                profit_pct = (price_now - self.last_entry_price) / self.last_entry_price
                if profit_pct > self.trail_start_pct:
                    new_trail = price_now - price_now * self.trail_step_pct
                    if self.trailing_stop is None or new_trail > self.trailing_stop:
                        self.trailing_stop = new_trail
                if self.trailing_stop and price_now < self.trailing_stop:
                    self.position.close()
            elif self.position.is_short:
                profit_pct = (self.last_entry_price - price_now) / self.last_entry_price
                if profit_pct > self.trail_start_pct:
                    new_trail = price_now + price_now * self.trail_step_pct
                    if self.trailing_stop is None or new_trail < self.trailing_stop:
                        self.trailing_stop = new_trail
                if self.trailing_stop and price_now > self.trailing_stop:
                    self.position.close()
            return

        #contract_precision = 0
        if long:
            print("Long cmd = ", long)
            size = self.equity * 0.4  / self.data.Close[i]
            size = round(size)
            #print(f"Size long  = {size}")
            self.buy(size=size)
            self.last_entry_price = self.data.Close[i]
            self.trailing_stop = self.last_entry_price - self.last_entry_price * self.stop_loss_pct
        elif short:
            print("Short cmd = ", i, short, self.data.Open[i], self.data.High[i], self.data.Low[i], self.data.Close[i])
            size = self.equity * 0.4  / self.data.Close[i]
            size = round(size)
            #print(f"Size short  = {size}")
            self.sell(size=size)
            self.last_entry_price = self.data.Close[i]
            self.trailing_stop = self.last_entry_price + self.last_entry_price * self.stop_loss_pct

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