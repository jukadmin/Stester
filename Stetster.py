from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import logging
import warnings
from indicators import resample_to_15min, generate_signals
from indicators import export_indicators_to_csv



class MyStrategy(Strategy):
    stop_loss_pct = 0.01   # начальный стоп 1%
    trail_start_pct = 0.01 # начать трейлинг после 1%
    trail_step_pct = 0.001 # шаг трейлинга 0.1%

    def init(self):
        df_15min = resample_to_15min(self.data.df)
        export_indicators_to_csv(df_15min)
        self.long_signal, self.short_signal = generate_signals(df_15min)
        self.last_entry_price = None
        self.trailing_stop = None

    def next(self):
        i = len(self.data.Close) - 1
        long = self.long_signal.iloc[-1]
        short = self.short_signal.iloc[-1]

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

        if long:
            size = self.equity * 0.4 * 10 / self.data.Close[i]
            self.buy(size=size)
            self.last_entry_price = self.data.Close[i]
            self.trailing_stop = self.last_entry_price - self.last_entry_price * self.stop_loss_pct
        elif short:
            size = self.equity * 0.4 * 10 / self.data.Close[i]
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
df.set_index('Date', inplace=True)
df = df.rename(columns=lambda x: x.capitalize())  # Убедимся, что заголовки: Open, High, Low, Close, Volume
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
ldf = len(df)

# Стандартная встроенная стратегия.
class SmaCross(Strategy):
    def init(self):
        # Инициализация индикаторов SMA
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(10).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(20).mean(), self.data.Close)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy(size=0.2)  # 20% от доступного капитала
        elif crossover(self.sma2, self.sma1):
            self.sell(size=0.2)

# Запуск бэктеста
#bt = Backtest(df, SmaCross, cash=50_000, commission=0.002)
#stats = bt.run()

bt = Backtest(df, MyStrategy, cash=10000, commission=0.002)
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


bt.plot(resample='15min')