import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time

plt.rcParams['axes.unicode_minus'] = False

# ЗАГРУЗКА ДАННЫХ
def get_moex_candles(ticker, start_date, end_date, interval=24):
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
    
    all_candles = []
    start = 0
    page_size = 100
    
    print(f"\nЗагружаю данные для {ticker}...")
    
    while True:
        params = {
            'from': start_date,
            'till': end_date,
            'interval': interval,
            'start': start,
            'iss.meta': 'off',
            'candles.columns': 'begin,open,high,low,close,volume'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'candles' not in data or 'data' not in data['candles']:
                print(f"Ошибка: Неверный формат ответа от сервера для {ticker}")
                return pd.DataFrame()
            
            candles_data = data['candles']['data']
            
            if not candles_data:
                break
                
            all_candles.extend(candles_data)
            
            if len(candles_data) < page_size:
                break
                
            start += page_size
            time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к MOEX ISS: {e}")
            return pd.DataFrame()
    
    if not all_candles:
        print(f"Данные для {ticker} не найдены. Проверьте правильность тикера.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['begin', 'open', 'high', 'low', 'close', 'volume'])
    df['begin'] = pd.to_datetime(df['begin'].str[:10])
    df.set_index('begin', inplace=True)
    
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
    
    df = df.dropna(subset=['close'])
    df = df.sort_index()
    
    print("Загрузка данных завершена!")
    return df

# ДОБАВЛЕНИЕ СИГНАЛОВ СТРАТЕГИИ ХЭЛЛОУИНА
def get_halloween_signal(date):
    month = date.month
    
    # Зимний период (ноябрь - апрель) - в рынке
    if month in [11, 12, 1, 2, 3, 4]:
        return 1  # Покупаем/держим
    # Летний период (май - октябрь) - вне рынка
    else:
        return 0  # Продаем/не покупаем

def add_halloween_signal(df):
    df['HalloweenSignal'] = df.index.to_series().apply(get_halloween_signal)
    
    df['Signal'] = 0
    df.loc[(df['HalloweenSignal'] == 1) & (df['HalloweenSignal'].shift(1) == 0), 'Signal'] = 1 # Покупаем
    df.loc[(df['HalloweenSignal'] == 0) & (df['HalloweenSignal'].shift(1) == 1), 'Signal'] = -1 # Продаем
    
    return df

# ТЕСТИРОВАНИЕ СТРАТЕГИИ
def backtest_strategy(df, ticker="SBER", initial_capital=10000):
    print(f"\nТестирование стратегии для {ticker}: ")
    
    position = 0
    capital = initial_capital
    shares = 0
    trades = []
    
    for index, row in df.iterrows():
        price = row['close']
        signal = row['Signal']
        
        if signal == 1 and position == 0:
            shares = capital / price
            capital = 0
            position = 1
            trades.append({'Date': index, 'Type': 'BUY', 'Price': price})
            
        elif signal == -1 and position == 1:
            capital = shares * price
            shares = 0
            position = 0
            trades.append({'Date': index, 'Type': 'SELL', 'Price': price})
    
    if position == 1:
        final_price = df.iloc[-1]['close']
        capital = shares * final_price
        trades.append({'Date': df.index[-1], 'Type': 'CLOSE', 'Price': final_price})
    
    total_return = (capital - initial_capital) / initial_capital * 100
    
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0 and len(trades_df[trades_df['Type'] == 'SELL']) > 0:
        sell_trades = trades_df[trades_df['Type'] == 'SELL']
        buy_trades = trades_df[trades_df['Type'] == 'BUY']
        
        if len(sell_trades) > 0:
            buy_prices = buy_trades['Price'].values[:len(sell_trades)]
            sell_prices = sell_trades['Price'].values
            trade_returns = (sell_prices - buy_prices) / buy_prices * 100
            
            win_rate = (trade_returns > 0).sum() / len(trade_returns) * 100
            avg_return = trade_returns.mean()
            print(f"Процент прибыльных сделок: {win_rate:.2f}%")
            print(f"Средняя доходность на сделку: {avg_return:.2f}%")
        else:
            win_rate, avg_return = 0, 0
            trade_returns = np.array([])
    else:
        win_rate, avg_return = 0, 0
        trade_returns = np.array([])
    
    print(f"Начальный капитал: {initial_capital:.2f} ₽")
    print(f"Конечный капитал: {capital:.2f} ₽")
    print(f"Общая доходность: {total_return:.2f}%")
    print(f"Количество сделок: {len(trades_df[trades_df['Type'] == 'SELL'])}")
    
    return trades_df, total_return, win_rate, avg_return, trade_returns

# ВИЗУАЛИЗАЦИЯ
def plot_results(df, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(df.index, df['close'], label='Цена закрытия', color='black', linewidth=1)
    
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=70, label='Покупка (1 ноября)', zorder=5)
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=70, label='Продажа (1 мая)', zorder=5)
    
    ax1.set_title(f'Стратегия Хэллоуина для {ticker}', fontsize=14)
    ax1.set_ylabel('Цена (₽)')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    halloween_numeric = df['HalloweenSignal'].map({1: 1, 0: 0})
    ax2.fill_between(df.index, 0, halloween_numeric, step="mid", color='green', alpha=0.5, label='В рынке (ноябрь-апрель)')
    ax2.fill_between(df.index, 0, 1 - halloween_numeric, step="mid", color='gray', alpha=0.3, label='Вне рынка (май-октябрь)')
    ax2.set_ylabel('Позиция')
    ax2.set_xlabel('Дата')
    ax2.legend(loc='best')
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()
    plt.show()

# СБОР СТАТИСТИКИ ПО ГРУППЕ АКЦИЙ
def collect_statistics(tickers, start_date, end_date):
    all_trades_returns = []
    
    for ticker in tickers:
        df = get_moex_candles(ticker, start_date, end_date)
        if df.empty:
            print(f"Не удалось загрузить данные для {ticker}. Пропускаем.")
            continue
            
        df = add_halloween_signal(df)
        trades_df, total_return, win_rate, avg_return, trade_returns = backtest_strategy(df, ticker)
        #plot_results(df, ticker)

        all_trades_returns.extend(trade_returns)
    
    return np.array(all_trades_returns)

# АНАЛИЗ И ВЫВОД
def analyze_results(all_returns):   
    if len(all_returns) > 0:
        mean_return = np.mean(all_returns)
        positive_share = (all_returns > 0).sum() / len(all_returns) * 100
        std_return = np.std(all_returns)
        
        print("\nСтатистика по всем сделкам: ")
        print(f"Всего сделок по всем акциям: {len(all_returns)}")
        print(f"Средняя доходность сделки: {mean_return:.2f}%")
        print(f"Доля прибыльных сделок: {positive_share:.2f}%")
        
        # Гистограмма доходностей
        plt.figure(figsize=(10, 6))
        plt.hist(all_returns, bins=20, edgecolor='black', color='skyblue')
        plt.axvline(x=0, color='red', linestyle='-')
        plt.axvline(x=mean_return, color='green', linestyle='-', label=f'Средняя доходность ({mean_return:.2f}%)')
        plt.title('Распределение доходностей всех сделок по всем акциям')
        plt.xlabel('Доходность сделки (%)')
        plt.ylabel('Количество сделок')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print("Прогноз на следующую сделку: ")
        print(f"Стандартное отклонение: {std_return:.2f}%")
        print(f"Вероятность положительной доходности: {positive_share:.1f}%")
        print(f"95% доверительный интервал: [{mean_return - 1.96*std_return:.2f}%, {mean_return + 1.96*std_return:.2f}%]")

    else:
        print("Недостаточно данных для анализа.")
    

# ОСНОВНОЙ ЗАПУСК
MAIN_TICKER = "SBER"
    
END_DATE = datetime.now()
START_DATE = END_DATE - relativedelta(years=5)
START_DATE_STR = START_DATE.strftime('%Y-%m-%d')
END_DATE_STR = END_DATE.strftime('%Y-%m-%d')
    
print(f"Период анализа: с {START_DATE_STR} по {END_DATE_STR}")
    
main_df = get_moex_candles(MAIN_TICKER, START_DATE_STR, END_DATE_STR)
    
if not main_df.empty:
    main_df = add_halloween_signal(main_df)
    trades, total_ret, win_r, avg_r, rets = backtest_strategy(main_df, MAIN_TICKER)
    plot_results(main_df, MAIN_TICKER)


    print("\nСбор статистики для других акций для проверки гипотезы")        
    TICKERS_TO_TEST = ["GAZP", "LKOH", "ROSN", "VTBR", "TATN"
        
        ]
    all_returns_array = collect_statistics(TICKERS_TO_TEST, START_DATE_STR, END_DATE_STR)
    analyze_results(all_returns_array)

    '''
    "NVTK", "MTSS", "CHMF", "SNGS", "GMKN", 
        "ALRS", "PLZL", "RUAL", "AFKS", "MOEX", 
        "YNDX", "AFLT", "IRAO", "MAGN", "PHOR", 
        "X5", "T", "BSPB", "TRNFP", "SVCB", 
        "VKCO", "OZON", "PIKK", "MSNG", "UGLD", 
        "POSI", "CNRU", "ENPG", "MDMG", "RENI", 
        "RTKM", "FLOT", "HEAD", "NLMK", "CBOM"
    '''
