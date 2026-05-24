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
            
            # проверка на последнюю страницу
            if len(candles_data) < page_size:
                break
                
            start += page_size
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к MOEX ISS: {e}")
            return pd.DataFrame()
    
    if not all_candles:
        print(f"Данные для {ticker} не найдены. Проверьте правильность тикера.")
        return pd.DataFrame()
    
    # Преобразуем в DataFrame
    df = pd.DataFrame(all_candles, columns=['begin', 'open', 'high', 'low', 'close', 'volume'])
    df['begin'] = pd.to_datetime(df['begin'].str[:10]) # берем только дату
    df.set_index('begin', inplace=True) # делаем индексом строк столбец даты и не создаем новую таблицу, а меняем старую
    
    # Приводим числовые данные к float/int
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int) # errors='coerce' приошибке делаем NaN, NaN меняем на 0 fillna(0)
    
    # Удаляем строки с NaN в цене закрытия
    df = df.dropna(subset=['close'])
    
    # Сортируем по дате (на всякий случай)
    df = df.sort_index()
    
    print("Загрузка данных завершена!")
    return df

# РАСЧЕТ ФАЗ ЛУНЫ
def get_moon_phase(date):
    # переводим обычную дату в юлианскую
    year = date.year
    month = date.month
    day = date.day
    
    if month < 3:
        year -= 1
        month += 12
    
    A = year // 100
    B = A // 4
    C = 2 - A + B
    E = int(365.25 * (year + 4716))
    F = int(30.6001 * (month + 1))
    julian_day = C + day + E + F - 1524.5
    
    lunar_cycle = 29.5   # Лунный цикл в днях
    known_new_moon = 2451549.5  # 6 января 2000, полдень, новолуние, наша точка отсчета
    
    phase_days = (julian_day - known_new_moon) % lunar_cycle # Сколько прошло дней от новолуния
    
    if phase_days < lunar_cycle / 2:
        return 'Waxing'  # Растущая
    else:
        return 'Waning'  # Убывающая

def add_moon_phase_column(df):
    df['MoonPhase'] = df.index.to_series().apply(get_moon_phase)
    return df

# ТЕСТИРОВАНИЕ СТРАТЕГИИ 
def backtest_strategy(df, ticker="SBER", initial_capital=10000):
    print(f"\nТестирование стратегии для {ticker}: ")
    
    # Определяем сигналы
    df['Signal'] = 0
    df.loc[(df['MoonPhase'] == 'Waxing') & (df['MoonPhase'].shift(1) == 'Waning'), 'Signal'] = 1  # Покупка
    df.loc[(df['MoonPhase'] == 'Waning') & (df['MoonPhase'].shift(1) == 'Waxing'), 'Signal'] = -1  # Продажа
    
    # Симуляция сделок
    position = 0  # 0 = нет акций, 1 = куплена акция
    capital = initial_capital
    shares = 0  # Количество акций в портфеле
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
    
    # Статистика по сделкам
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0 and len(trades_df[trades_df['Type'] == 'SELL']) > 0:
        sell_trades = trades_df[trades_df['Type'] == 'SELL']
        buy_trades = trades_df[trades_df['Type'] == 'BUY']

        if len(sell_trades) > 0:
            # берем покупок столькоже сколько и продаж (ведь CLOSE мы не включили)
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
    
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=70, label='Сигнал на покупку', zorder=5)
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=70, label='Сигнал на продажу', zorder=5)
    
    ax1.set_title(f'Стратегия торговли по фазам Луны для {ticker}', fontsize=14)
    ax1.set_ylabel('Цена (₽)')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    moon_numeric = df['MoonPhase'].map({'Waxing': 1, 'Waning': 0})
    ax2.fill_between(df.index, 0, moon_numeric, step="mid", color='gold', label='Растущая Луна')
    ax2.fill_between(df.index, 0, 1 - moon_numeric, step="mid", color='slategray', label='Убывающая Луна')
    ax2.set_ylabel('Фаза Луны')
    ax2.set_xlabel('Дата')
    ax2.legend(loc='best')
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.show()

# СБОР СТАТИСТИКИ ПО ГРУППЕ АКЦИЙ
def collect_statistics(tickers, start_date, end_date):
    all_trades_returns = []
    
    for ticker in tickers:
        df = get_moex_candles(ticker, start_date, end_date)
        if df.empty:
            print(f"Не удалось загрузить данные для {ticker}. Пропускаем.")
            continue
            
        df = add_moon_phase_column(df)
        trades_df, total_return, win_rate, avg_return, trade_returns = backtest_strategy(df,ticker)
        # plot_results(df,ticker)

        all_trades_returns.extend(trade_returns)
    
    return np.array(all_trades_returns)

# АНАЛИЗ И ВЫВОД
def analyze_results(all_returns):   
    if len(all_returns) > 0:
        mean_return = np.mean(all_returns)
        positive_share = (all_returns > 0).sum() / len(all_returns) * 100
        
        print("\nСтатистика по всем сделкам: ")
        print(f"Всего сделок по всем акциям: {len(all_returns)}")
        print(f"Средняя доходность сделки: {mean_return:.2f}%")
        print(f"Доля прибыльных сделок: {positive_share:.2f}%")
        
        # Гистограмма доходностей
        plt.figure(figsize=(10, 6))
        plt.hist(all_returns, bins=20, edgecolor='black', color='skyblue') #bins=20 количество столбцов
        plt.axvline(x=0, color='red', linestyle='-')
        plt.axvline(x=mean_return, color='green', linestyle='-', label=f'Средняя доходность ({mean_return:.2f}%)')
        plt.title('Распределение доходностей всех сделок по всем акциям')
        plt.xlabel('Доходность сделки (%)')
        plt.ylabel('Количество сделок')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("Недостаточно данных для анализа.")
    
    # ВЫВОД
    print("\nВывод: Стратегия не работает")


MAIN_TICKER = "SBER"
    
END_DATE = datetime.now()
START_DATE = END_DATE - relativedelta(years=5)
START_DATE_STR = START_DATE.strftime('%Y-%m-%d')
END_DATE_STR = END_DATE.strftime('%Y-%m-%d')
    
print(f"Период анализа: с {START_DATE_STR} по {END_DATE_STR}")
    
main_df = get_moex_candles(MAIN_TICKER, START_DATE_STR, END_DATE_STR)
    
if not main_df.empty:
    main_df = add_moon_phase_column(main_df)
    trades, total_ret, win_r, avg_r, rets = backtest_strategy(main_df)
    plot_results(main_df, MAIN_TICKER)

    print("\n")
    print("\nСбор статистики по первому эшелону")
        
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
