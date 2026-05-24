import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import openmeteo_requests
import requests_cache
from retry_requests import retry

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
            'start': start,            # Смещение (для пагинации)
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
            # делаем паузу чтоб нас не заблокали, 10 запросов в секунду безопасно
            time.sleep(0.1)
            
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

# ЗАГРУЗКА ПОГОДНЫХ ДАННЫХ ДЛЯ САНКТ-ПЕТЕРБУРГА
def get_weather_data(latitude, longitude, start_date, end_date):
    print(f"\nЗагружаю погодные данные для Санкт-Петербурга...")
    
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=1)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min"],
        "timezone": "Europe/Moscow"
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        daily = response.Daily()
        daily_temperature_mean = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_max = daily.Variables(1).ValuesAsNumpy()
        daily_temperature_min = daily.Variables(2).ValuesAsNumpy()
        
        daily_dates = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
        
        weather_df = pd.DataFrame({
            'date': daily_dates,
            'temp_mean': daily_temperature_mean,
            'temp_max': daily_temperature_max,
            'temp_min': daily_temperature_min
        })
        weather_df['date'] = pd.to_datetime(weather_df['date'].dt.date)
        weather_df.set_index('date', inplace=True)
        
        print(f"Загрузка погодных данных для Санкт-Петербурга завершена!")
        print(f"Диапазон температур: от {weather_df['temp_mean'].min():.1f}°C до {weather_df['temp_mean'].max():.1f}°C")
        return weather_df
        
    except Exception as e:
        print(f"Ошибка при загрузке погодных данных: {e}")
        return pd.DataFrame()

def add_weather_signal(df, weather_df, temp_threshold=10):
    combined_df = df.join(weather_df, how='left')
    
    # Заполняем пропуски температуры, предыдущим днем (иначе прошлым)
    combined_df['temp_mean'] = combined_df['temp_mean'].ffill().bfill()
    combined_df['WeatherSignal'] = (combined_df['temp_mean'] > temp_threshold).astype(int)

    combined_df['Signal'] = 0
    # Покупка: было холодно, стало тепло
    combined_df.loc[(combined_df['WeatherSignal'] == 1) & (combined_df['WeatherSignal'].shift(1) == 0), 'Signal'] = 1
    # Продажа: было тепло, стало холодно
    combined_df.loc[(combined_df['WeatherSignal'] == 0) & (combined_df['WeatherSignal'].shift(1) == 1), 'Signal'] = -1
    
    # Сохраняем температуру в основной DataFrame
    df['temp_mean'] = combined_df['temp_mean']
    df['WeatherSignal'] = combined_df['WeatherSignal']
    df['Signal'] = combined_df['Signal']
    
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
            temp = row.get('temp_mean', 0)
            trades.append({'Date': index, 'Type': 'BUY', 'Price': price, 'Temperature': temp})
            
        elif signal == -1 and position == 1:
            capital = shares * price
            shares = 0
            position = 0
            temp = row.get('temp_mean', 0)
            trades.append({'Date': index, 'Type': 'SELL', 'Price': price, 'Temperature': temp})
    
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(df.index, df['close'], label='Цена закрытия', color='black', linewidth=1)
    
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=70, label='Покупка (тепло в СПб)', zorder=5)
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=70, label='Продажа (холодно в СПб)', zorder=5)
    
    ax1.set_title(f'Стратегия торговли по погоде в Санкт-Петербурге для {ticker}', fontsize=14)
    ax1.set_ylabel('Цена (₽)')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # температура в Санкт-Петербурге
    if 'temp_mean' in df.columns:
        ax2.plot(df.index, df['temp_mean'], color='orange', linewidth=1)
        ax2.axhline(y=10, color='blue', linestyle='-', label='Порог 10°C')
        ax2.fill_between(df.index, 10, df['temp_mean'], where=(df['temp_mean'] > 10), 
                         color='red', label='Тепло в СПб (>10°C)')
        ax2.fill_between(df.index, 10, df['temp_mean'], where=(df['temp_mean'] <= 10), 
                         color='lightblue', label='Холодно в СПб (≤10°C)')
        ax2.set_ylabel('Температура в Санкт-Петербурге (°C)')
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='-')
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()
    plt.show()

# СБОР СТАТИСТИКИ
def collect_statistics(tickers, start_date, end_date, weather_df):
    all_trades_returns = []
    
    if weather_df.empty:
        print("Не удалось загрузить погодные данные для Санкт-Петербурга")
        return np.array([])
    
    for ticker in tickers:
        df = get_moex_candles(ticker, start_date, end_date)
        if df.empty:
            print(f"Не удалось загрузить данные для {ticker}. Пропускаем.")
            continue
        
        df = add_weather_signal(df, weather_df, temp_threshold=10)
        trades_df, total_return, win_rate, avg_return, trade_returns = backtest_strategy(df, ticker)
        # plot_results(df, ticker)
        
        all_trades_returns.extend(trade_returns)
    
    return np.array(all_trades_returns)

# АНАЛИЗ РЕЗУЛЬТАТОВ
def analyze_results(all_returns):
    if len(all_returns) > 0:
        mean_return = np.mean(all_returns)
        positive_share = (all_returns > 0).sum() / len(all_returns) * 100
        
        print("Статистика по всем сделкам:\n")
        print(f"Всего сделок по всем акциям: {len(all_returns)}")
        print(f"Средняя доходность сделки: {mean_return:.2f}%")
        print(f"Доля прибыльных сделок: {positive_share:.2f}%")
        
        # Гистограмма доходностей
        fig, (ax1) = plt.subplots(figsize=(10, 6))
        
        ax1.hist(all_returns, bins=20, edgecolor='black', color='skyblue')
        ax1.axvline(x=0, color='red', linestyle='-', linewidth=1, label='Ноль')
        ax1.axvline(x=mean_return, color='green', linestyle='-', linewidth=1, 
                   label=f'Средняя ({mean_return:.2f}%)')
        ax1.set_title('Распределение доходностей сделок (погода СПб)')
        ax1.set_xlabel('Доходность сделки (%)')
        ax1.set_ylabel('Количество сделок')
        ax1.legend()
        ax1.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        
        print("\nВывод для Санкт-Петербурга: Стратегия не работает")
            
    else:
        print("Недостаточно данных для анализа.")

MAIN_TICKER = "SBER"
    
END_DATE = datetime.now()
START_DATE = END_DATE - relativedelta(years=5)
START_DATE_STR = START_DATE.strftime('%Y-%m-%d')
END_DATE_STR = END_DATE.strftime('%Y-%m-%d')
    
print(f"Период анализа: с {START_DATE_STR} по {END_DATE_STR}")

SPB_LAT = 59.9311
SPB_LON = 30.3609
    
weather_df = get_weather_data(SPB_LAT, SPB_LON, START_DATE_STR, END_DATE_STR)
    
if not weather_df.empty: 
    main_df = get_moex_candles(MAIN_TICKER, START_DATE_STR, END_DATE_STR)
        
    if not main_df.empty:
        main_df = add_weather_signal(main_df, weather_df, temp_threshold=10)
        trades, total_ret, win_r, avg_r, rets = backtest_strategy(main_df, MAIN_TICKER)
        plot_results(main_df, MAIN_TICKER)
            
        print("\nСбор статистики для других акций для проверки гипотезы")
            
        TICKERS_TO_TEST = ["GAZP", "LKOH", "ROSN", "VTBR", "TATN",
        "NVTK", "MTSS", "CHMF", "SNGS", "GMKN", 
        "ALRS", "PLZL", "RUAL", "AFKS", "MOEX", 
        "YNDX", "AFLT", "IRAO", "MAGN", "PHOR", 
        "X5", "T", "BSPB", "TRNFP", "SVCB", 
        "VKCO", "OZON", "PIKK", "MSNG", "UGLD", 
        "POSI", "CNRU", "ENPG", "MDMG", "RENI", 
        "RTKM", "FLOT", "HEAD", "NLMK", "CBOM"
        ]
        all_returns_array = collect_statistics(TICKERS_TO_TEST, START_DATE_STR, END_DATE_STR,weather_df)
        analyze_results(all_returns_array)
    else:
        print("Не удалось загрузить данные.")
else:
    print("Не удалось загрузить погодные данные для Санкт-Петербурга.")
