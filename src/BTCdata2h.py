# import libraries
import pandas as pd
import cryptocompare
import datetime as dt
from datetime import date, timedelta

def get_BTC_data():
    # Get the API key from the Quantra file located inside the data_modules folder
    cryptocompare_API_key = 'cryptocompare_API_key'

    # Set the API key in the cryptocompare object
    cryptocompare.cryptocompare._set_api_key_parameter(cryptocompare_API_key)

    today = date.today()
    timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
    yesterday = today - timedelta(days = 1)

    # Define the ticker symbol and other details
    ticker_symbol = 'BTC'
    currency = 'USD'
    limit_value = 48
    exchange_name = 'CCCAGG'
    data=cryptocompare.get_historical_price_hour(ticker_symbol, currency, limit=limit_value, exchange=exchange_name, toTs=today)

    # convert the data into data frame
    df = pd.DataFrame.from_dict(data)
    dateHour = []
    for i in df['time']:
        dateHour.append(pd.Timestamp(i, unit='s'))
    df['dateHour'] = dateHour
    df = df[['dateHour','high','low','open','close']]
    df2h = df.groupby(df.dateHour.dt.floor('2h')).mean(numeric_only=True)
    return df2h