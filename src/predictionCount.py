import pandas as pd
import datetime as dt
from datetime import date, timedelta,datetime
import numpy as np
from io import StringIO
import os
import awswrangler as wr

def prediction_count():
    today = date.today()
    timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
    yesterday = today - timedelta(days = 1)
    yest = yesterday.strftime("%Y-%m-%d")
    df_pred = pd.read_csv(f"/home/ubuntu/Bert/df_predicted.csv",encoding="ISO-8859-1")
    dateHour = []
    for i in df_pred['created_utc']:
        dateHour.append(pd.Timestamp(i, unit='s'))
    df_pred['dateHour'] = dateHour
    df_pred = df_pred[['dateHour','prediction']]
    df_pred['prediction']=df_pred['prediction'].map({2:'Positive',1:'Neutral',0:'Negative'})
    df_pred = df_pred.set_index('dateHour')
    y = pd.get_dummies(df_pred.prediction)
    y = y.reset_index()
    df2h = y.groupby(y.dateHour.dt.floor('2H')).sum(numeric_only=True)
    df2h['redditTotal'] = df2h['Positive'] + df2h['Neutral'] + df2h['Negative']
    df2h.to_csv(f"/home/ubuntu/Bert/sentiCount_2h.csv")
    return df2h
