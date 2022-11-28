from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,accuracy_score
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import LSTM
import tensorflow as tf
import numpy as np
from datetime import date, timedelta,datetime
from io import StringIO
import os
import awswrangler as wr
import datetime
import re
import kaleido
import plotly
import plotly.graph_objects as go
import plotly.express as px

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# Here is created input columns which are (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# Here is created output/forecast column which are (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# def create_model():
#     model = Sequential()
#     model.add(LSTM(128,activation = 'relu', input_shape=(train2X.shape[1], train2X.shape[2]),return_sequences = True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(64, activation = 'relu',return_sequences = True))
#     model.add(Dropout(0.2))
#     model.add(Dense(1))
#     #model.add(Activation("relu"))#linear
#     opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.00005)
#     model.compile(loss='mae', optimizer=opt,metrics=[tf.keras.metrics.MeanSquaredError()])
#     return model

def predictPrice():
    today = date.today()
    timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
    yesterday = today - timedelta(days = 1)
    yest = yesterday.strftime("%Y-%m-%d")

    data = pd.read_csv("/home/ubuntu/Bert/redditBTC_20210331_current.csv",index_col=0)
    data = data[['open','high','low','Negative','Neutral','Positive','redditTotal','close']]
    data = data.iloc[3234:,:]

    dataset2 = data.iloc[-36:,:]
    values2 = dataset2.values
    # here checked values numeric format 
    values2 = values2.astype('float32')

    # Dataset values are normalized by using MinMax method
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(values2)
    #print(scaled)

    # Normalized values are converted for supervised learning 
    reframed2 = series_to_supervised(scaled,12,12)

    values2 = reframed2.values
    train2X, train2Y = values2[:,:-1], values2[:,191]

    # Train and Test datasets are reshaped in 3D size to be used in LSTM
    train2X = train2X.reshape((train2X.shape[0],1,train2X.shape[1]))

    # Define a simple sequential model
    model = Sequential()
    model.add(LSTM(128,activation = 'relu', input_shape=(train2X.shape[1], train2X.shape[2]),return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation = 'relu',return_sequences = True))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    #model.add(Activation("relu"))#linear
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.00005)
    model.compile(loss='mae', optimizer=opt,metrics=[tf.keras.metrics.MeanSquaredError()])
    #model = create_model()
    model.load_weights(tf.train.latest_checkpoint("/home/ubuntu/Bert/LSTM_model/"))
    PredictedTest = model.predict(train2X)

    train2X = train2X.reshape((train2X.shape[0],train2X.shape[2]))
    PredictedTest = PredictedTest.reshape(PredictedTest.shape[0],PredictedTest.shape[2]) 
    test2Predict = concatenate((PredictedTest, train2X[:, -7:]), axis=1)
    test2Predict = scaler.inverse_transform(test2Predict)
    test2Predict = test2Predict[:,0]

    delta = test2Predict[0] - data.iloc[-1,7]
    test2Predict = test2Predict -delta

    dateHour = []
    for i in range(len(test2Predict)+1)[1:]: 
        dateHour.append((datetime.datetime.strptime(data.index[-1],'%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=2*i)))
    df_pred = pd.DataFrame({'dateHour':dateHour,'close':test2Predict})
    df_pred['type'] = "Predicted"
    df_real = data.reset_index()[['dateHour','close']]
    df_real['type'] = 'Real'
    df_final = pd.concat([df_real,df_pred])
    df_final.drop_duplicates(subset = 'dateHour',keep="first",inplace=True)
    df_final.to_csv(f'/home/ubuntu/Bert/predicted/predict{yest}.csv',index=False)

    fig = px.line(df_final.iloc[-240:,:], x="dateHour", y="close",line_dash="type",labels={'x': 'date','y':'Close Price'},
                title='Real and Predicted Bitcoin price in USD')
    fig.write_image("/home/ubuntu/Bert/BTCprice.png")
    div = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
    div = div[33:-14] + r"\n\t"
    index = open("/home/ubuntu/Bert/index_noPlotly.html").read()
    index = re.sub(r'(<br>\n\t)',rf'\1{div}', index)
    with open("/home/ubuntu/Bert/index.html", "w") as file:
        file.write(index)
    wr.s3.upload(local_file='/home/ubuntu/Bert/BTCprice.png', path=f's3://circ-rna/static/images/BTCprice.png')
    wr.s3.upload(local_file=f'/home/ubuntu/Bert/predicted/predict{yest}.csv', path=f's3://bert-btc/daily_price_pred/predict{yest}.csv')
    wr.s3.upload(local_file=f'/home/ubuntu/Bert/predicted/predict{yest}.csv', path='s3://circ-rna/daily_BTC_pred/predictBTC.csv')