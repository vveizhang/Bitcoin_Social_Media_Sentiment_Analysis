# import libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sentiment import *
from wordCloud import *
from BTCdata2h import *
from predictionCount import *
from LSTM import *
import pandas as pd
import numpy as np
from PIL import Image
import time
import datetime as dt
from datetime import date, timedelta
import datetime as dt
from datetime import date, timedelta,datetime
import torch
from torch import nn
from io import StringIO
import os
import awswrangler as wr
from wordcloud import WordCloud, STOPWORDS
import transformers
from nltk.tokenize import RegexpTokenizer
from gensim.parsing.preprocessing import strip_punctuation
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    today = date.today()
    timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
    yesterday = today - timedelta(days = 1)
    yest = yesterday.strftime("%Y-%m-%d")

    # indicate the s3 data source
    data_key = f"daily_comments/df{yest}.csv"
    bucket = 's3://bert-btc'
    csv_path = os.path.join(bucket,data_key)
    df = wr.s3.read_csv(path=csv_path, encoding = "iso-8859-1")
    df.head()
    df.to_csv("df.csv", index=False)

    df['sentiment'] = 0
    df = df[['body','created_utc','sentiment']]
    dt_list = []
    for i in df['created_utc']:
        dt_list.append(datetime.fromtimestamp(i))
    df['datetime'] = dt_list

    # load the pretrained 'bert-base-cased' model
    BATCH_SIZE = 16
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    data_loader = create_data_loader(df, tokenizer, BATCH_SIZE, max_len=512)

    wr.s3.download(path = "s3://bert-btc/model.pth",local_file='model.pth')

    # load the fine-tuned model weights 
    trained_model = SentimentClassifier(3)
    trained_model.load_state_dict(torch.load("model.pth"))#,map_location=torch.device('cpu')))
    trained_model = trained_model.to(device)

    # do the prediction
    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(trained_model,data_loader)
    df['prediction'] = y_pred
    df = df[['body','created_utc','datetime','prediction']]
    df.to_csv(f"/home/ubuntu/Bert/df_predicted.csv")

    # save the predicted result to s3
    write_key =  f"daily_prediction/df{yest}.csv"
    write_path = os.path.join(bucket,write_key)
    wr.s3.to_csv(df,path=write_path)

    df['text'] = ""
    for i in range(len(df)):
        no_punc = strip_punctuation(str(df.iloc[i,0]))
        no_punc = no_punc.replace("bitcoin","")
        no_punc = no_punc.replace("btc","")
        df.iloc[i,4] = no_punc

    df2 = df[['datetime','text','prediction']]
    write_WC_key = f"daily_wordcloud/df{yest}.csv"
    write_WC_path = os.path.join(bucket,write_WC_key)
    wr.s3.to_csv(df2,path=write_WC_path)
    df_pred_count = prediction_count()
    wr.s3.to_csv(df_pred_count,path=f's3://bert-btc/daily_sentiCounts/sentiCount{yest}_2h.csv')
    word_cloud(df2)

    # download bitcoin historical data and save to s3
    btcData2h = get_BTC_data()
    btcData2h.to_csv(f"/home/ubuntu/Bert/dateBTC2h.csv")
    wr.s3.to_csv(btcData2h,path=f's3://bert-btc/daily_BTCdata/date{yest}BTC2h.csv')

    # merge the predicted sentiment data and historical data to predict bitcoin price
    dataToday = pd.merge(btcData2h,df_pred_count, on ='dateHour')
    dataToday.to_csv("/home/ubuntu/Bert/redditBTC_2h.csv")
    dataPast = pd.read_csv("/home/ubuntu/Bert/redditBTC_20210331_current.csv")
    dataToday = pd.read_csv("/home/ubuntu/Bert/redditBTC_2h.csv")
    data = pd.concat([dataPast,dataToday])
    data.to_csv("/home/ubuntu/Bert/redditBTC_20210331_current.csv",index=False)
    wr.s3.to_csv(data,path=f's3://bert-btc/BTCplusReddit/data{yest}.csv')
    predictPrice()
       