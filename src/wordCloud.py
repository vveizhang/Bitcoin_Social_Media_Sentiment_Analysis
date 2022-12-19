# import libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import time
import datetime as dt
from datetime import date, timedelta,datetime
from io import StringIO
import os
import awswrangler as wr
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import RegexpTokenizer
from gensim.parsing.preprocessing import strip_punctuation

def word_cloud(df2):
    today = date.today()
    timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
    yesterday = today - timedelta(days = 1)
    yest = yesterday.strftime("%Y-%m-%d")

    df2 = df2.dropna()
    df_positive = df2[df2['prediction'] == 2]
    allTextPos = " ".join(i for i in df_positive['text'])
    df_negative = df2[df2['prediction'] == 0]
    allTextNeg = " ".join(i for i in df_negative['text'])
    df_neutral = df2[df2['prediction'] == 1]
    allTextNeu = " ".join(i for i in df_neutral['text'])
    allText = " ".join(i for i in df2['text'])

    background = np.array(Image.open("/home/ubuntu/Bert/cloud.png"))
    stopwords = set(STOPWORDS)
    stopwords.add('S')
    stopwords.add('gt')
    stopwords.add('t')
    stopwords.add('https')
    stopwords.add('don')
    stopwords.add('u')
    stopwords.add('isn')
    stopwords.add('wouldn')
    stopwords.add('im')
    stopwords.add('m')
    stopwords.add('e')
    stopwords.add('Bitcoin')
    stopwords.add('BTC')
    stopwords.add('will')
    stopwords.add('reddit')
    stopwords.add('coin')
    stopwords.add('crypto')
    stopwords.add('people')
    stopwords.add('one')
    stopwords.add('think')

    wc = WordCloud(background_color="white",
                max_words=100,
                mask = background,
                stopwords = stopwords)

    plt.figure(figsize=(30, 30))
    plt.imshow(wc.generate(allTextPos))
    plt.title(f'{yest} "Positive" Comments World Cloud',fontdict={'fontsize': 40,'color':'red', 'fontweight':'bold'})
    plt.axis("off")
    plt.savefig(f'/home/ubuntu/Bert/{yest}Positive.png', bbox_inches='tight')               

    plt.figure(figsize=(30, 30))
    plt.imshow(wc.generate(allTextNeu))
    plt.title(f'{yest} "Neutral" Comments World Cloud',fontdict={'fontsize': 40,'color':'red', 'fontweight':'bold'})
    plt.axis("off")
    plt.savefig(f'/home/ubuntu/Bert/{yest}Neutral.png', bbox_inches='tight')

    plt.figure(figsize=(30, 30))
    plt.imshow(wc.generate(allTextNeg))
    plt.title(f'{yest} "Negative" Comments World Cloud',fontdict={'fontsize': 40,'color':'red', 'fontweight':'bold'})
    plt.axis("off")
    plt.savefig(f'/home/ubuntu/Bert/{yest}Negative.png', bbox_inches='tight')

    plt.figure(figsize=(30, 30))
    plt.imshow(wc.generate(allText))
    plt.title(f'{yest} All Comments World Cloud',fontdict={'fontsize': 40,'color':'red', 'fontweight':'bold'})
    plt.axis("off")
    plt.savefig(f'/home/ubuntu/Bert/{yest}all.png', bbox_inches='tight')

    time.sleep(60)

    wr.s3.upload(local_file=f'/home/ubuntu/Bert/{yest}all.png', path=f's3://bert-btc/daily_wordcloud/{yest}/{yest}all.png')
    wr.s3.upload(local_file=f'/home/ubuntu/Bert/{yest}Negative.png', path=f's3://bert-btc/daily_wordcloud/{yest}/{yest}Negative.png')
    wr.s3.upload(local_file=f'/home/ubuntu/Bert/{yest}Neutral.png', path=f's3://bert-btc/daily_wordcloud/{yest}/{yest}Neutral.png')
    wr.s3.upload(local_file=f'/home/ubuntu/Bert/{yest}Positive.png', path=f's3://bert-btc/daily_wordcloud/{yest}/{yest}Positive.png')

    wr.s3.upload(local_file=f'/home/ubuntu/Bert/{yest}all.png', path=f's3://circ-rna/static/images/all.png')
    wr.s3.upload(local_file=f'/home/ubuntu/Bert/{yest}Negative.png', path=f's3://circ-rna/static/images/Negative.png')
    wr.s3.upload(local_file=f'/home/ubuntu/Bert/{yest}Neutral.png', path=f's3://circ-rna/static/images/Neutral.png')
    wr.s3.upload(local_file=f'/home/ubuntu/Bert/{yest}Positive.png', path=f's3://circ-rna/static/images/Positive.png')
    return None