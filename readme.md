<p align="center">
<br>
</p>

# Predicting Bitcoin price with Reddit Sentiment

## Table of Contents

- [Predicting Bitcoin price with Reddit Sentiment](#predicting-bitcoin-price-with-reddit-sentiment)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Bitcoin](#11-bitcoin)
    - [1.2 Sentiment Analysis](#12-sentiment-analysis)
  - [1.3 Bert](#13-bert)
  - [2. Dataset](#2-dataset)
  - [3. Bert Sentiment Analysis API Demo](#3-bert-sentiment-analysis-api-demo)
  - [4. Training and deployment of Bert](#4-training-and-deployment-of-bert)
    - [4.1. Create an Amazon SageMaker notebook instance](#41-create-an-amazon-sagemaker-notebook-instance)
    - [4.2. Training and deployment](#42-training-and-deployment)
      - [4.2.1 Load the pre-trained Bert model and tokenizer](#421-load-the-pre-trained-bert-model-and-tokenizer)
      - [4.2.2 Model Hyperparameters Tuning using wandb sweep](#422-model-hyperparameters-tuning-using-wandb-sweep)
      - [4.2.3 Deploy the model](#423-deploy-the-model)
      - [4.2.4 Containerizing and Deployment using Amazon EC2 and Docker](#424-containerizing-and-deployment-using-amazon-ec2-and-docker)
  - [5. Predict Bitcoin price with historical bitcoin data and sentiment data](#5-predict-bitcoin-price-with-historical-bitcoin-data-and-sentiment-data)
    - [5.1 Convert sentiment prediction data into sentiment counts in every 2 hours](#51-convert-sentiment-prediction-data-into-sentiment-counts-in-every-2-hours)
    - [5.2 Download historical Bitcoin price data and convert into 2 hour windows](#52-download-historical-bitcoin-price-data-and-convert-into-2-hour-windows)
    - [5.3 Merge two dataset into one in 2 hour interval.](#53-merge-two-dataset-into-one-in-2-hour-interval)
  - [6. Time Series Forecasting of Bitcoin Price](#6-time-series-forecasting-of-bitcoin-price)
    - [6.1. LSTM model](#61-lstm-model)
    - [6.2. Run the LSTM model to predic Bitcoin price of next day on AWS EC2](#62-run-the-lstm-model-to-predic-bitcoin-price-of-next-day-on-aws-ec2)
  - [7. Automation](#7-automation)
    - [7.1 AWS Event bridge triger Lambda to scrape the reddit comments and save to AWS S3.](#71-aws-event-bridge-triger-lambda-to-scrape-the-reddit-comments-and-save-to-aws-s3)
    - [7.2 AWS EC2 Crontab automaticly run "main.py".](#72-aws-ec2-crontab-automaticly-run-mainpy)
  - [8. Dashboard](#8-dashboard)
  - [9. References](#9-references)
  - [Contact](#contact)

Empolyed python, Bert, AWS EC2, docker, lambda, crontab and Event bridge, I build a prediction system that will automatically download reddit comments about bitcoin, sentimental analysis of the comments, then use these sentiment data to predict bitcoin price, and update the result daily to a dashboard here: [dashboard](http://3.145.73.72:8080/)

## Update: Since reddit disabled third-party API, the official API doesn't have the function required for this project, the dashboard is discontinued.

## 1. Introduction

### 1.1 Bitcoin

Bitcoin is a digital currency which operates free of any central control or the oversight of banks or governments.It has a distributed network system, where people can control their funds in a transparent way. It is the leading cryptocurrency and has the highest market capitalization among digital currencies. Unlike repeating phenomena like weather, cryptocurrency values do not follow a repeating pattern and mere past value of Bitcoin does not reveal any secret of future Bitcoin value. 

<p align="center">
<img src="/imgs/Bitcoin.jpeg">
<br>
<em>Bitcoin</em></p>

The Transformer architecture follows an encoder-decoder structure, but does not rely on recurrence and convolutions in order to generate an output. In a nutshell, the task of the encoder, on the left half of the Transformer architecture, is to map an input sequence to a sequence of continuous representations, which is then fed into a decoder. 

The decoder, on the right half of the architecture, receives the output of the encoder together with the decoder output at the previous time step, to generate an output sequence.

### 1.2 Sentiment Analysis

Humans follow general sentiments and technical analysis to invest in the market. Hence Sentiment is an important factor, considering people’s sentiment can improve the prediction of bitcoin price.

## 1.3 Bert

BERT is the first deeply bidirectional, unsupervised language representation, pre-trained using only a plain text corpus. Pre-trained representations can either be context-free or contextual, and contextual representations can further be unidirectional or bidirectional. Context-free models such as word2vec or GloVe generate a single word embedding representation for each word in the vocabulary. BERT represents “bank” using both its previous and next context starting from the very bottom of a deep neural network, making it deeply bidirectional.

<p align="center">
<img src="/imgs/Bert.png">
<br>
<em>Bert</em></p>

## 2. Dataset

I will use PushshiftAPI from psaw package to scrape comments regarding bitcoin from reddit.

```python
# import PushshiftAPI and download all the comments regarding bitcoin in the given date range
from psaw import PushshiftAPI 
api = PushshiftAPI()
start_date = start_date
end_date = end_date
data = data_prep_comments("bitcoin", start_time=int(dt.datetime(start_date).timestamp()), end_time=  int(dt.datetime(end_date).timestamp()),filters = [], limit = 2000000)
```
Here shows how the scraped comment data looks like:
<p align="center">
<img src="/imgs/Reddit_Comments.png">
<br>
<em>scraped reddit comment data</em></p>

## 3. Bert Sentiment Analysis API Demo

I built an [Online bitcoin comments sentiment analyzer](http://3.144.212.91:8501/) using [Streamlit](https://streamlit.io/) running the trained model. You can input any comments about Bitcoin, the API will do the sentiment analysis for you.

<p align="center">
<img src="/imgs/BertAPI.png">
<br>
<em>Image by Author</em></p>


## 4. Training and deployment of Bert

### 4.1. Create an Amazon SageMaker notebook instance

Follow this [hands-on tutorial](https://aws.amazon.com/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/) from AWS to create an Amazon SageMaker notebook instance. Use "*transformer*" as the **instance name**, and "*ml.t3.medium*" as the **instance type**.

You can also check the finished [Sagemaker notebook](https://github.com/vveizhang/Bitcoin_Social_Media_Sentiment_Analysis/blob/main/src/pyTorchInference.ipynb) here:

### 4.2. Training and deployment
#### 4.2.1 Load the pre-trained Bert model and tokenizer

```python
# construct sentiment classifier model using nn.Module
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        """
        Sentiment Classifier using BERT model.

        Args:
            num_classes (int): Number of classes for the output layer.
            bert_model (str): Pretrained BERT model to use.
        """
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the sentiment classifier.

        Args:
            input_ids: Tensor of input IDs.
            attention_mask: Tensor representing attention mask.

        Returns:
            Tensor: Output from the classifier.
        """
        returned = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask)
        pooled_output = returned["pooler_output"]
        output = self.drop(pooled_output)
        return self.out(output)

# load pretrained BERT model
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
data_loader = create_data_loader(df, tokenizer, BATCH_SIZE, max_len=300)
```
#### 4.2.2 Model Hyperparameters Tuning using wandb sweep
Run [Sagemaker notebook](https://github.com/vveizhang/Bitcoin_Social_Media_Sentiment_Analysis/blob/main/src/pyTorchInference.ipynb) on SageMaker to train and deploy the transformer model. Read through it to get more details on the implementation.

```python
# import wandb and configure sweep, metric will be validation accurary, try different optimizer and learning rates and epochs.
import wandb
wandb.login()
sweep_config = {'method': 'grid'}
metric = {'name': 'val_acc','goal': 'maximize'}
sweep_config['metric'] = metric

# Defien the hyperparameters to fine tuning
parameters_dict = {
    'optimizer': {'values': ['adam', 'sgd',"AdamW"]},
    'learning_rate': {'values': [5e-3, 1e-4, 3e-5, 6e-5, 1e-5]},
    'epochs': {'values': [2,4,6,8,10]}}

sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweep")


# define the model training and validating function
def train(config=None):
  with wandb.init(config=config):
    config = wandb.config

    EPOCHS = config.epochs
    model = SentimentClassifier(3).to(device)
    optimizer = build_optimizer(model,config.optimizer,config.learning_rate)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(EPOCHS):
      print(f'Epoch {epoch + 1}/{EPOCHS}')
      print('-' * 10)
      train_acc, train_loss = train_epoch(
        model,train_data_loader,loss_fn,optimizer,device,scheduler,len(df_train))
      print(f'Train loss {train_loss} accuracy {train_acc}')
      val_acc, val_loss = eval_model(model,test_data_loader,loss_fn,device,len(df_test))
      print(f'Val   loss {val_loss} accuracy {val_acc}')
        
wandb.agent(sweep_id, train)
```
The wandb will generate a parallel coordinates plot, a parameter importance plot, and a scatter plot when you start a W&B Sweep job. 

<p align="center">
<img src="/imgs/para_coord-1127.png">
<br>
<em>parallel coordinates plot</em></p>

#### 4.2.3 Deploy the model

```python
# define function to load saved model
def load_model(model_dir=model_dir):    
    model = SentimentClassifier(3).to(device)
    with open(model_dir, "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

# define function to make prediction using the saved model
def get_predictions(model, data_loader):
  model = model.eval()
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []
  with torch.no_grad():
    for d in data_loader:
      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      real_values.extend(targets)
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values
```

#### 4.2.4 Containerizing and Deployment using Amazon EC2 and Docker

Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly. With Docker, you can manage your infrastructure in the same ways you manage your applications. By taking advantage of Docker’s methodologies for shipping, testing, and deploying code quickly, you can significantly reduce the delay between writing code and running it in production. Here I also deployed the trained transformer model using Docker on Amazon EC2 instance.

```bash
ssh -i ec2-gpt2-streamlit-app.pem ec2-user@your-instance-DNS-address.us-east-1.compute.amazonaws.com
```

Then, copy the code into the cloud using git:
```bash
git clone https://github.com/vveizhang/Bitcoin_Social_Media_Sentiment_Analysis.git
```

Afterwards, go into the ec2-docker folder to build and run the image:
```bash
cd ec2-docker/
docker image build -t streamlit:BertSentiment .
```

## 5. Predict Bitcoin price with historical bitcoin data and sentiment data
### 5.1 Convert sentiment prediction data into sentiment counts in every 2 hours
Here is the original data from the previous prediction looks like:
<p align="center">
<img src="/imgs/predictedSentiDF.png">
<br>
<em>original data from the previous sentiment prediction</em></p>

```python
# get the sentiment counts and convert into 2 hours interval time series
y = pd.get_dummies(df.prediction).reset_index()
df2h = y.groupby(y.dateHour.dt.floor('2H')).sum(numeric_only=True)
```
<p align="center">
<img src="/imgs/predictedSentiCounts2h.png">
<br>
<em>original data from the previous sentiment prediction</em></p>

### 5.2 Download historical Bitcoin price data and convert into 2 hour windows
Use cryptocompare_API to download historical Bitcoin price data in 1 hour interval.
```python
# define the function to get historical bitcoin data using cryptocompare API
def get_BTC_data():
    # Get the API key from the Quantra file located inside the data_modules folder
    cryptocompare_API_key = 'cryptocompare_API_key'

    # Set the API key in the cryptocompare object
    cryptocompare.cryptocompare._set_api_key_parameter(cryptocompare_API_key)

    today = date.today()
    timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
    ticker_symbol = 'BTC'
    currency = 'USD'
    limit_value = 48
    exchange_name = 'CCCAGG'
    data=cryptocompare.get_historical_price_hour(ticker_symbol, currency, limit=limit_value, exchange=exchange_name, toTs=today)
    df2h = df.groupby(df.dateHour.dt.floor('2h')).mean(numeric_only=True)
    return df2h
```

<p align="center">
<img src="/imgs/BitcoinPriceData2h.png">
<br>
<em>Bitcoin Price Data in 2 hour</em></p>

### 5.3 Merge two dataset into one in 2 hour interval.

## 6. Time Series Forecasting of Bitcoin Price



### 6.1. LSTM model 

```python
# define the LSTM model using Tensorflow
opt = Adam(learning_rate=0.00005)
model = Sequential()
model.add(LSTM(128,activation = 'relu', input_shape=(trainX.shape[1], trainX.shape[2]),return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation = 'relu',return_sequences = True))
model.add(Dropout(0.2))
model.add(Dense(1))
#model.add(Activation("relu"))#linear
model.compile(loss='mae', optimizer=opt,metrics=[tf.keras.metrics.MeanSquaredError()])

# Model checkpoint callback
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="/content/model/best_model",
    save_weights_only=True,
    monitor = "val_loss",
    mode='min',
    save_best_only=True)
# Dataset is trained by using trainX and trainY
history = model.fit(trainX, trainY, epochs=28, batch_size=32, validation_data=(testX, testY), verbose=2, shuffle=False,callbacks=[model_checkpoint_callback])
```

<p align="center">
<img src="/imgs/LSTM_Bitcoin_pred.png">
<br>
<em>Actual and predicted Bitcoin price data</em></p>

### 6.2. Run the LSTM model to predic Bitcoin price of next day on AWS EC2

```python
# run the LSTM model to make the time series forecasting
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

# convert the predicted result back by scaler.inverse_transform
train2X = train2X.reshape((train2X.shape[0],train2X.shape[2]))
PredictedTest = PredictedTest.reshape(PredictedTest.shape[0],PredictedTest.shape[2]) 
test2Predict = concatenate((PredictedTest, train2X[:, -7:]), axis=1)
test2Predict = scaler.inverse_transform(test2Predict)
test2Predict = test2Predict[:,0]
```


## 7. Automation

I would like to build a automatic prediction system, which will automaticly web scraping reddit comments data and Bitcoin price data of that day, and predicti the Bitcoin price next day. Then output the price plot in a dashboard. I will use AWS Lambda, Event bridge and EC2 crontab to achive automation.

### 7.1 AWS Event bridge triger Lambda to scrape the reddit comments and save to AWS S3.

```python
# define the Lambda function to download reddit comments regarding bitcoin daily
def data_prep_comments(term, start_time, end_time, filters, limit):
    """
    Prepare DataFrame of comments from Reddit.

    Args:
        api: PushshiftAPI instance.
        term (str): Search term.
        start_time (int): Start timestamp.
        end_time (int): End timestamp.
        filters (list): List of filters for comments.
        limit (int): Limit for the number of comments.

    Returns:
        DataFrame: DataFrame containing comments.
    """
    if (len(filters) == 0):
        filters = ['id', 'author', 'created_utc','body', 'permalink', 'subreddit']

    comments = list(api.search_comments(
        q=term, after=start_time,before=end_time, filter=filters,limit=limit))       
    return pd.DataFrame(comments)
    
def lambda_handler(event, context):
    """
    AWS Lambda handler function.

    Args:
        event: AWS event.
        context: AWS context.
        api: PushshiftAPI instance.
        bucket_name (str): S3 bucket name.
        limit (int): Limit for the number of comments.

    Returns:
        None
    """
    df = data_prep_comments("bitcoin", start_time=int(dt.datetime(int(yesterday.strftime("%Y")),int(yesterday.strftime("%m")),int(yesterday.strftime("%d")), 0,1).timestamp()), 
                            end_time=  int(dt.datetime(int(yesterday.strftime("%Y")),int(yesterday.strftime("%m")),int(yesterday.strftime("%d")), 23,59).timestamp()),filters = [], limit = limit)
    
    bucket = 'bert-btc'
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, f'daily_comments/df{yesterday.strftime("%Y-%m-%d")}.csv').put(Body=csv_buffer.getvalue())
```

### 7.2 AWS EC2 Crontab automaticly run "main.py".
Here is the code for Crontab automation.
```bash
# automation of the python script running using crontab
crontab -e
30 2 * * * python3 run main.py
```
main.py
```python
# define the main function to get comment data, make sentiment analysis, convert and combine data, get historical bitcoin data, make the prediction and upload the prediction to dashboard
if __name__ == "__main__":
    yest = yesterday.strftime("%Y-%m-%d")

    bucket = bucket
    csv_path = os.path.join(bucket,data_key)
    df = wr.s3.read_csv(path=csv_path)
    df.to_csv("df.csv", index=False)

    BATCH_SIZE = 16
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    data_loader = create_data_loader(df, tokenizer, BATCH_SIZE, max_len=512)

    wr.s3.download(path = "s3://bucket/model.pth",local_file='model.pth')
    trained_model = SentimentClassifier(3)
    trained_model.load_state_dict(torch.load("model.pth"))#,map_location=torch.device('cpu')))

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(trained_model,data_loader)
    df['prediction'] = y_pred
    df = df[['body','created_utc','datetime','prediction']]
    df.to_csv(f"/home/ubuntu/Bert/df_predicted.csv")

    write_key =  f"daily_prediction/df{yest}.csv"
    write_path = os.path.join(bucket,write_key)
    wr.s3.to_csv(df,path=write_path)

    df2 = df[['datetime','text','prediction']]
    write_WC_key = f"daily_wordcloud/df{yest}.csv"
    write_WC_path = os.path.join(bucket,write_WC_key)
    wr.s3.to_csv(df2,path=write_WC_path)
    df_pred_count = prediction_count()
    wr.s3.to_csv(df_pred_count,path=f's3://bucket/daily_sentiCounts/sentiCount{yest}_2h.csv')
    word_cloud(df2)

    btcData2h = get_BTC_data()
    btcData2h.to_csv(f"/home/ubuntu/Bert/dateBTC2h.csv")
    wr.s3.to_csv(btcData2h,path=f's3://bucket/daily_BTCdata/date{yest}BTC2h.csv')

    dataToday = pd.merge(btcData2h,df_pred_count, on ='dateHour')
    dataToday.to_csv("/home/ubuntu/Bert/redditBTC_2h.csv")
    dataPast = pd.read_csv("/home/ubuntu/Bert/redditBTC_20210331_current.csv")
    dataToday = pd.read_csv("/home/ubuntu/Bert/redditBTC_2h.csv")
    data = pd.concat([dataPast,dataToday])
    data.to_csv("/home/ubuntu/Bert/redditBTC_20210331_current.csv",index=False)
    wr.s3.to_csv(data,path=f's3://bucket/BTCplusReddit/data{yest}.csv')
    predictPrice()
```
This python script will do the sentimental prediction of reddit comments, upload the result to S3; download the historical bitcoin price data of that day, use LSTM model to predict the price for the next day; output the predicted price plot to a [dashboard](http://3.145.73.72:8080/), as well as the wordcloud of the reddit comment that day.

All source code can be found in this Github Repo: [https://github.com/vveizhang/Bitcoin_Social_Media_Sentiment_Analysis](https://github.com/vveizhang/Bitcoin_Social_Media_Sentiment_Analysis)-


## 8. Dashboard
[dashboard](http://3.145.73.72:8080/)

<p align="center">
<img src="/imgs/dashborad.png">
<br>
<em>Actual and predicted Bitcoin price data</em></p>




<p align="center">
<img src="/imgs/Wordcloud.png">
<br>
<em>Word Cloud</em></p>

## 9. References

- **Transformers**: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- **Introduction of LSTM**: [Understanding LSTM -- a tutorial into Long Short-Term Memory Recurrent Neural Networks](https://arxiv.org/abs/1909.09586)

- **Introduction of Bert**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

- **Train and deploy models on AWS SageMaker**: [https://medium.com/@thom.e.lane/streamlit-on-aws-a-fully-featured-solution-for-streamlit-deployments-ba32a81c7460](https://medium.com/@thom.e.lane/streamlit-on-aws-a-fully-featured-solution-for-streamlit-deployments-ba32a81c7460)
- **Deploy Streamlit app on AWS EC2**: [https://medium.com/usf-msds/deploying-web-app-with-streamlit-docker-and-aws-72b0d4dbcf77](https://medium.com/usf-msds/deploying-web-app-with-streamlit-docker-and-aws-72b0d4dbcf77)

## Contact

- **Author**: Wei Zhang
- **Email**: [zwmc@hotmail.com](zwmc@hotmail.com)
- **Github**: [https://github.com/vveizhang](https://github.com/vveizhang)
- **Linkedin**: [https://www.linkedin.com/in/wei-z-76253523/](https://www.linkedin.com/in/wei-z-76253523/)
