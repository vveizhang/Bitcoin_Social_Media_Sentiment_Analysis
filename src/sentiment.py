# import libraries
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
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# construct sentiment classifier model using nn.Module
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        returned = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask)
        pooled_output = returned["pooler_output"]
        output = self.drop(pooled_output)
        return self.out(output)

# define the function to load saved model
def load_model(model_dir):    
    model = SentimentClassifier(3).to(device)
    with open(model_dir, "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

# Define MyDataset from class Dataset
class redditDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len=512): #,):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.reviews)
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding= 'max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
        )
        return {
    'review_text': review,
    'input_ids': encoding['input_ids'].flatten(),
    'attention_mask': encoding['attention_mask'].flatten(),
    'targets': torch.tensor(target, dtype=torch.long)
    }

# define customer data loader
def create_data_loader(df, tokenizer, batch_size,max_len=512):
    ds = redditDataset(
    reviews=df.body.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=512)
    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=1
)

# define function to do the prediction
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