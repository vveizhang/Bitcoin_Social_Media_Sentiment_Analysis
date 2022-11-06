import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
import os
import json
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

@st.cache
def load_model(model_dir="https://www.dropbox.com/s/0nq3ukfhe99t9pv/model.pth?dl=1"):    
    model = SentimentClassifier(3).to(device)
    with open(model_dir, "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def input_fn(input_data, content_type= 'application/json'):
    input = json.loads(input_data)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    review = input["text"]
    encoding = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=300,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt')
    input_ids = encoding['input_ids'].flatten()
    attention_mask = encoding['attention_mask'].flatten()
    return {'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten()}


def run_model(input, model):
    with torch.no_grad():       
        prediction = model(input['input_ids'].unsqueeze(0).to(device),input['attention_mask'].unsqueeze(0).to(device))
        _,result = torch.max(prediction, dim=1)
        result = result.cpu().numpy()[0]
        return(result)

#Serialize the prediction result into the desired response content type
# def output_fn(prediction, accept="text/plain"):
#     #logger.info('Serializing the generated output.')
#     result = np.round(prediction.cpu().item())
# #     if result == 1.0:
# #         response = "Your inqury sequence IS circRNAs"
# #     else:
# #         response = "Your inqury sequence IS NOT circRNAs"    
#     return str(result)