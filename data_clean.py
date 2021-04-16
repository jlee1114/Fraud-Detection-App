import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import stopwords
import nltk
import requests, json, os, sys, time, re
from sklearn.model_selection import train_test_split, KFold

from contractions import contractions


def add_label(data):
    fraud_types = ['fraudster_event','fraudster','fraudster_att']
    fraud_or_not = []
    for i in range(0,len(data)):
        if data['acct_type'].iloc[i] in fraud_types:
            fraud_or_not.append(1)
        else: fraud_or_not.append(0)
    data['event_type'] = fraud_or_not
    return data

def convert_desc(description):
    soup = BeautifulSoup(description, 'html.parser')
    return re.sub(r'(\s+)',' ',soup.text).strip()

def clean_text(text):
    '''Text Preprocessing '''
    # Convert words to lower case
    text = text.lower()
    # Expand contractions
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    # remove stopwords
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
    # Tokenize each word
    text =  nltk.WordPunctTokenizer().tokenize(text)
    # Lemmatize each token
    lemm = nltk.stem.WordNetLemmatizer()
    text = list(map(lambda word:list(map(lemm.lemmatize, word)), text))
    ans = []
    for arr in text:
        str = ''
        for c in arr:
            str += c
        ans.append(str)
    return ans

def avg_costs(data):
    avg_costs = []
    for i in range(0,len(data['ticket_types'])):
        costs = []
        for j in range(0,len(data['ticket_types'][i])):
            costs.append(data['ticket_types'][i][j]['cost'])
        avg_costs.append(np.average(costs))
    return avg_costs

def avg_quantity(data):
    avg_quantity = []
    for i in range(0,len(data['ticket_types'])):
        quantity = []
        for j in range(0,len(data['ticket_types'][i])):
            quantity.append(data['ticket_types'][i][j]['quantity_total'])
        avg_quantity.append(np.average(quantity))
    return avg_quantity

if __name__ == '__main__':
    data = pd.read_json(r'../data/data.json')
    data = add_label(data)
    data.description = data.description.apply(lambda x: convert_desc(x))
    data['desc_cleaned'] = data['description'].apply(lambda x: clean_text(x))
    data['avg_costs'] = avg_costs(data)
    data['avg_quantity'] = avg_quantity(data)
    X = data.drop(['event_type','acct_type'], axis=1)
    y = data.event_type
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    X_train.to_csv(r'../data/X_train.csv', index=False)
    X_test.to_csv(r'../data/X_test.csv', index=False)
    y_train.to_csv(r'../data/y_train.csv', index=False)
    y_test.to_csv(r'../data/y_test.csv', index=False)






