import string
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess(X):
    X = X.lower()
    # use regex to get rid of mentions (e.g., @tomhanks)
    pattern = f'(@[a-zA-Z0-9-]*)|[{string.punctuation[1:]}]*'
    p = re.compile(pattern)
    X = p.sub('', X)
    
    X = word_tokenize(X)
    stopwords_list = stopwords.words('english') + ['sxsw']
    stopwords_list += list(string.punctuation[1:])
    X = [x for x in X if x not in stopwords_list]
    
    lemmatizer = WordNetLemmatizer()
    X = [lemmatizer.lemmatize(x) for x in X]
    X = ' '.join(X)
    return X;

def split(df, percent):
    X = df['tweet']
    y = df['emotion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent, random_state=42)
    return (X_train, X_test, y_train, y_test)
    
    
def tfidfVectorize(X_train, *X_test):
    vectorizer = TfidfVectorizer()
    results=[]
    results.append(vectorizer.fit_transform(X_train))
    for test in X_test:
        results.append(vectorizer.transform(test))
    return tuple(results)

def w2v_vectorize(wv, docs):
    w2v_docs = []
    for doc in docs:
        doc_vec = np.zeros(100)
        for word in doc:
            doc_vec+=wv[word]
        doc_vec/=len(doc)
        w2v_docs.append(doc_vec)
    return w2v_docs
    
number_to_sentiment = {0: 'Negative emotion', 1: 'No emotion toward brand or product', 2: 'Positive emotion'}
sentiment_to_number = {'Negative emotion': 0, 'No emotion toward brand or product':1, 'Positive emotion':2}

def sentiment_encoder(Y):
    return [sentiment_to_number[y] for y in Y]

def sentiment_decoder(Y):
    return [number_to_sentiment[y] for y in Y]

def ngrams(X, size):
    vectorizer = TfidfVectorizer(ngram_range=size)
    grams = vectorizer.fit_transform(X)
    sums = grams.sum(axis = 0) 
    features = vectorizer.get_feature_names()
    data = [] 
    for col, term in enumerate(features): 
        data.append( (term, sums[0,col] )) 
    ranking = pd.DataFrame(data, columns = ['term','rank']) 
    words = (ranking.sort_values('rank', ascending = False)) 
    return words