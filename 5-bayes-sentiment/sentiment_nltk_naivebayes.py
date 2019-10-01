# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

# Useful references:
# https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386

# NLTK provides for some other sorts of features
# from nltk.sentiment.util import (mark_negation, extract_unigram_feats)
# mark_negation(): Append _NEG suffix to words that appear in the scope between
# a negation and a punctuation mark. extract_unigram_feats():
# Populate a dictionary of unigram features, reflecting the presence/absence
# in the document of each of the tokens in unigrams.

# Data
PROC_DIR = 'data/'
TRAIN = PROC_DIR + 'train.csv'
DEV =  PROC_DIR + 'dev.csv'
# In a previous step, I tokenized and pre-processed data and written
# out to a csv file.

df_train = pd.read_csv(TRAIN)
df_dev = pd.read_csv(DEV)

# %%
df_train = pd.DataFrame(df_train,columns=['id','label','text'])

# %%
# Feature extraction
df_pos_train = df_train[df_train['label'] == 'positive']
pos_tweets = df_pos_train['text'].tolist()

df_neg_train = df_train[df_train['label'] == 'negative']
neg_tweets = df_neg_train['text'].tolist()

df_neutral_train = df_train[df_train['label'] == 'neutral']
neutral_tweets = df_neutral_train['text'].tolist()

# %%
# how balanced is this training set?
len(df_pos_train)

# %%
len(df_neg_train)

# %%
len(df_neutral_train)


# %%
def features(sentence):
    words = sentence.lower().split()
    return dict(('contains(%s)' % w, True) for w in words)

positive_featuresets = [(features(tweet),'positive') for tweet in pos_tweets]
negative_featuresets = [(features(tweet),'negative') for tweet in neg_tweets]
neutral_featuresets = [(features(tweet),'neutral') for tweet in neutral_tweets]
training_features = positive_featuresets + negative_featuresets + neutral_featuresets

# %%
len(training_features)

# %%
sentiment_analyzer = SentimentAnalyzer()
trainer = NaiveBayesClassifier.train
classifier = sentiment_analyzer.train(trainer, training_features)

# %%
# Create evaluation data

#df_dev = pd.DataFrame(df_dev,columns=['id','label','text'])
truth_list = list(df_dev[['text', 'label']].itertuples(index=False, name=None))
len(truth_list)

# %%
# sanity check to make sure we manipulated the dataframe properly
truth_list[100]

# %%
# The evaluation method needs the feature extractor that was run to train the classifier
# Specifically, it wants a list of tuples (features,truth), where features is a dict
for i, (text, expected) in enumerate(truth_list):
    text_feats = features(text)
    truth_list[i] = (text_feats, expected)
truth_list[100]

# %%
# evaluate and print out all metrics
sentiment_analyzer.evaluate(truth_list,classifier)

# %%
# example of how to get to individual metrics
for key,value in sorted(sentiment_analyzer.evaluate(truth_list).items()):
    print('{0}: {1}'.format(key, value))
