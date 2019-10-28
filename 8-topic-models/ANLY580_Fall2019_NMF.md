---
jupyter:
  jupytext:
    comment_magics: false
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

#### ANLY 580 - NLP for Data Analytics Fall Semester 2019


### This notebook relies on the preprocessing performed in the topic modeling using LDA notebook. The dictionary and corpus preprocessing is re-used in this notebook to demonstrate topic modeling using Non-negative Matrix Factorization (NNMF)

```python
import os
import pandas as pd
import numpy as np

import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel
from gensim.models.nmf import Nmf

from nltk.corpus import stopwords
import string
import re
import pprint

from collections import OrderedDict

import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
```

```python
#Modify these parmeters for the local directory structre where the data is located. 

data_file_path = "data"
TEMP_FOLDER = "backup"
```

```python
# Read in the parsed_tweets dataframe from the ldamodel
parsed_tweets_types = {'msgid':str, 'topic':str, 'sentiment':str, 'Tweet':list}
parsed_tweets = pd.read_csv(os.path.join(data_file_path, "parsed_tweets.csv"),
                            sep="\t",
                            index_col = 0,
                            dtype={'msgid':str, 'topic':str, 'sentiment':str, 'Tweet':str},
                            converters={"Tweet": lambda x: x.strip("[]").replace("'","").split(", ")})
parsed_tweets.head(15)
```

```python
parsed_tweets.tail(15)
```

```python
human_topics = list(set(parsed_tweets['topic'].tolist()))
print(human_topics)
```

```python
# Load in the dictionary and corpus computed for the lda model.

dictionary = corpora.Dictionary.load(os.path.join(TEMP_FOLDER, 'semval.dict'))  # load from dictionary
corpus = corpora.MmCorpus(os.path.join(TEMP_FOLDER, 'semval.mm'))  # load from disk
```

```python
# In NNMF factorization we will use term-frequency-inverse_document-frequency for weighting the term document matrix

tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]      # step 2 -- use the model to transform vectors
```

```python
# Change the total_topics number to examine impacts the number of topics has with NNMF

total_topics = 20
num_passes = 20
```

```python
# Build the model

nmf = Nmf(corpus, id2word = dictionary, passes = 20, num_topics = total_topics)
```

```python
#Show first n=10 important words in the topics:
nmf.show_topics(total_topics, num_words = 10)
```

```python
# Load the topic - term data into an pyton dictionary
data_nmf = {i: OrderedDict(nmf.show_topic(i,10)) for i in range(total_topics)}
#data_nmf
```

```python
# Use the ordered dictionary to load the data into a dataframe
data_nmf = pd.DataFrame(data_nmf)
data_nmf = data_nmf.fillna(0).T
print(data_nmf.shape)
```

```python
data_nmf.head(20)
```

```python
# Run the original documents back thru the model to infer the distribution of topics 
# according to the nnmf model

topics = []
probs = []
max_to_show = 20

for k, i in enumerate(range(len(parsed_tweets['Tweet']))):
    bow = dictionary.doc2bow(parsed_tweets['Tweet'][i])
    doc_topics = nmf.get_document_topics(bow, minimum_probability = 0.01)
    topics_sorted = sorted(doc_topics, key = lambda x: x[1], reverse = True)
    topics.append(topics_sorted[0][0])
    probs.append("{}".format(topics_sorted[0][1]))
    
    # Dump out the topic and probability assignments for the first 20 documents
    if k < max_to_show:
        print("Document {}: {}".format(k, topics_sorted))

parsed_tweets['NMFtopic'] = pd.Series(topics)
parsed_tweets['NMFprob'] = pd.Series(probs)
```

```python
# dump the topic assignments for the last document thru the previous loop

doc_topics
```

```python
# Resort the dataframe according to the human annotated topic and nnmf topic
parsed_tweets.sort_values(['topic', 'NMFtopic'], ascending=[True, False], inplace=True)
parsed_tweets.head(20)
```

```python
# Take a look at the distributions of human annotated topics in the data via a barplot

sns.set(rc={'figure.figsize':(12.7,9.27)})
by_topic = sns.countplot(x='NMFtopic', data=parsed_tweets)

for item in by_topic.get_xticklabels():
    item.set_rotation(90)
```

```python
# Resort the dataframe according to the the nnmf assigned topic and the human annotated topic

parsed_tweets.sort_values(['NMFtopic', 'topic'], ascending=[True, True], inplace=True)
parsed_tweets.head(20)
```

```python
# Resort the dataframe according to the the nnmf assigned topic and the assocoiated probability
parsed_tweets.sort_values(['NMFtopic', 'NMFprob'], ascending=[True, False], inplace=True)
parsed_tweets.head(20)
```

```python
# What do the topic distrubtions look like relative to the original human annotated/tagged topics

df2 = parsed_tweets.groupby(['NMFtopic', 'topic'])['NMFtopic'].count().unstack('topic')
topic_mixture = df2[human_topics].plot(kind='bar', stacked=True, legend = False)
```

```python
# What do the topic distrubtions look like relative to the original human annotated/tagged sentiment

human_sentiment = list(set(parsed_tweets['sentiment'].tolist()))
df2 = parsed_tweets.groupby(['NMFtopic', 'sentiment'])['NMFtopic'].count().unstack('sentiment')
topic_mixture = df2[human_sentiment].plot(kind='bar', stacked=True, legend = True)
```

```python

```
