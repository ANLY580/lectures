# -*- coding: utf-8 -*-
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
#     display_name: Python [conda env:root] *
#     language: python
#     name: conda-root-py
# ---

# %% [markdown]
# # Intro to Natural Language Processing with Python
#
# ## Info
# - Scott Bailey (CIDR), *scottbailey@stanford.edu*
# - Javier de la Rosa (CIDR), *versae@stanford.edu*
#
#
# ## What are we covering today?
# - What is NLP?
# - Options for NLP in Python
# - Tokenization
# - Part of Speech Tagging
# - Named Entity Recognition
# - Word transformations
# - Readability indices

# %% [markdown]
# ## Goals
#
# By the end of the workshop, we hope you'll have a basic understanding of natural language processing, and enough familiarity with one NLP package, SpaCy, to perform basic NLP tasks like tokenization and part of speech tagging. Through analyzing presidential speeches, we also hope you'll understand how these basic tasks open up a number of possibilities for textual analysis, such as readability indices. 

# %% [markdown]
# ## What is NLP
#
# NLP stands for Natual Language Processing and it involves a huge variety of tasks such as:
# - Automatic summarization.
# - Coreference resolution.
# - Discourse analysis.
# - Machine translation.
# - Morphological segmentation.
# - Named entity recognition.
# - Natural language understanding.
# - Part-of-speech tagging.
# - Parsing.
# - Question answering.
# - Relationship extraction.
# - Sentiment analysis.
# - Speech recognition.
# - Topic segmentation.
# - Word segmentation.
# - Word sense disambiguation.
# - Information retrieval.
# - Information extraction.
# - Speech processing.
#
# One of the key ideas is to be able to process text without reading it.

# %% [markdown]
# ## NLP in Python
#
# Python is builtin with a very mature regular expression library, which is the building block of natural language processing. However, more advanced tasks need different libraries. Traditionally, in the Python ecosystem the Natural Language Processing Toolkit, abbreviated as `NLTK`, has been until recently the only working choice. Now, though, there are a number of choices based on different technologies and approaches
#
# We'll a solution that appeared relatively recently, called `spaCy`, and it is much faster than NLTK since is written in a pseudo-C Python language optimized for speed called Cython.
#
# Both these libraries are complex and there exist wrappers around them to simplify their APIs. The two more popular are `Textblob` for NLTK and CLiPS Parser, and `textacy` for spaCy. In this workshop we will be using spaCy with a touch of textacy thrown in at the very end.

# %%
!pip install spacy

# %%
import spacy

# %%
!python -m spacy download en

# %%
nlp = spacy.load('en')

# %%
# helper functions
import requests

def get_text(url):
    return requests.get(url).text

def get_speech(url):
    page = get_text(url)
    full_text = page.split('\n')
    return " ".join(full_text[2:])


# %%
clinton_url = "https://raw.githubusercontent.com/sul-cidr/python_workshops/master/data/clinton2000.txt"
clinton_speech = get_speech(clinton_url)
clinton_speech

# %%
doc = nlp(clinton_speech)

# %% [markdown]
# ## Tokenization
#
# In NLP, the act of splitting text is called tokenization, and each of the individual chunks is called a token. Therefore, we can talk about word tokenization or sentence tokenization depending on what it is that we need to divide the text into.

# %%
# word level
for token in doc:
    print(token.text)

# %%
# sentence level
for sent in doc.sents:
    print(sent)

# %%
# noun phrases
for phrase in doc.noun_chunks:
    print(phrase)

# %% [markdown]
# ## Part of Speech Tagging
#
# SpaCy also allows you to perform Part-Of-Speech tagging, a kind of grammatical chunking, out of the box. 

# %%
# simple part of speech tag
for token in doc:
    print(token.text, token.pos_)

# %%
# detailed tag
# For what these tags mean, you might check out http://www.clips.ua.ac.be/pages/mbsp-tags
for token in doc:
    print(token.text, token.tag_)

# %%
# syntactic dependency
for token in doc:
    print(token.text, token.dep_)

# %%
# visualizing the sentence
from spacy import displacy

# %%
first_sent = list(doc.sents)[0]
first_sent

# %%
single_doc = nlp(str(first_sent))
options = {"compact": True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Source Sans Pro'}
displacy.render(single_doc, style="dep", jupyter=True, options=options)


# %% [markdown]
# <div style="font-size: 1em; margin: 1em 0 1em 0; border: 1px solid #86989B; background-color: #f7f7f7; padding: 0;">
# <p style="margin: 0; padding: 0.1em 0 0.1em 0.5em; color: white; border-bottom: 1px solid #86989B; font-weight: bold; background-color: #AFC1C4;">
# Activity
# </p>
# <p style="margin: 0.5em 1em 0.5em 1em; padding: 0;">
# Write a function `count_chars(text)` that receives `text` and returns the total number of characters ignoring spaces and punctuation marks. For example, `count_chars("Well, I am not 30 years old.")` should return `20`.
# <br/>
# * **Hint**: You could count the characters in the words.*
# </p>
# </div>

# %%
# Solution using two functions, one to get just words without punct, one to get chars
def return_words(doc):
    return [token.text for token in doc if token.pos_ is not 'PUNCT']

def count_chars(words):
    return sum(len(w) for w in words)

# count_chars("Well, I am not 30 years old.")
words = return_words(nlp("Well, I am not 30 years old."))
count_chars(words)

# %% [markdown]
# ## Named Entity Recognition 

# %%
# https://spacy.io/api/annotation#named-entities
# trained on OntoNotes corpus
for ent in doc.ents:
    print(ent.text, ent.label_)

# %%
# If you're working on tokens, you can still access entity type
# Notice, though that the phrase entities are broken up here because we're iterating over tokens
# https://spacy.io/api/annotation#named-entities
for token in doc:
    if token.ent_type_ is not '':
        print(token.text, token.ent_type_, "----------", spacy.explain(token.ent_type_))

# %%
# spacy comes with built in entity visualization
displacy.render(single_doc, style="ent", jupyter=True)

# %%
next_sent = list(doc.sents)[3]
next_doc = nlp(str(next_sent))
displacy.render(next_doc, style="ent", jupyter=True)

# %% [markdown]
# It is possible to train your own entity recognition model, and to train other types of models in spaCy, but you need sufficient labeled data to make it work well.

# %% [markdown]
# ## Word transformations

# %%
# lemmas
for token in doc:
    print(token.text, token.lemma_)

# %%
doc1 = nlp('here are octopi')
for token in doc1:
    print(token.lemma_)

# %%
doc1 = nlp('There have been many mice and geese surrounding the pond.')
for token in doc1:
    print(token, token.lemma_)

# %%
# say we just want to lematize verbs
for token in doc:
    if token.tag_ == "VBP":
        print(token.text, token.lemma_)

# %%
# If you're using the simple part of speech instead of the tags
for token in doc:
    if token.pos_ == "VERB":
        print(token.text, token.lemma_)

# %%
# lowercasing
for token in doc:
    print(token.text, token.lower_)

# %% [markdown]
# ## Counting

# %%
from collections import Counter

# %%
sample_sents = "One fish, two fish, red fish, blue fish. One is less than two."

# %%
# Create a spacy doc
new_doc = nlp(sample_sents)

# Create a list of the words without the punctuation
words = [token.text for token in new_doc if token.pos_ is not 'PUNCT']
words

# %%
counter = Counter(words)

# %%
counter.most_common(10)

# %%
counter["fish"]


# %% [markdown]
# ## Sentiment Analysis
#
# Right now, spacy doesn't include a model for sentiment analysis. From comments on the spacy github repo, the developers of spacy, Explosion are going to offer sentiment models as part of their commercial offerings.
#
# They have put out examples for how to do sentiment analysis: 
# - https://github.com/explosion/spaCy/blob/master/examples/deep_learning_keras.py
# - https://github.com/explosion/spaCy/blob/master/examples/training/train_textcat.py
#
# Both of these use some sort of deep learning/neural networks
#

# %% [markdown]
# <div style="font-size: 1em; margin: 1em 0 1em 0; border: 1px solid #86989B; background-color: #f7f7f7; padding: 0;">
# <p style="margin: 0; padding: 0.1em 0 0.1em 0.5em; color: white; border-bottom: 1px solid #86989B; font-weight: bold; background-color: #AFC1C4;">
# Activity
# </p>
# <p style="margin: 0.5em 1em 0.5em 1em; padding: 0;">
# Let's define the lexicon of a person as the number of different words she uses to speak. Write a function `get_lexicon(text, n)` that receives `text` and `n` and returns the lemmas of nouns, verbs, and adjectives that are used at least `n` times.
# <br/>
# </p>
# </div>

# %%
def get_lexicon(text, n):
    doc = nlp(text)
    
    # return a list of words that have the correct part of speech    
    words = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB"]]
    # count the words     
    counter = Counter(words)
    # filter by number
    filtered_words = [word for word in counter if counter[word] > n]
    return sorted(filtered_words)
    
get_lexicon(clinton_speech, 30)


# %% [markdown]
# ## Readability indices
#
# Readability indices are ways of assessing how easy or complex it is to read a particular text based on the words and sentences it has. They usually output scores that correlate with grade levels.
#
# A couple of indices that are presumably easy to calculate are the Auto Readability Index (ARI) and the Coleman-Liau Index:
#
# $$
# ARI = 4.71\frac{chars}{words}+0.5\frac{words}{sentences}-21.43
# $$
# $$ CL = 0.0588\frac{letters}{100 words} - 0.296\frac{sentences}{100words} - 15.8 $$
#
#
# https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
#
# https://en.wikipedia.org/wiki/Automated_readability_index

# %%
# problem: the tokens in spacy include punctuation. to get this right, we should remove punct
# we then have to make sure our functions handle lists of words rather than spacy doc objects

def coleman_liau_index(doc, words):
    return (0.0588 * letters_per_100(doc)) - (0.296 * sentences_per_100(doc, words)) - 15.8

def count_chars(words):
    return sum(len(w) for w in words)

def sentences_per_100(doc, words):
    return (len(list(doc.sents)) / len(words)) * 100

def letters_per_100(words):
    return (count_chars(words) / len(words)) * 100


# %%
# To get just the words, without punctuation tokens
def return_words(doc):
    return [token.text for token in doc if token.pos_ is not 'PUNCT']


# %%
fancy_doc = nlp("Regional ontology, clearly defined by Heidegger, equals, if not surpasses, the earlier work of Heidegger's own mentor, Husserl")
fancy_words = return_words(fancy_doc)
fancy_words

# %%
coleman_liau_index(fancy_doc, fancy_words)

# %%
doc = nlp(clinton_speech)
clinton_speech_words = return_words(doc)
coleman_liau_index(doc, clinton_speech_words)


# %% [markdown]
# <div style="font-size: 1em; margin: 1em 0 1em 0; border: 1px solid #86989B; background-color: #f7f7f7; padding: 0;">
# <p style="margin: 0; padding: 0.1em 0 0.1em 0.5em; color: white; border-bottom: 1px solid #86989B; font-weight: bold; background-color: #AFC1C4;">
# Activity
# </p>
# <p style="margin: 0.5em 1em 0.5em 1em; padding: 0;">
# Write a function `auto_readability_index(doc)` that receives a spacy `Doc` and returns the Auto Readability Index (ARI) score as defined above. 
# <br/>
# * **Hint**: Feel free to use functions we've defined before.*
#    
# </p>
# </div>

# %%
def auto_readability_index(doc):
    words = return_words(doc)
    chars = count_chars(words)
    words = len(words)
    sentences = len(list(doc.sents))
    return (4.71 * (chars / words)) + (0.5 * (words / sentences)) - 21.43


# %%
auto_readability_index(fancy_doc)

# %%
auto_readability_index(doc)

# %%
clinton_url = "https://raw.githubusercontent.com/sul-cidr/python_workshops/master/data/clinton2000.txt"
bush_url = "https://raw.githubusercontent.com/sul-cidr/python_workshops/master/data/bush2008.txt"
obama_url = "https://raw.githubusercontent.com/sul-cidr/python_workshops/master/data/obama2016.txt"
trump_url = "https://raw.githubusercontent.com/sul-cidr/python_workshops/master/data/trump.txt"

# %%
clinton_speech = get_speech(clinton_url)
bush_speech = get_speech(bush_url)
obama_speech = get_speech(obama_url)
trump_speech = get_speech(trump_url)

# %%
speeches = {
    "clinton": nlp(clinton_speech),
    "bush": nlp(bush_speech),
    "obama": nlp(obama_speech),
    "trump": nlp(trump_speech),
}

# %%
print("Name", "Chars", "Words", "Unique", "Sentences", sep="\t")
for speaker, speech in speeches.items():
    words = return_words(speech)
    print(speaker, count_chars(words), len(words), len(set(words)), len(list(speech.sents)), sep="\t")


# %% [markdown]
# <div style="font-size: 1em; margin: 1em 0 1em 0; border: 1px solid #86989B; background-color: #f7f7f7; padding: 0;">
# <p style="margin: 0; padding: 0.1em 0 0.1em 0.5em; color: white; border-bottom: 1px solid #86989B; font-weight: bold; background-color: #AFC1C4;">
# Activity
# </p>
# <p style="margin: 0.5em 1em 0.5em 1em; padding: 0;">
# Write a function `avg_sentence_length(blob)` that receives a spaCy `doc` and returns the average number of words in a sentence for the doc. You might need to use our `return_words` function.
# </p>
# </div>

# %%
# average sentence length
def avg_sentence_length(doc):
    return sum(len(return_words(s)) for s in doc.sents) / len(list(doc.sents))


# %%
for speaker, speech in speeches.items():
    print(speaker, avg_sentence_length(speech))

# %% [markdown]
# We might stop to ask why Obama's speech seems to have shorter sentences. Is it deliberate rhetorical choice? Or could it be an issue with the data itself?
#
# In this case, if we look closely at the txt file, we can see that the transcription of the speech included the world 'applause' as a one word sentence throughout the text. Let's see what happens if we filter that out. 

# %%
obama_clean_speech = obama_speech.replace("(Applause.)", "")

# %%
# Let's compare lengths of the texts. We should see a difference.

len(obama_speech), len(obama_clean_speech)

# %%
# Now let's recheck the average sentence length of Obama's speech.
avg_sentence_length(nlp(obama_clean_speech))

# %%
speeches = {
    "clinton": nlp(clinton_speech),
    "bush": nlp(bush_speech),
    "obama": nlp(obama_clean_speech),
    "trump": nlp(trump_speech),
}


# %% [markdown]
# Let's write a quick function to get the most common words used by each person

# %%
def most_common_words(doc, n):
    words = return_words(doc)
    c = Counter(words)
    return c.most_common(n)


# %%
for speaker, speech in speeches.items():
    print(speaker, most_common_words(speech, 10))

# %% [markdown]
# You can see quickly that we need to remove some of these most common words. To do this, we'll use common lists of stopwords.

# %%
from spacy.lang.en.stop_words import STOP_WORDS
print(STOP_WORDS)

# %%
# to make sure we've got all the punctuation out and to remove some contractions, we'll have a custom stoplist
custom_stopwords = [',', '-', '.', '’s', '-', ' ', '(', ')', '--', '---', 'n’t', ';', "'s", "'ve", "  ", "’ve"]


# %%
def most_common_words(doc, n):
    words = [token.text for token in doc if token.pos_ is not 'PUNCT' 
             and token.lower_ not in STOP_WORDS and token.text not in custom_stopwords]
    c = Counter(words)
    return c.most_common(n)


# %%
for speaker, speech in speeches.items():
    print(speaker, ": ", most_common_words(speech, 10), "\n")


# %% [markdown]
# This sort of exploratory work is often the first step in figuring out how to clean a text for text analysis. 

# %% [markdown]
# Let's assess the lexical richness, defined as the ratio of number of unique words by the number of total words.

# %%
def lexical_richness(doc):
    words = return_words(doc)
    return len(set(words)) / len(words)


# %%
for speaker, speech in speeches.items():
    print(speaker, lexical_richness(speech))

# %% [markdown]
# Let's look at the readbility scores for all four speeches now
#
# For the Automated Readability Index, you can get the appropriate grade level here: https://en.wikipedia.org/wiki/Automated_readability_index

# %%
for speaker, speech in speeches.items():
    words = return_words(speech)
    print(speaker, "ARI:", auto_readability_index(speech), "CL:", coleman_liau_index(speech, words))

# %% [markdown]
# To get some comparison, let's also look at some stats calculated through Textacy. We'll see the ARI and CL scores, which use the same formulas we used. However, you might notice that the scores are different. To understand why, you have to dig into the source code for Textacy, where you'll find that it filters out punctuation in creating the word list, which affects the number of characters. It also lowercases the punctuation-filtered words before creating the set of unique words, decreasing that number as well compared to how we calculated it here. These changes affect both the ARI and CL scores.

# %%
!pip install textacy

# %%
import textacy

# %%
# https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
# https://en.wikipedia.org/wiki/Automated_readability_index
txt_speeches = [clinton_speech, bush_speech, obama_clean_speech, trump_speech]
corpus = textacy.Corpus('en', txt_speeches)
for doc in corpus:
    stats = textacy.text_stats.TextStats(doc)
    print({
        "ARI": stats.automated_readability_index,
        "CL": stats.coleman_liau_index,
        "stats": stats.basic_counts
    })

# %% [markdown]
# Why do we have such a significant difference in the CL scores? Let's look quickly at the textacy implementation: https://github.com/chartbeat-labs/textacy/blob/5927d539dd989c090f8a0b0c06ba40bb204fce82/textacy/text_stats.py#L277

# %%
print("Name", "Chars", "Words", "Unique", "Sentences", sep="\t")
for speaker, speech in speeches.items():
    words = return_words(speech)
    print(speaker, count_chars(words), len(words), len(set(words)), len(list(speech.sents)), sep="\t")


# %%
# clinton, bush, obama, trump
for doc in corpus:
    stats = textacy.text_stats.TextStats(doc)
    print({
        "stats": stats.basic_counts
    })

# %% [markdown]
# Post-workshop eval:
#
# https://stanforduniversity.qualtrics.com/jfe/form/SV_aaZ76OCnWDqQbuR
