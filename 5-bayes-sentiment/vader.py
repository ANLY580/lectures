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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import nltk
# nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# %%
analyzer = SentimentIntensityAnalyzer()
# While this lexicon-based approach doesn't require training -- you can
# potentially improve Vader with your own lexical data
# new_words = {
#     'stoked': 2.0,
#     'lame': -3.4,
# }
# analyzer.lexicon.update(new_words)

# %%
# Let's take a look against some gold data
DIR = 'data/'
TRAIN = DIR + 'twitter-2016train-A.txt'

with open(TRAIN, 'r') as dataset:
    my_lines = [next(dataset) for x in range(10)]

# %%
for my_sentence in my_lines:
    my_sentence = my_sentence.strip().split('\t')
    id, label, text = my_sentence[0], my_sentence[1], ' '.join(my_sentence[2:])
    vs = analyzer.polarity_scores(text)
    print("{:-<65} {} {}".format(text, str(vs), label))

# %%
# Below from https://github.com/cjhutto/vaderSentiment

sentences = ["VADER is smart, handsome, and funny.",  # positive sentence example
             "VADER is smart, handsome, and funny!",  # punctuation emphasis handled correctly (sentiment intensity adjusted)
             "VADER is very smart, handsome, and funny.", # booster words handled correctly (sentiment intensity adjusted)
             "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
             "VADER is VERY SMART, handsome, and FUNNY!!!", # combination of signals - VADER appropriately adjusts intensity
             "VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!", # booster words & punctuation make this close to ceiling for score
             "VADER is not smart, handsome, nor funny.",  # negation sentence example
             "The book was good.",  # positive sentence
             "At least it isn't a horrible book.",  # negated negative sentence with contraction
             "The book was only kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
             "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
             "Today SUX!",  # negative slang with capitalization emphasis
             "Today only kinda sux! But I'll get by, lol", # mixed sentiment example with slang and constrastive conjunction "but"
             "Make sure you :) or :D today!",  # emoticons handled
             "Catch utf-8 emoji such as such as 💘 and 💋 and 😁",  # emojis handled
             "Not bad at all"  # Capitalized negation
             ]

# %%
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence, str(vs)))

# %%
# Other things you can do from nltk:
nltk.sentiment.util.demo_liu_hu_lexicon("The plot was good, but the characters are uncompelling and the dialog is not great.", plot=True)

# %%
