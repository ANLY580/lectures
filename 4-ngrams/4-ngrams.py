# -*- coding: utf-8 -*-
# %% [markdown] {"slideshow": {"slide_type": "skip"}}
#
# ---
# jupyter:
#   jupytext:
#     cell_markers: region,endregion
#     comment_magics: false
#     formats: ipynb,.pct.py:hydrogen,Rmd,md
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.1'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


# %% [markdown] {"slideshow": {"slide_type": "notes"}}
# # Preprocessing Text


# %% {"slideshow": {"slide_type": "notes"}}
# If you are working in binder, you can comment any import statements in the blocks below.

import nltk
from nltk.tokenize import RegexpTokenizer
import regex
import string
from nltk.corpus import stopwords

# %% [markdown]
# Last week we saw Unicode challenges with the python **re** library and moved to the **regex** library. In fact, we lost tokens that we hadn't intended.
#
# We also learned a slightly more efficient way to manage a large regular expression (though the one below could still benefit from documentation). This pattern was created to handle social text like that from Twitter before the availabilty of fairly good "Twitter-aware" tokenizers.

# %%
patterns = [
"(?:[°\p{Punctuation}\p{Modifier_Symbol}\p{Math_Symbol}}]+(?:[\p{Letter}\p{Number}][°\p{Punctuation}\p{Modifier_Symbol}\p{Math_Symbol}]+))",
"(?:\@+\p{Letter}+)",
"(?:\:\-\p{Letter}+)",
"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",
"(?:norm__?[_A-Z]+)",
"(?:\p{Letter}\.(?:\p{Letter}\.)+)",
"(?:\p{Letter}\/\p{Letter}(?:\/\p{Letter})*)",
"(?:\b(?:(?:http|ftp|file)s?:\/\/\S+\w+\.\w+\.(?:\w\.)?(?:com|edu|gov|org|info|biz|mil|net)?(?:[a-z]{2})?\S+)\b)|(?:&(?:#?[0-9a-f]+|[a-z]+);)",
"(?:\b[0-9]*(?:1st|2nd|3rd|11th|12th|13th|[4-9]th)\b)",
"(?:[.+\-]?\p{Number}+(?:[.,\-:\/]*\p{Number}+)*)",
"(?:(?:n[\'’]t\b)|(?:[\'’](?:[sdm]|(?:ld)|(?:ll)|(?:re)|(?:ve)|(?:nt))\b))",
"(?:[\p{Letter}\p{Mark}]+(?:[\-\'’][\p{Letter}\p{Mark}]+)*)",
"(?:\.\.+|--+|__+|~~+|!!+|\*\*+|\?\?+|//+)",
"(?:\.\.\.+|---+|___+|~~~+|!!!+|\*\*\*+|\?\?\?+|///+)",#"(?:<\w+\/>)",
"(?:[\@]?\p{Letter}+)",
#"(?:\p{Space}+)",
#"(?:\p{Separator}+)",
"(?:\p{Punctuation}+)"
]
big_regex = ('|').join(patterns)

# %%
input = "🥰 Hey!  @sima #roadtrip from 09/09-9/27, aren't you in dude :-D  ?!"

# %%
tokens = regex.findall(big_regex,input)
print(tokens)

# %% [markdown]
# There are a number of advantages and disadvantages to tokenizers in NLTK. For example, 
# - ToktokTokenizer() is very fast
# - MosesTokenizer() is backwards capable and can detokenize text
# - ReppTokenizer() is able to provide token offsets
#
# Actually - the MosesTokenizer seems to have been moved to: https://github.com/alvations/sacremoses

# %% [markdown]
# As you can see, for short texts (like those in twitter or other social media), there are some patterns that require specialized tokenizers and pre-processing steps.
#
# Below are the first few tweets from assignment 1 - Twitter English data.

# %%
with open('data/social.txt', encoding="utf-8") as file:
    tweets=[]
    data = file.readlines()
    for tweet in data:
        tweets.append(tweet)
    print(tweets)

# %% [markdown]
# Even for English short texts, there are potentially number of challenges for tokenization:
# - mentions/usernames
# - URLs, numbers
# - textual [emoticons](https://en.wikipedia.org/wiki/List_of_emoticons)
# - [emoji](https://en.wikipedia.org/wiki/Emoji)
# - words (including hyphenated words)
# - case-folding 
# - punctuation 
# - hashtags 
# - non-English words

# %% [markdown]
# Note for performance reasons, many tokenizers use **compiled regular expressions** which are cached and may result in substantial performance gain, depending on how often you use them and possibly how many you have.
#
# Check the source code here to examine:
# https://www.nltk.org/_modules/nltk/tokenize/casual.html#TweetTokenizer

# %%
# Here is NLTK's "tweet-aware" tokenizer

nltk_casual_tokens=[]
for tweet in tweets:
    nltk_casual_tokens.append(nltk.casual_tokenize(tweet))
print(nltk_casual_tokens)

# See if you can spot a potential problem with one of the emoji.

# %% [markdown]
# Because language is always changing, you may still need to customize tokenization to account for what you are trying to accomplish.
#
# Below, you can see a customization of the big_regex above using compiled regular expressions.

# %%
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'(?:[°\p{Punctuation}\p{Modifier_Symbol}\p{Math_Symbol}}]+(?:[\p{Letter}\p{Number}][°\p{Punctuation}\p{Modifier_Symbol}\p{Math_Symbol}]+))',
    r'(?:\@+\p{Letter}+)',
    r'(?:\:\-\p{Letter}+)',
    r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)',
    r'(?:norm__?[_A-Z]+)',
    r'(?:\p{Letter}\.(?:\p{Letter}\.)+)',
    r'(?:\p{Letter}\/\p{Letter}(?:\/\p{Letter})*)',
    #r'(?:\b(?:(?:http|ftp|file)s?:\/\/\S+\w+\.\w+\.(?:\w\.)?(?:com|edu|gov|org|info|biz|mil|net)?(?:[a-z]{2})?\S+)\b)|(?:&(?:#?[0-9a-f]+|[a-z]+);)',
    r'(?:\b[0-9]*(?:1st|2nd|3rd|11th|12th|13th|[4-9]th)\b)',
    r'(?:[.+\-]?\p{Number}+(?:[.,\-:\/]*\p{Number}+)*)',
    r'(?:(?:n[\'’]t\b)|(?:[\'’](?:[sdm]|(?:ld)|(?:ll)|(?:re)|(?:ve)|(?:nt))\b))',
    r'(?:[\p{Letter}\p{Mark}]+(?:[\-\'’][\p{Letter}\p{Mark}]+)*)',
    r'(?:\.\.+|--+|__+|~~+|!!+|\*\*+|\?\?+|//+)',
    r'(?:\.\.\.+|---+|___+|~~~+|!!!+|\*\*\*+|\?\?\?+|///+)',#'(?:<\w+\/>)',
    r'(?:[\@]?\p{Letter}+)',
#    r'(?:\p{Punctuation}+)'
]

tokens_regex = regex.compile(r'(' + '|'.join(regex_str) + ')', regex.VERBOSE | regex.IGNORECASE)
emoticon_regex = regex.compile(r'^' + emoticons_str + '$', regex.VERBOSE | regex.IGNORECASE)

def tokenize(s):
    return tokens_regex.findall(s)


# %%
tokenize(input)

# %% [markdown]
# Finally, if you are doing feature extraction for building a classifier, you may later want to intentionally remove tokens.

# %%
# process text
tokens = nltk.casual_tokenize(input)

punctuation = list(string.punctuation)

# remove stopwords
tokens = [term.lower() for term in tokens if term.lower() not in stopwords.words('english')]

# remove punctuation
tokens = [term for term in tokens if term not in punctuation]

# remove hashtags
tokens = [term for term in tokens if not term.startswith('#')]

# remove profiles
tokens = [term for term in tokens if not term.startswith('@')]

# %%
tokens

# %% [markdown]
# Regardless, you will want to compare your tokenized output with with "gold-standard" tokens, if possible. The NLTK corpus collection includes a sample of Penn Treebank data, including the raw Wall Street Journal text (nltk.corpus.treebank_raw.raw()) and the tokenized version (nltk.corpus.treebank.words()).
#
# Search https://www.nltk.org/genindex.html for tokenize and look closely at the number of tokenizers available.
#
# And if you look here, for example: https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.toktok.ToktokTokenizer.AMPERCENT you will see that many of the tokenizers included use lists of very carefully crafted regular expressions.
