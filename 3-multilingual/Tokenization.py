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

# %%
from nltk.tokenize import RegexpTokenizer

# %%
pattern1 = r'''(?x)     # set flag to allow verbose regexps**
     (?:[A-Z]\.)+       # abbreviations, e.g. U.S.A.
     | \w+(?:-\w+)*       # words with optional internal hyphens
     | \$?\d+(?:\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
'''

pattern2 = r'''(?x)
      (?:[A-Z]\.)+     # abbreviations, e.g. U.S.A
      | \w+(?:-\w+)*   # words with internal hyphens
      | \$?[0-9]+[,-\.]?[0-9]   # currency and percentages e.g., $12.40, 83%
'''


# %%
input = "🥰 Hey!  @sima #roadtrip from 09/09-9/27, aren't you in dude :-D  ?!"

# %%
tokenizer = RegexpTokenizer(pattern2)

# %%
my_tokens = tokenizer.tokenize(input)
my_normalized = [word.lower() for word in my_tokens]
print(my_normalized)

# %%
# WHAT?  There is disappearing text!

# %%
import regex
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
"(?:\p{Space}+)",
"(?:\p{Separator}+)",
"(?:\p{Punctuation}+)"
]
big_regex = ('|').join(patterns)

# %%
#tokenizer = RegexpTokenizer(pattern3)
tokens = regex.findall(big_regex,input)
print(tokens)

# %% [markdown]
# # Using Unicode hex codes for hard-to-type characters
# (AND EMOJIS)
#
# https://r12a.github.io/uniview/#title
#     
#     "\u231a" for 4 digit hex, with a lowercase 'u'
#     "\U0001f430" for 5 to 8 digit hex, with an uppercase 'U', pad with zero

# %%
myinput = "when ⌚ eh Δ? 🐰  woo 🤾🏽‍♀️‍🤾🏽‍♀️"
import regex
patterns = [
    "(?:\p{Other_Symbol}[\U0001f3fb-\U0001f3ff]?\uFE0F?(?:\u200D\p{Other_Symbol}[\U0001f3fb-\U0001f3ff]?\uFE0F?)*)",
    "\u231a",
    "\U00000394",
    "\U0001f430",
    "\p{Letter}+",
    "\p{Punctuation}"
]
big_regex = ('|').join(patterns)
tokens = regex.findall(big_regex,myinput)
print(myinput)
print(tokens)

# %% [markdown]
# # DO I REALLY have to maintain this gigantic regex?!
#
# No.  And you probably shouldn't.  But they are handy for various things such as token tests.  How about trying stanfordnlp's tokenizers and part of speech taggers?
#
# ## Stanford NLP - multi-language tokenization, lemmatization, etc.

# %%
import stanfordnlp

# %%
pipeline_settings = "tokenize,mwt,pos,lemma"
nlp = stanfordnlp.Pipeline(lang='en',processors=pipeline_settings) 
doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
doc.sentences[0].print_tokens()

# %%
# If you don't have this resource, you may need to download
# stanfordnlp.download('ar') 

# %%
pipeline_settings = "tokenize,mwt,pos,lemma"
nlp = stanfordnlp.Pipeline(lang='ar',processors=pipeline_settings) 
doc = nlp("أحب التجديف على نهر بلدي.")
doc.sentences[0].print_tokens()

# %% [markdown]
# # References
#
# StanfordNLP Tutorial and overview:  https://www.analyticsvidhya.com/blog/2019/02/stanfordnlp-nlp-library-python/
#
# Stanford NLP neural etc.  https://github.com/stanfordnlp/
#
# Stanford models:  https://stanfordnlp.github.io/stanfordnlp/models.html#human-languages-supported-by-stanfordnlp
#
# Tagset conversions for specific treebanks to Universal Dependencies: https://stanfordnlp.github.io/stanfordnlp/models.html#human-languages-supported-by-stanfordnlp
#
# A nice article on Chinese segmentation:  https://medium.com/the-artificial-impostor/nlp-four-ways-to-tokenize-chinese-documents-f349eb6ba3c3

# %%
