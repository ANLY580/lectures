<!-- #region {"slideshow": {"slide_type": "skip"}} -->

---
jupyter:
  jupytext:
    cell_markers: region,endregion
    comment_magics: false
    formats: ipynb,.pct.py:hydrogen,Rmd,md
    text_representation:
      extension: .py
      format_name: hydrogen
      format_version: '1.1'
      jupytext_version: 1.1.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---
<!-- #endregion -->


<!-- #region {"slideshow": {"slide_type": "notes"}} -->
# Preprocessing Text
<!-- #endregion -->


```python slideshow={"slide_type": "notes"}
# If you are working in binder, you can comment any import statements in the blocks below.

import nltk
import re
from nltk.corpus import brown
from nltk.tokenize import RegexpTokenizer
```

```python
# This is the file from last week.

with open('ca11.txt') as file:
    print(file.read(20))
    ca11_raw = file.read()
```


```python
# Recall how difficult it was to tokenize using a single regular expression and that we 
# couldn't even get all of the number formats in ca11.txt
```
```python
pattern = r'''(?x)     # set flag to allow verbose regexps**
     (?:[A-Z]\.)+       # abbreviations, e.g. U.S.A.
     | \w+(?:-\w+)*       # words with optional internal hyphens
     | \$?\d+(?:\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
'''

tokenizer = RegexpTokenizer(pattern)
my_ca11_tokens = tokenizer.tokenize(ca11_raw)
```

```python
my_ca11_tokens
```

The reality is that you will want to compare your tokenized output with with "gold-standard" tokens, if possible. The NLTK corpus collection includes a sample of Penn Treebank data, including the raw Wall Street Journal text (nltk.corpus.treebank_raw.raw()) and the tokenized version (nltk.corpus.treebank.words()).

Search https://www.nltk.org/genindex.html for tokenize and look closely at the number of tokenizers available.

And if you look here, for example: https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.toktok.ToktokTokenizer.AMPERCENT you will see that many of the tokenizers included use lists of very carefully crafted regular expressions.


TODO: compiled regular expressions


There are a number of advantages and disadvantages to tokenizers in NLTK. For example, 
- ToktokTokenizer() is very fast
- MosesTokenizer() is backwards capable and can detokenize text
- ReppTokenizer() is able to provide token offsets

Actually - the MosesTokenizer seems to have been moved to: https://github.com/alvations/sacremoses


Changing gears to short texts like those in twitter or other social media, there are some patterns that require specialized tokenizers and pre-processing steps.

Below are the first few tweets from assignment 1 - Twitter English data.

```python
with open('social.txt', encoding="utf-8") as file:
    tweets=[]
    data = file.readlines()
    for tweet in data:
        tweets.append(tweet)
    print(tweets)
```

Even for English, there are potentially number of challenges for tokenization:
- mentions/usernames
- URLs, numbers
- textual [emoticons](https://en.wikipedia.org/wiki/List_of_emoticons)
- [emoji](https://en.wikipedia.org/wiki/Emoji)
- words (including hyphenated words)
- case-folding 
- punctuation 
- hashtags 
- non-English words

```python
# First let's check out the Whitespace Tokenizer from last week
whitespace_tokens=[]
for tweet in tweets:
    whitespace_tokens.append(tweet.split())
print(whitespace_tokens)
```

Note that all of these tokenizers use **compiled regular expressions** which are cached and may result in substantial performance gain, depending on how often you use them and possibly how many you have.

```python
# NLTK also has a "tweet-aware" tokenizer with some options
# https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.casual.casual_tokenize

nltk_casual_tokens=[]
for tweet in tweets:
    nltk_casual_tokens.append(nltk.casual_tokenize(tweet))
print(nltk_casual_tokens)

# How does this differ from what you see above? One way to test is to compare tokens
# that don't occur from one to the other.
```

```python
TODO: re.UNICODE
```

```python
# A somewhat better way to write a tokenizer with multiple
# regular expressions is in this snippet below (shorter version of 
# http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py)

# The order is important (match from first to last)

# Keep usernames together (any token starting with @, followed by A-Z, a-z, 0-9)
regexes=(r"(?:@[\w_]+)",

# Keep hashtags together (any token starting with #, followed by A-Z, a-z, 0-9, _, or -)
r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",

# Keep words with apostrophes, hyphens and underscores together
r"(?:[a-z][a-z’'\-_]+[a-z])",

# Keep all other sequences of A-Z, a-z, 0-9, _ together
r"(?:[\w_]+)",

# Everything else that's not whitespace
r"(?:\S)"
)

big_regex="|".join(regexes)

my_extensible_tokenizer = re.compile(big_regex, re.VERBOSE | re.I | re.UNICODE)

def my_extensible_tokenize(text):
    return my_extensible_tokenizer.findall(text)

# Note re.I for performing Perform case-insensitive matching; 
# expressions like [A-Z] will match lowercase letters, too. 
# You get the same effect onnon-ASCII Unicode characters such as ü and Ü, 
# by adding the UNICODE flag.
```

```python
extensible_tokens=[]
for tweet in tweets:
    extensible_tokens.append(my_extensible_tokenize(tweet))
print(extensible_tokens)
```
