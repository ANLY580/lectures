```python slideshow={"slide_type": "skip"}
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
```

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
*ANLY 580: Natural Language Processing for Data Analytics* <br>
*Fall 2019* <br>
# 2. Tools
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Today's Plan
- Overview of tools
- Hands-on: Tokenization
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Basically, today I want to make sure that everyone is able to work with NLTK and Python.

If you are unfamiliar with Linux, it's important to work through the tutorial for that. I've included the some references for the commands that Jurafsky used for text processing in Linux in the [supplement file here](https://anyl580.github.io/syllabus/2-Tools.html).
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# What you should do BEFORE class

- Reading (you know this)
- Watch associated J&M videos
- Look at the lecture supplement
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
There's little point in my regurgitating the reading -- particularly when Jurafsky and Manning do so well explaining the concepts. Plus, your quizzes assume you did this work. So I'll be focusing on context -- what you can't get directly from J&M or the readings.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Tools
- Development tools
- Industrial tools
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Hands-on: Tokenization
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
For the remainder of this session, walk through the material here and in the supplemental and we'll make sure everyone is set with GitHub and the basics before we move on to language modeling (next week).
<!-- #endregion -->

```python slideshow={"slide_type": "notes"}
# If you are not working in binder, uncomment any import statements in the blocks below.

import nltk
```

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
You'll need also to have the brown corpus (or any others that interest you). Again, if you are not in binder, you can use the following to download corpora. But this command you type in your terminal. Make sure you are in the same environment as your Jupyter notebook. If you don't know what this means, it's likely not an issue.

**python -m nltk.downloader brown**

We've also included versions of the tagged and untagged brown corpora in Canvas and in the lectures repository. When you git clone the repo (https://help.github.com/en/desktop/contributing-to-projects/cloning-a-repository-from-github-to-github-desktop),you will get copies of these. If you are working in R, you may need to do this.

Also in the repo are two individual files from the brown corpus. If you are new to Python, you also need to learn how to load files not included in libraries. These will give you an opportunity to do so.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Hopefully, you've already browsed through chapters 1 & 2 in NLTK. We'll be working from material in **chapter 3** this week and again in the fourth session of this class when we begin to learn about language modeling.

<!-- #endregion -->

```python slideshow={"slide_type": "notes"}
from nltk.corpus import brown
```

```python slideshow={"slide_type": "notes"}
# In your anaconda distribution, you should be able to use basic unix commands. 
# For example, try the ls command below to make sure this is the case.

!ls
```

```python slideshow={"slide_type": "notes"}
# call.txt has no line breaks. Note what happens when you use the head command.
!head ca11.txt
```

```python slideshow={"slide_type": "notes"}
# Another way to view this file using standard Python is the following. 
# This time, we are only reading some number of chracters.
# I'm a fan of realpython and if you are uncomfortable with basics in Python, check it out.
# https://realpython.com/read-write-files-python/
# The command below ensures that your file is closed after the block is executed.

with open('ca11.txt') as file:
    print(file.read(20))
    ca11_raw = file.read()
```

```python slideshow={"slide_type": "notes"}
# That said, you have some neat functions already packaged in NLTK to make things easier.
# For example:

brown.sents('ca11')
```

```python slideshow={"slide_type": "notes"}
# Okay, so let's play with tokenizers on a single file from Brown

from nltk.text import Text
ca11 = nltk.Text(brown.words(fileids=['ca11']))

# Note that if you are using the Text object that your text in ca11 is accessible via methods on this object.
```

```python slideshow={"slide_type": "notes"}
ca11.count("the")
ca11.concordance("start")
ca11.vocab()
```

```python slideshow={"slide_type": "notes"}
# API documentation here: https://www.nltk.org/api/nltk.tokenize.html

# While NLTK gives you some nice capabilities with its included corpora, generally speaking
# you have to handle tokenization yourself via the use of APIs in tools like SpaCy. Later, 
# when we get to the module where we use SpaCy we'll do some performance comparisons
# between NLTK and SpaCy for tokenization. But for how, let's use NLTK.

from nltk.tokenize import WhitespaceTokenizer

WhitespaceTokenizer().tokenize(ca11_raw)
```

```python slideshow={"slide_type": "notes"}
# Exercise 1: Whitespace tokenization

# How many tokens are there?

# YOUR SOLUTION HERE

# What do you see that looks like a problem?
```

```python slideshow={"slide_type": "slide"}
# Exercise 2: Regular expression tokenization

#https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.regexp.RegexpTokenizer
from nltk.tokenize import RegexpTokenizer

# From the documentation this is a tokenizer that splits a string using a regular expression, 
# which matches either the tokens or the separators between tokens.

# tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')

# Tokenization is a task you need to become proficient with. Sometimes included
# tokenizers do what you want, but other times you will need to provide additional
# pre-processing to ensure text is tokenized the way you would like it to be.

# With some help from rexpr, write a regular expression that tokenizes the text, 
# handling the problems you noted above.

# YOUR SOLUTION HERE

# How many tokens are there?

# YOUR SOLUTION HERE

# Could you fix everything you noted was wrong from the Whitespace tokenizer?

```

```python slideshow={"slide_type": "notes"}
# Exercise 3: Tagged Text

# One of the really neat things about the Brown Corpus (and others like it), are the additional annotations
# that give you more information about the text and word distributions in it.
# NLTK includes a probability module with the ability to collect conditional frequency 
# distribution over tokens in text.

# Let's switch to the entire Brown corpus now, and also use the tokenization provided by 
# the Text object.

# API documentation here:
# https://www.nltk.org/api/nltk.html#nltk.probability.ConditionalFreqDist

from nltk.probability import ConditionalFreqDist
from nltk.probability import FreqDist

# From the documentation, A frequency distribution records the number of times 
# each outcome of an experiment has occurred. For example, a frequency distribution 
# could be used to record the frequency of each word type in a document.

# fdist = FreqDist(word.lower() for word in word_tokenize(sent))

# Conditional frequency distributions are used to record the number of times each sample 
# occurred, given the condition under which the experiment was run. 
# For example, a conditional frequency distribution could be 
# used to record the frequency of each word (type) in a document, given its length.


```

```python slideshow={"slide_type": "notes"}
# Exercise 3: Tagged text (continued)

# What we'd like to do now is look at combinations of word types and tags in the Brown Corpus.

freq_dist = FreqDist()
cond_freq_dist = ConditionalFreqDist()

# Tagged words are already in tuples
brown.tagged_words()[:10]
```

```python
# Given a word, list the possible tags for that word with its frequency count.

# Example: a particular word should generate a list like [('nn', 12), ('vb', 22)]
# Then you might need to sort and reverse a list of tuples
# such as word_freq = [(y,x) for (x,y) in freq_word]

# YOUR SOLUTION HERE
```

```python slideshow={"slide_type": "notes"}
# Exercise 3: Tagged text (continued)

# Write a function that takes a word and gives you a frequency distribution

# YOUR SOLUTION HERE


# Write a function that takes a word and gives you a probability distribution

# YOUR SOLUTION HERE

```

```python slideshow={"slide_type": "notes"}
# Exercise 4: Ambiguous words

# Find the word which have the greatest variety of tags.

# YOUR SOLUTION HERE

```

```python slideshow={"slide_type": "notes"}
# Exercise 5: Ambiguity in the corpus

# How many ambiguous word types are there? 

# YOUR SOLUTION HERE

# What is the percentage of ambiguous words across the entire vocabulary?

# YOUR SOLUTION HERE
```
