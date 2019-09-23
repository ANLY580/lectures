# -*- coding: utf-8 -*-
<!-- #region {"slideshow": {"slide_type": "skip"}} -->

---
jupyter:
  jupytext:
    cell_markers: region,endregion
    comment_magics: false
    formats: ipynb,.pct.py:hydrogen,Rmd,md
    text_representation:
      extension: .md
      format_name: percent
      format_version: '1.1'
      jupytext_version: 1.1.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
*ANLY 580: Natural Language Processing for Data Analytics* <br>
*Fall 2019* <br>
# 4. Language Modeling
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Objectives
* Big ideas so far
* Language modeling (ngrams)
* Norvig's Jupyter notebook
* Other associational models
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
# More detailed topics
* Big ideas so far
* A conundrum: Flesch reading score
* Language models
    * unigram model (bag of words)
    * noisy channel
    * ngrams
* Some properties of ngrams
* Calculating ngrams
* Evaluating model fit
* Other associational models
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Big ideas so far
* Variability
* Ambiguity
* Tokenizers
* Multilingual
* Pipelines
<!-- #endregion -->


<!-- #region {"slideshow": {"slide_type": "notes"}} -->
* **Language** is highly **variable** -- it requires context for understanding. The *distributional hypothesis* posits linguistic items with similar distributions share similar meanings. And distributional properties of language extend to other sorts of similarity measures (e.g., POS, named entities, etc.). This is one of the most important ideas in NLP today.

* In this chapter on language modeling, J&M make reference to this idea when talking about *Kneser-Ney discounting* (n-gram smoothing) such that we'd like to use **distributional properties** beyond **absolute frequency** when making predictions with bigrams. (The example was "kong" versus "glasses" when the phrase "Hong Kong" is frequent in a corpus, but "glasses" has a wider distribution.)

* One manifestation of the value of the distributional nature of language use is **ambiguity** of meaning in words (e.g., 'bank' institution versus 'bank' verb versus bank of a river). We use surrounding context for understanding.

* **There is no clear definition of a word**. You will always have decisions to make when tokenizing.

* Choose your tokenizer based on the data, task, and **fit** if you are matching on hashes, model, index etc.

* You can check your assumption by looking at **out-of-vocabulary** words.

* **Multilingual processing** requires even more care since each language and script pair has its own considerations and you will encounter situations where the problems are not **visible**.

* Language processing typically requires **pipeline processes** where analysis occurs in steps. For some processes, the order may matter.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Is this bad science?
![](../images/flesch.jpg)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
> The Globe reviewed the language used by 19 presidential candidates, Democrats and Republicans, in speeches announcing their campaigns for the 2016 presidential election. The review, using a common algorithm called the Flesch-Kincaid readability test that crunches word choice and sentence structure and spits out grade-level rankings, produced some striking results.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# What is Flesch-Kincaid?
![](../images/flesch-kincaid-chart.jpg)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
From Wikipedia:
> The Flesch-Kincaid readability tests are readability tests designed to indicate how difficult a passage in English is to understand. There are two tests, the Flesch Reading Ease, and the Flesch-Kincaid Grade Level. Although they use the same core measures (word length and sentence length), they have different weighting factors.

Original article on ["How to write plain English" (Flesch)](https://web.archive.org/web/20160712094308/http://www.mang.canterbury.ac.nz/writing_guide/writing/flesch.shtml)

From the paper:

> Step 1. Count the words.
Count the words in your piece of writing. Count as single words contractions, hyphenated words, abbreviations, figures, symbols and their combinations, e.g., wouldn't, full-length, TV, 17, &, $15, 7%.

> Step 2. Count the syllables.
Count the syllables in your piece of writing. Count the syllables in words as they are pronounced. Count abbreviations, figures, symbols and their combinations as one-syllable words. If a word has two accepted pronunciations, use the one with fewer syllables. If in doubt, check a dictionary.

>Step 3. Count the sentences.
Count the sentences in your piece of writing. Count as a sentence each full unit of speech marked off by a period, colon, semicolon, dash, question mark or exclamation point. Disregard paragraph breaks, colons, semicolons, dashes or initial capitals within a sentence. For instance, count the following as a single sentence:
You qualify if-
You are at least 58 years old; and
Your total household income is under $5,000.

>Step 4. Figure the average number of syllables per word.
Divide the number of syllables by the number of words.

>Step 5. Figure the average number of words per sentence.
Divide the number of words by the number of sentences.

>Step 6. Find your readability score.

>Find the average sentence length and word length of your piece of writing on the chart (below). Take a straightedge or ruler and connect the two figures. The intersection of the straightedge or ruler with the center column shows your readability score.
<!-- #endregion -->
<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/flesch-kincaid-formula.jpg)
<!-- #endregion -->

```python slideshow={"slide_type": "notes"}
import textstat

test_data = (
    "Playing games has always been thought to be important to "
    "the development of well-balanced and creative children; "
    "however, what part, if any, they should play in the lives "
    "of adults has never been researched that deeply. I believe "
    "that playing games is every bit as important for adults "
    "as for children. Not only is taking time out to play games "
    "with our children and other adults valuable to building "
    "interpersonal relationships but is also a wonderful way "
    "to release built up tension."
)

textstat.flesch_reading_ease(test_data)
```
```python slideshow={"slide_type": "notes"}
# There are also utilities in NLTK that make it easy for you to code this yourself. For example, you can count syllables using cmudict.

# For more languages, there is a library called pyphen. https://pyphen.org

from nltk.corpus import cmudict
from curses.ascii import isdigit

d = cmudict.dict()

def count_syllables(word):
    return([len(list(y for y in x if isdigit(y[-1]))) for x in d[word.lower()]][0])

num_syllables = count_syllables("estimation")
#num_syllables = count_syllables("supercalifragilisticexpialidocious")
num_syllables
```

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Is this bad science?
![](../images/social.jpg)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Articles like the one referenced earlier pop up on a seemingly regular basis. Often, the author is analyzing the *speech* of a speaker and not formal, written text.

Do you see a problem with this?
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/flesch-kincaid.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
<a target="_blank" href="https://www.bostonglobe.com/news/politics/2015/10/20/donald-trump-and-ben-carson-speak-grade-school-level-that-today-voters-can-quickly-grasp/LUCBY6uwQAxiLvvXbVTSUN/story.html">Boston Globe: Oct 20, 2015</a>

"The Globe reviewed the language used by 19 presidential candidates, Democrats and Republicans, in speeches announcing their campaigns for the 2016 presidential election. The review, using a common algorithm called the **Flesch-Kincaid readability test** that crunches word choice and sentence structure and spits out grade-level rankings, produced some striking results."

"The Republican candidates - like Trump - who are speaking at a level easily understood by people at the lower end of the education spectrum are outperforming their highfalutin opponents in the polls. Simpler language resonates with a broader swath of voters in an era of 140-character Twitter tweets and 10-second television sound bites, say specialists on political speech."
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
* Language models
<!-- #endregion -->
<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Flesch-Kincaid is a very simple model intended to capture readability. It's not very complicated, but it is nonetheless mis-used. If you are a statistician, you can think about this as a problem in validity.

Now we're going to discuss language models. The type of language modeling that we're discussing is a very general purpose technique for NLP, though it was designed early on for Speech Recognition.

Hopefully, you've read chapter 3 of J&M and also watched the videos. I'm not going to cover all those same details, but I will highlight a few important points.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slides"}} -->
# Norvig (2) Unigram model or bag of words
https://nbviewer.jupyter.org/url/norvig.com/ipython/How%20to%20Do%20Things%20with%20Words.ipynb
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slides"}} -->
![](../images/noisy-channel-model.jpg)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slides"}} -->
Images from: Julia Hirschberg, http://fayllar.org/julia-hirschberg-v2.html

The basic idea is that we have a noisy channel where we need to pick the best sentence/word/letter by picking the most likely.

It's easy to see the similarity between a noisy channel model for speech recognition, optical character recognition and machine translation.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slides"}} -->
![](../images/noisy-channel-spelling.jpg)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slides"}} -->
Images from:

Here is the same problem but in the context of spelling. Here you can imagine a document where there are spelling errors.

What you might want to do is pick the best hypothesis about the best word to pick.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slides"}} -->
# Let's say...
![](../images/argmax.png)
- A French preposition could be translated as **"in"** or **"on"**.
- And... let's say p(f | e) suggests both:
- in the end zone
- on the end zone
P(e) will prefer: **in** the end zone
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
In this formula,
- argmax is the candidate with the highest combined probability.

The basic idea is that you want to pick a word that maximizes the product of two factors:

- P(f | e) the likelihood and is also known as the **noisy channel model** (or error model) -- it accounts for the variants. (This is the probability of the incorrect word given the correct word)
- P(e) the prior (correct word). This is term is called the **language model** (the probability of the correct word)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Back to spelling...
![](../images/edit-distance-kcg.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Remember edit distance?

This is our noisy channel model!

If you would like to dig deeper into this case study, Manning has a great presentation here: https://slideplayer.com/slide/16489389/

He notes from a Kukich 1992 study that 25-40% of mis-spellings are actual words.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Scoring
![](../images/edit-distance2-kcg.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
The basic idea here is that we get a cost for the number of edits in Pr(t | c) and we multiply those times a measure of frequency for each word.

Both 'actress' and 'across' are very high probability.

More detailed explanation:
Each candidate correction, c, is scored by Pr(c) Pr(t | c), and then normalized by the sum of the scores for all proposed candidates. The prior, Pr(c), is estimated by (freq(c) + 0.5)/N, where freq(c) is the number of times that the word c appears in the 1988 AP corpus (N = 44 million words))
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
"a stellar and versatile **acress** whose combination of sass and glamour..."
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Unfortunately, in the original sentence 'across' is the wrong word.

The unigram model alone could be improved. And, in fact, the authors speculated that more context would improve this algorithm. And that's where we're headed below...
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/bigram-lm.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Norvig (3) Spelling correction (noisy channel model)

https://nbviewer.jupyter.org/url/norvig.com/ipython/How%20to%20Do%20Things%20with%20Words.ipynb
<!-- #endregion -->

So let's move on...
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Markov property
![](../images/markov-property.png)
<!-- #endregion -->
<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Image credit: https://medium.com/ymedialabs-innovation/next-word-prediction-using-markov-model-570fc0475f96

In a process wherein the next state depends only on the current state, such a process is said to follow Markov property.

- The only context considered is the previous observation.
- This is represented as a simple probability distribution where the sum is equal to 1:
> P(like | I) = 0.67
P(love | I) = 0.33
P(fruits | like) = P(Science | like) = 0.5
P(Mathematics | love) = 1
- A sequence of events which follow the Markov model is referred to as the Markov Chain.
- You also need a special symbol such as a STOP or sentence boundary in order to calculate the first observation in a sentence.
- To calculate the conditional probabilities, you first must... tokenize!
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Bigram probability
![](../images/bigram-probability.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
A bigram is simply a sequence of two observations (words). And as J&M note, we use the term "bigram" both as the object (word sequence) and as the predictive model.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Chain Rule
![](../images/chain-rule.jpg)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Image credit: https://www.ibm.com/developerworks/community/blogs/nlp/entry/the_chain_rule_of_probability?lang=en

To calculate a longer sequence than a bigram (such as a trigram), we start to run into some challenges.

Language is very productive!
1. We can't count every possible sentence (recall language is always changing).
2. And we also don't know what all the possible sentences are. We have to make some sort of estimate. This means that a joint probability is not exactly going to work.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Norvig (4)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
# Joint and Conditional Probabilities
![](../images/conditional-probability.jpg)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Credit: https://www.slideshare.net/asdkfjqlwef/text-mining-from-bayes-rule-to-de

Let's take a diversion at this point and look at joint and conditional probabilities.

Joint probability is the likelihood of two *independent* events happening at the same time. [Think of two dice rolls simultaneously.] But you need to know the probability of each event to calculate it.

We're interested in conditional probabilities. [Think of one dice roll following another.]

Multiplication principle:
P(A and B)= P(A) x P(B | A)
We need to take the first event into account when considering the probability of the second event.

J&M show us that it's in fact easier if we try to estimate -- or approximate the history of a sentence by using just the last couple of words. We do this using a maximum likelihood estimation.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# MLE
![]()
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
You may have noticed... we've moved beyond calculating results from our data to making hypotheses about our data.

The idea behind MLE is that to compute a particular bigram probability of a word y given a previous word x, **you can determine the count of the bigram C(xy) and normalize it by the sum of all the bigrams that share the same first-word x.**
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
<!-- #endregion -->


<!-- #region {"slideshow": {"slide_type": "slides"}} -->
Task 1: Probability of sentence (joint probability)
     P(W) = P(w1,w2,w3,w4,w5...wn)
Task 2: probability of an upcoming word:
      P(w5|w1,w2,w3,w4)
A model that computes either of these:
          P(W)     or     P(wn|w1,w2,...wn-1) is called a language model.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# [Google ngram viewer](https://books.google.com/ngrams)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
Shamelessly from [Wikipedia](https://en.wikipedia.org/wiki/Google_Ngram_Viewer):

You can try 1-5 grams.

> The Google Ngram Viewer or Google Books Ngram Viewer is an online search engine that charts the frequencies of any set of comma-delimited search strings using a yearly count of n-grams found in sources printed between 1500 and 2008[1][2][3][4][5] in Google's text corpora in English, Chinese (simplified), French, German, Hebrew, Italian, Russian, or Spanish.[2][6] There are also some specialized English corpora, such as American English, British English, English Fiction, and English One Million; and the 2009 version of most corpora is also available.[7]

> The program can search for a single word or a phrase, including misspellings or gibberish.[6] The n-grams are matched with the text within the selected corpus, optionally using case-sensitive spelling (which compares the exact use of uppercase letters),[3] and, if found in 40 or more books, are then plotted on a graph.[8]

> The Google Ngram Viewer, as of January 2016, supports searches for parts of speech and wildcards.[7]

[The pitfalls of using google ngram to study language](https://www.wired.com/2015/10/pitfalls-of-studying-language-with-google-ngram/)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# GDELT
https://www.forbes.com/sites/kalevleetaru/2019/09/02/using-the-cloud-to-explore-the-linguistic-patterns-of-half-a-trillion-words-of-news-homepage-hyperlinks/#74994b02342b
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Problem...
"The **computer** which I had just put into the machine room on the fifth floor **is** crashing."
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
As J&M note,

> We can extend to trigrams, 4-grams, 5-grams
In general this is an insufficient model of language because language has long-distance dependencies.

But we can often get away with N-gram models.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
# Calculating ngrams
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
# Norvig (9) Evaluating model fit
<!-- #endregion -->
```
