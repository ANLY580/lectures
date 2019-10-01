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
# 5. Sentiment Analysis and Naive Bayes
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Note - we're going to spend considerable time talking about the project for next week. I've dropped Python code into Canvas that gives you a start to the problem (basline solution), though with a bit of effort you should be able to do better.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Agenda
* Some types of text classification
* Deep dive into sentiment analysis
* Naive Bayes (theory)
* Project baseline algorithm
* Evaluation
* Practical considerations
* Finishing your project
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
While as data scientists you need to understand as much as possible in order to best select algorithms or trouble-shoot when you encounter problems, J&M leaves implementation up to others. So project # 1 was designed to help you practice skills while developing an understanding of theory.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# What do we mean by text classification
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->

<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Spam
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->

<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Authorship
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->

<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Gender
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->

<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Language
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->

<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Sentiment
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->

<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Topics
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# And other tasks
- Illocutionary acts - Apologizing, promising, ordering, answering, requesting, complaining, warning, inviting, refusing, and congratulating
- genre
- register
- style
- stance
- profiling
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
- First example from Bang & Lee - http://www.cs.cornell.edu/home/llee/papers/cutsent.pdf
- Second example from Bing Liu - https://www.cs.uic.edu/~liub/FBS/NLP-handbook-sentiment-analysis.pdf
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Sentiment Analysis
- Opinions
- Sentiment (polarity)
- Values
- Attitudes
- Feelings (emotions)
![](../images/opinion.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Hutto & Gilbert 2014 - http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf

This is a pretty good definition, though it seems the field is expanding:
> Sentiment analysis, or opinion  mining,is an active  area of study  in  the  field  of  natural  language  processing  that  analyzes people's  opinions,  sentiments,  evaluations,  attitudes, and  emotions  via  the  computational  treatment  of  subjectivity  in  text. 

E S. Kim and E. Hovy, “Automatic detection of opinion bearing words and sentences”, 2005
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Special Tasks
- Subjectivity / objectivity
 - movie review snippets vs plot summaries
 - "The protagonist tries to protect her good name" 
- Aspect-Based
 - "The battery life of this camera is too short"
 - Opinion spam
 - Comparisons
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Vader
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
More complete demo in the __main__ for vaderSentiment.py. The demo has:

        examples of typical use cases for sentiment analysis, including proper handling of sentences with:

                typical negations (e.g., "not good")
                use of contractions as negations (e.g., "wasn't very good")
                conventional use of punctuation to signal increased sentiment intensity (e.g., "Good!!!")
                conventional use of word-shape to signal emphasis (e.g., using ALL CAPS for words/phrases)
                using degree modifiers to alter sentiment intensity (e.g., intensity boosters such as "very" and intensity dampeners such as "kind of")
                understanding many sentiment-laden slang words (e.g., 'sux')
                understanding many sentiment-laden slang words as modifiers such as 'uber' or 'friggin' or 'kinda'
                understanding many sentiment-laden emoticons such as :) and :D
                translating utf-8 encoded emojis such as 💘 and 💋 and 😁
                understanding sentiment-laden initialisms and acronyms (for example: 'lol')

        more examples of tricky sentences that confuse other sentiment analysis tools

        example for how VADER can work in conjunction with NLTK to do sentiment analysis on longer texts...i.e., decomposing paragraphs, articles/reports/publications, or novels into sentence-level analyses

        examples of a concept for assessing the sentiment of images, video, or other tagged multimedia content

        if you have access to the Internet, the demo has an example of how VADER can work with analyzing sentiment of texts in other languages (non-English text sentences).
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Naive Bayes
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
- Based on Bayes rule and relies on a simple "bag of words" representation
 - set of words such as a vector of words
 - Could use all words or a subset
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Definition
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
- Definition: Compute for each class, the conditional probability of a class, given a document.
 - posterior (document given the class) = prior (class) x liklihood (class given the document) / evidence (probability of the document)
 - Most likely class (maximum a posteriori class) = out of all classes the one that maximizes the probability of that class given the document. 
  - We drop the denominator (d) because for each class, we have the same number of documents and its a constant.
  
- This is just like last week. "likelihood" (least errors) and most frequent.
 - In this case, we represent the document by a set of features.
 - P(c) is just like before... how often does the class occur? Just relative frequencies.

<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Simplifying Assumptions
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
- The likelihood has a lot of parameters. More so that what we discussed with edit distance. It's so large we could only estimate it with a HUGE number of training examples. So we make simplifying assumptions.
 - Bag-of-words: position of the word in the document doesn't matter.
 - Conditional independence: the different features are independent given the class.
  - P(x1) given the class x P(x2) given the class... etc. A whole joint string of features

Both assumptions are incorrect. But by making these assumptions we are still able to achieve success despite the incorrectness.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
- Best class that maximizes the prior multiplying by the probability of each feature given the class.
 - For all the classes - for each class, look for the probability of the class and for each position we'll look at the word in that position, what's the probability of the class given that word.
 - Has lots in common with LMs

*If we just use word features and all the the words in text, we have a kind of language model.*
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Naive Bayes baseline (project)
- Let's walk through some code! 
- Vader (lexicon-based)
- Naive Bayes
 - NLTK
 - Scikit-learn
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
I don't want coding to get in the way of learning concepts. You can use these examples for your project and try to improve them. Not sure you can do much to improve Vader, but you can examine results from both to compare the types of errors that you see.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Evaluation
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/eval-measures.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Slide from J&M.


<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/contingency-table.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
For each pair of classes <c1,c2> how many documents from c1 were incorrectly assigned to c2?
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/f-measure.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
a combined measure: F-measure: weighted harmonic mean between precision and recall
- why weighted?  in some applications you may care more about P or R
- why harmonic?  it's conservative -- lower than arith or geo mean
- if P and R are far apart, F tends to be near lower value
- in order to do well on F1, need to do well on BOTH P and R

That said... think about how you communicate f-measure scores. Their meaning will not be obvious so you should explain in plain language what is behind it.

Comment: when ppl say f-measure w/o specifying beta, they mean balanced, and this is by far the most common way of doing it
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/micro-macro.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Slide from J&M

- Macroaveraging is a good strategy for a task where all categories are imporatant and also where one might be smaller than another.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/cross-validation.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Slide from J&M
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/10-fold.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Slide from J&M

"It is common to create a fixed training set and test set, then do 10-fold cross-validation inside the training set, but compute error rate the normal way in the test set."
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Practical considerations
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/realworld1.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Slide from J&M
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/realworld2.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Slide from J&M
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/realworld3.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Slide from J&M
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/realworld4.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Slide from J&M
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/realworld5.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Slide from J&M
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/realworld6.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Slide from J&M
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/realworld7.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Slide from J&M
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
![](../images/realworld8.png)
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Slide from J&M
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Finishing your project

- You have to try to do better than:
 - the baseline naive bayes that I provided
 - the baseline macroaverage score of .33 for any category
- Add the macro-average score
- Generate an output file for your INPUT (test file) that has two columns: id, label
- Submit your report and output file
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
- You can directly use the examples I've given you and try to improve them
 - If you do, you will need to include your pre-processor. I used a separate block of code to generate csv files with features for the classifier.
- You can use your own code from scratch 

- We will score your INPUT file and set up a leaderboard for three categories: highest score, most interesting / innovative approach, best report
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Gold Files
- Why are there four?
- How can you use them?
- What you might look for...
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "notes"}} -->
Thoughts from discussion with an expert participant...
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Questions?
<!-- #endregion -->
