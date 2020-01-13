---
layout: post

title: TF-IDF with TextVectorization

tip-number: 09
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to calculate the TF-IDF matrix of an input text with TensorFlow?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---

`TextVectorization` is an experimental layer for raw text preprocessing: text normalization/standardization, tokenization, n-gram generation, and vocabulary indexing.

This layer can also be used to calculate the TF-IDF matrix of a corpus.

> TF-IDF is a score that intended to reflect how important a word is to a document in a collection or corpus.

First, import `TextVectorization` class which is in an experimental package for now.

```python
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
```

Second, define an instance that will calculate TF-IDF matrix by setting the `output_mode` properly.

```python
tfidf_calculator = TextVectorization(
  standardize = 'lower_and_strip_punctuation',
  split       = 'whitespace',
  max_tokens  = MAX_TOKENS,
  output_mode ='tf-idf',
  pad_to_max_tokens=False)
```

Third, we build the vocab.
```python
tfidf_calculator.adapt(text_input)
```

Finally, we call the layer on the text to get a dense TF-IDF matrix.
```python
tfids = tfidf_calculator(text_input)
```

Example notebook [here](https://github.com/dzlab/deepprojects/blob/master/tensorflow/wiki_clustering_projector.ipynb).