---
layout: post

title: Basic NLP with PyTorch Text

tip-number: 04
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: What should I use for text processing in PyTorch?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - pytorch
---

[PyTorch Text](https://github.com/pytorch/text) is a PyTorch package with a collection of text data processing utilities, it enables to do basic NLP tasks within PyTorch. It provides the following capabilities:
* Defining a text preprocessing pipeline: tokenization, lowecasting, etc.
* Building Batches and Datasets, and spliting them into (train, validation, test)
* Data Loader for a custom NLP dataset

```python
import torchtext
```

[torchtext.data.Field](torchtext.data.Field) is a base datatype of PyTorch Text that helps with text preprocessing: tokenization, lowercasting, padding, umericalizaion and Building vocabulary.

```python
TEXT = torchtext.data.Field(
  tokenize    = spacy_tokenizer,
  lower       = True,
  batch_first = True,
  init_token  = '<bos>',
  eos_token   = '<eos>',
  fix_length  = seq_len
)
```

Tokenizing and Lowercasting

```python
minibatch = [ 'The Brown Fox Jumped Over The Lazy Dog' ]
minibatch = list(map(TEXT.preprocess, minibatch))
```

Padding text sequence to match the fixed sequence length
```python
minibatch = TEXT.pad(minibatch)
```

Before being able to numericalize, we first need to build vocab:

1. Count the frequencies of tokens in all documents and build a vocab using the tokens frequencies
```python
tokens = functools.reduce(operator.concat, minibatch)
counter = Counter(tokens)
counter

TEXT.vocab = TEXT.vocab_cls(counter)
```

It is also possible to build a vocab directly as follows:
```python
TEXT.build_vocab(minibatch)
```

3. Finally numericalize using the constructed vocabulary
```python
TEXT.numericalize(minibatch)
```


Notebook - [link](https://github.com/pytorch/text)