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

## Text processing
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

1- Count the frequencies of tokens in all documents and build a vocab using the tokens frequencies

```python
tokens = functools.reduce(operator.concat, minibatch)
counter = Counter(tokens)
counter

TEXT.vocab = TEXT.vocab_cls(counter)
```

It is also possible to build a vocab directly as follows
```python
TEXT.build_vocab(minibatch)
```

2- Finally numericalize using the constructed vocabulary
```python
TEXT.numericalize(minibatch)
```

## Data Loader
Build a dataset given a training and validation text files, and using the previously built text processing pipeline.

```python
train_ds, valid_ds = tt.data.TabularDataset.splits(
  path       = PATH,
  train      = 'train.csv',
  validation = 'valid.csv',
  format     = 'csv',
  fields     = [('text', TEXT)]
)
```

### Data Loader for Language Modeling
This dataset can be used to build an iterator that produces data for multiple NLP Tasks. For instance, to build the samples to use for Language Modeling using [torchtext.data.BPTTIterator](https://torchtext.readthedocs.io/en/latest/data.html#bpttiterator).

```python
def dataset2example(dataset, field):
  examples = list(map(lambda example: ['<bos>']+ example.text + ['<eos>'], dataset.examples)
  examples = [item for example in examples for item in example]
  example = tt.data.Example()
  setattr(example, 'text', examples)
  return tt.data.Dataset([example], fields={'text': field})

train_example = dataset2example(train_ds, TEXT)
valid_example = dataset2example(valid_ds, TEXT)

train_iter, valid_iter = tt.data.BPTTIterator.splits(
  (train_example, valid_example),
  batch_size = batch_size,
  bptt_len   = 30
)
```

The resulting `train_iter` and `valid_iter` are iterators over batches of samples that can be used in a training loop.

Notebook - [link](https://github.com/dzlab/deepprojects/blob/master/nlp/Basic_NLP_with_PyTorch_Text.ipynb)