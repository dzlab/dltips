---
layout: post

title: Load datasets with TorchText

tip-number: 20
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to load text datasets in PyTorch?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - pytorch
---

```python
import torch
from torchtext import data
from torchtext import datasets
```

With TorchText using an included dataset like IMDb is straightforward, as shown  in the following example:
```python
TEXT = data.Field()
LABEL = data.LabelField()

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split()
```
We can also load other data format with TorchText like `csv`/`tsv` or `json`.

## CSV / TSV
TorchText can read CSV/TSV files where each line represent a data sample (optionally a header as first row), e.g.:
```
author  location  age  tweet
John    Rome      23   another lovely day.
Mary    London    19   what a rainy day!
```

Assuming that we have a `data` directory and inside it a bunch of tab-separated files for training: `train.tsv`, `valid.tsv`, `test.tsv`. The following snippet will load this example dataset:

```python
# create Field objects
AUTHOR = data.Field()
AGE = data.Field()
TWEET = data.Field()
LOCATION = data.Field()

# create tuples representing the columns
fields = [
  ('author', AUTHOR),
  ('location', LOCATION),
  (None, None), # ignore age column
  ('tweet', TWEET)
]

# load the dataset in json format
train_ds, valid_ds, test_ds = data.TabularDataset.splits(
   path = 'data',
   train = 'train.tsv',
   validation = 'valid.tsv',
   test = 'test.tsv',
   format = 'tsv',
   fields = fields,
   skip_header = True
)

# check an example
print(vars(train_ds[0]))
```

First, we define a list of tuples `fields` whre in each tuple, the first element represents a name to use as a batch object's attribute, and a `Field` object as second element.

Note, the tuples have to be in the same order as the columns of the tsv file. In the example we use  `(None, None)` to skip the age column. If we wanted to use the author and location columns, we could just use two tuples in the `fields` and the rest will be ignored by TorchText.

Second, we use `TabularDataset.splits` to load the .tsv files into train/validation and test sets. We set `skip_header` flag to `True` to ignore the first row of each file (by default it is set `False`).

Finally, we can check one sample of the training dataset and see how tokenization is applied.

## JSON
TorchText need the json file to have on object per line, as follows:
```
{"author": "John", "location": "Rome", "age": 23, "tweet": ["another", "lovely", "day", "."]}
{"author": "Mary", "location": "London", "age": 19, "tweet": ["what", "a", "rainy", "day", "!"]}
```

Assuming that we have a `data` directory and inside it a bunch of files for training: `train.jon`, `valid.json`, `test.json`. The following snippet will load this example dataset:

```python
# create Field objects
AUTHOR = data.Field()
AGE = data.Field()
TWEET = data.Field()
LOCATION = data.Field()

# create a dictionary representing the dataset
fields = {
  'author': ('author', AUTHOR),
  'age': ('age', AGE),
  'location': ('location', LOCATION),
  'tweet': ('tweet', TWEET)
}

# load the dataset in json format
train_ds, valid_ds, test_ds = data.TabularDataset.splits(
  path = 'data',
  train = 'train.json',
  validation = 'valid.json',
  test = 'test.json',
  format = 'json',
  fields = fields
)

# check an example
print(vars(train_ds[0]))
```
The way the fields are defined is a bit different to csv/tsv. Instead of a list of tuples, we create a python dictionary `fields` where:
* the keys are the same keys in the original json object, i.e. `author`, `location`, `tweet`.
* the values are tuples where the first element will be used as an attribute in each data batch, the second element is a `Field` object.

Then, use `TabularDataset.splits` to create train/test datasets by specifying the file for each dataset and the file format (`json` in  this case).


Finally, we can check one sample of the training dataset and see how tokenization is applied.

> In a JSON file, TorchText tokenize string fields but when given a field containing a list of strings it will assume that the field is already tokenized.

## Iterators
Before creating iterators of the Datasets we need to build the vocabulary for each `Field`  object:
```python
AUTHOR.build_vocab(train_data)
LOCATION.build_vocab(train_data)
TWEET.build_vocab(train_data)
```

To create iterators, we use `BucketIterator.splits` by specifying the datasets, batch size, and a lambda to tell TorchText what key to use for sorting validation/test sets (traning set is shuffled every epoch).

Finally, we can then iterate over batches of the datasets using those iterators.

```python
# determine what device to use
device = torch.device(
  'cuda' if torch.cuda.is_available() else 'cpu'
)

# create iterators for train/valid/test datasets
train_it, valid_it, test_it = data.BucketIterator.splits(
  (train_ds, valid_ds, test_ds),
  sort_key = lambda x: x.author
  sort = True,
  batch_size = 32,
  device = device
)

# iterate over training
for batch in train_it:
  pass
```