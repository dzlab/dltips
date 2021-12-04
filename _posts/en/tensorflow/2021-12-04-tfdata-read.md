---
layout: post

title: Read data into Tensorflow Dataset

tip-number: 34
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to use tf.data to read data from memory or disk
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---

In tensorflow `tf.data.Dataset` represents a collection of data that abstracts the complexity of the underlying pipeline needed to read data.

## Read from memory
For quick protyping/testing or just to play with Tensorflow Data API (e.g. batching), we can build a `Dataset` simply by transforming in-memory objects:

Suppose we have an `X` and `Y` tensors, e.g representing synthetic data like this:
```python
import tensorflow as tf

size = 10

X = tf.constant(range(size), dtype=tf.float32)
Y = X * 2 + 1
```

We can create a Dataset from those tensors using `from_tensor_slices`
```python
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
```

Now after creating the collection of examples, we can use `tf.data` to transform it as needed. For instance:
- Repeat the entire dataset multiple times (e.g. same as number of training epochs)
- Divide the data into equal size batches and dropping any remaining objects that does not add up to a complete batch
```python
dataset.repeat(epochs).batch(batch_size, drop_remainder=True)
```

We can check the output and confrim that each batch of the previous dataset is of same size
```python
for i, (x, y) in enumerate(dataset):
  print("x:", x.numpy(), "y:", y.numpy())
```

## Read from disk
In reality, your data would probably some files (e.g. csv), for instance split into train and test, or multiple files with a pattern like this

```
$ ls -l ../data/*.csv
-rw-r--r-- 1 jupyter jupyter 13590 Feb 16 11:37 ../data/train-01.csv
-rw-r--r-- 1 jupyter jupyter 79055 Feb 16 11:37 ../data/train-02.csv
-rw-r--r-- 1 jupyter jupyter 23114 Feb 16 11:37 ../data/train-03.csv
```

We can use `make_csv_dataset` to load those files into a single dataset as follows
```python
# Define column names in same order as in CSV file
columns = ['x1', 'x2', 'y']
# Define default values for each column
defaults = [['na'], ['na'], [0.0]]
# Define files search pattern
pattern = '../data/train-*.csv'
# Read csv file
trainDS = tf.data.experimental.make_csv_dataset(pattern, 1, columns, defaults)
```
We can print the schema of the dataset with `print(trainDS)`
```
<PrefetchDataset shapes: OrderedDict([(x1, (1,)), (x2, (1,)), (y, (1,))]), types: OrderedDict([(x1, tf.float32), (x2, tf.float32), (y, tf.float32)])>
```

We can iterate over the first few element of this dataset using `dataset.take(2)` and print
```python
for data in tempds.take(2):
  pprint({k: v.numpy() for k, v in data.items()})
  print("\n")
```
```
{'x1': array([1.], dtype=float32), 'x2': array([1.], dtype=float32), 'y': array([0.], dtype=float32)}

{'x1': array([2.], dtype=float32), 'x2': array([2.], dtype=float32), 'y': array([2.], dtype=float32)}
```