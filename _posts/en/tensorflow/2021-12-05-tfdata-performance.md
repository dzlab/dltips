---
layout: post

title: Improve read performance with TFRecordDataset

tip-number: 35
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Learn how to speed up data processing with TFRecordDataset
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---

Some tips to speed up data processing with `TFRecordDataset`

## Concurrent files processing with interleave
Use [interleave](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#interleave) in TFRecordDataset to process many input files concurrently


```python
filenames = ["./file-01.csv", "./file-02.csv", "./file-03.csv", "./file-04.csv", ...]
dataset = tf.data.Dataset.from_tensor_slices(filenames)
def read_file(filename):
  return tf.data.TFRecordDataset(filename)
dataset = dataset.interleave(lambda x: read_file(x)), cycle_length=2, block_length=4, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
```

In this example we preprocess 2 files concurrently with `cycle_length=2`, interleave blocks of 4 records from each file with `block_length=4`, and let Tensorflow decide how many parallel calls are needed with `num_parallel_calls=tf.data.AUTOTUNE`.


## Prefetch data to improve throughput
Use prefetch to improves latency and throughput during training and avoid GPU starvation.

```python
dataset = tf.data.Dataset.range(10)
dataset.prefetch(2) # prefetches 2 elements
dataset.batch(3).prefetch(2) # prefetches two batches of 3 elements
```

> Note using this comes at the cost of using additional memory to store prefetched elements.