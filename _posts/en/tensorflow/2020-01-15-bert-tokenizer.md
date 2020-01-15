---
layout: post

title: BERT Tokenization

tip-number: 11
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to create a BERT Tokenizer with TensorFlow Text and TensorFlow Hub
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---

As prerequisite, we need to install TensorFlow Text library as follows:
```
pip install tensorflow_text -q
```

Then import dependencies
```python
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tftext
```

## Download vocabulary
Download BERT vocabulary from a pretrained BERT model on TensorFlow Hub (BERT preptrained models can be found [here](https://tfhub.dev/s?module-type=text-embedding&q=bert))
```python
BERT_URL = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1'
bert_layer = hub.KerasLayer(BERT_URL, trainable=False)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
```

At this moment, the vocabulary file will be avilable at `vocab_file` location, and the `do_lower_case` flag will be indicating whether BERT pretrained model is case sensitive or not.
```python
print(f'BERT vocab is stored at     : {vocab_file}')
print(f'BERT model is case sensitive: {do_lower_case}')
```

## Build Tokenizer
First, we need to load the downloaded vocabulary file into a list where each element is a BERT token.
```python
def load_vocab(vocab_file):
  """Load a vocabulary file into a list."""
  vocab = []
  with tf.io.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = reader.readline()
      if not token: break
      token = token.strip()
      vocab.append(token)
  return vocab

vocab = load_vocab(vocab_file)
```

Second, build a vocab lookup table using as input the created `vocab` list
```python
def create_vocab_table(vocab, num_oov=1):
  """Create a lookup table for a vocabulary"""
  vocab_values = tf.range(tf.size(vocab, out_type=tf.int64), dtype=tf.int64)
  init = tf.lookup.KeyValueTensorInitializer(keys=vocab, values=vocab_values, key_dtype=tf.string, value_dtype=tf.int64)
  vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov, lookup_key_dtype=tf.string)
  return vocab_table

vocab_lookup_table = create_vocab_table(vocab)
```

Finally, we can create a `BertTokenizer` instance as follows
```python
tokenizer = tftext.BertTokenizer(
    vocab_lookup_table,
    token_out_type=tf.int64,
    lower_case=do_lower_case
  )
```

## Examples
```python
>>> tokenizer.tokenize(["the brown fox jumped over the lazy dog"])
<tf.RaggedTensor [[[1103], [3058], [17594], [4874], [1166], [1103], [16688], [3676]]]>
```

To learn more about TF Text check this detailed introduction - [link](https://dzlab.github.io/nlp/2019/12/25/tensorflow-text-intro/).