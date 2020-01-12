---
layout: post

title: How to use TextVectorization layer

tip-number: 07
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to preprocess text in TensorFlow using TextVectorization.
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---

> pip install tf-nightly -q

The release of TF 2.1 introduced many new features like the introduction of `TextVectorization`. This layer performs preprocessing of raw text: text normalization/standardization, tokenization, n-gram generation, and vocabulary indexing.

This class takes the following arguments:

| Argument        | Default | Description           |
| ------------- | ------------- | ------------- |
| `max_tokens` | `None` (no limit) | Maximum size of the vocabulary |
| `standardize` | `lower_and_strip_punctuation` | Function to call for text standardization, it can be None (no standardization). |
| `split` | `whitespace` | Function to use for splitting. |
| `ngrams` | `None` | Integer or tuple representing how many ngrams to create |
| `output_mode` | `int` | output of the layer, `int`: for token indices, `binary`: to output an array of 1s where each 1 means the token is available in the text. `count`: similary to `binary` except instead of 1s the output array will contain token count. `tf-idf` similar to `binary` except the values are calculated with the TF-IDF algorithm. |
| `output_sequence_length` | `None` | Valid for `int` mode, it will be used to pad the text up to this length. |
| `pad_to_max_tokens` | `True` | Valid for  `binary`, `count`, and `tf-idf` modes. A flag idicating whether or not to pad output up to `max_tokens`. |

## HowTo
First, look at the raw data (in training set) to figure out the type of normalization and tokenization needed as well as checking they are producing expected result.

Second, define a function that will get as input raw text and clean it, e.g. punctuations and any contain HTML tags.

```python
def normlize(text):
  remove_regex = f'[{re.escape(string.punctuation)}]'
  space_regex = '...'
  result = tf.strings.lower(text)
  result = tf.strings.regex_replace(result, remove_regex, '')
  result = tf.strings.regex_replace(result, space_regex, ' ')
  return result
```

Third, define a `TextVectorization` layer that will take the previously defined `normalize` function as well as define the shape of the output.

```python
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

vectorize_layer = TextVectorization(
  standardize=normlize,
  max_tokens=MAX_TOKENS_NUM,
  output_mode='int',
  output_sequence_length=MAX_SEQUENCE_LEN)
```

Forth, call the vectorization layer `adapt` method to build the vocabulry.
```python
vectorize_layer.adapt(text_dataset)
```

Finally, the layer can be used in a Keras model just like any other layer.
```python
MAX_TOKENS_NUM = 5000  # Maximum vocab size.
MAX_SEQUENCE_LEN = 40  # Sequence length to pad the outputs to.
EMBEDDING_DIMS = 100

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)
model.add(tf.keras.layers.Embedding(MAX_TOKENS_NUM + 1, EMBEDDING_DIMS))
```

One here that the input layer needs to have a shape of (1,) so that we have one string per item in a batch. Also, the embedding layer takes an input of `MAX_TOKENS_NUM+1` because we are counting the padding token.

Check TF 2.1.0 release note [here](https://github.com/tensorflow/tensorflow/releases/tag/v2.1.0).