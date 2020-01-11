---
layout: post

title: Text Preprocessing with Keras

tip-number: 06
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Keras provides a set of very helpful text preprocessing utilities.
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - keras
---

Preprocessing can be very tedious depending on the data format (e.g. json, xml, binary) and how your model is expecting it (e.g. a fixed sequence length). Keras provides an API for preprocessing different kind of raw data Image or Text that's very important to know about.

## Sequence Preprocessing
The `keras.preprocessing` package have a sequence processing helpers for sequence data preprocessing, either text data or timeseries.

### Sequence Padding
You can use `pad_sequences` to add padding to your data so that the result would have same format.
```python
from keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(X_train_raw, maxlen=80)
X_test  = pad_sequences(X_test_raw, maxlen=80)
```

### Skip Grams
You can use `skipgrams` to generate skipgram word pairs.

### Sampling
You can use `make_sampling_table` to enerate word rank-based probabilistic sampling table.

## Text Preprocessing
The Keras package `keras.preprocessing.text` provides many tools specific for text processing with a main class `Tokenizer`. In addition, it has following utilities:
* `one_hot` to one-hot encode text to word indices
* `hashing_trick` to converts a text to a sequence of indexes in a fixed- size hashing space

### Tokenization
* Use `fit_on_texts` to update the tokenizer internal vocabulary based on a list of texts.
* Use `fit_on_sequences` to update the tokenizer internal vocabulary based on a list of sequences.

### Numericalization
* Use  `texts_to_sequences` to transforms each string in a list of strings to sequence of integers
* Use `sequences_to_matrix` to convert a list of sequences into a Numpy matrix

#### One-Hot Encoding
The `keras.utils` package have processing helpers for categorical embedding. Example you can use `to_categorical` to transform an integer represening a class into a sparse vector with zeros everywhere but the index of the class.
```python
from keras.utils import to_categorical

num_classes = 10

Y_train = to_categorical(y_train_raw, num_classes)
Y_test = to_categorical(y_test_raw, num_classes)
```

## Examples
*Example 1*: dealing with already pre-processed text
```python
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

(X_train, y_train)= reuters.load_data(num_words=NUM_WORDS)

tokenizer = Tokenizer(num_words=NUM_WORDS)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
y_train = to_categorical(y_train, NUM_CLASSES)
```

*Example 2*: vectorizing raw text into a 2D integer tensor
```python
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

texts  = []  # list of text samples
labels = []  # list of label ids

tokenizer = Tokenizer(num_words=NUM_WORDS)

tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y_train = to_categorical(np.asarray(labels))
```
