---
layout: post

title: Introduction to Keras

tip-number: 05
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Keras is a popular Deep Learning framework with a user friendly API.
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - keras
---

Keras is an awesome deep learning framework that is focused on providing a user friendly API on top of DL backends like Theano and TensorFlow. Keras high-level API allows you:
* Use existing datasets, or build your own,
* Define a complex data processing pipeline for images and text data.
* Build deep learning models using Sequential or a Functional API
* Provides a variety of layers for different use case: Convolution, Recurrent or Linear.
* Makes it easy to train and evaluate models as well as exporting them to diffent storage formats.

A simple Perceptron Neural Network can be build with Keras like this:

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer

# prepare a dataset
features = np.random.random((1000, 10))
labels = np.random.randint(2, size=(1000, 1))

# create a model
model = Sequential([
  InputLayer(input_shape=(None, 10)),
  Dense(20, activation='relu'),
  Dense(1, activation='sigmoid')
])

# train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(features, labels, epochs=2, batch_size=10)

# run predictions
predictions = model.predict(features)
```

## Data
Data for your neural network can be in different form, stored as python objects with `pickle` or in raw files (e.g. images, text). Keras accept data as numpy arrays, so sometimes you may need some preprocessing which Keras provides a lot helper to facilitate this step.

### Datasets
For your NN, either use Data that's alreay ready to be used (e.g. Keras Datasets) or acquire data and use Keras preprocessing to prepare it.

#### Keras Datasets
Example of popular Keras Datasets include: Boston housing, MNIST, and IMDb.
```python
from keras.datasets import boston_housing, mnist, imdb, cifar10

(X_train,y_train), (X_test,y_test) = mnist.load_data()
(X_train,y_train), (X_test,y_test) = boston_housing.load_data()
(X_train,y_train), (X_test,y_test) = cifar10.load_data()
(X_train,y_train), (X_test,y_test) = imdb.load_data(num_words=10000)
```

#### Public Datasets
You can also use public datasets and prepare them before passing them into a NN

```python
from urllib.request import urlopen

DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/ pima-indians-diabetes.data"

data = np.loadtxt(urlopen(DATA_URL), delimiter=",")
X, y = data[:, 0:8], data [:, 8]
```

## Architectures

### Multilayer Perceptron (MLP)

#### Regression
A regresion model output a real value and should have:
* one output in the final layer,
* use `mse` Mean Square Error loss function.
* use `mae` Mean Absolute Error metric.

```python
from keras.models import Sequential
from keras.layers import Dense, InputLayer

model = Sequential()
model.add(InputLayer(input_shape=(None, 64)))
model.add(Dense(64, activation='relu')
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

#### Binary Classification
A binary classification model should have:
* one output in the final layer,
* use `sigmoid` activation function,
* use `binary_crossentropy` loss function.
* use `accuracy` metric.

```python
from keras.models import Sequential
from keras.layers import Dense, InputLayer

model = Sequential()
model.add(InputLayer(input_shape=(None, 64)))
model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

#### Multi-Class Classification
A multi-class classification model should have:
* more than one output in the final layer,
* use `softmax` activation function,
* use `categorical_crossentropy` or `sparse_categorical_crossentropy` loss functions.
* use `accuracy` metric.

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer

model = Sequential()
model.add(InputLayer(input_shape=(None, 10)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Convolutional Neural Network (CNN)
A example CNN for multi-class classification would combine:
* Convolution layers like Conv2D with padding
* Pooling layers like MaxPooling2D
* Dropout layers to fight overfitting

```python
from keras.layers import Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(InputLayer(input_shape=(None, 28, 28, 3)))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3), padding='same', activation='relu'))
model.add(Conv2D(64 ,(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
```

### Recurrent Neural Network (RNN)
A example RNN for binary-class classification would combine:
* An Embedding layer like that would embed every token in a vocabulary (e.g. of size 10k) into a high dimensional space (e.g. of size 100)
* An recurent network like LSTM that takes a padded sequence as input
* Dropout layers to fight overfitting

```python
from keras.klayers import Embedding,LSTM

model.add(Embedding(10000, 100))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, metrics=['mae'])
```

## Lifecycle
Keras provide an extensive API for every step in the lifecyle of a model

### Training
You can train and monitor your metric every epoch on the training and validation sets
```python
model.fit(X_train, y_train,
  batch_size=32, epochs=10, verbose=1,
  validation_data=(X_test,y_test)
)
```

### Prediction
You can run prediction on the test set
```python
model.predict(X_test, batch_size=16)
model.predict_classes(X_test, batch_size=16)
```

### Inspection
You can inspect the properties of the model (weights, shapes, layers, etc) with
```python
# Model output shape
model.output_shape

# Model summary representation
model.summary()

# Model configuration
model.get_config()

# Model weight tensors
model.get_weights()
```

You can plot the layers of the model as `.png` file
```python
from keras.utils import plot_model

plot_model(model, to_file='char-rnn.png', show_shapes=True)
```

### Save / Load
You can save and reload models
```python
from keras.models import load_model

model.save('model_file.h5')
model = load_model('model_file.h5')
```

### Callbacks
A callback is a function that will be invoked at given stages of the training procedure. They usually help to get an idea oof internal states and statistics of the model during training.

* `EarlyStopping` Stop training when a monitored quantity has stopped improving
* `LearningRateScheduler` a scheduler to control Learning rate and change it over time
* `TensorBoard` will report metrics to TensorBoard for basic visualizations

```python
from keras.callbacks import EarlyStopping

early_stopping_callback = EarlyStopping(patience=2)
model.fit(X_train, y_train,
  batch_size=16, epochs=10,
  validation_data=(X_test,y_test),
  callbacks=[early_stopping_monitor]
)
```