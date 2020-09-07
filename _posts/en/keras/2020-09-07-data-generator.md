---
layout: post

title: Custom Data Generator with keras.utils.Sequence

mathjax: true

tip-number: 27
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to create custom data loaders for Keras?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - keras
---

Data Generators are useful in many cases, need for advanced control on samples generation or simply the data does not fit in memory and have to be loaded dynamically.

Keras' `keras.utils.Sequence` is the root class for Data Generators and has few methods to be overrided to implement a custom data laoder. A basic structure of a custom implementation of a Data Generator would look like this:

```python
class CustomDataset(tf.keras.utils.Sequence):
  def __init__(self, batch_size, *args, **kwargs):
    self.batch_size = batch_size
    ...

  def __len__(self):
    # returns the number of batches
    ...
    return total_data / self.batch_size

  def __getitem__(self, index):
    # returns one batch
    ...
    return X, y

  def on_epoch_end(self):
    # option method to run some logic at the end of each epoch: e.g. reshuffling
    ...
```

Using the custom generator is fairly easy by passing it to `model.fit` or `model.fit_generator`:

```python
train_ds = CustomDataset(...)
valid_ds = CustomDataset(...)

# Train model on dataset
model.fit_generator(train_ds, validation_data=valid_ds)
```

It is also worth noting that Keras also provide builtin data generator that can be used for common cases. For instance with `ImageDataGenerator` one can easily load images from a directory and apply some basic transformations:
```python
datagen = ImageDataGenerator(
  rescale=1./255,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)
```

Learn more about Data Generators usage in tf-keras - [link](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence)