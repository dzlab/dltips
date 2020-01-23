---
layout: post

title: Use to expose and parse CLI arguments

tip-number: 16
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to parse command line arguments passed by user to your TensorFlow application
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---

You may have noticed `tf.app.flags` been used in every official TensorFlow v1 tutorial like CIFAR-10 example - [link](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10). Yet this module is not documented on the official TensorFlow website, the only documentation is about the [tf.app.run](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/app/run).

An example of using this module would be to have a `main.py` file that contains something like following:
```python
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epochs", 10, "number of training epochs")
flags.DEFINE_string("data_dir", "./data/", "path of data for training/validation/testing")
flags.DEFINE_string("log_dir", "./log/", "path of log output")
flags.DEFINE_boolean("is_train", True, "wheter to train with input data or do inference")
flags.DEFINE_float("lr", 0.9, "learning rate")

FLAGS = flags.FLAGS
if __name__ == "__main__":
  # Use the flags to control program execution
  if FLAGS.is_train:
    # Do training
```
After saving the file, you can check how the help output would look like with `python main.py -h`.

As of TensorFlow v2 this module does not exist and you may be looking for an alternative. Luckily Python has some popular modules for CLI argument parsing, e.g. [argparse](https://docs.python.org/3/library/argparse.html) or [Abseil](https://abseil.io/docs/python/guides/flags) which has similar use just like `tf.app.flags` which makes migration painless.

This is how `main.py` would look like if we use `Abseil`
```python
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 10, "number of training epochs")
flags.DEFINE_string("data_dir", "./data/", "path of data for training/validation/testing")
flags.DEFINE_string("log_dir", "./log/", "path of log output")
flags.DEFINE_boolean("is_train", True, "wheter to train with input data or do inference")
flags.DEFINE_float("lr", 0.9, "learning rate")

def main(argv):
  # Use the flags to control program execution
  if FLAGS.is_train:
    # Do training

if __name__ == ‘__main__’:
  app.run(main)
```

Alternatively, you can use [python-fire](https://github.com/google/python-fire) which is a Google library that can be used to turn function arguments into CLI arguments and do the parsing for you. The same `main.py` would look like this
```python
import fire

def main(epochs, data_dir, log_dir, is_train, lr):
  # do something

if __name__ == '__main__':
  fire.Fire(main)
```