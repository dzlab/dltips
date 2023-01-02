---
layout: post

title: Flax basics

tip-number: 37
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Learn the basics for building deep learning models with Flax
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - more
---

Installation the `jax`/`flax` and `optax` modules
```shell
@ pip install jax
$ pip install flax
$ pip install optax
```

Import the modules
```python
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state
import optax
import tensorflow_datasets as tfds
```

## Data pipeline
Flax does not provide an API for loading data, but we can build a data pipeline with TensorFlow data API and JAX. For instance, the following snippets load MNIST data into JAX numpy arrays:
```python
ds = tfds.load('mnist')
ds_builder = tfds.builder('mnist')
ds_builder.download_and_prepare()
train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
```
After that we can use JAX numpy to perform any required data processing: resizing, normalizing, cropping, etc.

## Model definition
Models in Flax can be defined using the `Setup` function where we will need to initialize all the layers

```python
class MyModel(nn.Module):
    def setup(self):
        self.lin = nn.Dense(10)

    def __call__(self, x):
        x = self.dense1(x)
        return x
```

Or we can define models in Flax using the `@nn.compact` annotation

```python
class MyModel(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(10)(x)
        return x
```

## Loss and metrics
Loss functions are available through the `optax` module, but can also be calculated manually using JAX numpy API
```python
def cross_entropy_loss(*, logits, y_true):
    y_true_onehot = jax.nn.one_hot(y_true, num_classes=2)
    return optax.softmax_cross_entropy(logits=logits, labels=y_true_onehot).mean()
```

Metrics need to be defined manually, for instance the accuracy could be calculated like this using JAX numpy API
```python
def compute_metrics(*, logits, y_true):
    accuracy = jnp.mean(jnp.argmax(logits, -1) == y_true)
    metrics = {
        'accuracy': accuracy,
    }
    return metrics
```


## Training with Flax
Training models in Flax, requires first the creation of a **TrainState** to hold any information that will be passed to the model during its training.  

```python
def create_train_state(rng):
    model = MyModel()
    params = model.init(rng, param1, param2, ...)['params']
    opt = optax.adam(0.01,0.99,0.999,2e-05)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)
```

Then, we define a training step where we do the forward pass, compute losses and corresponding gradients. After that, we use the gradients to update the model parameters.


```python
@jax.jit
def train_step(training_state, xb, y_true):
  def loss_fn(params):
    logits = Model().apply({'params': params}, xb)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(y_true, num_classes=10)))
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, y_true)
  return state, metrics
```

Otherwise we could use the [Elegy](https://github.com/poets-ai/elegy) which is a high-level API similar to Keras.