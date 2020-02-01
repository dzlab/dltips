---
layout: post

title: Keras Tuner for Hyperparameters tuning

tip-number: 19
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to find best hyperparameters to use during model training? Is there a better way then manual trial and error?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - keras
---

Finding the right hyperparameters (e.g. regularization parameter, learning rate, dropout rate) of a machine learning model is tricky as the space of values can be to large. [Keras Tuner](https://github.com/keras-team/keras-tuner) is a hyperparameter optimization framework that helps in hyperparameter search. It lets you define a search space and choose a search algorithm to find the best hyperparameter values.

```python
import kerastuner as kt
```

Keras Tuner includes different search algorithms: Bayesian Optimization, [Hyperband](https://arxiv.org/pdf/1603.06560.pdf), and Random Search. Furthermmore, Keras Tuner is extendable and lets you define your own search algorithm.

First, we need to define a model builder function that takes one argument `hp` that will be provided by the optimization algorithm and used to define the different choices. Here is a basic example of how to `build_model` would look like:

```python
def build_model(hp):
  model_type = hp.Choice('model_type', ['choice 1', 'choice 2'])
  if model_type == 'choice 1':
    with hp.conditional_scope('model_type', 'choice 1'):
      pass
  if model_type == 'choice 2':
    with hp.conditional_scope('model_type', 'choice 2'):
      pass
```

Alternatively, Keras Tuner includes built-in models that you can use as base: `HyperResnet` and `HyperXception`. You can get a tunable version of each as follows:
```python
build_resnet = kt.applications.HyperResNet(
  input_shape=(256, 256, 3), classes=10
)
build_xception = kt.applications.HyperXception(
  input_shape=(256, 256, 3), classes=10
)
```

Once we have a model builder, we can create a tuner based on search algorithm. Example, Bayesian Optimization will need an objective function (or string) and maximum number of trials:
```python
tuner = kt.tuners.BayesianOptimization(
  build_model,
  objective=kt.Objective('accuracy', 'val_accuracy'),
  max_trials=50
)
```
Similarly, Hyperband takes the model builder, an objective and additional a specific parammeter `hyperband_iterations`:
```python
tuner = kt.Hyperband(
  build_model,
  objective='val_accuracy',
  max_epochs=30,
  hyperband_iterations=2
)
```

Finally, we start the search where Keras Tuner will try different permutations of the underlying architecture to find the best one for you specific task.
```python
tuner.search(
  train_ds,
  validation_data=test_ds,
  epochs=10
)
```

More details on Keras Tuner website - [link](https://keras-team.github.io/keras-tuner/).