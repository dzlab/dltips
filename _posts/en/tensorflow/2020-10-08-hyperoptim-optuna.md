---
layout: post

title: Neural Architecture Search in Tensorflow with Optuna

tip-number: 32
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to use an objective optimization library like Optuna to come up with an optimal Neural Network architecture?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---

The Network Architecture Space is a “collection” of all possible neural network architectures. It is impossible, from a computation perspective, to try all architectures in this space to come up with the best one. Therefore, a clever search startegy should be used to eliminate non promising architectures from the space and converge to the good one. There are search approaches which are implemented by different libraries:
* Random search
* Bayesian optimization
* Evolutionary methods
* Reinforcement learning(RL)
* Gradient-based methods.
* Hierachical-based search

In this TIP, we pick [Optuna](https://optuna.org/) as the search tool. This library key features are:
* Automated search for optimal hyperparameters using Python constructs
* Efficient search on large spaces and pruning of unpromising trials
* Parallelized search over multiple threads or processes


To use Optuna to optimize a TensorFlow model's hyperparameters, (e.g. number of layers number of hidden nodes, etc.), Follow these steps:

1. Create an `objective` function that accepts an Optuna `trial` object:
  * Use the `trial` object to suggest values for your hyperparameters
  * Create a model, optimizer using the suggested hyperparameters
  * Train the model and calculate a metric (e.g. accuracy)
  *  Return the metric value (this will be the objective)
2. Create an Optuna `study` object and execute the optimization


## Installation
First, Install `optuna` as follows
```
$ pip install optuna
```

Import the `Optuna` package
```python
import optuna
```

## Helper functions

Second, create helper functions
```python
# Helper function to get data
def get_data():
  ...
  return train_ds, valid_ds
```

Helper function to create model and optimize the number of layers, numbers units.
```python
def create_model(trial):
  num_layers = trial.suggest_int("num_layers", 1, 5)
  model = tf.keras.Sequential()
  for i in range(n_layers):
    num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
    model.add(tf.keras.layers.Dense(num_hidden, activation="relu"))
    ...
  return model
```

Helper function to create optimizer and optimize the choice of optimizers as well as their parameters.
```python
def create_optimizer(trial):
  kwargs = {}
  optimizer_options = ["Adam", "SGD", ...]
  optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
  if optimizer_selected == "Adam":
    kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
  elif optimizer_selected == "SGD":
    kwargs["learning_rate"] = ...
    kwargs["momentum"] = ...
  elif optimizer_selected == "XYZ":
    ...

  optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
  return optimizer
```
Helper function to run training
```python
def learn(model, optimizer, dataset, mode="eval"):
  accuracy = tf.metrics.Accuracy("accuracy", dtype=tf.float32)

  for batch, (features, labels) in enumerate(dataset):
    with tf.GradientTape() as tape:
      logits = model(features, training=(mode == "train"))
      loss_value = ...
      ...

  if mode == "eval":
    return accuracy
```
Notice how `trial.suggest_int` is used to ask Optuna for the hyper-parameter's value.

Third, create the objective function that uses the previous helper functions
```python
def objective(trial):
  # Get train/valid data.
  train_ds, valid_ds = get_data()

  # Build model and optimizer.
  model = create_model(trial)
  optimizer = create_optimizer(trial)

  # Training and validating cycle.
  for _ in range(epochs):
    learn(model, optimizer, train_ds, "train")

  accuracy = learn(model, optimizer, valid_ds, "eval")

  # Return last validation accuracy.
  return accuracy.result()
```

## Run optimization
Finally, create an Optuna study and run the optimization.
```python
def search():
  study = optuna.create_study(direction="maximize")
  study.optimize(objective, n_trials=100)

  print("Number of finished trials: ", len(study.trials))

  print("Best trial:")
  trial = study.best_trial

  print("  Value: ", trial.value)

  print("  Params: ")
  for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
```

More examples on how to use Optuna can be found [here](https://github.com/optuna/optuna/blob/master/examples).