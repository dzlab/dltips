---
layout: post

title: Learning Rate Scheduling with Callbacks

mathjax: true

tip-number: 23
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to dynamically change the learning rate for gradient descent optimizers?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - keras
---

One of the usefull tweaks for faster training of neural networks is to vary (in often cases reduce) the learning rate hyperprameter which is used by Gradient-based optimization algorithms. 

Keras provide a callack function that can be used to control this hyperprameter over time (numer of iterations/epochs). To use this callback, we need to:
* Define a function that takes an epoch index as input and returns the new learning rate as output.
* Create an instance of `LearningRateScheduler` and pass the previously defined function as a parameter.

```python
def sechdule(epoch):
  ...

lr_callback = tf.keras.callbacks.LearningRateScheduler(sechdule, verbose=True)
```

## Scheduling functions
There is endless ways to schedule/control the learning rate, this section presents some examples.

### Constant Learning Rate
The following scheduling function keeps learning rate at constant value regardless of time.
```python
# Define configuration parameters
start_lr = 0.001

# Define the scheduling function
def schedule(epoch):
  return start_lr
```

### Time-based Decay
The following scheduling function gradually decreases the learning rate over time from a starting value. The mathematical formula is $$lr = \frac{lr_0}{(1+k*t)}$$ where $$lr_0$$ is the initial learning rate value, $$k$$ is a decay hyperparameter and $$t$$ is the epoch/iteration number.

```python
# Define configuration parameters
start_lr = 0.001
decay = 0.1

# Define the scheduling function
def schedule(epoch):
  previous_lr = 1
  def lr(epoch, start_lr, decay):
    nonlocal previous_lr
    previous_lr *= (start_lr / (1. + decay * epoch))
    return previous_lr
  return lr(epoch, start_lr, decay)
```

### Exponential Decay
The following scheduling function exponentially decreases the learning rate over time from starting point. Mathematically it can be reporesented as $$lr = lr_0 * \exp^{-k*t}$$ where $$lr_0$$ is the initial learning rate value, $$k$$ is a decay hyperparameter and $$t$$ is the epoch/iteration number.

```python
# Define configuration parameters
start_lr = 0.001
exp_decay = 0.1

# Define the scheduling function
def schedule(epoch):
  def lr(epoch, start_lr, exp_decay):
    return start_lr * math.exp(-exp_decay*epoch)
  return lr(epoch, start_lr, exp_decay)
```

### Constant then exponential decayed
The following scheduling function keeps the learning rate at starting value for the first ten epochs and after that will decrease it exponentially.

```python
# Define configuration parameters
start_lr = 0.001
rampup_epochs = 10
exp_decay = 0.1

# Define the scheduling function
def schedule(epoch):
  def lr(epoch, start_lr, rampup_epochs, exp_decay):
    if epoch < rampup_epochs:
      return start_lr
    else:
      return start_lr * math.exp(-exp_decay * epoch)
  return lr(epoch, start_lr, rampup_epochs, exp_decay)
```

### One Cycle Learning Rate
The following scheduling function gradually increases the learning rate from a starting point up to a max value during a period of epochs. After that it will decrease the learning rate exponentially and stabilise it to a minimum value. This scheduling algorithm is also known as One Cycle Learning Rate [source](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/fast-and-lean-data-science/07_Keras_Flowers_TPU_xception_fine_tuned_best.ipynb)
```python
# Define configuration parameters
start_lr = 0.0001
min_lr = 0.00001
max_lr = 0.001
rampup_epochs = 10
sustain_epochs = 0
exp_decay = 0.8

# Define the scheduling function
def schedule(epoch):
  def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
    if epoch < rampup_epochs:
      lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
      lr = max_lr
    else:
      lr = (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr
    return lr
  return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)
```

## Visualization
The following chart visualizes the learnining rate as it is scheduled by each of the previousy defined functions.

<div id="graph"></div>
<script>
var trace1 = {
  x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
  y: [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
  type: 'scatter',
  name: 'Constant Learning Rate'
};
var trace2 = {
  x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
  y: [0.001, 0.0009090909090909091, 0.0008333333333333334, 0.0007692307692307692, 0.0007142857142857144, 0.0006666666666666666, 0.000625, 0.000588235294117647, 0.0005555555555555556, 0.0005263157894736842, 0.0005, 0.0004761904761904762, 0.00045454545454545455, 0.0004347826086956522, 0.00041666666666666664, 0.0004, 0.0003846153846153846, 0.00037037037037037035, 0.0003571428571428572, 0.0003448275862068965],
  type: 'scatter',
  name: 'Time-based Decay'
};
var trace3 = {
  x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
  y: [0.001, 0.0009048374180359595, 0.0008187307530779819, 0.0007408182206817179, 0.0006703200460356394, 0.0006065306597126335, 0.0005488116360940264, 0.0004965853037914095, 0.0004493289641172216, 0.00040656965974059914, 0.00036787944117144236, 0.00033287108369807955, 0.00030119421191220205, 0.0002725317930340126, 0.00024659696394160646, 0.00022313016014842982, 0.00020189651799465538, 0.0001826835240527346, 0.00016529888822158653, 0.00014956861922263504],
  type: 'scatter',
  name: 'Exponential Decay'
};
var trace4 = {
  x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
  y: [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.00036787944117144236, 0.00033287108369807955, 0.00030119421191220205, 0.0002725317930340126, 0.00024659696394160646, 0.00022313016014842982, 0.00020189651799465538, 0.0001826835240527346, 0.00016529888822158653, 0.00014956861922263504],
  type: 'scatter',
  name: 'Constant then exponential decayed'
};
var trace5 = {
  x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
  y: [0.0001, 0.00019, 0.00028, 0.00036999999999999994, 0.00045999999999999996, 0.00055, 0.0006399999999999999, 0.00073, 0.00082, 0.00091, 0.001, 0.0008020000000000001, 0.0006436000000000001, 0.0005168800000000002, 0.0004155040000000001, 0.0003344032000000001, 0.0002695225600000001, 0.00021761804800000007, 0.00017609443840000007, 0.00014287555072000006],
  type: 'scatter',
  name: 'One Cycle Learning Rate'
};

var data = [trace1, trace2, trace3, trace4, trace5];
var layout = {
  title:'Learning Rate over time (epoch)'
}
Plotly.newPlot('graph', data, layout);
</script>

Learn more about LearningRateScheduler usage in tf-keras - [link](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler)