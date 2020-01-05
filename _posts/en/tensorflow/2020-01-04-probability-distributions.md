---
layout: post

title: Probability Distributions using TensorFlow

tip-number: 00
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Which probability distributions should I know, How do I know which one to use and when to use them?
tip-writer-support: https://www.patreon.com/dzlab

redirect_from:
  - /en/probability-distributions/

categories:
    - en
    - tensorflow
---

# Probability Distributions

Probability Distributions are mathematical functions that provides the probabilities of the occurrence of various possible outcomes in an experiment. Knowing at least the most common ones is very important, in fact they are used all over in Neural Networkse most importantly in weight initialization and normalization.

How do we know which probability distributions to use and when to use them?


In TensorFlow, the package [tf.random](https://www.tensorflow.org/api_docs/python/tf/random), provides an easy to use API to generate Tensors of different shapes following a chosen distribution. Also, [tfp.distributions](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions) package of TensorFlow Probability .


## Continuous probability distributions
A distribution is called continus when its random variable can take any real value.

### Uniform distribution
The uniform distribution is a type of continuous probability distributions. It describes an experiment where the outcomes have same sampling probability. It takes too parameters, the minimum value and maximum value of the interval from where values will be sampled.

```python
samples = tf.uniform.uniform([simple_size], minval=0.0, maxval=1.0)
```

### Normal distribution

Normal distribution, also known as the Gaussian distribution, is a type of continuous probability distribution for a real-valued random variable. It has two parameters the `mean` and the `standard deviation`.
* This distribution is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean.
* The standard distribution affects how large/wide is the distribution graph.


```python
samples = tf.random.normal([simple_size], mean=0.0, stddev=1.0)
```

## Discrete probability distribution
A distribution is called discrete when its random variable can take values from a accountable set.

### Poisson
[Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution), is a discrete probability distribution. It expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occurs every known constant mean rate and time-independent since the last event. It has one parameter `lam` which is the events mean rate (i.e. mean number of events in a unit of time).

```python
samples = tf.random.poisson([1000], lam=2)
```

### Binomial
[Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution), a descrete distribution that given a binary experiment (yes or no) counts the number of successes (yes). It has one parameter `p` which is the probability of success.

```python
samples = tf.keras.backend.random_binomial([1000], p=0.2)
```

Notebook can be found here - [link](https://github.com/dzlab/deepprojects/blob/master/tensorflow/Probability_Distributions_using_TensorFlow.ipynb)