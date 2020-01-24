---
layout: post

title: TimeDistributed

tip-number: 17
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to keep track of what hyper-parameters were used in what experiment?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - keras
---

[TimeDistributed](https://keras.io/layers/wrappers/#timedistributed) is a wrapper Layer that will apply a layer the temporal dimension of an input. To effectively learn how to use this layer (e.g. in Sequence to Sequence models) it is important to understand the expected input and output shapes.

* **input** a tensor of at least 3D, e.g. `(samples, timestamps, in_features, ...)`. In case, the input comes from an `LSTM` layer make sure to return sequences (e.g. `return_sequences=True`).
* **output** a tensor of shape 3D, e.g. `(samples, timestamps, out_features)`. The output `out_features` corresponds to the output of the wrapped layer (e.g. Dense) which will be applied (with same weights) to the LSTMs outputs one time step in `timestamps` at a time.

Example, if TimeDistributed receives data of shape (None, 100, 32, 256) then the wrapped layer (e.g. Dense) will be called for every slice of shape (None, 32, 256). Here none corresponds to samples/batch size.

Consider the following simple model:
```python
model = Sequential([
  LSTM(units,
    input_shape=(timestamps, 1),
    return_sequences=True
  ),
  TimeDistributed(Dense(1))
])
model.compile(optimizer='adam', loss='mse')
```
In this case, the input and output for each layer will be:

| Layer        | Input | Output           |
| ------------- | ------------- | ------------- |
| LSTM | `(samples, timestamps, 1)` | `(samples, timestamps, units)` |
| TimeDistributed | `(samples, timestamps, units)` | `(samples, timestamps, 1)` |

The training data need to be shaped as follows:
* X will be `(samples, timestamps, 1)`
* y will be `(samples, timestamps, 1)`