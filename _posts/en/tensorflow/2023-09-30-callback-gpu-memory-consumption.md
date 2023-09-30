---
layout: post

title: Track your TF model GPU memory consumption during training

tip-number: 39
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Learn how to track your model GPU memory consumption during training using get_memory_info
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---

TensorFlow provides an experimental [get_memory_info](https://www.tensorflow.org/api_docs/python/tf/config/experimental/get_memory_info) API that returns the current GPU memory consumption.

We can use this API in a custom TF Callback to track GPU memory usage at `peak` during training as follows:

```python
class GPUMemoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, target_batches, print_stats=False, **kwargs):
        super().__init__(**kwargs)
        self.target_batches = target_batches
        self.print_stats = print_stats

        self.memory_usage = []
        self.labels = []

    def _compute_memory_usage(self):
        memory_stats = tf.config.experimental.get_memory_info("GPU:0")
        # Convert bytes to GB and store in list.
        peak_usage = round(memory_stats["peak"] / (2**30), 3)
        self.memory_usage.append(peak_usage)

    def on_epoch_begin(self, epoch, logs=None):
        self._compute_memory_usage()
        self.labels.append(f"epoch {epoch} start")

    def on_train_batch_begin(self, batch, logs=None):
        if batch in self.target_batches:
            self._compute_memory_usage()
            self.labels.append(f"batch {batch}")

    def on_epoch_end(self, epoch, logs=None):
        self._compute_memory_usage()
        self.labels.append(f"epoch {epoch} end")
```

> Note: For simplicity we are assing there is a single GPU, `GPU:0`.

Here is an example show how to create an instance of such a callback to track consumption at various batches:

```python
gpu_memory_callback = GPUMemoryCallback(
    target_batches=[5, 10, 25, 50, 100, 150, 200, 300, 400, 500],
    print_stats=True,
)
```

Once the callback instance is created we can simply pass it to `model.fit` so it gets called during training to track GPU consumption

```python
model.compile(optimizer=optimizer, loss=loss, weighted_metrics=["accuracy"])

model.fit(train_ds, epochs=EPOCHS, callbacks=[gpu_memory_callback])
```

Once the model training finishes, we can access the consumption history as follows

```python
memory_usage = gpu_memory_callback.memory_usage
```

Then we can simply plot it with matplotlib

```python
plt.bar(memory_usage)
```

It is important to reset the `peak` memory usage to `current` memory usage before starting the training to make sure un-used memory is released and will not be accounted for in our callback.

```python
tf.config.experimental.reset_memory_stats("GPU:0")
```

One good use case for tracking GPU consumption is to be able to compare two (or more) models training based on their GPU memory consumption. For instance, comparing a distilled version of a bigger model.

The workflow could be like this

```python
gpu_memory_callback_1 = GPUMemoryCallback(...)
model_1.fit(train_ds, epochs=EPOCHS, callbacks=[gpu_memory_callback_1])

tf.config.experimental.reset_memory_stats("GPU:0")

gpu_memory_callback_2 = GPUMemoryCallback(...)
model_2.fit(train_ds, epochs=EPOCHS, callbacks=[gpu_memory_callback_2])

model_memory_usage_1 = gpu_memory_callback_1.memory_usage
model_memory_usage_2 = gpu_memory_callback_2.memory_usage
```

Then after training is done, we plot both consumptions to visually compare them:

```python
plt.bar(
    ["Model 1", "Model 2"],
    [max(model_memory_usage_1), max(model_memory_usage_2)],
    color=["red", "blue"],
)

plt.xlabel("Time")
plt.ylabel("GPU Memory Usage (in GB)")

plt.title("GPU Memory Usage Comparison")
plt.legend()
plt.show()
```
