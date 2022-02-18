---
layout: post

title: Resource Management with Quantization

mathjax: true

tip-number: 28
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Learn how to perform model quantization as an effective resource management technique.
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - keras
---

Managing resouces used by Machine Learning models is very curtial especially during inference. You don't want an experience where the user need to wait few minutes as he/she tries to type his message because your model needs more time to estimate best suggestion for next word. If your model will run on the cloud you want to optimize latency and reduce inference cost among other things. If it will run on edge devices, you need to worry in addition to that to the processing/memory restrictions, power-consumption, network usage, and storage space.

Likely, TensorFlow provides a toolkit called [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization) that can be used during training and also post-training tools in case the model is already trained to optimize it.

Install the toolkit with pip
```
$ pip install tensorflow_model_optimization
```

Import it as `tfmot`
```python
import tensorflow_model_optimization as tfmot
```

Now we can use many of the quantization functionalities for Keras which are available under `tfmot.quantization.keras`

We can quantize parts of a model by wrapping layers we want them to be quantized using `quantize_annotate_layer` from `tfmot.quantization.keras`

```python
from tfmot.quantization.keras import quantize_annotate_layer
from tfmot.quantization.keras import quantize_apply
# Create quantization wrapper
model = tf.keras.Sequential([
  ...
  # Only annotated layers will be quantized.
  quantize_annotate_layer(Conv2D()),
  quantize_annotate_layer(ReLU()),
  Dense(),
  ...
])
# Quantize the model
quantized_model = quantize_apply(model)
```

We can also quantize custom Keras layers using `quantize_scope` from `tfmot.quantization.keras`

```python
from tfmot.quantization.keras import quantize_annotate_layer
from tfmot.quantization.keras import quantize_annotate_model
from tfmot.quantization.keras import quantize_scope
from tfmot.quantization.keras import quantize_apply

model = quantize_annotate_model(tf.keras.Sequential([
  quantize_annotate_layer(CustomLayer(20, input_shape=(20,)), DefaultDenseQuantizeConfig()),
  tf.keras.layers.Flatten()
]))

# quantize_apply requires mentioning DefaultDenseQuantizeConfig with quantize_scope
with quantize_scope({'DefaultDenseQuantizeConfig': DefaultDenseQuantizeConfig, 'CustomLayer': CustomLayer}):
  # Use quantize_apply to actually make the model quantization aware.
  quant_aware_model = quantize_apply(model)
```


Learn more about quantization in tf-keras with the following resources
- Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference - [arxiv.org](https://arxiv.org/abs/1712.05877)
- Post-training quantization - [link](https://medium.com/tensorflow/introducing-the-model-optimization-toolkit-for-tensorflow-254aca1ba0a3)
- Quantization aware training - [link](https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html)