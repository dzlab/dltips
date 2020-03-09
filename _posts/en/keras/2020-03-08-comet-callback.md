---
layout: post

title: Track, compare, and optimize your models with Comet.ml

tip-number: 21
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to keep track of what hyper-parameters were used in what experiment?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - keras
---


[Comet.ml](https://comet.ml/) is a hosted service that let you keep track of your machine learning experiments, visualize and compare results of each experiment. Furthermore, Comet.ml provides an AutoML feature that let's you find the best hyper-prameters to train your model. And it's super easy to use, you simply initialize an experiment, Comet.ml will patch your code and add logging functionality. For instance it will automatically add callbacks to your Keras `model.fit` to log the training metrics and update update the experiment dashboard.

<h3 style="text-align:center;">
  <iframe class="post-content" width="1000" height="500" src="https://www.comet.ml/dzlab/tf-text-skipgram/" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</h3>

Install Comet.ml
```
pip install comet_ml -q
```

Setup Comet.ml:
* Login to the service https://comet.ml/ and and get an API key
* Initialize a new experiment with project names and additional flags

```python
from comet_ml import Experiment

COMET_API_KEY = 'xyz'
experiment = Experiment(
  api_key=COMET_API_KEY,
  project_name="<project-name>", workspace="<username>",
  auto_param_logging=True, auto_metric_logging=True
)
```

Initialize Comet.ml config to saves hyperparameters and inputs of the expriment
```python
config = {
  'batch_size': 1024,
  'train_split': 0.8
}

experiment.log_parameters(config)
```

After creating the model, you can upload model plot
```python
model = ...
plot_model(model, to_file='model.png', show_shapes=True)
experiment.log_image('model.png')
```

After training finishes, you can upload the model and its weights
```python
model.fit(train_ds, validation_data=valid_ds, epochs=10)
model.save('<model-name>.h5')
experiment.log_model('<model-name>', '<model-name>.h5', overwrite=True)
```

Learn more about what you can do with Comet - [link](https://medium.com/comet-ml/comet-ml-cheat-sheet-supercharge-your-machine-learning-experiment-management-7786e08f8e9e).