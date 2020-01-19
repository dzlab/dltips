---
layout: post

title: Weights & Biases callback for Keras

tip-number: 12
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to keep track of what hyper-parameters were used in what experiment?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - keras
---

[Weights & Biases](https://www.wandb.ai/) is a hosted service that let you keep track your machine learning experiments, visualize and compare results of each experiment. Basically, you log the hyper-parameters used in the experiment, the metrics from the training as well as the weights of the model itself. This tool let you also share the experiment results.


Install the W&B
```
pip install wandb -q
```

Setup W&B and login to the service
```python
import wandb

wandb.login()
```

Initialize a new W&B run with your user and project names:
```python
wandb.init(entity = "<username>", project = "<project-name>")
```

Initialize W&B config to saves hyperparameters and inputs of the expriment
```python
config = wandb.config
config.batch_size = 1024
config.train_split = 0.8
```

Create Keras callback to log training information
```python
wandb_callback = wandb.keras.WandbCallback(log_weights=True)
```

Add the callback to Keras `fit()` call
```python
model.fit(train_ds,
  validation_data=valid_ds,
  epochs=10,
  callbacks=[wandb_callback]
)
```

After training finishes, you can save the model to W&B
```python
model.save(os.path.join(wandb.run.dir, "<model-name>.h5"))
```

Learn more about what you can do with W&B - [link](https://docs.wandb.com/integrations/jupyter.html).