---
layout: post

title: Custom Training Loop

tip-number: 18
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to customize the training loop in TensorFlow? How to calulcate the gradients for each layer?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---

For most cases you would want to train your TensorFlow model using Keras API, i.e. `model.compile` and  `model.fit` and its variation. This is basic and good enough, as you can specify the loss function, optimization algorithm, provide training/test data, and possibly a callback. Thought there are cases where you may want more control on the training process, for instance:
* new optimization algorithm
* easily modify gradients and how to calculate loss
* speedup training  with all sort of tricks (e.g. teacher forcing)
* better hyperparameters tuning (e.g. use cyclic learning rate).

TensorFlow allow such customization through the `GradientTape` API. Here is a typical example:

```python
@tf.function
def training_loop(epochs, train_dataset, valid_dataset):
  # on every epoch run on entire training and validation
  for epoch in range(epochs):
    # enumerate the training set in batches
    for (batch, (features, labels)) in enumerate(train_dataset):
      train_loss = 0
      with tf.GradientTape() as tape:
        # forward pass in training mode
        logits = model(features, training=True)
        # caculate batch loss function
        loss = loss_func(labels, logits)
      # backprobagation: calculate the gradients and apply them for each layer
      grads = tape.gradient(loss, model.trainable_variables)
      # cumulate training loss
      train_loss += optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # calculate loss on validation set
    valid_loss = 0
    for (batch, (features, labels)) in enumerate(valid_dataset):
      # forward pass in inference mode
      logits = model(features, training=False)
      # cumulate validation loss
      valid_loss += loss_func(labels, logits)
```

What the code above is doing, is for every epoch it enumerates over the entire dataset in bacthes. For every batch, it does:
* A forward pass and record every operation in a tape
* Calculate the loss with respect to the actual labels
* Use recorded operations to perform a backpropagation and calulcate gradients
* Use the optimizer to adjust the layers weights by applying the gradients

Once, the pass on the entire training set finishes, the training loop performs a forward pass on the entire validation set in batches. For every batch, it does a forward pass and make sure the model is in an inference mode and calculate the validation loss of this epoch. It cumulates the losses to determine the validation loss of the current epoch.