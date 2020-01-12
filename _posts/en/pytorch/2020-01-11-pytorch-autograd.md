---
layout: post

title: Gradient computation with AutoGrad

tip-number: 08
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How do we compute the of a function in PyTorch?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - pytorch
---

AutoGrad in PyTorch allows the calculation of gradients (that will be used to updated the original variables) automatically.

Given two tensors X and Y, we can compute the gradient by simply computing a forward pass and a loss function. PyTorch will keep grad of all tensors manipulation and automatically calcultes the grad. Here is an example

```python
import torch

X = torch.rand(size=(10, 1))
y = torch.rand(size=(10, 1))
```

Define the tensors that need to have gradients (i.e. the weights) by setting the `requires_grad` flag to true.

```python
a = torch.ones(10, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

Define the forward and loss functions
```python
def forward(X):
  return X * a + b

def loss(X, y):
    y_pred = forward(X)
    return torch.sum(torch.abs((y_pred - y)))
```

Calculate the losses given the weights `a` and `b` then perform a backward pass
```python
loss = loss_func(X, y)

loss.backward()
```

At this moment, PyTorch have already claculated the grandients
```python
a.data, a.grad.data
```

Note, in a traning loop where the goal will be to optimize the weights (here `a` and `b`), after the backward pass and after having used the gradients to update the weights they need to be reset to 0.
```python
a.grad.zero_()
b.grad.zero_()
```