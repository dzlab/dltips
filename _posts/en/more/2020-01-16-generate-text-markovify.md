---
layout: post

title: Text Generation with Markovify

tip-number: 12
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to train a Markov Chain model to generate sentences given a corpus of text?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - more
---

A very simple way to train a model to generate random sentences given a corpus of text is to use a Markov Chain. In python, we can use `markovify` to build such models.

```
pip install markovify
```

Assuming that the training corpus is a collection of files, we first create a Markov Chain for each file as follows:

```python
import os
import markovify

PATH = '...'

chains = []
for filename in os.listdir(PATH):
  content = open(f'{PATH}{filename}', 'r').readlines()
  markov_chain = markovify.Text(content)
  chains.append(markov_chain)
```

Then we combine the different models into one larger Markov Chain as follows:

```python
model = markovify.combine(chains)
```

Finally, we can start generating random text:

```python
model.make_sentence()
```