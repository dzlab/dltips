---
layout: post

title: Create BERT vocabulary with Tokenizers

tip-number: 20
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to create a vocabulary file to use with TensorFlow Text BertTokenizer?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---

[Tokenizers](https://github.com/huggingface/tokenizers) is an easy to use and very fast python library for training new vocabularies and text tokenization. It can be installed simply as follows:

> pip install tokenizers -q

To generate the vocabulary of a text, we need to create an instance `BertWordPieceTokenizer` then train it on the input text file as follows. Once training done, it can take some time depending on the corpus size, we save the vocabulary to a file for later use. Here are the steps:

```python
from tokenizers import BertWordPieceTokenizer

# Initialize an empty BERT tokenizer
tokenizer = BertWordPieceTokenizer(
  clean_text=False,
  handle_chinese_chars=False,
  strip_accents=False,
  lowercase=True,
)

# prepare text files to train vocab on them
files = ['input.txt']

# train BERT tokenizer
tokenizer.train(
  files,
  vocab_size=100,
  min_frequency=2,
  show_progress=True,
  special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
  limit_alphabet=1000,
  wordpieces_prefix="##"
)

# save the vocab
tokenizer.save('.', 'bert')
```

The `vocab_size` (default is 30000) and `limit_alphabet` (default is 1000) are very important parameters that control the quality and richness of the generated vocabulary. You need to try different values for both parameters and play with the generated vocab.

Once we have the vocabulary file in hand, we can use to check the look of the encoding with some text as follows:
```python
# create a BERT tokenizer with trained vocab
vocab = 'bert-vocab.txt'
tokenizer = BertWordPieceTokenizer(vocab)

# test the tokenizer with some text
encoded = tokenizer.encode('...')
print(encoded.tokens)
```

After confirming that the trained vocabulary is good for the task in hand, we can use in a TensforFlow model via TensorFlow Text BertTokenizer as explained [in this tip]({{page.lang | append: "/tensorflow/bert-tokenizer/" | relative_url }}).

