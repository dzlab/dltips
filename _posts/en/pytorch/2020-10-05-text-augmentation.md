---
layout: post

title: Text data augmentation with Back Translation

tip-number: 31
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to augment a small corpus of text data for a task like text classification?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - pytorch
---

Data augmentation is an effective technique to reduce overfitting that consists of creating an additional slightly modified version of the available data. In NLP, **Back Translation** is one of such augmentation technique that works as follows:
- given an input text in some source language (e.g. English)
- translate this text to a temporary destination language (e.g. English -> French)
- translate back the previously translated text into the source language (e.g. French -> English)

The rest of this tip, will show you how to implement Back Translation using [MarianMT](https://marian-nmt.github.io/) and Hugging Face's transformers library.

First, install dependencies
```
$ pip install transformers
$ pip install mosestokenizer
```

Second, download the MarianMT model and tokenizer for translating from English to [Romance languages](https://en.wikipedia.org/wiki/Romance_languages), and the ones for translating from [Romance languages](https://en.wikipedia.org/wiki/Romance_languages) to English.
```python
from transformers import MarianMTModel, MarianTokenizer

# Helper function to download data for a language
def download(model_name):
  tokenizer = MarianTokenizer.from_pretrained(model_name)
  model = MarianMTModel.from_pretrained(model_name)
  return tokenizer, model

# download model for English -> Romance
tmp_lang_tokenizer, tmp_lang_model = download('Helsinki-NLP/opus-mt-en-ROMANCE')
# download model for Romance -> English
src_lang_tokenizer, src_lang_model = download('Helsinki-NLP/opus-mt-ROMANCE-en')
```

Third, define helper functions to translate texts to a target language then use it to implement the back translation logic.
```python
def translate(texts, model, tokenizer, language):
  """Translate texts into a target language"""
  # Format the text as expected by the model
  formatter_fn = lambda txt: f"{txt}" if language == "en" else f">>{language}<< {txt}"
  original_texts = [formatter_fn(txt) for txt in texts]

  # Tokenize (text to tokens)
  tokens = tokenizer.prepare_seq2seq_batch(original_texts)

  # Translate
  translated = model.generate(**tokens)

  # Decode (tokens to text)
  translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

  return translated_texts

def back_translate(texts, language_src, language_dst):
  """Implements back translation"""
  # Translate from source to target language
  translated = translate(texts, tmp_lang_model, tmp_lang_tokenizer, language_dst)

  # Translate from target language back to source language
  back_translated = translate(translated, src_lang_model, src_lang_tokenizer, language_src)

  return back_translated
```

Finally, we can run some tests, for instance using French as a temporary language:
```python
src_texts = ['I might be late tonight', 'What a movie, so bad', 'That was very kind']
back_texts = back_translate(src_texts, "en", "fr")

print(back_texts)
# ['I might be late tonight.', 'What a movie, so bad', 'That was very kind of you.']
```

And using Spanish as a temporary language:
```python
src_texts = ['I might be late tonight', 'What a movie, so bad', 'That was very kind']

back_texts = back_translate(src_texts, "en", "es")
print(back_texts)
# ['I could be late tonight.', 'What a bad movie!', 'That was very kind of you.']
```

Check other supported languages for instance to chain more translations (e.g. English -> French -> English -> Spanish -> English)
```python
tokenizer.supported_language_codes
```
