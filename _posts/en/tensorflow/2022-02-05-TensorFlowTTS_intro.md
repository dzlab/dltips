---
layout: post

title: English Text to speech with TensorFlowTTS

tip-number: 35
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Learn how to generate English speech from a text using TensorFlowTTS
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---


[TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS) is a Speech Synthesis library for Tensorflow 2, it can be used to generate speech in many languages including: English, French, Korean, Chinese, German. This library can also be easily adapted to generate speech in other languages.

In this tip, we will use TensorFlowTTS to generate english speech from a random text

First, we need to install the library

```
$ pip install git+https://github.com/TensorSpeech/TensorFlowTTS.git
$ pip install git+https://github.com/repodiac/german_transliterate.git#egg=german_transliterate
```

Then, we import the needed packages

```python
import tensorflow as tf

import yaml
import numpy as np

import IPython.display as ipd

from transformers import pipeline

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import AutoProcessor
```

Now, we load the pretrained model for the English language which was trained on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) corpus.

```python
tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en", name="tacotron2")
melgan = TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en", name="melgan")
```

We also, need to instantiate the inference model that will process the text

```python
processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
```

To simplify the generation of speech, we will define the following helper function which will call perform inference

```python
def text2speech(input_text, text2mel_model, vocoder_model):
    input_ids = processor.text_to_sequence(input_text)
    # text2mel part
    _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        tf.convert_to_tensor([len(input_ids)], tf.int32),
        tf.convert_to_tensor([0], dtype=tf.int32)
        )
    # vocoder part
    audio = vocoder_model(mel_outputs)[0, :, 0]
    return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
```


Finally we call the helper function on a random text to generate the corresponding speech:

```python
story = 'This story is called Breathe in Breathe out-of-Flight, a novel by George Kraszewski'.
mels, alignment_history, audios = text2speech(story, tacotron2, melgan)

ipd.Audio(audios, rate=22050)
```

Here are more inference examples with each model at [notebooks](https://github.com/tensorspeech/TensorFlowTTS/tree/master/notebooks). To convert the model to TF Lite format see [colab](https://colab.research.google.com/drive/1HudLLpT9CQdh2k04c06bHUwLubhGTWxA?usp=sharing). For language specific examples, see [colab](https://colab.research.google.com/drive/1akxtrLZHKuMiQup00tzO2olCaN-y3KiD?usp=sharing) (for English), [colab](https://colab.research.google.com/drive/1ybWwOS5tipgPFttNulp77P6DAB5MtiuN?usp=sharing) (for Korean), [colab](https://colab.research.google.com/drive/1YpSHRBRPBI7cnTkQn1UcVTWEQVbsUm1S?usp=sharing) (for Chinese), [colab](https://colab.research.google.com/drive/1jd3u46g-fGQw0rre8fIwWM9heJvrV1c0?usp=sharing) (for French), [colab](https://colab.research.google.com/drive/1W0nSFpsz32M0OcIkY9uMOiGrLTPKVhTy?usp=sharing) (for German).
