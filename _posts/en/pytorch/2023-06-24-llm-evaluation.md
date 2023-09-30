---
layout: post

title: Evaluating LLMs Qualitatively and Quantitatively

tip-number: 35
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Learn how to evaluate LLMs Qualitatively (human evaluation) and Quantitatively (with ROUGE metrics)
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - pytorch
---

In this tip, we will see how to evaluate an LLM Qualitatively (human evaluation) and Quantitatively (with ROUGE metrics).



First, install the required packages for the LLM, datasets and evaluation.

```python
pip install transformers datasets evaluate rouge_score
```

Then, import the necessary modules

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import pandas as pd
```

For evaluation, we can use custom entries or take few entries from an existent dataset. Let's use the [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) dataset from Hugging Face which contains 10,000+ dialogues with the corresponding manually labeled summaries and topics. 

```python
dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(dataset_name)
test_ds = dataset['test']
```

LLMs usually require the prompt to have a certain structure. Let's define a helper function that transforms the input to the expected prompt format:

```python
def prompt_func(dialogue):
  text = f"\nSummarize the following conversation.\n\n{dialogue}\n\nSummary:\n"
  return [text]
```

We need a pre-trained LLM, let's use a small version for the purpose of this article from the family of [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5).

```python
model_name='google/flan-t5-base'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## Qualitative Evaluation (Human Evaluation)

With the qualitative approach we ask ourselves the question "Is my model behaving the way it is supposed to?". This is usually a good starting point for an LLM evaluation.

For example, below we visually check how good our test model is able to create summaries of the dialogue compared:


```python
dialogue = test_ds[0]['dialogue']
human_baseline_summary = test_ds[0]['summary']

input_ids = tokenizer(prompt_func(dialogue), return_tensors="pt").input_ids

model_outputs = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
model_text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)

print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}\n')
print(f'MODEL OUTPUT:\n{model_text_output}\n')
```

This is an example output

```
BASELINE HUMAN SUMMARY:
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.

MODEL OUTPUT:
#Person1#: You'd like to upgrade your computer. #Person2: You'd like to upgrade your computer.
```

## Quantitative Evaluation (with ROUGE Metric)

The [ROUGE metric](https://en.wikipedia.org/wiki/ROUGE_(metric)) helps quantify how good is the model is output. It compares the generated output to a "baseline" output which is usually the label created by a human.

First, we need to load the ROUGE metric with the `evaluate` module

```python
rouge = evaluate.load('rouge')
```

Second, we generate the outputs for the examples in the test dataset, and save the results.

```python
dialogues = test_ds['dialogue']
baseline_summaries = test_ds['summary']

model_summaries = []

for _, dialogue in enumerate(dialogues):
    input_ids = tokenizer(prompt_func(dialogue), return_tensors="pt").input_ids

    model_outputs = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    model_text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    model_summaries.append(model_text_output)
```

We can optionally visualize both summaries side by side in a DataFrame

```python
pd.DataFrame(list(zip(baseline_summaries, model_summaries)), columns = ['baseline_summaries', 'model_summaries'])
```

Finally, we evaluate the models computing ROUGE metrics as follows:

```python
model_results = rouge.compute(
    predictions=model_summaries,
    references=baseline_summaries,
    use_aggregator=True,
    use_stemmer=True,
)

print('MODEL ROUGE Metrics:')
print(model_results)
```

This is an example output

```
ORIGINAL MODEL:
{'rouge1': 0.24223171760013867, 'rouge2': 0.10614243734192583, 'rougeL': 0.21380459196706333, 'rougeLsum': 0.21740921541379205}
```
