---
layout: post

title: LLMs fine-tuning

tip-number: 36
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Learn the techniques to fine-tune an LLM
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - pytorch
---

In this tip, we will fine-tune an LLM with two techniques; full fine-tuning and with Parameter Efficient Fine-Tuning (PEFT). We will use the [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) LLM, which is a high quality instruction tuned model. 


![Different LLM tuning techniques](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2023/09/11/Optimize_generative_AI_workloads_for_Sustainabilit_11092023_1-1.png)_Source [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/optimize-generative-ai-workloads-for-environmental-sustainability/)_

First, install the required packages for the LLM, datasets and PEFT.
```shell
pip install -q torch torchdata transformers datasets loralib peft
```

Then, import the necessary modules

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
```

Then, load the pre-trained [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) model from HuggingFace.

```python
model_name='google/flan-t5-base'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## Dataset

For fine-tuning, we can use custom entries or take few entries from an existent dataset. In our case we will use the [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) dataset from Hugging Face which contains 10,000+ dialogues with the corresponding manually labeled summaries and topics.

```python
dataset = load_dataset("knkarthick/dialogsum")
```

We need to convert the prompt-response pairs of the dataset into a instructions for the LLM. For instance:

```
Summarize the following conversation.

    Person 1: Hi.
    Person 2: Hi, how are you?
    
Summary:
Person 1 and Person 2 are greeting each other.
```

Let's define a helper function that takes an example from the dataset and convert it to a prompt

```python
def prompt_func(example):
    START = 'Summarize the following conversation.\n\n'
    END = '\n\nSummary: '
    prompt = [START + dialogue + END for dialogue in example["dialogue"]]
    return prompt
```

Then define a helper function to preprocess the prompt-response dataset into tokens:

```python
def tokenize_function(example):
    prompt = prompt_func(example)
    input_ids = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    labels = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return {'input_ids': input_ids, 'labels': labels}
```

Now, we apply the `tokenize_function` on the different splits in the dataset (train, validation and test) in batches.

```python
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

Optinally, we can subsample examples from the dataset:

```python
tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)
```


## Full Fine-Tuning

We will use Hugging Face [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)) for full-tuning as follows:

```python
output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=1,
    max_steps=1
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)
```

Then start the training by simply calling `train`:

```python
trainer.train()
```


## Parameter Efficient Fine-Tuning (PEFT)

PEFT is a form of instruction fine-tuning that is much more efficient than full fine-tuning - with comparable evaluation results. PEFT includes fine-tuning techniques like **Low-Rank Adaptation (LoRA)** and prompt tuning (which is NOT THE SAME as prompt engineering!). 

In our case we will use LoRA which allows the user to fine-tune their model using fewer compute resources (in some cases, a single GPU). Using LoRA, we freeze the underlying LLM and only training the adapter. After fine-tuning with LoRA, the result is that the original LLM remains unchanged and a newly-trained “LoRA adapter” emerges. This LoRA adapter is much, much smaller than the original LLM - on the order of a single-digit % of the original LLM size (MBs vs GBs).

At inference time, the LoRA adapter needs to be reunited and combined with its original LLM to serve the inference request. The benefit, however, is that many LoRA adapters can re-use the original LLM which reduces overall memory requirements when serving multiple tasks and use cases.

First, we need to set up the PEFT/LoRA model for fine-tuning with a new layer/parameter adapter

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)
```

> Note: the `r` hyper-parameter defines the rank/dimension of the adapter to be trained.

Then, we add the LoRA adapter layers/parameters to our base LLM

```python
peft_model = get_peft_model(original_model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))
```

Then, define training arguments and create `Trainer` instance.

```python
output_dir = f'./peft-DialogSum-training'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
    logging_steps=1,
    max_steps=1    
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)
```

And train the PEFT Adapter

```python
peft_trainer.train()
```

Once training finishes, we save the parameters

```python
peft_model_path="./peft-DialogSum-checkpoint"

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)
```

At infenrece, we need to add the PEFT adapter to the original LLM

```python
from peft import PeftModel, PeftConfig

base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_model = PeftModel.from_pretrained(
    base_model,
    peft_model_path,
    torch_dtype=torch.bfloat16,
    is_trainable=False # inference mode
    )
```

## Evaluation

For evaluating the resulting models from the fine-tuning techniques discusser earlier, you can refer to a previous tip on evaluating LLMs Qualitatively (human evaluation) and Quantitatively (with ROUGE metrics) - [link]({{ "/en/pytorch/llm-evaluation/" | absolute_url }}).

