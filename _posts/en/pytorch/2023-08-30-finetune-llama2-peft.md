---
layout: post

title: Fine tune Llama 2 on custom data with PEFT

tip-number: 39
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Learn how to fine tune any LLM such as Llama2 on custom data using PEFT
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - pytorch
---

In this tip, we will see how to fine tune Llama 2 (or any other foundational LLM) on custom datasets using a collection of libraries from HuggingFace: transformers, peft, etc.

First, install dependencies:
```shell
pip install -q huggingface_hub
pip install -q -U trl transformers accelerate peft
pip install -q -U datasets bitsandbytes einops wandb
pip install  -q ipywidgets
pip install -q scipy
```

and import all needed modules

```python
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
```

Depending on the foundational model you are using, you may optionally need to authenticate to HuggingFace
```shell
huggingface-cli login
```

## Data

A good format to structure a trainig dataset is to use a `.jsonl` structured file, where each row is a JSON object representing a training example with an input for the model and the associated output:

```json
{"input": "What is the 'ultraviolet catastrophe'?", "output": "It is the misbehavior of a formula for higher frequencies."}
{"input": "Where did Ibn Battuta travel to after his visit to the Chagatai Khanate?", "output": "Constantinople"}
{"input": "From where did Ibn Battuta travel to Yemen after the hajj?", "output": "He traveled via the Red Sea."}
```

Now we can load the dataset us the `datasets` library as follows

```python
train_dataset = load_dataset('json', data_files='train.jsonl', split='train')
test_dataset = load_dataset('json', data_files='test.jsonl', split='test')
```

To present the example as a prompt to the LLM, we need to create a formatting function `prompt_func`:
```python
def prompt_func(example):
  text = f"### Question: {example['input']}\n ### Answer: {example['output']}"
  return [text]
```

## Fine-tuning

Next, we load the foundational LLM as we well:

```python
base_model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
```

Also, load the tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
```

Then, we define the training arguments in a `TrainingArguments` object

```python
training_args = TrainingArguments(
    output_dir="./Llama-2-7b-hf-fine-tuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=50,
    max_steps=1000,
    logging_dir="./logs",         # Directory for storing logs
    save_strategy="steps",        # Save the model checkpoint every logging step
    save_steps=50,                # Save checkpoints every 50 steps
    evaluation_strategy="steps",  # Evaluate the model every logging step
    eval_steps=50,                # Evaluate and save checkpoints every 50 steps
    do_eval=True                  # Perform evaluation at the end of training
)

max_seq_length = 512
```

As well as the config for the Lora adapter:

```python
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
```

```python
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_func,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)
```

Finally, we kick start the fine-tuning:

```python
trainer.train()
```

## Inference
To use the fine-tuned version of the model, we need to load the weights of the base model and then merge it with the QLora weights which were saved by the PEFT library.

First step, load the base model
```python
base_model_name="meta-llama/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
```

Second step, load the QLora weights and merge them with the base model

```python
model = PeftModel.from_pretrained(base_model, "./Llama-2-7b-hf-fine-tuned")
```

Now, we can use the model to run inference

```python
eval_prompt = f"### Question: What is the stance on Ibn Battuta's Rihla?\n ### Answer: "

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    toks = model.generate(**model_input, max_new_tokens=100)[0]
    print(tokenizer.decode(toks, skip_special_tokens=True))
```
