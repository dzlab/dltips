---
layout: post

title: Fine-tuning Llama 2 with axolotl

tip-number: 38
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to fine-tune Llama 2 on custom dataset with axolotl
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - pytorch
---

```
                                                   dP            dP   dP 
                                                   88            88   88 
                        .d8888b. dP.  .dP .d8888b. 88 .d8888b. d8888P 88 
                        88'  `88  `8bd8'  88'  `88 88 88'  `88   88   88 
                        88.  .88  .d88b.  88.  .88 88 88.  .88   88   88 
                        `88888P8 dP'  `dP `88888P' dP `88888P'   dP   dP 
```

[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) is a library that makes fine-tuning LLMs an easy task by using a configuration-first approach. With a single YAML file we can customize the training by defining parameters for: 
- Huggingface models: llama, pythia, falcon, mpt
- Huggingface datasets and formats
- Single vs Multiple GPUs training via FSDP or Deepspeed
- Fine-tuning technique: fullfinetune, lora, qlora, relora, and gptq
- WandB configuration to log metrics, results, checkpoints

This tip, will show you how to fine-tune Llama 2 using axolotl. We will fine-tune Llama on the [knowrohit07/know_sql](https://huggingface.co/datasets/knowrohit07/know_sql) dataset which is a text to sql dataset.

Below are few examples from this datasets. It contains a `context` which represents the definition of one or multiple SQL tables, a `question` about those tables, and the ground truth SQL `answer`.

|answer|question|context|
|-|-|-|
|SELECT COUNT(district) FROM table_1341586_19 WHERE incumbent = "Lindy Boggs"|how many district with incumbent being lindy boggs|CREATE TABLE table_1341586_19 (district VARCHAR, incumbent VARCHAR)|
|SELECT result FROM table_1341586_19 WHERE candidates = "Billy Tauzin (D) Unopposed"|what's the result with candidates being billy tauzin (d) unopposed|CREATE TABLE table_1341586_19 (result VARCHAR, candidates VARCHAR)"|

> Note: you can choose any other dataset from [Hugging Face](https://huggingface.co/datasets).

Before starting the fine-tuning, first check that a GPU is available and that bf16 mode is supported

```python
import torch

print('GPU available?', torch.cuda.is_available())
print('BF16 is supported?', torch.cuda.is_bf16_supported())
```

Second, install dependencies including `axolotl` and `peft`

```shell
git clone -b main --depth 1 https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip install packaging
pip install -e '.[flash-attn,deepspeed]'
pip install -U git+https://github.com/huggingface/peft.git
```

`axolotl` uses a YAML file to configure the fine-tuning. You can see how such files looks like by visiting the [examples](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples) folder.

In our case, we will use the [llama-2/qlora.yml](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/qlora.yml) example and apply the following git patch that sets the base model to `meta-llama/Llama-2-7b-hf` and the dataset path and type.


```diff
diff --git a/content/axolotl/examples/llama-2/qlora.yml b/sql.yml
index 5425532..eb395cb 100644
--- a/content/axolotl/examples/llama-2/qlora.yml
+++ b/sql.yml
@@ -1,2 +1,2 @@
-base_model: NousResearch/Llama-2-7b-hf
-base_model_config: NousResearch/Llama-2-7b-hf
+base_model: meta-llama/Llama-2-7b-hf
+base_model_config: meta-llama/Llama-2-7b-hf
@@ -12,2 +12,2 @@ datasets:
-  - path: mhenrichsen/alpaca_2k_test
-    type: alpaca
+  - path: knowrohit07/know_sql
+    type: context_qa2
```

Now we can start the fine-tuning (depending on your system this may take around 1h)

```shell
accelerate launch -m axolotl.cli.train sql.yml
```

Once, `axolotl` is done the qlora weights of the model will be available in the `qlora-out` folder. We need to apply those weights to the original llama-2

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

qlora_model = './qlora-out'
base_model = 'meta-llama/Llama-2-7b-hf'
tokr = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map=0)
model = PeftModel.from_pretrained(model, qlora_model)
model = model.merge_and_unload()
```

Then save the new weights so we can reuse them in the future.

```python
model.save_pretrained('sql-model')
```

Now we are ready to test our fine-tuned text-to-sql model. Let's define a helper function that format the prompt we will pass to our model with the same format used in the fine-tuning dataset

```python
fmt = """SYSTEM: Use the following contextual information to concisely answer the question.

USER: {}
===
{}
ASSISTANT:"""

def sql_prompt(context, question): return fmt.format(context, question)
```

Let's verify that our helper prompt function works as expecting with the following example

```python
context = 'CREATE TABLE farm_competition (Hosts VARCHAR, Theme VARCHAR)'
question = 'Get the count of competition hosts by theme.'
print(sql_prompt(context, question))
```

The above snippet will generate the following output

```
SYSTEM: Use the following contextual information to concisely answer the question.

USER: CREATE TABLE farm_competition (Hosts VARCHAR, Theme VARCHAR)
===
Get the count of competition hosts by theme.
ASSISTANT:
```

Finally, we can pass this prompt to our fine-tuned model

```python
toks = tokr(sql_prompt(context, question), return_tensors='pt')
res = model.generate(**toks.to('cuda'), max_new_tokens=250).to('cpu')
print(tokr.batch_decode(res)[0])
```

This will output something like this

```
<s> SYSTEM: Use the following contextual information to concisely answer the question.

USER: CREATE TABLE farm_competition (Hosts VARCHAR, Theme VARCHAR)
===
Get the count of competition hosts by theme.
ASSISTANT: SELECT COUNT(Hosts), Theme FROM farm_competition GROUP BY Theme</s>
```

> Note: `<s>` and `</s>` are extra tokens added by the tokenizer that denotes the start and end of text respectively.
