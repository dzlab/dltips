---
layout: post

title: Merge LLMs with MergeKit

tip-number: 39
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to merge multiple LLMs into a single model with MergeKit
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - pytorch
---


[MergeKit](https://github.com/cg123/mergekit) is an open source CLI that can be used to merge LLMs without writing a single line of code. It takes as input a YAML configuration file that specifies the operations to perform in order to produce the merged model.


The following is an example configuration file that aims to merge three models using the `linear` merge method and the `float16` as a data type for the merge operatio. You can find more configuration files in the [examples](https://github.com/cg123/mergekit/tree/main/examples) folder.

```yaml
models:
  - model: psmathur/orca_mini_v3_13b
    parameters:
      weight: 1.0
  - model: WizardLM/WizardLM-13B-V1.2
    parameters:
      weight: 0.3
  - model: garage-bAInd/Platypus2-13B
    parameters:
      weight: 0.5
merge_method: linear
dtype: float16
```


Few things to consider when merging models with MergeKit is to have a system with minimum physical requirements:

- Enough memory to load all source models weights
- Enough disk space to save source models as well as the final model
- An Internet with fast upload speed for sharing the final model

For instance, to be able to merge two 7B models we will need ~15 GB of memory and at least 30 GB of disk space. If not enough memory is available, e.g. running on Google Colab which has around 12GB of memory, we can chunk the model into shards of size 1B. However, the merge process will be slower and take longer to finish.


To be able to use MergeKit, first clone the github repo of install the package like this:

```shell
pip install -e git+https://github.com/cg123/mergekit
```

Then, define the merge configuration file as illustrated above and then run `mergekit-yaml` command, for instance:

```shell
mergekit-yaml path/to/config.yml path/to/output-model-directory \
  --allow-crimes \
  --out-shard-size 1B
```

> Note: with the `--out-shard-size` option, we are specifying a shard size of 1B which may split the merged model into many chunks. But, we can also make the shard size even smaller, e.g. 500M or even 250M, which can be helpful when uploading the model chunks.


Optionally, upload the model to Hugging Face

1. Create a new model [here](https://huggingface.co/new)
2. Run the code below to install [huggingface-hub](https://pypi.org/project/huggingface-hub/) library

```shell
pip install -U huggingface-hub
```

3. Run the code below to login to Hugging Face Hub

```shell
huggingface-cli login
```

4. Upload the model weights to Hugging Face Hub

```python
REPO_ID = 'username/model'
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="mergedmodel",
    repo_id=REPO_ID,
    repo_type="model",
)
```
