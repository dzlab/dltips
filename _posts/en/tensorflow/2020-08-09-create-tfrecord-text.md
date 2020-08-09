---
layout: post

title: Create TFRecord from a text dataset

tip-number: 25
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to deal with large text datasets?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---

To deal with large datasets that does not fit in memory, we would need to encode them into TFRecord then load them during trainnig. The [tf_models](https://github.com/tensorflow/models) library includes some tools for processing and re-encoding a dataset into an dfrom TFRecords for efficient training.


> pip install tf-models -q

Import necessary packages from the tf_models library

```python
from official import nlp
from official.nlp import bert
import official.nlp.bert.run_classifier
import official.nlp.data.classifier_data_lib
```

First, we need to describe what features of the dataset will be transformed using one of the `DataProcessor` class. For each row of the input data, this class generates a `InputExample` instance (from `official.nlp.data.classifier_data_lib` package).

The `tf_models` library already has couple of implementation for specific Datasets, here is the list:

| Class Name | Dataset | Description |
|------------|---------|-------------|
| ColaProcessor | CoLA | [Corpus of Linguistic Acceptability](https://nyu-mll.github.io/CoLA/)|
| MnliProcessor | MultiNLI | [Multi-Genre Natural Language Inference](https://cims.nyu.edu/~sbowman/multinli/) |
| MrpcProcessor | MRPC | [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398) |
| PawsxProcessor | PAWS-X | [Paraphrase Adversaries from Word Scrambling](https://github.com/google-research-datasets/paws/tree/master/pawsx) |
| QnliProcessor | QNLI | [GLUE: A MULTI-TASK BENCHMARK AND ANALYSIS PLATFORM FOR NATURAL LANGUAGE UNDERSTANDING](https://arxiv.org/pdf/1804.07461.pdf)|
| QqpProcessor | QQP | [Indentification of Semantic Duplicates, or Dilettante Dive into the Quora Data Set](https://github.com/sdll/QQP) |
| RteProcessor | RTE | [Recognizing Textual Entailment](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment) |
| SstProcessor | SST-2 | [Stanford Sentiment Treebank](https://github.com/AcademiaSinicaNLPLab/sentiment_dataset) |
| StsBProcessor | STS-B | [STS Benchmark](https://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)|
| TfdsProcessor | TF Datasets | Generic processor for text classification and regression TF Datasets. |

For instance, to create a Data processor for the MRPC dataset use `TfdsProcessor` as follows
```python
processor = nlp.data.classifier_data_lib.TfdsProcessor(
  tfds_params="dataset=glue/mrpc,text_key=sentence1,text_b_key=sentence2",
  process_text_fn=bert.tokenization.convert_to_unicode
  )
```

Second, we apply this processor on the raw dataset to generate TFRecords for training/validation and test.

```python
# Generate and save training data into a tf record file
nlp.data.classifier_data_lib.generate_tf_record_from_data_file(
  processor = processor,
  data_dir = None,  # It is `None` because data is from tfds, not local dir.
  tokenizer = tokenizer,
  train_data_output_path = "./train.tf_record",
  eval_data_output_path = "./eval.tf_record",
  test_data_output_path = "./test.tf_record",
  max_seq_length = 128
  )
```

Finally, create a `tf.data` Dataset to load the TF Records from those TFRecord files using helper function `get_dataset_fn`:
```python
max_seq_length = 128
batch_size = 32
eval_batch_size = 16
train_ds = bert.run_classifier.get_dataset_fn(
    "./train.tf_record",
    max_seq_length,
    batch_size,
    is_training=True)()

eval_ds = bert.run_classifier.get_dataset_fn(
    "./eval.tf_record",
    max_seq_length,
    eval_batch_size,
    is_training=False)()

test_ds = bert.run_classifier.get_dataset_fn(
    "./test.tf_record",
    max_seq_length,
    eval_batch_size,
    is_training=False)()
```

Note: if you cannot find a DataProcessor implementation that works for your dataset you can build your own processor by subclassing `DataProcessor` class. For example:

```python
class MyDatasetProcessor(DataProcessor):
  def __init__(self, process_text_fn=bert.tokenization.convert_to_unicode):
    super(DataFrameProcessor, self).__init__(process_text_fn)
    ...

  def get_train_examples(self, data_dir = None):
    """Create training examples."""
    return self._create_examples("train")

  def get_dev_examples(self, data_dir = None):
    """Create evaluation examples."""
    return self._create_examples("dev")

  def get_test_examples(self, data_dir = None):
    """Create testing examples."""
    return self._create_examples("test")

  def get_labels(self):
    """Get the list of labels."""
    return [...]

  @staticmethod
  def get_processor_name():
    """Get the name of this processor."""
    return "MyDataset"

  def _create_examples(self, set_type):
    """Creates examples for the training/dev/test sets."""
    for i, data in enumarate(...):
      guid = "%s-%s" % (set_type, i)
      text_a = self.process_text_fn(data['text_a'])
      text_b = self.process_text_fn(data['text_b']) # or None if there is no text_b
      label = self.process_text_fn(data['label']) if set_type!="test" else None
      # construct an example
      yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
```