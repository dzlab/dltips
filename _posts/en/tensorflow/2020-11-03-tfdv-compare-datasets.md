---
layout: post

title: Comparing Datasets with TFDV

tip-number: 33
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to compare datasets and spot potential issues?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---

TFDV (TFX Data Validation) is a Python package that is part of TensorFlow eXtended ecosystem, and implement techniques for data validation and schema generation.

```bash
$ pip install tensorflow-data-validation
```

It is usually used in the data validation step of a TFX pipeline to check the data before it is feeded to the data processing and actual training steps.

It is also used to compare multiple datasets (e.g. training vs validation) and helps significantly different are they (e.g. different schema, missing values, etc).

In this TIP, we will use TFDV in a standalone mode to:
* Load two datasets from CSV files
* Generate satistics for each one
* Compare these statistics

```python
import tensorflow_data_validation as tfdv

# Load datasets and generate statistics
ds1_stats = tfdv.generate_statistics_from_csv(
  data_location='data_1.csv',
  delimiter=','
)
ds2_stats = tfdv.generate_statistics_from_csv(
  data_location='data_2.csv',
  delimiter=','
)

# Compare statistics
tfdv.visualize_statistics(
  lhs_statistics=ds1_stats, lhs_name='DS-I',
  rhs_statistics=ds2_stats, rhs_name='DS-II'
)
```

An example of an interactive visualization of the comparison of two datasets would look like this:

> Note how numercial data vs categorical data are compared

<h3 style="text-align:center;">
<iframe id="" width="100%" height="600px" srcdoc="<script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js&quot;></script><link rel=&quot;import&quot; href=&quot;https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html&quot;><facets-overview proto-input=&quot;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&quot;></facets-overview>"></iframe>
</h3>