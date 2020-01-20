---
layout: post

title: VizSeq on Google Colab

tip-number: 14
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: How to use VizSeq on Google Colab to visually analyze Text Generation Models?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - more
---

Facebook AI's [VizSeq](https://ai.facebook.com/blog/vizseq-a-visual-analysis-toolkit-for-accelerating-text-generation-research/) is an intuitive tool that helps analyzing the performance of text generation dataset and models under. This tool can be deployed as a web application but can also be used in Google Colab.

![](https://cdn-images-1.medium.com/max/800/1*Ff7BTxmEjUXHtYu9JkfClg.jpeg)

Install the package
```
!pip install -q git+https://github.com/facebookresearch/vizseq
```

Given a source text and a target text, VizSeq let you quickly visualize statistics on the dataset, like: number of examples, sequence lengths, etc.

```python
%matplotlib inline
vizseq.view_stats(sources, references)
```

VizSeq let you also visualize statistics on the ngrams on the dataset

```python
vizseq.view_n_grams(sources)
```

VizSeq let you also visualize of metrics (e.g. BLUE) to analyze the performance of your model

```python
vizseq.view_scores(actual_seq, predicted_seq, ['bleu', 'meteor'])
```

Another hady feature of VizSeq is the manual inspect of the dataset input/output examples of your model against true values:

```python
vizseq.view_examples(input_seq, actual_seq, predicted_seq, ['bleu'], page_sz=2, page_no=12)
```

Here is notebook example that can be run on Colab - [link](https://github.com/dzlab/deepprojects/blob/master/nlp/Hello_VizSeq.ipynb)