---
layout: post

title: TensorFlow Projector

tip-number: 10
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Once we have embeddings and their metadata how could we visualize them?
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - tensorflow
---



TensorFlow [Projector](http://projector.tensorflow.org/) is visuale tool that let the user intercat and analyze high demensional data (e.g. embeddings) and their metadata, by projecting them in a 3D space on the browser. Here is a preview of this tool:


<h3 style="text-align:center;">
  <iframe class="post-content" width="1000" height="600" src="https://projector.tensorflow.org/" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</h3>


To use this tool for analysing your data (e.g. [wiki emebeddings](https://github.com/dzlab/deepprojects/blob/master/tensorflow/wiki_clustering_projector.ipynb)), it is very simple:

1. Go to http://projector.tensorflow.org/
2. Upload a first .tsv file that contains the embedding vectors with tab separated cells.
3. Optionally, upload a .tsv metada file with each row corresponding to embeddings in the previous file.

Once loaded, the embeddings will become available in the projector canvas, after that the tool will start analyzing the data given the selected method (e.g. tSNE, PCA).

Each embedding will have a position and some points will be close to each other forming clusters that could be interepreted. For instance, if the uploaded data representing embeddings of words (e.g. word2vec data), then we can expect a cluster of points that group positive words and another one on the opposite side representing negative words.

By selecting one point in Projector, we can see what other points are close to alone with a similarity score. For instance, if we loaded a subset of word2vec, and selected a the `like` word then we may expect the closet point will represent likes or liked or likely. Also we would expect to see the further points to represent words like dislike or disliked.