---
layout: post

title: Image Preprocessing with Keras

tip-number: 
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Keras provides a set of very helpful images preprocessing utilities.
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - keras
---


## Image Preprocessing
* Use `image_load` to load an image into PIL format.
* Use `flow_images_from_data` or `flow_images_from_directory` to generates batches of augmented/normalized data from images and labels, or a directory
* Use `image_data_generator` to generate minibatches of image data with real-time data augmentation.
* Use `fit_image_data_generator` to fit image data generator internal statistics to some sample data
* Use  `generator_next` to retrieve the next item
* Use `image_to_array` or `image_array_resize` or `image_array_save` for 3D array representation

Data augmentation


https://github.com/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_with_vision_transformer.ipynb


```python
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)
```