import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from vgg16 import VGG16
from keras.preprocessing import image

from keras.layers import Input
from keras.models import Model


# Use VGG16 model as an image feature extractor 
image_input = Input(shape=(224, 224, 3))
model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
layer_name = 'fc2'
feature_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)


def get_feature_vector_fromPIL(img):
    feature_vector = feature_model.predict(img)
    assert(feature_vector.shape == (1,4096))
    return feature_vector

def calculate_similarity_cosine(vector1, vector2):
    return cosine_similarity(vector1, vector2)
