#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:13:48 2017

@author: user
"""

from keras.models import *
from keras.layers import *
from keras.preprocessing.image import *
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG19
import h5py


def gap(MODEL, image_size, lambda_func=None):
    
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory("train", image_size, shuffle=False,                                            batch_size=20)
    test_generator = gen.flow_from_directory("test", image_size, shuffle=False, 
                                             batch_size=20, class_mode=None)

    train = model.predict_generator(train_generator, 171)
    test = model.predict_generator(test_generator, 3)
    with h5py.File("gap_%s.h5"%MODEL.func_name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)


gap(ResNet50, (224, 224))
gap(VGG19, (224, 224))
gap(InceptionV3, (224, 224))

#model1 = ResNet50(weights='imagenet',include_top=False)
#model2 = InceptionV3(weights='imagenet',include_top=False)
#model3 = VGG19(weights='imagenet',include_top=False)