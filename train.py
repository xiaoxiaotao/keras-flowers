#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:57:14 2017

@author: user
"""

from keras.models import *
from keras.layers import *
import make_parallel
np.random.seed(2017)

inputs = Input(X_train.shape[1:])
x = Dropout(0.5)(inputs)
x = Dense(1024,activation='sigmoid')(x)
x = Dense(512,activation='sigmoid')(x)
x = Dense(5, activation='softmax')(x)
model = Model(inputs, x)
model_p=make_parallel.make_parallel(model,2)
model_p.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model_p.fit(X_train, y_train, batch_size=20, nb_epoch=500, validation_split=0.2)

model_p.save('second_model.h5')


