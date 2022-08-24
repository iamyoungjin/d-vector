import config as c
import os
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten, Conv2D, MaxPooling2D

class CNNModel:
    def __init__(self):
        self.inputShape = (c.INPUT_SHAPE)

    def build(INPUT_SHAPE):
        model = Sequential()
        model.add(Conv2D(256, 5, input_shape=(INPUT_SHAPE)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(256, 5, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))    
        model.add(Conv2D(256, 5, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))

        modelInput = Input(shape=(INPUT_SHAPE ))
        features = model(modelInput)
        embd_model = Model(inputs = modelInput, outputs=features)

        model1 = Activation('relu')(features)
        model1 = Dropout(0.2)(model1)
        model1 = Dense(5, activation='softmax')(model1)

        model = Model(inputs=modelInput, outputs=model1)
        return embd_model, model
