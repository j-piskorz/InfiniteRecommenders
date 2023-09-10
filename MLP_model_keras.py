import numpy as np
import keras
from keras import initializers
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Dense, Lambda, Activation
from keras.layers import Input, Dense
from keras.optimizers import Adam
from time import time

import scipy.sparse as sp
import pandas as pd


def make_MLP_model(X_train, y_train, hyper_params):

    # build model
    input = Input(shape=(hyper_params['num_items'],), dtype='float32', name = 'user_input')
    vector = input
    for w in range(hyper_params['depth']):
        layer = Dense(hyper_params['width'], kernel_regularizer=l2(hyper_params['reg']),
                      activation='relu', name = f'hidden_layer_{w}',
                      kernel_initializer=initializers.LecunUniform())
                      #bias_initializer=initializers.RandomNormal(stddev=0.1))
        vector = layer(vector)
    prediction = Dense(hyper_params['num_items'], activation=None,
                       kernel_initializer=initializers.LecunUniform(),
                       #bias_initializer=initializers.RandomNormal(stddev=0.1),
                       name = 'prediction')(vector)

    model = Model(inputs=input,
                  outputs=prediction)

    model.compile(optimizer=Adam(learning_rate=hyper_params['adam_lr']), loss='mse')

    # training
    hist = model.fit(X_train, #input
                     y_train, # labels
                     batch_size=hyper_params['batch_size'],
                     epochs=hyper_params['epochs'], verbose=0, shuffle=True)

    def predict_MLP(X_test):
      return model.predict(X_test, verbose=0)

    return predict_MLP

def make_MLP_model_last_layer(X_train, y_train, hyper_params):

    # build model
    input = Input(shape=(hyper_params['num_items'],), dtype='float32', name = 'user_input')
    vector = input
    for w in range(hyper_params['depth']):
        layer = Dense(hyper_params['width'], kernel_regularizer=l2(hyper_params['reg']),
                      activation='relu', name = f'hidden_layer_{w}',
                      kernel_initializer=initializers.LecunUniform(),
                      trainable = False)
                      #bias_initializer=initializers.RandomNormal(stddev=0.1))
        vector = layer(vector)
    prediction = Dense(hyper_params['num_items'], activation=None,
                       kernel_initializer=initializers.LecunUniform(),
                       #bias_initializer=initializers.RandomNormal(stddev=0.1),
                       name = 'prediction')(vector)

    model = Model(inputs=input,
                  outputs=prediction)

    model.compile(optimizer=Adam(learning_rate=hyper_params['adam_lr']), loss='mse')

    # training
    hist = model.fit(X_train, #input
                     y_train, # labels
                     batch_size=hyper_params['batch_size'],
                     epochs=hyper_params['epochs'], verbose=0, shuffle=True)

    def predict_MLP(X_test):
      return model.predict(X_test, verbose=0)

    return predict_MLP