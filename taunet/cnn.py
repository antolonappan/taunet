import numpy as np
import healpy as hp
import os
import pickle as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import random as python_random
import nnhealpix.layers
from tensorflow.keras import metrics

class TauNet:
    def __init__(self, nside=16,batch_size=32, max_epochs=45, reduce_lr_on_plateau=True,nmaps=4):
        self.nside = nside
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.nmaps = nmaps
        self.model = self.create_model()

    @staticmethod
    def new_loss(y_true, y_pred):
        squared_residual = tf.math.square(y_true[:, 0] - y_pred[:, 0])
        squared_sigma = tf.math.square(y_pred[:, 1])

        loss = tf.math.reduce_sum(squared_residual)
        loss += tf.math.reduce_sum(tf.math.square(squared_residual - squared_sigma))

        return loss

    def create_model(self):
        shape = (hp.nside2npix(self.nside), self.nmaps)
        inputs = tf.keras.layers.Input(shape)
        # nside 16 -> 8
        x = nnhealpix.layers.ConvNeighbours(self.nside, filters=32, kernel_size=9)(inputs)
        x = tf.keras.layers.Activation('relu')(x)
        x = nnhealpix.layers.Dgrade(self.nside, self.nside//2)(x)
        # nside 8 -> 4
        x = nnhealpix.layers.ConvNeighbours(self.nside//2, filters=32, kernel_size=9)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = nnhealpix.layers.Dgrade(self.nside//2, self.nside//4)(x)
        # nside 4 -> 2
        x = nnhealpix.layers.ConvNeighbours(self.nside//4, filters=32, kernel_size=9)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = nnhealpix.layers.Dgrade(self.nside//4, self.nside//8)(x)
        # nside 2 -> 1
        x = nnhealpix.layers.ConvNeighbours(self.nside//8, filters=32, kernel_size=9)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = nnhealpix.layers.Dgrade(self.nside//8, self.nside//16)(x)
        # dropout
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(48)(x)
        x = tf.keras.layers.Activation('relu')(x)
        out = tf.keras.layers.Dense(2)(x)

        tf.keras.backend.clear_session()
        model = tf.keras.models.Model(inputs=inputs, outputs=out)
        return model

    def compile_and_fit(self, X_train, y_train, X_valid, y_valid):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, mode='min'
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=5
        )

        self.model.compile(
            loss=self.new_loss,  # Use the class's static method
            optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.MeanSquaredError()]
        )

        callbacks = [early_stopping]
        if self.reduce_lr_on_plateau:
            callbacks.append(reduce_lr)

        history = self.model.fit(
            x=X_train, y=y_train,
            batch_size=self.batch_size, epochs=self.max_epochs,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks
        )

        return history
    
    def predict(self,X_test):
        return self.model.predict(X_test)
    
    def save(self,fname):
        self.model.save(fname)
    
    def load(self,fname):
        self.model = tf.keras.models.load_model(fname,custom_objects={'OrderMap': nnhealpix.layers.OrderMap,'new_loss':self.new_loss})