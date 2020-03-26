import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from util import get_data
from variables import *

from sklearn.utils import shuffle

class AutoEncoder(object):
    def __init__(self,M,encoder_id):
        self.M = M
        self.encoder_id = encoder_id

    def fit(self,X):
        M = self.M
        N , D = X.shape
        Wh = tf.random.normal(shape=(D, M))
        bh = np.zeros(M)
        bo = np.zeros(D)

        self.Wh = tf.Variable(Wh, dtype=tf.float32)
        self.bh = tf.Variable(bh, dtype=tf.float32)
        self.bo = tf.Variable(bo, dtype=tf.float32)

        train_loss = []

        self.optimizer = tf.optimizers.Adam(0.01)
        n_batches = N // batch_size

        for epoch in range(num_epochs):
            epoch_loss = 0
            X = shuffle(X)
            for n in range(n_batches):
                with tf.GradientTape() as tape:
                    batch = X[n*batch_size:(n+1)*batch_size].to_numpy()
                    Xbatch = tf.constant(batch, dtype = tf.float32, shape=[batch_size,D])
                    Xhat_logits = self.fordward_logits(Xbatch)
                    loss = AutoEncoder.loss_funaction(Xbatch, Xhat_logits)
                epoch_loss += float(loss.numpy())
                grads = tape.gradient(loss, [self.Wh, self.bh, self.bo])
                self.optimizer.apply_gradients(zip(grads,[self.Wh, self.bh, self.bo]))
            train_loss.append(epoch_loss)
            print("epoch: ", epoch, " loss :", epoch_loss)

        plt.plot(train_loss)
        plt.show()

    def hidden(self,X):
        return tf.nn.sigmoid(tf.matmul(X, self.Wh) + self.bh)

    def fordward_logits(self,X):
        Z = self.hidden(X)
        Xhat_logits_logits = tf.matmul(Z, tf.transpose(self.Wh)) + self.bo
        return Xhat_logits_logits

    def fordward(self,X):
        return tf.nn.softmax(self.fordward_logits(X))

    @staticmethod
    def loss_funaction(X, Xhat_logits):
        return tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=X,
                            logits=Xhat_logits,
            ))

    def train_or_load(self):
        if not os.path.exists(saved_weights):
            print("Training and saving")
            self.fit(Xtrain)
            np.save(saved_weights, [self.Wh, self.bh, self.bo])
        else:
            print("loading weights")
            self.Wh, self.bh, self.bo = np.load(saved_weights, allow_pickle=True)

    def test(self, X):
        N, D = X.shape
        l = int(np.sqrt(D))
        for _ in range(5):
            idx = np.random.choice(N)
            Ximage = tf.constant(X.iloc[idx,:].to_numpy().reshape(1, D), dtype = tf.float32, shape=[1,D] )
            Xhat_logits = self.fordward_logits(Ximage)

            Ximage = Ximage.numpy().reshape(l, l)
            Xhat_logits   = Xhat_logits.numpy().reshape(l, l)

            plt.subplot(1,2,1)
            plt.imshow(Ximage, cmap='gray')
            plt.title('Original image')

            plt.subplot(1,2,2)
            plt.imshow(Xhat_logits, cmap='gray')
            plt.title('Reconstructed image')

            plt.show()

if __name__ == "__main__":
    Xtrain, Ytrain, Xtest , Ytest = get_data()
    encoder = AutoEncoder(1000,0)
    encoder.train_or_load()
    encoder.test(Xtrain)