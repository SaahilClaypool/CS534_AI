# Python 3
#%%
import numpy as np
import os, sys
from PIL import Image, ImageOps
import csv
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Dropout, BatchNormalization
from keras.optimizers import  sgd
from keras.utils import to_categorical

def percent_correct(y, y_hat):
    # compute percent correct by first converting y_hat into a one-hot vector
    one_hot_y_hat = np.where(y_hat == y_hat.max(axis=0)[None, :], 1, 0)
    return np.mean(np.where(y == one_hot_y_hat, 1, 0))



def compute_predictions(x, w):
    # first compute preactivation vectors
    z = compute_preactivation(x, w)
    z_sum = np.sum(np.exp(z), axis=0)
    # normalize
    y_hat = np.exp(z) / z_sum
    return y_hat


def load_data(which):
    images = np.load('mnist/{}_images.npy'.format(which))
    labels = np.load('mnist/{}_labels.npy'.format(which))
    return images, labels

def onehot_to_name(y_hat, label_names):
    print(y_hat.shape)
    idx = np.argmax(y_hat)
    return label_names[idx]

#%%
if __name__ == "__main__":
    training_images, training_labels, = load_data("mnist_train")
    testing_images, testing_labels, = load_data("mnist_test")

    print(training_images.shape)

    print("Initializing Classifier")

    # I don't really know what i'm doing here. hurr durr more layers more things
    # classifier = load_model('saved_models/nn_mnist_1')
    classifier = Sequential()
    classifier.add(Dense(50, activation='relu',input_dim=training_images.shape[1]))
    classifier.add(Dropout(0.5))
    classifier.add(BatchNormalization())
    classifier.add(Dense(10, activation='softmax'))
    classifier.compile(optimizer=sgd(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    #for t in range(3):
    classifier.fit(x=training_images, y=training_labels, epochs=100, batch_size=256)
    classifier.save("saved_models/nn_mnist_1")#+str(t))
    print("Fit completed")

    training_pc = classifier.evaluate(x=training_images, y=training_labels, batch_size=128)
    print("Training percent correct: ", training_pc)

    testing_pc = classifier.evaluate(x=testing_images, y=testing_labels, batch_size=128)
    print("Testing percent correct: ", testing_pc)

