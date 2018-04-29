# Python 3
#%%
import numpy as np
import os, sys
from PIL import Image, ImageOps
import csv

def percent_correct(y, y_hat):
    # compute percent correct by first converting y_hat into a one-hot vector
    one_hot_y_hat = np.where(y_hat == y_hat.max(axis=0)[None, :], 1, 0)
    return np.mean(np.where(np.argmax(y, axis=0) == np.argmax(one_hot_y_hat, axis=0), 1, 0))


def cross_entropy_loss(y, y_hat):
    return -np.mean(np.sum(y * np.log(y_hat), axis=0))


def compute_preactivation(x, w):
    # compute all values of pre-activation at once
    return np.dot(w.T, x)


def compute_predictions(x, w):
    # first compute preactivation vectors
    z = compute_preactivation(x, w)
    z_sum = np.sum(np.exp(z), axis=0)
    # normalize
    y_hat = np.exp(z) / z_sum
    return y_hat


def compute_grad(x, y_hat, y):
    return np.dot(x, (y_hat - y).T) / x.shape[1]


def log_reg(data, labels):
    # hyper parameters
    learn_rate = 0.01
    minibatch_size = 500
    max_epochs = 150
    tol = 0.0002

    ce_diff = tol + 1
    epoch_count = 0

    # initialize weights
    w = 0.01 * np.random.randn(data.shape[0], labels.shape[0])
    # randomize order of images/labels
    r_perm = np.random.permutation(data.shape[1])
    data = np.take(data, r_perm, axis=1)
    labels = np.take(labels, r_perm, axis=1)
    test_data = data[:]

    y_hat = compute_predictions(data, w)
    prev_ce = cross_entropy_loss(labels, y_hat)

    while epoch_count < max_epochs and ce_diff > tol:
        # train on mini batches
        for i in range(0, data.shape[1], minibatch_size):
            y_hat = compute_predictions(data[:, i:i + minibatch_size], w)
            w = w - learn_rate * compute_grad(data[:, i:i + minibatch_size], y_hat, labels[:, i:i + minibatch_size])
        # get cross entropy loss for the overall data
        y_hat = compute_predictions(data, w)
        ce = cross_entropy_loss(labels, y_hat)
        ce_diff = np.abs(ce - prev_ce)
        prev_ce = ce
        epoch_count += 1
        pc = percent_correct(labels, y_hat)
        # print("Epoch: ", epoch_count)
        print("Cross entropy at epoch ", epoch_count, " : ", ce, " PC at epoch: ", pc)
    return w


def load_data(which):
    images = np.load('mnist/{}_images.npy'.format(which)).T
    labels = np.load('mnist/{}_labels.npy'.format(which)).T
    return images, labels

def onehot_to_name(y_hat, label_names):
    print(y_hat.shape)
    idx = np.argmax(y_hat)
    return label_names[idx]

def show_image(data, width=100, h=100):
    print("shape: " , data.shape)
    leng = data.shape[0]
    data = data.reshape((100,100,))
    img = Image.fromarray(data, "L")
    img.save('my.png')
    img.show()

#%% 
if __name__ == "__main__":
    training_images, training_labels, = load_data("mnist_train")
    testing_images, testing_labels, = load_data("mnist_test")

    print(training_images.shape, testing_images.shape)

    w = log_reg(training_images, training_labels)

    training_y_hat = compute_predictions(training_images, w)
    training_pc = percent_correct(training_labels, training_y_hat)
    training_ce = cross_entropy_loss(training_labels, training_y_hat)
    print("Training cross-entropy: ", training_ce)
    print("Training percent correct: ", training_pc)

    testing_y_hat = compute_predictions(testing_images, w)
    testing_pc = percent_correct(testing_labels, testing_y_hat)
    print("Testing percent correct: ", testing_pc)
