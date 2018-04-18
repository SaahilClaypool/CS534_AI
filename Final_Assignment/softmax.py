# Python 3
#%%
import numpy as np
import os, sys
from PIL import Image, ImageOps
import csv

def percent_correct(y, y_hat):
    # compute percent correct by first converting y_hat into a one-hot vector
    one_hot_y_hat = np.where(y_hat == y_hat.max(axis=0)[None, :], 1, 0)
    return np.mean(np.where(y == one_hot_y_hat, 1, 0))


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


def soft_max(data, labels):
    # hyper parameters
    learn_rate = 0.00001
    minibatch_size = 500
    max_epochs = 150
    tol = 0.0002

    ce_diff = tol + 1
    epoch_count = 0

    # initialize weights
    w = 0.00001 * np.random.randn(data.shape[0], labels.shape[0])
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
        #y_hat = compute_predictions(data, w)
        #ce = cross_entropy_loss(labels, y_hat)
        #ce_diff = np.abs(ce - prev_ce)
        #prev_ce = ce
        epoch_count += 1
        print("Epoch: ", epoch_count)
        #print("Cross entropy at epoch ", epoch_count, " : ", ce)
    return w


def load_data():
    """
    Load images and labels in as column vectors.
    labels are converted into one-hot encodings
    """
    folder = "../dog_data/train"

    dog_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    print("Working with {0} images".format(len(dog_files)))
    label_reader = csv.reader(open('../dog_data/labels.csv', 'r'))
    lab_dict = {}
    for row in label_reader:
        k, v = row
        lab_dict[k] = v

    images = np.empty(shape=(10000, 0))
    labels_str = []

    q = 0
    desired_size = 500
    for _file in dog_files:
        q += 1
        img = Image.open(folder + "/" + _file).convert('LA')
        old_size = img.size  # old_size[0] is in (width, height) format
        ratio = float(desired_size) / max(old_size)
        delta_w = desired_size - old_size[0]
        delta_h = desired_size - old_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        pad_img = ImageOps.expand(img, padding)
        pad_img = pad_img.resize((100, 100), Image.ANTIALIAS)

        data = np.asarray(pad_img, dtype="int32")
        images = np.append(images, np.array(data[:,:,0].flatten()).reshape((10000,1)), axis=1)

        labels_str.append(lab_dict[_file[:-4]])

    images = np.array(images[:])

    unique_labels = np.unique(np.array(labels_str))
    unique_labels = unique_labels.reshape((unique_labels.shape[0], 1))
    one_hot_labels = np.empty(shape=(unique_labels.shape[0], 0))
    for l in labels_str:
        one_hot_labels = np.append(one_hot_labels, np.where(l == unique_labels, 1, 0).reshape((unique_labels.shape[0], 1)), axis=1)
    print(images.shape, one_hot_labels.shape, unique_labels.shape)
    return images, one_hot_labels, unique_labels, dog_files

def onehot_to_name(y_hat, label_names):
    print(y_hat.shape)
    idx = np.argmax(y_hat)
    return label_names[idx]

def show_image(data, width=100, h=100):
    print("shape: " , data.shape)
    leng = data.shape[0]
    data = data.reshape((100,100,))
    img = Image.fromarray(data, "LA")
    img.save('my.png')
    img.show()

#%% 
if __name__ == "__main__":
    training_images, training_labels, label_names, dog_files = load_data()

    w = soft_max(training_images, training_labels)
    np.savetxt('weights.txt', w, delimiter=",")

    training_y_hat = compute_predictions(training_images, w)
    training_pc = percent_correct(training_labels, training_y_hat)
    training_ce = cross_entropy_loss(training_labels, training_y_hat)
    print("Training cross-entropy: ", training_ce)
    print("Training percent correct: ", training_pc)
    #
    # testing_y_hat = compute_predictions(testing_images, w)
    # testing_pc = percent_correct(testing_labels, testing_y_hat)
    # testing_ce = cross_entropy_loss(testing_labels, testing_y_hat)
    # print("Testing cross-entropy: ", testing_ce)
    # print("Testing percent correct: ", testing_pc)

#%%  Print the best doggo
# el = training_y_hat[:, 0].T
# onehot_to_name(el, label_names)
# show_image('../dog_data/train/000bec180eb18c7604dcecc8fe0dba07.jpg')
# training_images.shape
# dog_files[0]