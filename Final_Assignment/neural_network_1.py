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


def load_data(desired_size, shrink_size):
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

    images = np.empty(shape=(shrink_size**2, 0))
    labels_str = []

    q = 0
    for _file in dog_files:
        q += 1
        img = Image.open(folder + "/" + _file).convert('L')
        old_size = img.size  # old_size[0] is in (width, height) format
        ratio = float(desired_size) / max(old_size)
        delta_w = desired_size - old_size[0]
        delta_h = desired_size - old_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        pad_img = ImageOps.expand(img, padding)
        pad_img = pad_img.resize((shrink_size, shrink_size), Image.ANTIALIAS)
        #pad_img.show()
        data = np.asarray(pad_img)

        images = np.append(images, np.array(data[:,:]).reshape((shrink_size**2, 1)), axis=1)
        # recover_image = Image.fromarray(np.array(data[:,:]).reshape((10000, 1)).reshape((100, 100)), 'L')
        # recover_image.show()

        labels_str.append(lab_dict[_file[:-4]])
        print(q)

    images = np.array(images[:])

    unique_labels = np.unique(np.array(labels_str))
    #one_hot_labels = to_categorical(labels_str, num_classes=unique_labels.size)
    one_hot_labels = np.empty(shape=(unique_labels.shape[0], 0))
    for l in labels_str:
        one_hot_labels = np.append(one_hot_labels, np.where(l == unique_labels, 1, 0).reshape((unique_labels.shape[0], 1)), axis=1)
    print(images.shape, one_hot_labels.shape, unique_labels.shape)
    return images, one_hot_labels, unique_labels, dog_files

def onehot_to_name(y_hat, label_names):
    print(y_hat.shape)
    idx = np.argmax(y_hat)
    return label_names[idx]

#%%
if __name__ == "__main__":
    desired_size = 500
    shrink_size = 100
    training_images, training_labels, label_names, dog_files = load_data(desired_size, shrink_size)
    # TODO: use proper methods for cross-validation instead of just splitting data like this
    testing_images = training_images[:, -1222:]
    testing_labels = training_labels[:, -1222:]

    training_images = training_images[:, :9000]
    training_labels = training_labels[:, :9000]

    training_images = training_images.T
    training_labels = training_labels.T
    testing_images = testing_images.T
    testing_labels = testing_labels.T

    print("Initializing Classifier")


    # I don't really know what i'm doing here. hurr durr more layers more things
    classifier = load_model('saved_models/nn4_0')
    # classifier = Sequential()
    # classifier.add(Dense(30, activation='relu',input_dim=shrink_size**2))
    # classifier.add(Dropout(0.5))
    # classifier.add(BatchNormalization())
    # classifier.add(Dense(label_names.size, activation='softmax'))
    # classifier.compile(optimizer=sgd(lr=0.0002, decay=1e-6, momentum=0.9, nesterov=True),
    #                    loss='categorical_crossentropy',
    #                    metrics=['accuracy'])
    # for t in range(3):
    #    classifier.fit(x=training_images, y=training_labels, epochs=1000, batch_size=512)
    #    classifier.save("saved_models/nn4_"+str(t))
    print("Fit completed")

    training_pc = classifier.evaluate(x=training_images, y=training_labels, batch_size=128)
    print("Training percent correct: ", training_pc)

    testing_pc = classifier.evaluate(x=testing_images, y=testing_labels, batch_size=128)
    print("Testing percent correct: ", testing_pc)

