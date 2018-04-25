# Python 3
#%%
import numpy as np
import os, sys
from PIL import Image, ImageOps
import csv
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, LSTM, Dropout, BatchNormalization
from keras.optimizers import  sgd
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import metrics

def percent_correct(y, y_hat):
    # compute percent correct by first converting y_hat into a one-hot vector
    one_hot_y_hat = np.where(y_hat == y_hat.max(axis=0)[None, :], 1, 0)
    return np.mean(np.where(y == one_hot_y_hat, 1, 0))

# define path to save model


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

    images = np.empty(shape=(0, 1, shrink_size, shrink_size))
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
        # images = np.append(images, np.array(data[:,:])) # Leave images as 50 x 50
        im = np.array(data[:,:]).reshape((1, 1, shrink_size, shrink_size))
        images = np.append(images, im, axis=0)
        # recover_image = Image.fromarray(np.array(data[:,:]).reshape((10000, 1)).reshape((100, 100)), 'L')
        # recover_image.show()

        labels_str.append(lab_dict[_file[:-4]])
        print(q)
        # if q > 1000:
        #    break

    print(images.shape)
    images = np.array(images[:])

    unique_labels = np.unique(np.array(labels_str))
    #one_hot_labels = to_categorical(labels_str, num_classes=unique_labels.size)
    one_hot_labels = np.empty(shape=(unique_labels.shape[0], 0))
    for l in labels_str:
        one_hot_labels = np.append(one_hot_labels, np.where(l == unique_labels, 1, 0).reshape((unique_labels.shape[0], 1)), axis=1)
    print(images.shape, one_hot_labels.shape, unique_labels.shape)
    return images, one_hot_labels, unique_labels, dog_files

def onehot_to_name(y_hat, label_names):
    idx = np.argmax(y_hat)
    return label_names[idx]

model_path = './fm_cnn_BN.h5'
callbacks = [
    EarlyStopping(
        monitor='val_acc', 
        patience=10,
        mode='max',
        verbose=1),
    ModelCheckpoint(model_path,
        monitor='val_acc', 
        save_best_only=True, 
        mode='max',
        verbose=0)
]
#%%
if __name__ == "__main__":
    desired_size = 500
    shrink_size = 50
    # Images are now 1 x 50 x 50 so each row is an image
    training_images, training_labels, label_names, dog_files = load_data(desired_size, shrink_size)
    # TODO: use proper methods for cross-validation instead of just splitting data like this
    # testing_images = training_images[:, -1222:]
    # testing_labels = training_labels[:, -1222:]

    # training_images = training_images[:, :9000]
    # training_labels = training_labels[:, :9000]

    # training_images = training_images.T
    # training_labels = training_labels.T
    # testing_images = testing_images.T
    # testing_labels = testing_labels.T

    print("Initializing Classifier")
    img_rows, img_cols = 50, 50
    input_shape = (1, img_rows, img_cols)
    droprate = 0.1
    num_classes = label_names.shape[0]
    filter_pixel = 3 # I believe this is the sub matrix that we are using to identify features. 

    # I don't really know what i'm doing here. hurr durr more layers more things
    classifier = Sequential()
    classifier.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel),
        padding="same", activation="relu", 
        input_shape=input_shape))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(droprate))

    # layer 2
    classifier.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel), activation="relu", border_mode="same"))
    classifier.add(BatchNormalization())
    # classifier.add(MaxPooling2D()) # causes error?
    classifier.add(Dropout(droprate))#3

    # fully connected layer
    classifier.add(Flatten())
    classifier.add(Dense(500, use_bias=False))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(droprate))

    # Fully connected final layer
    classifier.add(Dense(num_classes))

    classifier.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.RMSprop(),
                metrics=[metrics.categorical_accuracy])

    classifier.summary()
    history = classifier.fit(training_images, training_labels.T,
            batch_size=128,
            epochs=100,
            verbose=1,
            validation_data=(training_images, training_labels.T),shuffle=True,
            callbacks=callbacks)
    print("fit done")

    # classifier.add(Dense(1024, activation='relu', input_dim=shrink_size**2))
    # classifier.add(BatchNormalization())
    # classifier.add(Dropout(0.1))
    # classifier.add(Dense(512, activation='relu'))
    # classifier.add(BatchNormalization())
    # classifier.add(Dropout(0.1))
    # classifier.add(Dense(512, activation='relu'))
    # classifier.add(BatchNormalization())
    # classifier.add(Dropout(0.1))
    # classifier.add(Dense(256, activation='relu'))
    # classifier.add(BatchNormalization())
    # classifier.add(Dropout(0.1))
    # classifier.add(Dense(label_names.size, activation='softmax'))
    # classifier.compile(optimizer=sgd(lr=0.00002, decay=1e-6, momentum=0.9, nesterov=True),
    #                    loss='categorical_crossentropy',
    #                    metrics=['accuracy'])
    # classifier.fit(x=training_images, y=training_labels, epochs=5000, batch_size=512)
    # print("Fit completed")

    training_pc = classifier.evaluate(x=training_images, y=training_labels.T, batch_size=128)
    print("Training percent correct: ", training_pc[1])

    # testing_pc = classifier.evaluate(x=testing_images, y=testing_labels, batch_size=128)
    # print("Testing percent correct: ", testing_pc)