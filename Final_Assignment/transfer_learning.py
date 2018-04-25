# Python 3
#%%
import numpy as np
import os, sys
from PIL import Image, ImageOps
import csv
import keras
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Activation, LSTM, Dropout, BatchNormalization, Input
from keras.optimizers import  sgd
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import metrics
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3


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
        if q > 1000:
           break

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
#%% loading cell
desired_size = 500
shrink_size = 50
# Images are now 1 x 50 x 50 so each row is an image
training_images, training_labels, label_names, dog_files = load_data(desired_size, shrink_size)
#%%

print("Initializing Classifier")
img_rows, img_cols = 50, 50
input_shape = (img_rows, img_cols, 1)
input_tensor = Input(shape=input_shape)  # this assumes K.image_data_format() == 'channels_last'
droprate = 0.1
num_classes = label_names.shape[0]
filter_pixel = 3 # I believe this is the sub matrix that we are using to identify features. 


# Pre trained model
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=True, input_shape=input_shape)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D(data_format='channels_last')(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# compile
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


model.summary()

#%% 
history = model.fit(training_images, training_labels.T,
        batch_size=128,
        epochs=30,
        verbose=1,
        validation_data=(training_images, training_labels.T),shuffle=True,
        callbacks=callbacks)
print("fit done")

training_pc = model.evaluate(x=training_images, y=training_labels.T, batch_size=32)
print("Training percent correct: ", training_pc[1])

# testing_pc = classifier.evaluate(x=testing_images, y=testing_labels, batch_size=128)
# print("Testing percent correct: ", testing_pc)