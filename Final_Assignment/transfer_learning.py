# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#%%
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
import os, sys
import csv
import numpy as np


im_size = 256
def load_labels():
    label_reader = csv.reader(open('../dog_data/labels.csv', 'r'))
    lab_dict = {}
    for row in label_reader:
        k, v = row
        lab_dict[k] = v
    folder = "../dog_data/train"
    dog_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    print("Working with {0} images".format(len(dog_files)))
    labels_str = []
    f = -1
    X = -1
    q = 0
    for _file in dog_files:
        q += 1
        labels_str.append(lab_dict[_file[:-4]])
        img = load_img('../dog_data/train/000bec180eb18c7604dcecc8fe0dba07.jpg', target_size=(im_size, im_size))  # this is a PIL image
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        if (f == -1):
            f = 1
            X = np.empty(shape=(x.shape))
        if (q == 1000):
            break
        X = np.append(X, x, axis=0)
    unique_labels = np.unique(np.array(labels_str))
    #one_hot_labels = to_categorical(labels_str, num_classes=unique_labels.size)
    one_hot_labels = np.empty(shape=(0,unique_labels.shape[0]))
    print(one_hot_labels.shape)
    for l in labels_str:
        one_hot_labels = np.append(one_hot_labels, 
            np.where(l == unique_labels, 1, 0).reshape((1, unique_labels.shape[0])), 
            axis=0)
    return X, one_hot_labels
x, lab = load_labels()
print(lab.shape)

datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x)

#%% 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(im_size, im_size, 3)))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))

model.add(Conv2D(32, (3, 3), data_format="channels_last"))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))

model.add(Conv2D(64, (3, 3), data_format="channels_last"))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(lab.shape[1]))
model.add(Activation('sigmoid'))

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr=0.001, clipnorm=1.)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['categorical_accuracy'])
model.summary()

#%%  FIT
batch_size = 20
model.fit_generator(
        datagen.flow(x, lab, batch_size=batch_size),
        steps_per_epoch=len(x) / batch_size, # batch_size,
        epochs=20
        ) # batch size
model.save_weights('first_try.h5')  # always save your weights after training or during training

# %%
print(x.shape)
print(lab)
for i in range(len(x)):
    p = np.argmax(model.predict(x[i:i+1]))
    a = np.argmax(lab[i:i+1])
    print("i: {} p: {} a: {}".format(i, p, a))