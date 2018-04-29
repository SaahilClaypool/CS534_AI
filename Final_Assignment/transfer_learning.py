# Note: much of this code is from an example on Kaggle. 
# Small modifications have been made, but this code
# has been extremely helpful in understanding transfer learning
# https://www.kaggle.com/gaborfodor/dog-breed-pretrained-keras-models-lb-0-3
# and keras documentation from 
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html%%
#%%
## Imports
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs
from os.path import join, exists, expanduser
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.linear_model import LogisticRegression
#%%  Cache things for speed
cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)

#%% Use only the top K classes
INPUT_SIZE = 224
NUM_CLASSES = 32
SEED = 1987
data_dir = '../dog_data/'
# This loads the labels into a table object
labels = pd.read_csv(join(data_dir, 'labels.csv'))

#%% Select top K breeds
# Pick the top K 
selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)
labels = labels[labels['breed'].isin(selected_breed_list)]
labels['target'] = 1
# Order by number of images in that breed
labels['rank'] = labels.groupby('breed').rank()['id']
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
np.random.seed(seed=SEED)
rnd = np.random.random(len(labels))
# Make training 80%, validation 80%
# Could turn into K fold validation
train_idx = rnd < 0.8
valid_idx = rnd >= 0.8
y_train = labels_pivot[selected_breed_list].values
ytr = y_train[train_idx]
yv = y_train[valid_idx]

def read_img(img_id, train_or_test, size):
    """Read and resize image.
    # Arguments
        img_id: string
        train_or_test: string 'train' or 'test'.
        size: resize the original image.
    # Returns
        Image as numpy array.
    """
    img = image.load_img(join(data_dir, train_or_test, '%s.jpg' % img_id), target_size=size)
    img = image.img_to_array(img)
    return img

#%% Xception feature

INPUT_SIZE = 299
POOLING = 'avg'
x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

# Load images and shape to fit xception, which takes 299 by 299 channel last images
for i, img_id in tqdm(enumerate(labels['id'])):
    img = read_img(img_id, 'train', (INPUT_SIZE, INPUT_SIZE))
    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    x_train[i] = x
print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

# Run through Xception to generates feature outputs
Xtr = x_train[train_idx]
Xv = x_train[valid_idx]
print((Xtr.shape, Xv.shape, ytr.shape, yv.shape))
xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)
train_x_bf = xception_bottleneck.predict(Xtr, batch_size=32, verbose=1)
valid_x_bf = xception_bottleneck.predict(Xv, batch_size=32, verbose=1)
print('Xception train bottleneck features shape: {} size: {:,}'.format(train_x_bf.shape, train_x_bf.size))
print('Xception valid bottleneck features shape: {} size: {:,}'.format(valid_x_bf.shape, valid_x_bf.size))

#%% Run logistic regression on the FEATURES that Xception outputs

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)
logreg.fit(train_x_bf, (ytr * range(NUM_CLASSES)).sum(axis=1))
valid_probs = logreg.predict_proba(valid_x_bf)
valid_preds = logreg.predict(valid_x_bf)
print('Validation Xception LogLoss {}'.format(log_loss(yv, valid_probs)))
print('Validation Xception Accuracy {}'.format(accuracy_score((yv * range(NUM_CLASSES)).sum(axis=1), valid_preds)))

#%% Visual confirmation of performance

def name_from_number(num, labels):
    return labels[selected_breed_list].axes[1][num]

def id_from_number(num, labels):
    return labels['id'][valid_idx].iloc[num]


def plot_idx(idx, g_model, labels): 
    predic = int(g_model.predict(valid_x_bf[idx:idx+1])[0])
    predic_label = name_from_number(predic, labels)
    
    fig = plt.figure(1, figsize=(16, 16))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.05)
    ax = grid[0]
    im_id = id_from_number(idx, labels)
    img = read_img(im_id, 'train', (224, 224))
    ax.imshow(img / 255.)
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    ax.text(10, 180, 'PREDICTION %s ' % predic_label, color='w', backgroundcolor='k', alpha=0.8, fontsize=20)
    breed_name = labels_pivot[labels_pivot['id'] == im_id][selected_breed_list].idxmax(axis=1)
    ax.text(10, 200, 'LABEL: %s' % breed_name, color='k', backgroundcolor='w', alpha=0.8, fontsize=20)
    ax.axis('off')
    plt.savefig('img/validation_{}.png'.format(idx))
    # plt.show()


# Plot first 30 
for i in range(30):
    plot_idx(i, logreg, labels_pivot)

valid_breeds = (yv * range(NUM_CLASSES)).sum(axis=1)
error_idx = (valid_breeds != valid_preds)

# Plot the errors
for img_id, breed, pred in zip(labels.loc[valid_idx, 'id'].values[error_idx],
                                [selected_breed_list[int(b)] for b in valid_preds[error_idx]],
                                [selected_breed_list[int(b)] for b in valid_breeds[error_idx]]):
    fig, ax = plt.subplots(figsize=(5,5))
    img = read_img(img_id, 'train', (299, 299))
    ax.imshow(img / 255.)
    ax.text(10, 250, 'Prediction: %s, Confidence" %s' % pred, color='w', backgroundcolor='r', alpha=0.8, fontsize=20)
    ax.text(10, 270, 'LABEL: %s' % breed, color='k', backgroundcolor='g', alpha=0.8, fontsize=20)
    ax.axis('off')
    plt.savefig('img/error_{}.png'.format(img_id))
    # plt.show()        

#%% K Fold
import numpy as np
A = np.arange(81).reshape(9,9)
B = np.arange(27).reshape(9,3)
def test_folds(x_train, y_train, k=5):
    folds = np.split(x_train, k)
    fold_labels = np.split(y_train, k)
    
    for i in range(k): 
        print("Fold {} of {}".format(i,k))
        validation = folds[i]
        v_labels = fold_labels[i]
        train = np.array(folds[:i] + folds[i:])
        t_labels = np.array(fold_labels[:i] + folds[i:])
        logloss, acc = run_fold(train, validation, t_labels, v_labels)
        print("\n")

def run_fold(Xtr, Xv, ytr, yv):
    POOLING = 'avg'
    print((Xtr.shape, Xv.shape, ytr.shape, yv.shape))
    xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)
    train_x_bf = xception_bottleneck.predict(Xtr, batch_size=32, verbose=1)
    valid_x_bf = xception_bottleneck.predict(Xv, batch_size=32, verbose=1)
    print('Xception train bottleneck features shape: {} size: {:,}'.format(train_x_bf.shape, train_x_bf.size))
    print('Xception valid bottleneck features shape: {} size: {:,}'.format(valid_x_bf.shape, valid_x_bf.size))

    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)
    logreg.fit(train_x_bf, (ytr * range(NUM_CLASSES)).sum(axis=1))
    valid_probs = logreg.predict_proba(valid_x_bf)
    valid_preds = logreg.predict(valid_x_bf)
    logloss = log_loss(yv, valid_probs)
    acc = accuracy_score((yv * range(NUM_CLASSES)).sum(axis=1), valid_preds)
    print('Validation Xception LogLoss {}'.format(logloss))
    print('Validation Xception Accuracy {}'.format(acc))
    return logloss, acc

test_folds(x_train, y_train)