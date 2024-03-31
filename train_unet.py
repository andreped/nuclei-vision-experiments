from math import ceil
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
#from keras.callbacks import ModelCheckpoint, Callback
from smistad.smistad_imgaug import UltrasoundImageGenerator
from smistad.smistad_dataset import get_dataset_files
from smistad.smistad_network import Unet # :(
from smistad.smistad_imgaug import *
import os
import sys
import numpy as np
from tensorflow.python.keras.optimizers import RMSprop, Adam, Adadelta
from keras.losses import categorical_crossentropy
from skimage.measure import regionprops, label
from numpy.random import shuffle
from batch_generator_tumor_random import *
import seaborn as sns
import pandas as pd


def dynamic_updownsample():
    new = []
    for uniq in uniques:
        curr = images[set_classes == uniq]
        np.random.shuffle(curr)
        if len(curr) >= max_limit:
            new.append(curr[:max_limit])
        else:
            factor = max_limit / len(curr)
            new.append(np.tile(curr, int(np.ceil(factor)))[:max_limit])

    print([len(x) for x in new])


server = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = server
print('Server: ' + server)

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# aug = {}, bs = 8

# RGB
name = '160420_multiclass_nuclei_seg_256_32_rgb_nbclasses_6_bs_96_all_augs'

tmp = name.split('_')[6]
if tmp == "rgb":
    nb_channels = 3
elif tmp == "gray":
    nb_channels = 1
else:
    raise Exception("please choose either {rgb, gray}")

window = int(name.split('_')[4])
nb_classes = int(name.split("_")[8])
input_shape = (window, window, nb_channels)
curr_datasets = "080420_binary_nuclei_seg_256_32_rgb"
N_train = 10000
N_val = 2000

# paths
data_path = '/home/andrep/workspace/nuclei/datasets/' + curr_datasets + '/'
save_model_path = '/home/andrep/workspace/nuclei/output/models/'
history_path = '/home/andrep/workspace/nuclei/output/history/'
datasets_path = '/home/andrep/workspace/nuclei/output/datasets/'

# assign WSIs randomly to train, val and test
images = os.listdir(data_path)
images = np.array([data_path + i for i in images])
shuffle(images)

set_classes = np.array([x.split("/")[-1].split(".")[0].split("_")[-1] for x in images])
uniques = np.unique(set_classes)
print(uniques)
print(len(uniques))

#sns.catplot(x="tissue", kind="count", palette="ch:.25", data=images)

#plt.hist(set_classes, bins=len(uniques))
#plt.show()

# undersample the two largest groups (Breast, Colon) to the third largest, and upsample all the smaller ones to the 3rd largest one
tmps = pd.factorize(set_classes)
print(np.histogram(tmps[0], bins=len(uniques)))
curr_hist = np.histogram(tmps[0], bins=len(uniques))
max_limit = np.sort(curr_hist[0])[-3]
print(max_limit)

train_dir = []
val_dir = []
test_dir = []
val1 = 0.8
val2 = 0.9
for uniq in uniques:
    #if uniq not in ['Breast']:  # , 'Colon']:
    #    continue
    tmp = images[set_classes == uniq]
    np.random.shuffle(tmp)
    N = len(tmp)
    cval1 = int(N*val1)
    cval2 = int(N*val2)
    train_dir.append(tmp[:cval1])
    val_dir.append(tmp[cval1:cval2])
    test_dir.append(tmp[cval2:])

# 
# <- As original images have the same size, the # of patches for each GT is equal, thus already balanced
# 


# make larger val_set:
#val_set = test_set + val_set # <- val_set = test_set -> cross validation

# save random generated data sets
sets = ["train", "val", "test"]
dirs = [train_dir, val_dir, test_dir]

f = h5py.File((datasets_path + 'dataset_' + name + '.h5'), 'w')
for i in range(len(sets)):
    curr_set = sets[i]
    curr_dir = dirs[i].copy()
    for j, tmp in enumerate(curr_dir):
        f.create_dataset(curr_set + "/" + str(uniques[j]), data=np.array(tmp).astype('S400'), compression="gzip", compression_opts=4)
f.close()

# define model
network = Unet(input_shape=input_shape, nb_classes=nb_classes)
network.encoder_spatial_dropout = 0.1  # 0.2
network.decoder_spatial_dropout = 0.1
if window == 512:
    network.set_convolutions([4, 8, 16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16, 8, 4])  # <- RGB 512x512
elif window == 256:
    network.set_convolutions([8, 16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16, 8])  # <- if 256x256
else:
    raise ValueError('Unsupported window size chosen, please choose 256 or 512')
model = network.create()
print(model.summary())

# load model <- if want to fine-tune, or train further on some previously trained model
#model.load_weights('/home/andre/Documents/Project/Andrep/lungmask/output/model_3d_11_01.h5', by_name=True)

model.compile(
    optimizer=Adam(1e-3),  # 1e-2 best?
    loss=network.get_dice_loss()
)

# augmentation
batch_size = 96  # 8
epochs = 1000

#aug = {}
train_aug = {'flip':1, 'rotate_ll':1, 'mult':[0.9,1.1], 'hsv':10, 'scale':[0.7,1.5], 'gauss':[0, 0, 20], 'jpeg':[10, 50, 1]}
#train_aug = {'flip':1, 'rotate_ll':1, 'mult':[0.9,1.1], 'hsv':10, 'gauss':[0, 0, 20], 'jpeg':[10, 50, 1]}
val_aug = {'flip':1, 'rotate_ll':1}
#train_aug = {}
#val_aug = {}

# define generators for sampling of data
train_gen = batch_gen2(train_dir, batch_size=batch_size, aug=train_aug, epochs=epochs, input_shape=(256, 256, 3),
                       nb_classes=6, N_samples=N_train)
val_gen = batch_gen2(val_dir, batch_size=batch_size, aug=val_aug, epochs=epochs, input_shape=(256, 256, 3),
                     nb_classes=6, N_samples=N_val)

save_best = ModelCheckpoint(
    save_model_path + 'model_' + name + '.h5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.losses.append(['loss', 'val_loss'])

    def on_epoch_end(self, batch, logs={}):
        self.losses.append([logs.get('loss'), logs.get('val_loss')])
        # save history:
        f = h5py.File((history_path + '/history_' + name + '.h5'), 'w')
        f.create_dataset("history", data=np.array(self.losses).astype('|S9'), compression="gzip", compression_opts=4)
        f.close()

history_log = LossHistory()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

history = model.fit_generator(
        train_gen,
        steps_per_epoch=int(ceil(N_train/batch_size)),
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=int(ceil(N_val/batch_size)),
        callbacks=[save_best, reduce_lr, history_log]
)