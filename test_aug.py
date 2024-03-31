

from math import ceil
from tensorflow.python.keras.callbacks import ModelCheckpoint
#from keras.callbacks import ModelCheckpoint, Callback
from smistad.smistad_imgaug import UltrasoundImageGenerator
from smistad.smistad_dataset import get_dataset_files
from smistad.smistad_network import Unet # :(
from smistad.smistad_imgaug import *
import os
import sys
import numpy as np
from sys import exit
from tensorflow.python.keras.optimizers import RMSprop, Adam, Adadelta
from keras.losses import categorical_crossentropy
from skimage.measure import regionprops, label
from numpy.random import shuffle
from batch_generator_tumor import *


# use single GPU (first one)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

### set name for model and history
name = '22_06_256_32_rgb'

# paths
data_path = '/hdd/NUCLEI/datasets/binary_nuclei_seg_512_32_rgb_22_06/'
save_model_path = '/hdd/NUCLEI/output/models/'
history_path = '/hdd/NUCLEI/output/history/'
datasets_path = '/hdd/NUCLEI/output/datasets/'

# assign WSIs randomly to train, val and test
images = os.listdir(data_path)
images = [data_path + i for i in images]
shuffle(images)

# only two split, 20 % in test, rest training
test_set = images[:7] # first 10 in test
val_set = images[:7] # next 10 in val
train_set = images[7:] # rest in training
# -> two-split, 10 in test, rest in training

# get the actual patches
sets = []
for t in test_set:
    for p in os.listdir(t):
        sets.append(t + '/' + p)
test_set = sets.copy()
val_set = test_set.copy()

sets = []
for t in train_set:
    print(t)
    for p in os.listdir(t):
        sets.append(t + '/' + p)
train_set = sets.copy()


if name.split('_')[3] == 'gray':
    channels = 1
else:
    channels = 3

window = int(name.split('_')[2])


# define model
network = Unet(input_shape=(window, window, channels), nb_classes=2)
network.encoder_spatial_dropout = 0.1 # 0.2
network.decoder_spatial_dropout = 0.1
#network.set_convolutions([4, 8, 16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16, 8, 4]) # <- RGB 512x512
network.set_convolutions([8, 16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16, 8]) # <- if 256x256
model = network.create()

# load model <- if want to fine-tune, or train further on some previously trained model
#model.load_weights('/home/andre/Documents/Project/Andrep/lungmask/output/model_3d_11_01.h5', by_name=True)

model.compile(
    #optimizer = Adam(1e-3), # 1e-2 best?
    optimizer = 'adadelta',
    #loss='binary_crossentropy'
    loss=network.get_dice_loss()
)

# augmentation
epochs = 1000

aug = {}
aug = {'flip':1, 'rotate_ll':1, 'mult':[0.8,1.2], 'hsv':30, 'scale':[1,2], 'gauss':[0, 0, 30]}
#aug = {'gauss':[0, 0, 30]}
aug = {'jpeg':[10, 50, 1]}
aug = {'flip':1, 'rotate_ll':1, 'mult':[0.7,1.3], 'hsv':30, 'scale':[1,2], 'gauss':[0, 0, 20], 'jpeg':[10, 50, 1]}


batch_size = 1
epochs = 4 #300
imgs = 5

shuffle(train_set)

train_orig = train_set.copy()

while True:

    shuffle(train_orig)

    train_set = train_orig[:imgs]
    print(train_set)

    origs = train_set.copy()

    augs_list = []

    for curr_orig in train_set:

        train_set = [curr_orig]

        # generate sample
        for curr in train_set:

            print(curr)
            file = h5py.File(curr, 'r')
            data = np.array(file['data'], dtype=np.float32)
            gt = np.array(file['label'], dtype=np.uint8)
            file.close()

            orig = data[0,...]

            # define generators for sampling of data
            train_gen = batch_gen2(train_set, batch_size=batch_size, aug=aug, epochs=epochs)

            cnt = 0
            tmps = []

            print('----')
            augs = []

            for im, gt in train_gen:
                print(11)

                tmp = im[0]
                #tmp = gt[0,...,0]
                augs.append(tmp.copy())

                print(im.shape)

                cnt += 1

            augs_list.append(augs)



    fig, ax = plt.subplots(imgs, epochs+1, figsize=(10, 10))
    plt.tight_layout()

    print(len(origs))
    print()
    print(origs)
    print()
    print(len(augs_list))

    for i in range(len(origs)):
        f = h5py.File(origs[i], 'r')
        data = np.array(f['data'])
        gt = np.array(f['label'])
        f.close()

        orig = data[0, ...]
        print(data.shape)
        ax[i,0].imshow(orig)
        ax[i,0].set_title(origs[i].split('/')[-1].split('.hd5')[0])

        print(i)
        print('-')

    print(len(augs_list))
    print(len(augs_list[0]))


    for i in range(len(augs_list)):
        for j in range(len(augs_list[0])):
            ax[i, j+1].imshow(augs_list[i][j])
            print(i, j)

    for i in range(imgs):
        for j in range(epochs+1):
            #ax[i,j].axis('off')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])


    plt.show()