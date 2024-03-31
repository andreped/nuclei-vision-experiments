from math import ceil
from keras.callbacks import ModelCheckpoint
#from keras.callbacks import ModelCheckpoint, Callback
#from smistad.smistad_imgaug import *
import os
from keras.optimizers import Adam
from numpy.random import shuffle
from batch_generator_yolo3 import *
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
#name = '230420_binary_nuclei_normal_yolov3_256_32_rgb_nbclasses_1_weights_only_max_80_overfit_bs_32_lr_1e-3_new_anchors_freeze_1_pretrained_breast_only_finetune_new_augs'
name = '230420_binary_nuclei_normal_yolov3_256_32_rgb_nbclasses_1_weights_only_max_80_overfit_bs_32_lr_1e-3_new_anchors_breast_only_rigid_augs'
name = '230420_binary_nuclei_normal_yolov3_256_32_rgb_nbclasses_1_weights_only_max_80_overfit_bs_48_lr_1e-3_new_anchors_rigid_augs_finetune'
name = '230420_binary_nuclei_tiny_yolov3_256_32_rgb_nbclasses_1_weights_only_max_80_overfit_bs_48_lr_1e-3_new_anchors_rigid_augs_finetune' # best single-class detector
name = '290520_binary_nuclei_tiny_yolov3_256_32_rgb_nbclasses_6_weights_only_max_80_overfit_bs_48_lr_1e-3_new_anchors_rigid_augs_finetune' # best multi-class detector ?

print(name)

tmp = name.split('_')[7]
if tmp == "rgb":
    nb_channels = 3
elif tmp == "gray":
    nb_channels = 1
else:
    raise Exception("please choose either {rgb, gray}")

window = int(name.split('_')[5])
nb_classes = int(name.split("_")[9])
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

#print(test_dir)
#exit()


if len(train_dir) == 1:
    N_train = len(train_dir[0])
    N_val = len(val_dir[0])

#
# <- As original images have the same size, the # of patches for each GT is equal, thus already balanced
#

# only keep a few


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


# define tiny yolo3 model
from yolo_v3.Utils.Train_Utils import get_anchors, create_tiny_model, create_model

tiny_yolo3_anchors_path = "./yolo_v3/keras_yolo3/model_data/tiny_yolo_anchors.txt"
yolo3_anchors_path = "./yolo_v3/keras_yolo3/model_data/yolo_anchors.txt"
new_yolo3_anchors_path = "./yolo_anchors_pannuke.txt"
new_tiny_yolo3_anchors_path = "./tiny_yolo_anchors_pannuke.txt"

anchors = get_anchors(new_tiny_yolo3_anchors_path)
#anchors = get_anchors(new_yolo3_anchors_path)
#anchors = get_anchors(yolo3_anchors_path)
#num_classes = 1  # only nuclei set to 1, else 6 multiclass

# pretrained weights
weights_path = "/home/andrep/workspace/nuclei/python/yolo/TrainYourOwnYOLO/Data/Model_Weights/trained_weights_final.h5"
#weights_path = "/home/andrep/workspace/nuclei/python/yolo3_weights/yolov3.weights"

# scale anchors to 256x256 input (originally 416x416)
#print(anchors)
#anchors = np.round(anchors * 256/416).astype(int)
#print(anchors)
#exit()


# pre-train to get stable loss
model = create_tiny_model(input_shape[:2], anchors, nb_classes, load_pretrained=False,
                          freeze_body=None, weights_path=weights_path)  # freeze_body=2
#model = create_model(input_shape[:-1], anchors, nb_classes, load_pretrained=True,
#                    freeze_body=2, weights_path=weights_path)  # freeze_body=2

model.compile(
    optimizer=Adam(1e-3),  # 1e-2 best?
    loss={
        "yolo_loss": lambda y_true, y_pred: y_pred
    },
)

# augmentation
batch_size = 48  # 8
#epochs = 1000
epochs = 30

#aug = {}
train_aug = {'mult':[0.8,1.2], 'hsv':10, 'gauss':[0, 0, 20], 'jpeg':[10, 50, 1]}
#train_aug = {'flip':1, 'rotate_ll':1, 'mult':[0.8,1.2], 'hsv':10, 'scale':[0.7,1.5], 'gauss':[0, 0, 20], 'jpeg':[10, 50, 1]}
#val_aug = {'flip':1, 'rotate_ll':1}
train_aug = {}
val_aug = {}

# define generators for sampling of data
train_gen = batch_gen2(train_dir, batch_size=batch_size, aug=train_aug, epochs=epochs,
                       input_shape=(256, 256, 3), nb_classes=nb_classes, N_samples=N_train,
                       anchors=anchors)
val_gen = batch_gen2(val_dir, batch_size=batch_size, aug=val_aug, epochs=epochs,
                     input_shape=(256, 256, 3), nb_classes=nb_classes, N_samples=N_val,
                     anchors=anchors)

save_best = ModelCheckpoint(
    save_model_path + 'model_' + name + '.h5',
    monitor='loss',  # <- TODO: CHANGED THIS TEMPORARILY, FROM VAL_LOSS -> LOSS (!)
    verbose=0,
    save_best_only=True,
    save_weights_only=True,  # TODO: Apparently, need to save weights and initialize during usage and load weights
    mode='auto',
    period=1
)

history = model.fit_generator(
        train_gen,
        steps_per_epoch=int(ceil(N_train/batch_size)),
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=int(ceil(N_val/batch_size)),
        callbacks=[save_best]
)


### then unfreeze all layers and fine-tune
print("\nFinetuning...\n")
if True:
    # unfreeze
    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    # recompile to apply the change
    model.compile(
        optimizer=Adam(lr=1e-4),  # <- now use lower learning rate, for fine-tuning
        loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        }
    )

    # start training
    history = model.fit_generator(
            train_gen,
            steps_per_epoch=int(ceil(N_train/batch_size)),
            epochs=1000,
            validation_data=val_gen,
            validation_steps=int(ceil(N_val/batch_size)),
            callbacks=[save_best]
    )

print("Finished!")
