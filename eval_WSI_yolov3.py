import h5py
import numpy as np
import cv2
from tensorflow.python.keras.models import load_model
# from keras.models import load_model
from numpy.random import shuffle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numba as nb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
    precision_recall_fscore_support
from tensorflow.keras.layers import Input
import matplotlib.patches as patches
from keras import backend as K
import openslide as ops


# import tensorflow as tf
# tf.enable_eager_execution()

@nb.jit(nopython=True)
def sc_any(array):
    for x in array.flat:
        if x:
            return True
    return False


def maxminscale(tmp):
    if sc_any(tmp):
        tmp = tmp - np.amin(tmp)
        tmp = tmp / np.amax(tmp)
    return tmp


def import_set(tmp, num=None, filter=False, shuffle_=False):
    f = h5py.File(datasets_path + 'dataset_' + name + '.h5', 'r')
    tmp = np.array(f[tmp])
    tmp = [tmp[i].decode("UTF-8") for i in range(len(tmp))]
    if shuffle_:
        shuffle(tmp)
    if filter:
        tmp = remove_copies(tmp)
    if num != None:
        tmp = tmp[:num]
    f.close()
    return tmp


def unique_patients(tmp):
    l = []
    for t in tmp:
        l.append(int(t.split('/')[-1].split('_')[0]))
    return np.unique(l)


def unique_classes(tmp):
    l = []
    for t in tmp:
        l.append(int(get_grade(t.split('/')[-2], grades)))  # .split('_')[-1]))
    return len(np.unique(l)), np.unique(l)


def get_grade(path, labels):
    tmp = int(path.split('_')[2])
    out = labels[labels[:, 0] == tmp, 1][0]
    return out


def remove_copies(tmp):
    return np.unique(tmp).tolist()


def _get_anchors_simple(anchor_path):
    with open(anchor_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)



def func(path):

    print(path)

    # image-specific params
    img_res = 40  # <- actual full image resolution/magnification
    image_plane_factor = 4  # <- actual downsampling factor between image planes

    # user-specific settings # 'tumor_classify_images_512_1.25x'
    image_plane = 0
    window_size = 256
    stride = 256

    # initialize reader
    reader = ops.OpenSlide(path)
    M, N = reader.dimensions

    # tissue detection
    image_plane_full = 2
    img = np.array(reader.read_region((0, 0), image_plane_full, reader.level_dimensions[image_plane_full]))[..., :3]
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)[..., 1]
    img = cv2.medianBlur(img, 7)
    _, tissue_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # calculate new ds
    ds = image_plane_factor ** image_plane_full

    #print(':)')

    del reader
    reader = ops.OpenSlide(path)

    print(M, N)

    # clear uneccessary stuff
    #del img #, grades

    # downsample factor for each axis
    image_plane_new = image_plane_factor ** image_plane

    #reader.close()

    # patchgen
    for x in range(int(np.ceil(M / stride / image_plane_new))):
        for y in range(int(np.ceil(N / stride / image_plane_new))):

            X = int(np.round(x * stride * image_plane_new))
            Y = int(np.round(y * stride * image_plane_new))

            Xg = int(np.round(x * stride * image_plane_new / ds))
            Yg = int(np.round(y * stride * image_plane_new / ds))

            new_size = int(np.round(window_size * image_plane_new / ds))

            # get current tissue
            tissue_curr = tissue_mask[Yg:(Yg + new_size), \
                          Xg:(Xg + new_size)]

            # count nonzero elements -> number of non-redundant tumor pixels
            num_tumor_pixels = np.count_nonzero(tissue_curr)
            tot = new_size ** 2
            fraction_tissue = num_tumor_pixels / tot

            print(fraction_tissue)

            # only accept tile if >= 25% tissue
            if fraction_tissue >= 0.75:

                # <--------------- PROBLEM HERE ???? WHY ???? (ITS NOT THE PROBLEM!!)
                # Read tile from .tif
                data = np.array(reader.read_region((X, Y), image_plane, (window_size, window_size)))[:, :, :3]
                data[data == (0, 0, 0)] = 255

                data = data.astype(np.float32) / 255.
                prediction = model.predict(np.expand_dims(data, axis=0))

                print(prediction[0].shape)
                # exit()

                iou = 0.0001  # 0.1  # 0.45
                score = 0.01  # 0.3
                # input_image_shape = (256, 256)
                input_image_shape = K.placeholder(shape=(2,))
                nb_classes = num_classes

                boxes, scores, classes = yolo_eval(model.output, anchors,
                                                   nb_classes, input_image_shape,
                                                   score_threshold=score, iou_threshold=iou,
                                                   max_boxes=200
                                                   )

                image_data = np.expand_dims(data, axis=0)

                sess = K.get_session()
                out_boxes, out_scores, out_classes = sess.run(
                    [boxes, scores, classes],
                    feed_dict={
                        model.input: image_data,
                        input_image_shape: [256, 256],
                        K.learning_phase(): 0
                    })

                print(out_boxes)
                print(out_scores)
                print(out_classes)

                pred_boxes = out_boxes.copy()

                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax.imshow(data)

                print(out_boxes.shape)

                cutoff = -1

                # order = np.argsort(out_scores)[::-1]
                # pred_boxes = out_boxes[order[:cutoff]]
                # print(out_boxes)
                # print(out_scores[order[:cutoff]])
                print("---")

                for i in range(pred_boxes.shape[0]):
                    l = pred_boxes[i].astype(np.int)
                    xmin = l[1]  # TODO: had to swap some orders here...
                    ymin = l[0]  # TODO: IS ORDER WRONG ???
                    xmax = l[3]
                    ymax = l[2]

                    print(l)
                    vals = 0.029
                    vals2 = 0.0075

                    curr_class = out_classes[i]

                    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2,
                                             edgecolor=colors[curr_class], facecolor='none')
                    ax.add_patch(rect)
                    props = dict(boxstyle='square', facecolor=colors[curr_class], alpha=1)
                    ax.text(xmin / 256 + vals2, 1 - ymin / 256 + vals, str(np.round(out_scores[i], 2)),
                            transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=props,
                            horizontalalignment='left', color='w')

                plt.show()

                skip_flag = True
                if skip_flag:
                    continue






# only use one
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# trained model
#name = '090420_binary_nuclei_tiny_yolov3_256_32_rgb_nbclasses_2_weights_only'
name = '100420_binary_nuclei_tiny_yolov3_256_32_rgb_nbclasses_2_weights_only_max_20_overfit'
name = '100420_binary_nuclei_tiny_yolov3_256_32_rgb_nbclasses_2_weights_only_max_80_overfit' #best?
#name = '110420_binary_nuclei_tiny_yolov3_256_32_rgb_nbclasses_2_weights_only_max_80_overfit_bs_64_no_anchor_fix_pretrained'
#name = '110420_binary_nuclei_tiny_yolov3_256_32_rgb_nbclasses_2_weights_only_max_80_overfit_bs_64_no_anchor_fix_pretrained_lr_1e-5_augs'
#name = '140420_binary_nuclei_tiny_yolov3_256_32_rgb_nbclasses_6_weights_only_max_80_overfit_bs_32_pretrained_lr_1e-3_no_augs_scaled_anchors'
#name = '140420_binary_nuclei_tiny_yolov3_256_32_rgb_nbclasses_1_weights_only_max_80_overfit_bs_32_pretrained_lr_1e-3_no_augs_scaled_anchors'
#name = '140420_binary_nuclei_tiny_yolov3_256_32_rgb_nbclasses_1_weights_only_max_80_overfit_bs_64_lr_1e-4_no_augs_scaled_anchors_breast_only_2'
#name = '140420_binary_nuclei_tiny_yolov3_256_32_rgb_nbclasses_1_weights_only_max_80_overfit_bs_64_lr_1e-4_no_augs_scaled_anchors_breast_only_2'
#name = '160420_binary_nuclei_normal_yolov3_256_32_rgb_nbclasses_1_weights_only_max_80_overfit_bs_64_lr_1e-4_no_augs_new_anchors_freeze_pretrained'
#name = '160420_binary_nuclei_normal_yolov3_256_32_rgb_nbclasses_1_weights_only_max_80_overfit_bs_32_lr_1e-3_no_augs_new_anchors_freeze_1_pretrained_breast_only'
name = '230420_binary_nuclei_tiny_yolov3_256_32_rgb_nbclasses_1_weights_only_max_80_overfit_bs_48_lr_1e-3_new_anchors_rigid_augs_finetune'

curr_datasets = "080420_binary_nuclei_seg_256_32_rgb"

# paths
data_path = '/home/andrep/workspace/nuclei/datasets/' + curr_datasets + '/'
save_model_path = '/home/andrep/workspace/nuclei/output/models/'
history_path = '/home/andrep/workspace/nuclei/output/history/'
datasets_path = '/home/andrep/workspace/nuclei/output/datasets/'

print(save_model_path + 'model_' + name + '.h5')

# from yolo_v3.keras_yolo3.yolo import *
from yolo_v3.yolo3.model import yolo_head
from keras.models import load_model
from yolo_v3.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body, yolo_head
# from yolo_v3.yolo3 import letterbox_image
from yolo_v3.keras_yolo3.yolo import tiny_yolo_body
from yolo_v3.yolo3.model import yolo_head
from keras.layers import Input
from yolo_v3.Utils.Train_Utils import get_anchors  # , create_tiny_model

# get trained model
input_shape = (256, 256, 3)
num_classes = int(name.split("nbclasses_")[-1].split("_")[0]) #  1 for all-in-one nuclei detector
num_anchors = 6  # 6 for tiny-yolo

# model = load_model(save_model_path + 'model_' + name + '.h5', compile=False)
#model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes)
model.load_weights(save_model_path + 'model_' + name + '.h5')  # make sure model, anchors and classes match
#model.save("./yolo_test_model.h5")



# try to attach yolo_head on top of body
from keras.layers import Lambda
#model.add(Lambda())


# load anchors
#'''
yolo3_anchors_path = "./yolo_v3/keras_yolo3/model_data/yolo_anchors.txt"
tiny_yolo3_anchors_path = "./yolo_v3/keras_yolo3/model_data/tiny_yolo_anchors.txt"

tiny_yolo3_anchors_path = "./yolo_v3/keras_yolo3/model_data/tiny_yolo_anchors.txt"
yolo3_anchors_path = "./yolo_v3/keras_yolo3/model_data/yolo_anchors.txt"
new_yolo3_anchors_path = "./yolo_anchors_pannuke.txt"
new_tiny_yolo3_anchors_path = "./tiny_yolo_anchors_pannuke.txt"
anchors = get_anchors(new_tiny_yolo3_anchors_path)
#anchors = get_anchors(new_yolo3_anchors_path)
#anchors = get_anchors(yolo3_anchors_path)

#anchors = np.round(anchors * 256/416).astype(int)  # scale anchors to 256x256 input (originally 416x416)
#'''

# define tiny yolo3 model
'''
from yolo_v3.Utils.Train_Utils import get_anchors, create_tiny_model

tiny_yolo3_anchors_path = "./yolo_v3/keras_yolo3/model_data/yolo_anchors.txt"
anchors = get_anchors(tiny_yolo3_anchors_path)
anchors = np.round(anchors * 256/416).astype(int) # scale anchors to 256x256 input (originally 416x416)
num_classes = 1  # only nuclei
weights_path = save_model_path + 'model_' + name + '.h5'

# TODO: Not using pre-trained model currently... set load_pretrained=True to use pretrained model
model = create_tiny_model(input_shape[:-1], anchors, num_classes, load_pretrained=True,
                          freeze_body=None, weights_path=weights_path)
'''

curr_set = ["/home/andrep/workspace/nuclei/A05.svs"]

print(model.summary())

th = 0.5

shuffle(curr_set)

dsc_list = []
sens_list = []
prec_list = []
imgs = []

colors = ['blue', 'green', 'orange', 'yellow', 'purple', 'magenta']

num = 3000
hit = 0
tot = 0
gts = []
preds = []

# run inference on WSI, patch-wise
for path in tqdm(curr_set):
    func(path)