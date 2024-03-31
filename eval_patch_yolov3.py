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
name = '290520_binary_nuclei_tiny_yolov3_256_32_rgb_nbclasses_6_weights_only_max_80_overfit_bs_48_lr_1e-3_new_anchors_rigid_augs_finetune' # best multi-class detector ?

curr_datasets = "080420_binary_nuclei_seg_256_32_rgb"

# paths
data_path = '/home/andrep/workspace/nuclei/datasets/' + curr_datasets + '/'
save_model_path = '/home/andrep/workspace/nuclei/output/models/'
history_path = '/home/andrep/workspace/nuclei/output/history/'
datasets_path = '/home/andrep/workspace/nuclei/output/datasets/'

# load test_data_set
sets = "test"
with h5py.File(datasets_path + 'dataset_' + name + '.h5', 'r') as f:
    curr_dir = []
    keys = list(f[sets].keys())
    for key in keys:
        curr_dir.append(np.array(f[sets + "/" + key]))
    curr_set = []
    for c in curr_dir:
        for c2 in c:
            curr_set.append(c2)
    curr_set = [curr_set[i].decode("UTF-8") for i in range(len(curr_set))]

np.random.shuffle(curr_dir)

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
model.load_weights(save_model_path + 'model_' + name + '.h5')  # make sure model, anchors and classes  match

# TODO: Convert weights -> .h5-model to be used for conversion!
model.save("./yolo_multiclass_nuclei_test_model.h5")



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

print(model.summary())

# choose data set
# curr_set = select_pats(curr_set, [101])
print('set chosen: ')
print(sets)

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
for path in tqdm(curr_set):  # tqdm(curr_set[:num]): # <- use some of the data
    # for path in tqdm(curr_set): # <- use all data

    f = h5py.File(path, 'r')
    data = np.array(f['image'], dtype=np.float32)
    gt_bboxes = np.array(f['bb_boxes'], dtype=np.uint8)
    f.close()

    print(data.shape)
    print(anchors)

    data = data.astype(np.float32) / 255.
    prediction = model.predict(np.expand_dims(data, axis=0))#.astype(np.float32)

    print(prediction[0].shape)
    #exit()

    iou = 0.0001 #0.1  # 0.45
    score = 0.3  # 0.3
    #input_image_shape = (256, 256)
    input_image_shape = K.placeholder(shape=(2, ))
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

    #order = np.argsort(out_scores)[::-1]
    #pred_boxes = out_boxes[order[:cutoff]]
    #print(out_boxes)
    #print(out_scores[order[:cutoff]])
    print("---")

    for i in range(gt_bboxes.shape[0]):
        l = gt_bboxes[i].astype(np.int)
        xmin = l[1]  # TODO: had to swap some orders here...
        ymin = l[0]
        xmax = l[3]
        ymax = l[2]
        curr_class = l[4]
        if nb_classes == 1:
            curr_class = 0

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2,
                                 edgecolor=colors[curr_class], facecolor='none')
        ax.add_patch(rect)
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
        ax.text(xmin/256+vals2, 1-ymin/256+vals, str(np.round(out_scores[i], 2)),
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props,
                horizontalalignment='left', color='w')

    plt.show()


    skip_flag = True
    if skip_flag:
        continue









    exit()

    print(len(prediction))
    print(prediction[0].shape)
    print(prediction[1].shape)

    feats = prediction.copy()
    print()
    print(feats[1].shape)

    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(
        feats[1], anchors, num_classes, input_shape[:2], calc_loss=False
    )

    print(box_xy)
    print(box_wh)
    print()

    exit()
    # print(prediction.shape)
    # print(prediction)

    '''
    iou = 0.45
    score = 0.3
    input_image_shape = (256, 256)
    nb_classes = 1

    boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                    nb_classes, input_image_shape,
                    score_threshold=score, iou_threshold=iou)
    '''

    exit()

    prediction = prediction.flatten()
    probs = prediction[:980]
    confs = prediction[980:(980 + 98)]
    preds = prediction[(980 + 98):]
    pred_boxes = []
    for i in range(98):
        pred_boxes.append(preds[int(i * 4):int((i + 1) * 4)])
    pred_boxes = np.array(pred_boxes)

    th = 0.4
    confs[np.isnan(confs)] = 0
    tmp = confs > th
    pred_boxes = pred_boxes[tmp]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(data)

    print(gt_bboxes.shape)
    print(pred_boxes.shape)

    for i in range(gt_bboxes.shape[0]):
        l = gt_bboxes[i].astype(np.int)
        xmin = l[1]  # TODO: had to swap some orders here...
        ymin = l[0]
        xmax = l[3]
        ymax = l[2]

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    for i in range(pred_boxes.shape[0]):
        l = pred_boxes[i].astype(np.int)
        xmin = l[1]  # TODO: had to swap some orders here...
        ymin = l[0]
        xmax = l[3]
        ymax = l[2]

        print(l)

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    plt.show()

print('Sens')
print(np.mean(sens_list))

print('Prec')
print(np.mean(prec_list))

print('DSC')
print(np.mean(dsc_list))

'''
order = np.argsort(dsc_list)[-6:]

print(order)

new_imgs = []
new_dsc_list = []
for i in order:
    new_imgs.append(imgs[i])
    new_dsc_list.append(dsc_list[i])

print(len(imgs))
print(len(new_dsc_list))

imgs = new_imgs
dsc_list = new_dsc_list

order = np.array(range(len(imgs)))
shuffle(order)

new_imgs = []
new_dsc_list = []
for i in order:
    new_imgs.append(imgs[i])
    new_dsc_list.append(dsc_list[i])

imgs = new_imgs
dsc_list = new_dsc_list

print(order)

print(np.sort(dsc_list)[::-1])

#imgs = [x for _, x in sorted(zip(order, imgs))]
#dsc_list = [x for _, x in sorted(zip(order, dsc_list))]
'''

'''
new_imgs = []
new_dsc_list = []
for i in tmp:
    new_imgs.append(imgs[i])
    new_dsc_list.append(dsc_list[i])

order = np.array(range(len(new_imgs)))
shuffle(order)
new_imgs = new_imgs[order]
new_dsc_list = new_dsc_list[order]
'''

'''
order = list(range(len(imgs)))
shuffle(order)

imgs = [x for _, x in sorted(zip(order, imgs))]
dsc_list = [x for _, x in sorted(zip(order, dsc_list))]
'''

cols = ['WSI', 'Confidence map', 'Binary prediction', 'Ground truth']

fig, ax = plt.subplots(6, 4, figsize=(12, 8))
size = fig.get_size_inches() * fig.dpi
plt.tight_layout()

for i in range(6):
    for j in range(4):
        if j == 0:
            ax[i, j].imshow(imgs[i][j])
        else:
            ax[i, j].imshow(imgs[i][j], cmap="gray")

        if j == 2:
            ax[i, j].set_title('DSC = ' + str(np.round(dsc_list[i], 3)))

cnt = 0
for i in range(6):
    for j in range(4):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        cnt += 1

cnt = 0
for axx, col in zip(ax[0], cols):
    if cnt == 2:
        axx.set_title(col + ' (DSC = ' + str(np.round(dsc_list[i], 3)) + ')')
    else:
        axx.set_title(col)
    cnt += 1

plt.show()

plt.hist(dsc_list)
plt.show()

exit()

print('recall in each class: ')
print(conf[0, 0] / np.sum(conf[0, :]))
print(conf[1, 1] / np.sum(conf[1, :]))
print(conf[2, 2] / np.sum(conf[2, :]))
# -> if : [[15, 3], [0, 3]] => perfect recall of class 3

print('overall accuracy: ')
acc = accuracy_score(gts, preds)
print(acc)

print('weighted overall accuracy: ')
wacc = accuracy_score(gts, preds, sample_weight=w)
print(wacc)

plt.figure()
plt.rc('font', family='serif')
plt.hist([gts, preds], bins=3, label=['gt', 'pred'])
plt.xticks(unique_classes(curr_set)[1])
plt.title('hits: ' + str(hit) + ', total: ' + str(tot) + ', acc: ' + str(np.round(acc, 4)) +
          ', wacc: ' + str(np.round(wacc, 4)), fontsize=16)
# plt.tick_params(axis='both', which='minor', labelsize=30)
# plt.text(min(unique_classes(tmp_set)[1])-0.5, 2,
#	'Results: \n hits: ' + str(hit) + '\n total: \n' + str(tot) + 'acc: \n' + str(hit/tot),
#	fontsize=14)
# plt.hist([counts_gt, counts_pred], bins=9, label=['gt', 'pred'])
# plt.subplots_adjust(left=0.25)

pats = unique_patients(curr_set)
pats_str = [str(i) for i in pats]
pats_str = '_'.join(pats_str)

plt.legend(loc='best')
# plt.savefig('/mnt/EncryptedData2/pathology/output/eval_hists/hist_01_03_pats_' + pats_str + '.png', bbox_inches='tight', dpi=600)
plt.show()

# gen = batch_gen2(train_set[:64], batch_size=64, aug = {}, shuffle_list = False, epochs = 1, classes=mask, grades=grades)

# cnt = 1
# for data, label in gen:
#    1


# from keras import backend as K
# K.clear_session()
