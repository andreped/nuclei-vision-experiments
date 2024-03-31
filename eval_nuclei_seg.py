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
import matplotlib.colors as mcolors


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
    f = h5py.File(datasets_path + 'dataset_' + name + '.hd5', 'r')
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


# only use one
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# trained model
name = '19_06_test_2'
name = '12_07_256_32_rgb'

# paths
data_path = '/hdd/NUCLEI/datasets/binary_nuclei_seg_256_rgb/'
save_model_path = '/hdd/NUCLEI/output/models/'
history_path = '/hdd/NUCLEI/output/history/'
datasets_path = '/hdd/NUCLEI/output/datasets/'


# load test_data_set
test_set = import_set('test', num=None, filter=True)
val_set = import_set('val', num=None, filter=True)
train_set = import_set("train", num=None, filter=True)

shuffle(test_set)
shuffle(val_set)
shuffle(train_set)

print(save_model_path + 'unet_nuclei_' + name + '.h5')

# get trained model
model = load_model(save_model_path + 'unet_nuclei_' + name + '.h5', compile=False)

print(model.summary())

# choose data set
# curr_set = select_pats(curr_set, [101])
set = 'val'
print('set chosen: ')
print(set)

if set == 'test':
    curr_set = test_set.copy()
elif set == 'val':
    curr_set = val_set.copy()
elif set == 'train':
    curr_set = train_set.copy()

# get unique patients for each set:
#print('unique patients:')
#print(unique_patients(curr_set))
# print()

# list of classes
# mask = np.array(unique_classes(curr_set.copy())[1])
#nb_classes, mask = unique_classes(curr_set)
# mask = np.array([1,3])
#print('mask/classes: ')
#print(mask)

# number of samples
#print('num samples: ' + str(len(curr_set)))

# pats = unique_patients(curr_set.copy())
# print('Samples used in eval: ')
# print(pats)

# make colormap to use for plotting pred on 2D-slice
np.random.seed(42)
vals = np.linspace(0,1,256)
np.random.shuffle(vals)
vals = plt.cm.jet(vals)
vals[0, :] = [0,0,0,1]
cmap = plt.cm.colors.ListedColormap(vals)

th = 0.5

shuffle(curr_set)

dsc_list  = []
sens_list = []
prec_list = []
imgs = []

num = 3000
hit = 0
tot = 0
gts = []
preds = []
for path in tqdm(curr_set): #tqdm(curr_set[:num]): # <- use some of the data
#for path in tqdm(curr_set): # <- use all data

    f = h5py.File(path, 'r')
    data = np.array(f['data'], dtype=np.float32)
    gt = np.array(f['label'], dtype=np.uint8)
    f.close()

    #data = np.squeeze(data, axis=0)

    data = data / 255

    #data = np.expand_dims(data, axis=0)

    #data = np.expand_dims(data, axis=0)
    #gt = np.expand_dims(gt, axis=0)

    pred = model.predict(data).astype(np.float32)

    p = np.squeeze(pred, axis=0)
    pred = np.argmax(p, axis=-1)
    #pred = pred[0,...,1]
    gt = gt[0,...,1]

    pred_bin = pred.copy()

    #pred_bin = pred.copy()
    #pred_bin[pred_bin <= th] = 0
    #pred_bin[pred_bin > th] = 1
    pred_bin = pred_bin.astype(int)

    print(pred_bin.shape)
    print(gt.shape)

    #gt = gt[0,...,0]
    #pred_bin = pred_bin[0,...,0]

    #dsc = DSC(pred_bin, gt)  # <- OBS! pred or pred_bin?
    #dsc_list.append(dsc)
    #print(dsc)

    # tumour segmentation by 2D-UNet
    pred = model.predict(data)

    # calculate recall, precision, f1-score
    metrics = precision_recall_fscore_support(gt.flatten(), pred_bin.flatten(), average='binary')
    sens = metrics[1]  # <- NB: prec is first! flipped...
    prec = metrics[0]
    f1_score = metrics[2]
    dsc = metrics[2]  # <- DSC = F1-score for tumour seg. accuracy

    print('Results: ')
    print(metrics)

    dsc_list.append(dsc)
    sens_list.append(sens)
    prec_list.append(prec)

    print(pred_bin.shape)
    print(pred.shape)

    imgs.append([data[0], pred[0,...,1], pred_bin, gt])

    '''
    fig, ax = plt.subplots(1, 4, figsize=(12, 8))
    ax[0].imshow(data[0])
    ax[1].imshow(pred, cmap="gray")
    ax[2].imshow(pred_bin, cmap="gray")
    ax[3].imshow(gt, cmap="gray")
    count = 0
    for i in range(2):
        for j in range(2):
            if not count == 2:
                1
                #ax[count].set_title(names[count])
            else:
                ax[count].set_title('DSC = ' + str(np.round(dsc, 3)))
                #ax[count].set_title(names[count] + ' (DSC = ' + str(np.round(dsc, 3)) + ')')
            ax[count].axis('off')
            count += 1
    #fig.suptitle('DSC: ' + str(np.round(dsc, 3)))
    fig.tight_layout()
    plt.show()
    '''

    fig, ax = plt.subplots(1, 4, figsize=(12, 8))
    ax[0].imshow(data[0])
    ax[1].imshow(pred, cmap=cmap)
    ax[2].imshow(pred_bin, cmap=cmap)
    ax[3].imshow(gt, cmap="gray")
    count = 0
    for i in range(2):
        for j in range(2):
            if not count == 2:
                1
                # ax[count].set_title(names[count])
            else:
                ax[count].set_title('DSC = ' + str(np.round(dsc, 3)))
                # ax[count].set_title(names[count] + ' (DSC = ' + str(np.round(dsc, 3)) + ')')
            ax[count].axis('off')
            count += 1
    # fig.suptitle('DSC: ' + str(np.round(dsc, 3)))
    fig.tight_layout()
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
size = fig.get_size_inches()*fig.dpi
plt.tight_layout()

for i in range(6):
    for j in range(4):
        if j == 0:
            ax[i,j].imshow(imgs[i][j])
        else:
            ax[i,j].imshow(imgs[i][j], cmap="gray")

        if j == 2:
            ax[i,j].set_title('DSC = ' + str(np.round(dsc_list[i], 3)))

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
