import numpy as np
import javabridge as jb
import bioformats as bf
import matplotlib.pyplot as plt
# import subprocess as sp
import multiprocessing as mp
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool

import itertools as it
import cv2
from tensorflow.python.keras.models import load_model
from tqdm import tqdm
import os
import h5py
import imageio
from skimage.transform.integral import integral_image, integrate
import numba as nb


@nb.jit(nopython=True)
def sc_any(array):
    for x in array.flat:
        if x:
            return True
    return False


@nb.jit(nopython=True)
def sc_all(array):
    for x in array.flat:
        if not x:
            return False
    return True


def maxminscale(tmp):
    # if (len(np.unique(tmp)) > 1):
    if sc_any(tmp):
        tmp = tmp - np.amin(tmp)
        tmp = tmp / np.amax(tmp)
    return tmp


# integrate over given window in integral image ii
# r0, c0 : top-left corner of block to be summed
# r1, c1 : bottom-right corner of block to be summed
def integrate_window(ii, r0, c0, r1, c1):
    return ii[c1, r1] - ii[c1, r0] - ii[c0, r1] + ii[c0, r0]


def minmaxvalues(tmp):
    return (np.amin(tmp), np.amax(tmp))


# hide GPU in prediction
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# folder name (specify output tile size here, i.e. images_tile-size_10x)
name = '11_06_tumor_images_512_0.625x_hdf5'
#name = '11_06_tumor_images_512_2.5x_hdf5'

print('----')

# paths to directories of relevant data
data_path = '/mnt/EncryptedData2/pathology/images/'
gt_path = '/mnt/EncryptedData2/pathology/annotations/exported_mask/'
end_path = '/mnt/EncryptedData2/pathology/datasets/' + name

if not os.path.isdir(end_path):
    os.makedirs(end_path)

# load grade-file
with h5py.File('/mnt/EncryptedData2/pathology/datasets/grades_gt_109.hd5', 'r') as file:
    grades = np.array(file['labels'])

# user-specific settings
ds = 10  # <- downscale factor used in extraction of mask from Qupath
print(name.split('_')[5].split('x')[0])
image_plane = int(np.log(40 / np.float32(name.split('_')[5].split('x')[0])) / np.log(2)) # <- which image plane to use when reading image, i.e. if 2 => 40x/2² = 10x magnified image
th = 0.5  # <- threshold for binarization of GT after interpolation
window_size = int(name.split('_')[4])
stride = window_size  # <- no overlap!
#stride = int(window_size/2) # 50 % overlap

print('--')
print(image_plane)
print(window_size)

# for all tumor-annotated vsi-images
datas = []
for d in os.listdir(data_path):
    d1 = d.split('.')[-1]
    if (d1 == 'vsi') and (len(d.split('_')) == 1):
        datas.append(data_path + d)

# start virtual machine
jb.start_vm(class_path=bf.JARS, max_heap_size="20G")

cnt = 0


# for each tumor-annotated slide image
for gt_dir in tqdm(os.listdir(gt_path)):

    # get patient ID
    image = gt_dir.split('/')[-1].split('.')[0]

    # read corresponding vsi-image
    path = data_path + image + '.vsi'

    # get corresponding histological grade label
    label = grades[grades[:, 0] == int(image), 1][0]
    # print(image, label)

    new_path = end_path + '/' + 'exported_tiles_' + str(image) + '_' + str(label)
    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    # read current gt
    gt = cv2.imread(gt_path + gt_dir)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    gt = maxminscale(gt).astype(np.int64)

    # initialize image reader
    ImageReader = bf.formatreader.make_image_reader_class()

    # Read downsampled image -> original vsi-image
    reader2 = ImageReader()
    reader2.setId(path)
    image_plane2 = 4  # <- image is effectively "downsampled" by 2^n (i.e. 2^7 = 128)
    reader2.setSeries(image_plane2)
    M2 = reader2.getSizeY()
    N2 = reader2.getSizeX()
    out = np.reshape(reader2.openBytes(0), (M2, N2, 3))
    out = np.uint8(255 * maxminscale(out))

    # apply tissue detector
    tmp = cv2.cvtColor(out.copy(), cv2.COLOR_RGB2HSV)  # RGB -> HSV
    tmp = cv2.medianBlur(tmp[..., 1], 7) # use saturation channel and smooth using median filter
    th, tissue_mask = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # <- apply otsu thresholding
    tissue_mask = cv2.resize(tissue_mask, gt.shape[::-1], cv2.INTER_LINEAR)  # reshape mask to fit tumour annotation mask
    tissue_mask[tissue_mask <= th] = 0
    tissue_mask[tissue_mask > th] = 1  # binarize
    tissue_mask = tissue_mask.astype(np.int64)

    # remove arrays not needed anymore
    del out, tmp

    # calculate integral images of both tissue_mask and GT
    #gt_int_img = integral_image(gt)
    #tissue_int_img = integral_image(tissue_mask)

    # remove arrays not needed anymore
    #del gt, tissue_mask

    # reader = bf.ImageReader(path)
    reader = ImageReader()
    reader.setId(path)
    reader.setSeries(image_plane)
    M = reader.getSizeY()
    N = reader.getSizeX()

    #f['array'][150]

    for x in range(int(np.ceil(N / stride)) - 1):
        for y in range(int(np.ceil(M / stride)) - 1):

            # image location
            X = int(x * stride)
            Y = int(y * stride)

            # GT location
            Xg = int(np.round(X / ds * 2 ** image_plane))
            Yg = int(np.round(Y / ds * 2 ** image_plane))

            #'''
            # read current GT and tissue patch
            gt_curr = gt[Yg:(Yg + int(np.round(window_size / ds * 2 ** image_plane))),
                      Xg:(Xg + int(np.round(window_size / ds * 2 ** image_plane)))]
            tissue_curr = tissue_mask[Yg:(Yg + int(np.round(window_size / ds * 2 ** image_plane))),
                      Xg:(Xg + int(np.round(window_size / ds * 2 ** image_plane)))]

            # count nonzero elements -> number of tumor pixels
            num_tumor_pixels = np.count_nonzero(gt_curr)
            tot = int(np.round(window_size / ds * 2 ** image_plane)) ** 2
            fraction_tumor = num_tumor_pixels / tot

            # count nonzero elements -> number of non-redundant tumor pixels
            num_tumor_pixels = np.count_nonzero(tissue_curr)
            tot = int(np.round(window_size / ds * 2 ** image_plane)) ** 2
            fraction_tissue = num_tumor_pixels / tot

            ''' # <- integral image approach was actually slower (!)
            # calculate percentage of tumor and tissue in current patch
            fraction_tumor = integrate_window(gt_int_img, Xg, Yg,\
                                Xg + int(np.round(window_size / ds * 2 ** image_plane)),\
                                Yg + int(np.round(window_size / ds * 2 ** image_plane))) / (int(np.round(window_size / ds * 2 ** image_plane))) ** 2
            fraction_tissue = integrate_window(tissue_int_img, Xg, Yg,\
                                              Xg + int(np.round(window_size / ds * 2 ** image_plane)),\
                                              Yg + int(np.round(window_size / ds * 2 ** image_plane))) / (int(np.round(window_size / ds * 2 ** image_plane))) ** 2
            '''

            #print(fraction_tumor, fraction_tissue)

            # condition on tumor tiles (accept also non-tumor tiles), and tissue tiles
            if True and (fraction_tissue >= 0.25):

                #print(fraction_tumor, fraction_tissue)

                # read tile directly from vsi
                data = np.reshape(reader.openBytesXYWH(0, X, Y, window_size, window_size),
                                  (window_size, window_size, 3))

                # handle incomplete boundary patches
                if data.shape != (window_size, window_size, 3):
                    curr = np.zeros((window_size, window_size, 3))
                    curr[Y:(Y + window_size), X:(X + window_size)] = data
                    data = curr.copy()
                    del curr

                # current name of tile
                tile_name = image + '_' + '(' + str(X) + ',' + str(Y) + ',' + str(window_size) + ',' + \
                            str(window_size) + ')' '_tumor_' + str(np.round(fraction_tumor, 2)) + \
                            '_tissue_' + str(np.round(fraction_tissue, 2))

                #plt.imshow(data)
                #plt.show()

                # save as PNG
                # imsave(new_path + '/' + tile_name + '.png', data)
                #imageio.imwrite(new_path + '/' + tile_name + '.png', data)

                ## save as hdf5
                gt_curr = cv2.resize(gt_curr.astype(np.uint8), data.shape[:-1][::-1], cv2.INTER_NEAREST).astype(np.float32)
                gt_curr = np.expand_dims(gt_curr, axis=-1)
                gt_curr = gt_curr.astype(np.float32)
                gt_curr = np.concatenate([gt_curr, 1 - gt_curr], axis=-1).astype(np.uint8)

                '''
                f = h5py.File(new_path + '/' + tile_name + '.h5', 'w')
                f.create_dataset("data", data=data, compression="gzip", compression_opts=3)
                f.create_dataset("label", data=gt_curr, compression="gzip", compression_opts=3)
                f.close()
                '''


                #'''
                tissue_curr = cv2.resize(tissue_curr.astype(np.uint8), data.shape[:-1][::-1], cv2.INTER_NEAREST).astype(np.float32)
                fig, ax = plt.subplots(1,3, figsize=(20,13))
                ax[0].imshow(data)
                ax[1].imshow(gt_curr[...,0])
                ax[2].imshow(tissue_curr)
                plt.show()
                #'''

                del data

    del reader, reader2

jb.kill_vm()
