

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import openslide as ops
import itertools as it
import cv2
from tqdm import tqdm
import os
import h5py
import imageio
import skimage.io as io
import numba as nb
import matplotlib.colors as mcolors
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
from xml.etree import ElementTree as ET


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


def func(gt_dir):

    file = gt_dir

    # set actual image resolution/magnification
    img_res = 20

    # folder name (specify output tile size here, i.e. images_tile-size_10x)
    #name = 'images_512_5x_PNG' # 20x, 5x, 1.25x available

    # paths to directories of relevant data
    data_path = '/hdd/PAIP/data/'
    gt_path = '/hdd/PAIP/data/'
    end_path = '/hdd/PAIP/datasets/' + name

    # user-specific settings
    image_plane = int(np.log(img_res / np.float32(name.split('_')[2].split('x')[0])) / np.log(4)) # 2 # <- which image plane to use when reading image, i.e. if 2 => 40x/2² = 10x magnified image
    window_size = int(name.split('_')[1])
    stride = window_size  # <- no overlap!
    th = 0.5  # <- threshold for binarization of GT after interpolation

    # current WSI
    label = file.split('.')[0].split('_')[2]

    # date
    date = '_'.join(file.split('_')[:2])

    new_path = end_path + '/' + date + '_' + label + '/'

    # where to save data and GT files for each WSI
    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    # read current gt
    gt = io.imread(data_path + date + '_' + label + '_whole' + '.tif')

    # downsample ds
    ds = 4 ** image_plane
    gt = cv2.resize(gt, (0,0), fx=1/ds, fy=1/ds, interpolation=cv2.INTER_NEAREST)

    # initialize reader
    reader = ops.OpenSlide(data_path + date + '_' + label + '.svs')
    M, N = reader.dimensions

    # tissue detection
    image_plane_full = 2
    img = np.array(reader.read_region((0,0), image_plane_full, reader.level_dimensions[image_plane_full]))[:,:,:3]

    rgb_th = (235, 210, 235)
    mask_tmp = 255 - cv2.inRange(img.astype(np.uint8), rgb_th, (255, 255, 255))
    mask_tmp[mask_tmp > 0] = 1

    # resize GT mask
    gt_rs = cv2.resize(gt, img.shape[:2][::-1], cv2.INTER_NEAREST)

    # apply tissue mask correction
    tissue_mask = mask_tmp.copy()
    mask_tmp *= gt_rs

    # remove fragments inside tumor annotation
    mask_tmp = remove_small_holes(mask_tmp.astype(bool), area_threshold = int(np.round(0.0005*np.prod(mask_tmp.shape)))).astype(int)

    # also fix GT by removing smallest generated candidates (fragments after initalization)
    mask_tmp = remove_small_objects(mask_tmp.astype(bool), min_size = int(np.round(0.00001*np.prod(mask_tmp.shape)))).astype(int)

    # update GT
    gt = mask_tmp.copy()
    del mask_tmp, gt_rs, img
    #gt = cv2.resize(mask_tmp, gt.shape[::-1], cv2.INTER_NEAREST)

    # downsample factor for each axis
    image_plane_new = 4 ** image_plane

    # patchgen
    for x in range(int(np.ceil(M / stride / image_plane_new))):
        for y in range(int(np.ceil(N / stride / image_plane_new))):

            X = int(np.round(x * stride * image_plane_new))
            Y = int(np.round(y * stride * image_plane_new))

            Xg = int(np.round(x * stride * image_plane_new / ds))
            Yg = int(np.round(y * stride * image_plane_new / ds))

            # get current GT
            gt_curr = gt[Yg:int(np.round(Yg + window_size * image_plane_new / ds)), \
            Xg:int(np.round(Xg + window_size * image_plane_new / ds))]

            # get current tissue
            tissue_curr = tissue_mask[Yg:int(np.round(Yg + window_size * image_plane_new / ds)), \
            Xg:int(np.round(Xg + window_size * image_plane_new / ds))]

            # if no tumor annotation for current tile, skip it (do this since extra step it's much faster than counting)
            if True: #not (np.all(gt_curr == 0)):

                # count nonzero elements -> number of non-redundant tumor pixels
                num_tumor_pixels = np.count_nonzero(tissue_curr)
                tot = int(np.round(window_size * image_plane_new / ds)) ** 2
                fraction_tissue = num_tumor_pixels / tot
                del tissue_curr

                # only accept tile if >= 25% tissue
                if fraction_tissue >= 0.25: #True: #(fraction >= 0.75):

                    #Read tile from .svs
                    data = np.array(reader.read_region((X,Y), image_plane, (window_size, window_size)))[:,:,:3]
                    data[data == (0,0,0)] = 255

                    if (gt_curr.shape != (window_size, window_size)) and (np.prod(gt_curr.shape) != 0):
                        curr = np.zeros((window_size, window_size))
                        curr[:gt_curr.shape[0], :gt_curr.shape[1]] = gt_curr
                        gt_curr = curr.copy()

                    #del curr # <- can't delete this variable for some reason...?

                    # current name of tile
                    tile_name = new_path + '(' + str(X) + ',' + str(Y) + ',' + str(window_size) + ',' + str(window_size) + ')'

                    '''
                    fig, ax = plt.subplots(1,2, figsize=(20,10))
                    ax[0].imshow(data)
                    ax[1].imshow(gt_curr)
                    plt.show()
                    '''

                    # save as PNG
                    # imsave(new_path + '/' + tile_name + '.png', data)
                    #imageio.imwrite(new_path + '/' + tile_name + '.png', data)
                    #imageio.imwrite(new_path_gt + '/' + tile_name + '.png', gt_curr)

                    # one-hot encoding
                    gt_curr = np.expand_dims(gt_curr, axis=-1)
                    gt_curr = gt_curr.astype(np.float32)
                    gt_curr = np.concatenate([gt_curr, 1 - gt_curr], axis=-1).astype(np.uint8)

                    # add dim in front
                    data = np.expand_dims(data, axis=0)
                    gt_curr = np.expand_dims(gt, axis=0)

                    # save as hd5
                    f = h5py.File(tile_name + '.h5', 'w')
                    f.create_dataset("data", data=data, compression="gzip", compression_opts=3)
                    f.create_dataset("label", data=gt_curr, compression="gzip", compression_opts=3)
                    f.close()

    reader.close()



if __name__ == "__main__":

    # hide GPU in prediction
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # folder name (specify output tile size here, i.e. images_tile-size_10x)
    name = 'images_512_1.25x_seg' # 20x, 5x, 1.25x

    # paths
    data_path = '/hdd/PAIP/data/'
    end_path = '/hdd/PAIP/datasets/' + name

    if not os.path.isdir(end_path):
        os.makedirs(end_path)

    # paths to directories of relevant data
    svs_files = []; xml_files = []; tif_files = []

    # get paths
    for path in os.listdir(data_path):
        curr = path.split('.')[-1].lower()
        if curr == 'svs':
            svs_files.append(path)
        elif curr == 'xml':
            xml_files.append(path)
        elif curr == 'tif':
            tif_files.append(path)

    # sort
    svs_files = sorted(svs_files); xml_files = sorted(xml_files); tif_files = sorted(tif_files)

    # list of paths to GT files
    gts = tif_files.copy()
    gts = [x for x in gts if "viable" not in x]

    # run processes in parallel
    proc_num = 8 #16
    p = mp.Pool(proc_num)
    num_tasks = len(gts)
    r = list(tqdm(p.imap(func, gts), "WSI", total=num_tasks)) #list(tqdm(p.imap(func,gts),total=num_tasks))
    p.close()
    p.join()