import numpy as np
import openslide as ops
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
import skimage.io as io
import matplotlib.colors as mcolors
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
import h5py
from xml.etree import ElementTree as ET
import numba as nb
from skimage.transform import resize


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


# make colormap to use for plotting pred on 2D-slice
colors = [(0,1,0,i) for i in np.linspace(0,1,3)]
cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)

# user params
window = 256
stride = 32

# current date
date = "080420"

# paths
#loc = "/hdd/NUCLEI/MoNuSeg Training Data/Tissue images/TCGA-38-6178-01Z-00-DX1.tif"
data_path = "/mnt/EncryptedPathology/pathology/data/pannuke/"
end_path = "/home/andrep/workspace/nuclei/datasets/" + date + "_binary_nuclei_seg_" + str(window) + \
 "_" + str(stride) + "_rgb/"

# RGB
channels = 3

if not os.path.exists(end_path):
	os.makedirs(end_path)

# for each fold
for fold in tqdm(os.listdir(data_path), "Fold: "):
	curr_fold = data_path + fold + "/"
	images_path = curr_fold + "images/" + fold.replace("_", "") + "/images.npy"
	types_path = curr_fold + "images/" + fold.replace("_", "") + "/types.npy"
	masks_path = curr_fold + "masks/" + fold.replace("_", "") + "/masks.npy"

	images = np.load(images_path)
	types = np.load(types_path)
	masks = np.load(masks_path)

	# for each patch
	for i in tqdm(range(len(types)), "Patch: "):
		mask = masks[i]
		binary_mask = (np.sum(mask[..., :-1], axis=-1) > 0).astype(int)

		# onehot binary mask
		tmp = np.zeros(binary_mask.shape + (2,), dtype=np.float32)
		tmp[..., 0] = 1 - binary_mask.astype(np.float32)
		tmp[..., 1] = binary_mask.astype(np.float32)
		binary_mask = tmp.copy()
		del tmp

		onehot_mask = (mask > 0).astype(int)
		onehot_mask = onehot_mask[..., np.roll(range(onehot_mask.shape[-1]), 1)]

		# extract bounding box coordinates for each connected component
		bb_boxes = []
		for ii in range(mask.shape[-1]-1):
			tmp = mask[..., ii]
			if len(np.unique(tmp)) == 1:
				continue
			regions = regionprops(tmp.astype(int))
			for region in regions:
				bb_boxes.append(region.bbox + (ii,))
				bb_boxes_labels.append(ii)
		bb_boxes = np.array(bb_boxes)

		# resize to 448x448 images
		image_448 = resize(images[i], (448, 448), order=0, anti_aliasing=False)
		mask_448 = resize(mask, (448, 448), order=0, anti_aliasing=False)

		# do the same but for the 448 images
		bb_boxes_448 = []
		for ii in range(mask_448.shape[-1] - 1):
			tmp = mask_448[..., ii]
			if len(np.unique(tmp)) == 1:
				continue
			regions = regionprops(tmp.astype(int))
			for region in regions:
				bb_boxes_448.append(region.bbox + (ii,))
		bb_boxes_448 = np.array(bb_boxes_448)

		curr_end_path = end_path + fold + "_" + str(i) + "_" + str(types[i]) + ".h5"
		with h5py.File(curr_end_path, "w") as f:
			f.create_dataset("image", data=images[i].astype(np.uint8), compression="gzip", compression_opts=3)
			f.create_dataset("mask", data=mask.astype(np.uint8), compression="gzip", compression_opts=3)
			f.create_dataset("type", data=np.array([types[i]]).astype('S200'), compression="gzip", compression_opts=3)
			f.create_dataset("binary_mask", data=binary_mask.astype(np.uint8), compression="gzip", compression_opts=3)
			f.create_dataset("multiclass_mask", data=onehot_mask.astype(np.uint8), compression="gzip", compression_opts=3)
			f.create_dataset("bb_boxes", data=bb_boxes.astype(np.int16), compression="gzip", compression_opts=3)
			f.create_dataset("image_448", data=image_448.astype(np.int16), compression="gzip", compression_opts=3)
			f.create_dataset("bb_boxes_448", data=bb_boxes_448.astype(np.int16), compression="gzip", compression_opts=3)
			f.close()
