

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
window = 512
stride = 32

# current date
date = '22_06'

# paths
#loc = "/hdd/NUCLEI/MoNuSeg Training Data/Tissue images/TCGA-38-6178-01Z-00-DX1.tif"
data_path = '/hdd/NUCLEI/MoNuSeg Training Data/Tissue images/'
gt_path = '/hdd/NUCLEI/MoNuSeg Training Data/Annotations/'
end_path = '/hdd/NUCLEI/datasets/binary_nuclei_seg' + '_' + str(window) + \
 '_' + str(stride) + '_rgb' + '_' + date + '/'

# RGB
channels = 3

if not os.path.exists(end_path):
	os.makedirs(end_path)

# if gray in end_path, generate gray output
if end_path.split('_')[-1].startswith('gray'):
	channels = 1

# for each WSI
for path in tqdm(os.listdir(data_path), "WSI"):

	if not path.startswith('.DS'):

		print(path)

		# current patch (of WSI)
		image = path.split('.tif')[0]
		new_path = end_path + image + '/'

		# create folder for data if not exist
		if not os.path.exists(new_path):
			os.makedirs(new_path)

		# get image patch
		data = plt.imread(data_path + path)#[..., :3]

		# get GT
		gt_loc = path.split('.tif')[0] + '.xml'
		tree = ET.parse(gt_path + gt_loc)
		root = tree.getroot()
		locations = []
		gt = np.zeros(data.shape[:-1])
		gts = []
		regions = root[0][1]

		# for each nuclei
		for region in regions[1:]:
			vertices = region[1]
			locs = []
			gt_curr = np.zeros_like(gt)

			# extract vertices for each nuclei
			for vert in vertices:
				vert = vert.attrib
				x = int(round(float(vert['X'])))
				y = int(round(float(vert['Y'])))
				locs.append((x,y))
				locations.append((x,y))

			# fill set of vertices to form segmented nuclei
			coords = np.array(locs)
			cv2.fillPoly(gt_curr, pts=[coords], color=(255,255,255))
			gts.append(gt_curr.astype(np.uint8))

			# add segmented nuclei to GT image
			gt += gt_curr
		gts = np.array(gts).astype(np.uint8)
		print(gts.shape)





		#'''
		#gt = np.zeros(data.shape[:-1])
		#for loc in locations:
		#	gt[loc[::-1]] = 1

		#gt = gt.astype(np.uint8)
		#gt = np.zeros(data.shape[:-1])
		#coords = []
		#for loc in locations:
		#	coords.append(loc)
		#coords = np.array(coords)
		#cv2.fillPoly(gt, pts = [coords], color=(255,255,255))
		#'''

		# binarize, neglect object wise GT
		gt[gt > 0] = 1

		# patch gen
		M, N = gt.shape

		#window = 256
		#stride = 32

		NN = int(np.ceil(N/stride))
		MM = int(np.ceil(M/stride))

		cnt = 0
		for i in range(MM):
			for j in range(NN):
				data_curr = np.zeros((window, window, 3))
				curr = data[int(np.round(i*stride)):int(np.round(i*stride+window)), int(np.round(j*stride)):int(np.round(j*stride+window))]
				data_curr[:curr.shape[0], :curr.shape[1]] = curr

				#if channels == 1:
				#data_curr = cv2.cvtColor(data_curr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
				#data_curr = np.expand_dims(data_curr, axis=-1)
				data_curr = data_curr.astype(np.uint8)

				# check proportion of "outside/black" pixels
				if np.count_nonzero(data_curr == 0) <= window*stride*5:

					# get current GT
					gt_curr = np.zeros((window, window))
					curr = gt[int(np.round(i*stride)):int(np.round(i*stride+window)), int(np.round(j*stride)):int(np.round(j*stride+window))]
					gt_curr[:curr.shape[0], :curr.shape[1]] = curr

					# one-hot encoding
					gt_curr = np.expand_dims(gt_curr, axis=-1)
					gt_curr = gt_curr.astype(np.float32)
					gt_curr = np.concatenate([1 - gt_curr, gt_curr], axis=-1).astype(np.uint8)

					# add dim in front
					data_curr = np.expand_dims(data_curr, axis=0)
					gt_curr = np.expand_dims(gt_curr, axis=0)

					# save as hd5
					f = h5py.File(new_path + str(cnt) + '.h5', 'w')
					f.create_dataset("data", data=data_curr, compression="gzip", compression_opts=3)
					f.create_dataset("label", data=gt_curr, compression="gzip", compression_opts=3)
					f.close()

					cnt += 1



					#plt.imshow(data_curr[0])
					#plt.show()

					#plt.imshow(gt_curr[0,...,0])
					#plt.show()








	'''
	from dynamic_watershed import post_process
	p1, p2 = 200, 0.5
	nuclei = post_process(gt, p1, thresh=p2)

	#nuclei = label(gt)


	vals = np.linspace(0,1,256)
	np.random.shuffle(vals)
	vals = plt.cm.jet(vals)
	vals[0, :] = [0,0,0,1]
	cmap2 = plt.cm.colors.ListedColormap(vals)

	names = ['WSI', 'Overlay', 'GT', 'GT fixed']
	fig, ax = plt.subplots(1, 4, figsize=(26,13))
	fig.tight_layout()
	ax[0].imshow(data)
	ax[1].imshow(data)
	ax[1].imshow(gt, cmap, alpha=0.8)
	ax[2].imshow(gt, cmap="gray")
	ax[3].imshow(nuclei, cmap=cmap2)

	for i,n in enumerate(names):
		ax[i].set_title(n)
		ax[i].set_xticks([])
		ax[i].set_yticks([])

	plt.show()
	'''



	print('---')

	#exit()
	#plt.imshow(data)
	#plt.show()

