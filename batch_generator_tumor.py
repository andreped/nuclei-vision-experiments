from smistad.smistad_dataset import get_dataset_files
import random
import h5py
import numpy as np
from scipy.ndimage.interpolation import rotate, shift, affine_transform, zoom
from numpy.random import random_sample, rand, random_integers, uniform
# import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy
from numpy.random import shuffle
import numba as nb
import PIL
from io import BytesIO


# have to do this to use matplotlib viewing with GPU for some reason (used to work before, thus didn't need to do this...)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")


# apply random gaussian noise to RGB image
def add_gaussian_blur2(input_im, output, sigmas_max):
    #blur = cv2.GaussianBlur(img, (5,5), 0)

    # oscillate around 0 => add or substract value with equal probability of add/sub, and as strong in both directions
    means = (0,0,0)

    # random sigma uniform [0, sigmas_max]
    sigmas = (round(uniform(0, sigmas_max[0])), round(uniform(0, sigmas_max[1])), round(uniform(0, sigmas_max[2])))

    # RGB -> HSV
    input_im = cv2.cvtColor(input_im.astype(np.uint8), cv2.COLOR_RGB2HSV)

    # apply random sigma gaussian blur
    input_im = np.clip(input_im.astype(np.float32) + cv2.randn(input_im.astype(np.float32), means, sigmas), a_min=0, a_max=255).astype(np.uint8)

    # HSV -> RGB
    input_im = cv2.cvtColor(input_im, cv2.COLOR_HSV2RGB)

    return input_im, output


# quite slow -> Don't use this! Not optimized and doesn't fit our problem!
def add_affine_transform2(input_im, output, max_deform):
    random_20 = uniform(-max_deform, max_deform, 2)
    random_80 = uniform(1 - max_deform, 1 + max_deform, 2)

    mat = np.array([[1, 0, 0],
                    [0, random_80[0], random_20[0]],
                    [0, random_20[1], random_80[1]]]
                   )
    input_im[:, :, :, 0] = affine_transform(input_im[:, :, :, 0], mat, output_shape=np.shape(input_im[:, :, :, 0]))
    output[:, :, :, 0] = affine_transform(output[:, :, :, 0], mat, output_shape=np.shape(input_im[:, :, :, 0]))
    output[:, :, :, 1] = affine_transform(output[:, :, :, 1], mat, output_shape=np.shape(input_im[:, :, :, 0]))

    output[output < 0.5] = 0
    output[output >= 0.5] = 1

    return input_im, output


"""
###
input_im:		input image, 5d ex: (1,64,256,256,1) , (dimi0, z, x, y, channel)
output:			ground truth, 5d ex: (1,64,256,256,2), (dimi0, z, x, y, channel)
max_shift:		the maximum amount th shift in a direction, only shifts in x and y dir
###
"""


# faster and sexier? - mvh André :)
def add_shift2(input_im, output, max_shift):
    # randomly choose which shift to set for each axis (within specified limit)
    sequence = [round(uniform(-max_shift, max_shift)), round(uniform(-max_shift, max_shift)), 0]

    # apply shift to RGB-image
    input_im = shift(input_im.copy(), sequence, order=0, mode='constant', cval=1) # <- pad with "white"
    
    output[..., 0] = shift(output.copy()[..., 0], sequence[:-1], order=0, mode='constant', cval=0)
    output[..., 1] = shift(output.copy()[..., 1], sequence[:-1], order=0, mode='constant', cval=1)

    return input_im, output


"""
####
input_im:		input image, 5d ex: (1,64,256,256,1) , (dimi0, z, x, y, channel)
output:			ground truth, 5d ex: (1,64,256,256,2), (dimi0, z, x, y, channel)
min/max_angle: 	minimum and maximum angle to rotate in deg, positive integers/floats.
####
"""


# -> Only apply rotation in image plane -> faster and unnecessairy to rotate xz or yz

def add_rotation2(input_im, output, max_angle):
    # randomly choose how much to rotate for specified max_angle
    angle_xy = round(uniform(-max_angle, max_angle))

    # rotate chunks
    input_im[..., 0] = rotate(input_im[..., 0], angle_xy, axes=(1, 2), reshape=False, mode='constant', order=1)

    output[..., 1] = rotate(output[..., 1], angle_xy, axes=(1, 2), reshape=False, mode='constant', cval=0,
                                order=1)
    output[..., 1][output[..., 0] <= 0.5] = 0
    output[..., 1][output[..., 0] > 0.5] = 1
    output[..., 0] = 1 - output[..., 1]
    output = output.astype(np.uint8)

    # output[:, :, :, 1] = rotate(output[:, :, :, 1], angle_xy, axes = (1,2), reshape = False, mode = 'constant', cval = 0, order = 1).astype(np.uint8)

    return input_im, output


"""
flips the array along random axis, no interpolation -> super-speedy :)
"""


def add_flip2(input_im, output):
    # randomly choose whether or not to flip
    if (random_integers(0, 1) == 1):
        # randomly choose which axis to flip against
        flip_ax = random_integers(0, high=1)

        # flip CT-chunk and corresponding GT
        input_im = np.flip(input_im, flip_ax)
        output = np.flip(output, flip_ax)

    return input_im, output


"""
performs intensity transform on the chunk, using gamma transform with random gamma-value
"""


def add_gamma2(input_im, output, r_limits):
    # limits
    r_min, r_max = r_limits

    # randomly choose gamma factor
    r = uniform(r_min, r_max)

    # RGB: float [0,1] -> uint8 [0,1]
    #input_im = (np.round(255. * input_im.copy())).astype(np.uint8)

    # RGB -> HSV
    input_im = cv2.cvtColor(input_im.astype(np.float32), cv2.COLOR_RGB2HSV).astype(np.float32)

    input_im[..., 2] = np.clip(np.round(input_im[..., 2] ** r), a_min=0, a_max=255)

    # HSV -> RGB
    input_im = cv2.cvtColor(input_im.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # need to normalize again after augmentation
    #input_im = (input_im.astype(np.float32) / 255)

    return input_im, output


def add_scaling2(input_im, output, r_limits):

    min_scaling, max_scaling = r_limits
    scaling_factor = np.random.uniform(min_scaling, max_scaling)

    def crop_or_fill(image, shape):
        image = np.copy(image)
        for dimension in range(2):
            if image.shape[dimension] > shape[dimension]:
                # Crop
                if dimension == 0:
                    image = image[:shape[0], :]
                elif dimension == 1:
                    image = image[:, :shape[1], :]
            else:
                # Fill
                if dimension == 0:
                    new_image = np.zeros((shape[0], image.shape[1], shape[2]))
                    new_image[:image.shape[0], :, :] = image
                elif dimension == 1:
                    new_image = np.zeros((shape[0], shape[1], shape[2]))
                    new_image[:, :image.shape[1], :] = image
                image = new_image
        return image

    input_im = crop_or_fill(scipy.ndimage.zoom(input_im.astype(np.float32), [scaling_factor,scaling_factor,1], order=1), input_im.shape)
    output = crop_or_fill(scipy.ndimage.zoom(output.astype(np.float32), [scaling_factor, scaling_factor, 1], order=0), output.shape)

    return input_im, output


"""
performs intensity transform on the chunk, using gamma transform with random gamma-value
"""


def add_brightness_mult2(input_im, output, r_limits):
    # limits
    r_min, r_max = r_limits

    # randomly choose multiplication factor
    r = uniform(r_min, r_max)

    # RGB: float [0,1] -> uint8 [0,1]
    #input_im = (np.round(255. * input_im.copy())).astype(np.uint8)

    # RGB -> HSV
    input_im = cv2.cvtColor(input_im.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

    input_im[..., 2] = np.clip(np.round(input_im[..., 2] * r), a_min=0, a_max=255)

    # HSV -> RGB
    input_im = cv2.cvtColor(input_im.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # need to normalize again after augmentation
    #input_im = (input_im.astype(np.float32) / 255)

    return input_im, output


def add_HEstain2(input_im, output):
    # RGB: float [0,1] -> uint8 [0,1] to use staintools
    #input_im = (np.round(255. * input_im.astype(np.float32))).astype(np.uint8)
    # input_im = input_im.astype(np.uint8)

    # standardize brightness (optional -> not really suitable for augmentation?
    # input_im = staintools.LuminosityStandardizer.standardize(input_im)

    # define augmentation algorithm -> should only do this the first time!
    if not 'augmentor' in globals():
        global augmentor
        # input_im = input_im[...,::-1]
        # augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2) # <- best, but slow
        augmentor = staintools.StainAugmentor(method='macenko', sigma1=0.1, sigma2=0.1)  # <- faster but worse

    # fit augmentor on current image
    augmentor.fit(input_im.astype(np.uint8))

    # extract augmented image
    input_im = augmentor.pop()

    #input_im = input_im.astype(np.float32) / 255.

    return input_im, output


'''
def add_HEstain2_all(input_im, output):
	input_shape = input_im.shape

	# for each image in stack -> transform to RGB uint8
	for i in range(input_im.shape[0]):
		input_im[i] *= 255
	input_im = input_im.astype(np.uint8)

	# define augmentation algorithm -> should only do this the first time!
	if not 'augmentor' in globals():
		global augmentor
		augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2)

	# apply augmentation on all slices
	augmentor.fit(input_im)

	# for each image extract augmented images
	input_out = np.zeros(input_shape)
	#for i in range()
'''


def add_rotation2_ll(input_im, output):
    # randomly choose rotation angle: 0, +-90, +,180, +-270
    k = random_integers(0, high=3)  # -> 0 means no rotation

    # rotate
    input_im = np.rot90(input_im, k)
    output = np.rot90(output, k)

    return input_im, output


def add_hsv2(input_im, output, max_shift):
    # RGB: float [0,1] -> uint8 [0,1]
    #input_im = (np.round(255. * input_im.copy())).astype(np.uint8)

    # RGB -> HSV
    input_im = cv2.cvtColor(input_im.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

    ## augmentation, only on Hue and Saturation channel
    # hue
    input_im[..., 0] = np.mod(input_im[..., 0] + round(uniform(-max_shift, max_shift)), 180)

    # saturation
    input_im[..., 1] = np.clip(input_im[..., 1] + round(uniform(-max_shift, max_shift)), a_min=0, a_max=255)

    # input_im = (np.round(255*maxminscale(input_im.astype(np.float32)))).astype(np.uint8)

    # input_im = np.round(input_im).astype(np.uint8)
    input_im = input_im.astype(np.uint8)

    # HSV -> RGB
    input_im = cv2.cvtColor(input_im, cv2.COLOR_HSV2RGB)

    # need to normalize again after augmentation
    #input_im = (input_im.astype(np.float32) / 255)

    return input_im, output


def add_jpeg2(input_im, output, params):
    # augmentation parameters: (default: min_comp = 10, max_comp = 50, prob = 0.5)
    min_comp = params[0]
    max_comp = params[1]
    prob = params[2]

    # randomly select which compression
    comp_val = int(np.random.randint(min_comp, max_comp, 1))
    #print(comp_val)

    # randomly to compression or not
    comp_it = np.random.rand() <= prob

    col_dim = input_im.shape[-1]
    if not comp_it:
        return input_im, output
    elif col_dim == 1:
        mode = 'L'
        image = PIL.Image.fromarray(np.squeeze(input_im, axis=-1).astype(np.uint8), mode)
    elif col_dim == 3:
        mode = 'RGB'
        image = PIL.Image.fromarray(input_im.astype(np.uint8), mode)
    else:
        raise ValueError('Unsupported nr of channels in JPEGCompression transform')

    with BytesIO() as f:
        image.save(f, format='JPEG', quality=100-comp_val)
        f.seek(0)
        image_jpeg = PIL.Image.open(f)
        result = np.asarray(image_jpeg).astype(np.float32)
        result = result.copy()
    return result.reshape(input_im.shape), output


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


def batch_gen2(file_list, batch_size, aug={}, epochs=1, input_shape=(256, 256, 3), nb_classes=2):
    for i in range(epochs):
        # for each epoch, clear batch and reset
        batch = 0
        im = np.zeros((batch_size, *input_shape), dtype=np.float32)
        gt = np.zeros((batch_size, *input_shape[:-1], nb_classes), dtype=np.uint8)

        np.random.shuffle(file_list)

        for filename in file_list:
            with h5py.File(filename, 'r') as file:
                input_im = np.array(file['image'], dtype=np.float32)
                if nb_classes == 2:
                    output = np.array(file['binary_mask'], dtype=np.float32)
                elif nb_classes == 6:
                    output = np.array(file['multiclass_mask'], dtype=np.float32)
                else:
                    raise Exception("please set nb_classes to either {2, 6}")

            # apply specified agumentation on both image stack and ground truth
            if 'stain' in aug:  # <- do this first
                input_im, output = add_HEstain2(input_im, output)

            if 'hsv' in aug:  # <- do this first
                input_im, output = add_hsv2(input_im, output, aug['hsv'])

            if 'gamma' in aug:
                input_im, output = add_gamma2(input_im, output, aug['gamma'])

            if 'mult' in aug:
                input_im, output = add_brightness_mult2(input_im, output, aug['mult'])

            if 'gauss' in aug:
                input_im, output = add_gaussian_blur2(input_im, output, aug['gauss'])

            if 'jpeg' in aug:
                input_im, output = add_jpeg2(input_im, output, aug['jpeg'])

            if 'rotate' in aug:  # -> do this last maybe?
                input_im, output = add_rotation2(input_im, output, aug['rotate'])

            if 'affine' in aug:
                input_im, output = add_affine_transform2(input_im, output, aug['affine'])

            if 'scale' in aug:
                input_im, output = add_scaling2(input_im, output, aug['scale'])

            if 'flip' in aug:
                input_im, output = add_flip2(input_im, output)

            if 'rotate_ll' in aug:
                input_im, output = add_rotation2_ll(input_im, output)

            if 'shift' in aug:
                input_im, output = add_shift2(input_im, output, aug['shift'])

            '''
            fig, ax = plt.subplots(1,2, figsize=(10,10))
            ax[0].imshow(input_im.astype(np.uint8))
            ax[1].imshow(output[..., 1], cmap="gray")
            plt.show()
            '''

            # normalize at the end
            input_im = input_im.astype(np.float32) / 255.

            # insert augmented image and GT into batch
            im[batch] = input_im
            gt[batch] = output

            del input_im, output

            batch += 1
            if batch == batch_size:
                # reset and shuffle batch in the end
                batch = 0
                yield im, gt