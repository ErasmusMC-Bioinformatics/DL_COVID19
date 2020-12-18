""" RGB to one-hot vector conversion AND one-hot vector to RGB conversion """
""" Applied to ground truth segmentation masks as pre-processing function """

import numpy as np
from COVID_Tissue_Code import *

def rgb_to_onehot(rgb_arr):
    color_dict= color_dict_hemorrhage
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]
    arr = np.zeros(shape, dtype=np.int8)
    for i, cls in color_dict.items():
        mask0 = rgb_arr[:, :, 0] == cls[0]
        mask1 = rgb_arr[:, :, 1] == cls[1]
        mask2 = rgb_arr[:, :, 2] == cls[2]
        mask = np.logical_and(np.logical_and(mask0, mask1), mask2)
        arr[mask] = i

    arr_vec = np.zeros((arr.shape[0], arr.shape[1], num_classes))
    for i in range(num_classes):
        mask = (arr == i)
        arr_vec[:, :, i][mask] = 1
    return arr_vec


def onehot_to_rgb(onehot):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2] + (3,))
    for k in color_dict.keys():
        output[single_layer == k] = color_dict[k]
    return np.uint8(output)
