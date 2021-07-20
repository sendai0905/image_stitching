import numpy as np
from numpy import linalg as LA
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
from numba import jit, prange

img1 = cv2.imread('data/stitching/img1_1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/stitching/img1_2.png', cv2.IMREAD_GRAYSCALE)

'''
feature detectionを行い、特徴点を抽出する
↓
各点に対して最小二乗法を用いてパラメータpを求める
'''


def Gaussianfilter(size, sigma):
    x = y = np.arange(0, size) - sigma
    X, Y = np.meshgrid(x, y)

    mat = np.exp(-(X ** 2 + Y ** 2) / (2 * (sigma ** 2))) / 2 * np.pi * (sigma ** 2)

    kernel = mat / np.sum(mat)
    return kernel


def scale_space(img, octave):
    h, w = img.shape
    img_list = np.array(img).reshape(1, h, w)
    for i in range(1, 5):
        convolved_img = ndimage.convolve(img, Gaussianfilter(3, 1.6 * 2 ** (octave - 1) * np.sqrt(2) ** i), mode='constant').reshape(1, h, w)
        img_list = np.append(img_list, convolved_img, axis=0)
        # print(img_list.shape)

    return img_list


def dog(img_list):
    dog_list = np.array([])
    l, h, w = img_list.shape
    for i in range(l - 1):
        dog_img = (img_list[i + 1].astype('int16') - img_list[i].astype('int16')).reshape(1, h, w)
        dog_img = np.where(dog_img < 0, 0, dog_img)
        dog_img = dog_img.astype('uint8')
        if not i:
            dog_list = dog_img
        else:
            dog_list = np.append(dog_list, dog_img, axis=0)

    return dog_list


@jit(nopython=True)
def detect_keypoint(dog_list):
    l, h, w = dog_list.shape
    keypoint_list = []
    for i in range(1, l - 1):
        under_img = dog_list[i - 1]
        focus_img = dog_list[i]
        upper_img = dog_list[i + 1]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                upper_area = upper_img[y-1:y+2, x-1:x+2]
                under_area = under_img[y-1:y+2, x-1:x+2]
                focus_area = focus_img[y-1:y+2, x-1:x+2]
                if focus_img[y, x] == np.max(np.array([np.max(upper_area), np.max(focus_area), np.max(under_area)])):
                    keypoint_list.append((i, y, x))

    return keypoint_list


h, w = img1.shape
img1_octave1 = scale_space(img1, 1)
img1_octave2 = scale_space(cv2.resize(img1, (int(w*0.5), int(h*0.5))), 2)
img1_octave3 = scale_space(cv2.resize(img1, (int(w*0.5**2), int(h*0.5**2))), 3)
img1_octave4 = scale_space(cv2.resize(img1, (int(w*0.5**3), int(h*0.5**3))), 4)

# print(img1_octave3)

dog_img1_octave1 = dog(img1_octave1)
dog_img1_octave2 = dog(img1_octave2)
dog_img1_octave3 = dog(img1_octave3)
dog_img1_octave4 = dog(img1_octave4)

# print(dog_img1_octave1)

img1_octave1_keypoint = detect_keypoint(dog_img1_octave1)
print('octave1 done')
img1_octave2_keypoint = detect_keypoint(dog_img1_octave2)
print('octave2 done')
img1_octave3_keypoint = detect_keypoint(dog_img1_octave3)
print('octave3 done')
img1_octave4_keypoint = detect_keypoint(dog_img1_octave4)
print('octave4 done')
print(len(img1_octave1_keypoint), len(img1_octave2_keypoint), len(img1_octave3_keypoint), len(img1_octave4_keypoint))
