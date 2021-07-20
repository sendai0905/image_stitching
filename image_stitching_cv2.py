import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

img1 = cv2.imread('data/stitching/img1.png')
img2 = cv2.imread('data/stitching/img2.png')
img3 = cv2.imread('data/stitching/img3.png')

stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
status, pano = stitcher.stitch([img1, img3])

if status != cv2.Stitcher_OK:
    print("Can't stitch images, error code = %d" % status)
    sys.exit(-1)

cv2.imwrite('img1_stitched.png', pano)
