from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import random
from matplotlib import pyplot as plt


## [Load image]
parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
args = parser.parse_args()

src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
## [Load image]

## [Convert to grayscale]
src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
## [Convert to grayscale]

## [Apply Histogram Equalization]
dst = cv.equalizeHist(src)
## [Apply Histogram Equalization]

## [Convert to bgrimage]
dst_bgr = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
## [Convert to bgrimage]

# Calculate Image Histogram 
# hist1 = cv.calcHist([src],[0],None,[256],[0,256])
# hist2 = cv.calcHist([dst],[0],None,[256],[0,256])
# hist3 = cv.calcHist([dst_bgr],[0],None,[256],[0,256])

# plt.subplot(221),plt.imshow(src, "gray"),plt.title('src')
# plt.subplot(222),plt.imshow(dst, "gray"),plt.title('dst')
# plt.subplot(223),plt.imshow(dst_bgr, "gray"),plt.title('dst_bgr')
# plt.subplot(224),
# plt.plot(hist1,color='r')
# plt.plot(hist2,color='g')
# plt.plot(hist3,color='b')
# plt.xlim([0,256])
# plt.show()

## [Display results]
cv.imshow('Source image', src)
cv.imshow('Equalized Image', dst)
cv.imshow('Equalized_bgr Image', dst_bgr)
## [Display results]

## [Wait until user exits the program]
cv.waitKey()
## [Wait until user exits the program]