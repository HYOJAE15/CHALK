import os

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2,40).__str__()

import cv2
import math 

import numpy as np
import slidingwindow as sw

from numba import njit

from PySide6.QtGui import QImage


def imread(path: str) -> np.array:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if img.ndim == 3 : 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    return img

def imwrite(path:str,
            img: np.array
            )-> None: 
    _, ext = os.path.splitext(path)

    if img.ndim == 3 : 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

    _, label_to_file = cv2.imencode(ext, img)
    label_to_file.tofile(path)

def createLayersFromLabel(label: np.array, 
                          num_class: int
                          ) -> list([np.array]):
    layers = []

    for idx in range(num_class):
        layers.append(label == idx)
        
    return layers


def cvtArrayToQImage(array: np.array) -> QImage:

    if len(array.shape) == 3 : 

        h, w, c = array.shape
        if c == 3:
            return QImage(array.data, w, h, 3 * w, QImage.Format_RGB888)
        elif c == 4: 
            return QImage(array.data, w, h, 4 * w, QImage.Format_RGBA8888)

    elif len(array.shape) == 2 :
        h, w = array.shape
        return QImage(array.data, w, h, QImage.Format_Mono)

@njit
def mapLabelToColorMap(label: np.array, 
                       colormap: np.array, 
                       palette: list(list([int, int, int]))
                       )-> np.array:
    
    assert label.ndim == 2, "label must be 2D array"
    assert colormap.ndim == 3, "colormap must be 3D array"
    assert colormap.shape[2] == 4, "colormap must have 4 channels"

    for x in range(label.shape[0]):
        for y in range(label.shape[1]):
            colormap[x, y, :3] = palette[label[x, y]][:3]

    return colormap
   

def convertLabelToColorMap(
        label: np.array,
        palette: list(list([int, int, int])),
        alpha: int) -> np.array:
    assert label.ndim == 2, "label must be 2D array"
    assert alpha >= 0 and alpha <= 255, "alpha must be between 0 and 255"

    colormap = np.zeros((label.shape[0], label.shape[1], 4), dtype=np.uint8)
    colormap = mapLabelToColorMap(label, colormap, palette)
    colormap[:, :, 3] = alpha

    return colormap


def generateForNumberOfWindows(data, dimOrder, windowCount, overlapPercent, transforms=[]):
	"""
	Generates a set of sliding windows for the specified dataset, automatically determining the required window size in
	order to create the specified number of windows. `windowCount` must be a tuple specifying the desired number of windows
	along the Y and X axes, in the form (countY, countX).
	"""
	
	# Determine the dimensions of the input data
	width = data.shape[dimOrder.index('w')]
	height = data.shape[dimOrder.index('h')]
	
	# Determine the window size required to most closely match the desired window count along both axes
	countY, countX = windowCount
	windowSizeX = math.ceil(width / countX)
	windowSizeY = math.ceil(height / countY)
	
	# Generate the windows
	return sw.generateForSize(
		width,
		height,
		dimOrder,
		0,
		overlapPercent,
		transforms,
		overrideWidth = windowSizeX,
		overrideHeight = windowSizeY
	)