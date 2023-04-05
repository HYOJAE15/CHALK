import os

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2,40).__str__()

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import  QPixmap, QColor, QPainter
from PySide6.QtWidgets import (
    QMainWindow, QFileSystemModel, QLabel, QGraphicsScene, QFileDialog, QGridLayout, QWidget
) 

from .ui_thumbnail_window import Ui_ThumbnailWindow
from .ui_main import Ui_MainWindow
from .ui_functions import UIFunctions
from .ui_brush_menu import Ui_BrushMenu
from .app_settings import Settings
from .dnn_functions import DNNFunctions

from .utils import imread, cvtArrayToQImage, cvtPixmapToArray, convertLabelToColorMap

from timeit import default_timer as timer
from numba import njit

import skimage.measure
import skimage.filters


@njit(fastmath=True)
def _applyBrushSize(brushSize):

    width = int(brushSize // 2)

    _X, _Y = [], []

    if brushSize % 2 == 0:
        for _x in range(-width, width):
            for _y in range(-width,width):
                _X.append(_x)
                _Y.append(_y)
    else:
        for _x in range(-width, width+1):
            for _y in range(-width, width+1):
                _X.append(_x)
                _Y.append(_y)

    return _X, _Y, width


def applyBrushSize(X, Y, brushSize, max_x, max_y, brushType = 'rectangle'): 

    _X, _Y, width = _applyBrushSize(brushSize)
    
    if brushType == 'circle' :
        
        _X, _Y = convetRectangleToCircle(_X, _Y, width)
        
        # dist = [np.sqrt(_x**2 + _y**2) for _x, _y in zip(_Y, _X)]
        # _Y =  [_y for idx, _y in enumerate(_Y) if dist[idx] < width]
        # _X = [_x for idx, _x in enumerate(_X) if dist[idx] < width]
        # _X, _Y = np.array(_X), np.array(_Y)

    return_x = []
    return_y = []
    
    for x, y in zip(X, Y):
        _x = x + _X
        _y = y + _Y

        return_x += _x.tolist()
        return_y += _y.tolist()

    return_x = np.array(return_x)
    return_y = np.array(return_y)

    return_x = np.clip(return_x, 0, max_x-1)
    return_y = np.clip(return_y, 0, max_y-1)

    _return = np.vstack((return_x, return_y))
    _return = np.unique(_return, axis=1)
    return_x , return_y = _return[0, :], _return[1, :]
    
    return return_x, return_y

@njit(fastmath=True)
def convetRectangleToCircle(X, Y, width):
    
    dist = [np.sqrt(_x**2 + _y**2) for _x, _y in zip(Y, X)]
    Y =  [_y for idx, _y in enumerate(Y) if dist[idx] < width]
    X = [_x for idx, _x in enumerate(X) if dist[idx] < width]
    return np.array(X), np.array(Y)

@njit(fastmath=True)
def fast_coloring(X, Y, array, label_palette, brush_class, alpha = 50):
    # print(X, Y)
    
    for x, y in zip(X, Y): 
        array[y, x, :3] = label_palette[brush_class]
        array[y, x, 3] = alpha

    return array 


def getScaledPoint_v2(event, scale):

    # scaled_event_pos = QPoint(round(event.pos().x() / scale), round(event.pos().y() / scale))
    x, y = round(event.x() / scale), round(event.y() / scale)

    return x, y 

def points_between(x1, y1, x2, y2):
    """
    coordinate between two points
    """

    d0 = x2 - x1
    d1 = y2 - y1
    
    count = max(abs(d1)+1, abs(d0)+1)

    if d0 == 0:
        return (
            np.full(count, x1),
            np.round(np.linspace(y1, y2, count)).astype(np.int32)
        )

    if d1 == 0:
        return (
            np.round(np.linspace(x1, x2, count)).astype(np.int32),
            np.full(count, y1),  
        )

    return (
        np.round(np.linspace(x1, x2, count)).astype(np.int32),
        np.round(np.linspace(y1, y2, count)).astype(np.int32)
    )


class BrushMenuWindow(QMainWindow, UIFunctions):
    def __init__(self):
        QMainWindow.__init__(self)

        self.ui = Ui_BrushMenu()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        self.settings = Settings()

        self.uiDefinitions()

        # RESIZE EVENTS
    # ///////////////////////////////////////////////////////////////
    def resizeEvent(self, event):
        # Update Size Grips
        self.resize_grips()

    # MOUSE CLICK EVENTS
    # ///////////////////////////////////////////////////////////////
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')


class ThumbnailGridWindow(QMainWindow, UIFunctions):
    def __init__(self):
        QMainWindow.__init__(self)

        self.ui = Ui_ThumbnailWindow()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        self.settings = Settings()

        self.uiDefinitions()

    def resizeEvent(self, event):
        # Update Size Grips
        self.resize_grips()

    # MOUSE CLICK EVENTS
    # ///////////////////////////////////////////////////////////////
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
            event_global = self.ui.pagesContainer.mapFromGlobal(event.globalPos())
            print(event_global.x(), event_global.y())

            
            height = self.ui.pagesContainer.height()
            row_height = height // self.ui.gridLayout.rowCount()

            width = self.ui.pagesContainer.width()
            col_width = width // self.ui.gridLayout.columnCount()

            
            self.ui.gridLayout.itemAtPosition()
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')


class ImageFunctions(DNNFunctions):
    def __init__(self):
        DNNFunctions.__init__(self)

        if not hasattr(self, 'ui'):
            QMainWindow.__init__(self)
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)
            
        self.ui.treeView.clicked.connect(self.openImage)
        self.fileModel = QFileSystemModel()
        self.alpha = 50
        self.scale = 1
        self.oldPos = None
        self.brush_class = 1
        
        self.ui.mainImageViewer.mouseMoveEvent = self.brushMoveEvent
        self.ui.mainImageViewer.mousePressEvent = self.brushPressPoint
        self.ui.mainImageViewer.mouseReleaseEvent = self.brushReleasePoint

        self.ControlKey = False

        self.brushSize = 10

        self.ui.brushButton.clicked.connect(self.openBrushMenu)

        self.BrushMenu = BrushMenuWindow()
        self.BrushMenu.ui.brushSizeSlider.valueChanged.connect(self.changeBrushSize)

        self.ui.addImageButton.clicked.connect(self.addNewImage)
        self.ui.deleteImageButton.clicked.connect(self.deleteImage)

        self.use_brush = True


        """
        Autolabel Tool
        """
        self.ui.autoLabelButton.clicked.connect(self.checkAutoLabelButton)
        self.use_autolabel = False

        """
        Label Enhancement Tool
        """
        self.ui.enhancementButton.clicked.connect(self.checkEnhancementButton)
        self.use_enhancement = False

        """
        Label GrapCut Tool
        """

        self.ui.grabCutButton.clicked.connect(self.checkGrabCutButton)
        self.use_grabcut = False

    
    def useEnhancement(self, event):
        """
        Label Enhancement Tool
        """
        
        event_global = self.ui.mainImageViewer.mapFromGlobal(event.globalPos())

        x, y = getScaledPoint_v2(event_global, self.scale)
        
        if (self.x != x) or (self.y != y) : 

            if self.x > x :
                min_x = x
                max_x = self.x
            else :
                min_x = self.x
                max_x = x

            if self.y > y :
                min_y = y
                max_y = self.y
            else :
                min_y = self.y
                max_y = y
    
            print(min_y, min_x, max_y, max_x)

            # get the region of interest
            img = cvtPixmapToArray(self.pixmap)
            img_roi = img[min_y:max_y, min_x:max_x, :3]
            label_roi = self.label[min_y:max_y, min_x:max_x]
            label_roi = label_roi.astype(np.uint8)

            label_roi = self.applyDenseCRF(img_roi, label_roi)

            self.label[min_y:max_y, min_x:max_x] = label_roi

            # update colormap
            self.updateColorMap()

        self.x = x
        self.y = y
        
    def enhanceLabel(self):
        """
        Label Enhancement Tool
        """
        # get the current image
        img = cvtPixmapToArray(self.pixmap)

        # get binary image of current label
        current_label = self.label == self.brush_class
        print(self.brush_class)

        # loop through each object in the label
        labeled = skimage.measure.label(current_label)

        for region in skimage.measure.regionprops(labeled):

            # get the bounding box coordinates
            min_y, min_x, max_y, max_x = region.bbox
            print(min_y, min_x, max_y, max_x)

            # get the region of interest
            img_roi = img[min_y:max_y, min_x:max_x, :3]
            label_roi = current_label[min_y:max_y, min_x:max_x]
            label_roi = label_roi.astype(np.uint8)

            # print the shape of the image
            print(img_roi.shape)
            print(label_roi.shape)
            print(label_roi.dtype)

            # run the enhancement algorithm
            label_roi = self.applyDenseCRF(img_roi, label_roi)

            # remain only the largest object
            label_roi = skimage.measure.label(label_roi)
            label_roi = label_roi == np.argmax(np.bincount(label_roi.flat)[1:]) + 1

            # smooth the label
            label_roi = skimage.morphology.binary_closing(label_roi, skimage.morphology.square(3))

            # update the label
            self.label[min_y:max_y, min_x:max_x] = label_roi * self.brush_class


        # update colormap
        self.updateColorMap()

    
    def checkAutoLabelButton(self):
        """
        Enable or disable auto label button
        """
        self.use_autolabel = True

        if self.use_brush:
            self.use_brush = False 

        if self.use_enhancement:
            self.use_enhancement = False


    def checkEnhancementButton(self):
        """
        Enable or disable enhancement button
        """
        self.use_enhancement = True

        if self.use_brush:
            self.use_brush = False

        if self.use_autolabel:
            self.use_autolabel = False


    def checkBrushButton(self):
        """
        Enabale or disable brush
        """
        self.use_brush = True

        if self.use_autolabel == True :
            self.use_autolabel = False 

        if self.use_enhancement == True :
            self.use_enhancement = False


    def checkGrabCutButton(self):
        """
        Enable or disable grabcut button
        """
        self.use_grabcut = True

        if self.use_brush:
            self.use_brush = False

        if self.use_autolabel:
            self.use_autolabel = False

        if self.use_enhancement:
            self.use_enhancement = False

        
    def openBrushMenu(self):
        self.BrushMenu.show()
        
        self.checkBrushButton()

    def changeBrushSize(self, value):
        self.brushSize = value
        self.BrushMenu.ui.brushSizeText.setText(str(value))


    def deleteImage(self, event):
        self.currentIndex = self.ui.treeView.currentIndex().data(QFileSystemModel.FilePathRole)
        self.imgFolderPath, filename = os.path.split(self.currentIndex)

        img_path = self.currentIndex
        label_path = self.convertImagePathToLabelPath(img_path)
        os.remove(img_path)
        os.remove(label_path)
        
        self.ui.treeView.model().removeRow(self.ui.treeView.currentIndex().row(), self.ui.treeView.currentIndex().parent())


    @staticmethod
    def convertImagePathToLabelPath(img_path):
        img_label_folder = img_path.replace('/leftImg8bit/', '/gtFine/')
        img_label_folder = img_label_folder.replace( '_leftImg8bit.png', '_gtFine_labelIds.png')
        return img_label_folder
    
    def addNewImage(self, event):

        self.currentIndex = self.ui.treeView.currentIndex().data(QFileSystemModel.FilePathRole)
        
        self.imgFolderPath, filename = os.path.split(self.currentIndex)

        if 'leftImg8bit.png' not in filename:
            self.imgFolderPath = self.currentIndex
        
        img_save_folder = self.imgFolderPath
        img_save_folder = img_save_folder.replace( '_leftImg8bit.png', '')  
    
        img_label_folder = img_save_folder.replace('/leftImg8bit/', '/gtFine/')
        img_label_folder = img_label_folder.replace( '_leftImg8bit.png', '')

        readFilePath = QFileDialog.getOpenFileNames(
                caption="Add images to current working directory", filter="Images (*.png *.jpg *.tiff)"
                )
        images = readFilePath[0]

        for img in images:
                
            temp_img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            img_filename = os.path.basename(img) # -> basename is file name
            img_filename = img_filename.replace(' ', '')
            img_filename = img_filename.replace('.jpg', '.png')
            img_filename = img_filename.replace('.JPG', '.png')
            img_filename = img_filename.replace('.tiff', '.png')
            img_filename = img_filename.replace('.png', '_leftImg8bit.png')

            img_gt_filename = img_filename.replace( '_leftImg8bit.png', '_gtFine_labelIds.png')
            gt = np.zeros((temp_img.shape[0], temp_img.shape[1]), dtype=np.uint8)

            _, org_img = cv2.imencode(".png", temp_img)
            org_img.tofile(os.path.join(img_save_folder, img_filename))

            _, gt_img = cv2.imencode(".png", gt)
            gt_img.tofile(os.path.join(img_label_folder, img_gt_filename))

        self.resetTreeView(refreshIndex=True)

        
    def resetTreeView(self, refreshIndex = False):
        """Reset the tree view
        Args:
            refreshIndex (bool, optional): Refresh the current index. Defaults to False.
        """
        
        self.ui.treeView.reset()
        self.fileModel = QFileSystemModel()
        _imgFolderPath = self.imgFolderPath.replace('/train', '')
        _imgFolderPath = _imgFolderPath.replace('/test', '')
        _imgFolderPath = _imgFolderPath.replace('/val', '')

        self.fileModel.setRootPath(_imgFolderPath)
        
        self.ui.treeView.setModel(self.fileModel)
        self.ui.treeView.setRootIndex(self.fileModel.index(_imgFolderPath))

        if refreshIndex:
            self.ui.treeView.setCurrentIndex(self.fileModel.index(self.currentIndex))


    def readImageToPixmap(self, path):
        """Read image to pixmap
        Args:
            path (str): Image path
        Returns:
            QPixmap: Image pixmap
        """
        img = imread(path)
        return QPixmap(cvtArrayToQImage(img))

    
    def openImage(self, index):
        
        self.imgPath = self.fileModel.filePath(index)

        self.labelPath = self.imgPath.replace('/leftImg8bit/', '/gtFine/')
        self.labelPath = self.labelPath.replace( '_leftImg8bit.png', '_gtFine_labelIds.png')

        self.pixmap = self.readImageToPixmap(self.imgPath)        
        
        self.label = imread(self.labelPath)

        # if self.label.shape[0] * self.label.shape[1] < 10000 * 10000:
        self.colormap = convertLabelToColorMap(self.label, self.label_palette, self.alpha)
        self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        
        self.scene = QGraphicsScene()
        self.pixmap_item = self.scene.addPixmap(self.pixmap)

        self.color_pixmap_item = self.scene.addPixmap(self.color_pixmap)

        self.ui.mainImageViewer.setScene(self.scene)

        self.scale = self.ui.scrollAreaImage.height() / self.label.shape[0]
        self.ui.mainImageViewer.setFixedSize(self.scale * self.color_pixmap.size())
        self.ui.mainImageViewer.fitInView(self.pixmap_item)
        
    def useGrabCut(self, event):

        """
        Label Enhancement Tool
        """
        import cv2
        
        event_global = self.ui.mainImageViewer.mapFromGlobal(event.globalPos())

        x, y = getScaledPoint_v2(event_global, self.scale)
        
        if (self.x != x) or (self.y != y) : 

            if self.x > x :
                min_x = x
                max_x = self.x
            else :
                min_x = self.x
                max_x = x

            if self.y > y :
                min_y = y
                max_y = self.y
            else :
                min_y = self.y
                max_y = y
    
            print(min_y, min_x, max_y, max_x)

            # get the region of interest
            img = cvtPixmapToArray(self.pixmap)
            img_roi = img[min_y:max_y, min_x:max_x, :3]
            label_roi = self.label[min_y:max_y, min_x:max_x]
            label_roi = label_roi.astype(np.uint8)

            mask = label_roi == self.brush_class
            
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)

            mask, bgdModel, fgdModel = cv2.grabCut(img_roi, mask, None, bgdModel, fgdModel,5, cv2.GC_INIT_WITH_MASK)

            mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

            self.label[min_y:max_y, min_x:max_x] = mask * self.brush_class

            # update colormap
            self.updateColorMap()

        self.x = x
        self.y = y

        
    def brushReleasePoint(self, event):

        if self.use_enhancement :
            self.useEnhancement(event)

        if self.use_grabcut : 
            self.useGrabCut(event)

        else: 
            self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))
            self.color_pixmap_item.setPixmap(QPixmap())
            self.color_pixmap_item.setPixmap(self.color_pixmap)
            

    
    def brushMoveEvent(self, event):

        if self.use_brush : 
            self.useBrush(event)

        

    
    def brushPressPoint(self, event):
        """
        Get the brush class and the point where the mouse is pressed
        """

        # Get the brush class
        self.brush_class = self.ui.classList.currentRow()

        event_global = self.ui.mainImageViewer.mapFromGlobal(event.globalPos())
        x, y = getScaledPoint_v2(event_global, self.scale)

        self.x = x
        self.y = y

        if self.use_autolabel : 
            self.useAutoLabel()
        

    def useAutoLabel(self):

        # img = self.pixmap.toImage()

        
        # width, height = img.width(), img.height()
        # buffer = img.bits().asstring(width * height * 4)
        # img = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))

        img = cvtPixmapToArray(self.pixmap)

        min_x = self.x - 128
        min_y = self.y - 128
        max_x = self.x + 128
        max_y = self.y + 128

        if min_x < 0 :
            min_x = 0
        if min_y < 0 :
            min_y = 0

        if max_x > img.shape[1] :
            max_x = img.shape[1]
        
        if max_y > img.shape[0] :
            max_y = img.shape[0]

        img = img[min_y:max_y, min_x:max_x, :]
        
        result = self.dnn_inference(img, do_crf=False)

        # update label with result

        idx = np.argwhere(result == 1)
        y_idx, x_idx = idx[:, 0], idx[:, 1]
        x_idx = x_idx + min_x
        y_idx = y_idx + min_y

        self.label[y_idx, x_idx] = self.brush_class
        self.colormap[y_idx, x_idx, :3] = self.label_palette[self.brush_class]


    def updateColorMap(self):
        """
        Update the color map
        """
        self.colormap = convertLabelToColorMap(self.label, self.label_palette, self.alpha)
        self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.color_pixmap_item.setPixmap(QPixmap())
        self.color_pixmap_item.setPixmap(self.color_pixmap)


    def useBrush(self, event):

        event_global = self.ui.mainImageViewer.mapFromGlobal(event.globalPos())

        x, y = getScaledPoint_v2(event_global, self.scale)
        
        if (self.x != x) or (self.y != y) : 

            x_btw, y_btw = points_between(self.x, self.y, x, y)

            x_btw, y_btw = applyBrushSize(x_btw, y_btw, self.brushSize, self.label.shape[1], self.label.shape[0])

            self.label[y_btw, x_btw] = self.brush_class
            self.colormap[y_btw, x_btw, :3] = self.label_palette[self.brush_class]

            self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))

            self.color_pixmap_item.setPixmap(QPixmap())
            self.color_pixmap_item.setPixmap(self.color_pixmap)

        self.x = x
        self.y = y


        


