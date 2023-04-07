from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QGraphicsScene

from .ui_main import Ui_MainWindow
from .ui_sam_window import Ui_SAMWindow
from .ui_functions import UIFunctions
from .app_settings import Settings

from mmseg.apis import inference_segmentor, init_segmentor

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from skimage.morphology import skeletonize

import pydensecrf.densecrf as dcrf 
import numpy as np

import skimage.morphology

from .utils import cvtPixmapToArray

from segment_anything import sam_model_registry, SamPredictor

class SAMWindow(QMainWindow, UIFunctions):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_SAMWindow()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        self.settings = Settings()

        self.uiDefinitions()

        # add qlabels to scroll area

    def resizeEvent(self, event):
        self.resize_grips()

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()

    def setScene(self, pixmap, color_pixmap, scale=1.0):
        """
        Set the scene of the image
        Args:
            pixmap (QPixmap): The pixmap of the image.
            color_pixmap (QPixmap): The pixmap of the color image.
            scale (float): The scale of the scene.
        """
        self.scene = QGraphicsScene()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.color_pixmap_item = self.scene.addPixmap(color_pixmap)
        self.ui.graphicsView.setScene(self.scene)
        self.scaleScene(scale=scale)

    def scaleScene(self, scale=1.0):
        """
        Scale the scene
        Args:
            scale (float): The scale of the scene.
        """
        self.ui.graphicsView.setFixedSize(scale * self.pixmap_item.pixmap().size())
        self.ui.graphicsView.fitInView(self.pixmap_item)



class DNNFunctions(object):
    def __init__(self):

        if not hasattr(self, 'ui'):
            QMainWindow.__init__(self)
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)

        self.SAMWindow = SAMWindow()

        self.mmseg_config = 'D:/chalk/dnn/checkpoints/cgnet_2048x2048_60k_CrackAsCityscapes.py'
        self.mmseg_checkpoint = 'D:/chalk/dnn/checkpoints/iter_60000.pth'
        self.sam_checkpoint = 'D:\CHALK\dnn\checkpoints\sam_vit_h_4b8939.pth'

        self.scale = 1.0

    
    def load_sam(self, checkpoint, mode='default'):
        """
        Load the sam model
        Args:
            mode (str): The mode of the sam model.
        """
        self.sam_model = sam_model_registry[mode](checkpoint=checkpoint)
        self.sam_model.to(device='cuda:0')
        self.sam_predictor = SamPredictor(self.sam_model)
        self.set_sam_image()

        
    def set_sam_image(self):
        image = cvtPixmapToArray(self.pixmap)
        image = image[:, :, :3]
        
        self.sam_predictor.set_image(image)


    def load_mmseg(self, config_file, checkpoint_file):
        """
        Load the mmseg model
        Args:
            config_file (str): The path to the config file.
            checkpoint_file (str): The path to the checkpoint file.
        """
        self.mmseg_model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    def inference_mmseg(self, img, do_crf=True):
        """
        Inference the image with the mmseg model

        Args:
            img (np.ndarray): The image to be processed.
            do_crf (bool): Whether to apply DenseCRF.

        Returns:
            mask (np.ndarray): The processed mask.

        """

        img = self.cvtRGBATORGB(img)

        result = inference_segmentor(self.mmseg_model, img)

        mask = result[0]

        if do_crf:
            crf = self.applyDenseCRF(img, mask)
            skel = skeletonize(mask)

            crf[skel] = 1
            mask = crf

        mask = skimage.morphology.binary_closing(mask, skimage.morphology.square(3))

        return mask

    @staticmethod
    def applyDenseCRF(img, label, num_iter=3):
        """
        Apply DenseCRF to the image and label

        Args:
            img (np.ndarray): The image to be processed.
            label (np.ndarray): The label to be processed.
            num_iter (int): The number of iterations.

        Returns:
            label (np.ndarray): The processed label.
        """
        num_labels = np.max(label) + 1

        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], num_labels)

        U = unary_from_labels(label, num_labels, gt_prob=0.7, zero_unsure=False)

        d.setUnaryEnergy(U)

        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(5, 5), schan=(5, 5, 5),
                                            img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = d.inference(num_iter)

        MAP = np.argmax(Q, axis=0)

        return MAP.reshape((img.shape[0], img.shape[1]))

    
    @staticmethod
    def cvtRGBATORGB(img):
        """Convert a RGBA image to a RGB image
        Args:
            img (np.ndarray): The image to be converted.

        Returns:
            img (np.ndarray): The converted image.
        
        """
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    

    


    