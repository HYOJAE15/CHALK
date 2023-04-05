
import sys

from PySide6.QtWidgets import QMainWindow

from .ui_main import Ui_MainWindow

from mmseg.apis import inference_segmentor, init_segmentor

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from skimage.morphology import skeletonize

import pydensecrf.densecrf as dcrf 
import numpy as np

import skimage.morphology

class DNNFunctions(object):
    def __init__(self):

        if not hasattr(self, 'ui'):
            QMainWindow.__init__(self)
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)

        config_file = 'D:/chalk/dnn/checkpoints/cgnet_2048x2048_60k_CrackAsCityscapes.py'
        checkpoint_file = 'D:/chalk/dnn/checkpoints/iter_60000.pth'

        self.dnn_model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
        print("Model is loaded.")


    def dnn_inference(self, img, do_crf=True):
        """
        Inference the image with the DNN model

        Args:
            img (np.ndarray): The image to be processed.
            do_crf (bool): Whether to apply DenseCRF.

        Returns:
            mask (np.ndarray): The processed mask.

        """

        img = self.cvtRGBATORGB(img)

        result = inference_segmentor(self.dnn_model, img)

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

    