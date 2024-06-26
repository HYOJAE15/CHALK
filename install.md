Install CHALK on Your Desktop
=============================

Follow these steps to install CHALK on your desktop:

1\. Install Required Packages
-----------------------------

Install the necessary packages by running the following command in your terminal or command prompt:

```
pip install -r requirements.txt
```

2\. Install PyTorch Compatible with Your Desktop Environment
------------------------------------------------------------

CHALK has been tested with PyTorch 1.13. To install the appropriate version of PyTorch for your system, visit the PyTorch previous versions page:

[https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)

Here's an example installation command for CUDA 11.6:


```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

3\. Install MMSegmentation Using the OpenMIM Installer
------------------------------------------------------

To install MMSegmentation and its required dependencies, run the following commands:


```
pip install -U openmim
mim install mmcv 
mim install mmsegmentation
```

4\. Install the Segment Anything Model from the FAIR GitHub Repository
----------------------------------------------------------------------

Install the Segment Anything Model by running this command:


```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

5\. Install PyDenseCRF from Lucas Beyer' GitHub Repository
----------------------------------------------------------

Install PyDenseCRF by running this command:

```
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

6\. Setup CHALK 
---------------

To setup CHALK, run the following command:

```
python setup.py develop
```

7\. Download checkpoint files for Autolabeling Function 
-------------------------------------------------------

Checkpoints for the autolabeling function are available at the following links and should be placed in the `dnn/checkpoints` directory:

[CGNet trained for Crack Detection](https://www.dropbox.com/s/okikp25lw58jg8y/cgnet.pth?dl=0)\
[Segment Anything Model (vit_h)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

Once you've completed these steps, CHALK should be installed and ready to use on your desktop.