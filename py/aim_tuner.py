####################################################################
####################################################################
from enum import Enum

class xform_nickname(Enum):
    XFORM0 = "Source"
    XFORM1 = "GrayScale"
    XFORM2 = "Threshold"
    XFORM3 = "Noise"
    XFORM4 = "Sharpen"
    XFORM5 = "Contrast"


####################################################################
from types import SimpleNamespace
import cv2

tuner = SimpleNamespace(

    # source image
    source_image_path='/content/gdrive/MyDrive/AIM/citizen_1864_rescan.jpg',
    # transform prefix
    xform_prefix_list=['xform0', 'xform1', 'xform2', 'xform3', 'xform4', 'xform5'],

    # pipeline file paths
    # gdrive_path = '/content/gdrive/MyDrive/AIM'
    source_base='citizen_',

    # RATING: percent prediction exact & partial match to groundtruth
    # noise -> xform3a with top percent->79.30
    # grayscale -> xform1c with top percent->78.19
    # source -> (no transform) xform0 with top percent->77.75
    # contrast -> xform5a with top percent->77.31
    # threshold -> xform2a with top percent->61.45
    # sharpen ->xform4b with top percent->55.50

    # XFORM1 = "GrayScale"
    grayscale_intensity_A=0.0,  # 77.53
    grayscale_intensity_B=0.5,  # 77.97
    grayscale_intensity_C=1.3,  # xform1c with top percent->78.19
    grayscale_intensity_D=1.8,  # 78.19

    # XFORM2 = "Threshold"
    thresh_val_A=127, thresh_type_A=cv2.THRESH_BINARY,      # xform5a with top percent->61.45
    thresh_val_B=50, thresh_type_B=cv2.THRESH_BINARY,       # 11.45
    thresh_val_C=127, thresh_type_C=cv2.THRESH_TRUNC,       # 61.45
    thresh_val_D=75, thresh_type_D=cv2.THRESH_TOZERO_INV,   # 30.61

    # XFORM3 = "Noise"
    noise_kernel_size_A=3,      # xform3a with top percent->79.30
    noise_kernel_size_B=7,      # 77.31
    noise_kernel_size_C=21,     # 45.59
    noise_kernel_size_D=31,     # 2.86

    # XFORM4 = "Sharpen"
    sharpen_kernel_size_A=3,    # 16.07
    sharpen_kernel_size_B=7,    # xform4b with top percent->55.51
    sharpen_kernel_size_C=21,   # 4.18
    sharpen_kernel_size_D=31,   # 53.08

    # XFORM5 = "Contrast"
    contrast_factor_A=1, brightness_factor_A=0,     # xform5a with top percent->77.31
    contrast_factor_B=1, brightness_factor_B=32,    # 76.43
    contrast_factor_C=2, brightness_factor_C=32,    # 27.53
    contrast_factor_D=2, brightness_factor_D=64,    # 32.15

)
#print(f"{aim_tuner}")
####################################################################