####################################################################
# from enum import Enum
#
class xform_nickname(Enum):
    XFORM0 = "Source"
    XFORM1 = "GrayScale"
    XFORM2 = "Threshold"
    XFORM3 = "Noise"
    XFORM4 = "Sharpen"
    XFORM5 = "Contrast"
####################################################################

####################################################################
from types import SimpleNamespace

tuner = SimpleNamespace(

    # source image
    source_image_path = '/content/gdrive/MyDrive/AIM/citizen_1864_rescan.jpg',
    # transform prefix
    transform_prefix = ['xform0', 'xform1', 'xform2', 'xform3', 'xform4', 'xform5'],

    # pipeline file paths
    #gdrive_path = '/content/gdrive/MyDrive/AIM'
    source_base = 'citizen_',
)
#print(f"{aim_tuner}")
####################################################################