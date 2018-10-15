import imreg_dft as ird
import numpy as np
import logging
from skimage import io
import sys
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegArray:
    def __init__(self, zoom, rotation, shift_x, shift_y):
        self.zoom = zoom
        self.rotation = rotation
        self.shift_x = shift_x
        self.shift_y = shift_y


def check_size(img:np.ndarray, template:np.ndarray) -> bool:
    """
    Check the shape for the arrays.
    :param img: numpy array
    :param template: numpy array
    :return: True if equal, False if different, raises AssertionError if different dimensions.
    """
    dims = (img.ndim, template.ndim)
    assert dims == (2, 2) , logger.error(f'Expected 2D arrays for both images, got {dims}')
    shape1, shape2 = img.shape, template.shape
    if shape1 == shape2:
        return True
    else:
        return False


def pad_inputs(img:np.ndarray, template:np.ndarray, mode='median') -> (np.ndarray, np.ndarray):
    """
    Pads two arrays to get the same size using np.pad function
    :param img: numpy array
    :param template: numpy array
    :param mode: same as numpy.pad, 'median' by default.
    :return: tuple with padded (img, template)
    """
    if not check_size(img,template):
        shape1, shape2 = img.shape, template.shape
        shape_out = (max(shape1[0],shape2[0]), max(shape1[1],shape2[1]))
        if shape1<shape_out:
            pad11 = (shape_out[0] - shape1[0])//2
            pad12 = shape_out[0] - shape1[0] - pad11
            pad21 = (shape_out[1] - shape1[1])//2
            pad22 = shape_out[1] - shape1[1] - pad21
            img = np.pad(array=img, pad_width=((pad11,pad12),(pad21,pad22)), mode=mode)
        if shape2<shape_out:
            pad11 = (shape_out[0] - shape2[0])//2
            pad12 = shape_out[0] - shape2[0] - pad11
            pad21 = (shape_out[1] - shape2[1])//2
            pad22 = shape_out[1] - shape2[1] - pad21
            template = np.pad(array=template, pad_width=((pad11,pad12),(pad21,pad22)), mode='median')
    return img, template


def register_imgs(main_cam_frame:np.ndarray, drift_cam_frame:np.ndarray) -> RegArray:
    if check_size(main_cam_frame, drift_cam_frame):
        i1,i2 = main_cam_frame, drift_cam_frame
    else:
        i1, i2 = pad_inputs(main_cam_frame, drift_cam_frame)
    result = ird.similarity(i1,i2)
    out = RegArray(zoom=result['scale'], rotation=result['angle'], shift_x=result['tvec'][0], shift_y=result['tvec'][1])
    return out


def register_files(path1:str, path2:str):
    f1 = io.imread(path1)
    f2 = io.imread(path2)
    arr = register_imgs(f1, f2)
    return arr


if __name__ == '__main__':
    args = ['/Volumes/Imod-grenier/Andrey/Phase retrieve/drift-correction/BFDC/bfdc/registration.py', '/Volumes/Imod-grenier/Mickael/yeast_imaging_EdU/andor_cam_transformmatrix/Pos0/img_000000000_Default0_000.tif', '/Volumes/Imod-grenier/Mickael/yeast_imaging_EdU/thorlabs_cam_bigbeads_transformmatrix/Pos0/img_000000000_Default0_000.tif']

    try:
        p1, p2 = args[1], args[2]
    except IndexError:
        print(f'Missing arguments path1, path2: {args}')

    arr = register_files(p1,p2)
    exit(0)