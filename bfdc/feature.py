import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter as gf
from skimage import filters
import matplotlib as mpl
mpl.use('TkAgg')
#mpl.use('PS')
import matplotlib.pyplot as plt
from bfdc.xcorr import get_abs_max
from read_roi import read_roi_file


class FeatureExtractor:
    """
    Finds a feature in calibration stack to use for drift correction
    """

    def __init__(self, cal_stack, extend = 0, invert=False):
        self.cal_stack = np.array(cal_stack,dtype='f')[:]
        self.featuremap = highlight_feature(self.cal_stack)
        self.mask = get_mask_otsu(self.featuremap,invert)
        self.labeled_mask = label_mask(self.mask)
        self.peak = get_abs_max(self.featuremap)
        self.segment = get_isolated_segment(self.labeled_mask, self.peak)
        self.boundaries = get_xy_boundaries(self.segment,extend=extend)
        self.crop_stack = crop_using_xy_boundaries(self.cal_stack, self.boundaries,extend=0)
        self.invert=invert

    def get_peak(self):
        if self.invert:
            peak = get_abs_max(self.featuremap)
    def get_crop(self):
        return self.crop_stack

    def plot(self):
        plt.figure(figsize=(9, 4))
        plt.subplot(131)
        plt.imshow(self.featuremap, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(self.labeled_mask)
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        plt.imshow(self.segment)
        plt.show()

        b = self.boundaries
        plt.imshow(self.segment[b['ymin']:b['ymax'], b['xmin']:b['xmax']])
        plt.show()

        for i in range(8, 12):
            plt.imshow(self.crop_stack[i])
            plt.show()


def get_mask_otsu(image):
    val = filters.threshold_otsu(image)

    mask = image > val
    return mask


def label_mask(mask):
    """
    see scipy.ndimage.label
    """
    labeled_mask, _ = ndi.label(mask)
    return labeled_mask


def get_isolated_segment(labeled_mask, peak):
    """
    provide pixel coodinates of a peak (normally get_abs_max(featuremap)),
    the segment among labeled_mask containing this peak will be isolated
    :return: mask_ with only one segment
    """
    seg_index = labeled_mask[peak]
    # print(seg_index)
    mask_ = labeled_mask == seg_index
    return mask_


def check_limits(img,coordinate):
    """
    Checks if the value is inside img borders
    :param img: 2d array
    :param coordinate: coordinate: int or tuple
    :return: coordinate within img.shape
    """
    if isinstance(coordinate,tuple):
        assert np.ndim(img) == len(coordinate)
        shape = img.shape
        out = np.array(coordinate)
        for i,c in enumerate(coordinate):
            c = max([0,c])
            out[i] = min([shape[i]-1,c])
        return tuple(out)

    elif isinstance(coordinate,int):
        assert np.ndim(img) == 1
        shape = len(img)
        c=coordinate
        c = max([0, c])
        out = min([shape - 1, c])
        return out


def get_xy_boundaries(segment, extend = 0):
    """
    Detects the xy baoundaries of a binary mask (segment)
    :segment: binary 2D dataset
    :return: dict{xmin,xmax,ymin,ymax}
    """
    assert np.ndim(segment) == 2, print("please provide 2D dataset")
    qy, qx = np.indices(segment.shape)
    e = extend
    xmin = qx[segment].min() - e
    xmax = qx[segment].max() + e
    ymin = qy[segment].min() - e
    ymax = qy[segment].max() + e
    xmin,ymin = check_limits(segment,(xmin,ymin))
    xmax,ymax = check_limits(segment,(xmax,ymax))
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def crop_2d_using_xy_boundaries(mask, boundaries):
    """
    :mask: any 2D dataset
    :boundaries: dict{xmin,xmax,ymin,ymax}
    :return: cropped mask
    """
    b = boundaries
    return mask[b['ymin']:b['ymax'], b['xmin']:b['xmax']]


def crop_using_xy_boundaries(mask, boundaries,extend=0):
    """
    :mask: any 2D or 3D dataset
    :boundaries: dict{xmin,xmax,ymin,ymax}
    :return: cropped mask
    """
    b = boundaries
    e = extend
    if np.ndim(mask) == 3:
        return mask[:, b['ymin']-e:b['ymax']+e, b['xmin']-e:b['xmax']+e]
    elif np.ndim(mask) == 2:
        return mask[b['ymin']-e:b['ymax']+e, b['xmin']-e:b['xmax']+e]
    else:
        raise (TypeError("Please use 2d or 3d data set"))


def highlight_feature(cal_stack):
    diff = []
    grad = []
    for i in range(len(cal_stack) - 1):
        dd = cal_stack[i] - cal_stack[i + 1]
        diff.append(dd)
        dd = get_grad(cal_stack[i])
        grad.append(dd)

    diff = np.array(diff).mean(axis=0)
    grad = np.array(grad).mean(axis=0)
    featuremap = gf(diff, 5) * gf(grad, 5)
    return featuremap


def get_grad(varray):
    vgrad = np.gradient(varray)
    fulgrad = np.sqrt(vgrad[0] ** 2 + vgrad[1] ** 2)
    return fulgrad


def read_roi(path):
    return read_roi_file(path)


def roi_to_boundaries(roi):
    roi = roi[list(roi)[0]]
    return dict(xmin = roi['left'],xmax = roi['left'] + roi['width'], ymin =  roi['top'], ymax =  roi['top'] + roi['height'])
