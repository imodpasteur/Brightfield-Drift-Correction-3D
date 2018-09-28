import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.feature import match_template
import logging
logger = logging.getLogger(__name__)


class LowXCorr(Exception):
    pass


def get_abs_max(data):
    """
    peaks up absolute maximum position in the stack/ 2D image
    First dimension comes first
    :param data: np.array
    :return: abs max coordinates in data.shape format
    """
    return np.unravel_index(np.argmax(data), data.shape)


def fit_gauss_3d(stack, radius_xy=4, radius_z=5, z_zoom=20, debug=False):
    """
    Detects maximum on the stack in 3D
    Fitting 2D gaussian in xy, 4-order polynomial in z
    Ouputs [x,y,z] in pixels
    """
    try:
        from bfdc import gaussfit
    except ImportError:
        raise ImportError('Missing gaussfit.py. Please download one from Zhuanglab Github')
    #cut_stack = np.zeros((1, 1, 1))
    assert np.ndim(stack) == 3, logger.error(f'fit_gauss_3d: input stack shape is wrong, expected 3 dim, got {stack.shape}')

    if debug:
        plt.imshow(stack.max(axis=0))
        plt.title('Max projection of cc-stack')
        plt.show()

    z_px, y_px, x_px = get_abs_max(stack)
    cc_value = np.max(stack)
    if cc_value < 0.2:
        #raise(LowXCorr("fit_gauss_3d: Cross corellation value os too low!"))
        logger.warning("fit_gauss_3d: Cross corellation value os too low!")
        return [0, 0, 0, False]

    if debug:
        print([z_px, y_px, x_px])
    r, rz = radius_xy, radius_z
    z_start = np.maximum(z_px - rz, 0)
    z_stop = np.minimum(z_px + rz + 2, len(stack) - 1)
    cut_stack = stack[z_start:z_stop, y_px - r :y_px + r + 1, x_px - r :x_px + r + 1]
    if debug: print(f'cut_stack shape {cut_stack.shape}')


    xy_proj = cut_stack.max(axis=0)
    z_proj = cut_stack.max(axis=(1, 2))
    # z_proj = cut_stack[:,r,r]

    #[(_min, _max, y, x, sig), good] = gaussfit.fitSymmetricGaussian(xy_proj,sigma=1)

    [(_min, _max, y, x, sigy,angle,sigx), good] = gaussfit.fitEllipticalGaussian(xy_proj)

    x_found = x - r + x_px
    y_found = y - r + y_px

    # [(_min,_max,z,sig),good] = gaussfit.fitSymmetricGaussian1D(z_proj)
    z_crop = z_proj
    x = np.arange(len(z_crop))
    x_new = np.linspace(0., len(z_crop), num=z_zoom * len(z_crop), endpoint=False)
    fit = np.polyfit(x, z_crop, deg=4)
    poly = np.poly1d(fit)
    z_fit = poly(x_new)
    z_found = (x_new[np.argmax(z_fit)] + z_start)

    if debug:

        fig = plt.figure(figsize=(10,3))
        fig.add_subplot(141)
        plt.imshow(xy_proj)
        plt.title('xy')

        fig.add_subplot(142)
        plt.imshow(cut_stack.max(axis=1))
        plt.title('zx')

        fig.add_subplot(143)
        plt.imshow(cut_stack.max(axis=2))
        plt.title('zy')

        print(f'z_start={z_start}, z_stop={z_stop}')

        fig.add_subplot(144)
        plt.plot(x, z_proj, '.-', label='cc curve')
        plt.plot(x_new, z_fit, '--', label='poly fit')
        plt.plot(x_new[np.argmax(z_fit)], max(z_fit), 'o', label='z found')
        plt.legend()
        plt.show()

    return [x_found, y_found, z_found, good]


def cc_template(image, template, plot=False):
    try:
        cc = match_template(image, template, pad_input=True)
        if plot:
            plt.imshow(image)
            plt.title('image')
            plt.show()
            plt.imshow(template)
            plt.title('template')
            plt.show()
            plt.imshow(cc)
            plt.title('cc')
            plt.colorbar()
            plt.show()
        return cc
    except ValueError:
        logging.error(f"Error in cc_template. Image shape {image.shape}, template shape {template.shape  }")




def cc_stack(image, stack, plot=False):
    image = np.array(image - np.mean(image), dtype='f')
    stack = np.array(stack - np.mean(stack), dtype='f')
    out = []
    for i, t in enumerate(stack):
        out.append(cc_template(image, t))
    out = np.array(out)
    if plot:
        fig = plt.figure(figsize=(6,2))

        fig.add_subplot(131)
        plt.imshow(out.max(axis=0))
        plt.title('max xy')
        plt.colorbar()

        fig.add_subplot(132)
        plt.imshow(out.max(axis=1))
        plt.title('max zx')
        plt.colorbar()

        fig.add_subplot(133)
        plt.imshow(out.max(axis=2))
        plt.title('max zy')
        plt.colorbar()
        plt.show()
    return out


def cc_max(cc_out):
    return np.unravel_index(np.argmax(cc_out), cc_out.shape)
