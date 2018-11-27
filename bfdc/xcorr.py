import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.feature import match_template
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


def fit_gauss_3d(stack, radius_xy=4, radius_z=8, z_zoom=20, min_xcorr = 0.5, z_crop = (None,None), debug=False):
    """
    Detects maximum on the stack in 3D
    Fitting 2D gaussian in xy, 4-order polynomial in z
    Ouputs [x,y,z] in pixels
    """
    
    from bfdc import gaussfit

    logger.debug(f'Start fit_gauss_3d with the stack shape zyx {stack.shape}')
    logger.debug(f'radius_xy={radius_xy}, radius_z={radius_z}, z_zoom={z_zoom}')
    assert np.ndim(stack) == 3, logger.error(f'fit_gauss_3d: input stack shape is wrong, expected 3 dim, got {stack.shape}')

    if False:
        plt.imshow(stack.max(axis=0))
        plt.title('Max xy projection of cc-stack')
        plt.show()

    z_px, y_px, x_px = get_abs_max(stack)
    logger.debug(f'Got absolute maximum xyz (px) {(x_px, y_px, z_px )}')
    cc_value = np.max(stack)
    if cc_value < min_xcorr:
        #raise(LowXCorr("fit_gauss_3d: Cross corellation value os too low!"))
        logger.error("fit_gauss_3d: Cross corellation value os too low!")
        return [0, 0, 0, False]
    else:
        logger.debug(f'cc peak value={cc_value}')

    r, rz = radius_xy, radius_z
    if z_crop == (None, None):
        z_start = max(z_px - rz, 0)
        z_stop = min(z_px + rz, len(stack))
        z_crop = (z_start, z_stop)
        logger.debug(f'Computing z boundaries before fit: z_start={z_start}, z_stop={z_stop}')
    else:
        z_start, z_stop = z_crop
        logger.debug(f'Using z boundaries: z_start={z_start}, z_stop={z_stop}')

    
    _, y_max, x_max = stack.shape
    y1 = max(0, y_px - r) 
    y2 = min(y_max, y_px + r)
    x1 = max(0, x_px - r) 
    x2 = min(x_max, x_px + r)
    cut_stack = stack[z_start:z_stop, y1:y2, x1:x2]
    logger.debug(f'After cutting x,y,z, we got cut_stack shape {cut_stack.shape}')
    if cut_stack.shape != (z_stop - z_start, 2 * r , 2 * r  ):
        logger.error(f'Wrong cut_stack shape: expected {(z_stop - z_start, 2 * r + 1, 2 * r +1 )}, got {cut_stack.shape}')
        return [-1, -1, -1, False, z_crop]

    xy_proj = cut_stack.max(axis=0)
    #z_proj = cut_stack.max(axis=(1, 2))
    z_proj = cut_stack[:,r].max(axis=1) #ignore y
    # z_proj = cut_stack[:,r,r]

    #[(_min, _max, y, x, sig), good] = gaussfit.fitSymmetricGaussian(xy_proj,sigma=1)
    logger.debug('Fit gauss xy')
    try:    
        [(_min, _max, y, x, sigy,angle,sigx), good] = gaussfit.fitEllipticalGaussian(xy_proj)
        logger.debug(f'raw xy {(x,y)}')
    except Exception as e:
        logger.error(e)
        return [-1, -1, -1, False, z_crop]
    x_found = x - r + x_px
    y_found = y - r + y_px
    logger.debug(f'xy found: {(x_found, y_found)}')

    # [(_min,_max,z,sig),good] = gaussfit.fitSymmetricGaussian1D(z_proj)
    
    polyfit = fit_poly_1D(z_proj, z_zoom, order=4)
    z_subpx = polyfit()
    z_found = z_subpx + z_start
    logger.debug(f'z_found = {z_subpx} + {z_start}')


    if debug:

        fig = plt.figure(figsize=(15,3))
        fig.add_subplot(141)
        plt.imshow(xy_proj)
        plt.title('xy')
        plt.colorbar()

        fig.add_subplot(142)
        plt.imshow(cut_stack.max(axis=1))
        plt.title('zx')
        plt.colorbar()

        fig.add_subplot(143)
        plt.imshow(cut_stack.max(axis=2))
        plt.title('zy')
        plt.colorbar()

        fig.add_subplot(144)
        plt.plot(polyfit.x + z_start, z_proj, 'o', label='cc curve')
        plt.plot(polyfit.x_new + z_start, polyfit.ext_curve, '--', label='poly fit')
        plt.plot(polyfit.x_new[np.argmax(polyfit.ext_curve)]+ z_start, max(polyfit.ext_curve), 'o', label='z found')
        plt.legend()

        plt.tight_layout()

    return x_found, y_found, z_found, good, z_crop


class fit_poly_1D:

    def __init__(self, curve, zoom, order=4, peak='max'):
        """
        Fits polynomial of given order to 1D curve and returns descreet absolute max/min after subsampling the fit.
        :param curve: 1D ndarraay
        :param zoom: upsampling
        :param order: polynomial order to fit
        :param peak: 'max' or 'min'
        :return z coordinate (first dimension with the best match)
        """
        self.curve = curve
        self.zoom = zoom
        self.order = order
        self.peak = peak

        self.interpolate()
        self.fit_poly()
        self.detect_peak()

    def interpolate(self):
        self.x = np.arange(len(self.curve))
        self.x_new = np.linspace(0., len(self.curve), num=self.zoom * len(self.curve), endpoint=False)

    def fit_poly(self):
        fit = np.polyfit(self.x, self.curve, deg=self.order)
        poly = np.poly1d(fit)
        self.ext_curve = poly(self.x_new)

    def detect_peak(self):
        if self.peak == 'max':
            self.subpx_fit = self.x_new[np.argmax(self.ext_curve)]
        elif self.peak == 'min':
            self.subpx_fit = self.x_new[np.argmin(self.ext_curve)]
        else:
            raise(ValueError(f'peak value \'{self.peak}\' not recognized. Expected \'max\' or \'min\''))

    def __call__(self):
        return self.subpx_fit

def fit_z_MSE(frame, template, zoom, order=4):
    """
    Fits MSE between frame 2D and template 3D to find z minimum
    :param frame: 2D ndarraay
    :param template: 3D array
    :param zoom: intepropation for the first dimension
    :param order: polynomial order to fit
    :return z coordinate (first dimension with the best match)
    """

    assert frame.ndim == 2
    assert template.ndim  == 3
    mse = (template - frame) ** 2 / len(template)
    curve = mse.mean(axis=(1, 2))
    z_px = fit_poly_1D(curve, zoom=20, order=4, peak='min')
    return z_px

    
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
