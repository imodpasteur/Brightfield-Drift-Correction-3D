import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.feature import match_template
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
# logger.setLevel(logging.DEBUG)


class LowXCorr(Exception):
    pass


class FitResult():

    def __init__(self,
                 x=None,
                 y=None,
                 z=None,
                 good=False,
                 result_fit=None,
                 z_px=None):
        self.x = x
        self.y = y
        self.z = z
        self.good = good
        self.result_fit = result_fit
        self.z_px = z_px

    def __iter__(self):
        for a in (self.x, self.y, self.z, self.good, self.result_fit, self.z_px):
            yield a

    def __repr__(self):
        return 'x={}, y={}, z={}, good={}, result_fit={}, z_px={}'.format(self.x,
                                                                          self.y, self.z, self.good, self.result_fit,
                                                                          self.z_px)


def get_abs_max(data):
    """
    peaks up absolute maximum position in the stack/ 2D image
    First dimension comes first
    :param data: np.array
    :return: abs max coordinates in data.shape format
    """
    return np.unravel_index(np.argmax(data), data.shape)


def fit_gauss_3d(stack: np.ndarray,
                 radius_xy: int = 4,
                 radius_z: int = 8,
                 z_zoom: int = 20,
                 min_xcorr: np.float = 0.5,
                 z_init: np.float = None,
                 fit_init: list = None,
                 debug=False) -> FitResult:
    """
    Detects maximum on the stack in 3D
    Fitting 2D gaussian in xy, 4-order polynomial in z
    Ouputs [x,y,z] in pixels
    """
    fit2d = False

    from bfdc import gaussfit

    logger.debug(f'Start fit_gauss_3d with the stack shape zyx {stack.shape}')
    logger.debug(f'radius_xy={radius_xy}, radius_z={radius_z}, z_zoom={z_zoom}')
    

    r, rz = radius_xy, radius_z
    
    cc_value = np.max(stack)
    if cc_value < min_xcorr:
        # raise(LowXCorr("fit_gauss_3d: Cross corellation value os too low!"))
        logger.error("fit_gauss_3d: Cross corellation value os too low!")
        return FitResult()
    else:
        logger.debug(f'cc peak value={cc_value}')

    if len(stack) == 1:
        logger.debug(f'fit_gauss_3d: switch to 2D mode as input stack shape {stack.shape}')
        assert stack.ndim == 3
        fit2d = True

    z_px, y_px, x_px = get_abs_max(stack)
    _, y_max, x_max = stack.shape
        
    y1 = max(0, y_px - r)
    y2 = min(y_max, y_px + r)
    x1 = max(0, x_px - r)
    x2 = min(x_max, x_px + r)
 
    if False:
        plt.imshow(stack.max(axis=0))
        plt.title('Max xy projection of cc-stack')
        plt.show()
    
    logger.debug(f'Got absolute maximum xyz (px) {(x_px, y_px, z_px )}')
    
    if z_init is None:
        z_start = max(z_px - rz, 0)
        z_stop = min(z_px + rz, len(stack))
        z_crop = (z_start, z_stop)
        logger.debug(f'Computing z boundaries from peak: z_start={z_start}, z_stop={z_stop}')
    else:
        z_init = int(z_init)
        z_start = max(z_init - rz, 0)
        z_stop = min(z_init + rz, len(stack))
        z_crop = (z_start, z_stop)
        logger.debug(f'Using z boundaries: z_start={z_start}, z_stop={z_stop}')

    
    cut_stack = stack[z_start:z_stop, y1:y2, x1:x2]
    logger.debug(f'After cutting x,y,z, we got cut_stack shape {cut_stack.shape}')
    if cut_stack.shape != (z_stop - z_start, 2 * r, 2 * r):
        logger.error(
            f'Wrong cut_stack shape: expected {(z_stop - z_start, 2 * r + 1, 2 * r +1 )}, got {cut_stack.shape}')
        return FitResult()

    xy_proj = cut_stack.max(axis=0)
    zx_proj = cut_stack.max(axis=1)
    zy_proj = cut_stack.max(axis=2)
    z_proj = cut_stack.max(axis=(1, 2))
    '''
    _, _x, _y = get_abs_max(cut_stack)
    logger.debug(f'Looking for xy peak {(_x, _y)}')

    z_crop = cut_stack[:, _y-1:_y+2, _x-1:_x+2]
    logger.debug(f'Crop {z_crop.shape}')

    z_proj = z_crop.mean(axis=(1, 2))
    '''
    # z_proj = cut_stack[:,r].max(axis=1) #ignore y
    # z_proj = cut_stack[:,r,r] #ignore xy
    # z_proj = cut_stack[:,r,r]

    # [(_min, _max, y, x, sig), good] = gaussfit.fitSymmetricGaussian(xy_proj,sigma=1)
    logger.debug('Fit gauss xy')
    try:
        [(_min, _max, y, x, _, _, _), good] = gaussfit.fitEllipticalGaussian(xy_proj)
        #[result_fit, good] = gaussfit.fitEllipticalGaussian3D(cut_stack, init=fit_init)
        #background, height, z, y, x, el_x, el_y, el_z, an_xy, an_yz, an_xz, ramp_x, ramp_y, ramp_z = result_fit

    except Exception as e:
        logger.error(f'Error in gaussian fit: {e}')
        #logger.error(f'result: {result_fit}')
        return FitResult()

    x_found = x - r + x_px
    y_found = y - r + y_px
    logger.debug(f'xy found: {np.round((x_found, y_found),2)}')

    if fit2d:
        z_found = 0
        z = 0
    else:
        zfitter = FitPoly1D(z_proj, zoom=20, radius=5)
        z = zfitter(plot=debug)
        z_found = z + z_start
    
    logger.debug(f'raw xyz {np.round((x, y, z),2)}')

    if debug and not fit2d:
        # fitted_ellipsoid = gaussfit.ellipticalGaussian3dOnRamp(*result_fit)(*np.indices(cut_stack.shape))

        # fit_residue = cut_stack - fitted_ellipsoid

        fig = plt.figure(dpi=72, figsize=(8, 3))

        fig.add_subplot(131)
        plt.imshow(xy_proj)
        plt.plot(x,y,'r+')
        plt.title('xy')
        plt.colorbar()

        fig.add_subplot(132)
        plt.imshow(zx_proj)
        plt.plot(x,z,'r+')
        plt.title('zx')
        plt.colorbar()

        fig.add_subplot(133)
        plt.imshow(zy_proj)
        plt.plot(y,z,'r+')
        plt.title('zy')
        plt.colorbar()
        '''
        fig.add_subplot(334)
        plt.imshow(fitted_ellipsoid.max(axis=0))
        plt.title('fit xy')
        plt.colorbar()

        fig.add_subplot(335)
        plt.imshow(fitted_ellipsoid.max(axis=1))
        plt.title('fit zx')
        plt.colorbar()

        fig.add_subplot(336)
        plt.imshow(fitted_ellipsoid.max(axis=2))
        plt.title('fit zy')
        plt.colorbar()
        
        fig.add_subplot(337)
        plt.imshow(fit_residue.max(axis=0))
        plt.title('residue xy')
        plt.colorbar()

        fig.add_subplot(338)
        plt.imshow(fit_residue.max(axis=1))
        plt.title('residue zx')
        plt.colorbar()

        fig.add_subplot(339)
        plt.imshow(fit_residue.max(axis=2))
        plt.title('residue zy')
        plt.colorbar()
        

        fig.add_subplot(144)
        plt.plot(z_proj, 'o', label='cc curve')
        plt.plot(gaussfit.symmetricGaussian1DonRamp(*params)(np.arange(len(z_proj))), '--', label='gauss fit')
        plt.legend()
        
        fig.add_subplot(144)
        try:
            plt.plot(zfitter.x + z_start, z_proj, 'o', label='cc curve')
            plt.plot(zfitter.x_new + z_start, zfitter.ext_curve, '--', label='poly fit')
            plt.plot(zfitter.x_new[np.argmax(zfitter.ext_curve)]+ z_start, max(zfitter.ext_curve), 'o', label='z found')
            plt.legend()
        except ValueError as e:
            logger.error(f'Error in subplot 144: {e}')
        '''
        plt.tight_layout()
        plt.show()
    elif debug and fit2d:
        fig = plt.figure(dpi=72, figsize=(3, 3))

        plt.plot(x,y,'r+')
        plt.title('xy')
        plt.colorbar()
        plt.show()
        
    return FitResult(x_found, y_found, z_found, good, None, z_px)


class FitPoly1D:

    def __init__(self, curve, zoom=10, order=4, radius=5, peak='max'):
        """
        Fits polynomial of given order to 1D curve and returns descreet absolute max/min after subsampling the fit.
        :curve: 1D ndarraay
        :zoom: upsampling
        :order: polynomial order to fit
        :peak: 'max' or 'min'
        :return: z coordinate (first dimension with the best match)
        """
        logger.debug(f'Start FitPoly1D init with a curve on length {len(curve)}')
        self.curve_full = curve
        self.curve = curve
        self.zoom = zoom
        self.order = order
        self.crop = 0

        if peak == 'max':
            self.peak = np.argmax
        elif peak == 'min':
            self.peak = np.argmin
        else:
            raise (ValueError(f'peak value \'{self.peak}\' not recognized. Expected \'max\' or \'min\''))

        if radius: self.do_crop(radius)
        self.interpolate()
        self.fit_poly()
        self.detect_peak()

    def do_crop(self, radius):
        logger.debug(f'FitPoly1D:do_crop')
        index = self.peak(self.curve)
        a = max(0, index - radius)
        b = min(len(self.curve), index + radius)
        self.curve = self.curve[a:b]
        self.crop = a

    def interpolate(self):
        logger.debug(f'FitPoly1D:interpolate')
        self.x_full = np.arange(len(self.curve_full))
        self.x = np.arange(self.crop, self.crop + len(self.curve))
        self.x_new = np.linspace(self.crop,
                                 len(self.curve) + self.crop,
                                 num=self.zoom * len(self.curve),
                                 endpoint=False,
                                 dtype='f')

    def fit_poly(self):
        logger.debug(f'FitPoly1D:fit_poly')
        fit = np.polyfit(self.x, self.curve, deg=self.order)
        poly = np.poly1d(fit)
        self.ext_curve = poly(self.x_new)

    def detect_peak(self):
        logger.debug(f'FitPoly1D:detect_peak')
        self.subpx_fit = self.x_new[self.peak(self.ext_curve)]
        logger.debug(f'FitPoly1D:detect_peak found {self.subpx_fit}')

    def plot(self):
        logger.debug(f'FitPoly1D:plot')
        plt.plot(self.x_full, self.curve_full, 'o', label='raw curve')
        plt.plot(self.x_new, self.ext_curve, '--', label='poly fit')
        plt.plot(self.subpx_fit, self.ext_curve[self.peak(self.ext_curve)], 'o', label=f'z found{self.subpx_fit}')
        plt.legend()
        plt.title('FitPoly1D plot')
        plt.show()

    def __call__(self, plot=False):
        if plot: self.plot()

        return self.subpx_fit


def fit_z_MSE(frame: np.ndarray, template: np.ndarray, zoom: int, order: int = 4, plot: bool = False) -> float:
    """
    Fits MSE between frame 2D and template 3D to find z minimum
    :frame: 2D ndarraay
    :template: 3D array
    :zoom: intepropation for the first dimension
    :order: polynomial order to fit
    :plot: bool, will show the curve and fit
    :return: z coordinate (first dimension with the best match)
    """
    z_px = 0.0
    assert np.ndim(frame) == 2, 'wrong frame shape'
    assert np.ndim(template) == 3
    mse = (template - frame) ** 2 / len(template)
    curve = mse.mean(axis=(1, 2))
    fitter = FitPoly1D(curve, zoom=20, order=4, peak='min')
    z_px = fitter(plot=plot)
    logger.debug(f'found z {z_px}')
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
    '''
    Correlates and image with 3D stack, returns 3D stack
    '''

    image = np.array(image - np.mean(image), dtype='f')
    stack = np.array(stack - np.mean(stack), dtype='f')
    if stack.ndim == 3:
        stack = stack
    elif stack.ndim == 2:
        stack = [stack]
    else:
        raise ValueError(f'Wrong stack with shape {stack.shape}, expected 2 or 3 dimensions')
    
    out = []
    try:
        for t in stack:
            out.append(cc_template(image, t))
        out = np.array(out)
    except Exception as e:
        logger.error('in cc_stack')
        raise e

    if plot and stack.ndim == 3:
        fig = plt.figure(figsize=(6, 2))

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
    elif plot and stack.ndim == 2:
        plt.imshow(out.max(axis=0))
        plt.title('cross-correlation xy')
        plt.colorbar()
        plt.show()

    return out


def cc_max(cc_out):
    return np.unravel_index(np.argmax(cc_out), cc_out.shape)
