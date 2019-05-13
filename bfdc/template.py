import numpy as np
from scipy import ndimage as ndi
from skimage.feature import match_template, peak_local_max


import logging

#logging.basicConfig(level='DEBUG')
logger = logging.getLogger(__name__)



def generate_ring_template(px_nm=20, 
                           size=200,
                           diam_nm=120,
                           thick_nm=20,
                           smooth_nm=20):
    _size=size // px_nm
    _diam=diam_nm // px_nm
    _thick=thick_nm // px_nm
    qy,qx = np.indices((_size,_size))
    c = _size/2.
    ring = np.ones((_size, _size))
    ring[((qx-c)**2+(qy-c)**2)<((_diam - _thick)/2)**2]=0
    ring[((qx-c)**2+(qy-c)**2)>((_diam + _thick)/2)**2]=0
    if smooth_nm: ring= ndi.gaussian_filter(ring,smooth_nm / px_nm)
    return ring

def generate_circle_template(px=20, 
                           size=200,
                           diam=120,
                           smooth=20,
                           shift=(0, 0)):
    _size=size // px
    _diam=diam // px
    qy,qx = np.indices((_size,_size))
    c = _size/2.
    ring = np.ones((_size, _size))
    ring[((qx - c - shift[0]) ** 2 + (qy - c - shift[1]) ** 2 ) > (_diam / 2) ** 2]=0
    if smooth: 
        ring = ndi.gaussian_filter(ring,smooth / px)
    return ring

def generate_moving_circle_2D(px=20, 
                            size=200,
                            diam=120,
                            smooth=20,
                            coordinates=np.cumsum(np.ones((10,2)), axis=0)):
    movie = list(map(lambda shift: generate_circle_template(px=20, 
                                                            size=200,
                                                            diam=120,
                                                            smooth=20,
                                                            shift=shift), 
                                                            coordinates))
    return movie

def conv(img, kern):
    return match_template(img, kern, pad_input=True)
