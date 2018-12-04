#!/Users/andrey/anaconda3/envs/pydrift-test/python3

from unittest import TestCase
import numpy as np
import bfdc.gaussfit as gf
#import matplotlib.pyplot as plt

class TestFitPoly1D(TestCase):

    def test_1D(self):
        background = 0.2
        height = 0.8
        center_x = 8
        width = 6
        ramp = 0.01
        expected = [background, height, center_x, width, ramp]
        x = np.arange(16, dtype='f')
        print(x)
        gauss1d = gf.symmetricGaussian1DonRamp(*expected)(x)
        #plt.plot(gauss1d, '.-')
        #plt.show()
        result, good = gf.fitSymmetricGaussian1DonRamp(gauss1d, 0.25)
        for i,j in zip(result, expected):
            self.assertAlmostEqual(first=i, second=j, delta=0.01*abs(i), msg=f'result = {result}, expected {expected}')

    def test_2D(self):
        background = 0.2
        height = 0.8
        center_x = 6
        center_y = 3
        c = .3
        b = .1
        a = .2
        ramp_x = 0.01
        ramp_y = -0.02
        expected = [background, height, center_x, center_y, a, b, c, ramp_x, ramp_y]
        x = np.indices((16,6), dtype='f')
        #print(x)
        gauss2d = gf.ellipticalGaussianOnRamp(*expected)(*x)
        #plt.imshow(gauss2d)
        #plt.show()
        result, good = gf.fitFixedEllipticalGaussianOnRamp(gauss2d)
        for i,j in zip(result, expected):
            self.assertAlmostEqual(first=i, second=j, delta=0.01*abs(i), msg=f'result = {result}, expected {expected}')

    def test_ramp_3D(self):
        background = 1
        ramp = np.array((1,2,3))
        shape = np.array([16,8,8])
        indices = np.indices(shape)
        bg_fn = lambda indices: background + (ramp.reshape((indices.shape[0],1,1,1)) * indices).sum(axis=0)
        z, y, x = indices
        bg = bg_fn(indices)
        self.assertEqual(bg.ndim, len(shape))
