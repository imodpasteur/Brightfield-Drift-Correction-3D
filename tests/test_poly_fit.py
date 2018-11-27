#!/Users/andrey/anaconda3/envs/pydrift-test/python3

from unittest import TestCase
import numpy as np
from bfdc.xcorr import fit_poly_1D

class TestMove_drift_to_zero(TestCase):

    def test_parabola_max(self):
        indices = np.arange(10)
        expected = 10
        zoom = 100
        curve = 5 - (indices-expected) ** 2
        fit = fit_poly_1D(curve=curve, zoom=zoom, order=2, peak='max')
        result = fit()
        self.assertAlmostEqual(result, expected, delta=1/zoom, msg=f'result = {result}, expected {expected}')

    def test_parabola_min(self):
        indices = np.arange(10)
        expected = 10
        zoom = 100
        curve = 5 + (indices - expected) ** 2
        fit = fit_poly_1D(curve=curve, zoom=zoom, order=2, peak='min')
        result = fit()
        self.assertAlmostEqual(result, expected, delta=1 / zoom, msg=f'result = {result}, expected {expected}')

    def test_parabola_min_4order(self):
        indices = np.arange(10)
        expected = 10
        zoom = 100
        curve = 5 + (indices-expected) ** 2
        fit = fit_poly_1D(curve=curve, zoom=zoom, order=4, peak='min')
        result = fit()
        self.assertAlmostEqual(result, expected, delta=1/zoom, msg=f'result = {result}, expected {expected}')

