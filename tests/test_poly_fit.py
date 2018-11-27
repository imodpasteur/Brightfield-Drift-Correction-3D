#!/Users/andrey/anaconda3/envs/pydrift-test/python3

from unittest import TestCase
import numpy as np
from bfdc.xcorr import FitPoly1D


class TestFitPoly1D(TestCase):

    def test_parabola_max(self):
        indices = np.arange(20)
        expected = 10
        zoom = 100
        curve = 5 - (indices-expected) ** 2
        fit = FitPoly1D(curve=curve, zoom=zoom, order=2, peak='max')
        result = fit()
        self.assertAlmostEqual(result, expected, delta=1/zoom, msg=f'result = {result}, expected {expected}')

    def test_parabola_min(self):
        indices = np.arange(20)
        expected = 10
        zoom = 100
        curve = 5 + (indices - expected) ** 2
        fit = FitPoly1D(curve=curve, zoom=zoom, order=2, peak='min')
        result = fit()
        self.assertAlmostEqual(result, expected, delta=1 / zoom, msg=f'result = {result}, expected {expected}')

    def test_parabola_min_4order(self):
        indices = np.arange(20)
        expected = 10
        zoom = 100
        curve = 5 + (indices - expected) ** 2
        fit = FitPoly1D(curve=curve, zoom=zoom, order=4, peak='min')
        result = fit()
        self.assertAlmostEqual(result, expected, delta=1/zoom, msg=f'result = {result}, expected {expected}')

    def test_parabola_min_plot(self):
        indices = np.arange(20)
        expected = 10
        zoom = 100
        curve = 5 + (indices - expected) ** 2
        fit = FitPoly1D(curve=curve, zoom=zoom, order=2, peak='min')
        result = fit(plot=True)
        self.assertAlmostEqual(result, expected, delta=1 / zoom, msg=f'result = {result}, expected {expected}')

    def test_parabola_min_plot_big_rad(self):
        indices = np.arange(20)
        expected = 10
        zoom = 100
        curve = 5 + (indices - expected) ** 2
        fit = FitPoly1D(curve=curve, zoom=zoom, order=2, peak='min', radius=20)
        result = fit(plot=False)
        self.assertAlmostEqual(result, expected, delta=1 / zoom, msg=f'result = {result}, expected {expected}')