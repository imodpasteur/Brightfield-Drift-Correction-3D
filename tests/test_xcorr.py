
from unittest import TestCase, skip
import numpy as np
from bfdc.xcorr import FitResult


class TestFitResult(TestCase):

    def test_init(self):
        a = FitResult()
        x,y,z,good,result_fit,z_px = a
        for a, b in zip((x,y,z,good,result_fit,z_px), (None, None, None, False, None, None)):
            self.assertEqual(a,b)
    def test_repr(self):
        a = FitResult()
        b = repr(a)
        self.assertEqual(b,'x={}, y={}, z={}, good={}, result_fit={}, z_px={}'.format(None, 
                            None, None, False, None, None))
