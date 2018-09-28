from unittest import TestCase
import numpy as np
from bfdc.feature import check_limits


class TestCheckLimits(TestCase):
    def test_check_limits_2d(self):
        img = np.zeros((512,512))
        peak = (-1,550)
        expected = (0,511)
        out = check_limits(img,peak)
        self.assertEqual(out,expected)

    def test_check_limits_linear(self):
        img = np.zeros(12)
        peak = 5
        expected = 5
        out = check_limits(img,peak)
        self.assertEqual(out,expected)

    def test_check_limits_linear0(self):
        img = np.zeros(12)
        peak = 0
        expected = 0
        out = check_limits(img,peak)
        self.assertEqual(out,expected)

    def test_check_limits_linear_neg(self):
        img = np.zeros(12)
        peak = -1
        expected = 0
        out = check_limits(img,peak)
        self.assertEqual(out,expected)

    def test_check_limits_linear_pos(self):
        img = np.zeros(12)
        peak = 15
        expected = 11
        out = check_limits(img,peak)
        self.assertEqual(out,expected)
