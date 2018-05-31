from unittest import TestCase
import numpy as np
from bfdc.feature import get_xy_boundaries


class TestGet_xy_boundaries(TestCase):
    def test_get_xy_boundaries(self):
        segment = np.zeros((10,10),dtype='bool')
        segment[3:6,3:6] = True
        out = get_xy_boundaries(segment)
        expected = dict(xmin=3,xmax=5,ymin=3,ymax=5)
        self.assertEqual(out,expected)

    def test_get_xy_boundaries_ext(self):
        segment = np.zeros((10,10),dtype='bool')
        segment[3:6,3:6] = True
        out = get_xy_boundaries(segment,extend=5)
        expected = dict(xmin=0,xmax=9,ymin=0,ymax=9)
        self.assertEqual(out,expected)

