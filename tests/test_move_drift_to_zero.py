from unittest import TestCase
import numpy as np
from bfdc.drift import move_drift_to_zero

class TestMove_drift_to_zero(TestCase):
    def test_move_drift_to_zero(self):
        table = np.ones((20,4))
        table[:,0] = np.arange(20)
        expected = table.copy()
        expected[:,1:] = 0
        result = move_drift_to_zero(table)
        self.assertTrue(np.array_equiv(result,expected))

    def test_move_drift_to_zero1(self):
        table = np.ones((20,4))
        table[:,0] = np.arange(1,21)
        expected = table.copy()
        expected[:,1:] = 0
        result = move_drift_to_zero(table)
        self.assertTrue(np.array_equiv(result,expected))

    def test_move_drift_to_zero_wrong_shape(self):
        table = np.ones((20, 5))
        table[:, 0] = np.arange(1, 21)
        with self.assertRaises(AssertionError):
            result = move_drift_to_zero(table)

    def test_move_drift_to_zero_wrong_shape1(self):
        table = np.ones((0, 5))
        with self.assertRaises(AssertionError):
            result = move_drift_to_zero(table)

