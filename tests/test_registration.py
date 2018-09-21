from bfdc.registration import check_size, pad_inputs
import unittest as ut
import numpy as np


class test_check_size(ut.TestCase):
    def test_wrong_dim(self):
        img = np.ones((10, 15))
        tmp = np.ones((5, 6, 7))
        with self.assertRaises(AssertionError):
            check_size(img, tmp)

    def test_wrong_size(self):
        img = np.ones((10, 15))
        tmp = np.ones((5, 20))
        out = check_size(img, tmp)
        self.assertFalse(out)

    def test_good_size(self):
        img = np.ones((10, 15))
        tmp = np.ones((10, 15))
        out = check_size(img, tmp)
        self.assertTrue(out)



class test_pad_inputs(ut.TestCase):
    def test_even_dim(self):
        img = np.ones((10, 16))
        tmp = np.ones((18, 14))
        new_img, new_tmp = pad_inputs(img,tmp)
        self.assertTupleEqual(new_img.shape, new_tmp.shape)

    def test_odd_dim(self):
        img = np.ones((10, 15))
        tmp = np.ones((17, 14))
        new_img, new_tmp = pad_inputs(img,tmp)
        self.assertTupleEqual(new_img.shape, new_tmp.shape)


if __name__ == '__main__':
    ut.main()