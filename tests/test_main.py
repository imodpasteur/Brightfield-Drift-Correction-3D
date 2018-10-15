from unittest import TestCase,skip
import numpy as np
from bfdc.iotools import check_stacks_size_equals
from bfdc.drift import main
import sys, os
import logging
logger = logging.getLogger(__name__)


class TestMain(TestCase):
    @skip
    def test_main(self):
        par = dict(command='trace',
                   dict='data/LED_stack_full_100nm.tif',
                   driftFileName='BFCC_table.csv',
                   movie='data/sr_2_LED_movie.tif',
                   nframes=10,
                   skip=0,
                   start=0,
                   xypixel=110,
                   zstep=100)
        sys.argv[1:] = ['trace',
               f'{par["dict"]}',
               f'{par["movie"]}',
               f'--nframes {par["nframes"]}']

        main(par)



class TestCheck_stacks_size_equals(TestCase):

    def test_check_stacks_size_equals(self):
        stack1 = np.zeros((10,15,15))
        stack2 = np.zeros((15,15,15))
        self.assertTrue(check_stacks_size_equals(stack1,stack2))

    def test_check_stacks_size_noequals(self):
        stack1 = np.zeros((10, 13, 13))
        stack2 = np.zeros((15, 15, 15))
        self.assertFalse(check_stacks_size_equals(stack1, stack2))

