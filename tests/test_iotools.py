from unittest import TestCase
import numpy as np
from bfdc.iotools import check_stacks_size_equals, check_multi_channel,skip_stack
import logging
from picasso import io as pio

logger = logging.getLogger(__name__)


class Test_check_multi_channel(TestCase):

    def test_single_channel(self):
        stack = np.zeros((10, 16, 16))
        out = check_multi_channel(stack).shape
        expected = (10, 16, 16)
        self.assertEqual(out, expected)

    def test_dual_channel(self):
        stack = np.zeros((10, 2, 16, 16))
        stack[:, 1] = 1
        out = check_multi_channel(stack)
        out_shape = check_multi_channel(stack).shape
        expected = np.ones((10, 16, 16))
        expected_shape = (10, 16, 16)
        res1 = np.unique(np.equal(out, expected))
        #logger.info(f"{__name__}: res1 {res1}")
        self.assertEqual(res1, [True])
        self.assertTupleEqual(out_shape, expected_shape)


class TestCheck_stacks_size_equals(TestCase):

    def test_check_stacks_size_equals(self):
        stack1 = np.zeros((10, 15, 15))
        stack2 = np.zeros((15, 15, 15))
        self.assertTrue(check_stacks_size_equals(stack1, stack2))

    def test_check_stacks_size_noequals(self):
        stack1 = np.zeros((10, 13, 13))
        stack2 = np.zeros((15, 15, 15))
        self.assertFalse(check_stacks_size_equals(stack1, stack2))


class TestSkip_stack(TestCase):

    def test_skip_stack_default(self):

        #stack = np.zeros((100,15,15))
        stack,[info] = pio.load_movie('LED_stack_full_100nm.tif')
        input = stack.shape
        n_frames = stack.n_frames
        expected_index_list = list(np.arange(n_frames))
        print(f'input shape {input}')
        start = 0
        skip = 0
        nframes = None
        new_stack,index_list = skip_stack(stack, start=start, skip=skip, maxframes=nframes)
        self.assertListEqual(expected_index_list,list(index_list),msg=f"test_skip_stack_default: {input}, {len(index_list)}")

    def test_skip_stack_skipping10(self):
        stack, [info] = pio.load_movie('../data/crop_frame/sr_2_LED_movie.tif')
        input = stack.shape
        start = 10
        skip = 9
        nframes = None
        n_frames = stack.n_frames
        expected_index_list = list(np.arange(start-1,n_frames,skip+1))
        print(f'input shape {input}')
        new_stack, index_list = skip_stack(stack, start=start, skip=skip, maxframes=nframes)
        self.assertListEqual(expected_index_list, list(index_list),
                             msg=f"test_skip_stack_default: {input}, {len(index_list)}")

