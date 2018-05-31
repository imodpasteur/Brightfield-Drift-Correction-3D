from unittest import TestCase
import numpy as np
from bfdc.iotools import check_stacks_size_equals, check_multi_channel,skip_stack
import logging

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
        stack = np.zeros((100,15,15))
        input = stack.shape
        start = 0
        skip = 0
        nframes = None
        out = skip_stack(stack,start=start,skip=skip,nframes=nframes).shape
        self.assertTupleEqual(input,out,msg=f"test_skip_stack_default: {input}, {out}")

    def test_skip_stack_skipping10(self):
        stack = np.zeros((100, 15, 15))
        input = stack.shape
        start = 10
        skip = 9
        nframes = None
        out = skip_stack(stack, start=start, skip=skip, nframes=nframes).shape
        expected = (10,15,15)
        self.assertTupleEqual(out, expected, msg=f"test_skip_stack_default: {out}, {expected}")

