from unittest import TestCase
import numpy as np
import bfdc.iotools as iot
import logging
import bfdc.picassoio as pio
import pandas as pd

logger = logging.getLogger(__name__)


class Test_pd_io(TestCase):

    def test_get_ext(self):
        path = 'safdsa/dsfadf.csv'
        ext = iot.get_ext(path)
        self.assertEqual(ext, '.csv')

    def test_csv_open(self):
        path = "data/NUP-20180612-FOV1/results/BFCC_table.csv"
        df = iot.open_localization_table(path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_txt_open(self):
        path = "data/NUP-20180612-FOV1/results/ZOLA_localization_table_ZolaProtocol.txt"
        with self.assertRaises(TypeError):
            df = iot.open_localization_table(path)


class Test_check_multi_channel(TestCase):

    def test_single_channel(self):
        stack = np.zeros((10, 16, 16))
        out = iot.check_multi_channel(stack).shape
        expected = (10, 16, 16)
        self.assertEqual(out, expected)

    def test_dual_channel(self):
        stack = np.zeros((10, 2, 16, 16))
        stack[:, 1] = 1
        out = iot.check_multi_channel(stack)
        out_shape = iot.check_multi_channel(stack).shape
        expected = np.ones((10, 16, 16))
        expected_shape = (10, 16, 16)
        res1 = np.unique(np.equal(out, expected))
        # logger.info(f"{__name__}: res1 {res1}")
        self.assertEqual(res1, [True])
        self.assertTupleEqual(out_shape, expected_shape)


class TestCheck_stacks_size_equals(TestCase):

    def test_check_stacks_size_equals(self):
        stack1 = np.zeros((10, 15, 15))
        stack2 = np.zeros((15, 15, 15))
        self.assertTrue(iot.check_stacks_size_equals(stack1, stack2))

    def test_check_stacks_size_noequals(self):
        stack1 = np.zeros((10, 13, 13))
        stack2 = np.zeros((15, 15, 15))
        self.assertFalse(iot.check_stacks_size_equals(stack1, stack2))


class TestSkip_stack(TestCase):

    def test_skip_stack_default(self):

        # stack = np.zeros((100,15,15))
        stack, [_] = pio.load_movie(
            './data/full_frame/sr_642_redLED_start_10_skip_10_SP_100nm_1_MMStack_Pos0-100f.ome.tif')
        input = stack.shape
        n_frames = stack.n_frames
        expected_index_list = list(np.arange(n_frames))
        print(f'input shape {input}')
        start = 0
        skip = 0
        nframes = 0
        index_list = iot.skip_stack(n_frames=len(stack), 
                                    start=start, 
                                    skip=skip, 
                                    maxframes=nframes)
        self.assertListEqual(expected_index_list, list(
            index_list), msg=f"test_skip_stack_default: {input}, {len(index_list)}")

    def test_skip_stack_skipping10(self):
        stack, [_] = pio.load_movie('./data/crop_frame/sr_2_LED_movie.tif')
        input = stack.shape
        start = 10
        skip = 10
        nframes = None
        n_frames = stack.n_frames
        expected_index_list = list(np.arange(start-1, n_frames, skip))
        print(f'input shape {input}')
        index_list = iot.skip_stack(n_frames=len(
            stack), start=start, skip=skip, maxframes=nframes)
        self.assertListEqual(expected_index_list, list(index_list),
                             msg=f"test_skip_stack_default: {input}, {len(index_list)}")
