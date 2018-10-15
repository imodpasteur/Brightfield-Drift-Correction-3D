import argparse

import numpy as np
import os
import matplotlib as mpl
mpl.use('TkAgg')
#mpl.use('PS')
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d as gf1
from skimage import io
import bfdc.picassoio as pio
import subprocess

import logging

logger = logging.getLogger(__name__)


def open_stack(path):
    if path.endswith('ome.tif'):
        return io.imread(path)
    elif os.path.isdir(path):
        stack = TiffStackOpener(path, virtual=False)
        return stack.get_stack()
    else:
        logger.error('Please provide either ome.tif or a folder with .tif files')
        raise TypeError

def open_virtual_stack_picasso(path):
    movie, [_] = pio.load_movie(path)
    return movie


def open_virtual_stack(path:str):
    '''
    Opens virtual stack, returns iterable
    :param path: string
    :return: iterable
    '''
    if os.path.isdir(path):
        stack = TiffStackOpener(path,virtual=True)
        return stack.picasso_ome
    elif path.endswith('ome.tif'):
        return open_virtual_stack_picasso(path=path)
    else:
        logger.error('Please provide either ome.tif or a folder with .tif files')
        raise TypeError


def virtual_stack_shape(folder_path, suffix='.tif'):
    fff = []
    flist = os.scandir(folder_path)
    for f in flist:
        if f.name.endswith(suffix):
            fff.append(f)
    shape = io.imread(fff[-1].path).shape
    depth = len(fff)
    return (depth,) + shape


class TiffStackOpener:
    """
    Check the content of the folder, reads both ome.tif stacks in virtual or real mode and sets of .tif files.
    """

    def __init__(self, path, virtual=True):
        self.path = path
        self.movie_template = 'img_{:0>9d}_Default_000.tif'
        self.stack_template = 'img_000000000_Default_{:0>3d}.tif'
        self.picasso_ome = None
        self.stack = None
        self.tif_set = None
        self.file_list = []
        if os.path.isdir(path):
            logger.info('Opening virtual tif set')
            self.open_tif_set()
            self.get_file_list()
            # self.tif_type_selector()
        elif path.endswith('ome.tif'):
            if virtual:
                logger.info('Opening virtual ome.tif stack')
                self.open_ome_tif_virtual()
                self.type = 'ome_virt'
            else:
                logger.info('Opening ome.tif stack to the memory')
                self.open_ome_tif_memory()
                self.type = 'ome_mem'

    def open_ome_tif_virtual(self):
        # Launch picasso tiff reader
        self.picasso_ome = open_virtual_stack_picasso(self.path)

    def open_ome_tif_memory(self):
        # Use skimage.io.imread
        self.stack = io.imread(self.path)
        return self.stack

    def open_tif_set(self):
        # scan thorugh the folder
        self.tif_set = self.tif_type_selector()

    def tif_movie_opener(self, limit=100000, template='img_{:0>9d}_Default_000.tif'):
        for i in range(limit):
            fname = template.format(i)
            path = os.path.join(self.path, fname)
            try:
                f = io.imread(path)
                yield f
            except IOError:
                logger.debug(f"file {os.path.join(path, fname)} doesn't exist, break")
                break

    def tif_zstack_opener(self):
        return self.tif_movie_opener(template='img_000000000_Default_{:0>3d}.tif')

    def tif_type_selector(self):
        if os.path.exists(os.path.join(self.path, self.movie_template.format(1))):
            logger.info('Tif movie detected')
            return self.tif_movie_opener()
        elif os.path.exists(os.path.join(self.path, self.stack_template.format(1))):
            logger.info('Tif stack detected')
            return self.tif_zstack_opener()
        else:
            logging.error(f'''File name pattern not recognized.\n
                          Tried path: {os.path.join(self.path, self.stack_template.format(0))}''')
            return -1

    def get_stack(self):
        if self.tif_set:
            stack = []
            try:
                for i, f in enumerate(self.tif_set):
                    stack.append(f)
                    print(f'\rReading {i}-th frame', end='')
            except KeyboardInterrupt:
                logging.error('\nUser interrupt...')
            finally:
                self.stack = np.array(stack)
                logging.info(f'\nStack shape {self.stack.shape}')
            return self.stack
        elif self.picasso_ome:
            stack = []
            try:
                for i, f in enumerate(self.picasso_ome):
                    stack.append(f)
                    print(f'\rReading {i}-th frame', end='')
            except KeyboardInterrupt:
                logging.error('\nUser interrupt...')
            finally:
                self.stack = np.array(stack)
                logging.info(f'\nStack shape {self.stack.shape}')
            return self.stack

    def get_file_list(self):
        for f in os.scandir(self.path):
            if f.name.endswith('.tif'):
                self.file_list.append(f.name)
        self.file_list = sorted(self.file_list)

    @property
    def n_frames(self):
        if self.picasso_ome:
            return self.picasso_ome.n_frames

        elif self.stack:
            return self.stack.shape[0]

        elif self.tif_set:
            return len(self.file_list)

        else:
            logger.error('Stack type not defined')
            raise AttributeError

    def __len__(self):
        return self.n_frames()

    def __iter__(self):
        for i in self.file_list:
            yield io.imread(os.path.join(self.path,i))

    @property
    def shape(self):
        if self.picasso_ome:
            return self.picasso_ome.shape

        elif self.stack:
            return self.stack.shape

        elif self.tif_set:
            depth = len(self.file_list)
            img = io.imread(os.path.join(self.path, self.file_list[0]))
            shape = img.shape
            return (depth,) + shape

        else:
            logger.error('Stack type not defined to invoke the shape')
            raise AttributeError('Unable to get stack shape as stack is not properly loaded')

    def __getitem__(self, i):
        # return n-th item
        if self.picasso_ome:
            return self.picasso_ome[i]
        elif self.stack:
            return self.stack[i]
        elif self.tif_set:
            return io.imread(os.path.join(self.path, self.file_list[i]))
        else:
            logger.error('Stack type not defined')
            raise AttributeError('Unable to get item as stack is not properly loaded')



def save_drift_table(table: np.ndarray, path):
    save_table(table, path, fmt='drift_table')


def save_table(table, path: str, fmt: str = 'drift_table'):
    if fmt == 'drift_table':
        try:
            logger.info(f'Saving results to {path}')
            np.savetxt(path + ".csv", table, fmt='%.1f', delimiter=',', comments='', newline='\r\n',
                       header='"frame","x [nm]","y [nm]","z [nm]"')
            print('')
        except IOError as e:
            logger.error('Problem saving drift table: ')
            logger.error(e.strerror)
    else:
        logger.error('Please specify valid format: ["drift_table"]')
        logger.error('File NOT saved!')


def get_abs_path(path):
    dir_path = os.path.abspath(path)
    return dir_path


def get_parent_path(path):
    return os.path.abspath(os.path.join(path, os.pardir))


def open_csv_table(path, show_header=False):
    """
    Loads thunderstorm compatible csv table into numpy array
    First line is omitted as contains header
    :param path: path to the table.csv
    :param show_header: bool, shows 'head path'
    :return: numpy array
    """
    if show_header:
        subprocess.check_output(['head', '-n', '1', path])
    return np.loadtxt(path, delimiter=',', skiprows=1)


def save_zola_table(table, path):
    header = 'id,frame,x [nm],y [nm],z [nm],intensity,background,chi2,crlbX,crlbY,crlbZ,driftX,driftY,driftZ,' \
             'occurrenceMerging '
    np.savetxt(path, table[:, :15], fmt='%.2f', delimiter=',', comments='', newline='\r\n', header=header)


def plot_drift(table):
    plt.plot(table[:, 0], table[:, 1:])
    plt.xlabel('frame')
    plt.ylabel('Drift, nm')
    plt.legend(['x', 'y', 'z'])
    plt.title('Drift BF, nm')
    plt.grid()


def save_drift_plot(table, path, callback=None):
    plot_drift(table)
    plt.savefig(path)
    plt.close()
    if callback: callback({"Plot":path})
    logger.info(f"Saved drift plot to {path}")


def interpolate_drift_table(table, start=0, skip=0, smooth=10):
    """
    Smooth and interpolate a table
    :param table: fxyz (nm) array
    :param start: in case of renumbering needed : first frame
    :param skip: how many frame were skipped
    :param smooth: gaussian smoothing sigma
    :return: interpolated table
    """
    w = table.shape[1]
    if smooth > 0:
        table = smooth_drift_table(table, sigma=smooth)

    table = update_frame_number(table, start=start, skip=skip)

    time = table[:, 0]
    # print(time.shape)
    time_new = np.arange(1, max(time) + 1)
    new_table = np.zeros((len(time_new), w))
    new_table[:, 0] = time_new
    for col in range(1, w):
        y = table[:, col]
        # print(y.shape)
        f = interpolate.interp1d(time, y, fill_value='extrapolate')
        ynew = f(time_new)
        new_table[:, col] = ynew
    logger.info(f'interpolating from {len(time)} to {len(ynew)} frames')
    return new_table


def smooth_drift_table(table, sigma):
    drift = gf1(table[:, 1:], sigma=sigma, axis=0)
    table_smooth = table.copy()
    table_smooth[:, 1:] = drift
    return table_smooth


def check_stacks_size_equals(cal_stack, movie):
    logger.info(f'check_stacks_size_equals: Input shapes {cal_stack.shape,movie.shape}')
    if len(cal_stack.shape) == len(movie.shape) == 3:
        x1, x2 = cal_stack.shape[1], cal_stack.shape[2]
        y1, y2 = movie.shape[1], movie.shape[2]
        return x1 == y1 and x2 == y2
    else:
        raise (ValueError('cal_stack.shape: wrong shapes!'))


def check_multi_channel(movie, channel=2, channel_position=1):
    """
    Checks if stack contains channels and returns single channel stack
    :param movie: numpy array zxy or zcxy
    :param channel: 1 - first channel, 2 - second channel, etc
    :param channel_position: 1 - for zcxy, 0 - for czxy
    :return: numpy array zxy
    """
    logger.info(f'check_multi_channel: Input shape {movie.shape}')
    ndim = len(movie.shape)
    if ndim == 3:
        logger.info(f'check_multi_channel: Returning shape {movie.shape}')
        return movie
    elif ndim == 4:
        if channel_position == 1:
            logger.info(f'check_multi_channel: Returning shape {movie[:,channel-1].shape}')
            return movie[:, channel - 1]
        elif channel_position == 0:
            logger.info(f'check_multi_channel: Returning shape {movie[channel-1].shape}')
            return movie[channel - 1]
    else:
        raise (TypeError(f'check_multi_channel: channel order not understood, movie shape {movie.shape}'))


def skip_stack(n_frames: int, start: int, skip: int, maxframes: int):
    """
    Now works with virtual stack
    :param n_frames: total frame number
    :param start: in case of skipping: first frame to pick up (starts form 1)
    :param skip: number of frames skipped to get the right frame
            (for example, ch2 with alternating illumination refers to start=2,skip=1)
    :param maxframes: maximum number of frames in case of cropped dataset
    :return: index list
    """
    # logger.info('skip_stack: starting frame skipping routines')

    index_list = np.arange(n_frames)
    if start > 0:
        start = start - 1
    if skip == 0:
        skip = None
    if maxframes == 0:
        maxframes = None
    index_list = index_list[start:maxframes:skip]
    logger.info(f'skip_stack: returning frame list with {len(index_list)} frames')
    return index_list


def update_frame_number(table, start, skip):
    """
    updates frame number int the table using skip/start frames indices
    :param table: fxyz array
    :param start: first frame of selection
    :param skip: every skip-th frame from selection
    :return: table with updated frame column
    """
    if skip > 0 or start > 0:
        if table[0, 0] == 1:
            table[:, 0] -= 1
        elif table[0, 0] == 0:
            pass
        else:
            raise (ValueError("update_frame_number: Wrong table. Expected frame numbers starting with 0 or 1"))
        table[:, 0] *= skip
        table[:, 0] += start - 1
        logger.info("update_frame_number: Updated frame numbers successfully")
    return table


def put_trace_lock(path, name="BFDC_.lock"):
    f = open(path + os.sep + name, mode='w')
    f.close()
    logger.info('Setting lock')
    return path + os.sep + name


def remove_trace_lock(path):
    try:
        os.remove(path)
        logger.info('Removing lock')
        return 0
    except IOError:
        logger.error('Problem removing lock')
        return 1


def parse_input():
    # Main parser
    parser = argparse.ArgumentParser('BFDC')
    subparsers = parser.add_subparsers(dest='command')

    for command in ['trace', 'apply', 'batch']:
        subparsers.add_parser(command)

    # trace
    trace_parser = subparsers.add_parser('trace', help='identify drift in 3D')
    trace_parser.add_argument('dict', type=str, default='data/LED_stack_full_100nm.tif',
                              help='calibration stack file')
    trace_parser.add_argument('roi', type=str, default='',
                              help='calibration file roi from ImageJ')
    trace_parser.add_argument('movie', type=str, default='data/sr_2_LED_movie.tif',
                              help='movie stack file')
    trace_parser.add_argument('-z', '--zstep', type=int, default=100, help='z-step in nm. Default: 100')
    trace_parser.add_argument('--zdirection', type=str, default='approach', help='Choose approach/retract for the direction of calibration. Default: approach')
    trace_parser.add_argument('-xypx', '--xypixel', type=int, default=110, help='xy pixel size in nm. Default: 110')
    trace_parser.add_argument('--nframes', type=int, default=None,
                              help='now many frames to analyse from the movie. Default: None')
    trace_parser.add_argument('--driftFileName', type=str, default='BFCC_table',
                              help='filename for the drift table. Default: "BFCC_table.csv"')
    trace_parser.add_argument('--minsignal', type=int, default=100,
                              help='Threshold of mean intensity to treat the image as brightfield. Default: 100')
    trace_parser.add_argument('--skip', type=int, default=0,
                              help='how many frames to skip form the movie. Default: 0')
    trace_parser.add_argument('--start', type=int, default=0,
                              help='how many frames to skip in the beginning of the movie. Default: 0')
    trace_parser.add_argument('--channel', type=int, default=2,
                              help='channel index (starts with 1) for the movie. Default: 2')
    trace_parser.add_argument('--channel_position', type=int, default=1,
                              help='channel position (starts with 0) for the movie. Default: 1')
    trace_parser.add_argument('--lock', type=int, default=0,
                              help='if 1, will create BFDC_.lock file in the movie folder. Default: 0')

    # apply
    apply_parser = subparsers.add_parser('apply', help='apply drift 3D to ZOLA table')
    apply_parser.add_argument('zola_table', type=str, default='',
                              help='ZOLA localization table, format ifxyz.......dxdydz')
    apply_parser.add_argument('drift_table', type=str, default='BFCC_table.csv',
                              help='3D drift table, format fxyz')
    apply_parser.add_argument('--skip', type=int, default=0,
                              help='how many frames to skip form the movie. Default: 0')
    apply_parser.add_argument('--start', type=int, default=0,
                              help='how many frames to skip in the beginning of the movie. Default: 0')

    apply_parser.add_argument('--smooth', type=int, default=0, help='gaussian smoothing for the drift. Default: 0')
    apply_parser.add_argument('--maxbg', type=int, default=0,
                              help='reject localizations with high background. Default: 0')
    apply_parser.add_argument('--zinvert', type=int, default=0, help='invert z axis for drift. Default: 0')

    # batch
    batch_parser = subparsers.add_parser('batch', help='''Batch trace and apply drift 3D to ZOLA table''')
    batch_parser.add_argument('batch_path', type=str, default='',
                              help='data path')
    batch_parser.add_argument('-z', '--zstep', type=int, default=100, help='z-step in nm. Default: 100')
    batch_parser.add_argument('--zdirection', type=str, default='approach',
                              help='Choose approach/retract for the direction of calibration. Default: approach')
    batch_parser.add_argument('-xypx', '--xypixel', type=int, default=110, help='xy pixel size in nm. Default: 110')
    batch_parser.add_argument('--skip', type=int, default=0,
                              help='how many frames to skip form the movie. Default: 0')
    batch_parser.add_argument('--start', type=int, default=0,
                              help='how many frames to skip in the beginning of the movie. Default: 0')

    batch_parser.add_argument('--fov_prefix', type=str, default='FOV',
                              help='Prefix of a folder with a single field of view. Default: FOV ')

    batch_parser.add_argument('--dict_folder_prefix', type=str, default='dict',
                              help='Prefix of the dictionary folder')

    batch_parser.add_argument('--ROI_suffix', type=str, default='roi',
                              help='Extension of roi file. Default: roi')

    batch_parser.add_argument('--dict_suffix', type=str, default='ome.tif',
                              help='Extension of dictionary stack file. Default: ome.tif')

    batch_parser.add_argument('--sr_folder_prefix', type=str, default='sr',
                              help='Prefix of the super-resolution movie folder. Default: sr')

    batch_parser.add_argument('--sr_movie_suffix', type=str, default='Pos0.ome.tif',
                              help='Extension of super-resolution movie stack file. Default: Pos0.ome.tif')

    batch_parser.add_argument('--zola_dc_filename', type=str, default='ZOLA*BFDC*.csv',
                              help='ZOLA DC table name. Default: ZOLA*BFDC*.csv')

    batch_parser.add_argument('--dc_table_filename', type=str, default='BFCC*.csv',
                              help='Drift track table name. Default: BFCC*.csv')

    batch_parser.add_argument('--zola_raw_filename', type=str, default='ZOLA_localization_table.csv',
                              help='ZOLA localization table name. Default: ZOLA_localization_table.csv')

    batch_parser.add_argument('--zola_lock_filename', type=str, default='ZOLA_.lock',
                              help='ZOLA DC table name. Default: ZOLA_.lock')

    batch_parser.add_argument('--smooth', type=int, default=50, help='gaussian smoothing for the drift. Default: 50')
    batch_parser.add_argument('--filter_bg', type=int, default=100,
                              help='''Use this value to detect BF frames. 
                                   Frames with the mean value more than this number are counted as bright field.
                                   In the reconstuction table, background more than this number will be filtered out.
                                   Default: 100
                                   ''')
    batch_parser.add_argument('--zinvert', type=int, default=0, help='invert z axis for drift. Default: 0')

    return parser
