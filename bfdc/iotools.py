import argparse

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d as gf1
import subprocess

import logging

logger = logging.getLogger(__name__)


def save_drift_table(table: np.ndarray, path):
    save_table(table, path, fmt='drift_table')


def save_table(table, path: str, fmt: str = 'drift_table'):
    if fmt == 'drift_table':
        try:
            logger.info(f'Saving results to {path}')
            np.savetxt(path+".csv", table, fmt='%.1f', delimiter=',', comments='', newline='\r\n',
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


def open_csv_table(path,showHeader = False):
    """
    Loads thunderstorm compatible csv table into numpy array
    First line is omitted as contains header
    :param path: path to the table.csv
    :param showHeader: bool, shows 'head path'
    :return: numpy array
    """
    if showHeader:
        subprocess.check_output(['head','-n','1',path])
    return np.loadtxt(path, delimiter=',', skiprows=1)


def save_zola_table(table, path):
    header = 'id,frame,x [nm],y [nm],z [nm],intensity,background,chi2,crlbX,crlbY,crlbZ,driftX,driftY,driftZ,' \
             'occurrenceMerging '
    np.savetxt(path, table[:,:15], fmt='%.2f', delimiter=',', comments='', newline='\r\n', header=header)


def save_drift_plot(table, path):
    fig = plt.figure()
    plt.plot(table[:, 0], table[:, 1:])
    plt.xlabel('frame')
    plt.ylabel('Drift, nm')
    plt.legend(['x', 'y', 'z'])
    plt.title('Drift BF, nm')
    plt.grid()
    plt.savefig(path)
    plt.close()
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
    timeNew = np.arange(1, max(time) + 1)
    newTable = np.zeros((len(timeNew), w))
    newTable[:, 0] = timeNew
    for col in range(1, w):
        y = table[:, col]
        # print(y.shape)
        f = interpolate.interp1d(time, y, fill_value='extrapolate')
        ynew = f(timeNew)
        newTable[:, col] = ynew
    logger.info(f'interpolating from {len(time)} to {len(ynew)} frames')
    return newTable


def smooth_drift_table(table, sigma):
    drift = gf1(table[:, 1:], sigma=sigma, axis=0)
    table_smooth = table.copy()
    table_smooth[:, 1:] = drift
    return table_smooth


def check_stacks_size_equals(cal_stack,movie):
    logger.info(f'check_stacks_size_equals: Input shapes {cal_stack.shape,movie.shape}')
    if len(cal_stack.shape) == len(movie.shape) == 3:
        x1,x2 = cal_stack.shape[1],cal_stack.shape[2]
        y1,y2 = movie.shape[1],movie.shape[2]
        return x1 == y1 and x2 == y2
    else:
        raise(ValueError('cal_stack.shape: wrong shapes!'))


def check_multi_channel(movie,channel = 2, channel_position = 1):
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
            return movie[:,channel-1]
        elif channel_position == 0:
            logger.info(f'check_multi_channel: Returning shape {movie[channel-1].shape}')
            return movie[channel-1]
    else:
        raise(TypeError(f'check_multi_channel: channel order not understood, movie shape {movie.shape}'))


def skip_stack(n_frames:int, start:int, skip:int, maxframes:int):
    """
    Now works with virtual stack
    :param n_frames: total frame number
    :param start: in case of skipping: first frame to pick up (starts form 1)
    :param skip: number of frames skipped to get the right frame (for example, ch2 with alternating illumination refers to start=2,skip=1)
    :param maxframes: maximum number of frames in case of cropped dataset
    :return: index list
    """
    #logger.info('skip_stack: starting frame skipping routines')

    index_list = np.arange(n_frames)
    if start > 0:
        start = start - 1
        index_list = index_list[start:maxframes:skip + 1]
        logger.info(f'skip_stack: returning frame list with {len(index_list)} frames')
    return index_list


def update_frame_number(table,start,skip):
    """
    updates frame number int the table using skip/start frames indices
    :param table: fxyz array
    :param start: first frame of selection
    :param skip: every skip-th frame from selection
    :return: table with updated frame column
    """
    if skip > 0 or start > 0:
        if table[0, 0] == 1:
            table[:, 0] -=1
        elif table[0, 0] == 0:
            pass
        else:
            raise(ValueError("update_frame_number: Wrong table. Expected frame numbers starting with 0 or 1"))
        table[:,0] *= skip
        table[:,0] += start - 1
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

    for command in ['trace', 'apply']:
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
    trace_parser.add_argument('-xypx', '--xypixel', type=int, default=110, help='xy pixel size in nm. Default: 110')
    trace_parser.add_argument('--nframes', type=int, default=None, help='now many frames to analyse from the movie. Default: None')
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
    apply_parser.add_argument('--maxbg', type=int, default=0, help='reject localizations with high background. Default: 0')
    apply_parser.add_argument('--zinvert',type=int, default=0, help='invert z axis for drift. Default: 0')

    return parser
