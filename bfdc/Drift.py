"""
TODO: Split track and apply functionality
TODO: Pass args to main
TODO: Integrity test

"""
import traceback

# from scipy.ndimage import gaussian_filter1d as gf1
from skimage import io

from bfdc.CrossCorrelation import *
from bfdc.feature import *
from bfdc.iotools import *
from picasso import io as pio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WrongCrop(Exception):
    pass


class DriftFitter:
    """
    Automatic Drift Fitter
    Crops the stack and tracks the drift
    """

    def __init__(self, cal_stack, roi):
        self.boundaries = roi_to_boundaries(roi)
        self.dict = crop_using_xy_boundaries(cal_stack, boundaries=self.boundaries)
        self.z_crop = (0, -1)
        self.x_correction = 0
        self.y_correction = 0
        self.zCenter = len(self.dict) // 2
        self.radius_xy = 3

    def doTrace(self, movie, frame_list, extend_xy=5, debug=False):
        logging.info(f"doTrace: got the movie with shape {movie.shape}, using {len(frame_list)} frames for tracing")
        # for i,frame in enumerate(movie):
        # crop frame with extended by 5px boundaries
        # run xcorr drift loc
        # extract central z frame
        # use +5-5 frames out of z stack
        # extract xy position
        # change boundaries if needed
        b = self.boundaries
        xc = np.mean([b["xmax"], -b["xmin"]])
        yc = np.mean([b["ymax"], -b["ymin"]])
        out = np.empty((0, 4))
        x_, y_, z_ = 0, 0, 0
        total = len(frame_list)
        problems = []
        i = 0

        try:
            for i, f in enumerate(frame_list):
                frame = movie[f]
                logging.debug(f'frame {i+1}')
                crop_frame = crop_using_xy_boundaries(frame, b, extend=extend_xy)

                logging.debug(f'Cropping frame {crop_frame.shape}')
                if min(crop_frame.shape) == 0:
                    raise (WrongCrop(f"doTrace: problem with boundaries: crop size hits 0 {crop_frame.shape}"))
                crop_dict = self.crop_dict()
                cc = cc_stack(crop_frame, crop_dict)
                # out.append(cc_max(cc))
                x, y, z = fit_gauss_3d(cc, radius_xy=self.radius_xy, radius_z=5, z_zoom=20, debug=debug)

                z_ = z + self.z_crop[0] - self.zCenter
                x_ = x + self.x_correction - xc - self.radius_xy
                y_ = y + self.y_correction - yc - self.radius_xy
                logger.debug(f"x_px = {x}, y_px = {y}, \
                                            x_correction = {self.x_correction}, y_correction = {self.y_correction}")

                out = np.append(out, np.array([i + 1, x_, y_, z_]).reshape((1, 4)), axis=0)
                logging.debug(f'found xyz {x,y,z}')
                self.update_z_crop(z + self.z_crop[0])
                self.update_xy_boundaries(x, y, extend_xy)

                print('\r{}/{}'.format(i + 1, total), end=' ')

        except LowXCorr:
            logging.warning(f'Low cross correlation value for the frame {i+1}. Filling with the previous frame values')
            if len(out):
                out = np.append(out, np.array([i + 1, x_, y_, z_]).reshape((1, 4)), axis=0)
            problems.append(i + 1)

        except Exception as e:
            print(e)
            traceback.print_stack()
            problems.append(i + 1)

        finally:
            n = len(problems)
            print(f'\nDone tracing with {n} problem frames')
            if n:
                print(problems)

            return np.array(out)

    def update_z_crop(self, z):
        z1, z2 = np.max([0, int(z - 5)]), np.min([int(z + 7), len(self.dict) - 1])
        logger.debug(f'update_z_crop:z boundaries {z1},{z2}')
        self.z_crop = (z1, z2)

    def crop_dict(self):
        return self.dict[self.z_crop[0]:self.z_crop[1]]

    def update_xy_boundaries(self, x, y, extend):
        b = self.boundaries
        e = extend
        x0 = (b['xmax'] - b['xmin']) / 2 + e
        y0 = (b['ymax'] - b['ymin']) / 2 + e
        xdif = int(np.round(x - x0, decimals=0))
        ydif = int(np.round(y - y0, decimals=0))
        if abs(xdif) > 0 or abs(ydif) > 0:
            logging.debug(f'moving xy boundary by {xdif},{ydif}')
            b['xmin'] += xdif
            b['xmax'] += xdif
            self.x_correction += xdif
            b['ymin'] += ydif
            b['ymax'] += ydif
            self.y_correction += ydif


def get_drift_3d(movie, frame_list, cal_stack, debug=False):
    """
    Computes drift on the movie vs cal_staack in 3D
    :param movie: time series stack
    :param frame_list: list with frame indices (starting with 0)
    :param cal_stack: calibration z stack
    :param debug: shows the fits
    :return: table[frame,x,y,z] in pixels
    """
    out = []
    total = len(movie)
    problems = []
    i = 0
    try:
        for i in frame_list:
            frame = movie[i]
            cc = cc_stack(frame, cal_stack)
            # out.append(cc_max(cc))
            x, y, z = fit_gauss_3d(cc, debug=debug)
            out.append([i + 1, x, y, -z])
            print('\r{}/{}'.format(i + 1, total), end=' ')
    except Exception as e:
        print(e)
        problems.append(i + 1)

    finally:
        n = len(problems)
        print(f'\nDone tracing with {n} problem frames')
        if n:
            print(problems)

        return np.array(out)


def trace_drift(args, cal_stack, movie, debug=False):
    """
    Computes 3D drift on the movie vs cal_stack
    :param debug: Plot data and fit if True
    :param args: dict[args.xypixel, args.zstep,args.nframes,args.skip,args.start]
    :param cal_stack: 3d stack dictionary
    :param movie: time series 3D stack
    :return: table[frame,x,y,z] in nm
    """
    print("tracing drift!")
    px = [args.xypixel, args.xypixel, args.zstep]
    skip = args.skip
    start = args.start
    nframes = args.nframes

    print(f'Pixel size xyz: {px}')
    drift_px = np.zeros(4)
    movie, frame_list = skip_stack(movie, start=start, skip=skip, nframes=nframes)
    try:
        drift_px = get_drift_3d(movie=movie, frame_list=frame_list, cal_stack=cal_stack, debug=debug)
    except KeyboardInterrupt as e:
        print(e)

    drift_nm = drift_px.copy()
    drift_nm[:, 1:] = drift_px[:, 1:] * px
    drift_nm = update_frame_number(drift_nm, start, skip)
    return drift_nm


def trace_drift_auto(args, cal_stack, movie, roi, debug=False):
    """
    Computes 3D drift on the movie vs cal_stack with auto crop
    :param debug: plot data and fit if True
    :param args: dict[args.xypixel, args.zstep]
    :param cal_stack: 3d z-stack
    :param movie: time series 3D stack
    :param roi: readout of IJ roi file
    :return: table[frame,x,y,z] in nm
    """
    print("tracing drift with automatic feature detection!")
    px = [args.xypixel, args.xypixel, args.zstep]
    skip = args.skip
    start = args.start
    nframes = args.nframes

    print(f'Pixel size xyz: {px}')
    drift_px = np.zeros((1, 4))

    fitter = DriftFitter(cal_stack, roi)

    frame_list = skip_stack(movie.n_frames, start=start, skip=skip, nframes=nframes)
    try:
        drift_px = fitter.doTrace(movie, frame_list=frame_list, debug=debug)
    except KeyboardInterrupt as e:
        print(e)

    drift_nm = drift_px.copy()
    drift_nm[:, 1:] = drift_px[:, 1:] * px
    drift_nm = update_frame_number(drift_nm, start, skip)
    return drift_nm


def move_drift_to_zero(drift_nm, ref_average=10):
    """
    moves to zero the start of the table
    :param drift_nm: fxyz table
    :param ref_average: how many frames to average to get zero
    :return: shifted table
    """
    drift_ref = drift_nm[0:ref_average, :].mean(axis=0)
    drift_ref[0] = 0  # frame number should be 0 for reference
    drift_ = drift_nm - drift_ref
    return drift_


def apply_drift(zola_table, bf_table):
    zola_frame_num = len(np.unique(zola_table[:, 1]))

    bf_frame_num = len(np.unique(bf_table[:, 0]))

    print(f'Frame number for zola/bf_DC : {zola_frame_num}/{bf_frame_num}')

    if bf_frame_num < zola_frame_num:
        print(f'Truncating ZOLA table to {bf_frame_num} frames')
        zola_table = zola_table[zola_table[:, 1] < bf_frame_num]
        fnum = len(np.unique(zola_table[:, 1]))
        print(f'New frame number: {fnum}')

    frame_nums = np.array(zola_table[:, 1], dtype='int')
    bf_drift_framed = bf_table[frame_nums - 1]

    zola_table_dc = zola_table.copy()
    zola_table_dc[:, [2, 3, 4]] = zola_table_dc[:, [2, 3, 4]] - bf_drift_framed[:, [1, 2, 3]]
    zola_table_dc[:, [11, 12, 13]] = bf_drift_framed[:, [1, 2, 3]]
    return zola_table_dc


def mymain(myargs=None):
    parser = parse_input()
    if myargs is None:
        args = parser.parse_args(myargs)
    else:
        args = parser.parse_args()
    logger.debug(args)

    if args.command == 'trace':
        cal_path = get_abs_path(args.dict)
        logger.info(f'\nOpening calibration {args.dict}')
        cal_stack = io.imread(cal_path)
        logger.info(f'Imported dictionary {cal_stack.shape}')

        roi_path = get_abs_path(args.roi)
        logger.info(f'\nOpening roi {args.roi}')
        roi = read_roi(roi_path)

        movie_path = get_abs_path(args.movie)
        logger.info(f'\nOpening movie {args.movie}')
        # movie = io.imread(movie_path)
        movie, [_] = pio.load_movie(movie_path)
        logger.info(f'Imported movie {movie.shape}')

        # ch_index = args.channel
        # ch_pos = args.channel_position

        # movie = check_multi_channel(movie,channel=ch_index,channel_position=ch_pos)
        size_check = check_stacks_size_equals(cal_stack, movie)

        if size_check:
            logger.info('Stack and movie of equal sizes, attempting auto crop')
            drift_ = trace_drift_auto(args=args, cal_stack=cal_stack, movie=movie, roi=roi, debug=False)
        else:
            logger.info('Stack and movie of different sizes, running on full size')
            drift_ = trace_drift(args, cal_stack, movie)

        if drift_.shape[0] > 0:
            movie_folder = get_parent_path(movie_path)
            save_path = os.path.join(movie_folder, args.driftFileName)
            save_drift_table(drift_, save_path)
            save_drift_plot(move_drift_to_zero(drift_, 10), save_path + "_2zero" + ".png")

            logger.info('Drift table saved, exiting')
        else:
            logger.info('Drift table empty, exiting')

    if args.command == 'apply':
        logger.debug(args)
        zola_path = get_abs_path(args.zola_table)
        bf_path = get_abs_path(args.drift_table)

        logger.info(f'Opening data')
        zola_table = open_csv_table(zola_path)
        logger.info(f'Zola table contains {len(zola_table)} localizations \
                    from {len(np.unique(zola_table[:,1]))} frames')
        bf_table = open_csv_table(bf_path)

        if args.smooth > 0:
            bf_table[:, 1:4] = gf1(bf_table[:, 1:4], sigma=args.smooth, axis=0)

        logger.info(f'Applying drift')
        zola_table_dc = apply_drift(bf_table=bf_table, zola_table=zola_table)

        path = os.path.splitext(zola_path)[0] + f'_BFDC_smooth_{args.smooth}.csv'
        logger.info(f'saving results to {path}')
        save_zola_table(zola_table_dc, path)
        save_drift_plot(bf_table, os.path.splitext(path)[0] + '.png')

    else:
        parser.print_help()

    return 0


if __name__ == '__main__':
    mymain()
