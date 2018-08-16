from bfdc.drift import *
from skimage import io
import bfdc.picassoio as pio
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        if args.lock:
            lock = put_trace_lock(os.path.dirname(movie_path))
        logger.info(f'\nOpening movie {args.movie}')
        # movie = io.imread(movie_path)
        movie, [_] = pio.load_movie(movie_path)
        logger.info(f'Imported movie {movie.shape}')

        # ch_index = args.channel
        # ch_pos = args.channel_position

        # movie = check_multi_channel(movie,channel=ch_index,channel_position=ch_pos)
        size_check = check_stacks_size_equals(cal_stack, movie)

        if size_check:
            logger.info('Stack and movie of equal sizes')
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


        if args.lock:
            unlock = remove_trace_lock(lock)

    if args.command == 'apply':
        logger.debug(args)
        zola_path = get_abs_path(args.zola_table)
        bf_path = get_abs_path(args.drift_table)

        logger.info(f'Opening data')
        zola_table = open_csv_table(zola_path)
        logger.info(f'Zola table contains {len(zola_table)} localizations from {len(np.unique(zola_table[:,1]))} frames')
        bf_table = open_csv_table(bf_path)

        if args.smooth > 0:
            bf_table[:, 1:4] = gf1(bf_table[:, 1:4], sigma=args.smooth, axis=0)

        logger.info(f'Applying drift')
        zola_table_dc, bf_table_int = apply_drift(bf_table=bf_table,
                                                  zola_table=zola_table,
                                                  start=args.start,
                                                  skip=args.skip,
                                                  maxbg=args.maxbg,
                                                  zinvert=args.zinvert)

        path = os.path.splitext(zola_path)[0] + f'_BFDC_smooth_{args.smooth}.csv'
        logger.info(f'saving results to {path}')
        save_zola_table(zola_table_dc, path)
        save_drift_plot(move_drift_to_zero(bf_table_int), os.path.splitext(path)[0] + '.png')

    else:
        parser.print_help()

    return 0


if __name__ == '__main__':
    mymain()