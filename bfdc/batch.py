from glob import glob
from bfdc.drift import *
import bfdc.drift as dr

def batch_drift(path,
                fov_prefix='FOV',
                dict_folder_prefix='dict_',
                roi_suffix='roi',
                dict_suffix='ome.tif',
                sr_folder_prefix='sr_',
                sr_movie_suffix='Pos0.ome.tif',
                zola_dc_filename="ZOLA*BFCC*.csv",
                dc_tabel_filename="BFCC*.csv",
                zola_raw_filename="ZOLA_localization_table.csv",
                zola_lock_filename="ZOLA_.lock",
                apply_smooth=50,
                filter_bg=100
                ):

    sep = os.path.sep
    base = lambda path: os.path.basename(os.path.normpath(path))
    relative = lambda path, parent: path[len(parent):]
    parent = os.path.dirname

    def parse_fovs(path):
        fov_list = glob(pathname=path + sep + fov_prefix + "*" + sep)
        logger.info(f'Found {len(fov_list)} folders starting with FOV')
        logger.info(fov_list)
        return fov_list

    def find_file(path, name, pattern, expected_number=None):
        try:
            flist = glob(path + sep + pattern)
            #flist[0]
            if expected_number:
                assert len(flist) == expected_number, f'Found {len(flist)} {name}s, while expected {expected_number}'
            for f in flist:
                logger.info(f'{base(path)}: Found {name}: {base(f)}')
            return flist
        except IndexError:
            logger.info(f'{base(path)}: No {name}')
            return None

    def find_dict(roi_path):
        return find_file(path=parent(roi_path),
                         name='dict',
                         pattern=dict_folder_prefix + "*" + dict_suffix,
                         expected_number=1)

    def find_roi(fov_path):
        return find_file(path=fov_path,
                         name='ROI',
                         pattern=dict_folder_prefix + "*" + sep + "*" + roi_suffix,
                         expected_number=None)

    def find_movies(fov_path):
        return find_file(path=fov_path,
                         name='movie',
                         pattern=sr_folder_prefix + "*" + sep + "*" + sr_movie_suffix)

    def batch_trace_drift(bfdict, roi, movie):
        bfout = glob(parent(movie) + sep + dc_tabel_filename)
        lock = glob(parent(movie) + sep + zola_lock_filename)
        if bfout == [] and lock == []:
            logger.info("start BF tracking")

            args = ["trace", bfdict, roi, movie, '--lock', '1']
            dr.commandline_main(args=args)

        elif len(lock) == 1:
            logger.info('Found BFDC_.lock --- skipping')

    def batch_apply_drift(movie:str,
                    smoothing:int=apply_smooth,
                    max_bg:int=filter_bg
                    ):
        zola_dc_table = glob(parent(movie) + sep + zola_dc_filename)
        if zola_dc_table == []:
            drift_tables = glob(parent(movie) + sep + dc_tabel_filename)
            for drift_table in drift_tables:
                logger.info(f"Found {base(drift_table)}")
                zola_table = glob(parent(drift_table) + sep + zola_raw_filename)
                z_lock = glob(parent(drift_table) + sep + zola_lock_filename)
                if len(zola_table) == 1 and len(z_lock) == 0:
                    logger.info(f'Found {base(zola_table[0])}')
                    logger.info("Apply drift")

                    args = ["apply", zola_table[0], drift_table, "--smooth", smoothing, "--maxbg", max_bg]
                    dr.commandline_main(args=args)
                elif len(z_lock) == 1:
                    logger.info(f'Found {base(z_lock[0])} --- skipping')
                else:
                    logger.info('No ZOLA table found')
        else:
            logger.info('Folder already processed')

    # root = tk.Tk()
    # root.withdraw()
    # path = filedialog.askdirectory()
    # path = input("Please provide a folder path:")
    # path = os.path.abspath(path)
    logger.info(path)
    fov_list = parse_fovs(path)
    logger.info(f'Found {len(fov_list)} folders starting with FOV')
    # print('#FOV,\tROI,\tlocs,\tgir(nm),\tellipticity')

    for fov in sorted(fov_list):
        roi = find_roi(fov)
        if roi:
            bfdict = find_dict(roi[0])[0]
            movies = find_movies(fov)
            for movie in movies:
                batch_trace_drift(bfdict, roi[0], movie)
                batch_apply_drift(movie)

            # print('')
        else:
            logger.info('No ROI')
    return 0



