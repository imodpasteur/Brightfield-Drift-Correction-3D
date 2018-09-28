from bfdc.drift import *
import tkinter as tk
from tkinter import filedialog
import os
import logging
from glob import glob

logger = logging.getLogger(__name__)

sep = os.path.sep
base = lambda path: os.path.basename(os.path.normpath(path))
relative = lambda path, parent: path[len(parent):]
parent = os.path.dirname


class BatchDrift:

    def __init__(self, batch_path,
                fov_prefix='FOV',
                dict_folder_prefix='dict_',
                roi_suffix='roi',
                dict_suffix='ome.tif',
                sr_folder_prefix='sr',
                sr_movie_suffix='Pos0.ome.tif',
                zola_dc_filename="ZOLA*BFDC*.csv",
                dc_table_filename="BFCC*.csv",
                zola_raw_filename="ZOLA_localization_table.csv",
                zola_lock_filename="ZOLA_.lock",
                smooth=50,
                filter_bg=100,
                **kwargs):
        self.batch_path = batch_path
        self.fov_prefix = fov_prefix
        self.dict_folder_prefix = dict_folder_prefix
        self.roi_suffix = roi_suffix
        self.dict_suffix = dict_suffix
        self.sr_folder_prefix = sr_folder_prefix
        self.sr_movie_suffix = sr_movie_suffix
        self.zola_dc_filename = zola_dc_filename
        self.dc_table_filename = dc_table_filename
        self.zola_raw_filename = zola_raw_filename
        self.zola_lock_filename = zola_lock_filename
        self.smooth = smooth
        self.filter_bg = filter_bg

        self.do_batch()

    def parse_fovs(self,path):
        fov_list = glob(pathname=path + sep + self.fov_prefix + "*" + sep)
        logger.info(f'Found {len(fov_list)} folders starting with FOV')
        for i,f in enumerate(sorted(fov_list)):
            logger.info(f'{i} -- {relative(f,path)}')
        return fov_list

    def find_file(self, path, name, pattern, expected_number=None):
        try:
            flist = glob(path + sep + pattern)
            if expected_number:
                assert len(flist) == expected_number, f'Found {len(flist)} {name}s, while expected {expected_number}'
            if flist:
                for f in flist:
                    logger.info(f'{base(path)}: Found {name}: {relative(f,path)}')
            else:
                logger.info(f'No {name} was found using pattern \"{pattern}\"')
            return flist
        except IndexError:
            logger.info(f'{base(path)}: No {name}')
            return None

    def find_dict(self, roi_path):
        return self.find_file(path=parent(roi_path),
                              name='dict',
                              pattern=self.dict_folder_prefix + "*" + self.dict_suffix,
                              expected_number=1)

    def find_roi(self, fov_path):
        return self.find_file(path=fov_path,
                              name='ROI',
                              pattern=self.dict_folder_prefix + "*" + sep + "*" + self.roi_suffix,
                              expected_number=None)

    def find_movies(self, fov_path):
        return self.find_file(path=fov_path,
                              name='movie',
                              pattern=self.sr_folder_prefix + "*" + sep + "*" + self.sr_movie_suffix)

    def batch_trace_drift(self, bfdict, roi, movie):
        bfout = glob(parent(movie) + sep + self.dc_table_filename)
        if not bfout:
            logger.info("start BF tracking")

            args = ["trace", bfdict, roi, movie, '--lock', '1']
            main(argsv=args)

        else:
            logger.info('Found BFDC table --- skipping')

    def batch_apply_drift(self, movie:str):
        zola_dc_table = glob(parent(movie) + sep + self.zola_dc_filename)
        if not zola_dc_table:
            drift_tables = glob(parent(movie) + sep + self.dc_table_filename)
            for drift_table in drift_tables:
                logger.info(f"Found {base(drift_table)}")
                zola_table = glob(parent(drift_table) + sep + self.zola_raw_filename)
                z_lock = glob(parent(drift_table) + sep + self.zola_lock_filename)
                if zola_table and not z_lock:
                    logger.info(f'Found {base(zola_table[0])}')
                    logger.info("Apply drift")

                    args = f"apply {zola_table[0]} {drift_table} \
                            --smooth={str(self.smooth)} --maxbg={str(self.filter_bg)}"
                    main(argsv=args.split())
                elif len(z_lock) == 1:
                    logger.info(f'Found {base(z_lock[0])} --- skipping')
                else:
                    logger.info('No ZOLA table found')
        else:
            logger.info('Folder already processed')

    def do_batch(self):
        logger.info(self.batch_path)
        fov_list = self.parse_fovs(self.batch_path)
        logger.info(f'Found {len(fov_list)} folders starting with FOV')

        for fov in sorted(fov_list):
            logger.info(f'Processing {relative(fov,self.batch_path)}')
            roi = self.find_roi(fov)
            if roi:
                bfdict = self.find_dict(roi[0])[0]
                movies = self.find_movies(fov)
                if movies:
                    for i,movie in enumerate(movies):
                        logger.info(f'Processing movie {i+1}/{len(movies)}: {relative(movie,self.batch_path)}')
                        self.batch_trace_drift(bfdict, roi[0], movie)
                        self.batch_apply_drift(movie)
                else:
                    logger.info(f'No movies found with \"{self.sr_folder_prefix}\" prefix')

            else:
                logger.info('No ROI')
        return 0


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory()
    BatchDrift(path)
