"""
    picasso.io
    ~~~~~~~~~~

    General purpose library for handling input and output of files

    :author: Joerg Schnitzbauer, 2015
    :copyright: Copyright (c) 2015 Jungmann Lab, Max Planck Institute of Biochemistry
"""
import os.path as _ospath
import numpy as _np
#import yaml as _yaml
import re as _re
import struct as _struct
import json as _json
import os as _os
import threading as _threading


class NoMetadataFileError(FileNotFoundError):
    pass


def multiple_filenames(path: str, index: object) -> object:
    base, ext = _ospath.splitext(path)
    filename = base + '_' + str(index) + ext
    return filename


def load_tif(path):
    movie = TiffMultiMap(path, memmap_frames=False)
    info = movie.info()
    return movie, [info]


def load_movie(path, prompt_info=None):
    base, ext = _ospath.splitext(path)
    ext = ext.lower()
    if ext == '.tif':
        return load_tif(path)


def load_info(path, qt_parent=None):
    path_base, path_extension = _ospath.splitext(path)
    filename = path_base + '.yaml'
    try:
        with open(filename, 'r') as info_file:
            info = list(_yaml.load_all(info_file))
    except FileNotFoundError as e:
        print('\nAn error occured. Could not find metadata file:\n{}'.format(filename))
        # if qt_parent is not None:
        # _QMessageBox.critical(qt_parent, 'An error occured', 'Could not find metadata file:\n{}'.format(filename))
        raise NoMetadataFileError(e)
    return info


class TiffMap:
    TIFF_TYPES = {1: 'B', 2: 'c', 3: 'H', 4: 'L', 5: 'RATIONAL'}
    TYPE_SIZES = {'c': 1, 'B': 1, 'h': 2, 'H': 2, 'i': 4, 'I': 4, 'L': 4, 'RATIONAL': 8}

    def __init__(self, path, verbose=False):
        if verbose:
            print('Reading info from {}'.format(path))
        self.path = _ospath.abspath(path)
        self.file = open(self.path, 'rb')
        self._tif_byte_order = {b'II': '<', b'MM': '>'}[self.file.read(2)]
        self.file.seek(4)
        self.first_ifd_offset = self.read('L')

        # Read info from first IFD
        self.file.seek(self.first_ifd_offset)
        n_entries = self.read('H')
        for i in range(n_entries):
            self.file.seek(self.first_ifd_offset + 2 + i * 12)
            tag = self.read('H')
            type = self.TIFF_TYPES[self.read('H')]
            count = self.read('L')
            if tag == 256:
                self.width = self.read(type, count)
            elif tag == 257:
                self.height = self.read(type, count)
            elif tag == 258:
                bits_per_sample = self.read(type, count)
                dtype_str = 'u' + str(int(bits_per_sample / 8))
                # Picasso uses internatlly exclusively little endian byte order...
                self.dtype = _np.dtype(dtype_str)
                # ... the tif byte order might be different, so we also store the file dtype
                self._tif_dtype = _np.dtype(self._tif_byte_order + dtype_str)
        self.frame_shape = (self.height, self.width)
        self.frame_size = self.height * self.width

        # Collect image offsets
        self.image_offsets = []
        offset = self.first_ifd_offset
        while offset != 0:
            self.file.seek(offset)
            n_entries = self.read('H')
            if n_entries is None:
                # Some MM files have trailing nonsense bytes
                break
            for i in range(n_entries):
                self.file.seek(offset + 2 + i * 12)
                tag = self.read('H')
                if tag == 273:
                    type = self.TIFF_TYPES[self.read('H')]
                    count = self.read('L')
                    self.image_offsets.append(self.read(type, count))
                    break
            self.file.seek(offset + 2 + n_entries * 12)
            offset = self.read('L')
        self.n_frames = len(self.image_offsets)

        self.lock = _threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):
        with self.lock:  # Otherwise we get messed up when reading frames from multiple threads
            if isinstance(it, tuple):
                if isinstance(it, int) or _np.issubdtype(it[0], _np.integer):
                    return self[it[0]][it[1:]]
                elif isinstance(it[0], slice):
                    indices = range(*it[0].indices(self.n_frames))
                    stack = _np.array([self.get_frame(_) for _ in indices])
                    if len(indices) == 0:
                        return stack
                    else:
                        if len(it) == 2:
                            return stack[:, it[1]]
                        elif len(it) == 3:
                            return stack[:, it[1], it[2]]
                        else:
                            raise IndexError
                elif it[0] == Ellipsis:
                    stack = self[it[0]]
                    if len(it) == 2:
                        return stack[:, it[1]]
                    elif len(it) == 3:
                        return stack[:, it[1], it[2]]
                    else:
                        raise IndexError
            elif isinstance(it, slice):
                indices = range(*it.indices(self.n_frames))
                return _np.array([self.get_frame(_) for _ in indices])
            elif it == Ellipsis:
                return _np.array([self.get_frame(_) for _ in range(self.n_frames)])
            elif isinstance(it, int) or _np.issubdtype(it, _np.integer):
                return self.get_frame(it)
            raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self):
        return self.n_frames

    def info(self):
        info = {'Byte Order': self._tif_byte_order, 'File': self.path, 'Height': self.height,
                'Width': self.width, 'Data Type': self.dtype.name, 'Frames': self.n_frames}
        # The following block is MM-specific
        self.file.seek(self.first_ifd_offset)
        n_entries = self.read('H')
        for i in range(n_entries):
            self.file.seek(self.first_ifd_offset + 2 + i * 12)
            tag = self.read('H')
            type = self.TIFF_TYPES[self.read('H')]
            count = self.read('L')
            if count * self.TYPE_SIZES[type] > 4:
                self.file.seek(self.read('L'))
            if tag == 51123:
                # This is the Micro-Manager tag. We generate an info dict that contains any info we need.
                readout = self.read(type, count).strip(b'\0')  # Strip null bytes which MM 1.4.22 adds
                mm_info = _json.loads(readout.decode())
                info['Micro-Manager Metadata'] = mm_info
                info['Camera'] = mm_info['Camera']
        return info

    def get_frame(self, index, array=None):
        self.file.seek(self.image_offsets[index])
        frame = _np.reshape(_np.fromfile(self.file, dtype=self._tif_dtype, count=self.frame_size), self.frame_shape)
        # We only want to deal with little endian byte order downstream:
        if self._tif_byte_order == '>':
            frame.byteswap(True)
            frame = frame.newbyteorder('<')
        return frame

    def read(self, type, count=1):
        if type == 'c':
            return self.file.read(count)
        elif type == 'RATIONAL':
            return self.read_numbers('L') / self.read_numbers('L')
        else:
            return self.read_numbers(type, count)

    def read_numbers(self, type, count=1):
        size = self.TYPE_SIZES[type]
        fmt = self._tif_byte_order + count * type
        try:
            return _struct.unpack(fmt, self.file.read(count * size))[0]
        except _struct.error:
            return None

    def close(self):
        self.file.close()

    def tofile(self, file_handle, byte_order=None):
        do_byteswap = (byte_order != self.byte_order)
        for image in self:
            if do_byteswap:
                image = image.byteswap()
            image.tofile(file_handle)


class TiffMultiMap:

    def __init__(self, path, memmap_frames=False, verbose=False):
        self.path = _ospath.abspath(path)
        self.dir = _ospath.dirname(self.path)
        base, ext = _ospath.splitext(_ospath.splitext(self.path)[0])  # split two extensions as in .ome.tif
        base = _re.escape(base)
        pattern = _re.compile(base + '_(\d*).ome.tif')  # This matches the basename + an appendix of the file number
        entries = [_.path for _ in _os.scandir(self.dir) if _.is_file()]
        matches = [_re.match(pattern, _) for _ in entries]
        matches = [_ for _ in matches if _ is not None]
        paths_indices = [(int(_.group(1)), _.group(0)) for _ in matches]
        self.paths = [self.path] + [path for index, path in sorted(paths_indices)]
        self.maps = [TiffMap(path, verbose=verbose) for path in self.paths]
        self.n_maps = len(self.maps)
        self.n_frames_per_map = [_.n_frames for _ in self.maps]
        self.n_frames = sum(self.n_frames_per_map)
        self.cum_n_frames = _np.insert(_np.cumsum(self.n_frames_per_map), 0, 0)
        self.dtype = self.maps[0].dtype
        self.height = self.maps[0].height
        self.width = self.maps[0].width
        self.shape = (self.n_frames, self.height, self.width)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):
        if isinstance(it, tuple):
            if it[0] == Ellipsis:
                stack = self[it[0]]
                if len(it) == 2:
                    return stack[:, it[1]]
                elif len(it) == 3:
                    return stack[:, it[1], it[2]]
                else:
                    raise IndexError
            elif isinstance(it[0], slice):
                indices = range(*it[0].indices(self.n_frames))
                stack = _np.array([self.get_frame(_) for _ in indices])
                if len(indices) == 0:
                    return stack
                else:
                    if len(it) == 2:
                        return stack[:, it[1]]
                    elif len(it) == 3:
                        return stack[:, it[1], it[2]]
                    else:
                        raise IndexError
            if isinstance(it[0], int) or _np.issubdtype(it[0], _np.integer):
                return self[it[0]][it[1:]]
        elif isinstance(it, slice):
            indices = range(*it.indices(self.n_frames))
            return _np.array([self.get_frame(_) for _ in indices])
        elif it == Ellipsis:
            return _np.array([self.get_frame(_) for _ in range(self.n_frames)])
        elif isinstance(it, int) or _np.issubdtype(it, _np.integer):
            return self.get_frame(it)
        raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self):
        return self.n_frames

    def close(self):
        for map in self.maps:
            map.close()

    def get_frame(self, index):
        # TODO deal with negative numbers
        for i in range(self.n_maps):
            if self.cum_n_frames[i] <= index < self.cum_n_frames[i + 1]:
                break
        else:
            raise IndexError
        return self.maps[i][index - self.cum_n_frames[i]]

    def info(self):
        info = self.maps[0].info()
        info['Frames'] = self.n_frames
        return info

    def to_file(self, file_handle, byte_order=None):
        for map in self.maps:
            map.tofile(file_handle, byte_order)


def get_movie_groups(paths):
    groups = {}
    if len(paths) > 0:
        pattern = _re.compile(
            '(.*?)(_(\d*))?.ome.tif')  # This matches the basename + an optional appendix of the file number
        matches = [_re.match(pattern, path) for path in paths]
        match_infos = [{'path': _.group(), 'base': _.group(1), 'index': _.group(3)} for _ in matches]
        for match_info in match_infos:
            if match_info['index'] is None:
                match_info['index'] = 0
            else:
                match_info['index'] = int(match_info['index'])
        basenames = set([_['base'] for _ in match_infos])
        for basename in basenames:
            match_infos_group = [_ for _ in match_infos if _['base'] == basename]
            group = [_['path'] for _ in match_infos_group]
            indices = [_['index'] for _ in match_infos_group]
            group = [path for (index, path) in sorted(zip(indices, group))]
            groups[basename] = group
    return groups
