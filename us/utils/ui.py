import os
import urllib.request
from urllib.parse import urlsplit
from tqdm import tqdm
from typing import Optional


def copyfileobj_with_progress(fsrc, fdst, total: Optional[int] = None, length: int = 16*1024) -> None:
    """
    Copy data from file-like object fsrc to file-like object fdst.
    Same as `shutil.copyfileobj` with progress bar: https://hg.python.org/cpython/file/eb09f737120b/Lib/shutil.py#l64
    :param fsrc: file-like object
    :param fdst: file-like object
    :param total:
    :param length
    :return:
    """
    bar_format = '    {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {rate_fmt}'
    with tqdm(total=total, unit='B', unit_scale=True, bar_format=bar_format, ncols=60) as pbar:
        while True:
            buf = fsrc.read(length)
            if not buf:
                break
            fdst.write(buf)
            pbar.update(len(buf))


def download_file(url: str, fdst):
    """
    Download data from `url` to file-like object fdst.
    :param url: str
    :param fdst: file-like object
    :return:
    """
    split = urlsplit(url)
    filename = os.path.basename(split.path)

    print('Downloading {}'.format(filename))

    with urllib.request.urlopen(url) as response:
        length = response.getheader('content-length')
        if length:
            total = int(length)
            copyfileobj_with_progress(response, fdst, total=total)
