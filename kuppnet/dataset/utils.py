import logging
import shutil
import typing as t
from pathlib import PosixPath, Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO, SeqRecord

from kuppnet.dataset.base import Interval

SeqRec = SeqRecord.SeqRecord


def download_text(url: str, **kwargs) -> str:
    """
    :param url: A valid url.
    """
    r = requests.get(url, stream=True, **kwargs)
    if r.ok:
        content = "".join(chunk.decode('utf-8') for chunk in r.iter_content(chunk_size=1024 * 8))
    else:
        raise RuntimeError(f'Download of {url} failed: status code {r.status_code} and output {r.text}')
    return content


def download_file(url, save_path) -> str:
    with requests.get(url, stream=True) as r:
        with open(save_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return save_path


def get_valid_dir(directory: t.Optional[str] = None) -> t.Union[TemporaryDirectory, PosixPath]:
    """
    :param directory: If None, creates a `TemporaryDirectory` instance.
    Otherwise, creates a `Path` instance and ensures it exists.
    """
    if directory is None:
        return TemporaryDirectory()
    if not isinstance(directory, str):
        raise ValueError('Wrong input for directory')
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f'The provided directory {directory.name} does not exist')
    return directory


def dump(
        path: str, resource_name: str,
        data: t.Union[pd.DataFrame, t.Dict, t.List[SeqRec]]) -> None:
    """
    A helper function to dump the resource's data.
    Consult with type annotations to check which data types are supported.
    :param path: A path to dump to.
    :param resource_name: A name of the resource for formatting errors and logging messages.
    :param data: Data to dump.
    """
    if isinstance(data, pd.DataFrame):
        data.to_csv(path, index=False, sep='\t')
    elif isinstance(data, t.Dict):
        pd.DataFrame(
            [list(data.keys()), list(data.values())]
        ).to_csv(path, index=False, sep='\t')
    elif isinstance(data, t.List) and data and isinstance(data[0], SeqRec):
        SeqIO.write(data, path, 'fasta')
    else:
        raise ValueError(f'{resource_name} -- dumping the input of such type is not supported')
    logging.info(f'{resource_name} -- saved parsed data to {path}')


def fetch_seqs(acc: t.Iterable[str]) -> str:
    """
    Fetch UniProt sequences in fasta format
    """
    url = 'https://www.uniprot.org/uploadlists/'
    params = {
        'from': 'ACC+ID',
        'to': 'ACC',
        'format': 'fasta',
        'query': ' '.join(acc)}

    return download_text(url, params=urlencode(params).encode('utf-8'))


def map_ids(ids: t.Iterable[str], db_from: str, db_to: str):
    url = 'https://www.uniprot.org/uploadlists/'
    params = {
        'from': db_from,
        'to': db_to,
        'format': 'tab',
        'query': ' '.join(ids)}
    return download_text(url, params=urlencode(params).encode('utf-8'))


def roll_window(
        array: t.Union[t.Sequence[t.Any], np.ndarray], window_size: int,
        window_step: int, stop_at=0) -> t.Iterator[Interval[np.ndarray]]:
    """
    rolls window over an array of objects (numbers, letters, etc.)
    :param array: array of objects
    :param window_size:
    :param window_step:
    :param stop_at: left-hand position to stop a window at;
    by default = window_size - 1 which means if will produce output of arrays with len = window_size
    if you want to guarantee that all array values will be in an output, use stop_at = window_size // 2
    :return: tuple of intervals array had been split at and list of arrays original array had been split to
    >>> a = np.arange(20)
    >>> roll_window(a, 10, 3)
    ([(0, 10), (3, 13), (6, 16), (9, 19)],
     [array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
      array([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
      array([ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15]),
      array([ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])])
    >>> roll_window(a, 10, 3, 5)
    ([(0, 10), (3, 13), (6, 16), (9, 19), (12, 22)],
     [array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
      array([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
      array([ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15]),
      array([ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
      array([12, 13, 14, 15, 16, 17, 18, 19])])
    """
    if not stop_at:
        stop_at = window_size - 1
    intervals = [(i, window_size + i) for i in range(0, len(array) - stop_at, window_step)]
    if not intervals:
        return iter([Interval(0, window_size, np.array(array, dtype=np.int32))])
    return (Interval(start, stop, array[start:stop]) for start, stop in intervals)


if __name__ == '__main__':
    raise RuntimeError
