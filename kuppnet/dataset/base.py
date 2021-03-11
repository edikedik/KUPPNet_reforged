import typing as t
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from Bio import SeqRecord

SeqRec = SeqRecord.SeqRecord

T = t.TypeVar('T')

AminoAcids = {
    'A': (1, 'ALA', 'alanine'),
    'R': (2, 'ARG', 'arginine'),
    'N': (3, 'ASN', 'asparagine'),
    'D': (4, 'ASP', 'aspartic acid'),
    'C': (5, 'CYS', 'cysteine'),
    'Q': (6, 'GLN', 'glutamine'),
    'E': (7, 'GLU', 'glutamic acid'),
    'G': (8, 'GLY', 'glycine'),
    'H': (9, 'HIS', 'histidine'),
    'I': (10, 'ILE', 'isoleucine'),
    'L': (11, 'LEU', 'leucine'),
    'K': (12, 'LYS', 'lysine'),
    'M': (13, 'MET', 'methionine'),
    'F': (14, 'PHE', 'phenylalanine'),
    'P': (15, 'PRO', 'proline'),
    'S': (16, 'SER', 'serine'),
    'T': (17, 'THR', 'threonine'),
    'W': (18, 'TRP', 'tryptophan'),
    'Y': (19, 'TYR', 'tyrosine'),
    'V': (20, 'VAL', 'valine')}


class RequiresAttributeError(Exception):
    def __init__(self, attribute):
        super().__init__(f'Requires non-empty attribute `{attribute}`')


class AbstractResource(metaclass=ABCMeta):
    """
    Abstract base class defining a basic interface for a Resource.
    """

    @abstractmethod
    def fetch(self):
        raise NotImplementedError

    @abstractmethod
    def prepare_data(self):
        raise NotImplementedError

    @abstractmethod
    def dump(self, dump_path: str):
        raise NotImplementedError


class AbstractDataset(metaclass=ABCMeta):
    """
    Abstract base class defining a basic interface of a Dataset.
    """

    @abstractmethod
    def combine_resources(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def fetch_sequences(self, download_dir: str):
        pass

    @abstractmethod
    def roll_window(self, size: int, step: int):
        pass

    @abstractmethod
    def cluster(self, ident: float, out_path: str):
        pass

    @abstractmethod
    def finalize(self):
        pass


class Interval(t.Container, t.Generic[T]):
    __slots__ = ('start', 'stop', 'data')

    def __init__(self, start: int, stop: int, data: t.Optional[T] = None):
        self.start = start
        self.stop = stop
        self.data = data

    def __contains__(self, item: T) -> bool:
        return False if self.data is None or item is None else self.data == item

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __eq__(self, other: 'Interval'):
        return (self.start, self.stop, self.data) == (other.start, other.stop, other.data)

    def __hash__(self):
        return hash((self.start, self.stop, self.data))

    def __len__(self):
        return self.stop - self.start

    def __bool__(self):
        return bool(len(self))

    def __and__(self, other: 'Interval'):
        # TODO docs
        first, second = sorted([self, other], key=lambda iv: iv.start)
        return type(self)(first.start, second.stop, [first.data, second.data])

    def __repr__(self):
        return f'{type(self).__name__}(start={self.start}, stop={self.stop}, data={self.data})'

    def reload(self, value: T):
        return type(self)(self.start, self.stop, value)


class NamedInterval(t.Container, t.Generic[T]):
    __slots__ = ('name', 'start', 'stop', 'data')

    def __init__(self, name: str, start: int, stop: int, data: t.Optional[T] = None):
        self.name = name
        self.start = start
        self.stop = stop
        self.data = data

    def __repr__(self):
        return f'{type(self).__name__}(start={self.start}, stop={self.stop}, ' \
               f'data={self.data}), name={self.name}'

    def __contains__(self, item: T) -> bool:
        return False if self.data is None or item is None else self.data == item

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __eq__(self, other: 'NamedInterval'):
        return (self.start, self.stop, self.data, self.name) == (
            other.start, other.stop, other.data, other.name)

    def __hash__(self):
        return hash((self.start, self.stop, self.data, self.name))

    def __len__(self):
        return self.stop - self.start

    def __bool__(self):
        return bool(len(self))


class SeqEncoder:
    def __init__(self, mapping: t.Optional[t.Mapping[str, int]] = None):
        self.mapping = mapping
        if mapping is None:
            self.mapping = {k: v[0] for k, v in AminoAcids.items()}
        self.default_missing = max(self.mapping.values())

    def __call__(self, seq: str) -> np.ndarray:
        return np.array([self.mapping.get(c, self.default_missing) for c in seq])


if __name__ == '__main__':
    raise RuntimeError
