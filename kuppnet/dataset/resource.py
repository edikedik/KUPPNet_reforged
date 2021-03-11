import logging
import typing as t
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import filterfalse
from pathlib import PosixPath, Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
from Bio import SeqRecord
from more_itertools import divide
from tqdm import tqdm

from kuppnet.dataset.base import AbstractResource
from kuppnet.dataset.utils import get_valid_dir, dump, download_file, map_ids

SeqRec = SeqRecord.SeqRecord


@dataclass
class Urls:
    dbPTM: str = 'http://dbptm.mbc.nctu.edu.tw/download/experiment/Phosphorylation.txt.gz'
    dbPAF: str = 'http://dbpaf.biocuckoo.org/TOTAL.zip'
    dbPSP: str = 'http://dbpsp.biocuckoo.cn/Download/PhosphorylationData/Total.txt'
    pELM: str = 'http://phospho.elm.eu.org/dumps/phosphoELM_all_latest.dump.tgz'
    BioGRID: str = 'https://downloads.thebiogrid.org/Download/BioGRID/' \
                   'Release-Archive/BIOGRID-4.3.194/BIOGRID-PTMS-4.3.194.ptm.zip'


class Resource(AbstractResource):
    """
    An intermediate class defining common methods
    """

    def __init__(self, resource_name: str, download_dir: t.Optional[str] = None,
                 download_name: str = 'resource', url: str = ''):
        self.resource_name = resource_name
        self.url = url
        self.parsed_data: t.Any = None
        self.download_dir = get_valid_dir(download_dir)
        self.download_name = download_name
        self.resource_path = f'{download_dir}/{download_name}'
        self.preprocessed_data = None
        self.prepared_data = None

    def fetch(self, force: bool = False) -> t.Optional[str]:
        """
        Downloads Resource from the default location.
        :return: Path to a downloaded file (should be the same as {self.download_dir}/{self.download_file_name})
        """
        if isinstance(self.download_dir, str):
            dir_path = self.download_dir
        elif isinstance(self.download_dir, PosixPath):
            dir_path = str(self.download_dir)
        elif isinstance(self.download_dir, TemporaryDirectory):
            dir_path = self.download_dir.name
        else:
            raise ValueError(f'Invalid type {type(self.download_dir)} of `download_dir` attribute')
        resource_path = f'{dir_path}/{self.download_name}'

        if Path(resource_path).exists() and not force:
            return None
        else:
            if force:
                logging.info(f'{self.resource_name} -- overwriting {resource_path}')
            else:
                download_file(self.url, self.resource_path)
                logging.info(f'{self.resource_name} -- downloaded resource from {self.url}')
        return resource_path

    def prepare_data(self):
        raise NotImplementedError

    def dump(self, dump_path: str):
        """
        Dumps the resource to `dump_path`.
        :param dump_path: A valid path.
        :return:
        """
        if self.parsed_data is None:
            raise ValueError(f'{self.resource_name} -- no parsed data to dump '
                             f'(hint: call `parse` method first)')
        dump(path=dump_path, data=self.parsed_data, resource_name=self.resource_name)


class PSP(Resource):
    def __init__(self, download_dir: t.Optional[str] = 'dataset/raw',
                 min_lt: int = 1, min_ms: int = 3, min_cst: int = 3, url=''):
        super().__init__('PhosphoSitePlus', download_dir, 'Phosphorylation_site_dataset', url)
        self.min_lt, self.min_ms, self.min_cst = min_lt, min_ms, min_cst

    def fetch(self, force=False):
        raise ValueError('PhosphoSitePlus requires you to login and fetch '
                         '`Phosphorylation_site_dataset` manually')

    def prepare_data(self):
        df = pd.read_csv(self.resource_path, sep='\t', skiprows=3)
        logging.info(f'{self.resource_name} -- initial records: {len(df)}')
        df = df[(df.LT_LIT >= self.min_lt) | (df.MS_LIT >= self.min_ms) | (df.MS_CST >= self.min_cst)]
        logging.info(f'{self.resource_name} -- filtered records: {len(df)}')
        df = df[['ACC_ID', 'MOD_RSD']]
        if any(x.split('-')[1] != 'p' for x in df['MOD_RSD']):
            raise RuntimeError(f'{self.resource_name} -- some sites are not phosphorylation sites')
        df['MOD_RSD'] = df['MOD_RSD'].apply(lambda x: int(x.split('-')[0][1:]))
        df.rename(columns={'ACC_ID': 'Acc', 'MOD_RSD': 'Pos'}, inplace=True)
        df = df.drop_duplicates()
        logging.info(f'{self.resource_name} -- removed duplicates; records: {len(df)}')
        self.prepared_data = df
        return df


class dbPAF(Resource):
    def __init__(self, download_dir: t.Optional[str] = 'dataset/raw', url=Urls.dbPAF):
        super().__init__('dbPAF', download_dir, 'dbPAF.zip', url)

    def prepare_data(self):
        df = pd.read_csv(self.resource_path, sep=r'\s+', skiprows=1,
                         usecols=[1, 2], names=['Acc', 'Pos'])
        logging.info(f'{self.resource_name} -- initial records: {len(df)}')
        df = df.drop_duplicates()
        logging.info(f'{self.resource_name} -- dropped duplicates; '
                     f'final records: {len(df)}')
        self.prepared_data = df
        return df


class dbPTM(Resource):
    def __init__(self, download_dir: t.Optional[str] = 'dataset/raw', url=Urls.dbPTM):
        super().__init__('dbPTM', download_dir, 'dbPTM.txt.gz', url)

    def prepare_data(self):
        raise NotImplementedError


class pELM(Resource):
    def __init__(self, download_dir: t.Optional[str] = 'dataset/raw', url=Urls.pELM):
        super().__init__('pELM', download_dir, 'p.ELM.tar.gz', url)

    def prepare_data(self):
        self.prepared_data = pd.read_csv(
            self.resource_path, sep=r'\s+',
            usecols=[0, 2], skiprows=1, names=['Acc', 'Pos'],
            encoding='utf-8')
        logging.info(f'{self.resource_name} -- finalized resource; '
                     f'total records: {len(self.prepared_data)}')
        return self.prepared_data


class BioGrid(Resource):
    def __init__(self, download_dir: t.Optional[str] = 'dataset/raw', url=Urls.BioGRID):
        super().__init__('BioGRID', download_dir, 'BioGRID.zip', url)

    def prepare_data(self):
        with ZipFile(self.resource_path) as archive:
            path = archive.extract(archive.filelist[0], self.download_dir)
            logging.info(f'{self.resource_name} -- extracted raw dataset to {path}')
        df = pd.read_csv(path, sep='\t', low_memory=False, usecols=[
            'BioGRID ID', 'Post Translational Modification', 'Position']).drop_duplicates()
        logging.info(f'{self.resource_name} -- loaded dataset; total records: {len(df)}')
        df = df[df['Post Translational Modification'] == 'Phosphorylation']
        logging.info(f'{self.resource_name} -- filtered phosphorylation sites; '
                     f'total records: {len(df)}')

        # Map BioGRID IDs to UniProt Acc
        bio_grid_ids = set(df['BioGRID ID'].astype(str))
        num_chunks = len(bio_grid_ids) // 500
        chunks = divide(num_chunks, bio_grid_ids)
        with ThreadPoolExecutor() as executor:
            results = "".join(
                map(lambda future: future.result(),
                    tqdm(
                        as_completed([executor.submit(
                            map_ids, chunk, 'BIOGRID_ID', 'ACC') for chunk in chunks]),
                        desc='Mapping chunks of IDs',
                        total=num_chunks)))
        mapping = dict(
            map(lambda x: (int(x[0]), x[1]),
                filterfalse(
                    lambda x: x[0] == 'From', filter(
                        lambda x: len(x) == 2, map(
                            lambda x: x.rstrip().split('\t'), results.split('\n'))))))

        logging.info(f'{self.resource_name} -- mapped {len(mapping)} out of {len(bio_grid_ids)} ids')
        df['Acc'] = [mapping[x] if x in mapping else np.nan for x in df['BioGRID ID']]

        df = df.drop_duplicates().dropna().rename(columns={'Position': 'Pos'})[['Acc', 'Pos']].sort_values('Acc')
        self.prepared_data = df
        logging.info(f'{self.resource_name} -- finalized resource; total records: {len(df)}')
        return df


if __name__ == '__main__':
    raise RuntimeError
