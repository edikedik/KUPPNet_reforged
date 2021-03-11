import logging
import subprocess as sp
import typing as t
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain, starmap
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO, SeqRecord
from more_itertools import divide, split_at
from tqdm import tqdm

from kuppnet.dataset.base import AbstractDataset, RequiresAttributeError, Interval, NamedInterval, SeqEncoder
from kuppnet.dataset.resource import Resource
from kuppnet.dataset.utils import fetch_seqs, roll_window

SeqRec = SeqRecord.SeqRecord


class Dataset(AbstractDataset):
    def __init__(self, resources: t.List[Resource]):
        self._resources = resources
        self.combined_resources: t.Optional[pd.DataFrame] = None
        self.seqs: t.Optional[t.Tuple[SeqRec]] = None
        self.rolled: t.Optional[t.List[t.List[NamedInterval]]] = None
        self.clusters: t.Optional[t.List[t.List[NamedInterval]]] = None
        self.dataset: t.Optional[pd.DataFrame] = None

    @property
    def resources(self) -> t.List[Resource]:
        return self._resources

    def combine_resources(self) -> pd.DataFrame:
        def validate(resource: Resource) -> Resource:
            if resource.prepared_data is None:
                raise RuntimeError(f'Resource {resource.resource_name} has no `prepared_data`')
            for col in ['Acc', 'Pos']:
                if col not in resource.prepared_data.columns:
                    raise RuntimeError(f'Resource {resource.resource_name} has no columns {col}')
            return resource

        def wrap(resource: Resource):
            df_ = resource.prepared_data.copy()
            df_['Resource'] = resource.resource_name
            return df_

        df = pd.concat([wrap(validate(r)) for r in self.resources])
        logging.info(f'Combined resources; total records: {len(df)}')
        df = df.groupby(['Acc', 'Pos'], as_index=False).agg(lambda x: ';'.join(x))
        logging.info(f"Aggregated observations' sources; total records: {len(df)}")
        self.combined_resources = df
        return df

    def fetch_sequences(self, download_dir: str = 'dataset/raw',
                        download_name: str = 'UniProt_seqs.fasta',
                        n_threads: t.Optional[int] = None) -> t.Tuple[SeqRec]:
        if self.combined_resources is None:
            raise RequiresAttributeError('combined_resources')
        acc = set(self.combined_resources['Acc'])
        logging.info(f'Total sequences to fetch: {len(acc)}')
        path = f'{download_dir}/{download_name}'
        if Path(path).exists() and Path(path).is_file():
            acc_exist = acc & {s.id.split('|')[1] for s in SeqIO.parse(path, 'fasta')}
            acc -= acc_exist
            logging.info(f'Found {len(acc_exist)} sequences in file {path}. '
                         f'Remaining sequences to fetch: {len(acc)}')
        if not acc:
            logging.info(f'No new sequences to fetch: returning existing ones.')
            self.seqs = tuple(SeqIO.parse(path, 'fasta'))
            return self.seqs
        num_chunks = len(acc) // 500
        chunks = divide(num_chunks, acc)
        logging.info(f'Split downloading list into {num_chunks} chunks')
        path = f'{download_dir}/{download_name}'
        with open(path, 'a+') as f:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                for future in tqdm(
                        as_completed([executor.submit(fetch_seqs, chunk) for chunk in chunks]),
                        desc='Fetching chunks of sequences',
                        total=num_chunks):
                    print(future.result(), file=f)

        self.seqs = tuple(SeqIO.parse(path, 'fasta'))
        logging.info(f'Fetched {len(self.seqs)} sequences to {path}')
        return self.seqs

    def roll_window(self, size: int, step: int) -> t.List[t.List[NamedInterval]]:
        def process_seq(seq: SeqRec):
            intervals: t.Iterator[Interval] = roll_window(seq.seq, size, step, size // step)
            valid_intervals = filter(
                lambda x: any(aa in x.data for aa in ['S', 'Y', 'T']), intervals)
            valid_intervals = (
                NamedInterval(seq.id.split('|')[1], x.start, x.stop, x.data) for x in valid_intervals)
            return list(valid_intervals)

        if self.seqs is None:
            raise RequiresAttributeError('seqs')
        self.rolled = list(map(process_seq, self.seqs))
        logging.info(f'Rolled window with size {size} and step {step} over sequences')
        return self.rolled

    def dump_intervals(self, path='dataset/raw/intervals.fasta') -> str:
        if self.seqs is None:
            raise RequiresAttributeError('seqs')
        if self.rolled is None:
            raise RequiresAttributeError('rolled')
        if len(self.seqs) != len(self.rolled):
            raise ValueError(f'Number of sequences {len(self.seqs)} does not match '
                             f'a number of interval groups {len(self.rolled)}')
        with open(path, 'w') as f:
            for interval in chain.from_iterable(self.rolled):
                rec_id = f'{interval.name}|{interval.start}-{interval.stop}'
                int_rec = SeqRec(id=rec_id, name=rec_id, description=rec_id, seq=interval.data)
                SeqIO.write([int_rec], f, 'fasta')
        logging.info(f'Saved sequence intervals to {path}')
        return path

    def cluster(self, ident: float = 0.95, out_path: str = 'dataset/raw/intervals.clusters.txt',
                n_threads: int = 0, max_mem_mb: int = 4000, min_seq_len: int = 50,
                input_path: t.Optional[str] = None) -> t.List[t.List[NamedInterval]]:

        def get_interval(cluster_rec: str, mapping: t.Mapping[t.Tuple[str, int, int], NamedInterval]) \
                -> NamedInterval:
            rec = cluster_rec.split()[2].rstrip('...').lstrip('>')
            name = rec.split('|')[0]
            start, stop = map(int, rec.split('|')[1].split('-'))
            return mapping[(name, start, stop)]

        if self.rolled is None:
            raise RequiresAttributeError('rolled')

        if input_path is None:
            input_path = self.dump_intervals()
        cmd = f'cd-hit -i {input_path} -o {out_path} -c {ident} ' \
              f'-T {n_threads} -M {max_mem_mb} -l {min_seq_len} -d 0'
        logging.info(f'Will run clustering command {cmd}')
        try:
            sp.run(cmd, shell=True, check=True)
        except sp.CalledProcessError as e:
            res = sp.run(cmd, shell=True, check=False, capture_output=True, text=True)
            raise RuntimeError(f'Command {cmd} failed with an error {e}, '
                               f'stdout {res.stdout} and stderr {res.stderr}')
        mapping_ = {(x.name, x.start, x.stop): x for x in chain.from_iterable(self.rolled)}
        with open(f'{out_path}.clstr') as f:
            clusters = list(map(
                lambda c: [get_interval(r, mapping_) for r in c],
                filter(bool, split_at(f, lambda x: x.startswith('>')))))
        logging.info(f'Clustered intervals into {len(clusters)} clusters')
        self.clusters = clusters
        return clusters

    def finalize(self, bar: bool = True, num_proc: t.Optional[int] = 4):
        if self.combined_resources is None:
            raise RequiresAttributeError('combined_resources')
        if self.clusters is None:
            raise RequiresAttributeError('clusters')
        Row = namedtuple('Row', ['Cluster_i', 'Acc', 'Start', 'Stop', 'Seq'])

        def wrap_cluster(cluster_i: int, cluster: t.List[NamedInterval]):
            def wrap_interval(interval: NamedInterval):
                return Row(cluster_i, interval.name, interval.start,
                           interval.start + len(interval.data),
                           str(interval.data))

            return map(wrap_interval, cluster)

        cl = pd.DataFrame(tqdm(
            chain.from_iterable(starmap(wrap_cluster, enumerate(self.clusters))),
            desc='Wrapping intervals into df',
            total=sum(1 for _ in chain.from_iterable(self.rolled))))

        df = pd.merge(self.combined_resources[['Acc', 'Pos']], cl[['Acc', 'Start', 'Stop']], on='Acc')
        df['Pos'] = df['Pos'].astype(int)
        df['WithinInterval'] = (df['Pos'] - 1 > df['Start']) & (df['Pos'] - 1 < df['Stop'])

        # TODO: this is still quite slow...
        logging.info(f'Distributing positive classes into intervals (might take a while)')
        df = df.groupby(
            ['Acc', 'Start', 'Stop'], as_index=False
        ).apply(
            lambda x: ';'.join(x.Pos[x.WithinInterval].astype(str).sort_values().unique())
        ).rename(
            columns={None: 'Pos'}
        )
        df['NumPos'] = df['Pos'].apply(lambda x: 0 if x == '' else len(x.split(';')))
        df = pd.merge(df, cl, on=['Acc', 'Start', 'Stop'])
        df = pd.merge(df, self.combined_resources[['Acc', 'Resource']], on='Acc')
        df = df.sort_values('NumPos', ascending=False).groupby('Cluster_i').head(1)
        logging.info(f'Selected representatives; total records: {len(df)}')

        self.dataset = df
        return df

    def encode(self):

        def encode_row(row, encode=SeqEncoder()):
            seq = encode(row.Seq)
            pos = np.array(
                [int(x) - row.Start - 1 for x in row.Pos.split(';') if x],
                dtype=np.int32)
            phos = np.zeros(len(seq), dtype=np.int8)
            if pos.any():
                try:
                    phos[pos] = 1
                except IndexError as e:
                    raise ValueError(f'Failed on the row {row} with an error {e}')
            mask = np.in1d(seq, [16, 17, 19]).astype(np.int8)
            if (phos & ~ mask).any():
                logging.warning(f'Row {row} contains positive classes at the wrong place')
            return seq, phos, mask

        if self.dataset is None:
            raise RequiresAttributeError('dataset')
        for col in ['Seq', 'Start', 'Pos']:
            if col not in self.dataset.columns:
                raise ValueError(f'Missing column {col}')

        return list(
            map(encode_row, tqdm(
                self.dataset.itertuples(),
                desc='Encoding dataset',
                total=len(self.dataset))))


if __name__ == '__main__':
    raise RuntimeError
