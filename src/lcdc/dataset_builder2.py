import re
from typing import List
from tqdm import tqdm
import os
import glob

import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from .lcdataset import LCDataset
from .utils.track import Track
from .utils.rso import RSO
from .utils.functions import load_rsos_from_csv, load_tracks_from_csv


from enum import StrEnum

class TableCols(StrEnum):
    NORAD_ID = 'norad_id'
    TIMESTAMP = 'timestamp'
    PERIOD = 'period'
    DATA = 'data'
    VARIABILITY = 'variability'
    NAME = 'name'
    LABEL = 'label'
    RANGE = 'range'

from abc import ABC, abstractmethod
from functools import partial, reduce

class Preprocessor(ABC):

    @abstractmethod
    def __call__() -> List[Track]:
        pass

class Compose(Preprocessor):

    def __init__(self, *funs: Preprocessor) -> None:
        self.funs = funs

    def __call__(self, track: Track, object: RSO) -> List[Track]:
        tracks = [track]
        for f in self.funs:
            f = partial(f, object=object)
            if (tracks := reduce(list.__add__, map(f, tracks))) == []:
                break

        return tracks


class DatasetBuilder2:
    
    def __init__(self, 
                 directory,
                 classes=None,
                 regexes=None,
                 norad_ids=None,
                 preprocessing=None,
                 statistics=[],
                 split_ratio=None,
                 mean_std=False,
                 lazy=False):
        
        self.dir = directory
        self.preprocessing = preprocessing
        self.statistics = statistics
        self.split_ratio = split_ratio
        self.compute_mean_std = mean_std
        self.lazy = lazy

        self.norad_ids = norad_ids
        assert classes is not None or norad_ids is not None, "Either classes or norad_ids must be provided"

        if norad_ids is not None:
            classes = "Unknown"
            regexes = [".*"]

        self.classes = {c:i for i, c in enumerate(classes)}


        regexes = regexes if regexes is not None else classes
        self.regexes = {c:re.compile(r) for c,r in zip(classes, regexes)}
        self.table = self.load_data()



        print(f"Loaded {len(self.objects)} objects and {len(self.tracks)} tracks")

    def load_data(self):

        parquet_files = glob.glob(f"{self.dir}/*.parquet")
        dataset = pq.ParquetDataset(parquet_files)
        table = dataset.read()


        if self.norad_ids is not None:
            mask = pc.is_in(table[TableCols.NORAD_ID], value_set=self.norad_ids)
            table = table.filter(mask)

        names = list(map(lambda x: x.as_py(), table[TableCols.NAME]))
        labels = list(map(self._get_label, names))
        table = table.append_column(TableCols.LABEL, pa.array(labels))

        if self.classes is not None:
            table = table.filter(pc.field(TableCols.LABEL) != 'Unknown')

        splits = [(0,len(x)) for x in table[TableCols.DATA]]
        table = table.append_column(TableCols.RANGE, pa.array(splits))

        return table
    
    def _get_label(self, rso):
        for c, r in self.regexes.items():
            if r.match(rso.name) is not None:
                return c
        return 'Unknown'
    
    def split_train_test(self, ratio=0.8, seed=None):

        data = []
        for t_id in self.tracks:
            t = self.tracks[t_id]
            if isinstance(t, list):
                t = t[0]
            c = self.labels[t.norad_id]
            data.append([t_id, self.classes[c]])

        data = np.array(data)

        X = data[:,0]
        y = data[:, 1]

        train, test, _, _ = train_test_split(X, y, test_size=1-ratio, stratify=y, random_state=seed)

        train_tracks = {t_id: self.tracks[t_id] for t_id in train}
        test_tracks  = {t_id: self.tracks[t_id] for t_id in test}

        return train_tracks, test_tracks
    
    def build_dataset(self) -> List[LCDataset]:
        def fun(ts: List[Track]):
            t = ts[0]
            if t.data is None:
                t.load_data_from_file(f"{self.dir}/data")
            parts = self.preprocessing(t, self.objects[t.norad_id])
            for p in parts:
                for stat_fun in self.statistics:
                    stat_fun(p)
                    
            if self.lazy:
                for p in parts:
                    if self.lazy: p.unload_data()
            return parts
        

                
        table = self.table
        if self.preprocessing is not None:

            new_table = None
            for i in range(len(table)):
                splits = table[TableCols.SPLITS][i].as_py()
                t = table.splice(i,1).to_pydict()
                data = t[TableCols.DATA].values.to_numpy(zero_copy_only=False)
                for a,b in splits:
                    t2 = t.copy()
                    t2[TableCols.DATA] = np.array(data[a:b])
                    records = self.preprocessing(t2)
                    '''
                    TODO
                    - [ ] Return list of records
                    - [ ] Add records to new table
                    '''
                    if records != []:
                        if new_table is None:
                            new_table = pa.table(records)
                        else:
                            new_table = pa.concat_tables([new_table, pa.table(records)])
                        



            new_tracks = {}
            for ts in tqdm(self.tracks.values(), desc="Preprocessing"):
                if (res := fun(ts)) != []:
                    new_tracks[ts[0].id] = res
                    
            # self.tracks = {ts[0].id: res for ts in self.tracks.values() if (res := fun(ts)) != []}
            self.tracks = new_tracks
            objects_to_be_removed = set(self.objects) - set([self.tracks[i][0].norad_id for i in self.tracks])
            for o_id in objects_to_be_removed:
                del self.objects[o_id]
        
        
        datasets = []
        if self.split_ratio is None:
            datasets = [LCDataset(self.tracks, self.labels, self.dir, self.compute_mean_std,"data")]
        else:
            train_tracks, test_tracks = self.split_train_test(self.split_ratio)
            datasets = [LCDataset(train_tracks, self.labels, 
                              self.dir, self.compute_mean_std, "train"), 
                        LCDataset(test_tracks, self.labels, 
                              self.dir, self.compute_mean_std, "test") ]
        return datasets
    
    def to_file(self, path, data_types=[]):
        datasets = self.build_dataset()

        for d in datasets:
            d.to_file(path, data_types)
    
    def to_dict(self, data_types=[]):
        datasets = self.build_dataset()
        if len(datasets) == 1:
            return datasets[0].to_dict(data_types)
        return [d.to_dict(data_types) for d in datasets]
