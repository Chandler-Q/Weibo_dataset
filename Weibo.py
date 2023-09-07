# -*- coding: utf-8 -*-
"""
Created on Sun May 28 10:37:14 2023

@author: c
"""

import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce


class Weibo(InMemoryDataset):

    def __init__(
        self,
        root: str,
        name: str = 'Weibo',
        feature: str = 'bert',
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.root = root
        self.name = name
        self.feature = feature
        super().__init__(root, transform, pre_transform, pre_filter)

        assert split in ['train', 'val', 'test']
        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')
        
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed', self.feature)

    @property
    def raw_file_names(self) -> List[str]:
        return ['profile','spacy','bert']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        raise ValueError('No URL found')

    def process(self):
        
        data_path = osp.join(self.raw_dir, self.feature)
            
        files = os.listdir(data_path)
            

        for path, split in zip(self.processed_paths, ['train', 'val', 'test']):
            
            data_list = []
            

            idx = np.load(osp.join(self.raw_dir,split+'_index.npy' )).tolist()
            split_files = [files[i] for i in idx]
            
            for i in range(len(split_files)):
                array = np.load(osp.join(data_path,files[i]),allow_pickle=True)
                
                x = torch.from_numpy(array[0]).to(torch.float)
                edge_index = torch.from_numpy(array[1]).to(torch.long)
                y = torch.from_numpy(array[2]).to(torch.long)
                
                data = Data(x = x , edge_index=edge_index, y=y)
                data_list.append(data)
                print(f'\r{i+1}/{len(split_files)}',end='')
                
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
                
            data, slices = self.collate(data_list)
            torch.save((data, slices), path)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, name={self.name}, '
                f'feature={self.feature})')

if __name__ == '__main__':
    Weibo('./data',feature='profile')

    Weibo('./data',feature='spacy')
    Weibo('./data',feature='bert') 

    