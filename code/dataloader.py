import numpy as np
import argparse
import os
from tqdm import tqdm
import time
import pickle as pkl
import torch

class Dataloader(object):
    def __init__(self, dataset_tuple, batch_size, shuffle) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_tuple = dataset_tuple
        self.dataset_tuple = [torch.from_numpy(t) for t in self.dataset_tuple]
        
        self.dataset_size = len(self.dataset_tuple[0])
        if self.dataset_size % self.batch_size == 0:
            self.total_step = int(self.dataset_size / self.batch_size)
        else:
            self.total_step = int(self.dataset_size / self.batch_size) + 1
        self.step = 0
    
    def __len__(self):
        return self.total_step

    def _shuffle_data(self):
        print('shuffling...')
        perm = np.random.permutation(self.dataset_size)
        for i in range(len(self.dataset_tuple)):
            self.dataset_tuple[i] = self.dataset_tuple[i][perm]

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.total_step:
            raise StopIteration

        left = self.batch_size * self.step
        if self.step == self.total_step - 1:
            right = self.dataset_size
        else:
            right = self.batch_size * (self.step + 1)
        
        self.step += 1
        batch_data = []
        for i in range(len(self.dataset_tuple)):
            batch_data.append(self.dataset_tuple[i][left:right])
        return batch_data

    def refresh(self):
        print('refreshing...')
        self.step = 0
        if self.shuffle:
            self._shuffle_data()
        print('refreshed')

class ChunkDataloader(object):
    def __init__(self, dataset_size, chunk_size, batch_size, data_prefix,
                start_chunk, end_chunk) -> None:
        self.batch_size = batch_size
        self.chunk_step = start_chunk #0
        self.start_chunk = start_chunk
        self.end_chunk = end_chunk

        self.data_prefix = data_prefix
        with open('{}_{}.pkl'.format(data_prefix, start_chunk), 'rb') as f:
            self.chunk = pkl.load(f)

        self.chunk = torch.from_numpy(self.chunk)

        self.dataset_size = dataset_size  # 739364600,13000000
        self.chunk_size = chunk_size  # 65000000,13000000
        self.chunk_num = int(self.dataset_size / self.chunk_size) + 1 if (dataset_size % chunk_size) else int(
            self.dataset_size / self.chunk_size)

        if self.dataset_size >= self.chunk_size:
            if self.chunk_size % self.batch_size == 0:
                self.chunk_total_step = int(self.chunk_size / self.batch_size)
            else:
                self.chunk_total_step = int(self.chunk_size / self.batch_size) + 1
        else:
            self.chunk_total_step = int(self.dataset_size/ self.batch_size) + 1 if self.dataset_size % self.batch_size else int(self.dataset_size/self.batch_size)

        if (self.dataset_size % self.chunk_size) % self.batch_size == 0:
            self.total_step = self.chunk_total_step * int(self.dataset_size / self.chunk_size) + int(
                (self.dataset_size % self.chunk_size) / self.batch_size)
        else:
            self.total_step = self.chunk_total_step * int(self.dataset_size / self.chunk_size) + int(
                (self.dataset_size % self.chunk_size) / self.batch_size) + 1

        self.step = 0

    def __len__(self):
        return self.total_step

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.chunk_total_step:
            if self.chunk_step == self.end_chunk: #self.chunk_num - 1:
                raise StopIteration
            else:
                self.chunk_step += 1
                self.step = 0
                with open('{}_{}.pkl'.format(self.data_prefix, self.chunk_step), 'rb') as f:
                    self.chunk = pkl.load(f)
                self.chunk = torch.from_numpy(self.chunk)
                self.chunk_size = len(self.chunk)
                self.chunk_total_step = int(self.chunk_size / self.batch_size) + 1 if (
                            self.chunk_size % self.batch_size) else int(self.chunk_size / self.batch_size)

        left = self.batch_size * self.step
        if self.step == self.chunk_total_step - 1:
            right = self.chunk_size
        else:
            right = self.batch_size * (self.step + 1)

        self.step += 1
        return self.chunk[left:right]

    def refresh(self):
        print('refreshing...')
        self.step = 0

        with open('{}_{}.pkl'.format(self.data_prefix, self.start_chunk), 'rb') as f:
            self.chunk = pkl.load(f)

        self.chunk = torch.from_numpy(self.chunk)

        self.chunk_size = len(self.chunk)
        self.chunk_total_step = int(self.chunk_size / self.batch_size) + 1 if (
                    self.chunk_size % self.batch_size) else int(self.chunk_size / self.batch_size)
        self.chunk_step = self.start_chunk

    
        print('refreshed')


class ChunkDataloader_train(object):
    def __init__(self, dataset_size, chunk_size, batch_size, data_prefix) -> None:
        self.batch_size = batch_size
        self.chunk_step = 0
        self.data_prefix = data_prefix
        with open('{}_0.pkl'.format(data_prefix), 'rb') as f:
            self.chunk = pkl.load(f)

        self.chunk = [torch.from_numpy(t) for t in self.chunk]

        self.dataset_size = dataset_size  # 739364600,13000000
        self.chunk_size = chunk_size  # 65000000,13000000

        self._shuffle_data()
        self.chunk_num = int(self.dataset_size / self.chunk_size) + 1 if (dataset_size % chunk_size) else int(
            self.dataset_size / self.chunk_size)

        if self.dataset_size >= self.chunk_size:
            if self.chunk_size % self.batch_size == 0:
                self.chunk_total_step = int(self.chunk_size / self.batch_size)
            else:
                self.chunk_total_step = int(self.chunk_size / self.batch_size) + 1
        else:
            self.chunk_total_step = int(self.dataset_size/ self.batch_size) + 1 if self.dataset_size % self.batch_size else int(self.dataset_size/self.batch_size)

        if (self.dataset_size % self.chunk_size) % self.batch_size == 0:
            self.total_step = self.chunk_total_step * int(self.dataset_size / self.chunk_size) + int(
                (self.dataset_size % self.chunk_size) / self.batch_size)
        else:
            self.total_step = self.chunk_total_step * int(self.dataset_size / self.chunk_size) + int(
                (self.dataset_size % self.chunk_size) / self.batch_size) + 1

        self.step = 0

    
    def _shuffle_data(self):
        print('shuffling...')
        perm = np.random.permutation(self.chunk_size)
        for i in range(len(self.chunk)):
            self.chunk[i] = self.chunk[i][perm]


    def __len__(self):
        return self.total_step

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.chunk_total_step:
            if self.chunk_step == self.chunk_num - 1:
                raise StopIteration
            else:
                self.chunk_step += 1
                self.step = 0
                with open('{}_{}.pkl'.format(self.data_prefix, self.chunk_step), 'rb') as f:
                    self.chunk = pkl.load(f)
                
                self.chunk = [torch.from_numpy(t) for t in self.chunk]
                self.chunk_size = len(self.chunk[0])
                self._shuffle_data()
                self.chunk_total_step = int(self.chunk_size / self.batch_size) + 1 if (
                            self.chunk_size % self.batch_size) else int(self.chunk_size / self.batch_size)

        left = self.batch_size * self.step
        if self.step == self.chunk_total_step - 1:
            right = self.chunk_size
        else:
            right = self.batch_size * (self.step + 1)

        self.step += 1
        batch_data = []
        for i in range(len(self.chunk)):
            batch_data.append(self.chunk[i][left:right])
        return batch_data

    def refresh(self):
        print('refreshing...')
        self.step = 0

        with open('{}_0.pkl'.format(self.data_prefix), 'rb') as f:
            self.chunk = pkl.load(f)

        self.chunk = [torch.from_numpy(t) for t in self.chunk]
        
        self.chunk_size = len(self.chunk[0])

        self._shuffle_data()
        self.chunk_total_step = int(self.chunk_size / self.batch_size) + 1 if (
                    self.chunk_size % self.batch_size) else int(self.chunk_size / self.batch_size)
        self.chunk_step = 0

    
        print('refreshed')

if __name__ == "__main__":
    # with open('../data/zj/input_data/train_set.pkl', 'rb') as f:
    #     dataset_tuple = pkl.load(f)
    # dl = Dataloader(dataset_tuple, 1024, True)
    # for batch in tqdm(dl):
    #     train_x, train_y = batch
    #     print(train_x.shape)
    #     print(train_y.shape)
    
    ck_dl = ChunkDataloader(739364600, 13000000, 13000, '../data/zj/input_data/test_set')
    for batch in tqdm(ck_dl):
        test_x = batch
        # print(test_x.shape)