import sys
sys.path.append('../utils')
import numpy as np
from utils import read_json
from features import get_mfcc_features
from build_data import get_data, pad
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os 


class SpeechDataset(Dataset):
    def __init__(self, json_file, max_sequence_len, sample_rate, n_mfcc, mode = 'train'):
        super(SpeechDataset, self).__init__()
        self.data = list(read_json(json_file).items())
        self.max_sequence_len = max_sequence_len
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.mode = mode

    def __getitem__(self, idx):
        pair, label = self.data[idx]
        pair = pair.split(';')
        file1,file2 = pair[0],pair[1]
        em1 = get_mfcc_features(file = file1,sample_rate = self.sample_rate, mode = self.mode, n_mfcc = self.n_mfcc).detach().numpy()
        em2 = get_mfcc_features(file = file2,sample_rate = self.sample_rate, mode = self.mode, n_mfcc = self.n_mfcc).detach().numpy()
        em1 = pad(em1,self.max_sequence_len)
        em2 = pad(em2,self.max_sequence_len)
        em1 = torch.from_numpy(em1.astype(np.float32))
        em2 = torch.from_numpy(em2.astype(np.float32))
        return em1, em2, torch.LongTensor([int(label)])

    def __len__(self):
        return len(self.data)
 

def get_loader(bs, nw, max_sequence_len, sample_rate, n_mfcc, SAVE_TRAIN_PATH, SAVE_VAL_PATH):  

  train_dataset = SpeechDataset(SAVE_TRAIN_PATH, max_sequence_len, sample_rate, n_mfcc, mode='train')
  test_dataset = SpeechDataset(SAVE_VAL_PATH, max_sequence_len, sample_rate, n_mfcc, mode='val')

  train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        num_workers=nw
  )

  test_loader = DataLoader(
          dataset=test_dataset,
          batch_size=bs,
          num_workers=nw
  )
  print("DONE LOADING DATA !")
  return train_loader,test_loader