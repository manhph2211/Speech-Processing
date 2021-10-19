import sys
sys.path.append('../utils')

from utils import read_json
from features import get_mfcc_features
from build_data import get_data, pad
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    def __init__(self, json_file, max_sequence_len = 40, mode = 'train'):
        super(SpeechDataset, self).__init__()
        self.data = list(read_json(json_file).items())
        self.max_sequence_len = max_sequence_len
        self.mode = mode
        self.up_channel = up_channel

    def __getitem__(self, idx):
        pair, label = self.data[idx]
        pair = pair.split(';')
        file1,file2 = pair[0],pair[1]
        em1 = get_mfcc_features(file = file1,mode = self.mode).detach().numpy()
        em2 = get_mfcc_features(file = file2,mode = self.mode).detach().numpy()
        em1 = pad(em1,self.max_sequence_len)
        em2 = pad(em2,self.max_sequence_len)
        em1 = torch.from_numpy(em1.astype(np.float32))
        em2 = torch.from_numpy(em2.astype(np.float32))
        return em1, em2, torch.LongTensor([int(label)])

    def __len__(self):
        return len(self.data)
 