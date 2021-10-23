import sys
sys.path.append('../utils')
import os  
import random
import numpy as np
from utils import write_json, get_prefix
import glob 
from tqdm import tqdm
from split_waves import split
import argparse


def get_data(wave_paths, save_train_file, save_test_file, limit, test_size, subjects, thres_len):
    train_total = {}
    test_total = {}
    test_len = int(limit*test_size)
    train_len = limit - test_len
    check_exist = []

    for idx in tqdm(range(limit)):
      file1 = wave_paths[idx]
      if os.stat(file1).st_size < thres_len:
        continue
      predix, subject1= get_prefix(file1)
      label = random.choice([0,1]) 

      try: 
        if label == 1:
          wav_files = np.array(glob.glob(predix+'/*.wav'))
          other_wav_files = wav_files[wav_files!=file1]
          file2 = random.choice(other_wav_files)
          if os.stat(file2).st_size < thres_len:
            continue
        elif label == 0:
          other_folders = subjects[subjects!=subject1]
          predix_2 = predix.replace(subject1,random.choice(other_folders))
          file2 = random.choice(glob.glob(predix_2+'/*.wav'))
          if os.stat(file2).st_size < thres_len:
            continue
        pair = file1+';'+file2
        check_exist_pair = set([file1,file2])
        if check_exist_pair not in check_exist:
          check_exist.append(check_exist_pair)
          if idx<=train_len:
            train_total[pair] = int(label)
          else:
            test_total[pair] = int(label)
      except:
            continue
          

    write_json(save_train_file,train_total)
    print(len(train_total))
    write_json(save_test_file,test_total)
    print("SAVE PAIRS !")


def pad(em, max_sequence_len = 40): 
    if em.shape[2] < max_sequence_len:
        pad = np.zeros((em.shape[0],em.shape[1],max_sequence_len - em.shape[2]))
        em = np.concatenate((em, pad), axis=2)
    else:
        em = em[:,:,:max_sequence_len] 
    assert em.shape == (1,em.shape[1],max_sequence_len)
    return em[0].T


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Processing data.')
  parser.add_argument('--data_root', type=str, default = '../../data/Train-Test-Data/dataset', help='path to your dataset')
  parser.add_argument('--training_pairs', type= int , default=1000, help='numbers of voice pairs you wanna train')
  parser.add_argument('--max_wav_len', type= int , default=1500, help='miliseconds - cut file into max_wav_len files')
  parser.add_argument('--test_size', type= float , default=0.2, help='test size to split train test')
  parser.add_argument('--thres_len', type= int , default=50000, help='min volumn size of file to accept')
  args = vars(parser.parse_args())
  DATA_ROOT = args['data_root'] 
  subject_folders = os.listdir(DATA_ROOT)
  # split(subject_folders, time_len = args['max_wav_len'])
  total_wav_files = glob.glob(os.path.join(DATA_ROOT,'*/*.wav')) 
  random.shuffle(total_wav_files)
  get_data(wave_paths = total_wav_files, save_train_file = '../../data/train_data.json', save_test_file = '../../data/test_data.json', limit = args['training_pairs'] , test_size = args['test_size'], subjects = np.array(subject_folders), thres_len=args['thres_len'])
