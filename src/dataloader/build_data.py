import sys
sys.path.append('../utils')
import os  
import numpy as np
from utils import write_json


def get_data(wave_paths, save_train_file = '../../data/train_data.json', save_test_file = '../../data/test_data.json', limit = 15000 , test_size = 0.2, subjects = np.array(subject_folders), thres_len=50000):
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
        if file1.endswith('wav') and file2.endswith('wav'):
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
    write_json(save_test_file,test_total)


def pad(em, max_sequence_len = 40): 
    if em.shape[2] < max_sequence_len:
        pad = np.zeros((em.shape[0],em.shape[1],max_sequence_len - em.shape[2]))
        em = np.concatenate((em, pad), axis=2)
    else:
        em = em[:,:,:max_sequence_len] 
    assert em.shape == (1,em.shape[1],max_sequence_len)
    return em[0].T
