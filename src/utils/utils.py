import os
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random 
import json
from split_waves import split
from features import get_mfcc_features
from pydub import AudioSegment
from pydub.playback import play


def read_json(file):
  with open(file,'r') as f:
    data = json.load(f)
  return data


def write_json(file,data):
  with open(file,'w') as f:
    json.dump(data,f,indent = 4)


def get_prefix(file):
  split = file.split('/')
  return '/'.join(split[:-1]),split[-3]


def listen(file):
  sound = AudioSegment.from_wav(file)
  play(sound)


def show_time_domain(file):
  signal,sr = librosa.load(file)
  plt.figure(figsize=(14, 5))
  librosa.display.waveplot(signal, sr)
  plt.show()


def show_frequency_domain(features):
  plt.figure(figsize=(14,5))
  librosa.display.specshow(features[0].numpy())
  plt.colorbar(format="%+2f")
  plt.show()


if __name__ == '__main__':

  DATA_ROOT = '../../data/Train-Test-Data/dataset'
  subject_folders = os.listdir(DATA_ROOT) # 400 subjects (speakers)
  total_wav_files = glob.glob(os.path.join(DATA_ROOT,'*/*.wav')) # 4K files ?, It should be 10k LOL
  file = total_wav_files[0]
  listen(file)
  show_time_domain(file)
  mfccs = get_mfcc_features(file)
  show_frequency_domain(mfccs)
  # split_waves(subject_folders, time_len = 1500)
