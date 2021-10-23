from comet_ml import Experiment
import os
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import json
from features import get_mfcc_features,augment_wav
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
  return '/'.join(split[:-1]),split[-2]


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
  subject_folders = os.listdir(DATA_ROOT)
  total_wav_files = glob.glob(os.path.join(DATA_ROOT,'*/*.wav')) 
  file = total_wav_files[0]
  listen(file)
  # show_time_domain(file)
  val_mfccs = get_mfcc_features(file,mode="val")
  train_mfccs = get_mfcc_features(file,mode="train")
  # show_frequency_domain(mfccs)
  experiment = Experiment(
      api_key="qEseycgDNNW4vbWOXXm0ctQYo",
      project_name="Speaker Verification",
      workspace="maxph2211",
  )
  experiment.log_dataset_info("dataset",DATA_ROOT)
  experiment.log_image(val_mfccs, name = "mfcc")
  experiment.log_image(train_mfccs, name = "augmented_mfcc") 
  experiment.log_audio(file)
  mes = f"There are {len(subject_folders)} subjects and {len(total_wav_files)} waves file in total"
  experiment.log_other("General info",mes)
  
