from pydub import AudioSegment
import os
from tqdm import tqdm


def chunk_(i,test_audio_file,save_folder='splitaudio',time_len = 1500):
  subject_folder = '/'.join(test_audio_file.split('/')[:-1])
  save_folder = os.path.join(subject_folder,save_folder)
  try:
    os.rmdir(save_folder)
  except:
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
  audio = AudioSegment.from_file(test_audio_file)
  lengthaudio = len(audio)
  start = 0
  threshold = time_len # 1.5s
  end = 0
  counter = 0

  while start < len(audio):
      end += threshold
      chunk = audio[start:end]
      filename = f'{i}_chunk{counter}.wav' 
      filename = os.path.join(save_folder,filename)
      try:
        chunk.export(filename, format="wav")
      except:
        pass
      counter +=1
      start += threshold


def split(subject_folders,save_folder,time_len):
  for subject_foder in tqdm(subject_foders):
    subject_foder = os.path.join(DATA_ROOT, subject_foder)
    wav_paths = glob.glob(os.path.join(subject_foder,'*.wav'))
    for i,wav_path in enumerate(wav_paths):
      chunk_(i,wav_path,save_folder,time_len)
