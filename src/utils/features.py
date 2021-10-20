from __future__ import division
import augment
import torchaudio
import torchaudio.transforms as tf
import torch   
import random


def augment_wav(signal,sr=16000):
    signal = torch.from_numpy(signal)
    reverb_signal = augment.EffectChain().reverb(50, 50, 50).channels(1).apply(signal, src_info={'rate': sr})
    noise_generator = lambda: torch.zeros_like(reverb_signal).uniform_()
    additive_noise_signal = augment.EffectChain().additive_noise(noise_generator, snr=random.uniform(15,25)).apply(reverb_signal, src_info={'rate': sr})
    time_dropout_signal =  augment.EffectChain().time_dropout(max_seconds=random.uniform(0.1,0.25)).apply(additive_noise_signal, src_info={'rate': sr})
    return time_dropout_signal,sr


def get_mfcc_features(file, sample_rate =  16000, mode = 'val', n_mfcc = 40):
  test , sr = torchaudio.load(file)
  test = tf.Resample(sr,sample_rate)(test)
  if mode == 'train':
    test, sr = augment_wav(test.numpy())
  test = tf.MFCC(sample_rate = sample_rate,n_mfcc = n_mfcc)(test)
  if test.shape[0] !=1:
    test = test[0].unsqueeze(dim=0)
  return test



