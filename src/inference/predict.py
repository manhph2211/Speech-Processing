import torch
import sys
sys.path.append("../utils")
sys.path.append("../train")
sys.path.append("../models")
sys.path.append("../dataloader")
from build_data import get_data, pad
from features import get_mfcc_features
from tqdm import tqdm
from xvectors import Xvector, Classifier
from sklearn.metrics import accuracy_score
import numpy as np 
from tqdm import tqdm
import argparse
import pandas as pd
import os  
import random
import glob


def predict(file1,file2,classifier,feature_extractor,n_mfcc, max_sequence_len, sample_rate, device):
    feature_extractor.eval()
    classifier.eval()  
    with torch.no_grad():
        em1 = get_mfcc_features(file = file1,sample_rate = sample_rate, mode = "test", n_mfcc = n_mfcc).detach().numpy()
        em2 = get_mfcc_features(file = file2,sample_rate = sample_rate, mode = "test", n_mfcc = n_mfcc).detach().numpy()
        
        em1 = pad(em1,max_sequence_len)
        em2 = pad(em2,max_sequence_len)

        em1 = torch.from_numpy(em1.astype(np.float32))
        em2 = torch.from_numpy(em2.astype(np.float32))

        em1 = em1.unsqueeze(dim=0).to(device)
        em2 = em2.unsqueeze(dim=0).to(device)

        xvec1 = feature_extractor(em1)
        xvec2 = feature_extractor(em2)

        xvec = torch.hstack((xvec1,xvec2))
        out = classifier(xvec.reshape((xvec.shape[0],-1)))          
        _, predict_ = out.max(dim=1)
    return predict_


def submission(save_file, classifier, feature_extractor, n_mfcc, max_sequence_len, sample_rate, device = 'cpu', limit = 5, csv_public_test='../../data/Train-Test-Data/public-test.csv',test_wave_folder = '../..data/Train-Test-Data/public-test'):
	test_files = pd.read_csv(csv_public_test)
	test_files['audio_1'] = test_files['audio_1'].apply(lambda x: os.path.join(test_wave_folder, x))
	test_files['audio_2'] = test_files['audio_2'].apply(lambda x: os.path.join(test_wave_folder, x))
	results = []
	with open(save_file,'w') as f:
	    for i,(file1,file2) in enumerate(tqdm(zip(test_files['audio_1'].to_numpy(),test_files['audio_2'].to_numpy()))):
	      out = predict(file1, file2, classifier, feature_extractor, n_mfcc, max_sequence_len, sample_rate, device)
	      file1 = file1.split('/')[-2]
	      file2 = file2.split('/')[-2]
	      results.append(out)
	      if i == limit:
	        break
	      f.write(file1 +','+ file2 +','+ str(out) + ' \n')


def main(hyper_params):

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	feature_extractor = Xvector(in_channels = hyper_params['in_channels'], lin_neurons = hyper_params['lin_neurons'])
	classifier = Classifier(input_shape=(2,1,2 * hyper_params['lin_neurons']),out_neurons = 2) 

	feature_extractor = feature_extractor.to(device)
	classifier = classifier.to(device)

	submission('results.txt', classifier, feature_extractor, hyper_params["n_mfcc"], hyper_params["max_sequence_len"], hyper_params["sample_rate"], device)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Testing...')
	parser.add_argument('--n_mfcc', type=int, default = 40, help='n_mfcc when calculate mfcc')
	parser.add_argument('--max_sequence_len', type= int , default=40, help='pad to max_wav_len')
	parser.add_argument('--sample_rate', type= int , default=16000, help='sample rate')
	parser.add_argument('--in_channels', type= int , default=40, help='numbers of in_channels - xvector')
	parser.add_argument('--lin_neurons', type= int , default=512, help='numbers of hidden in_channels - xvector')

	args = vars(parser.parse_args())
	main(args)
