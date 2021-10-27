from comet_ml import Experiment
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
import warnings
warnings.simplefilter("ignore", UserWarning)


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


def submission(save_file, experiment, classifier, feature_extractor, n_mfcc, max_sequence_len, sample_rate, device = 'cpu', limit = 5, csv_test_file='../../data/private-test.csv',test_wave_folder = '../../data/private-test'):
	test_files = pd.read_csv(csv_test_file)
	test_files['audio_1'] = test_files['audio_1'].apply(lambda x: os.path.join(test_wave_folder, x))
	test_files['audio_2'] = test_files['audio_2'].apply(lambda x: os.path.join(test_wave_folder, x))
	results = []
	with open(save_file,'w') as f:
	    for i,(file1,file2) in enumerate(tqdm(zip(test_files['audio_1'].to_numpy(),test_files['audio_2'].to_numpy()))):
	      out = predict(file1, file2, classifier, feature_extractor, n_mfcc, max_sequence_len, sample_rate, device)
	      mes = str(out.detach().cpu().numpy())
	      experiment.log_audio(audio_data = file1, file_name=f"Pair {i+1} - {mes}")
	      experiment.log_audio(audio_data = file2, file_name=f"Pair {i+1} - {mes}")
	      # experiment.log_other(f"Pair {i+1}",str(out.detach().cpu().numpy()))
	      file1 = file1.split('/')[-1]
	      file2 = file2.split('/')[-1]
	      results.append(out)
	      if i == limit:
	        break
	      f.write(file1 +','+ file2 +','+ str(out.detach().cpu().numpy()[0]) + ' \n')


def main(hyper_params,ckpt1 = '../ckpt/xvec.pth',ckpt2='../ckpt/classify.pth'):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	feature_extractor = Xvector(in_channels = hyper_params['in_channels'], lin_neurons = hyper_params['lin_neurons'])
	classifier = Classifier(input_shape=(64,1,2 * hyper_params['lin_neurons']),out_neurons = 2) 
	feature_extractor = feature_extractor.to(device)
	classifier = classifier.to(device)

	feature_extractor.load_state_dict(torch.load(ckpt1,map_location=torch.device('cpu')))
	classifier.load_state_dict(torch.load(ckpt2,map_location=torch.device('cpu')))
	experiment = Experiment(
	    api_key="",
	    project_name="Speaker Verification",
	    workspace="maxph2211",
	)
	submission('results.txt', experiment, classifier, feature_extractor, hyper_params["n_mfcc"], hyper_params["max_sequence_len"], hyper_params["sample_rate"], device, limit = hyper_params["limit"])


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Testing...')
	parser.add_argument('--n_mfcc', type=int, default = 40, help='n_mfcc when calculate mfcc')
	parser.add_argument('--max_sequence_len', type= int , default=40, help='pad to max_wav_len')
	parser.add_argument('--sample_rate', type= int , default=16000, help='sample rate')
	parser.add_argument('--in_channels', type= int , default=40, help='numbers of in_channels - xvector')
	parser.add_argument('--lin_neurons', type= int , default=512, help='numbers of hidden in_channels - xvector')
	parser.add_argument('--limit', type= int , default=200, help='numbers of predictions')

	args = vars(parser.parse_args())
	main(args)
