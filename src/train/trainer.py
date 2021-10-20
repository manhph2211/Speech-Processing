from comet_ml import Experiment
import sys
sys.path.append('../utils')
sys.path.append('../losses')
sys.path.append('../models')
sys.path.append('../dataloader')
import numpy as np
import os 
from dataset import get_data, get_loader
from xvectors import Xvector,Classifier
import torch
from train import Trainer
import torch.nn as nn
from contrastive import ContrastiveLoss
import argparse


def training_experiment(train_loader,test_loader,experiment,trainer,epoch_n,scheduler1,scheduler2):
	BEST_LOSS = np.inf
	for epoch in range(epoch_n):
	    with experiment.train():
	      mean_train_acc, train_loss_epoch = trainer.train_epoch(train_loader)
	      experiment.log_metrics({
	            "loss": train_loss_epoch,
	            "acc": mean_train_acc
	      }, epoch=epoch)

	    with experiment.test():
	      mean_val_acc, val_loss_epoch = trainer.val_epoch(test_loader)
	      scheduler1.step(val_loss_epoch)
	      scheduler2.step(val_loss_epoch)
	      trainer.save_checkpoint(experiment)
	      experiment.log_metrics({
	            "loss": val_loss_epoch,
	            "acc": mean_val_acc
	      }, epoch=epoch)
	    
	    print("EPOCH: ", epoch+1," - TRAIN_LOSS: ", train_loss_epoch," - TRAIN_ACC: ",mean_train_acc, " || VAL_LOSS: ", val_loss_epoch, " - VAL_ACC: ", mean_val_acc)


def train(hyper_params):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	TRAIN_FILE='../../data/train_data.json'
	VAL_FILE='../../data/test_data.json'	

	experiment = Experiment(
	    api_key="qEseycgDNNW4vbWOXXm0ctQYo",
	    project_name="Speaker Verification",
	    workspace="maxph2211",
	)

	experiment.log_parameters(hyper_params)

	feature_extractor = Xvector(in_channels = hyper_params['in_channels'], lin_neurons = hyper_params['lin_neurons'])
	classifier = Classifier(input_shape=(hyper_params['batch_size'],1,2 * hyper_params['lin_neurons']),out_neurons = 2) # n_class : 2

	feature_extractor = feature_extractor.to(device)
	classifier = classifier.to(device)

	criterion1 = ContrastiveLoss().to(device)
	criterion2 = nn.CrossEntropyLoss().to(device)

	optimizer1 = torch.optim.Adam(feature_extractor.parameters(), lr=hyper_params['learning_rate1'])
	optimizer2 = torch.optim.Adam(classifier.parameters(), lr=hyper_params['learning_rate2'])

	scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, factor=0.8, patience=3, verbose=True)
	scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, factor=0.8, patience=3, verbose=True)
	
	train_loader,test_loader = get_loader(hyper_params["batch_size"],hyper_params["worker_n"],hyper_params["max_sequence_len"], hyper_params["sample_rate"], hyper_params["n_mfcc"], TRAIN_FILE,VAL_FILE)
	trainer = Trainer(feature_extractor,classifier,criterion1,criterion2,optimizer1,optimizer2,hyper_params["loss_ratio"],hyper_params["clip_value"],device)

	training_experiment(train_loader,test_loader,experiment,trainer,hyper_params['epoch_n'],scheduler1,scheduler2)


def main():

	parser = argparse.ArgumentParser(description='Training...')
	parser.add_argument('--n_mfcc', type=int, default = 40, help='n_mfcc when calculate mfcc')
	parser.add_argument('--max_sequence_len', type= int , default=40, help='pad to max_wav_len')
	parser.add_argument('--sample_rate', type= int , default=16000, help='sample rate')
	parser.add_argument('--batch_size', type= int , default=64, help='batch size')
	parser.add_argument('--epoch_n', type= int , default=50, help='numbers of epochs')
	parser.add_argument('--worker_n', type= int , default=2, help='numbers of workers')
	parser.add_argument('--in_channels', type= int , default=40, help='numbers of in_channels - xvector')
	parser.add_argument('--lin_neurons', type= int , default=512, help='numbers of hidden in_channels - xvector')

	parser.add_argument('--loss_ratio', type= float , default=0.1, help='miliseconds - cut file into max_wav_len files')
	parser.add_argument('--learning_rate1', type= float , default=0.00001, help='lr of xvector')
	parser.add_argument('--learning_rate2', type= float , default=0.00005, help='lr of classifer')
	parser.add_argument('--clip_value', type= int , default=1, help='for clip gradient')
	
	args = vars(parser.parse_args())
	train(args)

if __name__ == '__main__':
	main()