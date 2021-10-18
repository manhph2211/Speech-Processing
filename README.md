Voice-Vertification :smiley:
=====

# Introduction

- This project bala ... 
- Thanks to ...

## MFCC

- Well, MFCC is ...

## Xvector

- Features ...

## Simease 

- For ...

# Dataset

- Data can be found at [link](https://drive.google.com/file/d/1oT_cvFRDLha0E0Yh66zKRrisGgIbgjmZ/view?usp=sharing)

# Usage

- You can follow colab files in `/notebooks` for quick end2end implementation, but if you wanna make it complicated, just look through the code in 	`src` and try step by step:

```

- For splitting wave files into Xsecond-files:  ** python3 utils.py --length X --save_folder splitaudio **

- For making dataset before training: ** python3 dataset.py --test_size 0.2 --save_train_json ../data/train.json --save_val_json ../data/val.json

- For training ** python3 train.py --embedding_type 'mfcc' --add_aug True --max_len 40 --up_channels False --resize None --bs 64 --n_epoch 300 --lr 0.0001 --lin_neurons 512**   

- For testing: ** python3 test.py -save_txt result.txt**

```

