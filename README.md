Voice-Vertification :smiley:
=====

# Introduction


## MFCC


## Xvector


## Simease 

# Dataset

- Data can be found at [link](https://drive.google.com/file/d/1oT_cvFRDLha0E0Yh66zKRrisGgIbgjmZ/view?usp=sharing)

# Usage

- You can follow colab files in `/notebooks` for quick end2end implementation, but if you wanna make it complicated, just look through the code in 	`src` and try step by step:

```

- Looking through the data ** python3 utils.py **

- For preparing dataset brefore training:  ** python3 build_data.py  --data_root  --training_pairs --max_wav_len **

- For training ** python3 train.py --n_mfcc --sample_rate --batch_size 64 --epoch_n --lin_neurons**   

- For testing: ** python3 predict.py --limit **

```

