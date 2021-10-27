Voice-Vertification :smiley:
=====

# Introduction

## Feature Extractor

### STFT

### Mel Spec

### MFCC

### GMM and GMM-UBM

### JFA

### I-vector 

### D-vector 

### X-vector

## Backend model

### VggVox

### Attention Backend

## Loss Functions

### Contrastive Loss

### Triplet Loss

### GE2E loss



# Dataset

- Data can be found at [link](https://drive.google.com/file/d/1oT_cvFRDLha0E0Yh66zKRrisGgIbgjmZ/view?usp=sharing)
- You should follow the data directory as in `./data`

# Usage

- You can follow colab files in `/notebooks` (not final yet LOL) for quick end2end implementation.
- If you wanna make it complicated, just look through the code in `src` and try step by step (make sure you're in the right folder before run commands):


  - Looking through the data ` python3 utils.py `

  - For preparing dataset brefore training:  ` python3 build_data.py  --data_root  --training_pairs --max_wav_len `

  - For training `` python3 train.py --n_mfcc --sample_rate --batch_size 64 --epoch_n --lin_neurons ``

  - For testing: `` python3 predict.py --limit ``


