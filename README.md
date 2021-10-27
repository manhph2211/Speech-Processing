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

### Wav2vec

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

# References

- Lots of useful tutroials [Youtube Channel](https://www.youtube.com/c/ValerioVelardoTheSoundofAI)
- Paper GMM [pdf](https://maelfabien.github.io/machinelearning/Speech1/#limits-of-gmm-ubm)
- Paper Adaptive GMM [pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.117.338&rep=rep1&type=pdf)
- Paper JFA [pdf](https://www1.icsi.berkeley.edu/Speech/presentations/AFRL_ICSI_visit2_JFA_tutorial_icsitalk.pdf)
- Paper I vector [pdf](http://groups.csail.mit.edu/sls/archives/root/publications/2010/Dehak_IEEE_Transactions.pdf) 
- Paper D vector [pdf](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf)
- Paper X vector [pdf](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)
- Paper attention backend with x vector [pdf](https://arxiv.org/pdf/2104.01541.pdf)
- Wav2vec [pdf](https://huggingface.co/transformers/model_doc/wav2vec2.html)
- Ge2e Loss Paper [pdf](https://arxiv.org/pdf/1710.10467.pdf)
- Vggvox Paper [pdf](https://arxiv.org/pdf/1806.05622.pdf)
