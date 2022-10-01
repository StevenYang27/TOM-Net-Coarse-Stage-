# TOM-Net-Coarse-Stage-
Simple reimplementation of Coarse Stage of TOM-Net: Learning Transparent Object Matting from a Single Image, CVPR 2018

## Introduction

This repository is a simple reimplementaion of Coarse Stage of [TOM-Net](https://guanyingc.github.io/TOM-Net/). The official codes can be found [here](https://github.com/guanyingc/TOM-Net). The official codes was written in [Torch](http://torch.ch/), while this repository uses [PyTorch](https://pytorch.org/) to implement the model. The codes are tested on macOS and Ubuntu 16.04.

The architechture of Coarse Stage of TOM-Net is as the left side of the following picture. Please refer to the original paper for further information.
<img width="1121" alt="image" src="https://user-images.githubusercontent.com/57172976/193403316-0344ddf2-cc8b-414c-962b-dfe856c8e301.png">

## Dependencies

- Python 3.7
- PyTorch 1.1
- Opencv-python 4.6
- tqdm
- matplotlib

## Training

To train the model, frist [download](https://drive.google.com/drive/folders/1LNvP5g_U8kO3zBhdKd4ur8mBVunWjNH4) and unzip the full dataset into the `data` direcotry. Then use following command to activate training session.

```
# refer to train.py for complete command line arguments 

python train.py --num_epochs [total number of training epochs] --batch_size [desired batch size]
```

Trained models will be located in the `output` directory.

## Inference

To evaluate the model and generate images. Please use following command.

```
# refer to inference.py for complete command line arguments 

python inference.py --model_path [model path to infer]
```
