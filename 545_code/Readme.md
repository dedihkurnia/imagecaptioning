# Caption Generation with Attention
* Show, Attend and tell: http://arxiv.org/abs/1502.03044
* Borrowed the idea from https://github.com/kelvinxu/arctic-captions

## Code
* make_flickr_dataset.py: Extracts conv5_3 layer activations of VGG Network for flickr30k images, and save them in 'data/feats.npy'
* model_tensorflow.py: Main codes

## Usage
* Download flickr30k Dataset.
* Extract VGG conv5_3 features using make_flickr_dataset.py
* Train: run train() in model_tensorflow.py
* Test: run test() in model_tensorflow.py

![alt tag](https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow/blob/master/attend.jpg)
