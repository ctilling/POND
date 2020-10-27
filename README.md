# POND

This is our implementation for the paper _Probabilistic Neural Kernel Tensor Decomposition_, by Conor Tillinghast, Shikai Fang, Kai Zheng, and Shandian Zhe @  IEEE International Conference on Data Mining (ICDM), 2020.

MIT license


# System Requirements

All code were tested under python 3.6 and TensorFlow 1.14.0. 

# Usage 

The main.py file takes five arguments, which are the path of input training file, the path of testing file, decomposition rank, tensor rank, and number of threads. Batch size, the number of epochs and learning rate are option arguments. The following would give a rank 3 decomposition of a (200,100,200) tensor with learning rate .01, batch size 128 for 50 epochs

		ex) python main.py -tr train.txt -te test-fold.txt -r 3 -dim 200 100 200 -lr .01 -ne 50 -bs 128
		
The default settings for a batch size of 256, a learning rate of .001 and 100 epochs.
	
We also include our implementation of GPTF


