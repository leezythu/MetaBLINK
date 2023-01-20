# Meta-Learning
In our paper, we apply the meta-learning technique on BLINK. Specifically, we use the meta-learning technique proposed {here}[https://arxiv.org/pdf/1803.09050.pdf] and modify the original training scripts of {biencoder}[https://github.com/facebookresearch/BLINK/blob/main/blink/biencoder/train_biencoder.py] and {crossencoder}[https://github.com/facebookresearch/BLINK/blob/main/blink/crossencoder/train_cross.py].

We use the {Higher}[https://github.com/facebookresearch/higher] tool, which can calculate the second derivative conveniently. {Here}[https://github.com/TinfoilHat0/Learning-to-Reweight-Examples-for-Robust-Deep-Learning-with-PyTorch-Higher] is an example of meta-learning implementation using Higher. 

`train_biencoder.py` contains the core code of our MetaBlink, and you can clone the original BLINK repository and replace the training file with ours. Please note that the 'train_biencoder.py' requires a large GPU memory, so we also provide a naive version of MetaBlink. Please see  'train_biencoder_bs_1.py'. It filters noisy samples one by one (corresponding to batch_size=1), using less GPU memory. 
In the code, the train_dataloader corresponds to the generated noisy samples, and the meta_dataloader corresponds to the few-shot golden samples.
The crossencoder can be coverted to an meta-learning version in the same way as biencoder.