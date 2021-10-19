# DD2424-project: Fader Nets
Group 58: Mauricio Byrd Victorica, Hugo Bellem Westin, Fredrik Diffner and Valter Lundegårdh

This repository contains the code for our project, a Tensorflow 2 implementation of "Fader Networks: Manipulating Images by Sliding Attributes" by Lample et. al (Guillaume Lample et al.Fader Networks: Manipulating Images by Sliding Attributes. 2018. arXiv: 1706.00409[cs.CV]). The original author's code is also publicly available at https://github.com/facebookresearch/FaderNetworks.

The code folder contains all the necessary files to execute training with different parameters as is done in a main file or notebook (an example notebook in Colab is available at https://colab.research.google.com/drive/1u0awW5H_UQGEef9I1rgEiht6mzDxiDYn?usp=sharing) and is described next:

training.py: contains the training class used to create training instances for the discriminator and auto-encoder as described in our report and the Fader Nets original paper. The code uses tensorboard to store losses and graphs in a provided "logdir" path.

training_mute.py: performs the same operations as training.py, but skips saving graphs and losses. It is meant to be called only for 1 epoch only to properly initialize a keras model where model weights can be loaded into.

training resume from 500.py: this file is adapted to take in a model that was trained for 500 epochs and continue its training.

loader.py: contains the functions used to process images and attribute data required to visualize network outputs or input images and attributes into the network.

model.py: contains the discriminator and auto-encoder models implemented as described in the report and their corresponding loss functions.

model_classic.py: contains an early implementation of the models where leaky ReLU activation functions where used in the decoder instead of regular ReLU.

Our final auto-encoder model's weights can also be downloaded from https://drive.google.com/file/d/11cxY2jnUSwY_NEakVLsi4rhpvb7H68eG/view?usp=sharing



