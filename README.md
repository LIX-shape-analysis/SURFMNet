# SURFMNet
This is a fork from Jean Michel repository https://github.com/JM-data/Unsupervised_DeepFunctionalMaps

Source code and data associated with the ICCV'19 oral paper "Unsupervised Deep Learning for Structured Shape Matching" will be maintained and updated here in future.

### Dependency

The code is tested under TF1.6 GPU version and Python 3.6 on Ubuntu 16.04, with CUDA 9.0 and cuDNN 7. It requires Python libraries `numpy`, `scipy`.

### Prepare Your Own Data

Please run  bash Prepare_data.sh

### Shape Matching

To train a DFMnet model to obtain matches between shapes without any ground-truth or geodesic distance matrix (using only a shape's Laplacian eigenvalues and eigenvectors and also Descriptors on shapes):

        python train_DFMnet.py

To obtain matches after training for a given set of shapes:

        python test_DFMnet.py
        
Visualization of functional maps at each training step is possible with tensorboard:

        tensorboard --logdir=./Training/


