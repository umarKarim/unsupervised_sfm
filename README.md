# Unsupervised Depth Estimation from Videos with PyTorch 
## Introduction 
This repository is for unsupervised depth extraction from videos. Code is clean and simple, and aimed towards understanding the mechanics of unsupervised SfM-based methods. Repo is inspired by numerous other repositories, especially ![link](https://github.com/JiawangBian/SC-SfMLearner-Release). 

## Requirements 
- PyTorch 
- Torchvision 
- NumPy 
- Matplotlib 

## Usage 
### Training 
For training with the KITTI dataset, download the KITTI dataset from ![kitti](https://1drv.ms/u/s!AiV6XqkxJHE2g1zyXt4mCKNbpdiw?e=ZJAhIl). Use the following bash command.

``` 
bash kitti_train.sh
```
Change path to root directory on your system.

For training with the rectified NYU dataset, download the dataset from ![rectified_nyu](https://1drv.ms/u/s!AiV6XqkxJHE2k3elbxAE9eE4IhRB?e=WoFpdF). Use the following bash command.

```
bash nyu_train.sh
```
Change path to root directory on your system. 

The intermediate results of training are stored in the *int_results* directory. The intermediate models are stored in *models* directory. Tensorboard can be included but I have not as I have issues with X on my server. Rest of the options for training are included in *options.py*.

### Validation 
For qualitative validation with kitti, use the following bash command. 

```
bash kitti_validate_qual.sh
```

For qualitative validation with rectified nyu dataset, use the following bash command.

```
bash nyu_validate_qual.sh
```
Change the path to root directories and models in both scripts. 

The output depth maps are stored in the *output_results* directory. Rest of the options for testing are in *test_options.py*. 

## Discussion 
I have not used the provided intrinsics with the datasets in this code. I am allowing the pose estimating neural network to estimate the intrinsics for general use.



