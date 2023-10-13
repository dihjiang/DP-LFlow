# DP-LFlow

This code repo corresponds to the paper "DP-LFlow: Differentially Private Latent Flow for Scalable Sensitive Image Generationg" in TMLR.

This repo is originally developed using Python 3.6 (but I believe it should also be compatible with higher versions of Python).

## Preparation
### 1. Pyvacy
[Pyvacy](https://github.com/ChrisWaites/pyvacy) (under Apache-2.0 license) is a PyTorch version of an older version of [Tensorflow Privacy](https://github.com/tensorflow/privacy). 
```
# compute privacy parameters
from pyvacy.analysis import epsilon
epsilon(10, 1, 1.25, 300) # => epsilon=10 (subsampling rate=1/10, noise level=1.25, num of iterations=300)
epsilon(10, 1, 5.1 , 100) # => epsilon=1  (subsampling rate=1/10, noise level=5.1 , num of iterations=100)
epsilon(10, 1, 8.5 , 300) # => epsilon=1  (subsampling rate=1/10, noise level=8.5 , num of iterations=300)
```

### 2. CelebA and CelebA-HQ dataset
Torchvision cannot download [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) automatically. You need to manually download this dataset, then specify its path in datasets.py (line 48). You can start with MNIST and FMNIST for convenience.

Similar for CelebA-HQ. Please follow [this link](https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch/blob/main/Face_Gender_Classification_Test_with_CelebA_HQ.ipynb) to download this dataset (arranged by the gender), then specify its path in datasets.py (line 80).


### 3. Install necessary packages
Install necessary packages in requirements.txt.

## Running DP-LFlow
### Training 
```
## MNIST: parse class number between 0 and 9 to --select to specify the subset to be trained on, which can be run in parallel
# epsilon=10 (noise level=1.25, #iterations=300)
python3 train_DPAENF_unconditional.py  --dataset mnist  --ae_lr 0.01  --nf_lr 0.005  --code_length 20  --ae_h 64  --nf_h 200  --num_blocks 9  --grad_clip 0.1  --iterations 302  --log_step 100  --batch_size 500  --noise 1.25  --select 0
# epsilon=1  (noise level=5.1 , #iterations=100)
python3 train_DPAENF_unconditional.py  --dataset mnist  --ae_lr 0.01  --nf_lr 0.005  --code_length 20  --ae_h 64  --nf_h 200  --num_blocks 9  --grad_clip 0.1  --iterations 102  --log_step 100  --batch_size 500  --noise 5.1  --select 0

## Fashion MNIST: parse class number between 0 and 9 to --select to specify the subset to be trained on, which can be run in parallel
# epsilon=10 (noise level=1.25, #iterations=300)
python3 train_DPAENF_unconditional.py  --dataset fmnist  --ae_lr 0.01  --nf_lr 0.005  --code_length 20  --ae_h 64  --nf_h 200  --num_blocks 9  --grad_clip 0.1  --iterations 302  --log_step 100  --batch_size 500  --noise 1.25  --select 0
# epsilon=1  (noise level=5.1 , #iterations=100)
python3 train_DPAENF_unconditional.py  --dataset fmnist  --ae_lr 0.01  --nf_lr 0.005  --code_length 20  --ae_h 64  --nf_h 200  --num_blocks 9  --grad_clip 0.1  --iterations 102  --log_step 100  --batch_size 500  --noise 5.1  --select 0

## CelebA: parse class number in {0,1} to --select to specify the subset to be trained on, which can be run in parallel
# epsilon=1  (noise level=8.5 , #iterations=300)
python3 train_DPAENF_unconditional.py  --dataset celeba  --ae_lr 0.005  --nf_lr 0.005  --weight_decay 1e-5  --code_length 32  --ae_h 256  --nf_h 200  --num_blocks 9  --grad_clip 0.01  --iterations 102  --log_step 300  --batch_size 5000  --noise 8.5  --select 0

# CelebA-HQ: the following two command lines can run in parallel
# epsilon=10 (noise level=1.25 , #iterations=300)
python3 train_DPAENF_unconditional.py  --dataset celebahq  --ae_lr 0.005  --nf_lr 0.005  --weight_decay 1e-5  --code_length 64  --ae_h 512  --nf_h 256  --num_blocks 12  --grad_clip 10.0  --iterations 302  --log_step 100  --batch_size 1400  --noise 1.25  --select 0
python3 train_DPAENF_unconditional.py  --dataset celebahq  --ae_lr 0.005  --nf_lr 0.005  --weight_decay 1e-5  --code_length 64  --ae_h 512  --nf_h 256  --num_blocks 12  --grad_clip 10.0  --iterations 302  --log_step 100  --batch_size 800  --noise 1.25  --select 1

```

### Test 
```
## Here we fix epsilon=1 for all following experiments. Refer to training parameters to change --noise and --num_iters accordingly for epsilon=10.

# Generate figures for all classes
python3 test_mnist.py   --dataset mnist  --grad_clip 0.1  --num_iters 100  --noise 5.1  --n_class 10  --ae_lr 0.01  --nf_lr 0.005  --weight_decay 1e-5  --code_length 20  --ae_h 64  --nf_h 200  --num_blocks 9  --batch_size 500  --sharpness 5.0
python3 test_mnist.py   --dataset fmnist  --grad_clip 0.1  --num_iters 100  --noise 5.1  --n_class 10  --ae_lr 0.01  --nf_lr 0.005  --weight_decay 1e-5  --code_length 20  --ae_h 64  --nf_h 200  --num_blocks 9  --batch_size 500  --sharpness 8.0
python3 test_celeba.py   --dataset celeba  --grad_clip 0.01  --num_iters 300  --noise 8.5  --n_class 2  --ae_lr 0.005  --nf_lr 0.005  --weight_decay 1e-5  --code_length 32  --ae_h 256  --nf_h 200  --num_blocks 9  --batch_size 5000  --sharpness 2.0

# Compute FID.
python3 test_mnist.py   --dataset mnist  --grad_clip 0.1  --num_iters 100  --noise 5.1  --n_class 10  --ae_lr 0.01  --nf_lr 0.005  --weight_decay 1e-5  --code_length 20  --ae_h 64  --nf_h 200  --num_blocks 9  --batch_size 500  --compute_FID  --num_samples 60000  --sharpness 5.0  --test
python3 -m pytorch_fid ./Imgs/mnist/True  ./Imgs/mnist/Gen/60000  --batch-size 100

python3 test_mnist.py   --dataset fmnist  --grad_clip 0.1  --num_iters 100  --noise 5.1  --n_class 10  --ae_lr 0.01  --nf_lr 0.005  --weight_decay 1e-5  --code_length 20  --ae_h 64  --nf_h 200  --num_blocks 9  --batch_size 500  --compute_FID  --num_samples 60000  --test  --sharpness 8.0
python3 -m pytorch_fid ./Imgs/fmnist/True  ./Imgs/fmnist/Gen/60000  --batch-size 100

python3 test_celeba.py   --dataset celeba  --grad_clip 0.01  --num_iters 300  --noise 8.5  --n_class 2  --ae_lr 0.005  --nf_lr 0.005  --weight_decay 1e-5  --code_length 32  --ae_h 256  --nf_h 200  --num_blocks 9  --batch_size 5000  --sharpness 2.0  --compute_FID  --num_samples 60000  --test
python3 -m pytorch_fid ./Imgs/celeba/True  ./Imgs/celeba/Gen/60000  --batch-size 100

# Run classification, change --seed five times to run the classification 5 times
python3 test_mnist.py   --dataset mnist  --grad_clip 0.1  --num_iters 100  --noise 5.1  --n_class 10  --ae_lr 0.01  --nf_lr 0.005  --weight_decay 1e-5  --code_length 20  --ae_h 64  --nf_h 200  --num_blocks 9  --batch_size 500  --ML_ACC  --num_samples 60000  --sharpness 5.0  --seed 1  --test
python3 test_mnist.py   --dataset fmnist  --grad_clip 0.1  --num_iters 100  --noise 5.1  --n_class 10  --ae_lr 0.01  --nf_lr 0.005  --weight_decay 1e-5  --code_length 20  --ae_h 64  --nf_h 200  --num_blocks 9  --batch_size 500  --ML_ACC  --num_samples 60000  --sharpness 8.0  --seed 1  --test
python3 test_celeba.py   --dataset celeba  --grad_clip 0.01  --num_iters 300  --noise 8.5  --n_class 2  --ae_lr 0.005  --nf_lr 0.005  --weight_decay 1e-5  --code_length 32  --ae_h 256  --nf_h 200  --num_blocks 9  --batch_size 5000  --ML_ACC  --num_samples 60000  --sharpness 2.0  --seed 1 --test  

# Run intra-dataset Out-of-distribution (OOD) detection
python3 test_mnist.py   --dataset mnist  --grad_clip 0.1  --num_iters 100  --noise 5.1  --n_class 10  --ae_lr 0.01  --nf_lr 0.005  --weight_decay 1e-5  --code_length 20  --ae_h 64  --nf_h 200  --num_blocks 9  --batch_size 500  --OOD  --test
python3 test_mnist.py   --dataset fmnist  --grad_clip 0.1  --num_iters 100  --noise 5.1  --n_class 10  --ae_lr 0.01  --nf_lr 0.005  --weight_decay 1e-5  --code_length 20  --ae_h 64  --nf_h 200  --num_blocks 9  --batch_size 500  --OOD  --test
