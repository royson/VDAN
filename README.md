# VDAN
This is an official implementation of VDAN:Temporal Kernel Estimation for Blind Video Super-Resolution 

The overall code frame work is based on [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch/tree/master/src)

The training data-preprocess is consist with previous works [IKC](https://github.com/yuanjunchai/IKC)  and [DAN](https://github.com/greatlog/DAN)

For the PCD and TSA module, we adopted implementation from [EDVR](https://github.com/xinntao/EDVR/)

## Dependencies
* Pytorch >= 1.3.1
* Nvidia GPU + CUDA
* Pacakges: numpy, imageio, scikit-image, tqdm, opencv

## Train

### Train for real world
```
python3.6 main.py --template=KSR_REDS_blur --model=VDAN --test_kernel_path=[benchmark_frames] --test_blur_path=[benchmark_kernels] --n_colors=3 --batch_size=1 --loss=L1 --n_sequence=3 --no_augment --noise
```

## Test
```
python3.6 main.py --template=KSR_REDS_blur --model=VDAN --test_kernel_path=[frames] --test_blur_path=[kernels] --dir_hr= --n_colors=3 --batch_size=1 --loss=L1 --real_kernel=0 --n_sequence=3 --no_augment --test_only --pre_train=[model_ckpt] --real --noise
```
