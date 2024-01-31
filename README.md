# HoVer-NeXt Training Repository

Contains necessary code to train and evaluate HoVer-NeXt on Lizard-Mitosis and Pannuke.
For inference, please check the [hover-next inference repository](https://github.com/digitalpathologybern/hover_next_inference)

## How to run



### pre-requisites

Setup the environment by running the following commands. Be careful to choose the right pytorch version for your installed CUDA Version.

```bash
conda env create -f environment.yml
conda activate hovernext
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
```

download the necessary data

lizard from here:

And then download the additional mitosis predictions from here:

merge and create folds using: TODO: merge_mitosis.py 


pannuke from here:

and convert using convert_pannuke_to_conic.py 

training parameters are defined in a .toml file. Please check out the examples in the `sample_configs/` folder.

### Train a model:
To start the training run:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py --config "sample_configs/train_pannuke.toml"
```

There is no default logger so if you want to run this script in the background and monitor the log, run:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py --config "sample_configs/train_pannuke.toml" >train.log 2>&1 &
```


### Singularity 

A singularity container can be downloaded from here:
TODO

Multi-GPU / node training is supported via torchrun

```bash
export APPTAINER_BIND="/storage," # make sure that your local FS is mounted
apptainer exec --nv nuc_torch_v16.sif \
    torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py \
	    --config "sample_configs/train_pannuke.toml"
```

## Training on other datasets

Follow along the pannuke code and replace hyperparameters and necessary preprocessing steps along the way.
Data should always be in the same format, see convert_pannuke_to_conic.py.
## Whole slide Inference

Please check the inference repository for WSI/large image inference:

[hover-next inference repository](https://github.com/digitalpathologybern/hover_next_inference)

# License
This repository is licensed under GNU General Public License v3.0. 
If you are intending to use this repository for commercial usecases, please check the licenses of all python packages referenced in the Setup section / described in the requirements.txt and environment.yml.

# Citation
If you are using this repository, please cite:
TODO
