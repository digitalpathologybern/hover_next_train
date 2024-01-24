# HoVer-NeXt Training Repository

Contains necessary code to train and evaluate HoVer-NeXt on Lizard-Mitosis and Pannuke.

## How to run



### pre-requisites

Setup the environment

```bash
conda env create -f environment.yml
conda activate hovernext
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
```

download the necessary data

training parameters are defined in a .toml file. Please check out the examples in the `sample_configs/` folder.

To start the training run:

```bash
python3 train.py --config "sample_configs/train_pannuke.toml"
```

There is no default logger so if you want to run this script in the background and monitor the log, run:

```bash
python3 train.py --config "sample_configs/train_pannuke.toml" >train.log 2>&1 &
```


### Singularity 

A singularity container can be downloaded from here:
TODO

Multi-GPU / node training is supported via torchrun

```bash
export APPTAINER_BIND="/storage:/storage" # make sure that your local FS is mounted
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
