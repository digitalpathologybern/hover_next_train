# HoVer-NeXt Training Repository

Contains necessary code to train and evaluate HoVer-NeXt on Lizard-Mitosis and Pannuke.
For inference, please check the [hover-next inference repository](https://github.com/digitalpathologybern/hover_next_inference)

Find the Publication here: [https://openreview.net/pdf?id=3vmB43oqIO](https://openreview.net/pdf?id=3vmB43oqIO)

## How to run

### pre-requisites

Setup the environment by running the following commands. Be careful to choose the right pytorch version for your installed CUDA Version.

```bash
conda env create -f environment.yml
conda activate hovernext
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
```

#### Lizard-Mitosis training
For Lizard-Mitosis and the mitosis data training, download [lizard_mitosis.zip](https://zenodo.org/records/10636591/files/lizard_mitosis.zip?download=1) and [mitosis_ds.zip](https://zenodo.org/records/10636591/files/mitosis_ds.zip?download=1) from [zenodo](https://zenodo.org/records/10636591)
and extract the folders.

#### PanNuke training
If available, download PanNuke from here: [TIA-Warwick](https://warwick.ac.uk/fac/cross_fac/tia/data/)
and convert using convert_pannuke_to_conic.py like so:

```bash
python3 convert_pannuke_to_conic.py --path "/path-to/pannuke/masks/"
```

training parameters are defined in a .toml file. Please check out the examples in the `sample_configs/` folder.

### Train a model:
To start the training run the following, but make sure to view and modify the ```*.toml``` before.

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py --config "sample_configs/train_pannuke.toml"
```

There is no default logger so if you want to run this script in the background and monitor the log, run:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py --config "sample_configs/train_pannuke.toml" >train.log 2>&1 &
```

### Run hyperparameter search:

After training a model, run hyperparamter search to find the best foreground and seed thresholds for watershed

```bash
python3 hp_search.py --config "your_experiment/params.toml"
```

### Run evaluation

After hyperparamter search, model evaluation can be done via

```bash
python3 python3 evaluate.py --exp "your_experiment" --tta 16
```

This creates a new folder within the experiment folder that contains the experiment results.

To run the eosinophil validation, download the [eos-val dataset](https://zenodo.org/records/10636591/files/eos_val.zip?download=1) and process it via the hover-next-inference pipeline. Afterwards run:

```bash
python3 python3 eos_eval.py --out "eos_results.csv" --root "/path-to-inference-results/" --val_root "/path-to/eos_val/"
```

### Singularity 

Download the singularity image from [Zenodo](https://zenodo.org/records/10649470/files/hover_next.sif)

Multi-GPU / node training is supported via torchrun. The defined batch size in the ```*.toml``` is the batch size per gpu.

```bash
# don't forget to mount your local directory
export APPTAINER_BINDPATH="/storage"
apptainer exec --nv /path-to-container/hover_next.sif \
    torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py \
	    --config "sample_configs/train_pannuke.toml"
```

## Training on other datasets

Follow along the pannuke code and replace hyperparameters and necessary preprocessing steps along the way.
Data should always be in the same format, see ```convert_pannuke_to_conic.py.```

## Using HoVer-NeXt to finetune on other datasets

Download a pre-trained checkpoint from [here](https://zenodo.org/records/10635618). 
In the training config, you can specify a checkpoint to be loaded, select the downloaded and extracted ```best_model``` file.

## Whole slide Inference

Please check the inference repository for WSI/large image inference:

[hover-next inference repository](https://github.com/digitalpathologybern/hover_next_inference)

# License

This repository is licensed under GNU General Public License v3.0 (See License Info).
If you are intending to use this repository for commercial usecases, please check the licenses of all python packages referenced in the Setup section / described in the requirements.txt and environment.yml.

# Citation

If you are using this code, please cite:
```
@inproceedings{baumann2024hover,
  title={HoVer-NeXt: A Fast Nuclei Segmentation and Classification Pipeline for Next Generation Histopathology},
  author={Baumann, Elias and Dislich, Bastian and Rumberger, Josef Lorenz and Nagtegaal, Iris D and Martinez, Maria Rodriguez and Zlobec, Inti},
  booktitle={Medical Imaging with Deep Learning},
  year={2024}
}
```
and
```
@INPROCEEDINGS{rumberger2022panoptic,
  author={Rumberger, Josef Lorenz and Baumann, Elias and Hirsch, Peter and Janowczyk, Andrew and Zlobec, Inti and Kainmueller, Dagmar},
  booktitle={2022 IEEE International Symposium on Biomedical Imaging Challenges (ISBIC)}, 
  title={Panoptic segmentation with highly imbalanced semantic labels}, 
  year={2022},
  pages={1-4},
  doi={10.1109/ISBIC56247.2022.9854551}}
```
