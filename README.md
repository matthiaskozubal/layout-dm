# LayoutDM: Discrete Diffusion Model for Controllable Layout Generation (CVPR2023)
This repository is an official implementation of the paper titled above.
Please refer to [project page](https://cyberagentailab.github.io/layout-dm/) or [paper](https://arxiv.org/abs/2303.08137) for more details.

## Setup
Here we describe the setup required for the model training and evaluation.

### Requirements
We check the reproducibility under this environment.
- Python3.7
- CUDA 11.3
- [PyTorch](https://pytorch.org/get-started/locally/) 1.12

We recommend using Poetry (all settings and dependencies in [pyproject.toml](pyproject.toml)).
Pytorch-geometry provides independent pre-build wheel for a *combination* of PyTorch and CUDA version (see [PyG:Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
) for details). If your environment does not match the one above, please update the dependencies.


### How to install
Tested on WSL

#### Repo
  ```bash
  git clone https://github.com/matthiaskozubal/layout-dm
  ```

#### Python 
- miniconda
  ```bash
  conda create -n layout-dm python=3.7 -y
  conda activate layout-dm
  ```

#### Packages
- poetry (see [official docs](https://python-poetry.org/docs/))
  ```bash
  pip install poetry==1.4.2
  poetry env use $(wchich python)
  poetry install  
  ```

**Optional**, instead of using poetry:
- manual 
  ```bash
  pip install fsspec omegaconf torch==1.12.1 seaborn matplotlib torch-geometric==2.3.1 torchaudio==0.12.1 torchvision==0.13.1 hydra-core==1.1.2
  ```

#### Fonts
  ```bash
  sudo apt install fontconfig
  fc-list | grep "LiberationSerif-Regular"
  sudo apt-get install fonts-liberation
  sudo fc-cache -f -v
```

#### Download resources and unzip
``` bash
wget https://github.com/CyberAgentAILab/layout-dm/releases/download/v1.0.0/layoutdm_starter.zip
unzip layoutdm_starter.zip
```

The data is decompressed to the following structure:
```
download
- clustering_weights
- datasets
- fid_weights
- pretrained_weights
```

## Experiment
**Important**: we find some critical errors that cannot be fixed quickly in using multiple GPUs. Please set `CUDA_VISIBLE_DEVICES=<GPU_ID>` to force the model use a single GPU.

Note: our main framework is based on [hydra](https://hydra.cc/). It is convenient to handle dozens of arguments hierarchically but may require some additional efforts if one is new to hydra.

### Demo
Please run a jupyter notebook in [notebooks/demo.ipynb](notebooks/demo.ipynb). You can get and render the results of six layout generation tasks on two datasets (Rico and PubLayNet).

### Training
You can also train your own model from scratch, for example by

```bash
bash bin/train.sh rico25 layoutdm
```

, where the first and second argument specifies the dataset ([choices](src/trainer/trainer/config/dataset)) and the type of experiment ([choices](src/trainer/trainer/config/experiment)), respectively.
Note that for training/testing, style of the arguments is `key=value` because we use hydra, unlike popular `--key value` (e.g., [argparse](https://docs.python.org/3/library/argparse.html)).

### Testing

```bash
poetry run python3 -m src.trainer.trainer.test \
    cond=<COND> \
    job_dir=<JOB_DIR> \
    result_dir=<RESULT_DIR> \
    <ADDITIONAL_ARGS>
```
`<COND>` can be: (unconditional, c, cwh, partial, refinement, relation)

For example, if you want to test the provided LayoutDM model on `C->S+P`, the command is as follows:
```
poetry run python3 -m src.trainer.trainer.test cond=c dataset_dir=./download/datasets job_dir=./download/pretrained_weights/layoutdm_rico result_dir=tmp/dummy_results
```

Please refer to [TestConfig](src/trainer/trainer/hydra_configs.py#L12) for more options available.
Below are some popular options for <ADDITIONAL_ARGS>
- `is_validation=true`: used to evaluate the generation performance on validation set instead of test set. This must be used when tuning the hyper-parameters.
- `sampling=top_p top_p=<TOP_P>`: use top-p sampling with p=<TOP_P>　instead of default sampling.

### Evaluation
```bash
poetry run python3 eval.py <RESULT_DIR>
```

## Run on custom input data
### Based on data objects 
1. Place your objects in proper /data/ subdirectories
  - header files
    - schema: `*.header`
  - images 
    - schema: `*_background.png`, `*_product.png`, or `*_logo.png`
  - text files
    - schema: `*.txt`
2. Generate data from the data stored in /data/ subdirectories
  - `generate_data()`
    - to get the input for the layout dm in a form of an instance of torch_geometric.data.data.Data
3. Make predictions
  - `pred = predict_layout()`
    - to get the predictions from the layour dm in a form of an instance of torch_geometric.data.data.Data
    - defaults:
      - model_name='layoutdm_rico'
      - cond_type='cwh' (no rescaling, just repositioning)
      - n_samples=1
      - 
  - access positions and sizes by `pred.x` and labels by `pred.y`
  - `output = save_pred_to_json(list_files, pred)`
    - save the prediction in a json format
  - `combine_elements_based_on_layout_dm(output)`
    - use the layoutdm output prediction in json format stored in /output/ to combine the input data from /data/ and save them as the final output in /output/. Example:
    - /output/output_option-1.json
    - /output/output_option-1.png 
4 Interpret the output
  - labels:
    - publaynet dataset: ['text', 'title', 'list', 'table', 'figure']
    - rico dataset: ['Text', 'Image', 'Icon', 'Text Button', 'List Item', 'Input', 'Background Image', 'Card', 'Web View', 'Radio Button', 'Drawer', 'Checkbox', 'Advertisement', 'Modal', 'Pager Indicator', 'Slider', 'On/Off Switch', 'Button Bar', 'Toolbar', 'Number Stepper', 'Multi-Tab', 'Date Picker', 'Map View', 'Video', 'Bottom Navigation']


## Citation

If you find this code useful for your research, please cite our paper:

```
@inproceedings{inoue2023layout,
  title={{LayoutDM: Discrete Diffusion Model for Controllable Layout Generation}},
  author={Naoto Inoue and Kotaro Kikuchi and Edgar Simo-Serra and Mayu Otani and Kota Yamaguchi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
  pages={10167-10176},
}
```
