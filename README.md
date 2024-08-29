# Collaborative group: Composed image retrieval via consensus learning from noisy annotations

[[KBS](https://www.sciencedirect.com/science/article/pii/S095070512400769X)] [[arxiv](https://arxiv.org/abs/2402.00086)] 

The directory contains source code of the published article:

Zhang et al's Collaborative group: Composed image retrieval via consensus learning from noisy annotations.

## Data


Download these datasets [Shoes](http://tamaraberg.com/attributesDataset/index.html), [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq), and [Fashion200k](https://github.com/xthan/fashion-200k) into the "data" directory.

## Environment Preparation

Please make sure you have installed anaconda or miniconda. The version about `pytorch` and `cudatoolkit` should be depended on your machine.

```shell
conda create -n triplet_image python=3.7 \
conda activate triplet_image \
pip3 install -r requirements.txt
```

## Overview of the workflow

Modify the config files in `config` directory

Run the following script to train and evluate the model:

```shell
CUDA_VISIBLE_DEVICES=3,0,2 python3 main.py --config_path=configs/Shoes_trans_g2_res50_config.json --experiment_description=layer3+4+0.1:10kl+text3+4--device_idx=3,0,2 --num_workers=8 --batch_size=30 --optimizer='Adam'
```
Example scripts are placed in the current directory named `shoes.sh`, `iq.sh`, and `200k.sh`. 

All the config files are placed in the `pretrain_finetune` folder. Using OpenNMT commands to run the codes and modifiing them according to the needs.


## Acknowledgement

CoSMo: https://github.com/postBG/CosMo.pytorch

RoBerta: https://huggingface.co/docs/transformers/model_doc/roberta


## 	Citation
If you use our code, please cite our work:
```
@article{ZHANG2024112135,
    title = {Collaborative group: Composed image retrieval via consensus learning from noisy annotations},
    journal = {Knowledge-Based Systems},
    volume = {300},
    pages = {112135},
    year = {2024},
    author = {Xu Zhang and Zhedong Zheng and Linchao Zhu and Yi Yang}
}
```
