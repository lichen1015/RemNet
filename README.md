# RemNet
the official implementation  of "RemNet: A lightweight backbone for UAV Object Detection"

# Environments
```shell
conda create -n remnet python=3.11 -y
conda activate remnet
pip3 install torch torchvision torchaudio # match your cuda version
pip install -U openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0,<3.1.0"
pip install -r requirements.txt
pip install einops timm albumentations
mim install -v -e .
python setup.py install
```

# Prepare VisDrone2019-DET Dataset
Download and extract [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) dataset in the following directory structure:
```txt
├── VisDrone2019-DET-COCO
    ├── train2019
        ├── 0000002_00005_d_0000014.jpg 
        ├── ...
    ├── val2019
        ├── 0000001_02999_d_0000005.jpg
        ├── ...
    └── annotations
        ├── VisDrone2019-DET_train_coco.json
        ├── VisDrone2019-DET_val_coco.json
```
# Train
Train with 4 GPUs in one node:
```shell
bash ./tools/dist_train.sh self_config/remnet_visdrone.py 4 --amp --work-dir ./output_dir
```

Train with Single GPU in one node:
```shell
python ./tools/train.py self_config/remnet_visdrone.py --amp --work-dir ./output_dir
```

# Acknowledgements
We thank but not limited to following repositories for providing assistance for our research:
* [MMYOLO](https://github.com/open-mmlab/mmyolo)
