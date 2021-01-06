# DAST_segmentation
The source code of DAST: Unsupervised Domain Adaptation in Semantic Segmentation Based on Discriminator Attention and Self-Training. 

This is a [pytorch](http://pytorch.org/) implementation. 

### Prerequisites
- Python 3.6
- GPU Memory >= 11G
- Pytorch 1.6.0

### Getting started

- Download [The GTA5 Dataset]( https://download.visinf.tu-darmstadt.de/data/from_games/ )

- Download [The SYNTHIA Dataset]( http://synthia-dataset.net/download-2/ )

- Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )

- Download [The imagenet pretraind model]( https://drive.google.com/drive/folders/1w7GZQTIkuGkNo4a87J3sSmPR2avdZw2_?usp=sharing )

The data folder is structured as follows:
```
├── data/
│   ├── Cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── GTA5/
|   |   ├── images/
|   |   ├── labels/
│   └── 			
└── model_weight/
│   ├── DeepLab_resnet_pretrained.pth
    ├── vgg16-00b39a1b-updated.pth
...
```
### Train
1. First train DA and choose the best weight evaluated by our established [validation data]( https://drive.google.com/file/d/1P6Kev8qkISm3BNShPNt9ugbSKuHGHKxj/view?usp=sharing )
```
CUDA_VISIBLE_DEVICES=0 python DA_train.py --snapshot-dir ./snapshots/GTA2Cityscapes
```
2. Then train DAST for several round using the above weight.
```
CUDA_VISIBLE_DEVICES=0 python DAST_train.py --snapshot-dir ./snapshots/GTA2Cityscapes
```

### Evaluate
```
CUDA_VISIBLE_DEVICES=0 python -u evaluate_bulk.py
CUDA_VISIBLE_DEVICES=0 python -u iou_bulk.py
```
Our pretrained model is available via [Google Drive]( https://drive.google.com/drive/folders/1w7GZQTIkuGkNo4a87J3sSmPR2avdZw2_?usp=sharing )

### Citation
This code is heavily borrowed from the baseline [AdaptSegNet]( https://github.com/wasidennis/AdaptSegNet ) and [BDL]( https://github.com/liyunsheng13/BDL )
