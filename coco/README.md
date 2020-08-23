
# COCO Experiment

The code is based on Detectron2 
https://github.com/facebookresearch/detectron2

## Installation

Installation for Detectron2 can be found in [Detectron2](https://github.com/facebookresearch/detectron2)

#### Minimal Installation guide
```
# install detectron with local files
cd coco
python -m pip install -e detectron2
# If you have 'AT_CHECK undefined' error, then change all AT_CHECK to TORCH_CHECK in detectron2/detectron2/layers/csrc/deformable/deform_conv.h and deform_conv.cu
# If you have "no module named 'fvcore'" error, please install pip install git+https://github.com/facebookresearch/fvcore.git
# If you have "no module named 'pycocotools'" error, please install https://github.com/cocodataset/cocoapi with Cython.

# symlink COCO dataset
cd detectron2
mkdir datasets
ln -s <PATH_TO_COCO_DATASET> ./datasets/coco
```

## Code changes from adding ACM

The current code is modified from the below commit.
```
git checkout f5c0c0979f261dcff448f6867175050ec681a584
```

#### Modified or added files

```
detectron2/config/defaults.py                                                                
detectron2/engine/defaults.py                                                                
detectron2/modeling/backbone/fpn.py                                                          
detectron2/modeling/backbone/resnet.py                                                       
detectron2/modeling/meta_arch/rcnn.py  
detectron2/modeling/backbone/context_module.py
detectron2/modeling/backbone/custom_bottleneck.py
configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_acm.yaml
```

## Train
Detectron2 uses yaml file to configure your run. Checkout the below example yaml files. 
- resnet50-baseline [mask_rcnn_R_50_FPN_1x.yaml](detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml).
- resnet50-ACM [mask_rcnn_R_50_FPN_1x_acm.yaml](detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_acm.yaml).

You may train the network with below script. 
```
cd detectron2
# baseline
python tools/train_net.py \
        --num-gpus 8 \
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
# acm
python tools/train_net.py \
        --num-gpus 8 \
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_acm.yaml
```

## Evaluate 

If you just want to evaluate the trained model, you may use the below script. 
You may download the ResNet50-ACM pretrained model from [here](https://drive.google.com/file/d/1pepXC0mkOm-zeEp3-CNavL8DQyKixkTB/view?usp=sharing).

And specify the WEIGHTS path in config file 
(ex: [mask_rcnn_R_50_FPN_1x_acm_eval.yaml](detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_acm_eval.yaml)).

```
# eval only
python tools/train_net.py \
        --num-gpus 8 \
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_acm_eval.yaml \
        --eval-only                           
```

##### Result: 

|     |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:---:|:------:|:------:|:------:|:------:|:------:|:------:|
| bbox| 39.942 | 61.582 | 43.304 | 24.271 | 43.533 | 51.154 |
| segm| 36.395 | 58.403 | 38.634 | 18.482 | 39.202 | 51.593 |
