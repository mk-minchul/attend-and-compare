#!/usr/bin/env bash

# baseline
python tools/train_net.py --num-gpus 8 --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml

# v35
python tools/train_net.py --num-gpus 8 --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_acm.yaml

