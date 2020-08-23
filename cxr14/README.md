
# CXR14 

## Task overview
Task Overview In this task, the objective is to identify the presence of 14 diseases in a given chest X-ray image. 
Chest X-ray14 dataset is the first largescale dataset on 14 common diseases in chest X-rays.
The dataset contains 112,120 images from 30,805 unique patients. 
Image-level labels are mined from image attached reports using natural language processing techniques (each image can
have multi-labels). 

## Dataset
Download the cxr14 dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC and un-tar all files. 
You can specify where the data is located in the run script with the below option.
```
--path_to_images <PATH_TO_CXR14_DATA>
```

## Environment
You may start with the below docker file and install additional packages. 

```dockerfile
FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel
```

```sh
pip install -r requirements.txt
```

Our code uses [wandb](https://app.wandb.ai/) for tracking metrics and losses. So you must login before starting the run.
```
wandb login <YOUR_WANDB_API_KEY>
```

## Train
To train resnet50-ACM from scratch, you may use the below code.
```
sh scripts/train.sh
```

## Evaluate 
If you just want to evaluate the trained model, you may use the below script. 
You may download the pretrained model from [here](https://drive.google.com/drive/folders/1h_lmNtpfb1KfOFzxapgWWTKF-nkzPhkM?usp=sharing). 


```
sh scripts/eval.sh
```

##### result

|label             |auc               |
|------------------|------------------|
|Atelectasis       |0.8342763556741988|
|Cardiomegaly      |0.9071881856141923|
|Consolidation     |0.8087410332298132|
|Edema             |0.9021789678229688|
|Effusion          |0.8866042854223145|
|Emphysema         |0.9482558484970659|
|Fibrosis          |0.8508758545762228|
|Hernia            |0.9477317680086773|
|Infiltration      |0.7193759339295139|
|Mass              |0.8628153089659782|
|Nodule            |0.8153894291376429|
|Pleural_Thickening|0.8007042676242088|
|Pneumonia         |0.7725149814338228|
|Pneumothorax      |0.8976015387655567|


#### Acknowledgement
This code is refactored from https://github.com/jrzech/reproduce-chexnet
