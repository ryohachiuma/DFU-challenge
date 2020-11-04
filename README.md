# DFU challenge repository
This repository is for DFU challenge (https://dfu2020.grand-challenge.org/evaluation/submissions/create/). This challenge will be held in MICCAI 2020 (https://www.miccai2020.org/).



This repository is based on [mmdetection](https://github.com/open-mmlab/mmdetection).
Please follow the instruction guide on [installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md)


# Quick Demo
If you want to test with your image, download the one of the pretrained weights of any model (e.g. Deformable Convolution) from the [link](https://keiojp0-my.sharepoint.com/:u:/g/personal/ryo-hachiuma_keio_jp/EcjiVmBnq4xEo-1cZQrTjHkBHpo_MqRMlzFZaWrHE2ts4A?e=3KQutX) and save to the MODEL_PATH. And assume that the image is placed at IMG_PATH. And run the following command.
```
python demo/image_demo.py IMG_PATH configs/feet/deform_low.py MODEL_PATH 
```
This will visualize the bounding boxes of the wound.



# Training models
Faster R-CNN, Deformable Conv2, Cascade R-CNN, pisa
```
python tools/train.py configs/feet/feet_low.py
python tools/train.py configs/feet/deform_low.py
python tools/train.py configs/feet/cascade.py
python tools/train.py configs/feet/pisa.py
```

The pretrained models can be downloaded from the [link](https://keiojp0-my.sharepoint.com/:f:/g/personal/ryo-hachiuma_keio_jp/EqttPiOf3C9DtC5FHZW2qPgB-XpBhjLKDhWPS3RVyY8hiQ?e=eR1MEf)

# Testing models
```
python demo/inference.py configs/feet/feet_low.py work_dirs/feet_low/epoch_17.pth --score-thr=0.7
python demo/inference.py configs/feet/deform_low.py work_dirs/deform_low/epoch_25.pth --score-thr=0.7
python demo/inference.py configs/feet/cascade.py work_dirs/cascade/epoch_23.pth --score-thr=0.7
python demo/inference.py configs/feet/pisa.py work_dirs/pisa/epoch_19.pth --score-thr=0.7
```

# Merging all models
```
python demo/ensemble_nms.py
```

