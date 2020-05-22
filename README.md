# CCAI_MASKGEN_ADVENT_TEMP
## What needs to be done before run the code?
1. Create a new subfolder under the root named **pretrained_models**  
2. Download the [pretrained DeepLabv2 model](https://github.com/valeoai/ADVENT/releases/download/v0.1/DeepLab_resnet_pretrained_imagenet.pth) and save it in the folder  
3. Change the comet_ml work_space and user in the config files (shared/config.yml)  
## What needs to be done before test the code?
1. Copy the model that you want to test to the root
2. The model should be named like **model_{30000}.pth**. 30000 can be replaced to any int
3. Go to shared/config.yml find TEST-Model-test_iter and change the number to the number in the name of your .pth file

## How to run the code?
```bash
$ pip install .
$ python train_CCAI.py
```
## How to test the code?
```bash
$ python test_CCAI.py
```
## How to uninstall the package?
```bash
$ pip uninstall ADVENT
```

