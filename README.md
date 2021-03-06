# Code corresponding to the paper "Deep Learning Methods Allow Fully Automated Segmentation of Metacarpal Bones to Quantify Volumetric Bone Mineral Density"
This repository contains the necessary parts to recreate the results of our paper.
The segmentation prediction on the right was produced by our method. On the left is the corresponding ground-truth manual segmentation.
![GT and model prediction](PredAndSegmentation.svg)

## Overview
The "deep_learning" folder contains all pieces necessary to train the networks mentioned in the paper
In "data_management" the components to interact with the propriatary file formats from the HR-pQCT scanner as well as dataset handling are included.

## Getting started
We recommend using conda for the management of installed packages.
To install the necessary packages run the following command
```
conda env create -f environment.yml
```

### Training
To train a network, use the file "ct_model.py" in the "deep_learning" folder.
```
python deep_learning/ct_model.py
```

### Validation
To validate the trained models on full resolution images, please use the "validate_full_res.py" file in the "deep_learning" folder.
```
python deep_learning/validate_full_res.py
```

### Inference
To use the trained models in a production environment please use "inference.py" in the "deep_learning" folder.
```
python deep_learning/inference.py
```
