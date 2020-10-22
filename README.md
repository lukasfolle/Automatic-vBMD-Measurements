# Code for the paper "Fully Automatic Bone Mineral Density Measurements using Deep Learning"
This is the repository corresponding to the above mentioned paper.
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

### Validation
To validate the trained models on full resolution images, please use the "validate_full_res.py" file in the "deep_learning" folder.

### Inference
To use the trained models in a production environment please use "inference.py" in the "deep_learning" folder.