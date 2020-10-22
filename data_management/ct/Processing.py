from typing import List, Dict
import numpy as np
from monai.transforms import Compose
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.spatial.dictionary import Resized, Rotate90d
from monai.transforms.utility.dictionary import ToNumpyd, CastToTyped, SqueezeDimd
from data_management.monai_master.dictionary import LoadDatad
from monai.transforms import ToTensord, AddChanneld
from deprecated import deprecated

from data_management.ct.VolumeIO import ScancoLoader
from data_management.arthritis_transforms import SegmentationAlignd, Contourized, CutToValidSlicesd, ConditionalFlipd, \
    KeepKeyd, SubtractAndDivide, ExtendToValidSlicesd


def build_transformation(transformation: dict):
    key = transformation.keys()
    if len(list(key)) > 1:
        raise ValueError("Only one key is valid for creation of a Transformation.")
    key = list(key)[0]
    return eval(key)(**transformation[key])


def build_processing_pipeline(transformation_settings: List[Dict]):
    return Compose([
        build_transformation(transform) for transform in transformation_settings
    ])
