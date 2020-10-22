import os
import numpy as np
import torch
import pandas as pd
from typing import Tuple
from monai.transforms import MapTransform
from monai.transforms.spatial.dictionary import Flipd
from monai.transforms.spatial.array import Flip
from scipy.ndimage import binary_closing
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour
from tqdm import tqdm
import SimpleITK as sitk

from arthritis_utils.General import path_exists_check


class SegmentationAlignd(MapTransform):
    @staticmethod
    def align_segmentation_with_volume(volume: torch.Tensor, segmentation: torch.Tensor,
                                       row: pd.DataFrame) -> torch.Tensor:
        aligned_segmentation = torch.zeros_like(volume, dtype=torch.bool)
        aligned_segmentation[:,
        row["pos_x"]:row["pos_x"] + row["dim_x"],
        row["pos_y"]:row["pos_y"] + row["dim_y"],
        row["pos_z"]:row["pos_z"] + row["dim_z"]] = segmentation
        return aligned_segmentation

    def __call__(self, data):
        data["segmentation"] = self.align_segmentation_with_volume(data["volume"], data["segmentation"], data["meta"])
        return data


class Closed(MapTransform):
    def __init__(self, keys, selem: np.ndarray = None):
        """
        :param keys: ...
        :param selem: The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped structuring element (connectivity=1).
        """
        super().__init__(keys)
        self.selem = selem

    def __call__(self, data):
        for key in self.keys:
            data[key] = binary_closing(data[key], self.selem)
        return data


class Contourized(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            data[key] = self._calculate_contour(data[key])
        return data

    def _calculate_contour(self, volume):
        volume_shape = volume.shape
        volume = volume.squeeze()
        contour = np.zeros_like(volume)
        for slice_ind in range(contour.shape[-1]):  # tqdm(range(contour.shape[-1]), desc="Contour extraction"):
            if len((x_non_zero_check := np.nonzero(volume[..., slice_ind])[0])) == 0:
                continue
            gimage = inverse_gaussian_gradient(volume[..., slice_ind], sigma=1.0)
            init_ls = np.zeros(volume[..., slice_ind].shape, dtype=np.int8)
            x_ind, y_ind = np.nonzero(volume[..., slice_ind])
            x_ind_min, x_ind_max = np.min(x_ind), np.max(x_ind)
            y_ind_min, y_ind_max = np.min(y_ind), np.max(y_ind)
            x_lower, x_upper, y_lower, y_upper = self._prune_indices(volume[..., slice_ind], int(0.9 * x_ind_min),
                                                                     int(x_ind_max * 1.1), int(y_ind_min * 0.9),
                                                                     int(y_ind_max * 1.1))
            init_ls[x_lower:x_upper, y_lower:y_upper] = 1
            contour[..., slice_ind] = morphological_geodesic_active_contour(gimage, 35, init_ls,
                                                                            smoothing=2, balloon=-1.4,
                                                                            threshold=.9)
        contour = contour.reshape(volume_shape)
        return contour

    @staticmethod
    def _prune_indices(image, x_lower, x_upper, y_lower, y_upper):
        image_shape = image.shape
        if x_lower < 0:
            x_lower = 0
        if x_upper >= image_shape[0]:
            x_upper = image_shape[0] - 1
        if y_lower < 0:
            y_lower = 0
        if y_upper >= image_shape[1]:
            y_upper = image_shape[1] - 1
        return x_lower, x_upper, y_lower, y_upper


class CutToValidSlicesd(MapTransform):
    def __call__(self, data):
        if "pos_z" in data["meta"].keys():
            valid_slice_ind_start = data["meta"]["pos_z"]
            valid_slice_ind_end = valid_slice_ind_start + data["meta"]["dim_z"]
        else:
            valid_slice_ind_start = int(data["meta"]["pos"][2])
            valid_slice_ind_end = valid_slice_ind_start + int(data["meta"]["dim"][2])
        data["volume"] = data["volume"][..., valid_slice_ind_start:valid_slice_ind_end]
        data["segmentation"] = data["segmentation"][..., valid_slice_ind_start:valid_slice_ind_end]
        return data


class ExtendToValidSlicesd(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            if "pos_z" in data["meta"].keys():
                valid_slice_ind_start = data["meta"]["pos_z"]
                valid_slice_ind_end = valid_slice_ind_start + data["meta"]["dim_z"]
            else:
                valid_slice_ind_start = int(data["meta"]["pos"][2])
                valid_slice_ind_end = valid_slice_ind_start + int(data["meta"]["dim"][2])
            zero_padded_tensor = np.zeros(data["meta"]["original_image_dim"])
            zero_padded_tensor[..., valid_slice_ind_start:valid_slice_ind_end] = data[key]
            data[key] = zero_padded_tensor
        return data


class LoadDicomSITKd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.image_reader = sitk.ImageSeriesReader()
        self.meta_reader = sitk.ImageFileReader()
        self.meta_reader.LoadPrivateTagsOn()

    def _read_dicom(self, dicom_folder):
        dicom_names = self.image_reader.GetGDCMSeriesFileNames(dicom_folder)
        self.image_reader.SetFileNames(dicom_names)
        image = self.image_reader.Execute()
        return image

    def _read_dicom_meta(self, dicom_folder):
        meta_data = dict()
        dicom_files = os.listdir(dicom_folder)
        for dicom_file in dicom_files:
            if dicom_file.endswith(".dcm"):
                self.meta_reader.SetFileName(os.path.join(dicom_folder, dicom_file))
                self.meta_reader.ReadImageInformation()
                for key in self.meta_reader.GetMetaDataKeys():
                    meta_data[key] = self.meta_reader.GetMetaData(key)
                return meta_data

    def __call__(self, data):
        for key in self.keys:
            path_exists_check(data[key])
            dicom_folder = data[key]
            image = self._read_dicom(dicom_folder)
            meta = self._read_dicom_meta(dicom_folder)
            data[key] = {"image": image, "meta": meta}
        return data


class SITKToArray(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            data[key]["image"] = sitk.GetArrayFromImage(data[key]["image"])
        return data


class ChangeDimensionOrder(MapTransform):
    def __init__(self, keys, source_order: Tuple, target_order: Tuple):
        super().__init__(keys)
        self.source_order = source_order
        self.target_order = target_order

    def __call__(self, data):
        for key in self.keys:
            data[key]["image"] = np.moveaxis(data[key]["image"], self.source_order, self.target_order)
        return data


class ConditionalFlipd(Flipd):
    def __init__(self, keys):
        super().__init__(keys, 0)

    def __call__(self, data):
        for key in self.keys:
            if data["meta"]["mezSeite"] == 1.0:
                flipper = Flip(spatial_axis=1)
                data[key] = flipper(data[key])
        return data


class KeepKeyd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        data_keep = dict().fromkeys(self.keys)
        for key in self.keys:
            data_keep[key] = data[key]
        return data_keep


class SubtractAndDivide(MapTransform):
    def __init__(self, keys, subtrahend: float, divisor: float):
        super().__init__(keys)
        self.subtrahend = subtrahend
        self.divisor = divisor

    def __call__(self, data):

        for key in self.keys:
            data[key] = (data[key] - self.subtrahend) / self.divisor
        return data
