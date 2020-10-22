import os

import numpy as np
import pytorch_lightning as pl
import torch
import json
from monai.transforms.spatial.array import Resize

from data_management.Dataset import get_persistent_dataset
from data_management.ct.Database import MCPDatabase
from data_management.ct.VolumeIO import ScancoLoader
from deep_learning.ct_model import DataModule, Model, Model3D
from deep_learning.visualize_model_predictions import get_latest_model


class FullResolutionDataModule(DataModule):
    def get_transformations(self):
        transform_dict = [
            {"LoadDatad": {"keys": ["volume", "segmentation"], "loader": ScancoLoader()}},
            {"ToNumpyd": {"keys": ["volume", "segmentation"]}},
            {"CastToTyped": {"keys": ["volume"], "dtype": np.float32}},
            {"CastToTyped": {"keys": ["segmentation"], "dtype": np.bool}},
            {"AddChanneld": {"keys": ["volume", "segmentation"]}},
            {"ToTensord": {"keys": ["volume", "segmentation"]}},
            {"SegmentationAlignd": {"keys": ["volume", "segmentation"]}},
            {"CutToValidSlicesd": {"keys": ["volume", "segmentation"]}},
            {"ToNumpyd": {"keys": ["segmentation"]}},  # NEW due to removing of Resized
            {"Rotate90d": {"keys": ["segmentation"], "k": 3}},
            {"ToNumpyd": {"keys": ["segmentation", "volume"]}},
            {"ConditionalFlipd": {"keys": ["volume", "segmentation"]}},
            {"SubtractAndDivide": {"keys": ["volume"],
                                   "subtrahend": self.hparams["global_mean"],
                                   "divisor": self.hparams["global_var"]}},
            {"KeepKeyd": {"keys": ["volume", "segmentation"]}},
            {"ToTensord": {"keys": ["volume", "segmentation"]}},
        ]
        return transform_dict

    def setup(self, stage=None):
        db = MCPDatabase("C:/ExcelDB.xlsx", "C:/ISQs", "C:/SEG_AIM_FROM_GOBJs", "C:/HEADER_FROM_GOBJ")

        self.dataset = get_persistent_dataset(db, transforms=self.preprocessing_pipeline, cache=True,
                                              cache_folder="MONAI_Full_Res")
        self.db_length = len(self.dataset)


class FullResolutionModel(Model):
    def __init__(self, hparams, phase: str, save_dir: str, **kwargs):
        super().__init__(hparams)
        self.downsize = Resize(spatial_size=(self.hparams["x_y_dim"], self.hparams["x_y_dim"], self.hparams["z_dim"]),
                               mode="trilinear", align_corners=False)
        self.image_logging = False
        self.phase = phase
        self.save_dir = save_dir
        self.test_device = self.device

    def prepare_input_data(self, input_data):
        input_data = input_data.detach().cpu()
        input_data = self.downsize(input_data.squeeze(dim=0))
        input_data = input_data.reshape((1, *input_data.shape))
        input_data = torch.from_numpy(input_data).to(self.device)
        return input_data

    def prepare_prediction(self, prediction, label):
        upsize = Resize(spatial_size=label.squeeze().shape,
                        mode="trilinear", align_corners=False)
        prediction = prediction.detach().cpu()
        prediction = upsize(prediction.squeeze(dim=0)).reshape(label.shape)
        prediction = torch.from_numpy(prediction).to(self.test_device)
        return prediction

    def test_step(self, batch, batch_idx, return_prediction=False):
        if self.phase == "val":
            return self.validation_step(batch, batch_idx)
        elif self.phase == "test":
            return super().test_step(batch, batch_idx, return_prediction)

    def test_epoch_end(self, outputs):
        if self.phase == "test":
            results = super().test_epoch_end(outputs)
        elif self.phase == "val":
            results = self.validation_epoch_end(outputs)
        print(results)
        path = os.path.join(self.save_dir, f"full_res_{self.phase}_result.json")
        with open(path, "w") as f:
            json.dump(str(results), f)
        return results


class FullResolutionModel3D(FullResolutionModel, Model3D):
    def __init__(self, hparams, phase, save_dir):
        super().__init__(hparams=hparams, phase=phase, save_dir=save_dir)


def validate(phase, checkpoint_path, model_dim):
    network_version = "version_" + checkpoint_path.split("version_")[1].split("\\")[0]
    save_dir = os.path.join(os.path.dirname(__file__), "test_runs", network_version)

    if model_dim == 2:
        best_model = FullResolutionModel.load_from_checkpoint(
            os.path.join(checkpoint_path, get_latest_model(checkpoint_path)), phase=phase, save_dir=save_dir)
    elif model_dim == 3:
        best_model = FullResolutionModel3D.load_from_checkpoint(
            os.path.join(checkpoint_path, get_latest_model(checkpoint_path)), phase=phase, save_dir=save_dir)
    else:
        raise NotImplemented()

    hparams = {"x_y_dim": 512, "z_dim": 80, "batch_size": 1, "global_mean": 562, "global_var": 1073,
               "lr": 1e-5}
    ct_dataset_full_resolution = FullResolutionDataModule(hparams)
    ct_dataset_full_resolution.setup(phase)

    trainer = pl.Trainer(gpus=-1, deterministic=True,
                         default_root_dir=save_dir,
                         num_sanity_val_steps=-1, checkpoint_callback=False, max_epochs=0)

    if phase == "val":
        dataloader = ct_dataset_full_resolution.val_dataloader()
    elif phase == "test":
        dataloader = ct_dataset_full_resolution.test_dataloader()
    else:
        raise NotImplemented()
    trainer.test(best_model, dataloader)


if __name__ == "__main__":
    model_dim = 2
    version = "version_16"
    current_phase = "test"
    checkpoint_path = os.path.join(os.path.dirname(__file__), "lightning_logs", version, "checkpoints")
    validate(current_phase, checkpoint_path, model_dim)
