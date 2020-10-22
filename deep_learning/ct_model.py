from copy import deepcopy
import sys
import os
import json

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import auroc
from segmentation_models_pytorch.unet.model import Unet
from torch.utils.data import DataLoader
import torchvision

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_management.ct.VolumeIO import ScancoLoader
from data_management.DataLoader import ArthritisDataLoader
from data_management.Dataset import get_persistent_dataset
from data_management.ct.Database import MCPDatabase
from data_management.ct.Processing import build_processing_pipeline
from deep_learning.Loss import dice_loss_functional
from deep_learning.Metrics import accuracy, iou
from deep_learning.UNet3D.IsenseeUnet import Modified3DUNet


class DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.batch_size = hparams["batch_size"]
        self.dataset = None
        self.split = [0.7, 0.2, 0.1]
        self.db_length = 0
        self.hparams = hparams
        self.transformations = self.get_transformations()
        self.transformation_hparams = json.loads(
            json.dumps(self.transformations, default=lambda o: '<not serializable>'))
        self.preprocessing_pipeline = build_processing_pipeline(self.transformations)

    def setup(self, stage=None):
        db = MCPDatabase("C:/ExcelDB.xlsx", "C:/ISQs", "C:/SEG_AIM_FROM_GOBJs", "C:/HEADER_FROM_GOBJ")

        self.dataset = get_persistent_dataset(db, transforms=self.preprocessing_pipeline)
        self.db_length = len(self.dataset)

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
            {"Resized": {"keys": ["volume"],
                         "spatial_size": (self.hparams["x_y_dim"], self.hparams["x_y_dim"], self.hparams["z_dim"]),
                         "mode": "trilinear", "align_corners": False}},
            {"Resized": {"keys": ["segmentation"],
                         "spatial_size": (self.hparams["x_y_dim"], self.hparams["x_y_dim"], self.hparams["z_dim"]),
                         "mode": "nearest"}},
            {"Rotate90d": {"keys": ["segmentation"], "k": 3}},
            {"ToNumpyd": {"keys": ["segmentation"]}},
            {"ConditionalFlipd": {"keys": ["volume", "segmentation"]}},
            {"SubtractAndDivide": {"keys": ["volume"],
                                   "subtrahend": self.hparams["global_mean"],
                                   "divisor": self.hparams["global_var"]}},
            {"KeepKeyd": {"keys": ["volume", "segmentation"]}},
            {"ToTensord": {"keys": ["volume", "segmentation"]}},
        ]
        return transform_dict

    def train_dataloader(self) -> DataLoader:
        train_dataset = deepcopy(self.dataset)
        train_dataset.data = train_dataset.data[0:int(self.db_length * self.split[0])]
        return ArthritisDataLoader(train_dataset, shuffle=True, batch_size=self.batch_size,
                                   num_workers=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        val_dataset = deepcopy(self.dataset)
        start_ind = int(self.db_length * self.split[0])
        end_ind = start_ind + int(self.db_length * self.split[1])
        val_dataset.data = val_dataset.data[start_ind:end_ind]
        return ArthritisDataLoader(val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        test_dataset = deepcopy(self.dataset)
        start_ind = int(self.db_length * self.split[2])
        test_dataset.data = test_dataset.data[-start_ind:]
        return ArthritisDataLoader(test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, hparams, pretrained=True):
        super().__init__()
        self.hparams = hparams
        encoder_weights = None
        if pretrained:
            encoder_weights = "imagenet"
        self.model = Unet(in_channels=1, activation="sigmoid", encoder_weights=encoder_weights)
        self.image_logging = True

    def forward(self, volume):
        return self.model(volume)

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                lr=self.hparams["lr"])

    def log_image(self, x, y, y_hat):
        def norm(a):
            return (a - a.min()) / (a.max() - a.min())

        x = norm(x)
        grid = torchvision.utils.make_grid([x, y, y_hat])
        self.logger.experiment.add_image("val/image", grid, self.global_step)

    def prepare_input_data(self, input_data):
        return input_data

    def prepare_prediction(self, prediction, label):
        return prediction

    def predict(self, input, slice_ind: int):
        if slice_ind != -1:
            y_hat = self(input[..., slice_ind])
        else:
            y_hat = torch.zeros_like(input, device=self.device)
            slices = input.shape[-1]
            for i in range(slices):
                y_hat[..., i] = self(input[..., i])
        return y_hat

    def training_step(self, batch, batch_idx):
        x = batch["volume"]
        y = batch["segmentation"]
        # U-Net is only 2D -> Slice the volume
        slices = x.shape[-1]
        slice_ind = np.random.randint(0, slices)
        y_hat = self.predict(x, slice_ind)
        loss = dice_loss_functional(y_hat, y[..., slice_ind])
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x = batch["volume"]
        y = batch["segmentation"]

        loss = torch.tensor([0.0], device=self.device, requires_grad=False)
        auc = torch.tensor([0.0], device=self.device, requires_grad=False)
        accuracy_metric = torch.tensor([0.0], device=self.device, requires_grad=False)
        iou_metric = torch.tensor([0.0], device=self.device, requires_grad=False)

        x = self.prepare_input_data(x)
        y_hat = self.predict(x, -1)
        y_hat = self.prepare_prediction(y_hat, y)
        slices = y_hat.shape[-1]
        for i in range(slices):
            loss += dice_loss_functional(y_hat[..., i], y[..., i])
            try:
                auc += auroc(y_hat[..., i].flatten(), y[..., i].flatten())
            except ValueError:
                print("WARNING: AUC validation value not reliable.")
            accuracy_metric += accuracy(y_hat[..., i], y[..., i])
            iou_metric += iou(y_hat[..., i], y[..., i])
            if batch_idx == 0 and i == int(slices / 2) and self.image_logging:
                self.log_image(x[0, ..., i], y[0, ..., i], y_hat[..., i][0])
        loss /= slices
        auc /= slices
        accuracy_metric /= slices
        iou_metric /= slices

        return {'val_loss': loss, "val_auc": auc, "val_accuracy": accuracy_metric, "val_iou": iou_metric}

    def test_step(self, batch, batch_idx, return_prediction=False):
        x = batch["volume"]
        y = batch["segmentation"]

        loss = torch.tensor([0.0], device=self.device, requires_grad=False)
        auc = torch.tensor([0.0], device=self.device, requires_grad=False)
        accuracy_metric = torch.tensor([0.0], device=self.device, requires_grad=False)
        iou_metric = torch.tensor([0.0], device=self.device, requires_grad=False)

        x = self.prepare_input_data(x)
        y_hat = self.predict(x, -1)

        y_hat = self.prepare_prediction(y_hat, y)
        slices = y_hat.shape[-1]
        valid_auc_slices = 0
        for i in range(slices):
            loss += dice_loss_functional(y_hat[..., i], y[..., i])
            try:
                auc += auroc(y_hat[..., i].flatten(), y[..., i].flatten())
                valid_auc_slices += 1
            except ValueError:
                print("WARNING: AUC validation value not reliable.")
            accuracy_metric += accuracy(y_hat[..., i], y[..., i])
            iou_metric += iou(y_hat[..., i], y[..., i])

        loss /= slices
        auc /= valid_auc_slices
        accuracy_metric /= slices
        iou_metric /= slices
        if not return_prediction:
            return {'test_loss': loss, "test_auc": auc, "test_accuracy": accuracy_metric, "test_iou": iou_metric}
        else:
            return y_hat

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'loss': avg_loss}
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_auc = torch.stack([x['val_auc'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        avg_iou = torch.stack([x['val_iou'] for x in outputs]).mean()
        tensorboard_logs = {'val/loss': avg_loss, "val/auc": avg_auc, "val/accuracy": avg_accuracy,
                            "val/iou": avg_iou}
        return {'val_loss': avg_loss, "val_auc": avg_auc, "val_accuracy": avg_accuracy, "val_iou": avg_iou,
                'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_auc = torch.stack([x['test_auc'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        avg_iou = torch.stack([x['test_iou'] for x in outputs]).mean()
        tensorboard_logs = {'test/loss': avg_loss, "test/auc": avg_auc, "test/accuracy": avg_accuracy,
                            "test/iou": avg_iou}
        return {'test_loss': avg_loss, "test_auc": avg_auc, "test_accuracy": avg_accuracy, "test_iou": avg_iou,
                'log': tensorboard_logs}


class Model3D(Model):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams)
        self.model = Modified3DUNet(1, 1, base_n_filter=2)
        self.image_logging = False

    def training_step(self, batch, batch_idx):
        x = batch["volume"]
        y = batch["segmentation"]

        y_hat = self.predict(x, -1)

        loss = dice_loss_functional(y_hat, y)
        return {'loss': loss}

    def predict(self, input, slice_ind):
        if slice_ind == -1:
            return self(input)
        else:
            raise NotImplemented()


if __name__ == "__main__":
    for random_seed in [42, 43, 44]:
        pl.seed_everything(random_seed)
        hparams = {"x_y_dim": 512, "z_dim": 80, "batch_size": 1, "global_mean": 562, "global_var": 1073,
                   "lr": 1e-4}
        ct_dataset = DataModule(hparams)
        hparams["transformation_hparams"] = ct_dataset.transformation_hparams
        model = Model3D(hparams)
        trainer = pl.Trainer(gpus=-1, deterministic=True, default_root_dir=os.path.dirname(__file__),
                             num_sanity_val_steps=-1, max_epochs=250)
        trainer.fit(model, ct_dataset)
        del trainer, model
