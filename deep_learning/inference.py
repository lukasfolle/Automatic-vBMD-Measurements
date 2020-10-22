import argparse
import os

import torch

from arthritis_utils.General import path_exists_check
from data_management.ct.Processing import build_processing_pipeline
from data_management.ct.VolumeIO import ScancoLoader
from data_management.ct.VolumeIO import prepare_info_dict, write_volume
from data_management.ct.scanco_script.parse_header_file import parse_header
from deep_learning.validate_full_res import FullResolutionDataModule
from deep_learning.validate_full_res import FullResolutionModel
from deep_learning.visualize_model_predictions import get_latest_model


class Inference:
    def __init__(self):
        self.model = self.load_model()
        self.hparams = self.model.hparams
        self.scanco_loader = ScancoLoader()
        self.preprocessing_pipeline = self.get_preprocessing_pipeline()
        self.postprocessing_pipeline = self.get_postprocessing_pipeline()

    def read_image(self, image_path: str):
        images, dims = self.scanco_loader(image_path)
        return images

    def get_preprocessing_pipeline(self):
        preprocessing_transforms = FullResolutionDataModule(self.hparams).get_transformations()
        transforms = preprocessing_transforms[2:]  # Skip load and to_numpy
        drop_transforms = ["SegmentationAlignd"]
        transforms = [transform for transform in transforms if list(transform.keys())[0] not in drop_transforms]
        preprocessing_pipeline = build_processing_pipeline(transforms)
        return preprocessing_pipeline

    def apply_preprocessing_pipeline(self, images, mez_side, header_path):
        header_dict = parse_header(header_path)
        monai_format = {"volume": images, "segmentation": images, "meta": dict({"mezSeite": mez_side}, **header_dict)}  # Ignore segmentation
        monai_format = self.preprocessing_pipeline(monai_format)
        images = monai_format["volume"]
        images = images.squeeze()
        return images

    @staticmethod
    def get_postprocessing_pipeline():
        transforms = [
            {"AddChanneld": {"keys": ["prediction"]}},
            # Revert spatial changes applied in preprocessing
            {"Rotate90d": {"keys": ["prediction"], "k": 1}},
            {"ConditionalFlipd": {"keys": ["prediction"]}},
            {"SqueezeDimd": {"keys": ["prediction"], "dim": 0}},
            {"ExtendToValidSlicesd": {"keys": ["prediction"]}}
        ]
        postprocessing_pipeline = build_processing_pipeline(transforms)
        return postprocessing_pipeline

    def apply_postprocessing_pipeline(self, images, mez_side, header_path, original_image_dim):
        header_dict = parse_header(header_path)
        monai_format = {"prediction": images, "meta": dict({"mezSeite": mez_side,
                                                            "original_image_dim": original_image_dim}, **header_dict)}
        monai_format = self.postprocessing_pipeline(monai_format)
        images = monai_format["prediction"]
        return images

    @staticmethod
    def load_model():
        phase = "inference"
        checkpoint_path = os.path.join(os.path.dirname(__file__), "lightning_logs", "version_4", "checkpoints")
        best_model = FullResolutionModel.load_from_checkpoint(
            os.path.join(checkpoint_path, get_latest_model(checkpoint_path)), phase=phase, save_dir="")
        best_model.freeze()
        best_model = best_model.cuda()
        return best_model

    def infer_model(self, images: torch.tensor, threshold):
        if len(images.shape) > 3:
            raise ValueError(f"Input volume should have 3 dimensions but got {len(images.shape)}.")
        labels = torch.zeros_like(images)
        images = images.view((1, 1, *images.shape))
        images = self.model.prepare_input_data(images)
        images = images.cuda()
        predictions = torch.zeros_like(images, device="cuda:0")
        for slice in range(images.shape[-1]):
            predictions[..., slice] = self.model(images[..., slice])
        predictions = self.model.prepare_prediction(predictions, labels)
        predictions = predictions.detach().cpu().numpy()
        predictions = (predictions > threshold) * 1000
        return predictions

    def get_target_path(self, image_path):
        target_path = os.path.dirname(image_path)
        file_name = os.path.basename(image_path)
        base_name, extension = os.path.splitext(file_name)
        target_path = os.path.join(target_path, base_name + "_PRED" + ".AIM")
        return target_path

    def write(self, predictions, target_path):
        print(f"INFO: Writing prediction to {target_path}.")
        info_dict = prepare_info_dict(predictions, target_path)
        write_volume(predictions, info_dict)

    def infer(self, image_paths, header_paths, mez_sides, threshold):
        for image_path, header_path, mez_side in zip(image_paths, header_paths, mez_sides):
            path_exists_check(image_path)
            path_exists_check(header_path)
            images = self.read_image(image_path)
            original_image_dim = images.shape
            images = self.apply_preprocessing_pipeline(images, mez_side, header_path).squeeze()
            predictions = self.infer_model(images, threshold)
            predictions = self.apply_postprocessing_pipeline(predictions, mez_side, header_path, original_image_dim)
            target_path = self.get_target_path(image_path)
            self.write(predictions, target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference of the trained model.\n'
                                                 'Predictions will be save at the same location as the input images.\n'
                                                 'For a patient the z-dimensions need to be specified by annotating'
                                                 ' something in the first and last slice of interest. Then the header '
                                                 'of that file has to be stored.')
    parser.add_argument('--image_paths', type=str, nargs='+',
                        help='One or more volumes to run the models inference.')
    parser.add_argument('--header_paths', type=str, nargs='+',
                        help='Path(s) to the header files of seg aims corresponding to the image_paths.\n'
                             'Header files are created in ipl using write_part header.')
    parser.add_argument('--mez_seite', type=int, nargs='+',
                        help='Side(s) of the patient hand corresponding to the image_paths.')
    parser.add_argument('--threshold', type=float,
                        help='Threshold for binarization of network predictions. In range 0 to 1.')

    args = parser.parse_args()
    image_paths_arg = args.image_paths
    mez_seite_arg = args.mez_seite
    header_paths_arg = args.header_paths
    threshold_arg = args.threshold
    if not (len(mez_seite_arg) == len(image_paths_arg) and len(image_paths_arg) == len(header_paths_arg)):
        raise ValueError(f"Length of arg image_path image_paths_arg, and mez_seite has to be identical but "
                         f"got ({len(image_paths_arg)}) ({len(header_paths_arg)}), and ({len(mez_seite_arg)}).")
    inference = Inference()
    inference.infer(image_paths_arg, header_paths_arg, mez_seite_arg, threshold_arg)
