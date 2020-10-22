import os
import numpy as np
from monai.transforms.compose import Transform

from include.ClinicModules.FileImportScancoAIM import import_aim, write_header_aim
from include.ClinicModules.FileImportScancoISQ import import_isq
from arthritis_utils.General import path_exists_check


def load_volume(file_path):
    path_exists_check(file_path)
    filename, extension = os.path.splitext(file_path)
    if ".AIM" in extension:
        return import_aim(file_path)
    elif ".ISQ" in extension:
        return import_isq(file_path)
    else:
        raise NotImplementedError(f"No support for {extension}.")


def write_volume(volume: np.ndarray, info: dict):

    def prepare_volume(volume):
        volume = np.moveaxis(volume, [2, 1, 0], [0, 1, 2])
        volume = volume.reshape((info["dim"][0], info["dim"][1], info["dim"][2]))
        volume = volume.astype("int16")
        array_bytes = np.ndarray.tobytes(volume)
        return array_bytes

    with open(info["file_name"], "wb") as f:
        raw_data_offset = write_header_aim(f, info)
        f.seek(raw_data_offset)
        f.write(prepare_volume(volume))
    print(f"Write to {info['file_name']} complete.")


def prepare_info_dict(volume: np.ndarray, aim_save_location):
    info = {"file_name": aim_save_location,
            "length_data": len(np.ndarray.tobytes(volume.astype("int16"))),
            "position": (0, 0, 0),
            "dim": volume.shape,
            "offset": (0, 0, 0),
            "supdim": (0, 0, 0),
            "suppos": (0, 0, 0),
            "subdim": (0, 0, 0),
            "testoff": (0, 0, 0),
            "el_size_mm": (4020125351, 4020125351, 4020125351),
            }
    return info


class ScancoLoader(Transform):
    def __call__(self, filename):
        file_info, volume = load_volume(filename)
        file_info = {"dimensions": file_info}
        return volume, file_info


if __name__ == "__main__":
    volume = np.zeros((512, 512, 40))
    volume[170:340, 120:340, 10:30] = 1000.0
    info = prepare_info_dict(volume, "test.AIM")
    write_volume(volume, info)
