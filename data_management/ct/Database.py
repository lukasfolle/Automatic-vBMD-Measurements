import os

import pandas as pd

from arthritis_utils.General import path_exists_check
from data_management.Database import AbstractDatabase
from data_management.ct.scanco_script.parse_header_file import get_df_from_header_folder


class MCPDatabase(AbstractDatabase):
    def __init__(self, excel_db_path: str, volumes_path: str, segmentation_path: str, header_folder_path: str):
        self.excel_db_path = excel_db_path
        self.volumes_path = volumes_path
        self.segmentation_path = segmentation_path

        self._header_df = get_df_from_header_folder(header_folder_path)
        self._remove_missing_files()

        self.db = self._load_excel_db_path()
        self._merge_dbs()

    def _load_excel_db_path(self):
        path_exists_check(self.excel_db_path)
        return pd.read_excel(self.excel_db_path, dtype={"patGeschlecht": "category", "patHdg": "category",
                                                        "mezSeite": "category", "motMotiongradeA": "category",
                                                        "motMotiongradeB": "category"})

    def _remove_missing_files(self):

        def _check_file_for_existence(row, path, ending):
            if ending in "_SEG.AIM":
                ending += ";" + str(row["version"])
            if os.path.exists(full_path := os.path.join(path, row["file_name"] + ending)):
                return pd.Series([True, full_path])
            return pd.Series([False, "None"])

        self._header_df[["found_seg_aim", "seg_aim_path"]] = self._header_df.apply(_check_file_for_existence,
                                                                                   args=(
                                                                                       self.segmentation_path,
                                                                                       "_SEG.AIM",),
                                                                                   axis=1)
        self._header_df[["found_isq", "isq_path"]] = self._header_df.apply(_check_file_for_existence,
                                                                           args=(self.volumes_path, ".ISQ",), axis=1)
        print(f"INFO: Could find {self._header_df['found_seg_aim'].sum()} SEG.AIMs.")
        print(f"INFO: Could find {self._header_df['found_isq'].sum()} ISQs.")
        self._header_df = self._header_df[self._header_df["found_seg_aim"]]
        self._header_df = self._header_df[self._header_df["found_isq"]]

    def _merge_dbs(self):

        def _get_dicom_num_from_filename(row):
            return int(row["file_name"][1:])

        self._header_df["mezDicom"] = self._header_df.apply(_get_dicom_num_from_filename, axis=1)
        self.db = self.db.merge(self._header_df, how="inner", on="mezDicom")

    def get_db_monai_format(self):
        self.db["mezDatum"] = self.db["mezDatum"].astype(str)
        monai_rows = self.db.iterrows()

        monai_db = [{"volume": monai_row[1].to_dict()["isq_path"],
                     "segmentation": monai_row[1].to_dict()["seg_aim_path"],
                     "label": monai_row[1].to_dict()["patHdg"],
                     "meta": monai_row[1].to_dict()}
                    for monai_row in monai_rows]
        return monai_db
