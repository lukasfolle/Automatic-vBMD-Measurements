import os
import tempfile
from monai.data import PersistentDataset

from data_management.Database import AbstractDatabase


def get_persistent_dataset(db: AbstractDatabase, transforms, cache: bool = True, cache_folder: str = "MONAI_no_contour"):
    db = db.get_db_monai_format()
    cache_location = None
    if cache:
        cache_location = os.path.join(tempfile.gettempdir(), cache_folder)
        if not os.path.exists(cache_location):
            os.mkdir(cache_location)
        print(f"INFO: Dataset will be cached in {cache_location}")
    persistent_ds = PersistentDataset(db, transform=transforms, cache_dir=cache_location)
    return persistent_ds
