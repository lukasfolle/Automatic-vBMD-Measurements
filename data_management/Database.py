import abc
from typing import List, Dict


class AbstractDatabase(abc.ABC):
    @abc.abstractmethod
    def get_db_monai_format(self) -> List[Dict]:
        """
        Should yield a list of dicts with similar keys
                [{"volume": ..., "segmentation": ..., "label": ..., "meta": ...}, ...]
        """

        pass
