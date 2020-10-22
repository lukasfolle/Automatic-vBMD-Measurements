from monai.transforms import MapTransform
from typing import Callable
from monai.config import KeysCollection


class LoadDatad(MapTransform):
    """
    Base class for dictionary-based wrapper of IO loader transforms.
    It must load image and metadata together. If loading a list of files in one key,
    stack them together and add a new dimension as the first dimension, and use the
    meta data of the first image to represent the stacked result. Note that the affine
    transform of all the stacked images should be same. The output metadata field will
    be created as ``key_{meta_key_postfix}``.
    """

    def __init__(
            self,
            keys: KeysCollection,
            loader: Callable,
            meta_key_postfix: str = "meta_dict",
            overwriting: bool = False
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            loader: callable function to load data from expected source.
                typically, it's array level transform, for example: `LoadNifti`,
                `LoadPNG` and `LoadNumpy`, etc.
            meta_key_postfix: use `key_{postfix}` to store the metadata of the loaded data,
                default is `meta_dict`. The meta data is a dictionary object.
                For example, load Nifti file for `image`, store the metadata into `image_meta_dict`.
            overwriting: whether allow to overwrite existing meta data of same key.
                default is False, which will raise exception if encountering existing key.
        Raises:
            TypeError: When ``loader`` is not ``callable``.
            TypeError: When ``meta_key_postfix`` is not a ``str``.
        """
        super().__init__(keys)
        if not callable(loader):
            raise TypeError(f"loader must be callable but is {type(loader).__name__}.")
        self.loader = loader
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_key_postfix = meta_key_postfix
        self.overwriting = overwriting

    def __call__(self, data):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.
        """
        d = dict(data)
        for key in self.keys:
            data = self.loader(d[key])
            assert isinstance(data, (tuple, list)), "loader must return a tuple or list."
            d[key] = data[0]
            assert isinstance(data[1], dict), "metadata must be a dict."
            key_to_add = f"{key}_{self.meta_key_postfix}"
            if key_to_add in d and not self.overwriting:
                raise KeyError(f"Meta data with key {key_to_add} already exists and overwriting=False.")
            d[key_to_add] = data[1]
        return d
