from monai.data import DataLoader


class ArthritisDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         pin_memory=True)
