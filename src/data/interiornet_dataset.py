"""Placeholder InteriorNet dataset loader."""

from torch.utils.data import Dataset


class InteriorNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, input_size=384):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.input_size = input_size
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("InteriorNetDataset loading not implemented yet.")

