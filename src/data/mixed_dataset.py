import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MixedDataset(Dataset):
    """
    A Dataset wrapper that samples from multiple underlying datasets
    according to specified probability weights.
    """
    def __init__(self, datasets: dict[str, Dataset], weights: dict[str, float]):
        self.dataset_names = list(datasets.keys())
        self.datasets = list(datasets.values())
        
        # Calculate lengths
        self.lengths = [len(ds) for ds in self.datasets]
        # Total virtual length (sum of all datasets)
        self.total_len = sum(self.lengths)
        
        # Normalize weights
        raw_weights = [weights.get(name, 0.0) for name in self.dataset_names]
        total_weight = sum(raw_weights)
        if total_weight <= 0:
            raise ValueError("Sum of dataset weights must be > 0")
        self.probs = [w / total_weight for w in raw_weights]
        
    def __len__(self):
        return self.total_len
        
    def __getitem__(self, idx):
        # 1. Sample which dataset to pull from based on probabilities
        ds_idx = np.random.choice(len(self.datasets), p=self.probs)
        dataset_name = self.dataset_names[ds_idx]
        dataset = self.datasets[ds_idx]
        
        # 2. Sample a random index from that specific dataset
        # We ignore the passed `idx` because it doesn't map cleanly to proportional sampling.
        # This means __getitem__ is stochastic, which is perfectly fine for training.
        # (For validation, we shouldn't use MixedDataset anyway, we evaluate them separately).
        sample_idx = np.random.randint(0, len(dataset))
        
        sample = dataset[sample_idx]
        
        # 3. Add origin tag
        # PyTorch dataloaders collate function doesn't like strings, so we can
        # assign an integer ID for the dataset origin if needed, or just let it be.
        # We'll assign a dataset_id to be safe.
        sample['dataset_id'] = torch.tensor(ds_idx, dtype=torch.long)
        
        return sample

def get_mixed_loader(
    data_roots: dict[str, str],
    batch_size: int,
    split: str = 'train',
    num_workers: int = 4,
    input_size: int = 384,
    cache_max_items: int = 512,
    mix_weights: dict[str, float] = None,
    **kwargs
) -> DataLoader:
    """
    Returns a DataLoader that dynamically samples from Hypersim and MIDIntrinsics
    (and potentially others in the future) according to mix_weights.
    """
    if split != 'train':
        raise ValueError("get_mixed_loader should only be used for training.")
        
    if mix_weights is None:
        mix_weights = {'hypersim': 0.5, 'midintrinsic': 0.5}

    from src.data.hypersim_dataset import HypersimDataset
    from src.data.midintrinsic_dataset import MIDIntrinsicDataset
    
    datasets = {}
    
    if 'hypersim' in mix_weights and mix_weights['hypersim'] > 0:
        datasets['hypersim'] = HypersimDataset(
            root_dir=data_roots.get('hypersim', '../../../datasets/hypersim'),
            split='train',
            input_size=input_size,
            cache_max_items=cache_max_items,
            crop_mode_train='hybrid',
            augment_train=True,
            split_file=kwargs.get('split_file', 'hypersim_split.json'),
            split_seed=kwargs.get('split_seed', 42),
            split_ratio=kwargs.get('split_ratio', 0.9),
            strict_split=kwargs.get('strict_split', True),
            max_hdf5_retries=kwargs.get('max_hdf5_retries', 1),
            skip_corrupt_samples=kwargs.get('skip_corrupt_samples', True),
            load_geometry=kwargs.get('load_geometry', True),
            load_normals=kwargs.get('load_normals', False),
            color_pair_prob=kwargs.get('hypersim_color_pair_prob', 0.0),
            color_tint_min=kwargs.get('hypersim_color_tint_min', 0.8),
            color_tint_max=kwargs.get('hypersim_color_tint_max', 1.25),
        )
        
    if 'midintrinsic' in mix_weights and mix_weights['midintrinsic'] > 0:
        datasets['midintrinsic'] = MIDIntrinsicDataset(
            root_dir=data_roots.get('midintrinsic', '../../../datasets/MIDIntrinsics'),
            split='train',
            input_size=input_size,
            crop_mode_train='hybrid',
            use_paired=kwargs.get('use_mid_paired', False),
            pair_mode=kwargs.get('mid_pair_mode', 'raw'),
            chromatic_aug=kwargs.get('mid_chromatic_aug', False),
            raw_color_pair=kwargs.get('mid_raw_color_pair', False),
        )
        
    if 'interiorverse' in mix_weights and mix_weights['interiorverse'] > 0:
        from src.data.interiorverse_dataset import InteriorVerseDataset
        datasets['interiorverse'] = InteriorVerseDataset(
            root_dir=data_roots.get('interiorverse', '../../../datasets/InteriorVerse'),
            split='train',
            input_size=input_size,
            crop_mode_train='hybrid',
        )

    if 'openrooms' in mix_weights and mix_weights['openrooms'] > 0:
        from src.data.openrooms_dataset import OpenRoomsDataset
        datasets['openrooms'] = OpenRoomsDataset(
            root_dir=data_roots.get('openrooms', '../../../datasets/OpenRooms'),
            split='train',
            input_size=input_size,
            crop_mode_train='hybrid',
        )

    if 'front3d' in mix_weights and mix_weights['front3d'] > 0:
        from src.data.front3d_dataset import Front3DDataset
        datasets['front3d'] = Front3DDataset(
            root_dir=data_roots.get('front3d', '../../../datasets/front3d_iid'),
            split='train',
            input_size=input_size,
            crop_mode_train='hybrid',
            cache_max_items=kwargs.get('front3d_cache_max_items', 0),
        )

    mixed_dataset = MixedDataset(datasets, mix_weights)
    
    return DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle is True, though __getitem__ is already stochastic
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
