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
    seed: int | None = None,
    strict_split: bool = True,
    use_mid_paired: bool = False,
    mid_raw_color_pair: bool = False,
) -> DataLoader:
    """Training loader that samples each item from one of the datasets by mix_weights.

    Datasets: hypersim, midintrinsic, interiorverse. Only those with a positive weight are
    instantiated. MID yields cross-illumination pairs when use_mid_paired is True. The
    trifactor model adds 3D-Front-IID v2 through src/data/trifactor_dataset.py.
    """
    if split != 'train':
        raise ValueError("get_mixed_loader should only be used for training.")
    if mix_weights is None:
        mix_weights = {'hypersim': 0.5, 'midintrinsic': 0.5}
    unknown = [k for k, w in mix_weights.items()
               if w > 0 and k not in ('hypersim', 'midintrinsic', 'interiorverse')]
    if unknown:
        raise ValueError(f'unknown datasets in mix_weights: {unknown}')

    from src.data.hypersim_dataset import HypersimDataset
    from src.data.midintrinsic_dataset import MIDIntrinsicDataset

    datasets = {}

    if mix_weights.get('hypersim', 0) > 0:
        datasets['hypersim'] = HypersimDataset(
            root_dir=data_roots.get('hypersim', '../datasets/hypersim'),
            split='train',
            input_size=input_size,
            cache_max_items=cache_max_items,
            crop_mode_train='hybrid',
            augment_train=True,
            strict_split=strict_split,
            load_geometry=False,
            load_normals=False,
        )

    if mix_weights.get('midintrinsic', 0) > 0:
        datasets['midintrinsic'] = MIDIntrinsicDataset(
            root_dir=data_roots.get('midintrinsic', '../datasets/MIDIntrinsics'),
            split='train',
            input_size=input_size,
            crop_mode_train='hybrid',
            use_paired=use_mid_paired,
            pair_mode='raw',
            raw_color_pair=mid_raw_color_pair,
        )

    if mix_weights.get('interiorverse', 0) > 0:
        from src.data.interiorverse_dataset import InteriorVerseDataset
        datasets['interiorverse'] = InteriorVerseDataset(
            root_dir=data_roots.get('interiorverse', '../datasets/InteriorVerse'),
            split='train',
            input_size=input_size,
            crop_mode_train='hybrid',
        )

    mixed_dataset = MixedDataset(datasets, mix_weights)

    # SEEDING. __getitem__ is stochastic (crop, augmentation, MID pair draw) and uses
    # numpy's GLOBAL RNG inside each worker process. Without a worker_init_fn, numpy in
    # a worker is seeded from OS entropy at fork, so the sampling stream was not
    # reproducible even when torch was seeded -- and torch was never seeded either, so
    # `seed: 42` / `deterministic: true` in the configs were dead settings describing a
    # guarantee that did not exist.
    #
    # Passing seed=None preserves the old behaviour exactly (unseeded, every run
    # different), so this change cannot silently alter any run that does not ask for a
    # seed. With a seed, each worker gets a DISTINCT but DERIVED stream (seed + worker
    # id), which is what makes replicates both independent within a run and
    # reproducible across runs.
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

        def worker_init_fn(worker_id, _seed=int(seed)):
            import numpy as _np
            import random as _random
            s = (_seed + worker_id) % (2 ** 31 - 1)
            _np.random.seed(s)
            _random.seed(s)
            torch.manual_seed(s)

    return DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle is True, though __getitem__ is already stochastic
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
