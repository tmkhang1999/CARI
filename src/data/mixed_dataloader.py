"""Weighted mixed dataloader with homogeneous per-dataset batches."""

import numpy as np
from torch.utils.data import DataLoader


class MixedDataloader:
    """
    Samples one dataset per step using configurable probabilities.
    Each yielded batch is homogeneous (single dataset).
    """

    def __init__(self, datasets, weights, batch_size=8, num_workers=4, pin_memory=True, seed=42):
        valid_datasets = {
            name: ds for name, ds in datasets.items()
            if hasattr(ds, '__len__') and len(ds) > 0
        }
        dropped = sorted(set(datasets.keys()) - set(valid_datasets.keys()))
        if dropped:
            print(f"[MixedDataloader] Skipping empty datasets: {dropped}")
        if not valid_datasets:
            raise ValueError("MixedDataloader requires at least one non-empty dataset.")

        self.loaders = {
            name: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
            )
            for name, dataset in valid_datasets.items()
        }
        self._iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        self.rng = np.random.default_rng(seed)
        self.sample_counts = {name: 0 for name in self.loaders}
        self.set_weights(weights)

    def set_weights(self, weights):
        names, probs = [], []
        for name, w in weights.items():
            if name not in self.loaders:
                continue
            fw = float(w)
            if fw < 0.0:
                raise ValueError(f"MixedDataloader weight must be non-negative, got {name}={w}")
            if fw > 0.0:
                names.append(name)
                probs.append(fw)
        if not names:
            raise ValueError("MixedDataloader received no positive weights for available datasets.")
        probs = np.asarray(probs, dtype=np.float64)
        self.names = names
        self.probs = probs / probs.sum()

    def get_sampling_stats(self):
        """Return cumulative per-dataset sample counts since initialization."""
        return dict(self.sample_counts)

    def _next_from(self, name):
        try:
            return next(self._iterators[name])
        except StopIteration:
            self._iterators[name] = iter(self.loaders[name])
            return next(self._iterators[name])

    def next_batch(self):
        last_exc = None
        # Retry a few times so one bad sample does not kill a long training run.
        for _ in range(max(4, len(self.names) * 2)):
            name = str(self.rng.choice(self.names, p=self.probs))
            try:
                batch = self._next_from(name)
                self.sample_counts[name] += 1
                return batch, name
            except Exception as exc:
                last_exc = exc
                print(f"[MixedDataloader][warn] failed to fetch batch from '{name}': {exc}")
                continue
        raise RuntimeError(f"MixedDataloader failed to fetch batch after retries: {last_exc}") from last_exc


def build_mixed_dataloaders(datasets, batch_size=8, num_workers=4):
    """Backward-compatible helper that returns one plain DataLoader per dataset."""
    valid_datasets = {
        name: ds for name, ds in datasets.items()
        if hasattr(ds, '__len__') and len(ds) > 0
    }
    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        for name, dataset in valid_datasets.items()
    }
