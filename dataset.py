import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Tuple


# ── Dataset ────────────────────────────────────────────────────────────────────
class FloodMapDataset(Dataset):
    def __init__(
        self,
        spatial      : np.ndarray,
        labels       : np.ndarray,
        rainfall     : np.ndarray,
        conditioning : Optional[np.ndarray] = None,
    ):
        super().__init__()

        assert spatial.ndim == 4,      f"spatial must be (S,3,H,W), got {spatial.shape}"
        assert labels.ndim == 3,       f"labels must be (S,H,W), got {labels.shape}"
        assert rainfall.ndim == 2,     f"rainfall must be (S,13), got {rainfall.shape}"
        assert spatial.shape[0] == labels.shape[0] == rainfall.shape[0], \
            "Scenario count mismatch across spatial/labels/rainfall"

        if conditioning is not None:
            assert conditioning.ndim == 2 and conditioning.shape[1] == 4, \
                f"conditioning must be (S,4), got {conditioning.shape}"
            assert conditioning.shape[0] == spatial.shape[0], \
                "Scenario count mismatch: conditioning vs spatial"

        self.spatial      = torch.from_numpy(spatial.astype(np.float32))       # (S,3,H,W)
        self.labels       = torch.from_numpy(labels.astype(np.int64))          # (S,H,W)
        self.rainfall     = torch.from_numpy(rainfall.astype(np.float32))      # (S,13)
        self.conditioning = (
            torch.from_numpy(conditioning.astype(np.float32))
            if conditioning is not None
            else torch.zeros(spatial.shape[0], 4, dtype=torch.float32)        # (S,4) zeros
        )
        self.has_conditioning = conditioning is not None
        self.n_scenarios      = spatial.shape[0]
        self.H, self.W        = spatial.shape[2], spatial.shape[3]

        self._print_summary()

    def __len__(self) -> int:
        return self.n_scenarios

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.spatial[idx],       # (3, H, W)
            self.rainfall[idx],      # (13,)
            self.conditioning[idx],  # (4,)
            self.labels[idx],        # (H, W)
        )

    def _print_summary(self):
        cond_str = f"(S,4) ✓" if self.has_conditioning else "(S,4) zeros — no conditioning"
        print(f"[FloodMapDataset]  {self.n_scenarios} scenarios")
        print(f"  Spatial      : {tuple(self.spatial.shape)}")
        print(f"  Rainfall     : {tuple(self.rainfall.shape)}")
        print(f"  Conditioning : {cond_str}")
        print(f"  Labels       : {tuple(self.labels.shape)}")

        # Class distribution across all valid pixels
        valid = self.labels[self.labels >= 0]
        total = valid.numel()
        names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
        print(f"  Class distribution (valid pixels):")
        for c, name in enumerate(names):
            cnt = (valid == c).sum().item()
            pct = 100.0 * cnt / total if total > 0 else 0
            print(f"    Class {c} ({name:<10}): {cnt:>9,}  ({pct:>5.1f}%)")


def create_cnn_datasets(
    cnn_train : Dict[str, np.ndarray],
    cnn_test  : Dict[str, np.ndarray],
) -> Tuple[FloodMapDataset, FloodMapDataset]:

    print("\n  Building Train dataset:")
    train_ds = FloodMapDataset(
        spatial      = cnn_train['spatial'],
        labels       = cnn_train['labels'],
        rainfall     = cnn_train['rainfall'],
        conditioning = cnn_train.get('conditioning'),
    )

    print("\n  Building Test dataset:")
    test_ds = FloodMapDataset(
        spatial      = cnn_test['spatial'],
        labels       = cnn_test['labels'],
        rainfall     = cnn_test['rainfall'],
        conditioning = cnn_test.get('conditioning'),
    )

    print(f"\n  ✓ Datasets ready")
    print(f"    Train : {len(train_ds)} scenarios  |  map {train_ds.H}×{train_ds.W}")
    print(f"    Test  : {len(test_ds)} scenarios   |  map {test_ds.H}×{test_ds.W}")

    return train_ds, test_ds
