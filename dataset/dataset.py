import torch
import numpy as np
from typing import Optional, Callable, Dict
from torch.utils.data import Dataset
import random

class WaferDataset_utils:
    @staticmethod
    def transform_three_channels(img: np.ndarray) -> np.ndarray:
        ch0 = (img == 0).astype(np.float32)
        ch1 = (img == 1).astype(np.float32)
        ch2 = (img == 2).astype(np.float32)
        return np.stack([ch0, ch1, ch2], axis=0)


class WaferDataset(Dataset):
    def __init__(self, wmaps, labels):
        self.wmaps = np.array(wmaps)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.wmaps)

    def __getitem__(self, idx):
        img = self.wmaps[idx]
        x = WaferDataset_utils.transform_three_channels(img)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class AugmentedWaferDataset(Dataset):
    def __init__(self, wmaps, labels, target_counts : Optional[Dict] = None, augment_transform : Optional[Callable] = None, seed: int = 42):

        self.augment_transform = augment_transform
        self.target_counts = target_counts if target_counts is not None else {}
        self.class_data_dict = self._group_by_class(wmaps, labels)        
        self.augmented_data, self.augmented_labels = self._build_balanced_dataset()

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        img, y = self.augmented_data[idx], self.augmented_labels[idx]
        x = WaferDataset_utils.transform_three_channels(img) # [1, H, W] -> # [3, H, W]
        x = torch.tensor(x, dtype=torch.float32)        
        y = torch.tensor(y, dtype=torch.long)        
        return x, y

    def _group_by_class(self, wmaps, labels):
        class_data_dict = {label: [] for label in set(labels)}

        for wmap, label in zip(wmaps, labels):
            class_data_dict[label].append(wmap)
        return class_data_dict # dict{0: [data...], 1: [data...], ... ,8: [data...]}


    def _build_balanced_dataset(self):
        """Generate a balanced dataset across all classes"""

        augmented_data, augmented_labels = [], []

        for label, wmap_list in self.class_data_dict.items():
            d, l = self._balance_class(label, wmap_list)
            augmented_data.extend(d)
            augmented_labels.extend(l)
        return augmented_data, augmented_labels
    
    def _balance_class(self, label, wmap_list):
        """Augment a specific class up to the target count"""    

        augmented_data, augmented_labels = [], []
        cur_count = len(wmap_list)
        target_count = self.target_counts.get(label, cur_count)

        # add original data
        augmented_data.extend(wmap_list)
        augmented_labels.extend([label] * cur_count)

        # add original data
        need_counts = max(0, target_count - cur_count)
        for _ in range(need_counts):
            wmap_sample = random.choice(wmap_list)
            augmented_data.append(self._augment_sample(wmap_sample))
            augmented_labels.append(label)        
        return augmented_data, augmented_labels 
    

    def _augment_sample(self, wmap_sample):
        x = torch.tensor(wmap_sample, dtype=torch.float32)  # [H,W]
        if self.augment_transform:
            x = x.unsqueeze(0)           # [1,H,W]
            x = self.augment_transform(x)
            x = x.squeeze(0)             # [H,W]
            x = x.round().clamp(0, 2) # preserve wafer-map values after augmentation(Affine)
        return x.numpy()