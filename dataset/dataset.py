import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torchvision

from dataset.utils import *


class TemporalDataset(Dataset):
    def __init__(self, modality_groups=None, modalityb="dx", test_size=0.2, mode="train", volume=True):
        self.patient_dict, (self.x, self.y, self.z) = get_tabular_data()
        self.data = []
        self.modalityb = modalityb
        self.volume = volume

        if modality_groups is None:
            modality_groups = {"dx": ["DX", "ADAS13", "Ventricles", "EXAMDATE"],
                               "cog": ["CDRSB", "ADAS11", "MMSE", "RAVLT_immediate"],
                               "vol": ["Hippocampus", "WholeBrain", "Entorhinal", "MidTemp"],
                               "pet": ["FDG", "AV45"],
                               "bio": ["ABETA_UPENNBIOMK9_04_19_17", "TAU_UPENNBIOMK9_04_19_17", "PTAU_UPENNBIOMK9_04_19_17"],
                               "demo": ["APOE4", "AGE"]}

        for i, (patient, v) in enumerate(self.patient_dict.items()):
            for j in range(len(v)):
                cur = v[j]
                row = {}
                row["ptid"] = i
                row["path"] = cur["path"]
                row["dx"] = one_hot_encode(int(cur["DX"]), 3)

                for group, submodalities in modality_groups.items():
                    row[group] = []
                    for submodality in submodalities:
                        row[group].append(cur[submodality])

                self.data.append(row)

        # train test split no label
        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size, random_state=42)
        if mode == "train":
            self.data = self.train_data
        else:
            self.data = self.test_data
        # row structure: for each data point, we have ptid, path and each group

    def __getitem__(self, index):
        cur = {}
        cur["dx"] = self.data[index]["dx"]
        volume, image = self.process_image(self.data[index]["path"])
        cur["modalityb"] = self.data[index][self.modalityb]
        if self.volume:
            cur["image"] = volume
        else:
            cur["image"] = image

        return cur

    def __len__(self):
        return len(self.data)

    def process_image(self, path):
        volume = get_volume(path)
        volume = volume.astype(np.float32)
        volume = transpose_volume(volume)
        volume = slice_volume(volume, self.x, self.y, self.z)
        image = get_frames(np.expand_dims(volume.squeeze(0), -1))
        return volume, image
