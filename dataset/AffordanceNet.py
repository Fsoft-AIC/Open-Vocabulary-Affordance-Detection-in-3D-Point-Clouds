import os
from os.path import join as opj
import numpy as np
from torch.utils.data import Dataset
import pickle as pkl


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


class AffordNetDataset(Dataset):
    def __init__(self, data_dir, split, partial=False):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.partial = partial

        self.load_data()
        self.affordances = self.all_data[0]["affordance"]
        return

    def load_data(self):
        self.all_data = []

        if self.partial:
            with open(opj(self.data_dir, 'partial_view_%s_data.pkl' % self.split), 'rb') as f:
                temp_data = pkl.load(f)
        else:
            with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split), 'rb') as f:
                temp_data = pkl.load(f)
        for _, info in enumerate(temp_data):
            if self.partial:
                partial_info = info["partial"]
                for view, data_info in partial_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["view_id"] = view
                    temp_info["data_info"] = data_info
                    self.all_data.append(temp_info)
            else:
                temp_info = {}
                temp_info["shape_id"] = info["shape_id"]
                temp_info["semantic class"] = info["semantic class"]
                temp_info["affordance"] = info["affordance"]
                temp_info["data_info"] = info["full_shape"]
                self.all_data.append(temp_info)

    def __getitem__(self, index):

        data_dict = self.all_data[index]
        modelid = data_dict["shape_id"]
        modelcat = data_dict["semantic class"]

        data_info = data_dict["data_info"]
        model_data = data_info["coordinate"].astype(np.float32)
        labels = data_info["label"]
        
        temp = labels.astype(np.float32).reshape(-1, 1)
        model_data = np.concatenate((model_data, temp), axis=1)

        datas = model_data[:, :3]
        targets = model_data[:, 3:]

        datas, _, _ = pc_normalize(datas)

        return datas, datas, targets, modelid, modelcat

    def __len__(self):
        return len(self.all_data)