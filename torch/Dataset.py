import torch
import pandas as pd
class train_datasets(torch.utils.data.Dataset):
    def __init__(self, transform = None):
        self.transform = transform

        data = pd.read_excel()
        data = pd.read_csv()

        self.data = [1, 2, 3, 4, 5, 6]
        self.label = [0, 1, 0, 1, 0, 1]


    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label
class test_datasets(torch.utils.data.Dataset):
    def __init__(self, transform = None):
        self.transform = transform

        data = pd.read_excel()
        data = pd.read_csv()

        self.data = [1, 2, 3, 4, 5, 6]
        self.label = [0, 1, 0, 1, 0, 1]


    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label
