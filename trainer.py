import torch
from torch.utils.data import Dataset, random_split
import random
import pickle
import numpy as np
from utils import balance_data


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(self.data['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        torch_local_context = torch.from_numpy(np.array(self.data['local_context'][idx])).permute(0,3,1,2).float()
        torch_pose = torch.from_numpy(np.array(self.data['pose'][idx])).float()
        torch_bbox = torch.from_numpy(np.array(self.data['bbox'][idx])).float()
        torch_speed = torch.from_numpy(np.array(self.data['speed'][idx])).unsqueeze(-1).float()
        torch_labels = torch.from_numpy(np.array(self.data['labels'][idx])).float()


        return {
            'local_context': torch_local_context,
            'pose': torch_pose,
            'bbox': torch_bbox,
            'speed': torch_speed,
            'label': torch_labels
        }
    



def train_test_split(checkpoint_path):
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)


    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    data=balance_data(data, remove_n_samples=100)
    dataset=CustomDataset(data)

    # Split intotrain, validation and test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    #print(train_dataset.__len__())

    return train_dataset, val_dataset, test_dataset