import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self) -> None:
        # Data Loading
        xy = np.loadtxt("Files\wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # size = n_samples, 1
        self.n_samples = xy.shape[0]

    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    dataset = WineDataset()
    # Load whole dataset with DataLoader
    # shuffle: shuffle data, good for training
    # num_workers: faster loading with multiple subprocesses
    # !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
    data_loader = DataLoader(dataset=dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=2)

    # convert to an iterator and look at one random sample
    num_epochs = 2
    batch_size = 4
    total_samples = len(dataset)
    n_iters = math.ceil(total_samples/batch_size)
    print(total_samples, n_iters)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(data_loader):
            if (i+1) % 5 == 0:
                print(f"epoch {epoch + 1}/{num_epochs}, step {i+1}/{n_iters}, inputs {inputs.shape}")