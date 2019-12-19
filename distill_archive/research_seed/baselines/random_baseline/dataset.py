import torch
from torch.utils.data import Dataset, DataLoader

class RandomCifarDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, height=32, width=32, channels=3, length=1000):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_size = [[self.channels, self.height, self.width]]

        sample = [torch.rand(*in_size) for in_size in input_size]
        sample = torch.Tensor(*sample)

        return sample

# dataset = RandomCifarDataset()
# data_loader = DataLoader(dataset, batch_size=4,
#                         shuffle=True, num_workers=4)

# # for sample in data_loader:
# #     print(sample.size())
# import torchvision
# import torchvision.transforms as transforms


# transform_test = transforms.Compose([
#                 transforms.Resize((32, 32)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             ])

# dataset = torchvision.datasets.STL10('./data', split='unlabeled', folds=None, transform=transform_test, target_transform=None, download=True)
# # dataset = RandomCifarDataset()
# data_loader = DataLoader(dataset, batch_size=4,
#                 shuffle=True, num_workers=4)

# for sample, target in data_loader:
#     print(sample.size())
#     print(target)