import glob
import random
from typing import Tuple, Any
import sys

sys.setrecursionlimit(3000)  # 默认递归深度是 1000，根据需要适当调大

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch_snippets import fname, parent, read
from sklearn.model_selection import train_test_split


class AlphabetDataset(Dataset):
    """
    Dataset for Images of English Alphabets (A-Z)
    """

    def __init__(self, file_path: str, transform=None):
        super(AlphabetDataset, self).__init__()

        self.files = file_path  # glob.glob(f"{file_path}/*/*.png")  # 收集所有图像文件
        self.transform = transform
        self.letter_2_index = {
            letter: idx for idx, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        }
        self.num_classes = len(self.letter_2_index)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, index):
        fpath = self.files[index]
        letter_name = fname(parent(fpath))  # 获取字母名称（文件夹名）
        img = read(fpath, 1)  # 读取图像
        return img, letter_name

    def __len__(self):
        return len(self.files)

    def choose(self):
        rand_index = random.randint(0, len(self.files) - 1)
        return self[rand_index]

    def collate_fn(self, batch):
        _imgs, targets = list(zip(*batch))
        if self.transform:
            imgs = [self.transform(img)[None] for img in _imgs]

        targets = [
            torch.tensor([self.letter_2_index[target]]).to(self.device)
            for target in targets
        ]

        imgs, targets = [torch.cat(i).to(self.device) for i in [imgs, targets]]

        return imgs, targets, _imgs


# 定义数据加载函数
def load_alphabet_data(
    file_path: str,
) -> Tuple[DataLoader[Any], DataLoader[Any], AlphabetDataset, AlphabetDataset]:
    """
    Loads all alphabet images under the given file path
    :param file_path: the path of the alphabet images
    :return: Tuple of 2 data loaders and 2 datasets
    """
    all_files = glob.glob(f"{file_path}/*/*.png")
    train_files, val_files = train_test_split(all_files)
    train_ds = AlphabetDataset(train_files, train_transform)
    val_ds = AlphabetDataset(val_files, val_transform)
    train_ld, val_ld = (
        DataLoader(train_ds, shuffle=True, collate_fn=train_ds.collate_fn),
        DataLoader(val_ds, shuffle=True, collate_fn=val_ds.collate_fn),
    )
    return train_ld, val_ld, train_ds, val_ds


train_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(64),
        T.CenterCrop(64),
        T.ColorJitter(
            brightness=(0.95, 1.05),
            contrast=(0.95, 1.05),
            saturation=(0.95, 1.05),
            hue=0.05,
        ),
        T.RandomAffine(degrees=5, translate=(0.01, 0.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

val_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(64),
        T.CenterCrop(64),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
