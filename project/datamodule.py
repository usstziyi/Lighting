"""
datamodule.py - CIFAR-10 数据模块
===================================
"""

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10
import lightning as L


class CIFAR10DataModule(L.LightningDataModule):
    """CIFAR-10 数据模块。

    功能:
    - 自动下载 CIFAR-10 数据集
    - 数据增强（随机裁剪、水平翻转等）
    - 标准化处理
    - 划分训练/验证集
    """

    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)
    NUM_CLASSES = 10
    IMAGE_SIZE = 32

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: float = 0.1,
        augment: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if augment:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

    def prepare_data(self):
        """下载数据集。"""
        CIFAR10(root=self.hparams.data_dir, train=True, download=True)
        CIFAR10(root=self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """设置数据集。"""
        if stage == "fit" or stage is None:
            train_full = CIFAR10(
                root=self.hparams.data_dir,
                train=True,
                transform=self.train_transform,
            )
            val_full = CIFAR10(
                root=self.hparams.data_dir,
                train=True,
                transform=self.test_transform,
            )

            val_size = int(self.hparams.val_split * len(train_full))
            train_size = len(train_full) - val_size

            indices = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(42))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            self.train_dataset = Subset(train_full, train_indices)
            self.val_dataset = Subset(val_full, val_indices)

        if stage == "test" or stage is None:
            self.test_dataset = CIFAR10(
                root=self.hparams.data_dir,
                train=False,
                transform=self.test_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
