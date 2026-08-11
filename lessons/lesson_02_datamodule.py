"""
Lesson 02: LightningDataModule 数据模块
==========================================

本课程学习:
1. LightningDataModule 的作用和结构
2. 如何将数据处理逻辑模块化
3. 数据预处理和数据增强
4. 使用 DataModule 配合 LightningModule

运行方式:
    python lessons/lesson_02_datamodule.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L


# ============================================================
# 1. 自定义数据集
# ============================================================
class SyntheticDataset(Dataset):
    """合成数据集，生成二分类数据。"""

    def __init__(self, n_samples: int = 1000, input_dim: int = 10, seed: int = 42):
        self.n_samples = n_samples
        self.input_dim = input_dim

        # 使用固定种子确保可复现
        g = torch.Generator().manual_seed(seed)

        # 生成特征
        self.X = torch.randn(n_samples, input_dim, generator=g)

        # 生成标签：两个高斯分布
        # 类别 0: 均值为 [-2, -2, 0, ...] 的分布
        # 类别 1: 均值为 [2, 2, 0, ...] 的分布
        self.y = torch.zeros(n_samples, dtype=torch.long)
        half = n_samples // 2

        self.X[:half] += torch.tensor([-2.0] + [0.0] * (input_dim - 1))
        self.X[half:] += torch.tensor([2.0] + [0.0] * (input_dim - 1))
        self.y[half:] = 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 2. LightningDataModule 的结构
# ============================================================
# LightningDataModule 将数据处理分为 5 个方法:
#   - __init__:      保存配置参数
#   - prepare_data:   下载数据或一次性预处理（只执行一次）
#   - setup:          划分数据集（每个 GPU 执行）
#   - train_dataloader:   返回训练 DataLoader
#   - val_dataloader:     返回验证 DataLoader
#   - test_dataloader:    返回测试 DataLoader


class SyntheticDataModule(L.LightningDataModule):
    """合成数据的 DataModule。"""

    def __init__(
        self,
        n_samples: int = 2000,
        input_dim: int = 10,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """一次性数据准备（下载、预处理等）。
        注意：此方法只在单个进程中执行一次。
        """
        # 对于合成数据，无需下载
        pass

    def setup(self, stage=None):
        """划分数据集。
        注意：此方法在每个 GPU 上执行。

        Args:
            stage: 'fit', 'validate', 'test', or None
        """
        if stage == "fit" or stage is None:
            # 创建训练和验证数据集
            full_train = SyntheticDataset(
                n_samples=self.hparams.n_samples,
                input_dim=self.hparams.input_dim,
                seed=42,
            )

            # 80% 训练，20% 验证
            train_size = int(0.8 * len(full_train))
            val_size = len(full_train) - train_size

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test" or stage is None:
            # 创建测试数据集
            self.test_dataset = SyntheticDataset(
                n_samples=500,
                input_dim=self.hparams.input_dim,
                seed=99,  # 使用不同种子
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )


# ============================================================
# 3. 配合 LightningModule 使用
# ============================================================
class Classifier(L.LightningModule):
    """二分类模型。"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # 添加学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]


# ============================================================
# 4. 主程序
# ============================================================
def main():
    print("=" * 60)
    print("Lesson 02: LightningDataModule 数据模块")
    print("=" * 60)

    # 创建 DataModule
    datamodule = SyntheticDataModule(
        n_samples=3000,
        input_dim=10,
        batch_size=32,
        num_workers=0,
    )

    # 创建模型
    model = Classifier(input_dim=10, hidden_dim=64)

    # 创建 Trainer
    trainer = L.Trainer(
        max_epochs=15,
        accelerator="auto",
        devices=1,
        precision="32-mix",
        gradient_clip_val=1.0,
    )

    # 方式 1: 使用 datamodule 自动加载数据
    # Trainer 会自动调用 datamodule.setup() 和 dataloader 方法
    print("\n方式 1: 使用 DataModule 训练")
    trainer.fit(model, datamodule=datamodule)

    # 测试
    print("\n在测试集上评估...")
    trainer.test(model, datamodule=datamodule)

    # 方式 2: 手动调用 setup 并使用 dataloader
    print("\n" + "=" * 60)
    print("方式 2: 手动控制 DataModule 流程")
    print("=" * 60)

    datamodule.setup(stage="fit")

    # 检查数据
    for batch in datamodule.train_dataloader():
        x, y = batch
        print(f"  训练 batch: x shape={x.shape}, y shape={y.shape}")
        print(f"  标签分布: {torch.bincount(y)}")
        break

    for batch in datamodule.val_dataloader():
        x, y = batch
        print(f"  验证 batch: x shape={x.shape}, y shape={y.shape}")
        print(f"  标签分布: {torch.bincount(y)}")
        break

    print("\n" + "=" * 60)
    print("Lesson 02 完成!")
    print("=" * 60)
    print("\n关键要点:")
    print("  1. DataModule 将数据处理与模型训练解耦")
    print("  2. setup() 根据 stage 参数执行不同的初始化")
    print("  3. 可以轻松在不同项目间复用数据处理逻辑")
    print("  4. 支持多 GPU 训练时的数据分发")


if __name__ == "__main__":
    main()
