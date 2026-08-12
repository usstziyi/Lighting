"""
Lesson 01: PyTorch Lightning 基础入门
========================================

本课程学习:
1. LightningModule 的核心结构
2. Trainer 的基本使用
3. 如何将普通 PyTorch 代码转换为 Lightning 风格

运行方式:
    python lessons/lesson_01_basic.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar


# ============================================================
# 1. 定义 LightningModule
# ============================================================
# LightningModule 是 PyTorch Lightning 的核心，它将模型分为 5 个部分:
#   - __init__: 定义模型结构（神经网络层）
#   - forward:  定义前向传播逻辑
#   - training_step:   定义训练循环
#   - validation_step: 定义验证循环
#   - test_step:        定义测试循环
#   - configure_optimizers: 配置优化器和学习率调度器


class SimpleMLP(L.LightningModule):
    """一个简单的多层感知机，用于演示 LightningModule 的基本结构。"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, output_dim: int = 2):
        super().__init__()
        self.save_hyperparameters()  # 自动保存超参数，便于重现

        # 定义网络结构
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

        # 用于记录指标的字典
        self.train_loss_epoch = []
        self.val_loss_epoch = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def _shared_step(self, batch, batch_idx):
        """训练和验证的共用逻辑"""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        loss, acc = self._shared_step(batch, batch_idx)

        # 记录指标
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        loss, acc = self._shared_step(batch, batch_idx)

        # on_epoch=True 会在整个 epoch 结束后聚合指标
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        loss, acc = self._shared_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# ============================================================
# 2. 准备数据
# ============================================================
def create_dummy_data(n_samples: int = 1000, input_dim: int = 10):
    """创建模拟数据"""
    # 生成随机特征
    X = torch.randn(n_samples, input_dim)
    # 生成标签：基于前两个特征的线性组合
    weights = torch.randn(input_dim)
    logits = X @ weights
    y = (logits > 0).long()

    # 划分数据集
    train_size = int(0.6 * n_samples)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    return train_dataset, val_dataset


# ============================================================
# 3. 使用 Trainer 进行训练
# ============================================================
def main():
    print("=" * 60)
    print("Lesson 01: PyTorch Lightning 基础入门")
    print("=" * 60)

    # 创建数据
    train_dataset, val_dataset = create_dummy_data(n_samples=200000, input_dim=10)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 创建模型
    model = SimpleMLP(input_dim=10, hidden_dim=32, output_dim=2)

    print(f"\n模型结构:\n{model}")
    print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建 Trainer
    # Trainer 封装了所有训练工程细节: GPU管理、分布式训练、日志记录等
    trainer = L.Trainer(
        max_epochs=10,           # 最大训练轮数
        accelerator="auto",      # 自动选择设备（GPU/CPU）
        devices="auto",              # 使用的设备数量
        gradient_clip_val=1.0,  # 梯度裁剪，防止梯度爆炸
        log_every_n_steps=10,    # 每多少步记录一次日志
        callbacks=[RichProgressBar(leave=True)],  # 保留每个 epoch 的进度条，不覆盖
    )

    # 开始训练
    print("\n开始训练...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # # 测试模型
    # print("\n在验证集上测试...")
    # trainer.test(model, dataloaders=val_loader)

    # # 演示如何保存和加载模型
    # print("\n保存模型...")
    # trainer.save_checkpoint("lesson_01_model.ckpt")

    # # 从检查点加载
    # print("\n从检查点加载模型...")
    # loaded_model = SimpleMLP.load_from_checkpoint("lesson_01_model.ckpt")

    # # 推理示例
    # loaded_model.eval()
    # test_input = torch.randn(5, 10)
    # with torch.no_grad():
    #     predictions = loaded_model(test_input)
    #     predicted_classes = torch.argmax(predictions, dim=1)

    # print(f"\n推理示例 (5个样本):")
    # print(f"  预测类别: {predicted_classes}")

    # print("\n" + "=" * 60)
    # print("Lesson 01 完成!")
    # print("=" * 60)


if __name__ == "__main__":
    main()
