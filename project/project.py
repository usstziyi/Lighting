"""
实战项目: CIFAR-10 图像分类
============================

本项目演示了 PyTorch Lightning 的完整使用流程:
1. LightningModule - 定义模型 (ResNet)
2. LightningDataModule - 处理 CIFAR-10 数据
3. Trainer - 训练控制器
4. Callbacks - 检查点、早停、学习率监控
5. Loggers - TensorBoard、CSV 日志

项目结构:
    project/
    ├── model.py          # ResNet 模型定义
    ├── datamodule.py    # CIFAR-10 数据模块
    ├── train.py          # 训练入口（命令行接口）
    ├── utils.py          # 工具函数
    └── project.py        # 本文件 - 快速演示

运行方式:
    # 完整训练（推荐）
    python project/train.py --max-epochs 20 --batch-size 128
    
    # 快速演示
    python project/project.py
    
    # 使用多 GPU
    python project/train.py --devices 2 --strategy ddp
    
    # 从检查点预测
    python project/train.py --predict ./checkpoints/.../best-xxx.ckpt
"""

import torch
import lightning as L
from model import ResNet
from datamodule import CIFAR10DataModule
from utils import get_cifar10_classes


def quick_demo():
    """快速演示：训练 3 个 epoch。"""
    print("=" * 60)
    print("CIFAR-10 图像分类 - 快速演示")
    print("=" * 60)

    # 设置随机种子
    L.seed_everything(42, workers=True)

    # 创建数据模块
    datamodule = CIFAR10DataModule(
        data_dir="./data",
        batch_size=64,
        num_workers=0,  # 使用 0 避免多进程问题
        val_split=0.1,
        augment=True,
    )

    # 创建模型
    model = ResNet(
        num_blocks=2,       # 使用较小的模型快速演示
        num_classes=10,
        learning_rate=0.01,
        weight_decay=1e-4,
        dropout_rate=0.2,
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

    # 创建 Trainer（简化配置）
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        precision="32-mix",
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    # 训练
    print("\n开始训练 (3 epochs)...")
    trainer.fit(model, datamodule=datamodule)

    # 测试
    print("\n测试结果:")
    results = trainer.test(model, datamodule=datamodule)
    print(f"  Test Loss: {results[0]['test/loss']:.4f}")
    print(f"  Test Accuracy: {results[0]['test/acc']:.4f}")

    # 类别名称
    class_names = get_cifar10_classes()
    print(f"\nCIFAR-10 类别: {class_names}")

    print("\n" + "=" * 60)
    print("演示完成! 运行完整训练请使用:")
    print("  python project/train.py --max-epochs 20")
    print("=" * 60)


if __name__ == "__main__":
    quick_demo()
