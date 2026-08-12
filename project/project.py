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
from lightning.pytorch.loggers import TensorBoardLogger
from torchinfo import summary


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
        num_workers=4,
        val_split=0.1,
        augment=True, # 是否进行数据增强
    )

    # 创建模型
    model = ResNet(
        num_blocks=2,       # 使用较小的模型快速演示
        num_classes=10,
        learning_rate=0.01,
        weight_decay=1e-4,
        dropout_rate=0.2,
    )

    # torchinfo
    summary(
        model, 
        input_size=(1, 3, 32, 32),
        col_names=["input_size", "output_size", "num_params","trainable"],
        verbose=1,
    )


    # 创建 Logger（启用计算图记录到 TensorBoard）
    logger = TensorBoardLogger("lightning_logs", log_graph=True)

    # 提供示例输入，TensorBoard 才能追踪计算图
    model.example_input_array = torch.randn(1, 3, 32, 32)

    # 创建 Trainer（简化配置）
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        enable_progress_bar=True,
        logger=logger,
    )



    # 训练（fit 时自动将计算图记录到 TensorBoard）
    print("\n开始训练 (3 epochs)...")
    trainer.fit(model, datamodule=datamodule)

    return

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
