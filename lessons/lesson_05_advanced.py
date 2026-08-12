"""
Lesson 05: 进阶训练技巧
==========================

本课程学习:
1. 多 GPU 训练策略
   - 单 GPU 训练
   - DDP (Distributed Data Parallel)
2. 混合精度训练 (AMP)
3. 梯度累积 (模拟大 batch size)
4. 随机种子设置确保可复现
5. 调试模式
6. 梯度检查点 (节省显存)

运行方式:
    python lessons/lesson_05_advanced.py
    
    多 GPU 训练:
    python lessons/lesson_05_advanced.py --strategy ddp --devices 2
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


# ============================================================
# 1. 定义模型
# ============================================================
class AdvancedClassifier(L.LightningModule):
    """演示多种训练技巧的分类模型。"""

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: list = [128, 64],
        output_dim: int = 10,
        dropout: float = 0.3,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 构建网络
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 使用梯度检查点节省显存,用时间换空间
        if self.hparams.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self.net, x, use_reentrant=False)
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # 记录指标
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_acc", (torch.argmax(logits, dim=1) == y).float().mean(), 
                 prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).float().mean()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # 使用不同的优化器配置
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        # CosineAnnealingWarmRestarts 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }


# ============================================================
# 2. 准备数据
# ============================================================
def create_data(n_samples=5000, input_dim=32, output_dim=10):
    g = torch.Generator().manual_seed(42)
    X = torch.randn(n_samples, input_dim, generator=g)
    weights = torch.randn(input_dim, output_dim, generator=g)
    logits = X @ weights
    y = torch.argmax(logits, dim=1)

    train_size = int(0.8 * n_samples)
    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    val_dataset = TensorDataset(X[train_size:], y[train_size:])

    return (
        DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0),
        DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0),
    )


# ============================================================
# 3. 演示不同的训练策略
# ============================================================

def demo_single_gpu():
    """演示单 GPU 训练。"""
    print("\n" + "=" * 60)
    print("演示 1: 单 GPU 训练")
    print("=" * 60)

    train_loader, val_loader = create_data(n_samples=3000)
    model = AdvancedClassifier(input_dim=32, hidden_dims=[128, 64], output_dim=10)

    # 在 CPU 上使用 32-true，在 GPU 上使用 16-mixed
    has_gpu = torch.cuda.is_available()
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="gpu" if has_gpu else "cpu",
        devices=1,
        precision="16-mixed" if has_gpu else "32-true",
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def demo_ddp():
    """演示 DDP 分布式训练 (需要多 GPU)。"""
    print("\n" + "=" * 60)
    print("演示 2: DDP 分布式训练")
    print("=" * 60)

    if torch.cuda.device_count() < 2:
        print("  [跳过] 需要至少 2 个 GPU")
        print(f"  当前 GPU 数量: {torch.cuda.device_count()}")
        return

    train_loader, val_loader = create_data(n_samples=5000)
    model = AdvancedClassifier(input_dim=32, hidden_dims=[128, 64], output_dim=10)

    trainer = L.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=2,
        strategy="ddp",      # DDP 策略
        precision="16-mixed",
        gradient_clip_val=1.0,
        sync_batchnorm=True,  # 同步 BatchNorm
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def demo_gradient_accumulation():
    """演示梯度累积 (模拟大 batch size)。"""
    print("\n" + "=" * 60)
    print("演示 3: 梯度累积")
    print("=" * 60)

    train_loader, val_loader = create_data(n_samples=3000)
    model = AdvancedClassifier(input_dim=32, hidden_dims=[128, 64], output_dim=10)

    # 实际 batch_size = 32, 但通过累积模拟 32 * 4 = 128 的 batch size
    has_gpu = torch.cuda.is_available()
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="gpu" if has_gpu else "cpu",
        devices=1,
        accumulate_grad_batches=4, # 梯度累积步数
        precision="16-mixed" if has_gpu else "32-true",
        gradient_clip_val=1.0,
    )

    print("  实际 batch_size = 32")
    print("  梯度累积步数 = 4")
    print("  等效 batch_size = 128")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def demo_mixed_precision():
    """演示不同精度的训练。"""
    print("\n" + "=" * 60)
    print("演示 4: 混合精度训练")
    print("=" * 60)

    train_loader, val_loader = create_data(n_samples=3000)

    # FP32 训练
    print("\n  使用 FP32 精度:")
    model_fp32 = AdvancedClassifier(input_dim=32, hidden_dims=[128, 64], output_dim=10)
    trainer_fp32 = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        precision="32-true",  # 纯 FP32
    )
    trainer_fp32.fit(model_fp32, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # FP16 混合精度训练（在支持 CUDA 时）
    if torch.cuda.is_available():
        print("\n  使用 FP16 混合精度:")
        model_fp16 = AdvancedClassifier(input_dim=32, hidden_dims=[128, 64], output_dim=10)
        trainer_fp16 = L.Trainer(
            max_epochs=3,
            accelerator="gpu",
            devices=1,
            precision="16-mixed",  # FP16 混合精度
        )
        trainer_fp16.fit(model_fp16, train_dataloaders=train_loader, val_dataloaders=val_loader)


def demo_reproducibility():
    """演示确保可复现性的设置。"""
    print("\n" + "=" * 60)
    print("演示 5: 可复现性设置")
    print("=" * 60)

    # 设置全局随机种子
    # workers=True ：除了固定 PyTorch、NumPy、Python 等全局随机种子外，
    # 还会把 DataLoader 的 num_workers 线程也种上种子。
    # 这样每个数据加载 worker 生成的随机顺序/数据也是确定的。
    L.seed_everything(42, workers=True)

    train_loader, val_loader = create_data(n_samples=2000)

    # 第一次训练
    model1 = AdvancedClassifier(input_dim=32, hidden_dims=[64, 32], output_dim=10)
    trainer1 = L.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        deterministic=True,  # 启用确定性算法
        benchmark=False,     # 禁用 benchmark 模式
    )
    trainer1.fit(model1, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 第二次训练（相同种子，相同配置）
    L.seed_everything(42, workers=True)  # 重置种子
    model2 = AdvancedClassifier(input_dim=32, hidden_dims=[64, 32], output_dim=10)
    trainer2 = L.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        deterministic=True,
        benchmark=False,
    )
    trainer2.fit(model2, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 比较结果
    val_acc1 = trainer1.callback_metrics.get("val_acc")
    val_acc2 = trainer2.callback_metrics.get("val_acc")

    print(f"\n  第一次训练 val_acc: {val_acc1:.4f}" if val_acc1 else "")
    print(f"  第二次训练 val_acc: {val_acc2:.4f}" if val_acc2 else "")
    if val_acc1 and val_acc2:
        print(f"  结果一致: {torch.isclose(val_acc1, val_acc2)}")


def demo_debugging():
    """演示调试模式。"""
    print("\n" + "=" * 60)
    print("演示 6: 调试模式")
    print("=" * 60)

    train_loader, val_loader = create_data(n_samples=500)  # 使用更小的数据集
    model = AdvancedClassifier(input_dim=32, hidden_dims=[32, 16], output_dim=10)

    # Fast-dev-run: 只运行 2 个 batch 来验证代码是否正确
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        fast_dev_run=True,  # 快速开发模式
    )

    print("  fast_dev_run=True: 只运行 1 个 batch 验证流程")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Limit batches
    trainer2 = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        limit_train_batches=10,   # 只使用前 10 个 batch
        limit_val_batches=5,     # 只使用前 5 个 batch
    )

    print("\n  limit_batches: 限制 batch 数量")
    trainer2.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


# ============================================================
# 4. 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="PyTorch Lightning 进阶训练技巧")
    parser.add_argument("--demo", type=str, default="all",
                       choices=["single", "ddp", "accumulate", "precision", 
                                "reproducible", "debug", "all"],
                       help="选择演示内容")
    args = parser.parse_args()

    print("=" * 60)
    print("Lesson 05: 进阶训练技巧")
    print("=" * 60)

    if args.demo == "single":
        demo_single_gpu()
    elif args.demo == "ddp":
        demo_ddp()
    elif args.demo == "accumulate":
        demo_gradient_accumulation()
    elif args.demo == "precision":
        demo_mixed_precision()
    elif args.demo == "reproducible":
        demo_reproducibility()
    elif args.demo == "debug":
        demo_debugging()
    elif args.demo == "all":
        # 依次演示所有内容
        demo_single_gpu()
        demo_gradient_accumulation()
        demo_mixed_precision()
        demo_reproducibility()
        demo_debugging()

    print("\n" + "=" * 60)
    print("Lesson 05 完成!")
    print("=" * 60)
    print("\n关键要点:")
    print("  1. strategy='ddp' 实现多 GPU 训练")
    print("  2. precision='16-mixed' 混合精度加速训练")
    print("  3. accumulate_grad_batches 模拟大 batch size")
    print("  4. L.seed_everything() 确保可复现性")
    print("  5. fast_dev_run=True 快速调试")
    print("  6. gradient_clip_val 防止梯度爆炸")
    print("  7. deterministic=True 启用确定性算法")


if __name__ == "__main__":
    main()
