"""
Lesson 03: 回调函数 (Callbacks)
=================================

本课程学习:
1. ModelCheckpoint - 保存最佳模型
2. EarlyStopping - 早停机制
3. LearningRateMonitor - 监控学习率
4. 自定义回调 - 实现特定功能

运行方式:
    python lessons/lesson_03_callbacks.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback,
)


# ============================================================
# 1. 定义模型
# ============================================================
class SimpleClassifier(L.LightningModule):
    """简单分类模型，用于演示回调。"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, output_dim: int = 3):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        # 使用余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ============================================================
# 2. 准备数据
# ============================================================
def create_data(n_samples=1500, input_dim=10, output_dim=3):
    g = torch.Generator().manual_seed(42)
    X = torch.randn(n_samples, input_dim, generator=g)
    weights = torch.randn(input_dim, output_dim, generator=g)
    logits = X @ weights
    y = torch.argmax(logits, dim=1)

    train_size = int(0.8 * n_samples)
    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    val_dataset = TensorDataset(X[train_size:], y[train_size:])

    return (
        DataLoader(train_dataset, batch_size=32, shuffle=True),
        DataLoader(val_dataset, batch_size=32, shuffle=False),
    )


# ============================================================
# 3. 自定义回调
# ============================================================
class AccuracyThresholdCallback(Callback):
    """当验证准确率达到阈值时，记录信息。"""

    def __init__(self, threshold: float = 0.8):
        super().__init__()
        self.threshold = threshold

    def on_validation_end(self, trainer, pl_module):
        """验证结束时检查指标。"""
        # trainer.callback_metrics — 返回一个字典，
        # 包含本轮训练/验证中所有通过 self.log() 记录的指标（如 val_acc 、 val_loss ）。
        # 在回调里可以通过 trainer 访问它，因为每个回调方法都会接收 trainer 参数。
        metrics = trainer.callback_metrics
        val_acc = metrics.get("val_acc")

        if val_acc is not None and val_acc >= self.threshold:
            print(f"\n{'='*50}")
            print(f"✓ 验证准确率达到 {self.threshold:.0%} 阈值!")
            print(f"  当前 val_acc = {val_acc:.4f}")
            print(f"  可以考虑停止训练或调整学习率")
            print(f"{'='*50}")


class TimerCallback(Callback):
    """记录训练各阶段的耗时。"""

    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.use_cuda_events = torch.cuda.is_available()

    def on_train_epoch_start(self, trainer, pl_module):
        """每个 epoch 开始时记录时间。"""
        import time
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        """每个 epoch 结束时计算耗时。"""
        import time
        elapsed = time.time() - self._epoch_start_time
        self.epoch_times.append(elapsed)
        print(f"[Timer] Epoch {trainer.current_epoch + 1} 耗时: {elapsed:.2f}秒")

    def on_train_end(self, trainer, pl_module):
        """训练结束时汇总。"""
        if self.epoch_times:
            total_time = sum(self.epoch_times)
            avg_time = total_time / len(self.epoch_times)
            print(f"\n[Timer] 训练完成!")
            print(f"  总耗时: {total_time:.2f}秒")
            print(f"  平均每 epoch: {avg_time:.2f}秒")


# ============================================================
# 4. 使用所有回调进行训练
# ============================================================
def main():
    print("=" * 60)
    print("Lesson 03: 回调函数 (Callbacks)")
    print("=" * 60)

    # 创建数据
    train_loader, val_loader = create_data(n_samples=1500, input_dim=10, output_dim=3)

    # 创建模型
    model = SimpleClassifier(input_dim=10, hidden_dim=32, output_dim=3)

    # ============================================================
    # 创建各种回调
    # ============================================================

    # 1. ModelCheckpoint - 保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",          # 监控的指标
        mode="max",                 # 'max' 表示指标越大越好
        save_top_k=3,               # 保存最佳的 3 个模型
        save_last=True,             # 也保存最后一个 epoch 的模型
        filename="best-{epoch:02d}-{val_acc:.4f}",  # 文件名格式
        dirpath="./checkpoints",    # 保存目录
    )

    # 2. EarlyStopping - 早停机制
    early_stopping = EarlyStopping(
        monitor="val_loss",         # 监控的指标
        patience=5,                 # 容忍多少个 epoch 没有改善
        min_delta=0.001,            # 最小改善量
        mode="min",                 # 'min' 表示指标越小越好
        verbose=True,               # 打印早停信息
    )

    # 3. LearningRateMonitor - 监控学习率
    lr_monitor = LearningRateMonitor(
        logging_interval="epoch",  # 记录间隔: 'step' 或 'epoch'
        log_momentum=True,          # 是否记录动量
    )

    # 4. 自定义回调
    accuracy_callback = AccuracyThresholdCallback(threshold=0.7)
    timer_callback = TimerCallback()

    # ============================================================
    # 创建 Trainer 并使用回调
    # ============================================================
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=1.0,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            lr_monitor,
            accuracy_callback,
            timer_callback,
        ],
    )

    # 开始训练
    print("\n开始训练（带各种回调）...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ============================================================
    # 使用保存的最佳模型
    # ============================================================
    print("\n" + "=" * 60)
    print("使用最佳模型进行测试")
    print("=" * 60)

    # checkpoint_callback.best_model_path 自动保存了最佳模型路径
    print(f"\n最佳模型路径: {checkpoint_callback.best_model_path}")
    print(f"最佳模型分数: {checkpoint_callback.best_model_score:.4f}")

    # 从最佳模型路径加载
    best_model = SimpleClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()

    # 测试
    results = trainer.test(best_model, dataloaders=val_loader)
    print(f"\n测试结果: {results}")

    # 演示从 last.ckpt 恢复训练
    print("\n" + "=" * 60)
    print("演示从检查点恢复训练")
    print("=" * 60)

    last_ckpt_path = "./checkpoints/last.ckpt"
    # 如果存在 last.ckpt，可以恢复训练
    import os
    if os.path.exists(last_ckpt_path):
        print(f"\n找到 last.ckpt，演示恢复训练...")
        # 注意：这里只是演示，实际使用时可以继续训练
        # trainer.fit(model, ckpt_path=last_ckpt_path)
        print("  (已注释掉恢复训练代码，取消注释即可使用)")

    print("\n" + "=" * 60)
    print("Lesson 03 完成!")
    print("=" * 60)
    print("\n关键要点:")
    print("  1. ModelCheckpoint 自动保存最佳模型")
    print("  2. EarlyStopping 防止过拟合")
    print("  3. LearningRateMonitor 监控学习率变化")
    print("  4. 可以自定义回调实现任意功能")
    print("  5. 回调在训练循环的特定阶段自动触发")


if __name__ == "__main__":
    main()
