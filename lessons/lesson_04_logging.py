"""
Lesson 04: 日志记录 (Logging)
================================

本课程学习:
1. TensorBoardLogger - 使用 TensorBoard 可视化
2. CSVLogger - 将指标保存为 CSV 文件
3. 自定义日志记录
4. 如何在 LightningModule 中记录不同类型的数据

运行方式:
    python lessons/lesson_04_logging.py
    
    启动 TensorBoard 查看日志:
    tensorboard --logdir=./lightning_logs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger


# ============================================================
# 1. 定义一个记录丰富指标的模型
# ============================================================
class AdvancedModel(L.LightningModule):
    """演示如何记录各种类型的指标。"""

    def __init__(self, input_dim: int = 20, hidden_dims: list = [64, 32], output_dim: int = 4):
        super().__init__()
        self.save_hyperparameters()

        # 构建网络
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # 用于计算 F1 score
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        # 收集预测用于计算 epoch 级指标
        self.train_preds.append(preds)
        self.train_labels.append(y)

        # 记录 step 级指标
        self.log("train/loss_step", loss, on_step=True, on_epoch=False)
        self.log("train/acc_step", (preds == y).float().mean(), on_step=True, on_epoch=False)

        # 记录额外的自定义指标
        # 例如：预测概率的均值和标准差
        probs = F.softmax(logits, dim=1)
        self.log("train/prob_mean", probs.mean(), on_step=True, on_epoch=False)
        self.log("train/prob_std", probs.std(), on_step=True, on_epoch=False)

        return loss

    def on_train_epoch_end(self):
        """epoch 结束时计算汇总指标。"""
        if self.train_preds:
            all_preds = torch.cat(self.train_preds)
            all_labels = torch.cat(self.train_labels)

            # 计算准确率
            acc = (all_preds == all_labels).float().mean()

            # 计算 F1 score（手动计算）
            # 判断样本 i 是否属于类别 cls：
            # preds[i] == cls ?   模型认不认它是 cls（模型侧）
            # labels[i] == cls ?  它实际上是不是 cls（事实侧）
            f1_scores = []
            for cls in range(self.hparams.output_dim):
                tp = ((all_preds == cls) & (all_labels == cls)).sum().float() # True Positive
                fp = ((all_preds == cls) & (all_labels != cls)).sum().float() # False Positive
                fn = ((all_preds != cls) & (all_labels == cls)).sum().float() # False Negative
                tn = ((all_preds != cls) & (all_labels != cls)).sum().float() # True Negative

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                f1_scores.append(f1)

            macro_f1 = torch.tensor(f1_scores).mean()

            # 记录 epoch 级指标
            self.log("train/acc_epoch", acc, on_epoch=True)
            self.log("train/f1_epoch", macro_f1, on_epoch=True)

            # 清空缓存
            self.train_preds.clear()
            self.train_labels.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_preds.append(preds)
        self.val_labels.append(y)

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """验证 epoch 结束时计算汇总指标。"""
        if self.val_preds:
            all_preds = torch.cat(self.val_preds)
            all_labels = torch.cat(self.val_labels)

            acc = (all_preds == all_labels).float().mean()

            # 计算每个类别的准确率
            class_accs = []
            for cls in range(self.hparams.output_dim):
                mask = all_labels == cls
                if mask.sum() > 0:
                    cls_acc = (all_preds[mask] == cls).float().mean()
                    class_accs.append(cls_acc)
                else:
                    class_accs.append(torch.tensor(0.0))

            avg_class_acc = torch.tensor(class_accs).mean()

            self.log("val/acc", acc, prog_bar=True)
            self.log("val/avg_class_acc", avg_class_acc)

            # 记录直方图数据（用于 TensorBoard）
            if self.logger:
                self.logger.experiment.add_histogram(
                    "val/prediction_distribution",
                    all_preds.float(),
                    self.current_epoch,
                )

            self.val_preds.clear()
            self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("test/loss", loss)
        self.log("test/acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }


# ============================================================
# 2. 准备数据
# ============================================================
def create_data(n_samples=2000, input_dim=20, output_dim=4):
    g = torch.Generator().manual_seed(42)
    X = torch.randn(n_samples, input_dim, generator=g)
    weights = torch.randn(input_dim, output_dim, generator=g)
    logits = X @ weights
    y = torch.argmax(logits, dim=1)

    train_size = int(0.8 * n_samples)
    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    val_dataset = TensorDataset(X[train_size:], y[train_size:])

    return (
        DataLoader(train_dataset, batch_size=64, shuffle=True),
        DataLoader(val_dataset, batch_size=64, shuffle=False),
    )


# ============================================================
# 3. 主程序：演示各种日志器
# ============================================================
def main():
    print("=" * 60)
    print("Lesson 04: 日志记录 (Logging)")
    print("=" * 60)

    # 创建数据
    train_loader, val_loader = create_data(n_samples=2000, input_dim=20, output_dim=4)

    # 创建模型
    model = AdvancedModel(input_dim=20, hidden_dims=[64, 32], output_dim=4)

    # ============================================================
    # 创建日志器
    # ============================================================

    # 1. TensorBoardLogger
    tb_logger = TensorBoardLogger(
        save_dir="./lightning_logs",       # 保存目录
        name="tensorboard_logs",           # 实验名称
        version="v1",                       # 版本号
        log_graph=True,                     # 是否记录模型图
    )

    # 2. CSVLogger
    csv_logger = CSVLogger(
        save_dir="./csv_logs",              # 保存目录
        name="csv_logs",                    # 实验名称
        version="v1",                       # 版本号
    )

    # 可以同时使用多个日志器
    loggers = [tb_logger, csv_logger]

    # ============================================================
    # 创建 Trainer 并使用日志器
    # ============================================================
    trainer = L.Trainer(
        max_epochs=15,
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        logger=loggers,  # 使用多个日志器
        log_every_n_steps=5,
    )

    # ============================================================
    # 自定义超参数记录
    # ============================================================
    extra_hparams = {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "architecture": "MLP-64-32-4",
    }

    # 在训练前记录额外的超参数
    for logger in loggers:
        logger.log_hyperparams(extra_hparams)

    # 开始训练
    print("\n开始训练...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ============================================================
    # 训练完成后的日志记录
    # ============================================================
    print("\n训练完成，查看日志:")
    print(f"  TensorBoard 日志: ./lightning_logs/tensorboard_logs/v1/")
    print(f"  CSV 日志: ./csv_logs/csv_logs/v1/metrics.csv")

    # 打印 CSV 日志内容
    print("\n" + "=" * 60)
    print("CSV 日志示例 (最后几行):")
    print("=" * 60)

    import csv
    csv_path = "./csv_logs/csv_logs/v1/metrics.csv"
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            # 打印表头和最后 5 行
            for row in rows[:1]:  # 表头
                print("  " + " | ".join(row))
            print(f"  ... (共 {len(rows) - 1} 行数据)")
            for row in rows[-5:]:  # 最后 5 行
                print("  " + " | ".join(row))
    except FileNotFoundError:
        print("  CSV 文件未找到")

    # ============================================================
    # 如何查看 TensorBoard
    # ============================================================
    print("\n" + "=" * 60)
    print("查看 TensorBoard 的方法:")
    print("=" * 60)
    print("  1. 在终端运行:")
    print("     tensorboard --logdir=./lightning_logs")
    print("  2. 浏览器访问: http://localhost:6006")
    print("  3. 在 Jupyter Notebook 中:")
    print("     %load_ext tensorboard")
    print("     %tensorboard --logdir=./lightning_logs")

    # 测试
    print("\n在测试集上评估...")
    trainer.test(model, dataloaders=val_loader)

    print("\n" + "=" * 60)
    print("Lesson 04 完成!")
    print("=" * 60)
    print("\n关键要点:")
    print("  1. TensorBoardLogger 提供丰富的可视化")
    print("  2. CSVLogger 便于数据分析和导入 Excel")
    print("  3. 可以同时使用多个日志器")
    print("  4. self.log() 支持 on_step/on_epoch 控制")
    print("  5. 可以自定义记录直方图、图像等")
    print("  6. hparams 自动记录模型配置")


if __name__ == "__main__":
    main()
