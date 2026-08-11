"""
train.py - 训练入口
===================

运行方式:
    python project/train.py --max-epochs 20 --batch-size 128
    
    使用多 GPU:
    python project/train.py --devices 2 --strategy ddp
    
    预测:
    python project/train.py --predict ./checkpoints/cifar10_resnet/best-epoch=xx-val_acc=xxxx.ckpt
"""

import argparse
import time
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from model import ResNet
from datamodule import CIFAR10DataModule
from utils import (
    get_cifar10_classes,
    print_classification_report,
    create_confusion_matrix,
    format_time,
)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="CIFAR-10 图像分类 - PyTorch Lightning 实战项目"
    )

    parser.add_argument("--max-epochs", type=int, default=20,
                       help="最大训练轮数")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="批大小")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                       help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="权重衰减")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout 率")
    parser.add_argument("--num-blocks", type=int, default=3,
                       help="ResNet 每层的残差块数量")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                       help="梯度裁剪值")

    parser.add_argument("--data-dir", type=str, default="./data",
                       help="数据保存目录")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="数据加载线程数")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="验证集比例")
    parser.add_argument("--no-augment", action="store_true",
                       help="不使用数据增强")

    parser.add_argument("--accelerator", type=str, default="auto",
                       choices=["auto", "gpu", "cpu", "mps"],
                       help="加速器类型")
    parser.add_argument("--devices", type=int, default=1,
                       help="使用的设备数量")
    parser.add_argument("--strategy", type=str, default="auto",
                       choices=["auto", "ddp", "ddp_find_unused_parameters"],
                       help="分布式训练策略")
    parser.add_argument("--precision", type=str, default="32-mix",
                       choices=["32-true", "32-mix", "16-mix"],
                       help="训练精度")

    parser.add_argument("--patience", type=int, default=5,
                       help="早停耐心值")
    parser.add_argument("--accumulate-grad", type=int, default=1,
                       help="梯度累积步数")

    parser.add_argument("--log-dir", type=str, default="./lightning_logs",
                       help="日志保存目录")
    parser.add_argument("--exp-name", type=str, default="cifar10_resnet",
                       help="实验名称")

    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--fast-dev-run", action="store_true",
                       help="快速开发模式（只运行 1 个 batch）")
    parser.add_argument("--predict", type=str, default=None,
                       help="仅进行预测（指定检查点路径）")

    return parser.parse_args()


def train(args):
    """训练模型。"""
    start_time = time.time()

    L.seed_everything(args.seed, workers=True)

    print("=" * 60)
    print("CIFAR-10 图像分类 - PyTorch Lightning 实战项目")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  最大 epoch: {args.max_epochs}")
    print(f"  批大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  精度: {args.precision}")
    print(f"  加速器: {args.accelerator}")
    print(f"  设备数量: {args.devices}")

    datamodule = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        augment=not args.no_augment,
    )

    model = ResNet(
        num_blocks=args.num_blocks,
        num_classes=10,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename="best-{epoch:02d}-{val_acc:.4f}",
        dirpath=f"./checkpoints/{args.exp_name}",
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=args.patience,
        min_delta=1e-4,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_name,
        log_graph=True,
    )

    csv_logger = CSVLogger(
        save_dir="./csv_logs",
        name=args.exp_name,
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy if args.devices > 1 else "auto",
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=[tb_logger, csv_logger],
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
        deterministic=False,
        benchmark=True,
    )

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    trainer.fit(model, datamodule=datamodule)

    training_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {format_time(training_time)}")

    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型")
    print("=" * 60)

    best_model_path = checkpoint_callback.best_model_path
    print(f"\n最佳模型路径: {best_model_path}")
    if checkpoint_callback.best_model_score:
        print(f"最佳验证准确率: {checkpoint_callback.best_model_score:.4f}")

    best_model = ResNet.load_from_checkpoint(best_model_path)
    best_model.eval()

    test_results = trainer.test(best_model, datamodule=datamodule)

    print("\n" + "=" * 60)
    print("详细分类报告")
    print("=" * 60)

    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    all_preds = []
    all_labels = []

    device = next(best_model.parameters()).device
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = best_model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    class_names = get_cifar10_classes()
    print_classification_report(all_preds, all_labels, class_names)

    cm = create_confusion_matrix(all_preds, all_labels)
    print("\n混淆矩阵 (行=真实, 列=预测):")
    header = "     " + "".join(f"{c:>6}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:5s}" + "".join(f"{v:6d}" for v in row))

    print("\n" + "=" * 60)
    print("项目完成!")
    print("=" * 60)
    print("\n查看日志:")
    print(f"  TensorBoard: tensorboard --logdir={args.log_dir}")
    print(f"  CSV 日志: ./csv_logs/{args.exp_name}/")


def predict(args):
    """使用训练好的模型进行预测。"""
    if args.predict is None:
        print("请使用 --predict 指定检查点路径")
        return

    print(f"加载模型: {args.predict}")
    model = ResNet.load_from_checkpoint(args.predict)
    model.eval()

    device = next(model.parameters()).device

    datamodule = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    class_names = get_cifar10_classes()

    print("\n" + "=" * 60)
    print("预测示例 (前 10 个样本)")
    print("=" * 60)

    for i in range(min(10, len(all_preds))):
        true_label = class_names[all_labels[i].item()]
        pred_label = class_names[all_preds[i].item()]
        confidence = all_probs[i][all_preds[i].item()].item()
        correct = "✓" if all_preds[i] == all_labels[i] else "✗"
        print(f"  样本 {i:2d}: 真实={true_label:12s} 预测={pred_label:12s} "
              f"置信度={confidence:.4f} {correct}")

    acc = (all_preds == all_labels).float().mean().item()
    print(f"\n总体准确率: {acc:.2%}")


def main():
    args = parse_args()

    if args.predict:
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
