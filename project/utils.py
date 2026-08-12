"""
utils.py - 工具函数
===================
"""

import torch
import numpy as np
from PIL import Image


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_cifar10_classes():
    """获取 CIFAR-10 类别名称。"""
    return CIFAR10_CLASSES


def denormalize_image(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)):
    """反归一化图像张量，用于可视化。

    Args:
        tensor: 归一化后的图像张量 (C, H, W)
        mean: 归一化均值
        std: 归一化标准差

    Returns:
        反归一化后的 numpy 数组PIL Image (H, W, C)
    """
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)

    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        tensor = tensor.cpu().numpy()

    image = tensor * std + mean
    image = np.clip(image, 0, 1)
    return image


def save_image(tensor, path, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)):
    """保存图像张量为文件。

    Args:
        tensor: 图像张量 (C, H, W) 或 (1, C, H, W)
        path: 保存路径
        mean: 归一化均值
        std: 归一化标准差
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    image = denormalize_image(tensor, mean, std)
    # PIL 无法解释 [0,1] 的浮点像素值，需要转换为 uint8 类型
    image_uint8 = (image * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def compute_accuracy(preds, labels):
    """计算准确率。

    Args:
        preds: 预测标签
        labels: 真实标签

    Returns:
        准确率 (float)
    """
    return (preds == labels).float().mean().item()


def compute_per_class_accuracy(preds, labels, num_classes=10):
    """计算每个类别的准确率。

    Args:
        preds: 预测标签
        labels: 真实标签
        num_classes: 类别数量

    Returns:
        每个类别的准确率列表
    """
    class_acc = []
    for cls in range(num_classes):
        # 找出所有属于第 cls 类的样本索引掩码
        mask = labels == cls
        if mask.sum() > 0:
            acc = (preds[mask] == cls).float().mean().item()
            class_acc.append(acc)
        else:
            class_acc.append(0.0)
    return class_acc


def create_confusion_matrix(preds, labels, num_classes=10):
    """创建混淆矩阵。

    Args:
        preds: 预测标签
        labels: 真实标签
        num_classes: 类别数量

    Returns:
        混淆矩阵 (numpy array)
    zip(labels_np, preds_np)
    # labels: [0, 1, 1, 2, 2, 2]
    # preds:  [0, 1, 2, 2, 0, 2]
    (0, 0) → 样本1: 真实0, 预测0
    (1, 1) → 样本2: 真实1, 预测1
    (1, 2) → 样本3: 真实1, 预测2
    (2, 2) → 样本4: 真实2, 预测2
    (2, 0) → 样本5: 真实2, 预测0
    (2, 2) → 样本6: 真实2, 预测2
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels.cpu().numpy(), preds.cpu().numpy()):
        cm[true_label, pred_label] += 1 # "真实为 i 类、但被预测为 j 类"的样本数
    return cm


def print_classification_report(preds, labels, class_names):
    """打印分类报告。

    Args:
        preds: 预测标签 tensor
        labels: 真实标签 tensor
        class_names: 类别名称列表
    """
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)

    for i, class_name in enumerate(class_names):
        mask = labels == i
        if mask.sum() > 0:
            class_preds = preds[mask]
            correct = (class_preds == i).sum().item()
            total = mask.sum().item()
            acc = correct / total
            print(f"  {class_name:12s}: {correct:4d}/{total:4d}  ({acc:.2%})")

    overall_acc = compute_accuracy(preds, labels)
    print(f"\n  总体准确率: {overall_acc:.2%}")


def get_device():
    """获取可用设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def format_time(seconds):
    """格式化时间。"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}时{minutes}分"
