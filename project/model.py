"""
model.py - ResNet 模型定义
===========================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Optional


class BasicBlock(nn.Module):
    """ResNet 基本残差块。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 升维/降采样
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 不变维/不变采样
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        # f(x) = x
        identity = x # 恒等映射 / 恒等连接

        # F(x) = x'
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # O(x) = F(x) + f(x) = x' + x
        out += identity
        out = F.relu(out, inplace=True)

        return out


class ResNet(L.LightningModule):
    """用于 CIFAR-10 的 ResNet 模型。

    这是一个简化版的 ResNet-26，适配 32x32 分辨率的图像。
    """

    def __init__(
        self,
        num_blocks: int = 3,
        num_classes: int = 10,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = 64
        # 输入shape(batch_size, 3, 32, 32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 输出shape(batch_size, 64, 32, 32)

        self.layer1 = self._make_layer(64, num_blocks, stride=1)
        self.layer2 = self._make_layer(128, num_blocks, stride=2)
        self.layer3 = self._make_layer(256, num_blocks, stride=2)
        self.layer4 = self._make_layer(512, num_blocks, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出shape(batch_size, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes), # 输出shape(batch_size, num_classes)
        )

        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []

    def _make_layer(self, out_channels, num_blocks, stride):
        """创建一个残差层。"""
        strides = [stride] + [1] * (num_blocks - 1) # [stride, 1, 1, 1, ...]
        layers = []
        for stride_val in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride_val))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_preds.append(preds)
        self.train_labels.append(y)

        self.log("train/loss", loss, prog_bar=True, on_step=True)

        return loss

    def on_train_epoch_end(self):
        """计算训练指标。"""
        if self.train_preds:
            all_preds = torch.cat(self.train_preds)
            all_labels = torch.cat(self.train_labels)
            acc = (all_preds == all_labels).float().mean()
            self.log("train/acc", acc, prog_bar=True, on_epoch=True)

            self.train_preds.clear()
            self.train_labels.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_preds.append(preds)
        self.val_labels.append(y)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """计算验证指标。"""
        if self.val_preds:
            all_preds = torch.cat(self.val_preds)
            all_labels = torch.cat(self.val_labels)

            acc = (all_preds == all_labels).float().mean()

            class_accs = []
            for cls in range(self.hparams.num_classes):
                mask = all_labels == cls
                if mask.sum() > 0:
                    cls_acc = (all_preds[mask] == cls).float().mean()
                    class_accs.append(cls_acc)
                else:
                    class_accs.append(torch.tensor(0.0))

            avg_class_acc = torch.tensor(class_accs).mean()

            self.log("val/acc", acc, prog_bar=True)
            self.log("val/avg_class_acc", avg_class_acc)

            self.val_preds.clear()
            self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        """预测步骤。"""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    def configure_optimizers(self):
        """配置优化器和学习率调度器。"""
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
            nesterov=True,
        )

        # 使用合理的默认值，防止 self.trainer 未初始化时出错
        max_epochs = getattr(self.trainer, 'max_epochs', 20) or 20

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
