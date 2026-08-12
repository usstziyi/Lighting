
已确认，这些是**库预定义的固定字符串**，不是自定义的。

源码在 [enums.py](file:///Users/usst_ziyi/Programs/trae/DeepL/Lighting/.venv/lib/python3.14/site-packages/torchinfo/enums.py#L24-35)，定义在 `ColumnSettings` 枚举里：

```python
class ColumnSettings(str, Enum):
    KERNEL_SIZE = "kernel_size"
    INPUT_SIZE = "input_size"
    OUTPUT_SIZE = "output_size"
    NUM_PARAMS = "num_params"
    PARAMS_PERCENT = "params_percent"
    MULT_ADDS = "mult_adds"
    TRAINABLE = "trainable"
```

torchinfo 共支持 7 种可用的列名，你可以按需组合：

| 字符串 | 含义 |
|---|---|
| `"input_size"` | 每层输入形状 |
| `"output_size"` | 每层输出形状 |
| `"num_params"` | 每层参数量 |
| `"params_percent"` | 每层参数占比 |
| `"kernel_size"` | 卷积核尺寸 |
| `"mult_adds"` | 乘加运算量 |
| `"trainable"` | 是否可训练 |

`summary()` 内部会用 `ColumnSettings(value)` 校验传入的字符串，传了不认识的列名会直接抛 `ValueError`。所以这些名字要严格照着写（大小写敏感），不能随便自定义。如果你不想记字符串，也可以用枚举本身：`from torchinfo import ColumnSettings`，然后 `col_names=[ColumnSettings.INPUT_SIZE]`，效果一样，还带 IDE 自动补全。