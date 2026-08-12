查到了，本机安装的是 **Lightning 2.6.5**。`precision` 参数的合法选项来自 [precision.py](file:///Users/usst_ziyi/Programs/trae/DeepL/Lighting/.venv/lib/python3.14/site-packages/lightning/fabric/plugins/precision/precision.py#L22-L35)。

## 合法选项（字符串形式）

| 取值 | 含义 |
|---|---|
| `"32-true"` | 纯 FP32（默认） |
| `"64-true"` | 纯 FP64（双精度） |
| `"16-true"` | 纯 FP16（模型、计算全用 half，可能溢出） |
| `"16-mixed"` | FP16 混合精度（AMP，推荐） |
| `"bf16-true"` | 纯 BF16 |
| `"bf16-mixed"` | BF16 混合精度（AMP） |
| `"transformer-engine"` | 英伟达 Transformer Engine（需 TE 库） |
| `"transformer-engine-float16"` | TE + FP16 |

另有两个已废弃的简写：整数 `16`（等价 `16-mixed`）和字符串 `"bf16"`（等价 `bf16-mixed`），以及 `32`→`32-true`、`64`→`64-true`。

## 你选中的这行其实有个 Bug ⚠️

```python
precision="32-mix" if has_gpu else "32-true",
```

`"32-mix"` **不是合法取值**。我刚实测验证，会直接抛错：

```
ValueError: Precision '32-mix' is invalid. Allowed precision values: ('transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true')
```

原因：混合精度（mixed）指的是 **16 位 + 32 位混用**（AMP），不存在"32 位混合精度"这个概念。这段代码的意图明显是**有 GPU 时想用 FP16 混合精度**，所以：

- 正确写法应该是 `precision="16-mixed"`（GPU 上才有效）
- 或者如果就是想用纯 FP32：`precision="32-true"`（无 GPU 分支的写法没错）

同一文件里 [第 216 行](file:///Users/usst_ziyi/Programs/trae/DeepL/Lighting/lessons/lesson_05_advanced.py#L216) 也是同样的 `"32-mix"` 问题，而 [第 254 行](file:///Users/usst_ziyi/Programs/trae/DeepL/Lighting/lessons/lesson_05_advanced.py#L254) 用的 `"16-mixed"` 才是正确的。

需要我把这几处 `"32-mix"` 修正为 `"16-mixed"` 吗？