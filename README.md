# Project 1：用 NumPy 复现 OpenCLIP ViT-B/32 图像编码器

本项目在**不依赖 PyTorch / OpenCLIP** 的前提下，仅用 NumPy 实现 OpenCLIP
ViT-B/32 的图像塔前向计算，输出与课程提供的 FP32 参考结果逐级对齐的
512 维单位向量。模型参数全部由课程提供且固定，本项目**不包含任何训练**。

> 课程 Project 1 代码作业。报告见 `report/main.tex`。

---

## 1. 目录结构

```text
project-finish/
├── image_encoder.py          # 唯一需要实现的文件（6 个 TODO）
├── helpers.py                # 提供的 LayerNorm / GELU / softmax / 单位化
├── preprocess_image.py       # 提供的 resize + center crop + 归一化
├── runtime.py                # 提供的数据/权重加载
├── check.py                  # 提供的逐级数值检查
├── run.py                    # 命令行：图片 -> 512 维输出
├── numpy_homework.py         # 12 道 NumPy 练习（另交）
├── config.json               # 固定模型结构
├── weights/
│   ├── image_encoder.npz     # 152 个视觉张量（课程提供，勿上传 GitHub）
│   └── manifest.json
├── assets/
│   ├── images/               # 10 张示例图
│   ├── images.json
│   └── reference_outputs.npz # 逐级参考输出（勿上传 GitHub）
├── report/
│   ├── main.tex              # 实验报告（Overleaf / XeLaTeX）
│   └── figs/pipeline.pdf
├── ASSIGNMENT.md             # 课程原始题目说明
└── README.md
```

## 2. 环境

- Python 3.9 及以上（开发环境为 Python 3.14.6）
- NumPy（开发环境为 2.4.6）
- Pillow（读取图片）
- 不需要 GPU、PyTorch、OpenCLIP 或联网

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 只包含 `numpy`、`Pillow`、`imageio`。

## 3. 权重与数据

课程原始压缩包已包含：

```text
weights/image_encoder.npz      # 约 326 MB，152 个视觉张量
assets/images/                 # 10 张示例图
assets/reference_outputs.npz   # 逐级 FP32 参考数组
```

把它们放在本项目对应目录即可，代码通过 `runtime.py` 中的
`ROOT = Path(__file__).resolve().parent` 定位，**不依赖当前工作目录**。
仓库不包含上述两个大文件；从课程附件获取后原样放入即可。

## 4. 快速开始

在项目根目录执行：

```bash
# 1) 校验资源：权重哈希、张量名、图片与参考顺序
python check.py --stage assets

# 2) 逐级检查（推荐按顺序排查）
python check.py --stage preprocess
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch

# 3) 一次跑完全部检查
python check.py --stage all

# 4) 对 10 张示例图生成输出
python run.py --output output_tokens.npz

# 5) 对自己的图片生成输出
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

## 5. 预期结果

`python check.py --stage all` 的正常输出：

```text
assets       PASS: 152 matching visual tensors and ten image/reference pairs
preprocess   PASS shape=(10, 3, 224, 224) max_abs=0.000e+00 mean_abs=0.000e+00
patches      PASS shape=(2, 49, 3072)     max_abs=0.000e+00 mean_abs=0.000e+00
tokens       PASS shape=(2, 50, 768)      max_abs=4.578e-05 mean_abs=1.310e-06
block0       PASS shape=(2, 50, 768)      max_abs=6.348e-06 mean_abs=3.280e-07
last_tokens  PASS shape=(2, 50, 768)      max_abs=4.864e-05 mean_abs=2.739e-06
output       PASS shape=(10, 512)         max_abs=4.917e-07 mean_abs=7.999e-08
batch        PASS shape=(3, 512)          max_abs=4.247e-07 mean_abs=7.784e-08
```

容差：输出 `atol=2e-5, rtol=2e-4`；内部 token `atol=2e-4, rtol=2e-4`；
预处理 `atol=2e-6`。`batch` 检查会打乱顺序并重复图片，用于确认实现
没有按位置缓存结果。

`run.py` 的输出是 `float32`、形状 `(B, 512)`，每行 L2 范数为 1：

```python
import numpy as np
d = np.load("output_tokens.npz")
print(d["ids"])                    # 10 个图片 ID，顺序与输入一致
print(d["output_tokens"].shape)    # (10, 512)
print(np.linalg.norm(d["output_tokens"], axis=1))  # 全为 1.0
```

10 张图的完整前向在普通 CPU 上约 **6 秒**。

## 6. 实现要点

只修改 `image_encoder.py`，六个 TODO 的对应关系：

| TODO | 方法 | 关键点 |
| --- | --- | --- |
| 01 | `ImageEncoder.patchify` | `[B,3,224,224] → [B,3,7,32,7,32] → transpose(0,2,4,1,3,5) → [B,49,3072]` |
| 02 | `MultiHeadAttention.split_heads` | `[B,N,768] → [B,N,12,64] → transpose(0,2,1,3) → [B,12,N,64]` |
| 03 | `MultiHeadAttention.__call__` | `Q·Kᵀ/√64 → softmax → ·V → 合并头 → out_proj` |
| 04 | `TransformerBlock.__call__` | 两个残差；`GELU(uW1ᵀ+b1)W2ᵀ+b2` |
| 05 | `ImageEncoder.embed` | patch 投影无 bias；CLS 在 token 0；加 `[50,768]` 位置表 |
| 06 | `ImageEncoder.__call__` | 取 `tokens[:,0]` → `ln_post` → `@ proj[768,512]` → 单位化 |

三个容易写错的地方：

1. **patch 顺序**：直接 `reshape(B,49,3072)` 会把通道和空间交错，`patches`
   检查会得到 `max_abs≈3.94`。必须先拆成 6 维再 transpose。
2. **多头转置**：`k` 是 4 维 `[B,H,N,64]`，只能交换最后两维
   `k.transpose(0,1,3,2)`，不能写成二维的 `k.T`。
3. **权重方向**：除 `visual.proj` 是 `[768,512]`（`[in,out]`）外，其余线性层
   都是 `[out,in]`，要乘 `weight.T`；`visual.proj` **不要转置**。

## 7. 12 道 NumPy 练习

```bash
python numpy_homework.py
```

覆盖 `numpy_homework.py` 中的 NP-E1~E8、NP-M1~M4：数组/切片、逐元素乘、
点乘、夹角、距离、单位化、axis 汇总、广播、batch、批量夹角、Top-k 余弦
检索、距离矩阵、两层前向。零向量、维度不匹配、非法 `k` 都会抛出
`ValueError`；M1 的零向量行返回 `np.nan`。

## 8. 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `assets FAIL: Wrong parameter file` | `weights/image_encoder.npz` 不完整或被改过，重新解压课程附件 |
| `FileNotFoundError: weights/...` | 没有把权重放到 `weights/`，或从错误目录运行（脚本本身按文件位置定位，不要求 cwd） |
| `patches FAIL` | patch 顺序错，检查 6 维 reshape 与 transpose |
| `tokens FAIL` | patch 投影用了 bias，或位置编码加错轴 |
| `output FAIL` 但 `last_tokens PASS` | 取错 token（应取 `[:,0]`）、忘记 `ln_post`，或错误地转置了 `visual.proj` |
| `GELU` 很慢 | `helpers.GELU` 是逐标量 `math.erf` 的教学实现，属正常现象 |

## 9. 提交与仓库

- 代码：`image_encoder.py`
- 复现说明：本文件
- 报告：`report/main.tex`（Overleaf 选 **XeLaTeX** 编译）
- 练习：`numpy_homework.py`

GitHub 仓库（请替换为你的地址）：

```text
https://github.com/xizixuan0103/project1-numpy-encoder
```

**不要提交** `weights/image_encoder.npz` 与 `assets/reference_outputs.npz`，
`.gitignore` 已排除；README 说明了如何从课程附件获取。

## 10. 说明

模型参数、`helpers.py`、`check.py`、`run.py`、`preprocess_image.py` 均为课程
提供。本仓库只包含本人实现的 `image_encoder.py`、`numpy_homework.py`、
复现说明与报告。开发过程中使用 AI 编程助手辅助整理与调试，全部数值结果
均在本地用 `check.py` 复现。



