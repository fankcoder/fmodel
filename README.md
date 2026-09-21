# Tiny GPT from scratch

一个用于学习 Decoder-only Transformer 训练全流程的约 14M 参数 GPT。

配置固定为：词表 8,000、6 层、`d_model=384`、6 个注意力头、`FFN=1536`、上下文长度 256，并且 Token Embedding 与 LM Head 权重共享。

## 1. 安装

建议在独立的 Conda 环境中安装依赖（当前机器已检测到可用 CUDA）：

```powershell
pip install -r requirements.txt
```

## 2. 获取 TinyStories

从 [TinyStories 数据集](https://huggingface.co/datasets/roneneldan/TinyStories) 下载训练文本后，将纯 UTF-8 文本保存为：

```text
data/raw/tinystories.txt
```

如果有验证集，则保存为 `data/raw/tinystories_valid.txt`。没有验证集时，训练脚本会从训练文本末尾切出 1% 作为验证数据。

也可以用项目内下载器获取官方的 train / validation 文本（约 1.94 GB，下载完成前不会开始训练）：

```powershell
python download_data.py
```

## 3. 训练 BPE tokenizer

```powershell
python tokenizer.py --input data/raw/tinystories.txt --output-dir artifacts/tokenizer --vocab-size 8000
```

## 4. 训练模型

3070 Ti 8GB 的推荐起点：

```powershell
python train.py --train-text data/raw/tinystories.txt --tokenizer artifacts/tokenizer/tinystories.model --batch-size 8 --grad-accum-steps 4 --max-steps 20000
```

完整 TinyStories 不应直接整体加载进内存。正式训练前，先生成内存映射 token 文件：

```powershell
python prepare_data.py --input D:\code\TinyStories-train.txt --tokenizer artifacts/tokenizer/tinystories.model --output-dir artifacts/data
python train.py --train-bin artifacts/data/train.bin --valid-bin artifacts/data/valid.bin --tokenizer artifacts/tokenizer/tinystories.model --batch-size 8 --grad-accum-steps 4 --max-steps 20000
```

显存不足时仅依次降低 `--batch-size`（8 → 4 → 2）。训练输出保存在 `artifacts/checkpoints/`；可以通过 `--resume` 恢复训练。

`last.pt` 是可恢复训练的完整 checkpoint（含 AdamW 状态，因此比权重本身大得多）；`model.pt` 是仅供生成和导出的模型权重，FP32 时约 53MB。

## 5. 生成

```powershell
python generate.py --checkpoint artifacts/checkpoints/model.pt --tokenizer artifacts/tokenizer/tinystories.model --prompt "Once upon a time there was" --max-new-tokens 120 --temperature 0.8 --top-k 50
```

## 6. 导出 ONNX（第二阶段）

```powershell
pip install onnx
python export_onnx.py --checkpoint artifacts/checkpoints/model.pt --output artifacts/tinygpt.onnx
```

这份 ONNX 接受 `int64` 的 `token_ids`，输出 `(batch, sequence, vocab)` logits。INT8 量化应在导出后按目标运行时（ONNX Runtime 或设备厂商编译器）执行，避免保存一个无法实际推理的“伪量化”文件。

## 学习地图

- `model.py`: Q/K/V 投影、缩放点积、因果 Mask、多头合并、残差、LayerNorm、GELU FFN。
- `data.py`: `x = tokens[t:t+256]`，`y = tokens[t+1:t+257]` 的 next-token 监督。
- `train.py`: CrossEntropy、AdamW、FP16 AMP、梯度累积、梯度裁剪、checkpoint。

第一版刻意不使用 RoPE、RMSNorm、SwiGLU、GQA 或 FlashAttention。跑通后再用每个版本的验证 loss 和生成结果做对照实验。
