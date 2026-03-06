# TrafficFormer 运行步骤（中文）

本文档只保留“怎么跑”的步骤，按顺序执行即可。

## 1) 环境准备

在项目根目录执行：

```bash
pip install -r requirements.txt
```

> 建议 Python 3.8+。

---

## 2) 预训练数据准备

准备好预训练语料文本（例如 `corpus.txt`）后，执行：

```bash
python3 pre-training/preprocess.py \
  --corpus_path corpus.txt \
  --vocab_path models/encryptd_vocab.txt \
  --seq_length 512 \
  --dataset_path dataset.pt \
  --processes_num 80 \
  --target bertflow
```

运行完成后会得到 `dataset.pt`。

---

## 3) 开始预训练

### 多卡示例

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python3 pre-training/pretrain.py \
  --dataset_path dataset.pt \
  --vocab_path models/encryptd_vocab.txt \
  --output_model_path model.bin \
  --world_size 3 --gpu_ranks 0 1 2 --master_ip tcp://localhost:12345 \
  --total_steps 90000 --save_checkpoint_steps 10000 --batch_size 64 \
  --embedding word_pos_seg --encoder transformer --mask fully_visible --target bertflow
```

### 单卡示例

```bash
CUDA_VISIBLE_DEVICES=0 python3 pre-training/pretrain.py \
  --dataset_path dataset.pt \
  --vocab_path models/encryptd_vocab.txt \
  --output_model_path model.bin \
  --world_size 1 --gpu_ranks 0 \
  --total_steps 90000 --save_checkpoint_steps 10000 --batch_size 64 \
  --embedding word_pos_seg --encoder transformer --mask fully_visible --target bertflow
```

预训练结束后得到模型文件（如 `model.bin`）。

---

## 4) 准备微调数据

你需要准备这 3 个 TSV 文件：

- `train_dataset.tsv`
- `valid_dataset.tsv`
- `test_dataset.tsv`

并保证格式为常见分类数据格式（含标签列与文本列）。

---

## 5) 开始微调与测试

```bash
CUDA_VISIBLE_DEVICES=0 python3 fine-tuning/run_classifier.py \
  --vocab_path models/encryptd_vocab.txt \
  --train_path train_dataset.tsv \
  --dev_path valid_dataset.tsv \
  --test_path test_dataset.tsv \
  --pretrained_model_path model.bin \
  --output_model_path models/finetuned_model.bin \
  --epochs_num 4 --earlystop 4 --batch_size 128 \
  --embedding word_pos_seg --encoder transformer --mask fully_visible \
  --seq_length 320 --learning_rate 6e-5
```

运行中会输出验证/测试指标（Acc、Precision、Recall、F1 等）。

---

## 6) 常见问题

### 6.1 报错 `No available GPUs.`

- 检查是否安装 GPU 版 PyTorch。
- 检查 `CUDA_VISIBLE_DEVICES` 是否正确。
- 检查 `world_size` 和 `gpu_ranks` 是否一致。

### 6.2 显存不足（OOM）

- 降低 `batch_size`。
- 降低 `seq_length`。
- 先用单卡跑通再扩展到多卡。

### 6.3 训练太慢

- 先用小数据集验证流程。
- 适当减少训练步数（`--total_steps`）做快速验证。

---

## 7) 最小可运行顺序（复制即用）

```bash
pip install -r requirements.txt

python3 pre-training/preprocess.py \
  --corpus_path corpus.txt \
  --vocab_path models/encryptd_vocab.txt \
  --seq_length 512 \
  --dataset_path dataset.pt \
  --processes_num 80 \
  --target bertflow

CUDA_VISIBLE_DEVICES=0 python3 pre-training/pretrain.py \
  --dataset_path dataset.pt \
  --vocab_path models/encryptd_vocab.txt \
  --output_model_path model.bin \
  --world_size 1 --gpu_ranks 0 \
  --total_steps 1000 --save_checkpoint_steps 500 --batch_size 16 \
  --embedding word_pos_seg --encoder transformer --mask fully_visible --target bertflow

CUDA_VISIBLE_DEVICES=0 python3 fine-tuning/run_classifier.py \
  --vocab_path models/encryptd_vocab.txt \
  --train_path train_dataset.tsv \
  --dev_path valid_dataset.tsv \
  --test_path test_dataset.tsv \
  --pretrained_model_path model.bin \
  --output_model_path models/finetuned_model.bin \
  --epochs_num 1 --earlystop 1 --batch_size 16 \
  --embedding word_pos_seg --encoder transformer --mask fully_visible \
  --seq_length 320 --learning_rate 6e-5
```
