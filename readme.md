# StructCheck-RL

**StructCheck-RL** 提供面向施工现场/建造场景的多模态 **施工进度（construction progress）** 评估流程：在视觉–语言模型（如 **MiniCPM-V**）上做 **LoRA 监督微调（SFT）**、批量 **推理**，并输出结构化 **JSON 预测**与 **分类指标**（completion + 简短 reason）。

**English:** Multimodal construction-progress assessment — SFT, batched inference, and JSON metrics on top of VLMs.

---

## 目录

- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始：单机训练与评估](#快速开始单机训练与评估)
- [Docker](#docker)
- [Slurm + 容器（可选）](#slurm--容器可选)
- [仓库结构](#仓库结构)
- [依赖与可复现环境](#依赖与可复现环境)
- [实验记录](#实验记录)
- [维护者：从集群镜像刷新 pip 锁](#维护者从集群镜像刷新-pip-锁)
- [许可证](#许可证)

---

## 环境要求

| 项目 | 说明 |
|------|------|
| **Python** | **3.10+**（与参考集群 conda 环境一致；3.11 未在此仓库内做兼容性保证） |
| **系统** | 推荐 **Linux x86_64 + NVIDIA GPU** |
| **CUDA / PyTorch** | `requirements.txt` 针对 **CUDA 12.8** 的 PyTorch 轮子（`cu128` extra index） |
| **数据** | JSONL，字段约定见 [`docs/DATA_CONSTRUCTION_GUIDE.md`](docs/DATA_CONSTRUCTION_GUIDE.md) |
| **基座模型** | 需自备 **MiniCPM-V**（或兼容接口的 VLM）权重目录 |

---

## 安装

在**仓库根目录**执行：

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

依赖只有 **[`requirements.txt`](requirements.txt)** 一份：顶部为中文说明，下方为全量 `包==版本`，可直接 `pip install -r` 建环境。

**检查：**

```bash
python3 -c "import structcheck; import torch; print('structcheck OK', torch.__version__)"
```

**校验 JSONL（可选）：**

```bash
PYTHONPATH=src python3 -m structcheck.data.validate_jsonl --help
```

若系统未安装 `python3-venv`，可用 `apt install python3.10-venv`（Debian/Ubuntu）后再创建虚拟环境。

---

## 快速开始：单机训练与评估

将下面路径换成你的模型与数据路径。

**多卡训练（示例 8 卡）：**

```bash
export PYTHONPATH=src
torchrun --nproc_per_node=8 --nnodes=1 -m structcheck.train.sft_train \
  --model-path /path/to/MiniCPM-V-2_6 \
  --train-jsonl /path/to/train.jsonl \
  --eval-jsonl /path/to/eval.jsonl \
  --output-dir /path/to/out_exp \
  --epochs 1 --max-steps 800 \
  --per-device-batch-size 1 --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 --max-length 2048 \
  --lora-r 32 --lora-alpha 0
```

**推理 + 指标：**

```bash
PYTHONPATH=src python3 -m structcheck.eval.run_inference \
  --base-model-path /path/to/MiniCPM-V-2_6 \
  --adapter-path /path/to/out_exp \
  --input-jsonl /path/to/eval.jsonl \
  --output-jsonl /path/to/out_exp/predictions_eval_2stage.jsonl \
  --require-image --compact-prompt \
  --max-length 2048 --max-new-tokens 128

PYTHONPATH=src python3 -m structcheck.eval.evaluate_predictions \
  --predictions-jsonl /path/to/out_exp/predictions_eval_2stage.jsonl \
  --output-json /path/to/out_exp/metrics_eval_2stage.json
```

各子命令支持 **`--help`**，可查看 LoRA、数据增强、投票推理等全部参数。

---

## Docker

```bash
docker build -t structcheck-rl:local -f Dockerfile .
docker run --gpus all -it -v "$(pwd)":/work/repo structcheck-rl:local bash
# 容器内若挂载了本仓库：cd /work/repo && pip install -e .
```

镜像内通过 **`pip install -r requirements.txt`** 安装完整依赖。若需额外 wheel，见 [`docker/wheels/README.txt`](docker/wheels/README.txt)。

---

## Slurm + 容器（可选）

示例脚本在 [`scripts/slurm/`](scripts/slurm/)，默认假设 **Apptainer/Singularity** 类 **`srun --container-image`** 工作流；Python 解释器默认可在镜像内 **`/opt/conda/envs/llm/bin/python3.10`**，也可用环境变量 **`PY` / `PIP` / `TORCHRUN`** 覆盖。

1. 按集群修改各 `.sbatch` 中的 **`#SBATCH`**（分区、账号、QoS 等）；站点策略示例见 [`scripts/slurm/SITE_POLICY.txt`](scripts/slurm/SITE_POLICY.txt)。
2. 提交前导出路径，例如：

```bash
export REPO_ROOT=/path/to/StructCheck-RL-main   # 或在仓库根目录 sbatch，使用默认 SLURM_SUBMIT_DIR
export CONTAINER_IMAGE=/path/to/structcheck-rl.sqsh
export CONTAINER_MOUNTS="${HOME}:${HOME},/data:/data"
export BASE_MODEL=/path/to/MiniCPM-V-2_6
export TRAIN_JSONL=/path/to/train.jsonl
export EVAL_JSONL=/path/to/eval.jsonl
export OUT_DIR=/path/to/outputs/my_run   # 可选
export LOG_ROOT="${HOME}/logs"
sbatch scripts/slurm/train_structcheck_sft_1node_8gpu_nogres.sbatch
```

仅评估已有 adapter：

```bash
export ADAPTER_DIR=/path/to/sft/output
export BASE_MODEL=/path/to/MiniCPM-V-2_6
export EVAL_JSONL=/path/to/eval.jsonl
export CONTAINER_IMAGE=/path/to/structcheck-rl.sqsh
sbatch scripts/slurm/eval_adapter_1gpu_test_s.sbatch
```

完整环境变量表见 [`scripts/slurm/README.md`](scripts/slurm/README.md)。

---

## 仓库结构

| 路径 | 说明 |
|------|------|
| [`src/structcheck/`](src/structcheck/) | 核心库与 `python -m` 入口（`train` / `eval` / `data`） |
| [`docs/DATA_CONSTRUCTION_GUIDE.md`](docs/DATA_CONSTRUCTION_GUIDE.md) | 数据与标注约定 |
| [`RESULTS.md`](RESULTS.md) | 实验结果与复现说明（文中路径多为示例） |
| [`requirements.txt`](requirements.txt) | **唯一**依赖文件：说明 + 全量 pin；Docker 与本地共用 |
| [`tools/extract_sqsh_requirements_lock.py`](tools/extract_sqsh_requirements_lock.py) | 从参考 `.sqsh` **覆盖重写** `requirements.txt` 中的 pin 列表（先备份） |
| [`tools/diff_freeze_vs_lock.py`](tools/diff_freeze_vs_lock.py) | `pip freeze` 与 `requirements.txt` 对比，列出缺失包 |

---

## 依赖与可复现环境

- **安装**：仅 **`pip install -r requirements.txt`** + PyTorch **`cu128`** 索引（见 [安装](#安装)）。
- **实验里多装的包**：`python3 tools/diff_freeze_vs_lock.py freeze.txt requirements.txt` 查看相对当前文件的增量，将需要的行**手工追加**到 `requirements.txt` 后提交。重新运行 **`extract_sqsh_requirements_lock.py` 会整文件覆盖**，请先备份或改 sqsh 后再 extract。

---

## 实验记录

论文/报告用的跑分与配置摘要见 **[`RESULTS.md`](RESULTS.md)**（其中部分绝对路径为作者环境示例，克隆后请替换为你的路径）。

---

## 维护者：从集群镜像刷新 pip 锁

本机需安装 **`unsquashfs`** 与 **`sqfscat`**（或等价工具）：

```bash
python3 tools/extract_sqsh_requirements_lock.py /path/to/structcheck-rl.sqsh
```

会**覆盖写入**根目录 **`requirements.txt`**（保留固定注释头 + 自 sqsh 解析的全部 pin）。

---

## 许可证

发布到 GitHub 时请在仓库根目录添加 **`LICENSE`** 文件并在此处替换本说明（例如 MIT、Apache-2.0 等）。

---

## 引用

若使用本仓库进行研究，请按需引用你的论文或本项目（可在发表后在此补充 BibTeX）。
