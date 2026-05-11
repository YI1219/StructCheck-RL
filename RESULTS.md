# StructCheck — MiniCPM-V SFT 实验结果与复现

离线 Slurm + 容器下的多模态 **completion + reason** 监督微调（LoRA）与评估汇总，供论文与后续迭代对照。

> **克隆仓库后如何安装与运行**：见根目录 [`README.md`](README.md)。本文中的路径多为作者侧数据与输出目录示例。

**输出根目录**：`ROOT=/work/home/polyuser6/scripts/dev/outputs/structcheck`  
下文表格中 **Dir** 列为 `ROOT/` 下的子目录名；**全路径** = `ROOT/<Dir>/`。

---

## 1. 任务与指标

- **输入**：现场/渲染图（JSONL 中 `images[0]`）、BIM（`meta`）、点云摘要（`point_cloud_summary`）、可选时间线（`temporal_history`）。
- **输出**：JSON，`completion` ∈ {Not Started, In Progress, Completed, Uncertain}，`reason` 为短英文说明。
- **主指标**：`completion_accuracy`、`completion_macro_f1`（`structcheck.eval.evaluate_predictions`）。
- **Eval 文件**：`/work/home/polyuser6/data/structure/splits_labeled_norm/eval.jsonl`（与训练 split 对应）。

---

## 2. 如何对比数字（必读）

| 要点 | 说明 |
|------|------|
| **`--require-image`** | 为 `yes` 时只评 **71** 条（有有效 `images[0]`）；为 `no` 时常为 **85** 条。**不可混比**。 |
| **指标文件名** | 同一 run 可能有多份 `metrics_*.json`（截断修复、compact 重评等）。表格 **Metrics** 列标明**应引用的 canonical 文件**。 |
| **对齐 vs 旧 adapter** | **当前** `sft_train` 与默认 `run_inference` 在 `Expected JSON:` 前含 **schema hint + `sft_json_tail_reminder()`**。用默认 eval 去评 **`174818` 等早期 run** 会分布外、塌到 **Uncertain**；此类 adapter 需 **`--legacy-sft-prompt`** 或历史 **`metrics_eval_2stage_compact.json`**。 |
| **冠军不可兼得** | **最高 accuracy**（165603）与 **最高 macro-F1**（174818 compact）取舍不同，见 §4。 |

---

## 3. 主表 — 全部实验结果

**列说明**：**Type** = 训练+内置评测（8×GPU 或 1×GPU）或仅 **Re-eval**（已有 adapter 上重跑推理）。**Img** = 评测是否 `--require-image`。**ρ(reason)** = `pred_reason_nonempty_rate`。**Pred 摘要** = 预测类计数简写（NS=Not Started，IP=In Progress，C=Completed，U=Uncertain）。

| Dir | Type | Slurm | GPUs | Steps | Img | N | Acc | Macro-F1 | ρ(reason) | Pred 摘要 | Metrics（canonical） | 备注 |
|-----|------|-------|-----:|------:|-----|--:|----:|---------:|----------:|----------|----------------------|------|
| `exp_sft_1n8g_nogres_20260508_101227` | SFT-8G | — | 8 | 30 | no | 85 | 0.1412 | 0.1566 | 1.00 | IP80 C5 | `metrics_eval_2stage.json` | 步数过低 → 塌 **In Progress** |
| `exp_sft_1n8g_nogres_20260508_130951` | SFT-8G | — | 8 | 800 | **yes** | 71 | 0.6761 | 0.3271 | 1.00 | NS47 IP5 C19 | 同上 | 早期较好 8G 基线 |
| `exp_sft_1n8g_nogres_20260509_150748` | SFT-8G | — | 8 | 800 | **yes** | 71 | 0.7042 | 0.3269 | 1.00 | NS50 IP1 C20 | 同上 | capped oversample |
| `exp_sft_1n8g_nogres_20260509_163417` | SFT-8G | — | 8 | 800 | **yes** | 71 | 0.7042 | 0.3215 | 0.99 | NS49 C22 | 同上 | 保守 Completed rubric |
| `exp_sft_1n8g_nogres_20260509_165603` | SFT-8G | — | 8 | 800 | **yes** | 71 | **0.7324** | 0.3121 | 0.48 | NS59 C12 | **`metrics_eval_2stage_fixlen.json`** | **accuracy 冠军**；IP **recall 0**；默认 metrics 曾因截断失真 → 以 fixlen 为准 |
| `exp_sft_1n8g_nogres_20260509_174818` | SFT-8G | — | 8 | 800 | **yes** | 71 | **0.6901** | **0.3704** | 0.77 | NS48 IP13 C9 U1 | **`metrics_eval_2stage_compact.json`** | **macro-F1 冠军**；job 内默认 metrics 误导 → 以 **compact** 为准 |
| `exp_sft_1n8g_nogres_20260509_190000` | SFT-8G | 3123 | 8 | 800 | **yes** | 71 | 0.6901 | 0.3207 | 1.00 | NS58 IP9 U4 | `metrics_eval_2stage.json` | **预测 C=0**；缺 tail reminder + 左截断 |
| `exp_sft_1n8g_nogres_20260509_192603` | SFT-8G | 3124 | 8 | 800 | **yes** | 71 | 0.0704 | 0.0421 | 0.15 | U65 NS4 C2 | 同上 | **失败**：右截断吃掉 JSON 尾 |
| `exp_sft_1n8g_nogres_20260509_194858` | SFT-8G | 3125 | 8 | 800 | **yes** | 71 | 0.1127 | 0.0660 | 0.17 | U57 NS7 IP4 C3 | 同上 | **失败**：训练/推理提示未对齐（旧 infer） |
| `exp_sft_1n8g_nogres_20260509_202052` | SFT-8G | — | 8 | 800 | **yes** | 71 | 0.0704 | 0.0545 | 0.14 | IP43 U24 NS4 | 同上 | 对齐 infer + 左截断 train；仍差 → 见当时诊断 |
| `exp_balanced_800steps_1514` | SFT-1G | — | 1 | 800 | no | 85 | 0.6588 | 0.3525 | 1.00 | NS59 IP7 C19 | `metrics_eval_2stage.json` | 1GPU balanced；与 71 条 **不可直接比** |
| 同上（Re-eval） | Re-eval | **3172** | — | — | **yes** | 71 | 0.0986 | 0.0562 | **0.00** | U63 NS8 | `metrics_eval_2stage_metrics_realign_v1.json` | Adapter **174818** + **默认对齐** infer → **无效** |
| 同上（Re-eval） | Re-eval | **3173** | — | — | **yes** | 71 | 0.1972 | 0.1044 | 0.03 | U56 NS14 IP1 | `metrics_eval_2stage_metrics_vote3.json` | 同上 + **votes=3**，仍塌 |
| 同上（Re-eval） | Re-eval | **3176** | — | — | **yes** | 71 | 0.5211 | 0.3176 | 0.03 | NS42 U19 C5 IP5 | `metrics_eval_2stage_legacy_v1.json` | **`LEGACY_SFT_PROMPT=1`**；仍低于 compact **0.69/0.37** |

**Slurm 仅 Re-eval 行**：adapter 目录均为 `exp_sft_1n8g_nogres_20260509_174818`；metrics 文件在该目录下。

---

## 4. 冠军与权衡

- **Completion accuracy（71 条、可比集）**：**0.7324** — `165603`（`metrics_eval_2stage_fixlen.json`）。代价：**In Progress** 在预测分布中消失（recall 0），macro-F1 较低，reason 非空率约 0.48。
- **Completion macro-F1（同上）**：**0.3704** — `174818`（**必须**引用 `metrics_eval_2stage_compact.json`）。Accuracy 0.6901，四类更均衡，Completed F1 仍弱于 165603。
- **新增训练/代码改动**（mask、LoRA、votes、extra copies 等）在截至文档更新时 **未产生高于上列 champion 的可比主表条目**；Re-eval 3172–3173 因 adapter/提示不匹配 **不进入冠军比较**。

---

## 5. 已知失效模式（简表）

| 现象 | 可能原因 | 缓解（代码/流程） |
|------|----------|-------------------|
| 全程 **In Progress** | 步数过少 | ≥800 steps 量级 |
| **Uncertain** 主导 | 推理截断无 headroom；或训练截断掉 JSON 尾；或 train/infer 结构不一致 | `prompt_max_len`；训练 **左截断** + **`sft_json_tail_reminder`**；对齐 `run_inference` 与 `sft_train` |
| **Completed=0** | 左截断去掉 rubric 且无尾提醒 | tail reminder + 提高 `MAX_LENGTH` |
| 旧 adapter + 新 eval 塌 | 分布外提示 | **`--legacy-sft-prompt`** 或历史 compact 指标 |
| macro-F1 不稳 | **Uncertain** 在 eval 极少 | 增采真实 Uncertain；固定 `--require-image` 再比 |

---

## 6. 复现（Slurm）

- **集群策略**：**不要**在 `#SBATCH` 或 `srun` 里写 **`--gres`**（见 `scripts/slurm/SITE_POLICY.txt`）。
- **8×GPU 训练+评测**：`scripts/slurm/train_structcheck_sft_1node_8gpu_nogres.sbatch`；作业名建议 `test_s_<num>`（`./scripts/slurm/submit_test_s_sft.sh <num>`）。
- **当前训练默认（摘要）**：compact instruction、`MAX_LENGTH=2048`、训练左截断、`sft_json_tail_reminder`、**`--mask-instruction-labels` 默认开**、LoRA **r=32**（`LORA_R`/`LORA_ALPHA` 可改）、`SFT_EXTRA_ARGS` 默认含 **`--balance-max-factor 8`**、`COMPLETION_EXTRA_COPIES` 默认 **`Completed:2,In Progress:1`**；可选 **`EXTRA_TRAIN_JSONL`**、**`COMPLETION_VOTES`**（推理）。
- **仅 1×GPU 重评**：`scripts/slurm/eval_adapter_1gpu_test_s.sbatch`；旧 adapter 设 **`LEGACY_SFT_PROMPT=1`**。

```bash
# 新 adapter（与当前 sft_train 对齐）
ADAPTER_DIR="$ROOT/<your_exp_dir>" sbatch --job-name=test_s_100 scripts/slurm/eval_adapter_1gpu_test_s.sbatch

# 例如 174818：legacy 提示
ADAPTER_DIR="$ROOT/exp_sft_1n8g_nogres_20260509_174818" \
  LEGACY_SFT_PROMPT=1 PRED_SUFFIX=legacy_eval \
  sbatch --job-name=test_s_103 scripts/slurm/eval_adapter_1gpu_test_s.sbatch
```

日志：`~/logs/test_s/slurm/`、`~/logs/test_s/ranks/`。

### Docker 发版镜像（与 `structcheck-rl.sqsh` 对齐）

- **`Dockerfile`** / 通用安装：仅根目录 **`requirements.txt`**（中文说明 + 全量 pin，约 **107** 条）。安装时使用 **`--extra-index-url https://download.pytorch.org/whl/cu128`**。**`docker/wheels/*.whl`** 在 **`pip install -r requirements.txt` 之后**再装。
- **捕获「跑实验中」多装的包**：训练 sbatch 在 **`OUT_DIR`** 写 **`pip_freeze_llm_*`**；eval 在 **`ADAPTER_DIR`** 写带时间戳的 freeze。与当前 pin 做差：`python3 tools/diff_freeze_vs_lock.py OUT_DIR/pip_freeze_llm_end.txt requirements.txt`，将输出行**追加进 `requirements.txt`** 后 **`docker build`**。
- **从 sqsh 重生依赖文件**（会覆盖 **`requirements.txt`**）：`python3 tools/extract_sqsh_requirements_lock.py /path/to/structcheck-rl.sqsh`
- **构建镜像**：`docker build -t structcheck-rl:release -f Dockerfile .`

---

## 7. 数据与校验

- **标注与扩数据**：`docs/DATA_CONSTRUCTION_GUIDE.md`
- **JSONL 检查**：`PYTHONPATH=src python3 -m structcheck.data.validate_jsonl --jsonl PATH [--require-image] [--check-reason-min-words 10]`

---

## 8. 附录：深入解读（可选）

- **步数**：30 step smoke 与 800 step 对比说明主表；长文分析已并入 §5、§4。
- **165603**：原始 `metrics_eval_2stage.json` 因推理长度曾失真；**论文/对比请用 `metrics_eval_2stage_fixlen.json`**（或与之一致的推理配置）。
- **174818**：Slurm 产出的默认 `metrics_eval_2stage.json` 不可代表该模型；**请用 `metrics_eval_2stage_compact.json`**。
- **硬例挖掘**（注意 eval 泄漏）：`python -m structcheck.data.mine_eval_errors`；示例补充集：`data/structure/splits_labeled_norm/hard_mined_from_174818_compact.jsonl`。
- **推理投票**：`run_inference.py --completion-votes K`；在 **与训练对齐的 adapter** 上评测才有意义。

---

*文档结构更新：合并为单主表（§3），并与 Slurm 监测 job 3172/3173/3176 对齐；若新增 run，请按 §3 增行并更新 §4。*
