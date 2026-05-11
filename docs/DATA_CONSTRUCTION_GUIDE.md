# StructCheck 数据构造指引

面向 **MiniCPM-V 多模态 SFT**：每条样本 = **现场/渲染图** + **BIM 与点云摘要文本** + **completion / reason 标注**。按此指引扩充数据，可优先改善 **Completed / In Progress** 与 **宏观 F1**。

---

## 1. 单条样本：JSONL 结构

每行一个 JSON 对象（UTF-8）。训练脚本读取字段如下。

### 1.1 必填（当前 pipeline 常用）

| 字段 | 类型 | 说明 |
|------|------|------|
| `element_id` | string | 构件唯一 ID；**切分 train/eval 时按此分组**，同一构件不要同时出现在两边（或按你们的 split 规则整组分配）。 |
| `images` | string[] | 本地**绝对路径**列表；**训练/推理默认使用 `images[0]`**。路径必须在集群上可读。 |
| `meta` | object | BIM 属性；至少保证 `Name`、`IfcType` 等有可读值（见 `formatting.build_bim_text`）。 |
| `target` | object | 见下节。 |

### 1.2 `target` 对象

| 字段 | 类型 | 说明 |
|------|------|------|
| `completion` | string | **必须严格为四选一**（大小写与空格与下列一致）：`Not Started`、`In Progress`、`Completed`、`Uncertain`。 |
| `reason` | string | 英文短说明；建议 **≥10 词**，说明**图像 + 点云/ BIM 证据**如何支持该标签（与 `formatting.completion_rubric()` 一致）。 |
| `stage_label` | string（可选） | 工序/工作包名称，会进入 prompt 的 “Stage / work package” 行。 |

### 1.3 强烈建议（提升效果）

| 字段 | 类型 | 说明 |
|------|------|------|
| `point_cloud_summary` | object | 至少包含 **`summary_line`**（一句话摘要）；**`metrics`** 为 dict，推荐数值键：`coverage`、`coverage_hit`、`voxel_recall`、`voxel_iou`（float）。这些会进入 **PointCloudMetrics** 行，与 rubric 中 “强对齐时倾向 Completed” 一致。 |
| `temporal_history` | array | 历史决策列表；每项建议含 `date`、`completion`。无历史可用 `[]`（会显示为无记录）。 |

### 1.4 其它常见字段（可保留）

`sample_id`、`run_date`、`storey`、`ifc_type`、`meta_path`、`point_cloud_summary_path`、`contour_path` 等可保留作溯源；**当前 `build_instruction` 不依赖全部**，但利于复现与审计。

---

## 2. 标签定义（与训练 rubric 对齐）

标注前请打印或对照 `src/structcheck/data/formatting.py` 中的 **`completion_rubric()`**（会进入完整 instruction）。摘要：

- **Not Started**：目标构件**尚未安装**（空壳、仅龙骨、或画面中**看不到**对应完工实体）。
- **In Progress**：针对**该目标构件**的安装**进行中**（脚手架、临时措施、**明显未完工面**，还不是最终完工形态）。
- **Completed**：**已完工、可对应 BIM 构件类型**的安装实体清晰可见，且与点云指标/图像一致；**仅粗糙预埋/龙骨不要标 Completed**。
- **Uncertain**：遮挡、缺图、点云与图像**冲突**、或信息不足以在另三类中择一；**不要勉强猜**。

**易混淆对（建议多采这些场景的样本）**

1. **Completed vs In Progress**：收口面层、门扇/设备是否到位、是否仍为临时封堵。  
2. **Completed vs Not Started**：点云 IoU 高但**相机未拍到**构件 → 往往应 **Uncertain** 或 **Not Started**（需 reason 写清）。  
3. **In Progress vs Not Started**：仅有邻近工种、非本构件的半成品，不要标成 In Progress。

---

## 3. `reason` 写法建议

- 用 **英文**，1–3 句即可，但建议 **≥10 词**（与 schema hint 一致，减少空 reason）。  
- 必须**可审计**：提到 **图像里看到什么**（或看不到什么）+ **PointCloudMetrics 高/低** 如何支持该标签。  
- 避免复制 rubric 原文；写**本条样本特有**的证据。  
- **Uncertain** 也要写清**缺什么信息**或**矛盾点**。

---

## 4. 图像与点云数据

- **首张图**：保证 `images[0]` **对焦目标区域**、分辨率足够；模糊/过曝样本会降低 Completed 类质量。  
- **多视角**：若一条有 `images[1..]`，当前训练仍主要用第一张；后续可扩展 multi-image，但采集时仍建议保证 **第一张**质量。  
- **路径**：与 Slurm/容器挂载一致（如 `/work/home/...`），避免相对路径导致 `--require-image` 丢样本。  
- **点云指标**：尽量保证 `metrics` 与真实对齐流程一致；**错误指标**会 teach 模型错误关联，损害 Completed / Not Started。

---

## 5. 数据集规模与类别平衡（实操建议）

- **总样本**：在现有基础上，优先增加 **数百条** 高质量样本比盲目上万条低质样本更有效。  
- **类别**：当前 eval 里 **Uncertain** 极少，macro-F1 会抖；若业务允许，**有意增加少量真实 Uncertain**（遮挡/冲突）。  
- **Completed / In Progress**：你们是主要瓶颈；建议 **Completed 与 In Progress 的条数不低于 Not Started 的 15–25%**（可按项目再调），或通过 **更多日期/视角** 覆盖边界案例。  
- **划分**：**按 `element_id`（或 `element_id + 项目`）分组** 再划分 train/eval，避免同一构件泄漏到验证集。  
- **时间**：同一构件多 `run_date` 可做成多条样本，但划分时仍要注意 **泄漏**（eval 应用未见过的构件或未见过的阶段策略需统一）。

---

## 6. 采集优先级清单（按 ROI 排序）

1. **Completed**：点云指标高 + 图像清晰显示完工构件（多种 IfcType）。  
2. **In Progress**：同一 IfcType 的**明确施工中**状态（与 Completed 成对更佳）。  
3. **Not Started**：早期现场 + 点云低覆盖/低 IoU，且图像无完工实体。  
4. **Uncertain**：刻意收集遮挡、夜间、仅局部、BIM 与扫描明显不一致。  
5. **硬例**：人工在 **Completed/In Progress** 边界上犹豫的样本（先标 Uncertain 或双人 adjudication）。

---

## 7. 与训练脚本的衔接

- 合并新数据：生成新的 `train.jsonl` / `eval.jsonl`（或追加后重新 split）。  
- 训练：`structcheck.train.sft_train --train-jsonl ... --eval-jsonl ...`（见 `RESULTS.md`）。  
- 可选：**错题挖掘** `python -m structcheck.data.mine_eval_errors` 生成补充 JSONL，再用 `--extra-train-jsonl`（注意 **eval 泄漏**，仅内部调参或把 ID 移回正式 train）。  
- 校验：使用仓库内 **`python -m structcheck.data.validate_jsonl`**（见下文）在合并后跑一遍。

---

## 8. 版本与提示词

当前默认 SFT/infer 使用 **`build_response_schema_hint()` + `sft_json_tail_reminder()`**；新采集的标签与 reason 应与 **`completion_rubric()`** 一致，避免旧口径与新训练冲突。

若需把本指引同步给标注外包，可附带 **`formatting.py` 中 `completion_rubric` 全文** 与 **四选一标签拼写**。
