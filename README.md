# RFDETR-FallGuard

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![QA](https://img.shields.io/badge/QA-107%20passed-brightgreen)
![GPU](https://img.shields.io/badge/validated-RTX%205070%20Ti-76B900?logo=nvidia&logoColor=white)
![Status](https://img.shields.io/badge/status-研究验证中-orange)

RFDETR-FallGuard 是一个面向毕业设计和可复现实验的跌倒检测级联系统。项目不仅判断单帧中
是否有人倒地，而是将姿态检测、人物跟踪、时序确认、多图语义复核和告警管理组成一条
可审计的完整链路。

> 当前阶段：已在 RTX 5070 Ti 上跑通 RF-DETR Small/Nano → ByteTrack → Temporal →
> Event → Qwen3.5-4B → Alert，正在进入人工标注和三折人物隔离验证。仓库中的
> 小样本指标只用于证明级联工程链路已跑通，不是论文最终准确率。

## 目录

- [项目能做什么](#项目能做什么)
- [系统架构](#系统架构)
- [最新进展](#最新进展)
- [快速开始](#快速开始)
- [真实模型与数据](#真实模型与数据)
- [运行级联验证](#运行级联验证)
- [下一步工作](#下一步工作)
- [项目结构](#项目结构)
- [实验边界与隐私](#实验边界与隐私)

## 项目能做什么

- 使用微调后的 [RF-DETR](https://github.com/roboflow/rf-detr) Nano/Small 检测
  `fallen`、`lying`、`sitting` 和 `standing` 四类人体姿态。
- 使用 ByteTrack 保持人物 ID，并保留每个轨迹的姿态、置信度、位置和时间历史。
- 结合框宽高比、中心点位移、垂直速度和持续时间进行 Temporal 时序确认。
- 为每个候选事件提取“发生前—发生中—发生后”三张人物裁剪图。
- 使用本地 [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) 做多图零样本语义复核，
  区分意外跌倒与主动坐下、弯腰、躺下等日常动作。
- 使用严格 JSON Schema 保存结论、置信度、风险等级、原因、耗时和调用状态。
- 支持 Mock、OpenAI、DeepSeek 和 Local Qwen 语义供应器；本地模型不需要 API Key。
- 提供 Gradio 界面、命令行工具、可复现配置、模型/数据哈希和分阶段评测报告。

## 系统架构

```text
视频 / 摄像头片段
        │
        ▼
RF-DETR 四类姿态检测
        │
        ▼
ByteTrack 人物身份关联
        │
        ▼
Temporal 时序状态机
        │
        ▼
Event 候选事件 + 前/中/后关键帧
        │
        ▼
Qwen3.5-4B 多图语义复核
        │
        ▼
AlertManager 告警决策 + 可审计日志
```

语义模型只返回建议，最终是否告警由应用层 `AlertManager` 决定，避免将业务权限
交给外部模型。详细边界见 [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) 和
[`docs/architecture.md`](docs/architecture.md)。

## 最新进展

实验日期：**2026-08-12**；验证主机：**NVIDIA GeForce RTX 5070 Ti 16 GB**。

### 1. 前端漏检分阶段诊断

对已打开的 Figshare Fall29 锁定集中 64 个已知漏检视频进行了仅诊断归因：

- 50/64 已进入 Temporal 候选，但未被确认为事件。
- 14/64 没有进入 `suspected` 状态。
- 63/64 在生产置信度下已有 `fallen/lying` 检测；剩余 1 个在 0.05 诊断阈值下也有输出。

因此当前主要问题是 **Temporal 确认规则过严**，而不是 RF-DETR 完全看不到人物。
这些数据已被打开，今后只能用于错误分析，不得继续调参或用作最终测试集。

### 2. 新的非泄漏验证协议

[`configs/validation/gmdcsa24_recovery_cv_v2.yaml`](configs/validation/gmdcsa24_recovery_cv_v2.yaml)
将 GMDCSA-24 按人物划分：

- Subjects 1–3：三折人物隔离交叉验证。
- Subject 4：保持锁定，在模型和阈值冻结前不允许打开。
- Small 是论文主模型，Nano 是轻量化对照。

### 3. 高召回前端小样本筛查

| 模型 | 分区 | 视频数 | TP | FP | FN | TN | 召回率 | 特异度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | Subjects 1–2 开发 | 8 | 4 | 3 | 0 | 1 | 1.00 | 0.25 |
| Small | Subject 3 筛查 | 4 | 2 | 1 | 0 | 1 | 1.00 | 0.50 |
| Nano | Subjects 1–2 开发 | 8 | 4 | 3 | 0 | 1 | 1.00 | 0.25 |
| Nano | Subject 3 筛查 | 4 | 2 | 0 | 0 | 2 | 1.00 | 1.00 |

前端刻意偏向高召回，允许一部分 ADL 进入后续语义复核。当前 `0.4` 是候选筛查阈值，
**不是已确定的论文正式阈值**。

### 4. Qwen3.5-4B 本地多图零样本复核

- 模型：`Qwen/Qwen3.5-4B`
- 固定 revision：`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- 输入：同一事件的 before / during / after 三张人物图。
- 每张关键帧都重新运行检测器，不复用旧边界框。
- Small：16/16 次调用成功且 JSON 结构有效，平均约 13.68 秒/事件。
- Nano：14/14 次调用成功且 JSON 结构有效，平均约 12.95 秒/事件。
- 两条 12 视频完整级联在弱标签上均为 6 TP、0 FP、0 FN、6 TN。

最后一项只是小样本工程观察，不得表述为“模型准确率 100%”。完整记录见
[`docs/experiment_status_20260812.md`](docs/experiment_status_20260812.md)。

### 5. 工程质量

RTX 5070 Ti Windows 环境的最新回归结果：

```text
ruff format --check .   117 files already formatted
ruff check .            All checks passed
mypy src scripts        75 source files, 0 issues
pytest -q                107 passed, 2 deselected, 1 warning
```

2 个默认不运行的测试是可能产生费用的真实 API 集成测试。唯一警告来自
`supervision==0.30.0` 对 ByteTrack 旧入口的弃用预告，当前功能仍通过真实集成测试。

## 快速开始

### Linux / macOS

```bash
git clone git@github.com:Aidenwu0209/RFDETR-FallGuard.git
cd RFDETR-FallGuard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

### Windows PowerShell

```powershell
git clone git@github.com:Aidenwu0209/RFDETR-FallGuard.git
Set-Location RFDETR-FallGuard
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

按需安装真实推理、跟踪、界面和本地多模态依赖：

```bash
python -m pip install -e '.[rfdetr,tracking,ui,local-vlm]'
```

RF-DETR 训练依赖单独安装：

```bash
python -m pip install 'rfdetr[train]==1.9.1'
```

### 无 GPU/无权重的快速链路

```bash
python scripts/run_pipeline.py \
  --mode mock \
  --config configs/profiles/development.yaml \
  --output-dir artifacts/mock-run
```

该命令只验证跟踪、Temporal、Event、关键帧、Mock 语义和告警日志的代码链路，所有输出
都会标记为 `MOCK`，不能用于科研指标。

## 真实模型与数据

仓库不包含数据集、视频、检测器权重、Qwen 权重、API Key 或生成的实验报告。请将资产
放入以下目录：

```text
data/raw/                         原始数据，不覆盖
data/processed/                   规范化后的派生数据
weights/official/                 RF-DETR 官方预训练权重
checkpoints/nano/                 Nano 姿态微调 checkpoint
checkpoints/small/                Small 姿态微调 checkpoint
weights/vlm/Qwen3.5-4B/           本地 Qwen3.5-4B
artifacts/                        评测、诊断、关键帧和告警输出
```

### 规范化 Fallen Person COCO 导出

```bash
python scripts/normalize_fallen_person.py \
  --source-dir data/raw/fallen-person \
  --output-dir data/processed/fallen-person/dataset \
  --source-archive 'data/raw/fallen-person-archive/Fallen Person.v1i.coco.zip'

python scripts/prepare_fallen_person.py \
  --dataset-dir data/processed/fallen-person/dataset \
  --output-dir data/processed/fallen-person
```

规范化过程保留原始导出，记录类别映射和 SHA-256，并在派生副本上进行图像、注释、
边界框和跨集重复审计。

### 下载固定版本的 Qwen3.5-4B

```bash
python scripts/download_qwen35.py \
  --repo-id Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir weights/vlm/Qwen3.5-4B
```

下载工具会生成 `download-manifest.json`，记录每个文件的大小和 SHA-256，避免在不同主机上
误用不同权重。

## 运行级联验证

### 1. 生成 Subjects 1–3 三折隔离清单

```bash
python scripts/prepare_gmdcsa24_recovery_cv.py \
  --manifest data/processed/gmdcsa24/manifest.json \
  --output-dir data/processed/gmdcsa24-recovery-cv-v2
```

### 2. 运行姿态 → ByteTrack → Temporal → Event

```bash
python scripts/validate_grouped_pipeline.py \
  --config configs/validation/gmdcsa24_semantic_high_recall_small.yaml \
  --manifest data/processed/gmdcsa24-recovery-cv-v2/manifest-fold-s3.json \
  --dataset-root data/raw/gmdcsa24/extracted/ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-d3edb5d \
  --weights checkpoints/small/checkpoint_best_total.pth \
  --model-variant small \
  --partition threshold_development \
  --all-videos \
  --output-json artifacts/validation/fold-s3-small-development.json
```

Nano 对照实验仅替换配置、权重和 `--model-variant nano`。正式折内评估必须使用
`--all-videos`，不能只运行小样本。

### 3. 生成语义候选并运行本地 Qwen

```bash
python scripts/prepare_semantic_candidate_bundle.py \
  --report artifacts/validation/fold-s3-small-development.json \
  --manifest data/processed/gmdcsa24-recovery-cv-v2/manifest-fold-s3.json \
  --dataset-root data/raw/gmdcsa24/extracted/ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-d3edb5d \
  --config configs/validation/gmdcsa24_semantic_high_recall_small.yaml \
  --weights checkpoints/small/checkpoint_best_total.pth \
  --output-dir artifacts/semantic/fold-s3-small

python scripts/evaluate_local_qwen_zero_shot.py \
  --candidate-manifest artifacts/semantic/fold-s3-small/semantic-candidate-manifest.json \
  --model-path weights/vlm/Qwen3.5-4B \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --output-json artifacts/semantic/fold-s3-small/qwen-zero-shot.json
```

### 4. 界面演示

```bash
python -m pip install -e '.[ui]'
python app/gradio_app.py --host 127.0.0.1 --port 7860
python scripts/smoke_gradio.py
```

当前界面支持上传视频和摄像头录制的有限片段，不声称已实现 24 小时连续实时监控。

## 下一步工作

按以下顺序执行，不要提前打开 Subject 4：

- [ ] **人工事件标注**：审核 Small 16 个、Nano 14 个语义候选及三张关键帧。
- [ ] **全量三折评估**：Small 和 Nano 都运行 Subjects 1–3 的全部视频，不允许人物/视频泄漏。
- [ ] **冻结模型与阈值**：按折均事件 F1、最低折召回率、误报数和最差折特异度选择唯一方案。
- [ ] **决定是否 QLoRA**：只有零样本 Qwen 出现稳定、可重复的语义错误时才进入 Linux/WSL QLoRA。
- [ ] **锁定测试**：生成阈值确认文件后，Subject 4 只运行一次。
- [ ] **论文正式报告**：报告每阶段数量、事件级指标、置信区间、语义有效率、延迟和失败案例。

**当前 QLoRA 状态：`DEFERRED_NOT_JUSTIFIED`。** 如果人工标注和全量交叉验证表明零样本已足够，
就保留零样本 Qwen3.5-4B，不为了形式上的“微调”增加无依据的复杂度。

## 项目结构

```text
RFDETR-FallGuard/
├── app/                  Gradio 界面
├── configs/              开发、实验、验证和 QLoRA 配置
├── datasets/             数据集下载/格式说明，不包含数据本体
├── docs/                 架构、时序、隐私、评测和实验记录
├── scripts/              推理、训练、审计、验证和评测入口
├── src/fallguard/
│   ├── detection/       RF-DETR 适配层
│   ├── tracking/        ByteTrack 与轨迹管理
│   ├── temporal/        时序特征与状态机
│   ├── events/          事件生命周期与关键帧
│   ├── semantic/        Mock/OpenAI/DeepSeek/Local Qwen
│   └── evaluation/      检测、事件和部署指标
├── tests/                单元与集成测试
├── BLOCKERS.md           外部阻塞项
├── IMPLEMENTATION_STATUS.md
└── pyproject.toml
```

## 实验边界与隐私

- 官方 COCO 预训练权重只能证明“检测到 person”，不能直接推导跌倒。
- `posture_multiclass` 必须使用与四类元数据一致的姿态微调 checkpoint。
- Fallen Person 数据存在跨集近重复，其检测指标只能作为工程训练证据。
- 已打开的 Fall29 结果只作诊断，不得再用于阈值优化。
- 当前数据只有视频级标签，没有经人工确认的跌倒起始时间，因此不报告正式检测延迟。
- 云端图像调用需要配置允许、每次请求的用户同意，以及显式的付费测试开关。
- API Key 不写入报告、界面或 Git；本地 Qwen 全程无云端图像传输。
- 系统是研究原型，不是医疗诊断或经认证的安防产品。

## 常用质量命令

```bash
ruff format --check .
ruff check .
mypy src scripts
pytest -q
python -m compileall -q src app scripts tests datasets
python scripts/check_environment.py
```

真实云端集成测试默认排除，只有在明确接受可能产生费用后才能运行：

```bash
RUN_PAID_API_INTEGRATION_TESTS=1 python -m pytest -m api
```

## 更多文档

- [`docs/experiment_status_20260812.md`](docs/experiment_status_20260812.md)：最新实验结果和 QLoRA 决策。
- [`docs/architecture.md`](docs/architecture.md)：系统模块和依赖边界。
- [`docs/temporal_design.md`](docs/temporal_design.md)：坐标、速度、平滑、超时和状态转移。
- [`docs/semantic_review.md`](docs/semantic_review.md)：语义供应器、回退、隐私和告警权限。
- [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md)：事件匹配和指标可用性。
- [`docs/qlora.md`](docs/qlora.md)：QLoRA 数据清单、干跑和执行边界。
- [`docs/privacy.md`](docs/privacy.md)：图像保留、日志和云端同意机制。
- [`datasets/README.md`](datasets/README.md)：外部数据集要求。

## 上游与数据来源

- [Roboflow RF-DETR](https://github.com/roboflow/rf-detr)
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [GMDCSA-24 v2.0](https://doi.org/10.5281/zenodo.12921216)

如果你只想继续当前论文实验，请从「[下一步工作](#下一步工作)」开始，不要绕过人工标注、
三折验证和 Subject 4 锁定门。
