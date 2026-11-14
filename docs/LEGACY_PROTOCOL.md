# 保留的实验协议（当前安装及CLI默认值见主README）

这是一个与原始提交材料完全分离的、确定性的、可断点续跑的命令行运行器。它只读取原数据，所有缓存、水印数据和结果均写入新的 `--output-dir`。

## 覆盖范围

- RQ1：Synthetic、Covertype、HOG、Housing 的原始/水印/扰动数据检测；Synthetic 和 Housing 各 400 个邻近数据集的 ROC/AUC。
- RQ2：Covertype/XGBoost、HOG/Random Forest、Housing/Linear Regression、Zoo/Random Forest 的共享干净测试集效用对比。
- RQ3：Covertype、HOG、Housing、Zoo 的 alteration；Covertype insertion/deletion 的 2/3 字段匹配检测；Iris `k=3` 匿名化和可审计的合成攻击。
- RQ5：论文取值的 `p`、`n`、`gamma` sweeps，同时记录攻击后 z-score 和 Logistic Regression 效用。

RQ4 对比方法和 HoloClean 不在本运行器范围内。

## 显式CUDA配置示例（发布验证未覆盖GPU）

```bash
cd /path/to/tabularmark_full_runner
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python run_full.py \
  --data-dir /path/to/TabularMark/datasets \
  --output-dir /path/to/tabularmark-full-results \
  --device cuda \
  --gpu-id 0,1
```

公开版本`--device`默认为`cpu`，`--gpu-id`默认为`0`；多卡需要显式指定。Covertype 的 XGBoost fit 会按任务标识稳定地分配到这两张卡；单次 XGBoost fit 使用一张卡。如果 CUDA/XGBoost 运行失败，该 fit 会自动在 CPU 上重试，并把完整的回退原因写入阶段 CSV/JSON，不会静默回退。其他模型按论文协议在 CPU 上运行。

## 数据发现和列名

运行器会递归搜索 `--data-dir`，优先读取已有的原始 CSV，并排除 `watermarked/`、`alteration/`、`perturbed/` 等中间目录。目标列和水印列按数据集别名动态解析；出现缺列或重名列时会明确报错，不会猜测。同一文件名优先级下发现多个候选时，运行器会比对 SHA-256：内容不同则拒绝运行并列出冲突路径/哈希，内容完全相同才按稳定路径顺序选择一份。每个阶段 JSON 保存实际数据路径、SHA-256、shape、最终列解析结果和 `source_resolution`候选仲裁记录。

显式指定`--allow-downloads`时允许把缺失的公开数据缓存到 `OUTPUT/input_cache/`：Covertype 来自 scikit-learn 的 `fetch_covtype`，Zoo 来自 UCI，Iris 来自 scikit-learn，HOG 的后备方案为记录了参数的 digits+HOG 生成。严格离线时使用 `--no-allow-downloads`；此时缺数据会使相应阶段失败并写入 manifest。

## 断点续跑

`OUTPUT/manifest.json` 记录每个阶段的 `pending/running/completed/failed`、尝试次数、错误堆栈、产物大小和 SHA-256。重新执行同一命令时，只有“状态完成且所有产物哈希正确”的阶段才会跳过。

```bash
# 只跑指定阶段
python run_full.py --stages rq1_detectability rq1_roc

# 强制重跑一个阶段
python run_full.py --force-stage rq3_alteration

# 某阶段失败后仍继续其他独立阶段
python run_full.py --keep-going
```

如果计算参数改变，运行器会拒绝混用旧 manifest；请指定新的 `--output-dir`。这避免了把不同参数的 CSV 误当为同一次实验。

## 阶段与产物

| 阶段 | 主要产物 |
|---|---|
| `rq1_detectability` | `rq1_detectability.csv/.json` |
| `rq1_roc` | `rq1_roc_neighbors.csv`, `rq1_roc_curve.csv`, `rq1_roc.json` |
| `rq2_utility` | `rq2_utility.csv/.json` |
| `rq3_alteration` | `rq3_alteration.csv/.json` |
| `rq3_row_attacks` | `rq3_row_attacks.csv/.json` |
| `rq3_iris_attacks` | `rq3_iris_attacks.csv/.json` |
| `rq5_sweeps` | `rq5_sweeps.csv/.json` |

共用的水印数据和密钥记录位于 `OUTPUT/artifacts/`，环境、依赖版本和 `nvidia-smi` 快照位于 `OUTPUT/environment.json`。

### 密钥安全

`OUTPUT/artifacts/keys/*.json` 包含完整的 `indices` 和 `divide_seeds`，是私密密钥材料。运行器写出后会强制设为 `0600`（仅所有者可读写）。**不应将这些文件公开、上传到公共仓库或同步到公共共享目录**；对外分发结果前应排除整个 `artifacts/keys/` 目录。

## 快速验证

```bash
python -m unittest discover -s tests -v

# 用真实数据做低成本 smoke run；不会等价于全量结果
python run_full.py --quick --stages rq1_detectability rq1_roc --device cpu
```

## 可复现口径与限制

- 嵌入/检测核心从本工作区独立校正版 the independent local reproduction 复制并注明来源，保留“所有 key-cell 修改在同一深拷贝累积”的修复；本目录不导入或修改原提交。
- 分类域严格复现官方 notebook 的分区：绿色类别数为 `floor(类别数 × nominal_gamma)`。因此类别数为奇数时，z-score 的零假设概率使用实际可实现的 `effective_gamma = 绿色类别数 / 类别数`，而不是直接使用请求值；例如 Covertype 的 7 类、`nominal_gamma=0.5` 对应 `effective_gamma=3/7`。所有检测 CSV/JSON 都显式记录 `nominal_gamma` 和 `effective_gamma`；为兼容旧结果，`gamma` 列仍表示 `nominal_gamma`。
- Covertype 匹配采用公开 notebook 实际执行的精确复合字段值，并分别报告 2/3 字段；论文正文称其为 MSB 匹配，因此 JSON 明确记录这一区别。删除后未匹配的 key cell 按非绿色计入原始 `n` 的 z 统计量。
- Iris 合成攻击采用可运行、可审计的类别条件多元高斯生成器。论文未公开足以进行比特级复刻的生成器配置；合成记录与原记录没有一一对应，因而非盲检测的 z-score 被明确记为未定义，而不是伪造一个检测值。
- 随机源、划分和模型 seed 都是固定的；RQ2 同一数据集、同一 repeat 的 original/watermarked 共享完全相同的模型 seed，保证成对比较不混入模型随机性。相同软件/硬件栈结果可重复；不同 CUDA/XGBoost 驱动组合仍可能存在末位浮点差异，`environment.json` 用于追溯。
- Housing 的 Linear Regression 效用评估与官方 notebook 一致：先使用共享索引划分 train/test，再分别对模型实际使用的全部 predictors + `MEDV` 做 complete-case `dropna`，不做中位数填补。该特例仅用于 Housing regression；其他数据集/模型的缺失处理保持不变。RQ2/RQ3 的 CSV 记录每次 fit 的输入、有效和删除行数，JSON 的 `housing_utility_missing_data` 元数据同时记录策略、complete-case 列与逐 fit 样本数。
