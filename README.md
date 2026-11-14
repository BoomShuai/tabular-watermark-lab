# Tabular Watermark Lab · 表格水印实验平台

[![CI](https://github.com/BoomShuai/tabular-watermark-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/BoomShuai/tabular-watermark-lab/actions/workflows/ci.yml)

独立整理的TabularMark方法复现与鲁棒性实验工具，包含主运行器、独立对比扩展和Notebook静态审计。保留确定性随机种子、输入仲裁、数据指纹、阶段清单、断点续跑及密钥权限处理；公开版本只带代码和合成样例生成逻辑。

这不是原作者官方实现，也不宣称方法为本仓库原创。方法来源和源码发布边界见[ATTRIBUTION.md](ATTRIBUTION.md)。

## 离线跑通

Python3.10+，推荐3.12：

```bash
python -m pip install -e .
python -m tabularmark_runner.demo --output private/demo
python -m tabularmark_runner --data-dir private/demo --output-dir outputs/smoke --quick --stages rq1_detectability rq1_roc --device cpu --workers 1
```

安装后也可用`tabular-watermark-demo`和`tabular-watermark`命令。默认CPU、默认禁止下载。演示CSV是虚构fixture，不是原始Covertype/HOG/Housing/Zoo，也不能作为论文效果复现成绩。Iris后备输入可来自scikit-learn自带数据，不需要联网。

## 保留的功能

| 模块 | 能力 |
| --- | --- |
| 主运行器 | RQ1可检测性/ROC、RQ2下游效用、RQ3修改/插入/删除/匿名化/合成攻击、RQ5参数扫描 |
| 对比扩展 | 三种方案的独立实验实现，平衡子集、共享嵌套攻击计划、0–100% alteration、可选XGBoost效用 |
| 静态审计 | 扫描用户本地Notebook/Python单元的输入与历史输出，缺输入标BLOCKED，旧输出不算本轮实测 |
| 运行管理 | 输入CSV同优先级哈希仲裁、模型seed成对复用、阶段状态、产物SHA-256、验证后续跑、跨平台进程锁 |

相同命令再次运行，只有完成且产物哈希正确的阶段会跳过；配置改变时拒绝混用已有清单。`--force-stage`显式重算指定阶段。详细实验口径与原协议差异见[保留协议](docs/LEGACY_PROTOCOL.md)和[实现边界](docs/METHODS.md)。

## 独立对比扩展

```bash
python scripts/run_covertype_compare.py --input private/demo/cover_type_with_columns.csv --output-dir outputs/compare --per-class 20 --n-key-cells 40 --watermark-length 20 --semmark-votes-per-bit 2 --alteration-percentages 0:100:50 --attack-repeats 1 --skip-ml
```

使用自己准备的真实数据时按原协议选择规模，核对数据来源和许可。`--skip-ml`明确记录未执行效用模型，不填虚值；部分位可能缺证据，输出保留undecidable诊断。

可选CPU/CUDA实验依赖：

```bash
python -m pip install -e '.[ml]'
```

可用`tabular-compare`命令。CUDA为显式选择，失败后CPU回退会记录原因；发布验证不包含真实GPU全量运行。scikit-image用于显式允许后备数据生成时的HOG适配。

## Notebook审计

```bash
python scripts/audit_official_notebooks.py --official-root private/upstream --json-output outputs/audit.json --csv-output outputs/audit.csv
```

或用`tabular-audit`。本仓库不打包上游源码；由用户自行准备本地输入。审计不执行cell，输出必须是审计树外的新文件，不用execution count或已保存输出冒充本次完成。

## 密钥与数据

主运行器的`outputs/.../artifacts/keys/`和对比扩展的`watermark_state_*.json`包含可用于检测的私密状态。Unix以0600写入，Windows需依赖账户/目录ACL。演示随机种子用于复现，不具备密码学安全性。不要把真实密钥、watermarked CSV或整份运行目录公开。

环境和manifest会记录本机路径与输入元数据，方便本地审计；这些是私有运行产物，不是可直接公开的报告。公开仓库忽略数据、权重、模型、输出、历史报告与私钥材料。

## 验证

```bash
python -m unittest discover -s tests -v
python scripts/privacy_check.py --history
```

测试验证算法往返、有效gamma、行匹配、数据仲裁、种子配对、恢复与哈希、对比确定性以及静态审计；可选依赖安装后增加XGBoost设备配置检查。测试与合成烟雾运行不等于所有研究问题的全量复现。

独立项目代码与自写文档按[MIT](LICENSE)发布，不重新授权上游仓库、论文或数据。[来源说明](ATTRIBUTION.md) · [隐私边界](PRIVACY.md) · [贡献说明](CONTRIBUTING.md)
