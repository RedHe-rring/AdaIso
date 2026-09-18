# AdaIso 研究与实验记录入口

本目录保存跨代码版本仍需长期查阅的计划、方法说明、实验清单和结论。
代码可以回退或切换分支，实验记录应始终通过明确的 Git commit 与代码对应。

## 工作规范

- [版本与实验工作流](versioning_and_experiment_workflow.md)
- [实验清单模板](experiment_manifest.template.yaml)
- [2026-09-18 工作区盘点](workspace_inventory_2026-09-18.md)
- [本地网格数据清单](local_mesh_dataset_manifest.csv)

根目录的 `AGENTS.md` 是修改代码和提交前应阅读的简版规则；完整解释以
版本与实验工作流为准。

## 版本记录

| 主题 | 记录 | 对应代码 |
|---|---|---|
| 论文方法整理 | [Lv, 2024, AdaIso](Lv,2024,adaiso.md) | 文献说明 |
| 复现方法计划 | [AdaIso 复现方法计划](Lv,2024,adaiso-repro-method-plan.md) | 多版本 |
| C++ T0-T10 结果 | [C++ T0-T10 结果](Lv,2024,adaiso-cpp-t0-t10-results.md) | 历史实验 |
| v5.3.7 整体流程 | [v5.3.7 整体流程方案](cpp_adaiso_v5_3_7_overall_flow_plan.md) | `1ea06a8`，待最终核对 |
| v6.0.0 基线 | `cpp_adaiso/修改计划与实验结果/v6.0.0 基线记录.md` | `3af46f8` |

## 新实验的最小流程

1. 为实验分配独立编号，例如 `E-2026-001-stage-a-ablation`。
2. 复制实验清单模板，填写基线 commit、输入哈希、固定参数和变量。
3. 在干净工作树上构建并运行测试。
4. 执行参数矩阵，将原始产物写入 Git 外的 artifact 目录。
5. 将清单、紧凑指标、失败记录和结论纳入 Git。

大型网格、构建目录、依赖源码和逐轮输出不应放入本目录。
