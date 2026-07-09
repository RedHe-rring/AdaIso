# Lv 全量补跑与 cpp_adaiso 同源指标报告

## 实验设置

- 模型范围：`cpp_adaiso/examples` 下全部 `.obj/.off/.ply`，共 `55` 个。
- Lv exe：`E:\Remeshing\2024Lv_Adaptively-Isotropic-Remeshing-main\Adaptively-Isotropic-Remeshing-main\SHFVCG2021remesh_Hist\x64\Debug\SHFVCGRemesh_Hist.exe`
- Lv 参数：`preprocessing=0`、`isotropic_judge=2`、`meshscale=1.2`。
- 输出目录：`compare/Lv_adaiso/batch_run/results_iter5/`
- 指标工具：`cpp_adaiso/build/adaiso_eval.exe`，直接调用 cpp_adaiso `summarizeMesh` 和 `areaSampledChamferDistance`。
- CD 采样：每方向 `100000` 点。
- PLY 输入评估：`3` 个 PLY 的 GT 先用 trimesh 转成 OBJ，再交给 `adaiso_eval`，避免 binary PLY 读取差异。

## 完整性检查

| 检查项 | 结果 |
|---|---:|
| 当前 examples 模型数 | 55 |
| Lv 成功输出数 | 26 |
| Lv 失败或缺失数 | 29 |
| 成功输出 + 失败或缺失 | 55 |
| 跳过既有输出数 | 26 |
| metrics JSON 成功数 | 26 |
| metrics 失败或缺失数 | 29 |

- 运行尝试日志：`compare/Lv_adaiso/batch_run/results_iter5/batch_run/all_examples_attempts.csv`
- 全量状态 CSV：`compare/Lv_adaiso/batch_run/results_iter5/all_examples_status.csv`
- 指标汇总 CSV：`compare/Lv_adaiso/batch_run/results_iter5/lv_metrics_summary.csv`

## Lv 指标汇总

| 指标 | 数值 |
|---|---:|
| 平均 Qavg | 0.8472992830 |
| 平均 Qmin | 0.0161450003 |
| 最低 Qmin | 0.0000007239 |
| 平均 CD | 1.3999835346E-02 |
| 最大 CD | 1.7207307680E-01 |
| 平均 RMSE | 4.4231013742E-02 |

- Qmin 最低：`model_001` (7.23921e-07), `hand` (1.21955e-06), `model_003` (1.36621e-06), `model_007` (1.47089e-06), `model_009` (1.5181e-06), `model_010` (1.67267e-06), `chair36` (2.93086e-06), `dragon` (3.67027e-06), `model_005` (4.99546e-06), `T2` (2.26164e-05)
- CD 最大：`model_001` (0.172073), `model_005` (0.0900808), `model_003` (0.0328953), `model_007` (0.0185229), `model_002` (0.0135792), `model_010` (0.011937), `model_004` (0.0108875), `model_009` (0.00658449), `model_006` (0.00640374), `wine_glass3` (0.000451584)

| Model | Input | Lv Out V/F | Qavg | Qmin | CD | RMSE | Status |
|---|---|---:|---:|---:|---:|---:|---|
| bunny | .obj | 63115/126216 | 0.881041 | 0.000134 | 3.123535E-05 | 3.951920E-03 | skipped_existing_success |
| chair36 | .ply | 11929/23479 | 0.819690 | 0.000003 | 1.117371E-05 | 2.363653E-03 | skipped_existing_success |
| dragon | .obj | 698634/1396963 | 0.901111 | 0.000004 | 9.312916E-06 | 2.157883E-03 | skipped_existing_success |
| hand | .obj | 15796/31542 | 0.870833 | 0.000001 | 1.598057E-05 | 2.826709E-03 | skipped_existing_success |
| model_001 | .off | 15049/27253 | 0.841556 | 0.000001 | 1.720731E-01 | 2.933199E-01 | skipped_existing_success |
| model_002 | .off | 8281/14206 | 0.838867 | 0.000066 | 1.357920E-02 | 8.239902E-02 | skipped_existing_success |
| model_003 | .off | 11695/22261 | 0.875615 | 0.000001 | 3.289530E-02 | 1.282484E-01 | skipped_existing_success |
| model_004 | .off | 15348/30708 | 0.889542 | 0.413054 | 1.088746E-02 | 7.378164E-02 | skipped_existing_success |
| model_005 | .off | 14594/24453 | 0.825766 | 0.000005 | 9.008079E-02 | 2.122272E-01 | skipped_existing_success |
| model_006 | .off | 28401/56800 | 0.889980 | 0.000041 | 6.403735E-03 | 5.658505E-02 | skipped_existing_success |
| model_007 | .off | 14219/25765 | 0.863078 | 0.000001 | 1.852292E-02 | 9.623649E-02 | skipped_existing_success |
| model_009 | .off | 24395/41767 | 0.824994 | 0.000002 | 6.584487E-03 | 5.737807E-02 | skipped_existing_success |
| model_010 | .off | 16127/27499 | 0.825310 | 0.000002 | 1.193696E-02 | 7.725594E-02 | skipped_existing_success |
| T0 | .off | 7978/15946 | 0.860456 | 0.000756 | 2.228272E-05 | 3.337867E-03 | skipped_existing_success |
| T1 | .off | 7460/14913 | 0.864301 | 0.000774 | 2.636821E-05 | 3.630992E-03 | skipped_existing_success |
| T2 | .off | 9546/19083 | 0.878806 | 0.000023 | 2.674621E-05 | 3.656926E-03 | skipped_existing_success |
| T3 | .off | 9198/18389 | 0.873529 | 0.001221 | 3.863066E-06 | 1.389796E-03 | skipped_existing_success |
| T4 | .off | 5976/11921 | 0.857239 | 0.000056 | 2.473468E-05 | 3.516723E-03 | skipped_existing_success |
| T5 | .off | 4516/9012 | 0.863758 | 0.000570 | 2.436880E-05 | 3.490616E-03 | skipped_existing_success |
| T6 | .off | 5173/10324 | 0.848340 | 0.000093 | 3.919526E-05 | 4.426921E-03 | skipped_existing_success |
| T7 | .off | 6106/12197 | 0.831280 | 0.000024 | 3.391004E-06 | 1.302114E-03 | skipped_existing_success |
| T8 | .off | 7694/15374 | 0.851751 | 0.000563 | 2.585661E-05 | 3.595595E-03 | skipped_existing_success |
| T9 | .off | 5423/10826 | 0.845975 | 0.000405 | 3.242173E-05 | 4.026272E-03 | skipped_existing_success |
| T10 | .off | 7734/15465 | 0.878289 | 0.000906 | 9.448109E-06 | 2.173489E-03 | skipped_existing_success |
| wine_glass3 | .ply | 825/1630 | 0.790959 | 0.000083 | 4.515842E-04 | 1.502638E-02 | skipped_existing_success |
| wine_glass24 | .ply | 733/1461 | 0.637716 | 0.000982 | 2.738176E-04 | 1.170080E-02 | skipped_existing_success |

## 失败与缺失

| Model | Lv status | Metric status | Log | Error |
|---|---|---|---|---|
| C0 | failed_or_pending | missing_lv_output | `compare/Lv_adaiso/batch_run/results_iter5/batch_run/logs/C0_20260709_163144.log` | return_code=3221225786, output_exists=False, output_bytes=0 |
| C1 | failed_or_pending | missing_lv_output |  | Lv output missing |
| C2 | failed_or_pending | missing_lv_output |  | Lv output missing |
| C3 | failed_or_pending | missing_lv_output |  | Lv output missing |
| C4 | failed_or_pending | missing_lv_output |  | Lv output missing |
| C6 | failed_or_pending | missing_lv_output |  | Lv output missing |
| C7 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T21 | failed_or_pending | missing_lv_output | `compare/Lv_adaiso/batch_run/results_iter5/batch_run/logs/T21_20260709_170222.log` | return_code=0, output_exists=False, output_bytes=0 |
| T30 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T33 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T36 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T38 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T44 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T48 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T51 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T53 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T67 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T71 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T72 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T82 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T86 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T87 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T96 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T118 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T119 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T124 | failed_or_pending | missing_lv_output |  | Lv output missing |
| T125 | failed_or_pending | missing_lv_output |  | Lv output missing |
| Th1 | failed_or_pending | missing_lv_output |  | Lv output missing |
| Th2 | failed_or_pending | missing_lv_output |  | Lv output missing |

## T 系列参考对比

下表仅作参考：Lv 使用原始外部实现 `iter=5, meshscale=1.2`；cpp_adaiso v2.8.1 汇总使用当前项目 `iter=10` 的 percentile target 实现。两者方法、迭代次数和尺寸场实现不同，不能视为单变量消融。

| Model | Lv V | cpp V | dV | Lv Qmin | cpp Qmin | dQmin | Lv CD | cpp CD | dCD% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | 7978 | 10375 | -2397 | 0.000756 | 0.443923 | -0.443168 | 2.228272E-05 | 1.862930E-05 | +19.61% |
| T1 | 7460 | 8558 | -1098 | 0.000774 | 0.471096 | -0.470322 | 2.636821E-05 | 2.314450E-05 | +13.93% |
| T2 | 9546 | 10390 | -844 | 0.000023 | 0.426014 | -0.425991 | 2.674621E-05 | 2.577870E-05 | +3.75% |
| T3 | 9198 | 9721 | -523 | 0.001221 | 0.304574 | -0.303353 | 3.863066E-06 | 4.127664E-06 | -6.41% |
| T4 | 5976 | 10104 | -4128 | 0.000056 | 0.401156 | -0.401100 | 2.473468E-05 | 2.037019E-05 | +21.43% |
| T5 | 4516 | 10414 | -5898 | 0.000570 | 0.451554 | -0.450984 | 2.436880E-05 | 1.294604E-05 | +88.23% |
| T6 | 5173 | 10325 | -5152 | 0.000093 | 0.300117 | -0.300024 | 3.919526E-05 | 1.928089E-05 | +103.29% |
| T7 | 6106 | 7998 | -1892 | 0.000024 | 0.470474 | -0.470451 | 3.391004E-06 | 2.591653E-06 | +30.84% |
| T8 | 7694 | 9882 | -2188 | 0.000563 | 0.434198 | -0.433635 | 2.585661E-05 | 2.211055E-05 | +16.94% |
| T9 | 5423 | 9753 | -4330 | 0.000405 | 0.284023 | -0.283617 | 3.242173E-05 | 1.894002E-05 | +71.18% |
| T10 | 7734 | 9458 | -1724 | 0.000906 | 0.346195 | -0.345289 | 9.448109E-06 | 9.051577E-06 | +4.38% |
