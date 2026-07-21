# N=2000 T0-T5 inverse-linear + splitFactor=1.33 实验报告

## 设置

将当前默认尺寸场从 `inverse-sqrt` 改为 `inverse-linear`，并保持 code variant `splitFactor=1.33`。随后在 N=2000 下重新运行 T0-T5。

运行参数：

```text
--target-faces 2000
--iters 10
--preprocess 0
--curvature-mode robust-mean
--csf-alpha 2.0
--cd-samples 100000
```

- 新结果目录：`cpp_adaiso/results/v5_N2000_T0_T5_compare/v5.0.2_linear_split133/`
- 对比 CSV：`cpp_adaiso/results/v5_N2000_T0_T5_compare/linear_split133_comparison.csv`

## 平均指标

| Version | mapping | splitFactor | avg faces | avg abs(faces-2000) | avg CD | avg qmin | avg viol | total ops |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v5.0.0_sqrt_split165 | inverse-sqrt | 1.65 | 1809 | 190.6667 | 8.245e-05 | 0.254921 | 0.058603 | 56220 |
| v5.0.1_sqrt_split165 | inverse-sqrt | 1.65 | 1807 | 193.3333 | 8.240e-05 | 0.221189 | 0.059131 | 56190 |
| v5.0.2_sqrt_split165 | inverse-sqrt | 1.65 | 1592 | 408.3333 | 9.559e-05 | 0.249689 | 0.043076 | 56205 |
| v5.0.2_sqrt_split133 | inverse-sqrt | 1.33 | 2152 | 160.3333 | 7.000e-05 | 0.160095 | 0.189215 | 55103 |
| v5.0.2_linear_split133 | inverse-linear | 1.33 | 2166 | 167.0000 | 8.686e-05 | 0.176767 | 0.257572 | 57283 |

## 每模型 CD 对比

| Model | v5.0.1 sqrt165 | v5.0.2 sqrt133 | v5.0.2 linear133 | linear vs sqrt133 | linear winner? | faces linear | qmin linear | viol linear |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| T0 | 1.122e-04 | 8.902e-05 | 1.017e-04 | +14.26% | no | 2158 | 0.201553 | 0.239728 |
| T1 | 8.771e-05 | 7.625e-05 | 1.097e-04 | +43.88% | no | 2220 | 0.044416 | 0.275976 |
| T2 | 1.298e-04 | 1.104e-04 | 1.360e-04 | +23.16% | no | 2188 | 0.129373 | 0.264168 |
| T3 | 1.571e-05 | 1.324e-05 | 1.889e-05 | +42.63% | no | 2156 | 0.240725 | 0.275819 |
| T4 | 6.086e-05 | 5.392e-05 | 6.836e-05 | +26.78% | no | 2276 | 0.148770 | 0.282660 |
| T5 | 8.812e-05 | 7.716e-05 | 8.650e-05 | +12.12% | no | 1996 | 0.295767 | 0.207081 |

## 结论

- linear+split133 相比 sqrt+split133：0/6 个模型 CD 更低。
- linear+split133 相比 v5.0.1 sqrt+split165：2/6 个模型 CD 更低。
- 平均 CD：sqrt+split133 为 `7.000e-05`，linear+split133 为 `8.686e-05`。
- 平均 faces：sqrt+split133 为 `2152`，linear+split133 为 `2166`。
- 平均 qmin：sqrt+split133 为 `0.160095`，linear+split133 为 `0.176767`。

因此，在 N=2000、splitFactor=1.33 的条件下，`inverse-linear` 默认尺寸场没有收益。它让平均 qmin 略好，但 CD 明显差于 `inverse-sqrt`，final violation 也更高。更合理的默认仍是 `inverse-sqrt + splitFactor=1.33`，如果要继续探索 linear，应单独调低 splitFactor 或 collapseFactor，而不是直接替换默认尺寸场。
