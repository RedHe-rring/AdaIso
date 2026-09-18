# cpp_adaiso v5.3.7 整体流程方案

本文档按当前代码版本 `cpp_adaiso` v5.3.7 编写，目标是让读者能够根据源码和命令复现完整流程。当前实现的主线是：从输入三角网格估计曲率，经过 CSF 平滑得到目标尺寸场，再用两阶段局部算子让输出网格同时接近尺寸场并保持三角形质量。

对应源码主要在：

- `cpp_adaiso/src/main.cpp`：命令行参数、输入输出、JSON 汇总。
- `cpp_adaiso/src/AdaIsoRemesher.cpp`：目标尺寸场、两阶段 remesh、局部算子。
- `cpp_adaiso/src/CsfSolver.cpp`：CSF 稀疏线性系统。
- `cpp_adaiso/src/Geometry.cpp`、`cpp_adaiso/src/Metrics.cpp`：几何量和评估指标。
- `cpp_adaiso/include/adaiso/AdaIsoRemesher.hpp`：默认参数和统计字段。

## 1. 输入、输出与默认参数

输入为三角网格 `M=(V,F)`，支持 `.obj/.ply/.off`。顶点为 `p_i in R^3`，三角面为 `(i,j,k)`。

默认输出包括：

- 重网格化后的三角网格。
- 标准输出中的 JSON summary，或通过 `--summary-json <path>` 写入文件。
- 可选尺寸场诊断：`--export-target-field <dir>`。
- 可选迭代网格：`--export-iteration-meshes <dir>`、`--export-stage-meshes <dir>`、`--export-all-stage-meshes <dir>`。

当前默认关键参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--variant` | `code` | 使用工程版本阈值 |
| `--curvature-mode` | `principal-max` | 原始曲率信号 |
| `--csf-alpha` | `2.0` | CSF 平滑强度 |
| `--target-error-delta` | `0` | 为 0 时自动估计误差容差 |
| `--auto-delta-quantile` | `0.5` | 自动容差的面积加权分位数 |
| `--target-h-max-factor` | `5.0` | `h_max = 5 * mean_edge_length` |
| `--iters` | `5` | size stage 和 quality stage 的最大外层轮数 |
| `--split-factor` | `1.33` | `variant=code` 的长边分裂阈值 |
| `--collapse-factor` | `0.8` | 短边折叠阈值 |
| `--stage-a-policy` | `split_collapse_no_guard` | Stage A 默认先 split 后 collapse |
| `--min-angle-floor-degrees` | `10` | collapse/flip/relocate 的质量安全底线 |
| `--split-min-angle-floor-degrees` | `1` | 弱 split guard 使用的角度底线 |
| `--max-flip-dihedral-deg` | `45` | flip 的二面角限制 |
| `--feature-aware-collapse` | `1` | 折叠点是否偏向强特征端点 |
| `--collapse-feature-omega` | `0.15` | 特征强度差异阈值 |
| `--relocation-mls` | `0` | 是否启用 MLS 投影 |

## 2. 总体算法

伪代码如下：

```text
Input: triangular mesh M0
Output: remeshed mesh M

M <- removeDegenerateFaces(M0)
M_original <- M

if preprocess:
    M <- subRemesh(M)
    M <- removeDegenerateFaces(M)

kappa_raw <- computeRawCurvatureField(M)
kappa <- solveScreenedCSF(M, kappa_raw)
h_i <- targetErrorCurvatureLengths(M, kappa)
H(x) <- build surface-barycentric target length field from h_i

for iter = 1..global_iterations:       # Stage A: size field fitting
    splitLongEdges(M, H)
    collapseShortEdges(M, H)
    record size metrics
    stop if no split and no collapse

for iter = 1..global_iterations:       # Stage B: quality repair
    flipEdges(M, H)
    relocateVertices(M, H, M_original)
    record quality metrics
    stop if no flip and no relocation

M <- removeDegenerateFaces(M)
write M and summary
```

v5.3.7 的关键变化是把局部尺寸场距离统一为连续目标度量的对数能量：

```math
E_{\mathrm{size}} = \frac{1}{|E_P|}\sum_{e\in E_P} \left(\log r_e\right)^2
```

其中 `P` 是局部 patch，`E_P` 是 patch 内去重后的边集合，`r_e` 是目标度量边长。Stage A 的 collapse 必须降低局部 `E_size`；Stage B 的 flip 和 relocate 在改善质量时也不能让 `E_size` 恶化。

## 3. 曲率场构建

代码支持三种曲率模式。

### 3.1 `principal-max`

默认模式调用 libigl 的 `principal_curvature`，得到两个主曲率 `k_1(i), k_2(i)`，原始曲率为：

```math
\kappa_i^{0}=\max(|k_1(i)|, |k_2(i)|)
```

若 libigl 返回非有限值，则用内部下界 `epsilon_k = 1e-12` 保护。

### 3.2 `normal-angle`

先计算面积加权顶点法向 `n_i`，再对每个顶点邻域取法向夹角均值：

```math
\kappa_i^0 =
\frac{1}{|N(i)|}
\sum_{j\in N(i)}
\arccos(\mathrm{clamp}(n_i\cdot n_j,-1,1))
```

该模式更像离散弯折强度，量纲和 `principal-max` 不同，主要用于对照实验。

### 3.3 `robust-mean`

先计算重心顶点面积：

```math
A_i=\sum_{f\ni i}\frac{A_f}{3}
```

三角形顶点 `a` 处余切为：

```math
\cot\theta_a=
\frac{(p_b-p_a)\cdot(p_c-p_a)}
{\|(p_b-p_a)\times(p_c-p_a)\|}
```

代码只保留正余切并截断：

```math
\widehat{\cot\theta}=
\begin{cases}
0, & \cot\theta \le 0 \\
\min(\cot\theta,10), & \cot\theta > 0
\end{cases}
```

边权为相邻对角余切贡献的二分之一累加。顶点拉普拉斯向量：

```math
L_i = \sum_{j\in N(i)} w_{ij}(p_j-p_i)
```

面积下界为：

```math
A_{\mathrm{floor}}=10^{-8}\bar l^2
```

其中 `bar l` 是全局平均边长。最终曲率信号：

```math
\kappa_i^0 =
\frac{1}{2\max(A_i,A_{\mathrm{floor}})}
\|L_i\|\bar l
```

## 4. CSF 平滑

CSF 使用 cotangent 权重图和顶点质量矩阵求解 screened scalar field。设 `m_i` 为重心顶点面积，`w_ij` 为边权，`alpha` 为 `--csf-alpha`，`bar l` 为平均边长：

```math
\lambda = \alpha \bar l^2
```

求解线性系统：

```math
(M+\lambda L)\tilde\kappa = M\kappa^0
```

展开到每个顶点：

```math
m_i(\tilde\kappa_i-\kappa_i^0)
+
\lambda \sum_{j\in N(i)} w_{ij}(\tilde\kappa_i-\tilde\kappa_j)
=0
```

矩阵装配形式为：

```math
A_{ii}=m_i+\lambda\sum_{j\in N(i)}w_{ij},\qquad
A_{ij}=-\lambda w_{ij}
```

求解器使用 Eigen `SimplicialLDLT`。默认缩放残差阈值是 `1e-10`：

```math
r_{\mathrm{scaled}} =
\frac{2\|A\tilde\kappa-b\|}
{\|A\|_\infty\|\tilde\kappa\|_\infty+\|b\|_\infty+\epsilon}
```

## 5. 目标尺寸场

目标尺寸场由平滑曲率 `tilde kappa_i` 和目标误差容差 `delta` 决定。

首先做安全曲率：

```math
\kappa_i=\max(\tilde\kappa_i, 10^{-12})
```

计算每个顶点的局部输入分辨率 `ell_i`，即 incident edge length 的中位数。局部自动容差为：

```math
\delta_i = \frac{\kappa_i \ell_i^2}{6}
```

若用户传入 `--target-error-delta > 0`，则直接使用该值；否则按顶点重心面积 `m_i` 取面积加权分位数：

```math
\delta =
Q_q^{area}(\{\delta_i\}, \{m_i\})
```

默认 `q=0.5`。如果自动结果非正或非有限，则回退：

```math
\delta = \frac{10^{-12}\bar l^2}{6}
```

尺寸上下界：

```math
h_{\min}=0.25\bar l,\qquad
h_{\max}=5.0\bar l
```

每个顶点目标边长：

```math
h_i =
\mathrm{clamp}
\left(
\sqrt{\frac{6\delta}{\kappa_i}-3\delta^2},
h_{\min},
h_{\max}
\right)
```

如果观测最大曲率不大于 `1e-12`，说明曲率场基本为平，代码进入 flat fallback：

```math
h_i=h_{\max}
```

目标面数只作为诊断量，不再驱动优化：

```math
\hat N =
\sum_{f\in F}
\frac{A_f}
{(\sqrt 3/4)h_f^2},
\qquad
h_f=\frac{h_i+h_j+h_k}{3}
```

`TargetLengthField` 将 `h_i` 建在原始目标场网格上。任意查询点 `x` 先用 AABB 找最近三角形，再对最近点做重心插值：

```math
H(x)=\beta_0h_i+\beta_1h_j+\beta_2h_k
```

## 6. 目标度量边长与尺寸能量

对边 `e=(i,j)`，令实际长度为：

```math
l_{ij}=\|p_i-p_j\|
```

端点目标长度为 `h_i=H(p_i), h_j=H(p_j)`。代码假设边上目标长度线性变化，使用积分形式的目标度量边长：

```math
r_{ij}=
\int_0^{l_{ij}}\frac{ds}{h(s)}
```

离散实现为：

```math
r_{ij}=
\begin{cases}
l_{ij}/h_i, & |h_i-h_j|\approx 0 \\
l_{ij}\dfrac{\log h_j-\log h_i}{h_j-h_i}, & \text{otherwise}
\end{cases}
```

理想状态是 `r_ij = 1`。短边、长边阈值分别为：

```math
c=0.8,\qquad s=1.33
```

`variant=paper` 时 `s=5/3`。诊断用 hinge violation：

```math
v_{ij}=\max(0,c-r_{ij})+\max(0,r_{ij}-s)
```

v5.3.7 的主要尺寸距离指标为：

```math
E_{\mathrm{size}}(P)=
\frac{1}{|E_P|}
\sum_{(i,j)\in E_P}
\left(\log r_{ij}\right)^2
```

该能量连续惩罚偏离 `r=1` 的边，不只统计是否越过 `[c,s]` 阈值。

## 7. Stage A：尺寸场拟合

Stage A 只执行 split 和 collapse，主职责是改变采样密度。

### 7.1 长边分裂

候选边条件：

```math
r_{ij}>s
```

候选按 `r_ij` 从大到小排序，并用 `used_vertices` 做批处理冲突过滤，保证同一批中没有共享端点。

分裂点不是普通中点，而是按两端目标尺寸偏移：

```math
t=
\mathrm{clamp}
\left(
\frac{\sqrt{h_i}}{\sqrt{h_i}+\sqrt{h_j}},
0.25,
0.75
\right)
```

```math
p_m=(1-t)p_i+tp_j
```

若 `h_i < h_j`，则 `t` 较小，新点更靠近 `i`，也就是更靠近目标尺寸更小、需要更密采样的一侧。

默认 `stage_a_policy=split_collapse_no_guard` 时 split 不启用角度 guard；另外两个实验策略会检查分裂后的局部最小角不低于 `split_min_angle_floor_degrees`。

### 7.2 短边折叠

候选边条件：

```math
r_{ij}<c
```

候选按 `r_ij` 从小到大排序。边界端点不折叠；非流形、link condition 不满足、局部 patch 退化、角度安全不满足、尺寸能量不下降都会拒绝。

拓扑合法性采用 link condition。对待折叠边 `(a,b)`，令 `N(a)`、`N(b)` 为邻接顶点集合，`O(a,b)` 为该边相邻三角形中的对顶点集合。必须满足：

```math
(N(a)\cap N(b))\setminus\{a,b\}=O(a,b)
```

折叠候选点默认取中点：

```math
p_c=\frac{p_a+p_b}{2}
```

开启 feature-aware collapse 时先计算顶点特征强度。顶点角亏损：

```math
D_i=\Theta_i^{ref}-\sum_{f\ni i}\theta_{i,f}
```

其中内部点 `Theta_i^{ref}=2pi`，边界点 `Theta_i^{ref}=pi`。顶点最大二面角为 `phi_i^{max}`。截断映射：

```math
\tau(x)=\min(\pi,2x)
```

特征强度：

```math
K_i=\tau(|D_i|),\qquad
E_i=\tau(\phi_i^{max})
```

```math
F_i=(K_i+1)(E_i+1)-1
```

若两端特征强度差异显著：

```math
|F_a-F_b| \ge \omega\max(F_a,F_b)
```

则折叠点取特征更强的端点，否则取中点。默认 `omega=0.15`。

collapse 的核心接受准则是局部尺寸能量严格下降：

```math
E_{\mathrm{size}}^{after}
<
E_{\mathrm{size}}^{before}
-10^{-8}
```

同时满足最小角安全：

```math
\theta_{\min}^{after}
\ge
\min(\theta_{\min}^{before}, \theta_{\mathrm{floor}})
```

默认 `theta_floor=10` 度。

每个 collapse pass 还包含冲突区域 mini-round。普通批处理因 footprint 冲突跳过的候选会把局部顶点加入种子集合，随后最多执行 1 个 retry mini-round，重新建拓扑并只检查冲突区域邻近短边。代码常量 `kCollapseConflictMiniRounds=2`，包含普通轮和一次 retry 轮。

## 8. Stage B：质量修复

Stage B 只执行 flip 和 relocate，不改变目标尺寸场，主职责是改善连接和点位质量，同时不破坏尺寸能量。

### 8.1 翻边

只考虑有两个相邻面的内部边 `(a,b)`。设两个对顶点为 `c,d`，翻边后新边为 `(c,d)`。基本拒绝条件：

- 原两面二面角大于 `max_flip_dihedral_degrees`。
- 新边已存在。
- 新目标度量边长不在 `[c,s]`。
- 新两面二面角超限。
- `avoid_obtuse_flip=true` 时，新三角形出现钝角。

局部 patch 接受逻辑：

1. 局部最小角质量必须通过：

```math
\theta_{\min}^{after}
>
\theta_{\min}^{before}
\quad \text{if } \theta_{\min}^{before}<\theta_{\mathrm{floor}}
```

否则：

```math
\theta_{\min}^{after}\ge\theta_{\min}^{before}
```

2. 尺寸能量不能恶化：

```math
E_{\mathrm{size}}^{after}
\le
E_{\mathrm{size}}^{before}+10^{-8}
```

3. 若当前低于角度底线，角度改善即可接受；否则按以下顺序接受：

- 最小角严格改善。
- 尺寸能量严格下降。
- 尺寸能量近似相等时，valence 目标更好。

valence 目标为：

```math
J=\sum_{v\in\{a,b,c,d\}}|\deg(v)-d_v^*|
```

内部点目标度数 `d_v^*=6`，边界点目标度数 `d_v^*=4`。

### 8.2 顶点重定位

先根据特征强度分类顶点：

- 边界顶点：`Corner`。
- 特征强度接近 0：`Smooth`。
- 其余顶点统计邻域中满足 `F_j >= 0.5 F_i` 的高特征邻居数 `m_i`。

分类距离：

```math
d_{\mathrm{corner}}=m_i,\qquad
d_{\mathrm{crease}}=|m_i-2|,\qquad
d_{\mathrm{smooth}}=|m_i-|N(i)||
```

选择距离最小的类别。只有 `Smooth` 顶点参与切向平滑。

对 smooth 顶点 `i`，按相邻面面积和邻居目标长度加权得到邻域中心：

```math
c_i=
\frac{\sum_{j\in N(i)} w_{ij} p_j}
{\sum_{j\in N(i)} w_{ij}},
\qquad
w_{ij}=\sum_{f\ni i,j}A_f\cdot H(p_j)
```

位移：

```math
d_i=c_i-p_i
```

去掉法向分量后取切向位移：

```math
d_i^T=d_i-(d_i\cdot n_i)n_i
```

候选点：

```math
p_i'=p_i+\lambda_t d_i^T
```

默认 `lambda_t=0.6`。

重定位接受条件：

```math
E_{\mathrm{size}}^{after}
\le
E_{\mathrm{size}}^{before}+10^{-8}
```

并通过与 flip 相同的最小角质量规则。

若 `--relocation-mls 1`，还会对 smooth 顶点尝试 MLS 投影回原始网格附近；投影点同样必须通过尺寸能量和角度质量检查。

## 9. 质量与误差指标

### 9.1 三角形质量

三角形质量定义为：

```math
q =
\frac{6A}
{\sqrt 3\,s\,l_{\max}}
```

其中 `A` 为三角形面积，`s=(l_0+l_1+l_2)/2` 为半周长，`l_max` 为最长边。等边三角形时 `q=1`，退化时接近 `0`。

summary 会输出：

- `qavg`：平均三角形质量。
- `qmin`：最小三角形质量。
- `min_angle_degrees`：全局最小内角。
- `angle_lt_10_pct`、`angle_lt_20_pct`：所有角中小于阈值的百分比。

### 9.2 尺寸场指标

对全部唯一边统计：

```math
\bar r=\frac{1}{|E|}\sum_e r_e
```

```math
\sigma_r=
\sqrt{\frac{1}{|E|}\sum_e(r_e-\bar r)^2}
```

同时输出 `p05_ratio`、`p95_ratio`、`violation_rate`、`short_violation_rate`、`long_violation_rate` 和：

```math
target\_log\_energy =
\frac{1}{|E|}\sum_e(\log r_e)^2
```

### 9.3 Chamfer Distance

若传入 `--gt <mesh>`，CLI 使用面积采样计算双向 Chamfer。每个方向默认 `100000` 个样本。采样序列使用 radical inverse：

- 面选择：base 2。
- 三角形内部参数：base 3 和 base 5。

双向均方最近邻距离：

```math
d_{out\to gt} =
\frac{1}{N}\sum_{x\in S_{out}}\min_{y\in S_{gt}}\|x-y\|^2
```

```math
d_{gt\to out} =
\frac{1}{N}\sum_{y\in S_{gt}}\min_{x\in S_{out}}\|y-x\|^2
```

Chamfer 和 symmetric RMSE：

```math
CD=d_{out\to gt}+d_{gt\to out}
```

```math
RMSE=\sqrt{\frac{CD}{2}}
```

## 10. 可复现实验步骤

### 10.1 构建

在仓库根目录执行：

```powershell
cmake -S cpp_adaiso -B cpp_adaiso/build2 -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_adaiso/build2 --config Release
ctest --test-dir cpp_adaiso/build2 -C Release --output-on-failure
```

项目依赖：

- C++17。
- Eigen3。
- nanoflann。
- Libigl。

`CMakeLists.txt` 会在找不到 Eigen3 或 nanoflann 包时通过 `FetchContent` 获取；Libigl 当前要求 `find_package(Libigl CONFIG REQUIRED)`。

### 10.2 单模型复现

示例命令：

```powershell
cpp_adaiso/build2/Release/adaiso_cli.exe `
  cpp_adaiso/examples/T21.off `
  -o cpp_adaiso/results/repro_T21/output.ply `
  --summary-json cpp_adaiso/results/repro_T21/summary.json `
  --export-target-field cpp_adaiso/results/repro_T21/target_field `
  --export-iteration-meshes cpp_adaiso/results/repro_T21/iters `
  --export-stage-meshes cpp_adaiso/results/repro_T21/stage `
  --curvature-mode principal-max `
  --csf-alpha 2.0 `
  --auto-delta-quantile 0.5 `
  --target-h-max-factor 5.0 `
  --iters 5
```

如果有 GT 网格，加入：

```powershell
--gt cpp_adaiso/examples/T21.off --cd-samples 100000
```

### 10.3 输出文件核查

推荐检查以下文件：

- `summary.json`：参数、输入输出统计、每轮迭代统计、stage 汇总、CSF 残差。
- `target_field/vertices.csv`：每个输入顶点的 raw curvature、CSF curvature、local delta、target h。
- `target_field/edges.csv`：输入尺寸场边上的 `target_ratio` 和 `h_slope`。
- `target_field/field_stats.json`：目标尺寸场范围、估计面数、局部分位数。
- `target_field/output_edge_ratios.csv`：输出网格每条边的目标度量长度。
- `iters/*.ply`：Stage A 和 Stage B 结束后的网格快照。
- `stage/*.ply`：第一轮中 split、collapse、flip、smooth 后的局部阶段快照。

### 10.4 最小复现实验记录模板

每次实验建议记录：

```text
version: cpp_adaiso 5.3.7
input: <path>
command: <full adaiso_cli command>
parameters:
  curvature_mode:
  csf_alpha:
  target_error_delta:
  auto_delta_quantile:
  target_h_max_factor:
  split_factor:
  collapse_factor:
  global_iterations:
outputs:
  output_mesh:
  summary_json:
  target_field_dir:
key metrics:
  output_faces:
  qavg:
  qmin:
  min_angle_degrees:
  angle_lt_10_pct:
  angle_lt_20_pct:
  mean_ratio:
  violation_rate:
  short_violation_rate:
  long_violation_rate:
  target_log_energy:
  chamfer_distance:
  symmetric_rmse:
```

## 11. 复现时的注意事项

1. 当前版本不是目标面数驱动。`estimated_faces` 是由目标尺寸场反推的诊断量，不是输入约束。
2. 默认 curvature mode 是 `principal-max`，如果切换到 `robust-mean` 或 `normal-angle`，尺寸场和输出面数会明显变化。
3. `target_error_delta=0` 表示自动容差。想严格复现某次实验，应记录 summary 中实际写出的 `target_error_delta`。
4. v5.3.7 的主尺寸指标是 `target_log_energy`，旧的 `violation_rate` 仍保留，但更多用于阈值越界诊断。
5. Stage A 和 Stage B 的职责不同：split/collapse 负责尺寸密度，flip/relocate 负责质量修复。不要把四个算子理解为同一个加权能量的四种优化步。
6. 若要比较不同版本或不同方法，应同时报告面数、质量、尺寸场偏差、几何误差和 runtime；只比较 Chamfer 或只比较面数都不完整。
