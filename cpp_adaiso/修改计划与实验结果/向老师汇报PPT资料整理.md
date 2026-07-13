# 向老师汇报 PPT 资料整理

本文档用于整理当前 AdaIso C++ 复现与改进工作的汇报材料。汇报主线以当前 v2.10.2 工作区方案为主，穿插 v2.6.0 到 v2.10.2 的关键实验，重点说明每一步“怎么做”和“为什么这样做”。

建议 PPT 控制在 10-12 页，每页一个核心方法点。不要只展示最终指标，要把曲率计算、CSF 扩散、尺寸场构造和局部算子约束讲清楚。

## 1. 研究目标

### PPT 标题

基于 Curvature Smoothed Field 的自适应各向同性重网格复现与改进

### 讲述重点

- 目标是复现并工程化实现论文中的自适应各向同性重网格流程。
- 输入为三角网格，输出为边长更接近目标尺寸场、三角形质量更好的重网格。
- 关注的不只是几何误差，还包括三角形质量、目标边长满足程度和特征保留。

### 评价指标

| 指标 | 含义 | 作用 |
|---|---|---|
| CD / RMSE | 输出网格与原始网格的几何误差 | 衡量贴合程度 |
| Qavg / Qmin | 三角形质量平均值和最小值 | 衡量网格单元质量 |
| edge-target violation | 边长落在目标范围外的比例 | 衡量尺寸场满足程度 |
| feature preservation | 尖锐边、边界、细节是否保留 | 衡量视觉和拓扑合理性 |

### 实现入口

当前主流程入口是：

```cpp
Mesh AdaIsoRemesher::remesh(const Mesh& input);
```

位置：`cpp_adaiso/src/AdaIsoRemesher.cpp`。

### 建议配图

- 左侧放原始模型。
- 右侧放 remesh 结果。
- 下方列出 CD、Qavg、Qmin、violation 四个核心指标。

## 2. 整体算法流程

### PPT 标题

整体流程：曲率场驱动的局部重网格迭代

### 当前流程

```text
输入三角网格
  -> removeDegenerateFaces()
  -> 可选 preprocess / sub-remesh
  -> 计算 raw curvature
  -> CSF 扩散得到 diffused curvature
  -> 曲率映射为 target h
  -> 构建可查询的目标尺寸场
  -> 全局迭代:
       splitLongEdges()
       collapseShortEdges()
       flipEdges()
       relocateVertices()
       removeDegenerateFaces()
  -> 输出网格
```

### 关键设计

- 目标尺寸场只在 remesh 开始时构建一次。
- 后续顶点移动后，通过输入曲面上的最近三角形 barycentric 插值查询目标 h。
- 每轮迭代后记录局部操作数量和质量指标，用于定位问题。

### 每轮统计

每一轮记录：

- split 数量
- collapse 数量
- flip 数量
- vertices / faces
- Qavg / Qmin
- mean edge-target ratio
- p05 / p95 edge-target ratio
- violation rate

### 建议配图

用流程图展示：

```text
raw curvature -> CSF curvature -> target h -> split/collapse/flip/smooth
```

## 3. 原始曲率计算

### PPT 标题

原始曲率：从 normal-angle 到 robust-mean

### 早期 normal-angle 模式

早期曲率用顶点法向变化估计：

```text
kappa_i = average_j acos(n_i · n_j)
```

其中 `j` 是顶点 `i` 的一环邻点。

特点：

- 实现简单。
- 能反映法向突变。
- 但受网格采样和局部三角形质量影响较大。
- 曲率值没有明确的尺度归一化物理意义。

### 当前默认 robust-mean 模式

当前默认使用 cotan Laplacian mean curvature magnitude：

```text
H_i = 0.5 * || sum_j w_ij (p_j - p_i) || / A_i
raw_kappa_i = H_i * mean_edge_length
```

$
H_i = 0.5 * || \sum_j w_{ij} (p_j - p_i) || / A_i  \\
raw_{kappa_i} = H_i * mean_edge_length
$

代码内部使用
$
w_{ij} = 1/2 * (cot \alpha_{ij} + cot \beta_{ij})
$
因此
$||H_i|| = \frac{1}{2}\frac{|| \sum_j w_{ij} (x_j - x_i) ||}{A_i}
$
这与常见写法
w_ij = cot alpha_ij + cot beta_ij,
|H_i| = ||sum_j w_ij(x_j-x_i)|| / (4A_i)
在系数上等价。

其中：

- `p_i`：顶点坐标。
- `A_i`：barycentric vertex area。
- `w_ij`：cotangent 权重。
- `mean_edge_length`：平均边长，用于得到无量纲曲率信号。

### 权重保护

`w_ij` 采用正 cotangent 权重，并做上限截断：

```text
w = clamp_positive(cot(theta), max = 10.0)
```

这样避免极端瘦长三角形产生过大的 cotangent 权重，导致曲率场爆炸。

### 实现依据

- `computeNormalAngleCurvatureField()`
- `computeRobustMeanCurvatureField()`
- `robustMeanCotangentWeights()`

位置：`cpp_adaiso/src/AdaIsoRemesher.cpp`。

### 建议配图

一页放两张 heat map：

- normal-angle raw curvature
- robust-mean raw curvature

讲述时强调 robust-mean 对高曲率区域响应更明显，但它不是完整解决方案。

## 4. 为什么改成 robust-mean

### PPT 标题

v2.6.0 实验：只改曲率还不够

### T21 曲率模式实验

v2.6.0 中对 T21 做了 normal-angle 与 robust-mean 对比。隔离实验设置 `iters=0`，只观察曲率场和 target field。

| 模式 | raw p95 / max | CSF p95 / max | observed h min / max |
|---|---:|---:|---:|
| normal-angle | 0.704071 / 2.069673 | 0.434008 / 0.674452 | 0.007634 / 0.022825 |
| robust-mean | 1.491122 / 5.073890 | 0.793727 / 1.222756 | 0.007658 / 0.022711 |

结论：

- robust-mean 的 raw 曲率范围更大。
- CSF 扩散后仍保持更高曲率上界。
- 但当时 target_h 范围接近，说明仅增大曲率绝对值不会自动带来更强尺寸分层。

### T21 iter10 实验

| 模式 | Qavg | Qmin | violation | CD |
|---|---:|---:|---:|---:|
| normal-angle | 0.890741 | 0.170683 | 0.038010 | 1.691229e-05 |
| robust-mean | 0.876847 | 0.214335 | 0.060745 | 1.901861e-05 |

判断：

- robust-mean 提高了 Qmin。
- 但 CD 没有改善，violation 上升。
- 因此曲率定义只是第一步，后续还必须处理尺寸映射和局部操作约束。

### model_001 观察

v2.6.0 的 `model_001` 实验显示，即使 robust-mean 曲率场和 target field 正常生成，最终结果仍存在特征约束丢失。

这说明问题不应只归因于曲率场，而要检查：

- split 是否继承特征。
- collapse 是否跨越高曲率脊线。
- flip 是否翻转特征边。
- smoothing 是否抹平特征。
- projection 是否只贴回曲面而没有维护特征线。

### 建议配图

- T21 的 normal-angle / robust-mean 曲率 heat map。
- model_001 原始与 remesh 对比截图，标注特征丢失位置。

## 5. CSF 曲率扩散

### PPT 标题

CSF 扩散：把噪声曲率变成平滑曲率场

### 目标

raw curvature 对局部噪声和采样质量敏感，直接映射 target h 会导致尺寸场跳变。CSF 的目标是得到平滑但仍保留主要高曲率结构的曲率场。

### 当前方程

当前实现使用 screened sparse solve：

```text
(M + alpha * h^2 * L)c = M r
```

其中：

- `r`：raw curvature。
- `c`：diffused curvature。
- `M`：barycentric vertex mass matrix。
- `L`：cotangent Laplacian。
- `h`：mean edge length。
- `alpha`：扩散强度，当前默认 `2.0`。

### 能量形式

这个线性系统可以从下面的优化目标理解：

```text
E(c) = sum_i M_i (c_i - r_i)^2
     + lambda * sum_(i,j) w_ij (c_i - c_j)^2

lambda = alpha * h^2
```

其中第一项是数据项，要求扩散后的曲率 `c_i` 不要偏离原始曲率 `r_i` 太多；第二项是平滑项，要求相邻顶点的曲率值不要跳变太大。`w_ij` 是 cotangent 边权重，`M_i` 是顶点面积质量。

对 `c_i` 求导并令导数为 0，可得到每个顶点上的离散方程：

```text
M_i (c_i - r_i) + lambda * sum_j w_ij (c_i - c_j) = 0
```

整理后：

```text
(M_i + lambda * sum_j w_ij) c_i
  - lambda * sum_j w_ij c_j
  = M_i r_i
```

这就是代码中逐行装配稀疏矩阵的形式。

### 逐顶点矩阵装配

对每个顶点 `i`：

```text
diagonal_i = M_i
rhs_i = M_i * r_i

for each neighbor j:
    diagonal_i += lambda * w_ij
    A_ij = -lambda * w_ij

A_ii = diagonal_i
```

所以最终矩阵满足：

```text
A c = b

A_ii = M_i + lambda * sum_j w_ij
A_ij = -lambda * w_ij
b_i  = M_i r_i
```

这与整体公式：

```text
(M + lambda L)c = M r
```

一致，其中 `lambda = alpha * h^2`。

### 方程含义

该方程可以理解为两个目标的平衡：

```text
数据项:     c 接近 raw curvature r
平滑项:     c 在网格上变化平滑
```

`alpha` 越大，曲率场越平滑；`alpha` 越小，越贴近原始曲率。

### 为什么乘以 h^2

`h` 使用输入网格的平均边长。`lambda = alpha * h^2` 的作用是让扩散强度与模型尺度和网格尺度匹配：

- 如果网格整体缩放，`h^2` 随面积尺度变化，扩散项也随之缩放。
- `alpha` 保持为无量纲调节量，方便跨模型使用同一默认值。
- 当前默认 `alpha=2.0`，表示比原始曲率更平滑，但仍保留主要高曲率结构。

### 工程实现

- cotangent 权重构建稀疏邻接。
- barycentric vertex area 构建质量项。
- 使用 `Eigen::SimplicialLDLT` 分解求解。
- 求解后记录 assembly / factorization / solve 时间、非零元数量和 residual。

### 实际计算流程

```text
输入: mesh, raw curvature r, csf_alpha

1. 检查 mesh 和 raw curvature 是否有效。
2. 计算 mean_edge_length h。
3. 构建 cotangent 权重图:
     w_ij = 0.5 * (cot alpha_ij + cot beta_ij)
   若权重非正或非有限，则替换为很小正数 eta=1e-4。
4. 计算 barycentric vertex mass:
     M_i = sum incident_face_area / 3
5. 设置 lambda:
     lambda = csf_alpha * h^2
6. 装配稀疏矩阵 A 和右端 b:
     A_ii = M_i + lambda * sum_j w_ij
     A_ij = -lambda * w_ij
     b_i  = M_i * r_i
7. 使用 Eigen::SimplicialLDLT 求解:
     A c = b
8. 检查解是否 finite，并计算 residual:
     residual = ||A c - b||
9. 输出 diffused curvature c 和求解统计。
```

### 和普通 Laplacian 平滑的区别

普通显式平滑通常类似：

```text
c_i <- average(neighbor values)
```

这种方法容易受迭代次数影响，也可能过度抹平特征。当前 screened solve 是一次全局稀疏线性求解，同时保留数据项：

```text
c 接近 r，同时 c 在网格上平滑
```

因此它比简单邻域平均更稳定，也更容易通过 `alpha` 控制扩散强度。

### PPT 讲述用简化版

汇报时可以用下面这段话：

```text
我没有直接使用 raw curvature 生成尺寸场，而是先解一个 screened Laplacian 系统。
这个系统一边让扩散后的曲率 c 接近原始曲率 r，一边惩罚相邻顶点曲率差。
最后得到的 c 比 raw curvature 更平滑，可以避免 target h 在相邻三角形之间剧烈跳变。
```

### 建议配图

三列图：

```text
raw curvature -> CSF curvature -> target h
```

讲述时强调 CSF 是连接几何特征和尺寸场的中间层。

## 6. 目标尺寸场计算

### PPT 标题

目标尺寸场：从曲率到目标边长 h

### v2.10.0 方案

v2.10.0 使用 target-face inverse-sqrt 映射：

```text
h_i = lambda / sqrt(kappa_i)
```

并通过 `target_faces` 控制整体密度。未显式指定时：

```text
target_faces = cleaned_input_face_count
```

### v2.10.1 修改

v2.10.1 去掉 robust-mean raw curvature 中的 `log1p` 压缩：

```text
旧: raw_kappa = log1p(mean_curvature * mean_edge_length)
新: raw_kappa = mean_curvature * mean_edge_length
```

T0 上 `target_h max/min` 从 `3.01x` 提高到 `3.38x`，但中间 90% 的 h 差异仍有限。

### 当前 v2.10.2 映射

当前工作区使用固定指数响应：

```text
h_i = clamp(lambda / pow(max(kappa_i, kappa_min), alpha),
            h_min, h_max)

alpha = 0.75
h_min = 0.05 * mean_edge_length
h_max = 5.0 * mean_edge_length
```

说明：

- 曲率越大，目标边长越小。
- `alpha=0.75` 比 `sqrt` 响应更强。
- 该参数目前是实验常量，不应汇报为最终最优参数。

### lambda 的求解

`lambda` 不是手动调节，而是通过二分求解，使估计面数接近 `target_faces`：

```text
estimated_faces += face_area / (sqrt(3)/4 * h_face^2)
```

其中 `h_face` 是三角形三个顶点目标 h 的平均值。

### 为什么 lambda 可以二分

在当前公式中：

```text
h_i(lambda) = clamp(lambda / kappa_i^alpha, h_min, h_max)
```

当 `lambda` 增大时，每个顶点的目标边长 `h_i` 不会变小，只会变大或因为 clamp 保持不变。因此三角形上的 `h_face` 也不会变小。

估计面数使用：

```text
estimated_faces = sum_f area_f / (sqrt(3)/4 * h_f^2)
```

所以 `h_f` 越大，单位面积允许的三角形越少，估计面数越小。也就是说：

```text
lambda 越小 -> h 越小 -> estimated_faces 越大
lambda 越大 -> h 越大 -> estimated_faces 越小
```

这构成了一个单调关系，因此可以用二分法寻找使 `estimated_faces` 接近 `target_faces` 的 `lambda`。

### lambda 的上下界

代码先构造两个极端尺寸场：

```text
lambda = 0
  -> lambda / kappa^alpha = 0
  -> 所有顶点被 clamp 到 h_min
  -> 得到最高密度 max_density_faces
```

```text
hi_lambda = h_max * max_kappa^alpha
```

因为最大曲率点的目标边长为：

```text
h = hi_lambda / max_kappa^alpha = h_max
```

其他曲率更小的点会得到更大的理论 h，然后被 clamp 到 `h_max`。因此：

```text
lambda = hi_lambda
  -> 所有顶点被 clamp 到 h_max
  -> 得到最低密度 min_density_faces
```

这就得到二分区间：

```text
lo = 0
hi = h_max * max_kappa^alpha
```

### 边界情况

如果用户要求的 `target_faces` 超过 `h_min` 能达到的最高密度：

```text
target_faces >= max_density_faces
```

代码直接使用：

```text
lambda = 0
h_i = h_min
```

表示已经达到允许的最小目标边长，无法再通过尺寸场生成更高理论面数。

如果用户要求的 `target_faces` 小于 `h_max` 对应的最低密度：

```text
target_faces <= min_density_faces
```

代码直接使用：

```text
lambda = hi_lambda
h_i = h_max
```

表示已经达到允许的最大目标边长，无法再通过尺寸场生成更低理论面数。

### 二分过程

在正常情况下，目标面数位于两个极端之间：

```text
min_density_faces < target_faces < max_density_faces
```

代码执行 64 次二分：

```text
lo = 0
hi = hi_lambda

repeat 64 times:
    mid = 0.5 * (lo + hi)
    h_i = clamp(mid / kappa_i^alpha, h_min, h_max)
    estimate = estimateTargetFaceCount(h)

    if estimate > target_faces:
        # 面数估计太多，说明 h 还太小，需要增大 lambda
        lo = mid
    else:
        # 面数估计太少，说明 h 太大，需要减小 lambda
        hi = mid

lambda = 0.5 * (lo + hi)
```

最终用这个 `lambda` 重新计算每个顶点的 `target_h`，并记录 `estimated_faces`。

### PPT 讲述用简化版

汇报时可以这样讲：

```text
lambda 控制整体网格密度。曲率只决定哪里更密、哪里更疏；
lambda 决定整体 h 的尺度。我们不是手动调 lambda，而是用目标面数反推它。
先计算全 h_min 时的最高理论面数，再计算全 h_max 时的最低理论面数。
如果目标面数在两者之间，就利用 estimated_faces 随 lambda 单调下降的性质二分求解。
```

### 需要特别说明

`target_faces` 只用于求尺寸场的理论预算，不等于最终输出面数。最终面数还受 split/collapse/flip 迭代次数、局部质量保护、特征保护和边界条件影响。

### 建议配图

- 曲率到 h 的函数曲线：`alpha=0.5` 与 `alpha=0.75` 对比。
- T0 或 T44 的 target_h heat map。

## 7. 尺寸场查询

### PPT 标题

目标 h 如何作用到后续重网格顶点

### 问题

target h 初始定义在输入网格顶点上，但 remesh 过程中会产生新顶点，原始顶点也会移动。因此每次局部操作都需要查询当前点对应的目标 h。

### 当前方法

对任意查询点 `p`：

1. 用 Libigl AABB 在输入网格上找最近三角形。
2. 得到最近点 `p'`。
3. 计算 `p'` 在该三角形上的 barycentric 坐标。
4. 用三角形三个顶点的 target h 做 barycentric 插值。

```text
h(p) = b0 * h0 + b1 * h1 + b2 * h2
```

### 设计优点

- 查询连续，不会简单跳到最近顶点。
- 局部操作始终参考原始曲面上的目标尺寸场。
- 新插入点和移动点都能获得 target h。

### 设计限制

- 尺寸场只在 remesh 开始时构建一次。
- 后续不会根据当前拓扑重新估计曲率。
- 如果局部拓扑已经偏离目标，尺寸场本身不会主动修正。

### 建议配图

画一个三角形，标出三个顶点的 `h0/h1/h2`，一个查询点通过 barycentric 插值得到 `h(p)`。

## 8. Split 算子约束

### PPT 标题

Split：消除过长边

### 触发条件

```text
edge_length > split_factor * min(h_a, h_b)
```

当前默认：

```text
variant = code
split_factor = 1.65
```

### 操作方式

1. 遍历所有边，收集超过阈值的长边。
2. 按边长从大到小排序。
3. 每个 pass 只选择不共享端点的边。
4. 对选中边批量 split，在中点插入新顶点。
5. 重建受影响三角形。

### 为什么批量 split

- 逐条 split 在大网格上很慢。
- 批量选择不共享端点的边，可以减少冲突。
- 优先处理最长边，有利于快速降低极端 edge-target ratio。

### 需要说明的局限

当前 split 主要按长度和 target h 判断，没有显式 feature edge 标记继承机制。因此特征保留主要依赖后续 feature strength 和局部操作保护，而不是完整的 feature graph。

### 建议配图

一条长边被分裂成两条边，旁边标出：

```text
L > 1.65 * min(h_a, h_b)
```

## 9. Collapse 算子约束

### PPT 标题

Collapse：消除过短边，同时保护拓扑和质量

### 触发条件

```text
edge_length < collapse_factor * target_reference
collapse_factor = 0.8
```

`target_reference`：

- `variant=code`：使用 `min(h_a, h_b)`。
- `variant=paper`：使用 `max(h_a, h_b)`。

### 候选选择

- 跳过边界顶点。
- 候选边按 `length / target` 从小到大排序。
- 每个 pass 选择 footprint 不冲突的边。
- footprint 包括两个端点及其一环邻居。

### 有效性约束

collapse 前必须满足：

- 边存在 incident faces。
- 非流形边跳过。
- link condition 成立。
- collapse 后相关邻边不能超过 split 阈值。
- 相关局部三角形不能退化。
- 局部最小质量不能低于 `min_quality_after_collapse=0.05`。

### feature-aware collapse

先计算每个顶点的 feature strength：

```text
feature = (angle_defect_feature + 1) * (dihedral_feature + 1) - 1
```

如果两个端点特征强度差异明显：

```text
abs(fa - fb) >= omega * max(fa, fb)
omega = 0.15
```

则 collapse 点保留在特征更强的端点；否则取中点。

### 汇报判断

collapse 是最容易破坏特征和拓扑的操作，所以当前约束最多。v2.6.0 的 model_001 特征丢失现象说明，仅依赖目标 h 不够，必须加入 feature-aware 和局部质量保护。

### 建议配图

展示短边 collapse 前后，并用红色标注被保护的高特征顶点。

## 10. Flip 与 Relocation 约束

### PPT 标题

Flip 和 Relocation：优化连接关系与顶点位置

### Flip 约束

只处理内部边：

```text
incident_face_count == 2
```

新边必须满足：

- 新边不存在。
- 新边长度在合法范围内：

```text
collapse_factor * target <= new_edge_length <= split_factor * target
```

- flip 后 valence energy 必须下降。
- flip 前后二面角不能超过 `max_flip_dihedral_degrees=45`。
- 默认避免生成钝角三角形。

### Valence energy

目标 valence：

```text
内部点: 6
边界点: 4
```

只有 flip 后四个相关点的 valence 偏差总和下降，才接受 flip。

### Relocation 约束

先把顶点分为：

- `Corner`
- `Crease`
- `Smooth`

分类依据来自 feature strength 和一环邻居中的重要特征邻点数量。

当前 tangent smoothing 只处理 `Smooth` 点：

```text
p_new = p + tangent_lambda * tangent_displacement
tangent_lambda = 0.6
```

MLS relocation 当前默认关闭：

```text
enable_relocation_mls = false
```

如果开启，只对 smooth 点投影。

### 建议配图

- Flip 前后两三角形示意。
- 一个顶点切平面 smoothing 示意。

## 11. 实验演进与结论

### PPT 标题

实验演进：从曲率定义到尺寸场响应

### v2.6.0：曲率模式与特征问题

结论：

- robust-mean 曲率场可用，能增强高曲率响应。
- 但 T21 上 CD 没有改善，violation 上升。
- model_001 仍出现特征约束丢失。
- 判断：只换曲率不够，需要检查局部操作约束。

### v2.10.0：target-face inverse-sqrt

方案：

```text
h = lambda / sqrt(kappa)
target_faces = cleaned input face count
```

全 T 系列对比 v2.8.1：

| 指标 | 结果 |
|---|---:|
| CD 明显改善 | 30 / 31 |
| 平均 dCD% | -20.03% |
| Qmin 明显下降 | 30 / 31 |
| 平均 Qmin | 0.0256859295 |
| 平均 violation | 0.0233352443 |

判断：

- 几何误差改善明显。
- 但三角形最低质量下降，是后续要控制的问题。

### v2.10.1：去掉 log 压缩

T0 结果：

| 指标 | v2.10.0 | v2.10.1 |
|---|---:|---:|
| target_h max/min | 3.01x | 3.38x |
| target_h p95/p05 | 1.43x | 1.50x |
| CD | 1.661244e-05 | 1.661196e-05 |
| Qmin | 0.102274 | 0.040125 |

判断：

- 去 log 是正确方向。
- 但尺寸分层增强有限。
- Qmin 下降明显。

### v2.10.2：指数 alpha=0.75

全 T 系列对比 v2.10.0：

| 指标 | v2.10.2 - v2.10.0 |
|---|---:|
| CD 降低模型数 | 2 / 31 |
| CD 升高模型数 | 29 / 31 |
| 平均 dCD% | +1.72% |
| 平均 dviolation | +0.016980 |
| 平均 h max/min 增量 | +5.023930 |

判断：

- 尺寸场对比度明显增强。
- 但没有带来整体 CD 收益。
- violation 上升，说明尺寸场梯度过强会增加局部操作压力。

### 建议配图

用一张版本演进表：

```text
v2.6.0 robust-mean
  -> v2.10.0 target-face inverse-sqrt
  -> v2.10.1 logless curvature
  -> v2.10.2 inverse-power alpha=0.75
```

## 12. 最终判断与下一步

### PPT 标题

当前判断：框架闭环，但仍需平衡几何误差、质量和特征

### 当前已完成

- 曲率计算：normal-angle 与 robust-mean 两种模式。
- 曲率扩散：screened CSF sparse solve。
- 尺寸场：target-face 驱动的 inverse-power 映射。
- 查询方式：输入曲面 AABB 最近三角形 + barycentric 插值。
- 局部算子：split / collapse / flip / tangent smoothing。
- 诊断输出：raw curvature、CSF curvature、target h、edge/face field stats、summary JSON、CD。

### 当前主要矛盾

不是单一曲率公式的问题，而是三者之间的平衡：

```text
更强自适应尺寸分层
  <-> 更高尺寸场梯度
  <-> 更难满足局部边长约束
  <-> 更容易降低 Qmin 或增加 violation
```

### 推荐下一步

以 v2.10.0 作为稳定基线，做小批量实验：

- 调 `alpha`：例如 `0.55 / 0.60 / 0.65 / 0.70`，不要直接只看 `0.75`。
- 加曲率 cap：例如 p99 或 p99.5，避免少数尖峰支配 lambda。
- 加强 feature constraint：尤其是 feature edge 的显式识别、继承和 flip/collapse 禁止规则。
- 控制最终面数偏差：target_faces 是理论尺寸场预算，不等于最终输出面数。
- 先用 T0 / T21 / T125 / model_001 小批量验证，再跑全 T 系列。

### 汇报结论句

当前实现已经形成完整的曲率驱动重网格闭环。实验表明，增强曲率响应可以提高自适应尺寸分层，但不会自动改善几何误差；下一阶段应围绕尺寸场平滑度、局部算子约束和特征保留做联合优化。

## 附录 A：当前默认参数

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `variant` | `code` | 工程默认路径 |
| `curvature_mode` | `robust-mean` | 当前默认曲率 |
| `csf_alpha` | `2.0` | CSF 平滑强度 |
| `target_faces` | 输入清理后面数 | 未显式指定时 |
| `target_curvature_exponent` | `0.75` | v2.10.2 实验常量 |
| `h_min` | `0.05 * mean_edge_length` | 目标边长下界 |
| `h_max` | `5.0 * mean_edge_length` | 目标边长上界 |
| `split_factor` | `1.65` | code variant |
| `collapse_factor` | `0.8` | 短边收缩阈值 |
| `max_flip_dihedral_degrees` | `45` | flip 二面角约束 |
| `feature_aware_collapse` | `true` | collapse 点保护 |
| `collapse_feature_omega` | `0.15` | 特征强度差异阈值 |
| `min_quality_after_collapse` | `0.05` | collapse 后局部质量下限 |
| `tangent_lambda` | `0.6` | tangent smoothing 步长 |
| `relocation_mls` | `false` | 当前默认关闭 |

## 附录 B：局部算子约束汇总表

| 算子 | 触发条件 | 接受条件 / 保护 |
|---|---|---|
| Split | `L > split_factor * min(h_a,h_b)` | 候选边按长度降序；同 pass 不共享端点；批量 split |
| Collapse | `L < 0.8 * target_reference` | 跳过边界；跳过非流形；link condition；局部质量下限；collapse 后不产生长边；feature-aware collapse point |
| Flip | 内部边，incident face 为 2 | 新边不存在；新边长度合法；valence energy 下降；二面角不超过 45 度；避免钝角 |
| Smoothing | smooth 类顶点 | 只做切向位移；corner/crease 不平滑；步长 `0.6` |
| MLS relocation | 默认关闭 | 如果开启，只投影 smooth 点，避免直接移动 feature/corner |

## 附录 C：建议 PPT 图表清单

1. 算法整体流程图。
2. raw curvature / CSF curvature / target h 三联图。
3. normal-angle 与 robust-mean 曲率 heat map 对比。
4. `alpha=0.5` 与 `alpha=0.75` 的曲率到 h 映射曲线。
5. split / collapse / flip 三个局部算子示意图。
6. model_001 特征丢失对比图。
7. v2.10.0、v2.10.1、v2.10.2 关键指标表。
8. 最终 tradeoff 图：CD、Qmin、violation、h max/min。

## 附录 D：汇报时避免的表述

- 不要说 v2.10.2 是最终最优方案。更准确的说法是：v2.10.2 是增强尺寸场响应的探索实验。
- 不要只说 robust-mean 改善了曲率。要说明它没有单独解决特征丢失，需要配合局部算子约束。
- 不要把 `target_faces` 说成最终输出面数。它是尺寸场二分求解的理论预算，最终面数还受 split/collapse/flip 迭代影响。
- 不要只展示 CD。需要同时展示 Qmin 和 violation，否则会掩盖质量退化。
