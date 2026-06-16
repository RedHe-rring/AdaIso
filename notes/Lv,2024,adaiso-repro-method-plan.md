# Lv, 2024, AdaIso 复现级 Method 与代码复现计划

论文：Chenlei Lv, Weisi Lin, Jianmin Zheng, **Adaptively Isotropic Remeshing Based on Curvature Smoothed Field**, IEEE TVCG, 2024.

相关笔记：[Lv,2024,adaiso.md](Lv,2024,adaiso.md)

作者代码：`https://github.com/vvvwo/Adaptively-Isotropic-Remeshing`

本文档依据：

- 本地 PDF：`papers/Lv 等 - 2024 - Adaptively Isotropic Remeshing Based on Curvature Smoothed Field.pdf`
- 作者仓库浅克隆检查版本：`53cbd9afd429e67bd1736e63594797cea702ceb0`，提交时间 `2022-05-07 15:52:32 +0800`

## 1. 原笔记的 Method 能否支撑复现

不能完全支撑复现。

原笔记适合理解论文，但不足以直接复现，原因是它缺少以下信息：

- 数据结构：顶点、面、邻接表如何存储和更新。
- 具体参数：迭代次数、直方图 bin 数、边分裂/折叠阈值、平滑步数、切向平滑权重、MLS 投影邻居数等。
- 操作顺序：每轮到底是 split -> collapse -> face check -> neighbor check -> flip -> tangent smoothing，还是别的顺序。
- 终止条件：CSF 平滑何时停止、split/collapse/flip 的内部 while 何时停止、整体重网格化跑几轮。
- 边界处理：边界点如何识别，哪些点不能 collapse 或 tangent smoothing。
- 作者源码与论文公式的差异：源码中部分曲率、阈值和平滑实现并不完全等同于论文文字。
- 实验复现计划：如何准备 SHREC/ModelNet、如何运行 baseline、如何计算 `Hd/Md/Qavg/Qmin/curvature sensitivity/runtime`。

因此需要单独写一个复现级 method。下面按“论文忠实复现”和“作者源码等价复现”两条线组织。若目标是复现论文思想，优先用论文公式；若目标是复现作者公开代码输出，优先按源码细节。

## 2. 复现目标分层

建议把复现分成三个层级，不要一开始就追求论文表格完全一致。

| 层级 | 目标 | 成功标准 |
|---|---|---|
| L0 smoke test | 能运行作者 EXE 或源码，对自带 `hand.obj` 生成结果 | 输出 OBJ，顶点/面数合理，不崩溃 |
| L1 方法复现 | 自己实现 CSF + histogram edge reconnection + sub-remeshing | 在若干网格上生成自适应各向同性结果，指标趋势符合论文 |
| L2 论文实验复现 | 复现 SHREC/ModelNet 上与 baseline 的对比 | 得到 `Hd/Md/Qavg/Qmin/curvature sensitivity/runtime` 表格和图，数值接近论文 |

L2 最难，因为作者仓库只带一个 `hand.obj` 示例，没有论文完整数据、baseline 代码和评估脚本。

## 3. 输入、输出和核心数据结构

### 3.1 输入

输入是三角网格：

```text
V = {v_i in R^3}
F = {(i, j, k)}
```

作者代码使用 VCG 的 importer 读取 `obj` 和 `ply`。源码入口参数是：

```text
SHFVCGRemesh_Hist.exe <input_mesh> <preprocess> <mode> <meshScale>
```

参数含义：

| 参数 | 含义 |
|---|---|
| `input_mesh` | 输入网格路径，支持 `obj`、`ply` |
| `preprocess` | `0` 不做预处理；`1` 做重复点/面检查、三角形检查和 sub-remeshing |
| `mode` | `1` isotropic；`2` adaptively isotropic |
| `meshScale` | 目标边长相对平均边长的缩放，README 说默认 `1`，越小越精细 |

示例：

```powershell
.\SHFVCGRemesh_Hist.exe Data/Mesh/hand.obj 0 2 1
```

### 3.2 输出

作者代码输出：

```text
<input_file_without_ext>_Remesh_<meshScale>.obj
```

例如：

```text
hand_Remesh_1.000000.obj
```

作者仓库自带样例计数：

| 文件 | 顶点数 | 法向数 | 面数 |
|---|---:|---:|---:|
| `hand.obj` | 20470 | 20470 | 40936 |
| `hand_Remesh_1.000000.obj` | 21605 | 0 | 43206 |

### 3.3 邻接结构

复现时建议维护以下结构：

```text
vertices: vector<Vec3>
faces: vector<Triangle>
pointNeighbor[i]: flat vector storing pairs (p2, p3) for each incident face (i, p2, p3)
pointEdge[i]: unique 1-ring vertex list
boundaryVertices / points_Keep: boundary points that should not be collapsed or tangentially smoothed
localTargetLength[i]: 每个顶点的目标局部边长
```

作者源码中 `pointNeighbor[i]` 不是单纯邻接点列表，而是每两个元素表示一个以 `i` 为中心的邻接三角形：

```text
pointNeighbor[i] = [b2, b3, b4, b5, ...]
means faces (i, b2, b3), (i, b4, b5), ...
```

这个设计会影响 split/collapse/flip 的更新方式。若重新实现，建议用 half-edge 或 `trimesh`/`OpenMesh` 这类结构简化实现；若对齐作者源码，需要复刻这种 pair-list 邻接。

## 4. 复现级 Method

### 4.1 总流程

```text
Load mesh
Build adjacency
Optionally preprocess:
  remove repeated vertices/faces
  check/repair triangle connectivity
  sub-remeshing for long edges
Compute average edge length L_ave
Compute CSF or code-equivalent smoothed curvature
Build histogram and assign local target edge length
Build KD-tree over original/current target-length vertices
Repeat global optimization for 5 iterations:
  split long edges
  collapse short edges
  remove invalid/repeated faces
  repair neighbor structure
  flip edges to improve valence
  tangent smoothing
  update boundary vertices
Rebuild face list
Project vertices back to original surface with MLS
Save OBJ
Evaluate
```

### 4.2 预处理：TriangleRemoveRepeat、TriangleCheck、Sub-remeshing

论文只重点描述 sub-remeshing；作者代码在 `preprocess=1` 时额外调用：

```text
TriangleRemoveRepeat
TriangleCheck
SubRemeshing
```

复现最低要求：

1. 去掉重复顶点或完全重复三角形。
2. 删除退化三角形：任意两顶点相同、面积接近 0、索引非法。
3. 修复或移除孤立面、重复面、方向明显冲突的面。
4. 对极长边执行 sub-remeshing。

Sub-remeshing 论文版：

1. 计算平均边长 `Eave`。
2. 找出 `length(e) > 2 * Eave` 的边。
3. 对每条长边插入 `floor(length(e) / Eave)` 个点。
4. 对包含长边的三角形按插点重新连接。
5. 生成新三角形时尽量避免钝角。
6. 重复检查新边，直到没有超过阈值的极长边。

Sub-remeshing 作者代码版：

1. `SHFRemeshing_SubRemeshing()` 中调用 `Sub_Remeshing_init(pointSet, faceSet, 1.5)`。
2. 因此实际使用 `Eave_code = 1.5 * Eave`。
3. 长边检测为 `length(e) > 2 * Eave_code`。
4. 插点数为 `int(length(e) / Eave_code)`。
5. 插点位置按线性比例放在边上。
6. 对每个受影响三角形，先找最长边，再按最长边与另外两条边上的插点关系重连。
7. 若三角形中间角 `angMiddle > 1.5` 弧度，按钝角三角形逻辑建立配对点。
8. `Sub_TriangularNormal` 和 `Sub_Triangular_Insert_NewFace` 用于维持新三角形方向。

复现建议：

- 若输入是普通研究数据，先用 `preprocess=0` 与论文主流程对齐。
- 若输入是 CAD 或人工编辑网格，使用 `preprocess=1`。
- 自己实现时保留 `subremesh_multiplier = 1.5` 作为可配置项。

### 4.3 曲率与 CSF

#### 4.3.1 论文版曲率

对每个顶点 `x_i`，计算 area-weighted vertex normal `N(x_i)`，然后定义：

```math
s(x_i) = max_{x_j in R(x_i)} arccos(N(x_i) dot N(x_j))
```

其中 `R(x_i)` 是去重后的 1-ring 邻居。

这个定义强调最大法向变化，适合捕捉局部高曲率。

#### 4.3.2 论文版 CSF 平滑

对每个顶点：

```math
tilde{s_i} = sum_{j in R_i} (w_ij / w_i) * s_j - s_i
w_i = sum_{j in R_i} w_ij
```

cotangent weight：

```math
w_ij = (cot alpha + cot beta) / 2, if cot alpha + cot beta > 0
w_ij = h, otherwise
```

默认：

```text
h = 0.0001
mu = 0.5
```

更新：

```math
s_i <- s_i + (1 - mu) * tilde{s_i}
```

为了防止所有曲率收敛为常数：

1. 固定曲率最大顶点 `vmax`。
2. 固定一个不在 `vmax` 的 1-ring 邻域内的曲率最小顶点 `vmin`。
3. 对其他点迭代平滑。

终止条件建议：

```text
max_iter_csf = 20
stop if mean(abs(tilde{s_i})) < 0.005
```

论文图 15 说明大多数测试模型 8 次迭代后可达到平均更新量小于 `0.005`。

#### 4.3.3 作者代码版 CSF/曲率近似

源码中的实现与论文公式有差异。核心在 `Mesh_Geometric.cpp::MeshGeometric_Harmonic_N_Value()` 和 `AdpIsotropic.cpp::MeshOptimization_Apt_L_init()`。

作者代码主要步骤：

1. 对每个顶点计算 area-weighted vertex normal。
2. 对每个 incident triangle 的两个邻点，计算当前顶点法向与邻点法向的夹角。
3. 初始曲率类似：

```text
N_Value[i] = angleSum / pointNeighbor[i].size()
```

注意：这里不是论文公式中的 1-ring 最大法向夹角。

4. 构造 cotangent weight：

```text
angle = innerAngle(opposite, i, neighbor)
tanValue = tan(abs(angle))
tanValue clipped to [0.1, 10]
ctan_weight += 1 / tanValue
```

5. 加入距离权重并归一化：

```text
w_ij = (ctan_weight_ij / distance_ij) / sum_j(ctan_weight_ij / distance_ij)
```

6. 平滑 3 次，`lambda = 0.5`：

```text
HN[i] = lambda * HN[i] + (1 - lambda) * sum_j(w_ij * HN[j])
```

7. 在 `AdpIsotropic.cpp` 中又做 3 次邻域平均，权重是边长：

```text
curv[i] = sum_j((edgeLength(i,j) / sumEdgeLength) * curv_old[j])
```

复现建议：

- 要复现论文文字：用 4.3.1 和 4.3.2。
- 要复现作者代码输出：用 4.3.3。
- 两条路线的结果不会完全一致，文档和实验中必须标注。

### 4.4 直方图分组与目标边长

论文版：

1. 对 CSF 曲率值做直方图。
2. 取计数最多的 bin 作为 base value；如果多个最大值，取中间位置。
3. 以 base value 为中心，把曲率分成 `k=5` 个区间。
4. 对应乘法因子：

```text
m = {1.8, 1.4, 1.0, 0.8, 0.6}
```

低曲率对应 `1.8`，高曲率对应 `0.6`。目标是让高曲率区域边长更短，低曲率区域边长更长。

作者代码版：

1. 固定使用 40 个直方图 bin：

```text
unitStep = (max_curv - min_curv) / 40
```

2. 找计数最多的 bin，取该 bin 中点为 `c_base`。
3. 在排序后的曲率数组中找到 `c_base` 插入位置 `indexC_Base`。
4. 定义分界：

```text
maxCu = number_of_vertices
unitMax_indexC_Base = (maxCu - indexC_Base) / 5
li1 = sorted[indexC_Base / 5 * 2]
li2 = sorted[indexC_Base / 5 * 4]
li3 = sorted[indexC_Base + unitMax_indexC_Base]
li4 = sorted[indexC_Base + unitMax_indexC_Base * 3]
```

5. 映射到局部目标边长：

```text
curv < li1              -> localTargetLength = 1.8 * L_ave
li1 <= curv < li2       -> localTargetLength = 1.4 * L_ave
li2 <= curv < li3       -> localTargetLength = 1.0 * L_ave
li3 <= curv < li4       -> localTargetLength = 0.8 * L_ave
curv >= li4             -> localTargetLength = 0.6 * L_ave
```

6. `MeshOptimization_Start(meshScale, 5, false)` 中还会把 `L_ave` 和所有 `localTargetLength` 乘以 `meshScale`：

```text
localTargetLength[i] *= meshScale
```

因此最终局部目标边长是：

```text
L_i = m_i * L_ave * meshScale
```

### 4.5 局部目标边长查询

作者代码为初始点集构建 KD-tree。后续 split/collapse/flip/smoothing 中，对于任意当前点位置 `p`：

```text
nearest original/current target-length vertex index = KDTree.nearest(p)
L(p) = localTargetLength[index]
```

对一条边 `(a, b)`，源码通常使用：

```text
L_edge = min(L(a), L(b))
```

这意味着只要边的一端落在高曲率区域，目标长度就会偏短，从而促使细分。

### 4.6 主循环

作者代码固定执行 5 轮：

```text
for iter in 1..5:
  Split
  Collapse
  FaceCheck
  NeighborCheck
  Flip
  TangentSmoothing
  EdgePointCheck
```

其中 `Split`、`Collapse` 和 `Flip` 内部各自有 while 循环。

### 4.7 Edge split

论文文字：

```math
length(e) > (5/3) * l * min(m_a, m_b)
```

作者源码实际逻辑：

```text
if length(a,b) >= 1.33 * min(L(a), L(b)):
    split edge
```

Split 内部最多重复 10 次，直到没有边满足条件。

操作：

1. 遍历无向边，避免重复处理，使用 `a < b`。
2. 若边长超阈值，把 midpoint 插入顶点表。
3. 若两个端点都是边界点，则新点也标记为边界点。
4. 更新两个端点及相邻三角形的 `pointNeighbor`。
5. 对新增点的邻接做去重。

复现注意：

- 如果用 half-edge，split 会更可靠。
- 若按源码复现，需要小心 `pointNeighbor` pair-list 更新，否则 face 重建会错。

### 4.8 Edge collapse

论文文字：

```math
length(e) < (4/5) * l * max(m_a, m_b)
```

作者源码实际逻辑：

```text
if length(a,b) < 0.8 * min(L(a), L(b)):
    candidate collapse
```

约束：

1. 边的 incident face 数 `valenceE <= 2`。
2. 任一端点是边界点则跳过。
3. 两个端点在本轮都仍 active。
4. collapse 到 midpoint。
5. collapse 后 midpoint 到所有相关邻点的边不能太长：

```text
radius(midpoint, neighbor) <= 1.33 * min(L(midpoint), L(neighbor))
```

Collapse 内部最多重复 10 次。作者主调用 `judge=false`，即不强制恢复到某个目标顶点数，但源码中仍会维护 `CollapseNum = points.size() - originalNumber`。

### 4.9 FaceCheck 与 NeighborCheck

FaceCheck：

- 删除邻接为空或异常的点。
- 更新点索引。
- 更新邻接表。
- 更新边界点索引。

NeighborCheck：

- 遍历 `pointNeighbor` 中的每个三角形 `(b1,b2,b3)`。
- 检查该三角形是否也存在于 `b2` 和 `b3` 的邻接结构中。
- 若缺失，则把缺失的邻接关系补回去。

复现建议：

- 每轮 split/collapse 后必须做拓扑一致性检查。
- 建议额外检查：重复面、反向重复面、孤立点、非流形边、退化三角形。

### 4.10 Edge flip

Flip 对共享边 `(b1,b2)` 的两个三角形：

```text
(b1,b2,b3), (b2,b1,b4)
```

尝试把对角线从 `(b1,b2)` 换成 `(b3,b4)`。

作者源码条件：

1. 新边长度必须在局部目标范围内：

```text
0.8 * min(L(b3), L(b4)) <= length(b3,b4) <= 1.33 * min(L(b3), L(b4))
```

2. valence energy 必须下降。

内部顶点目标 valence：

```text
interior vertex target valence = 6
boundary vertex target valence = 4
```

能量：

```text
E = sum(abs(current_valence(v) - target_valence(v))) for v in {b1,b2,b3,b4}
```

如果 flip 后 `E_after < E_before`，执行 flip。

论文还说若 flip 会生成钝角，则不处理。源码中主要用新边长度范围和 valence energy 控制，没有看到显式钝角检测。因此如果按论文复现，建议增加 obtuse check；如果按代码复现，不额外加。

Flip 内部重复执行，直到一轮没有任何 flip。

### 4.11 Tangent smoothing

作者源码：

```text
lambda = 0.6
```

边界点跳过。

对顶点 `p_i`：

1. 收集去重 1-ring 邻居。
2. 对每个邻居计算一个面积权重，来自相邻三角形面积。
3. 再乘以该邻居的局部目标边长 `L(neighbor)`。
4. 得到加权质心 `g_i`：

```text
g_i = sum_j(weight_j * p_j) / sum_j(weight_j)
weight_j = area_j * L(p_j)
```

5. 计算顶点法向 `n_i`，使用 area-weighted incident face normal。
6. 只保留 `g_i - p_i` 在切平面内的分量：

```text
v = g_i - p_i
normal_component = dot(v, n_i) * n_i
tangent_component = v - normal_component
p_i_new = p_i + lambda * tangent_component
```

### 4.12 边界点处理

作者代码判断边界点：

```text
如果顶点存在一条 incident edge，其 incident face 数为 1，则该顶点是 boundary point
```

边界点：

- 不参与 tangent smoothing。
- collapse 中如果任一端点是边界点，则跳过。
- split 中若边两个端点都是边界点，新 midpoint 也标为边界点。

### 4.13 重建 face list

主循环结束后，从 `pointNeighbor` 重建 `faceInfor`：

1. 遍历每个顶点 `i` 的 pair-list。
2. 生成候选面 `(i, b2, b3)`。
3. 将同一三角形在其他顶点邻接中的重复记录标成 `-1`，避免重复输出。
4. 最终输出 OBJ。

### 4.14 MLS 投影回原始表面

作者代码在自适应模式结束后执行：

```text
Update point location
for each output point:
    MeshOptimization_PointUpdate(point)
```

作用：把重网格化后的顶点投回原始网格附近，提高几何一致性。

源码流程：

1. 对原始点集建立 PCL KD-tree。
2. 对输出点查询 `K=8` 个最近邻。
3. 如果最近邻距离为 0，直接返回该点。
4. 否则用最多 `10` 次迭代的 MLS 风格投影。
5. 停止阈值：

```text
errorExist = 0.0001
```

6. 权重：

```text
w_j = exp(-(distance(p, q_j) / h)^2)
```

7. 计算加权平均点 `a(p)` 和加权平均法向 `n(p)`。
8. 更新：

```text
p <- p_original - dot(n(p), p_original - a(p)) * n(p)
```

重要风险：

- `SHFRemesh.cpp` 中成员变量 `double h;` 没有在该文件内明显初始化。
- `geodesic.hpp` 中有类似 `h = dis_h + (dis_h - dis_h2)` 的逻辑，但 `SHFRemesh.cpp` 未直接使用。
- 复现作者源码时需要检查运行时 `h` 的实际值。若重写实现，建议设置：

```text
h = distance to the 8th nearest original point
```

或者：

```text
h = d_K + (d_K - d_{K-1})
```

这与仓库其他模块中的写法一致。

## 5. 论文指标复现方法

### 5.1 几何一致性：Hd 与 Md

论文指标：

```text
Hd = Hausdorff distance
Md = mean distance
```

实现建议：

1. 用 `trimesh` 或 CGAL/AABB tree 实现 point-to-triangle distance。
2. 为避免只比较顶点导致偏差，对两个网格表面都均匀采样点。
3. 计算：

```text
Hd = max(max_{p in sampled original} d(p, remesh), max_{q in sampled remesh} d(q, original))
Md = mean_{q in sampled remesh} d(q, original)
```

4. 若要尽量贴近论文，距离映射使用 MLS surface mapping；若工程上先复现趋势，可先用 point-to-triangle distance。

### 5.2 网格质量：Qavg 与 Qmin

每个三角形：

```math
Q(t) = (6 / sqrt(3)) * S_t / (p_t * h_t)
```

其中：

- `S_t` 是三角形面积。
- `p_t` 是半周长。
- `h_t` 是最长边。
- `Q=1` 表示等边三角形。

统计：

```text
Qavg = mean(Q(t))
Qmin = min(Q(t))
```

### 5.3 曲率敏感性

复现论文 Fig. 12：

1. 对 remeshed mesh 计算每个顶点曲率，建议先用论文 Eq. (1)。
2. 对每个三角形取三个顶点曲率平均值。
3. 记录三角形面积。
4. 面积归一化到 `[0,1]`：

```text
A_norm = (A - A_min) / (A_max - A_min)
```

5. 按曲率分 bin，计算每个 bin 的平均归一化面积。
6. 理想曲线应随曲率增加而下降。

可补充一个标量指标，便于自动比较：

```text
Spearman correlation(curvature, -area)
```

值越大，说明曲率越高面积越小的单调关系越强。

### 5.4 运行时间

记录：

```text
load_time
preprocess_time
csf_time
split_collapse_flip_smooth_time
mls_projection_time
total_time
input_vertices
output_vertices
input_faces
output_faces
meshScale
preprocess flag
```

论文报告平台是：

```text
Intel Xeon W-2133 3.6GHz
32 GB RAM
Quadro P620
Windows 10
Visual Studio 2019 64-bit
```

如果复现平台不同，运行时间只比较趋势，不要求数值一致。

## 6. 代码复现详细计划

### 阶段 A：锁定复现标准

1. 明确目标：

```text
目标 1：复现作者公开代码输出。
目标 2：复现论文算法思想。
目标 3：复现论文实验表格。
```

2. 本项目建议先做目标 1 和目标 2，再做目标 3。
3. 在报告中明确区分：

```text
Paper-faithful implementation
Code-faithful implementation
```

### 阶段 B：建立目录结构

建议新增：

```text
repro/
  author_code/
  data/
    raw/
      shrec/
      modelnet/
      examples/
    processed/
  outputs/
    ours_author/
    ours_reimpl/
    baselines/
  scripts/
    run_author_exe.ps1
    run_reimpl.py or run_reimpl.cpp
    eval_quality.py
    eval_distance.py
    plot_curvature_sensitivity.py
    collect_results.py
  reports/
    figures/
    tables/
```

不要直接把临时克隆目录当正式复现目录。正式复现时应重新 clone 并记录 commit hash。

### 阶段 C：跑通作者 EXE

1. 克隆作者仓库：

```powershell
git clone https://github.com/vvvwo/Adaptively-Isotropic-Remeshing.git repro/author_code
cd repro/author_code
git checkout 53cbd9afd429e67bd1736e63594797cea702ceb0
```

2. 进入 `EXE` 目录。
3. 使用仓库自带 `hand.obj` 测试。注意 Windows 下建议用正斜杠路径：

```powershell
.\SHFVCGRemesh_Hist.exe ../SHFVCG2021remesh_Hist/Data/Mesh/hand.obj 0 2 1
```

4. 检查输出 OBJ 是否生成。
5. 统计输出顶点和面数，参考仓库自带结果：

```text
hand_Remesh_1.000000.obj: 21605 vertices, 43206 faces
```

6. 如果 EXE 因 DLL 缺失失败：

- 先检查 `EXE` 目录已有 DLL。
- 用 Dependencies 或 `dumpbin /dependents` 查缺失 DLL。
- 重点补齐 PCL debug DLL、GMP、MPFR、VC runtime。

### 阶段 D：从源码构建作者代码

源码构建适合要修改/调试 method 的情况。

环境：

```text
Windows 10/11
Visual Studio 2019, PlatformToolset v142
Windows SDK 10.0
CUDA 11.1, 如果保留 CUDA 文件
VCG library
PCL 1.8+，且最好匹配 Debug x64 库名
CGAL
Eigen3
Boost 1.67+
GMP/MPFR
```

工程文件检查到的配置：

```text
PlatformToolset: v142
WindowsTargetPlatformVersion: 10.0
OpenMPSupport: true
AdditionalIncludeDirectories: \VCG; \Eigen3; CUDA v11.1 include
AdditionalDependencies: pcl_*_debug.lib, libgmp-10.lib, libmpfr-4.lib
```

构建步骤：

1. 打开 `SHFVCG2021remesh_Hist/SHFVCGRemesh_OM.sln`。
2. 选择 `x64 Debug`。
3. 修改 include/lib 路径，不要使用工程里写死的 `\VCG`、`\Eigen3`。
4. 如果不做 CUDA，对 `.cu` 文件和 CUDA build customization 做隔离或移除。
5. 编译。
6. 用 `hand.obj` 跑通。
7. 对比 EXE 输出与源码输出顶点/面数、指标和视觉结果。

代码审计项：

- `SHFRemesh.cpp` 中 MLS 投影的 `h` 是否初始化。
- `MeshOptimization_FaceCheck()` 中 `if (bi1New > 0 && bi2New > 0)` 是否错误跳过索引 0。
- collapse 后是否可能产生非流形边。
- flip 是否需要补论文中的 obtuse check。

### 阶段 E：实现自己的复现版

如果目标是可维护复现，建议重新实现，而不是直接维护作者源码。

推荐语言：

- C++：OpenMesh/libigl/CGAL，性能好，贴近论文。
- Python：trimesh + numpy + scipy，开发快，适合先验证指标趋势。

模块拆分：

```text
mesh_io
adjacency
geometry
sub_remeshing
curvature_csf
histogram_target_length
edge_operations
projection
metrics
visualization
runner
```

实现顺序：

1. `mesh_io`：读取/写出 OBJ/PLY。
2. `adjacency`：构建 vertex-face、vertex-vertex、edge-face、boundary vertex。
3. `geometry`：面积、边长、法向、角度、三角形质量。
4. `curvature_csf`：先实现论文版 Eq. (1)-(4)，再可选实现代码版。
5. `histogram_target_length`：实现 `k=5` 和 40-bin 代码版。
6. `sub_remeshing`：先只处理长边，再实现钝角/插点配对。
7. `edge_operations`：split/collapse/flip。
8. `tangent_smoothing`：切平面平滑。
9. `projection`：MLS 投影。
10. `metrics`：Hd/Md/Qavg/Qmin/曲率敏感性。

每个模块必须有小测试：

- 单三角形：面积、法向、邻接。
- 正四面体或二十面体：valence 和边界检测。
- 平面网格：CSF 应接近低曲率。
- 球面网格：曲率分布应较均匀。
- 人工长边三角形：sub-remeshing 后最长边降低。

### 阶段 F：复现论文实验

1. 数据准备：

- 下载 SHREC 2010 或与论文一致的 SHREC 模型。
- 下载 ModelNet，并转换到三角网格格式。
- 选取论文中的类别和模型：SHREC 的 centaur/crane/elephant/hand/tyrannosaurus/ant/bird/crocodile/light/spider；ModelNet 的 bed/bottle/bathtub/bowl/car/airplane/person/sofa。

2. 统一预处理：

- 确保单位尺度一致。
- 确保面片均为三角形。
- 删除重复顶点、退化三角形。
- 记录输入顶点/面数。

3. 运行作者方法：

```powershell
.\SHFVCGRemesh_Hist.exe <mesh.obj> 0 2 1
.\SHFVCGRemesh_Hist.exe <mesh.obj> 1 2 1
```

两组都跑，确认论文实验更接近哪一组。

4. 输出点数控制：

论文公平比较要求输出约 `9000-13000` 点。用 `meshScale` 扫描：

```text
meshScale in {0.6, 0.8, 1.0, 1.2, 1.5}
```

选输出点数落在目标范围内的结果。

5. Baseline：

需要获取或实现：

- Adaptive Remeshing [13]
- CVT [49]
- Particle-based resampling [43]
- Delaunay-GPU [28]

如果无法获得原实现，至少记录替代实现来源，并在报告中标注“不完全等价 baseline”。

6. 指标计算：

对每个输入/方法输出：

```text
vertices_out
faces_out
Hd
Md
Qavg
Qmin
curvature_area_spearman
runtime
crash_or_nan
```

7. 生成表格：

- SHREC 几何一致性表。
- SHREC 网格质量表。
- ModelNet 几何一致性表。
- ModelNet 网格质量表。
- 运行时间表。

8. 生成图：

- remeshing 可视化。
- minimum inner angle color map。
- curvature-area 曲线。
- runtime 曲线。

### 阶段 G：验收标准

L0：

- 作者 EXE 能对 `hand.obj` 输出 OBJ。
- 输出无明显空文件、无非法 face index。

L1：

- 自己实现能在 `hand.obj`、球、bunny 或任意 5 个网格上稳定运行。
- `Qavg` 相比输入提升。
- 高曲率区域平均面积低于低曲率区域。
- 输出没有大量退化三角形。

L2：

- 至少 18 个论文模型均跑通或明确记录失败原因。
- 作者方法相对 baseline 的趋势与论文一致。
- 表格中 `Nan`/crash 情况按实际记录。
- 论文中的“不是所有单项都第一，但综合更平衡”的结论可从指标复现中支持。

## 7. 风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| 作者仓库缺论文完整数据 | 不能直接复现表格 | 自行下载 SHREC/ModelNet，记录模型来源和预处理 |
| baseline 代码不可得 | 对比表难以严格复现 | 找公开替代实现，报告中标注差异 |
| 源码与论文公式不一致 | 结果不一致 | 分 paper-faithful 和 code-faithful 两条线 |
| MLS 中 `h` 未初始化 | 输出可能不稳定 | 审计并显式设置 `h` |
| 旧依赖难装 | 源码构建成本高 | 先用 EXE smoke test，再做重实现 |
| edge operation 容易破坏拓扑 | 非流形或崩溃 | 用 half-edge 结构，加入拓扑不变量检查 |

## 8. 最小可执行复现路线

如果只想尽快得到一个可信结果，按这个顺序做：

1. 使用作者 EXE 跑 `hand.obj`。
2. 统计输出顶点/面数，与仓库示例输出比较。
3. 写 `eval_quality.py` 计算 `Qavg/Qmin`。
4. 写 `plot_curvature_sensitivity.py` 画 curvature-area 曲线。
5. 对 3 个自选网格运行 `meshScale={0.8,1.0,1.2}`。
6. 用结果确认：高曲率处三角形更小，整体三角形质量提升。
7. 再开始重写 method 或扩展到论文数据集。

## 9. 复现报告建议结构

最终报告可以按以下结构写：

```text
1. Environment
2. Data
3. Implementation Variant
   3.1 Paper-faithful
   3.2 Code-faithful
4. Parameters
5. Results
   5.1 Geometry consistency
   5.2 Mesh quality
   5.3 Curvature sensitivity
   5.4 Runtime
6. Difference from Paper
7. Failure Cases
8. Conclusion
```

必须明确写出：

- 使用的是作者 EXE、作者源码，还是重实现。
- 是否启用 `preprocess=1`。
- `meshScale` 取值。
- CSF 是论文版还是源码版。
- 是否修复了 MLS 的 `h`。
- baseline 是否为原论文同一实现。

