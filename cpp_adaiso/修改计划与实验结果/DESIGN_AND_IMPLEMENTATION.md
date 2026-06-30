# AdaIso C++ 复现代码设计与实现说明

本文档说明 `cpp_adaiso/` 当前 C++17 版本的工程结构、算法实现口径、关键模块、测试状态和后续可优化点。

## 1. 复现目标

本项目复现 Lv 等人在 2024 年论文 *Adaptively Isotropic Remeshing Based on Curvature Smoothed Field* 中的自适应各向同性重网格化流程。

当前 C++ 实现的目标是：

- 以“贴近作者源码行为”为主，而不是逐行移植作者原始 C++。
- 提供可构建、可运行、可测试的命令行工具 `adaiso_cli`。
- 支持常见三角网格输入并统一输出 OBJ。
- 保留论文和作者源码中的关键算法结构：曲率平滑场、自适应目标边长、split/collapse/flip/smooth 全局迭代、MLS 投影。
- 避免复刻作者源码中较脆弱的数据结构和未初始化变量行为。

当前默认参数偏向作者源码行为：

```text
variant = code
mesh_scale = 1.0
global_iterations = 5
target_min_factor = 0.6
target_max_factor = 1.8
target_gamma = 1.0
split_factor = 1.33
collapse_factor = 0.8
tangent_lambda = 0.6
mls_k = 8
mls_iterations = 10
```

## 2. 工程结构

```text
cpp_adaiso/
  CMakeLists.txt
  README.md
  DESIGN_AND_IMPLEMENTATION.md
  examples/
    pyramid.obj
    T0.off ... T10.off
  include/adaiso/
    AdaIsoRemesher.hpp
    Geometry.hpp
    IO.hpp
    KDTree.hpp
    Mesh.hpp
    Metrics.hpp
  src/
    AdaIsoRemesher.cpp
    Geometry.cpp
    IO.cpp
    KDTree.cpp
    Mesh.cpp
    Metrics.cpp
    main.cpp
  tests/
    basic_tests.cpp
```

构建产物位于 `cpp_adaiso/build/`，运行结果位于 `cpp_adaiso/results/`，这些目录不应进入 Git。

## 3. 构建设计

构建系统使用 CMake，要求 C++17。

依赖：

- Eigen3：线性代数和几何向量运算。
- nanoflann：KD-tree 最近邻查询。

`CMakeLists.txt` 的依赖策略是：

1. 优先 `find_package(Eigen3 CONFIG QUIET)` 和 `find_package(nanoflann CONFIG QUIET)`。
2. 如果使用 vcpkg toolchain，则直接使用 vcpkg 已安装包。
3. 如果找不到包，则通过 `FetchContent` 下载 Eigen 3.4.0 和 nanoflann v1.5.5。

已验证的本机命令：

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake -S cpp_adaiso -B cpp_adaiso/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=e:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows"

cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake --build cpp_adaiso/build --config Release"

cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && ctest --test-dir cpp_adaiso/build --output-on-failure"
```

## 4. 核心数据结构

### 4.1 Mesh

定义在 `include/adaiso/Mesh.hpp`：

```cpp
struct Mesh {
  std::vector<Eigen::Vector3d> vertices;
  std::vector<Eigen::Vector3i> faces;
};
```

约定：

- 顶点坐标使用 double。
- 面固定为三角面。
- face index 使用 0-based 内部索引。
- 输出 OBJ 时转为 1-based 索引。

辅助函数：

- `validate()`：检查 face index 是否越界。
- `removeDegenerateFaces()`：删除重复顶点面、小面积面、重复无向面，并 compact 顶点。
- `compactVertices()`：删除未被 face 引用的顶点并重排索引。
- `uniqueEdges()`：提取无向唯一边。

### 4.2 Edge

无向边使用：

```cpp
using Edge = std::pair<int, int>;
```

通过 `makeEdge(a, b)` 统一排序，保证 `(min, max)` 存储。

### 4.3 AdaIsoConfig

定义在 `include/adaiso/AdaIsoRemesher.hpp`，集中保存算法参数。

重要字段：

- `variant`：`code` 或 `paper`。
- `mesh_scale`：全局目标边长缩放。
- `global_iterations`：全局重网格化迭代次数。
- `target_min_factor` / `target_max_factor`：连续目标边长范围。
- `target_gamma`：连续曲率响应指数。
- `project_to_surface`：是否做 MLS 投影。
- `min_quality_after_collapse`：collapse 局部质量保护阈值。

`variant=code` 和 `variant=paper` 的主要差别：

- `code` split 阈值为 `1.33 * min(local_L)`。
- `paper` split 阈值为 `5/3 * min(local_L)`。
- `code` collapse 参考 `min(local_L)`。
- `paper` collapse 参考 `max(local_L)`。

## 5. 输入输出

实现文件：`src/IO.cpp`

支持输入：

- OBJ：支持 `v`、`f`，支持 `v/vt/vn` token，支持负索引，支持多边形扇形三角化。
- ASCII PLY：读取 vertex 前三个坐标，读取 face list 并扇形三角化。
- ASCII OFF：读取标准 `OFF` 文件，支持注释和空行，支持多边形扇形三角化。

不支持：

- Binary PLY。
- COFF/NOFF/STOFF 等扩展 OFF。
- 非三维顶点属性的语义处理。

输出：

- 统一输出 OBJ。

## 6. 几何与质量指标

实现文件：`src/Geometry.cpp`、`src/Metrics.cpp`

主要几何函数：

- `faceAreas()`：三角面面积。
- `faceNormals()`：面法向。
- `vertexNormals()`：面积加权顶点法向。
- `edgeFaces()`：无向边到 incident faces 的映射。
- `vertexNeighbors()`：顶点一环邻居。
- `vertexFaces()`：顶点 incident faces。
- `boundaryVertices()`：边界顶点检测。
- `cotangentEdgeWeights()`：cotangent 权重。
- `triangleQuality()`：三角形质量指标。
- `maxInnerAngle()`：最大内角。

质量指标：

```text
Q = 6 * area / (sqrt(3) * half_perimeter * longest_edge)
```

`Metrics.cpp` 汇总：

- 顶点数。
- 面数。
- `Qavg`。
- `Qmin`。

## 7. KD-tree

实现文件：`src/KDTree.cpp`

`NearestNeighborIndex` 是 nanoflann 的薄封装。

用途：

- 目标边长查询：将初始目标长度绑定到原始目标点，后续网格操作后通过最近邻查当前点的局部目标边长。
- MLS 投影：对原始网格顶点做 KNN 查询。

注意：

- nanoflann 返回的是平方距离。
- MLS 中会显式 `sqrt()` 后再计算带宽和权重。

## 8. AdaIso 主算法流程

核心实现：`src/AdaIsoRemesher.cpp`

入口：

```cpp
Mesh AdaIsoRemesher::remesh(const Mesh& input);
```

总体流程：

```text
1. removeDegenerateFaces(input)
2. 保存原始网格用于最终 MLS 投影
3. 可选 subRemesh()
4. buildTargetLengths()
   4.1 计算平均边长
   4.2 计算 CSF 曲率平滑场
   4.3 使用经验分位数连续映射目标边长
   4.4 为每个目标点保存目标边长
   4.5 构建目标点 KD-tree
5. 全局迭代 global_iterations 次：
   5.1 splitLongEdges()
   5.2 collapseShortEdges()
   5.3 removeDegenerateFaces()
   5.4 flipEdges()
   5.5 tangentSmoothing()
   5.6 removeDegenerateFaces()
6. 可选 MLS 投影
7. 返回输出网格
```

## 9. 曲率平滑场 CSF

函数：

```cpp
std::vector<double> AdaIsoRemesher::computeCsf(const Mesh& mesh);
```

步骤：

1. 计算面积加权顶点法向。
2. 对每个顶点计算与一环邻居法向的夹角。
3. 将平均夹角作为 raw curvature。
4. 使用 screened sparse solve 平滑：
   `(M + alpha h^2 L)c = M r`
5. `M` 使用 barycentric vertex mass，`L` 使用 clamped cotangent weights。
6. 记录 assembly/factorization/solve 时间、residual、非零元数量和 solver success。

`variant` 不再影响 CSF 曲率定义，只继续影响 split/collapse 阈值口径。

## 10. 目标边长分配

函数：

```cpp
std::vector<double> adaiso::detail::continuousTargetLengths(...);
```

流程：

1. 对 CSF 曲率排序，计算每个顶点的经验分位数 `q_i`。
2. 重复曲率值共享 tie block 的平均 rank。
3. 计算 `alpha_i = q_i ^ target_gamma`。
4. 在 log length 空间中从 `h_max` 连续插值到 `h_min`：

```text
h_i = exp((1 - alpha_i) * log(h_max) + alpha_i * log(h_min))
```

其中：

```text
h_min = mean_edge_length * mesh_scale * target_min_factor
h_max = mean_edge_length * mesh_scale * target_max_factor
```

常量曲率场直接映射为 `mean_edge_length * mesh_scale`。

后续网格拓扑变化后，当前顶点的目标边长通过 KD-tree 最近邻查询原始目标点得到。

## 11. 局部网格操作

### 11.1 Split

函数：

```cpp
int AdaIsoRemesher::splitLongEdges(Mesh& mesh);
```

规则：

```text
edge_length > splitFactor() * min(target_L(a), target_L(b))
```

实现要点：

- 每轮收集候选长边。
- 按边长从大到小排序。
- 选择互不共享端点的边批量 split。
- `splitEdgesBatch()` 一次性插入 midpoint 并重建受影响 face。

批量 split 是大网格性能的关键优化之一。早期逐条 split 在 T0.off 上完整一轮会超过 5 分钟；优化后 T0.off 一轮约 2.5 秒。

### 11.2 Collapse

函数：

```cpp
int AdaIsoRemesher::collapseShortEdges(Mesh& mesh);
```

规则：

```text
edge_length < collapseFactor() * target_reference
```

其中 `target_reference`：

- `code` variant 使用 `min(target_L(a), target_L(b))`。
- `paper` variant 使用 `max(target_L(a), target_L(b))`。

保护条件：

- 跳过边界顶点。
- 跳过非流形边。
- 检查 link condition。
- 检查 collapse 后相关邻边不会立刻变成长边。
- 检查局部受影响三角形质量不低于 `min_quality_after_collapse`。

性能优化：

- 每个 collapse pass 只构建一次 `edgeFaces`、`vertexNeighbors`、`vertexFaces`。
- 同一 pass 内批量选择 footprint 不冲突的短边并统一 collapse。
- 质量保护只检查受 collapse 影响的局部三角形，而不是复制并清理整个网格。

### 11.3 Flip

函数：

```cpp
int AdaIsoRemesher::flipEdges(Mesh& mesh);
```

规则：

- 只处理内部边，即 incident face 数为 2 的边。
- 新边不能已存在。
- 新边长度必须处于 collapse/split 阈值范围内。
- flip 后 valence energy 必须下降。
- 可选避免生成钝角三角形。

目标 valence：

```text
内部点: 6
边界点: 4
```

性能优化：

- 每轮 flip 构建无向边集合，查询新边是否存在从全网格扫描改为哈希查询。

### 11.4 Tangent Smoothing

函数：

```cpp
void AdaIsoRemesher::tangentSmoothing(Mesh& mesh);
```

规则：

- 跳过边界顶点。
- 对 incident faces 的邻点按面面积和目标边长加权。
- 得到邻域加权质心。
- 位移投影到顶点法向的切平面。
- 更新：

```text
p_new = p + tangent_lambda * tangent_displacement
```

默认 `tangent_lambda = 0.6`。

## 12. MLS 投影

函数：

```cpp
std::vector<Eigen::Vector3d> AdaIsoRemesher::mlsProject(...);
```

用途：

- 将 remesh 后的点投影回原始曲面附近。

实现：

1. 在原始网格顶点上构建 KD-tree。
2. 对每个输出点查询 KNN。
3. 使用 Gaussian 权重估计局部点 `a` 和局部法向 `n`。
4. 沿法向投影。
5. 最多迭代 `mls_iterations` 次。

带宽：

```text
h = dK + max(0, dK - dK-1)
```

这避免复刻作者源码中未初始化 `h` 的风险行为。

质量保护：

- 投影前记录 `Qmin`。
- 投影后重新清理并计算 `Qmin`。
- 如果投影导致全局 `Qmin` 下降到投影前以下，则拒绝投影结果，保留投影前网格。

## 13. CLI 设计

入口文件：`src/main.cpp`

使用方式：

```powershell
adaiso_cli <input.obj|input.ply|input.off> -o <output.obj> `
  --mesh-scale 1.0 `
  --variant code `
  --preprocess 0 `
  --iters 5 `
  --project 1 `
  --summary-json summary.json
```

默认行为：

- `variant=code`
- `mesh_scale=1.0`
- `iters=5`
- `preprocess=0`
- `project=1`

输出：

- OBJ 文件。
- 可选 JSON 摘要。
- JSON 中包括输入/输出路径、运行时间、参数、输入/输出质量指标、CSF 迭代次数。

## 14. 测试

测试文件：`tests/basic_tests.cpp`

覆盖：

- 基础几何指标。
- degenerate face 清理。
- edge-face 邻接。
- OBJ roundtrip。
- OFF 多边形扇形三角化。
- pyramid remesh smoke test。

已通过：

```text
ctest: 1/1 passed
```

## 15. 运行结果记录

### 15.1 Pyramid smoke test

参数：

```text
--variant code --mesh-scale 0.8 --iters 2 --preprocess 1 --project 1
```

结果：

```text
input:  5 vertices / 6 faces
output: 8 vertices / 12 faces
Qavg:   0.8709762509 -> 0.9087994822
Qmin:   0.7174389352 -> 0.7207681178
```

### 15.2 T0-T10 一轮测试

参数：

```text
--variant code --mesh-scale 1.0 --iters 1 --preprocess 0 --project 1
```

所有 `T0.off` 到 `T10.off` 均已跑通，输出 OBJ 面索引合法。

典型结果：

```text
T0:
input:  10002 vertices / 20000 faces
output: 19063 vertices / 38122 faces
Qavg:   0.698318 -> 0.767294
Qmin:   0.036781 -> 0.157208
time:   2.537 s
```

### 15.3 T0 十轮测试

参数：

```text
--variant code --mesh-scale 1.0 --iters 10 --preprocess 0 --project 1
```

结果：

```text
input:  10002 vertices / 20000 faces
output: 19988 vertices / 39972 faces
Qavg:   0.6983181594 -> 0.7961670004
Qmin:   0.0367809701 -> 0.2812970214
CLI runtime_sec: 19.3726645 s
external wall time: 21.368 s
```

输出文件：

```text
cpp_adaiso/results/T0_T10/T0_iter10_project1.obj
cpp_adaiso/results/T0_T10/T0_iter10_project1.json
```

这些输出属于运行产物，不建议提交到 Git。

## 16. 当前限制

- 不是作者源码逐行移植，不复刻 `pointNeighbor` pair-list。
- 不实现论文 baseline 对比方法。
- 不支持 binary PLY。
- OFF 仅支持标准 ASCII OFF，不支持 COFF/NOFF/STOFF。
- 输入 mesh 默认已经是合理流形三角网格；复杂非流形数据只做有限保护。
- 目标边长场只在 remesh 开始时构建一次，不在每轮拓扑变化后重估曲率场。

## 17. 后续优化方向

可继续优化的方向：

- 为 collapse 增加更细粒度的局部拓扑缓存，减少 batch pass 之间的邻接重建成本。
- 增加空间哈希或局部拓扑缓存，进一步减少重建邻接次数。
- 增加更严格的 manifold 检查和边界保持策略。
- 实现 `--variant paper` 的完整论文参数实验脚本。
- 增加和 Python 版的自动对齐测试。
- 增加批处理 CLI 或脚本，统一跑 `T0-T10` 并生成 CSV。
- 支持 binary PLY 和更多 OFF 扩展格式。

## 18. 关键工程判断

当前实现有几处有意偏离作者源码：

- 不使用 pair-list `pointNeighbor`，改用显式邻接 map/set，便于检查和维护。
- 不保留未初始化 MLS 带宽行为，改为显式计算 `h`。
- 不引入 VCG/PCL/CGAL/CUDA 等重依赖。
- 对 MLS 投影增加质量保护，避免投影破坏 remesh 后的三角质量。
- 对大网格热路径做批量 split、局部 collapse 检查和哈希边查询，保证 1 万点/2 万面级别数据可以在秒级完成单轮迭代。
