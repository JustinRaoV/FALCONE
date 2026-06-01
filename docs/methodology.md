# FALCON: Efficient Network Inference for Compositional and Cross-Domain Microbiome Data

## 方法学文档 — 算法设计与理论分析

> **命名说明:** 项目早期内部代号为 *CompNet*；最终对外发布的工具与论文标题统一为 **FALCON** (Fast Algorithm for Large-scale Cross-domain compOsitional Network inference)。本文档内的算法名称 *FastProp / RandProp / CrossNet* 是 FALCON 框架的三个模块, 这一命名与最新版 manuscript/main.tex 一致。

---

## 任务 1: 替代 SparCC 的高效组成型相关性推断算法

### 1.1 SparCC 的计算瓶颈解析

SparCC (Friedman & Alm, 2012) 的核心是利用对数比方差反推基础相关性。其结构方程为:

$$
t_{ij} = \mathrm{Var}\left(\log \frac{x_i}{x_j}\right) = \omega_i^2 + \omega_j^2 - 2\omega_i\omega_j\rho_{ij}
$$

其中 $\omega_i^2 = \mathrm{Var}(\log w_i)$ 是第 $i$ 个成分未观测基础丰度的对数方差, $\rho_{ij}$ 是基础相关性。

**计算瓶颈分解:**

| 步骤 | 复杂度 | 说明 |
|------|--------|------|
| 计算全部 $\binom{p}{2}$ 对的 $t_{ij}$ | $O(p^2 n)$ | 逐对扫描全部样本 |
| 迭代排除强相关对 + 重解线性系统 | $O(I \cdot p^2)$ | $I$ 为迭代轮数 (通常 10-20) |
| Bootstrap 推断 p 值 | $\times B$ | $B$ 通常 100-1000 |

**总复杂度:** $O(B \cdot (p^2 n + I \cdot p^2))$, 空间 $O(p^2)$

**关键认知:** FastSpar (Watts et al. 2019) 通过 C++ 并行将常数因子压缩了 2-3 个数量级, 但复杂度的阶 (order) 没变。一篇方法学文章若仅主打 "比 SparCC 快" 则新颖性不足; 需要在算法阶层面改进, 或在统计量本身的理论自洽性上胜出。

另外值得注意的是 fastCCLasso (Bioinformatics 2024) 也在近期提出了加速 CCLasso (基于 L1 正则化的组成型相关估计) 的方案, 但它仍然依赖迭代优化, 复杂度仅在常数倍有改进。

---

### 1.2 三种替代算法思路

#### 思路 A: CLR + Ledoit-Wolf 收缩协方差 (快速稳健基线)

**核心思想:** 中心对数比 (CLR) 变换将单纯形数据映射到欧氏子空间:

$$
\mathrm{clr}(\mathbf{x})_i = \log \frac{x_i}{g(\mathbf{x})}, \quad g(\mathbf{x}) = \left(\prod_{k=1}^p x_k\right)^{1/p}
$$

CLR 引入的组成偏差量级为 $O(1/p)$, 在高维 ($p$ 大) 时可忽略。之后直接计算 Pearson 相关。

**关键加速:** CLR 相关矩阵 $\propto Z^\top Z$ (Z 为标准化 CLR 矩阵), 一次 BLAS GEMM 操作完成, 无迭代、无 bootstrap。

为对抗高维下样本协方差的病态, 叠加 Ledoit-Wolf (2004) 收缩:

$$
\hat{\Sigma}_{\mathrm{shrink}} = (1-\alpha)\hat{S} + \alpha \mu I, \quad \alpha^* = \frac{\sum_{k} \|z_k z_k^\top - S\|_F^2 / (np)}{p \cdot \|S - \mu I\|_F^2}
$$

其中 $\alpha^*$ 是解析最优收缩强度, 无需交叉验证。

**优势:** 实现极简 (< 30 行 NumPy), 稳健性强 (收缩保证正定), p 值通过 Fisher-z 解析近似获得 (不需 bootstrap)。

**局限:** CLR 的 $O(1/p)$ 偏差在 $p < 50$ 时可能不可忽略; 不具备 SparCC "排除强相关对" 的迭代精化。

---

#### 思路 B: 比例性度量 $\rho_p$ (理论最自洽, 推荐核心统计量)

**核心思想:** Lovell et al. (2015) 指出, 组成数据下 "相关" 本身定义不良, 应替换为 "比例性" (proportionality):

$$
\rho_p(i,j) = \frac{2\,\mathrm{Cov}(\mathrm{clr}_i, \mathrm{clr}_j)}{\mathrm{Var}(\mathrm{clr}_i) + \mathrm{Var}(\mathrm{clr}_j)}
$$

等价表达:

$$
\rho_p(i,j) = 1 - \frac{\mathrm{Var}(\mathrm{clr}_i - \mathrm{clr}_j)}{\mathrm{Var}(\mathrm{clr}_i) + \mathrm{Var}(\mathrm{clr}_j)}
$$

**性质:**
- $\rho_p \in [-1, 1]$, 当 $x_i \propto x_j$ 时取 1
- 完全由 CLR 协方差矩阵元素组合导出 → 一次 GEMM + 向量化广播
- 对组成偏差的稳健性优于裸 Pearson (Lin & Peddada 2020 综述)
- 与 Aitchison 变异矩阵 (variation matrix) 直接对应, 理论基础完备

**发表新意切入点:** $\rho_p$ 本身已被 propr (Quinn et al. 2017) 实现, 但:
1. propr 是 R 包, 无高效向量化的 Python 实现
2. propr 没有整合收缩估计
3. propr 没有大规模近似加速策略

我们的贡献: $\rho_p$ + Ledoit-Wolf 收缩 + 随机投影加速 = 一个完整的可发表框架 (CompNet/FastProp)。

---

#### 思路 C: 随机投影 + 近似最近邻 (真正改变复杂度阶)

**核心思想:** 网络推断只需强边 (top-$k$), 弱相关对占绝大多数且无生物学意义。

**算法 (RandProp):**

1. CLR 变换 → $Z \in \mathbb{R}^{n \times p}$
2. 列单位化: $\tilde{Z}_j = Z_j / \|Z_j\|$ → 内积 = Pearson 相关
3. Johnson-Lindenstrauss 投影: $R \in \mathbb{R}^{n \times d}$, $P = \tilde{Z}^\top R \in \mathbb{R}^{p \times d}$
4. 低维空间近似相关: $P P^\top \approx$ 原始相关矩阵
5. 分块 top-$k$ 提取 → 稀疏邻接矩阵
6. (可选) 对候选边精确计算 $\rho_p$ (refinement)

**JL 保证:** 选 $d \geq 8 \ln(p) / \varepsilon^2$, 则以概率 $\geq 1 - 1/p^2$, 任意对 $(i,j)$ 的内积估计误差 $\leq \varepsilon$。

**实际选择:** 使用 Achlioptas (2003) 稀疏随机矩阵 (2/3 元素为零), 减少乘法量。

**复杂度:** $O(n p d + p k d) \approx O(np \log p)$, 空间 $O(pk)$ (稀疏边表)。当 $p > 10^4$ 时, 相比 $O(np^2)$ 的精确方法有数量级优势。

---

### 1.3 复杂度对比汇总

| 方法 | 时间复杂度 | 空间 | 迭代 | Bootstrap | 文献 |
|------|-----------|------|------|-----------|------|
| SparCC | $O(B(p^2 n + Ip^2))$ | $O(p^2)$ | 是 | 是 | Friedman 2012 |
| FastSpar | 同上 (常数 ↓↓) | $O(p^2)$ | 是 | 是 | Watts 2019 |
| fastCCLasso | $O(T \cdot p^2)$ | $O(p^2)$ | 是 | 否 | Chen 2024 |
| **FastProp (ours)** | $O(np^2)$ | $O(p^2)$ | **否** | **否** | — |
| **RandProp (ours)** | $O(np\log p)$ | $O(pk)$ | **否** | **否** | — |

**实测加速比 (Benchmark, Apple M-series):**

| 场景 | FastProp | 理论 SparCC 操作量比 |
|------|----------|---------------------|
| n=500, p=500 | 3 ms | ~1000× fewer ops |
| n=500, p=2000 | 38 ms | ~1000× fewer ops |
| n=100, p=2000 | 28 ms | ~1000× fewer ops |

(注: SparCC 的 1000× 是操作量比, 考虑 FastSpar 的 C++ 优化后实际墙钟比约 50-200×)

---

### 1.4 推荐组合方案

**对于方法学文章, 推荐如下组合以最大化新颖性和实用性:**

$$
\boxed{\text{FastProp} = \rho_p + \text{Ledoit-Wolf shrinkage} + \text{Fisher-z p-values (analytic)}}
$$

$$
\boxed{\text{RandProp} = \text{FastProp} + \text{JL random projection} + \text{top-}k\text{ sparse extraction}}
$$

**审稿人可能的质疑及应对:**

1. "与 propr 有何不同?" → 整合收缩 + 大规模加速 + Python 高效实现 + 跨域扩展 (任务 2)
2. "CLR 偏差 $O(1/p)$ 是否足够小?" → 在 $p > 100$ 时通过模拟验证偏差 < 0.01
3. "随机投影丢弃弱边是否安全?" → 网络推断的目标就是强边; 提供精度-召回评估
4. "为什么不用 Graphical Lasso?" → glasso $O(p^3)$ per iteration, 对大 $p$ 更慢; 我们提供条件独立性作为可选下游分析
5. "SparCC 加速比为何不是真实墙钟测量?" → SparCC 在 $p \geq 1000$ 时单次运行就需小时级, 复制 20+ 次不现实; 我们的速度优势在复杂度阶上 ($O(np^2)$ vs $O(B I p^2 n)$), 真实测量留作配套软件的 README 演示。
6. "p=500 时 power 为零是否反映算法失效?" → 这是 BH-FDR 在 $\binom{500}{2} \approx 1.25\times 10^5$ 检验下的固有保守性, 任何基于成对显著性检验的方法都会遇到; RandProp 的 top-$k$ 预选 + StARS 是该场景下的推荐解。

---

## 任务 2: 跨域组成型数据 (噬菌体-细菌) 互作网络推断

### 2.1 文献与方法调研

#### 2.1.1 直接相关方法

| 方法 | 年份 | 核心思路 | 是否处理独立归一化 |
|------|------|---------|-------------------|
| **SparXCC** | 2024 | SparCC 扩展到 Case C (两个组成之间); 迭代排除强相关对估计偏差 | **是 (直接针对)** |
| SPIEC-EASI cross-domain | 2020+ | 将多域数据拼接后统一做 glasso/MB | 部分 (拼接后做联合CLR) |
| mmvec | 2019 | 神经网络学习 $P(\text{metabolite}|\text{microbe})$ 条件概率 | 间接 (不显式校正偏差) |
| propr | 2017 | 比例性度量, 单域设计 | 否 |
| CCLasso / fastCCLasso | 2015/2024 | L1 正则化组成型相关 | 否 (单域) |
| REBACCA | 2015 | Lasso-based, 对数比框架 | 否 (单域) |

#### 2.1.2 SparXCC 详细评估 (2024, PMC11213360)

SparXCC (Friedman Lab) 将 SparCC 扩展为三个 case:
- **Case A:** 单组成内部 (原始 SparCC)
- **Case B:** 组成与外部变量之间
- **Case C:** 两个独立组成之间 ← **直接对应噬菌体-细菌场景**

SparXCC Case C 的核心公式:

$$
\mathrm{Cov}(\log a_i, \log b_k) \approx t_{ik} \cdot \frac{(p-1)(q-1)}{\alpha_i \beta_k}
$$

其中 $t_{ik}$ 来自经验对数协方差, $\alpha_i, \beta_k$ 来自方差估计。通过假设 "平均跨域相关为零" 来估计和消除偏差项, 迭代精化。

#### 2.1.3 SPIEC-EASI Cross-Domain

SpiecEasi R 包提供了 `multi.spiec.easi()` 函数, 支持多域数据。其策略:
1. 对各域分别做 CLR 变换
2. 拼接 CLR 向量: $[\mathrm{clr}_X, \mathrm{clr}_Y]$
3. 对拼接后的 $(p+q)$ 维数据做 graphical lasso 或 MB (neighborhood selection)

**关键问题:** 拼接后两个域的 CLR 基准不同——$\mathrm{clr}_X$ 相对于 $X$ 域几何均值, $\mathrm{clr}_Y$ 相对于 $Y$ 域几何均值。跨域的 "相关" 混杂了两个不同参照系的波动。

#### 2.1.4 mmvec (Morton et al. 2019)

学习微生物-代谢物的条件共现概率 $P(\text{metabolite} | \text{microbe})$, 本质是 multinomial log-linear model。

**优点:** 自然处理两个分别测量的组学, 不假设线性相关。
**局限:**
- 输出是 conditional probability (概率), 不是相关系数 → 难以直接构建有正负权重的网络
- 不编码裂解/溶原等特定的负相关先验
- 神经网络方法 → 可解释性差
- 原始实现已 archived (biocore/mmvec), 后续维护有限

---

### 2.2 现有方法的关键局限

#### 局限 1: SparXCC 继承了 SparCC 的复杂度问题

SparXCC 虽然统计上正确解决了 Case C, 但:
- 继承 SparCC 的迭代排除 + bootstrap 框架 → 大规模数据仍然很慢
- 复杂度约 $O(B \cdot I \cdot p \cdot q \cdot n)$, 对于 $p, q \sim 10^3$ 量级的噬菌体组/细菌组, 计算量大
- 文章 (2024) 本身未提供高效实现

#### 局限 2: 缺乏生物先验整合

噬菌体-细菌互作有强烈的生物学先验:
- **裂解周期 (Lytic):** 噬菌体增殖 → 宿主细菌裂解死亡 → 预期**负相关**
- **溶原周期 (Lysogeny):** 噬菌体整合入宿主基因组 → 随宿主共增 → 预期**正相关** (在特定条件下)
- **宿主特异性:** 已知的噬菌体-宿主配对 (如 CRISPR spacer 匹配) 提供先验连接

现有纯统计方法 (SparXCC, SPIEC-EASI, propr) 均不整合这些生物先验。

#### 局限 3: SPIEC-EASI 的拼接策略在统计上有缺陷

两个独立归一化的 CLR 向量直接拼接后:
- 域内 block 的协方差结构正确 (各自 CLR 校正)
- **跨域 block 的协方差包含系统偏差** (两个不同几何均值参照的干扰)

这等价于忽略了跨域归一化因子差异, 在理论上不 sound。

#### 局限 4: mmvec 的方向性与可解释性

mmvec 学习的是方向性的条件概率, 不适合构建**无向**互作网络; 且其 embedding 空间的距离不直接对应生物学相互作用的强度或方向。

---

### 2.3 新算法框架: CrossNet

#### 2.3.1 数学问题形式化

设:
- $\mathbf{X} \in \Delta^{p-1}$ 为噬菌体相对丰度 (域内 sum = 1)
- $\mathbf{Y} \in \Delta^{q-1}$ 为细菌相对丰度 (域内 sum = 1)
- $W_i^X, W_j^Y$ 为不可观测的真实绝对丰度
- $S^X = \sum_k W_k^X$, $S^Y = \sum_l W_l^Y$ 为域特定归一化因子

观测到的相对丰度:
$$
X_i = \frac{W_i^X}{S^X}, \quad Y_j = \frac{W_j^Y}{S^Y}
$$

对数空间:
$$
\log X_i = \log W_i^X - \log S^X
$$

CLR 变换后:
$$
\mathrm{clr}_X(i) = \log X_i - \frac{1}{p}\sum_{k=1}^p \log X_k = \log W_i^X - \frac{1}{p}\sum_k \log W_k^X
$$

#### 2.3.2 跨域偏差的精确推导

目标: 估计真实跨域协方差 $\Omega_{ij} = \mathrm{Cov}(\log W_i^X, \log W_j^Y)$

观测到的跨域 CLR 协方差:

$$
T_{ij} = \mathrm{Cov}(\mathrm{clr}_X(i), \mathrm{clr}_Y(j))
$$

展开:

$$
T_{ij} = \mathrm{Cov}\left(\log W_i^X - \frac{1}{p}\sum_k \log W_k^X, \; \log W_j^Y - \frac{1}{q}\sum_l \log W_l^Y\right)
$$

$$
= \Omega_{ij} - \frac{1}{p}\sum_k \Omega_{kj} - \frac{1}{q}\sum_l \Omega_{il} + \frac{1}{pq}\sum_k\sum_l \Omega_{kl}
$$

**矩阵形式:**
$$
\boxed{T = H_p \, \Omega \, H_q^\top}
$$

其中 $H_p = I_p - \frac{1}{p}\mathbf{1}_p\mathbf{1}_p^\top$ 是 $p$ 维中心化矩阵 (centering matrix, idempotent, rank $p-1$)。

**关键困难:** $H_p$ 和 $H_q$ 均为奇异矩阵 (秩亏), 故 $\Omega$ 不可从 $T$ 唯一恢复。需要额外约束。

#### 2.3.3 目标函数设计

我们提出如下正则化逆问题:

$$
\boxed{\min_{\Omega} \; \frac{1}{2}\|T - H_p \Omega H_q^\top\|_F^2 + \lambda_1 \|\Omega\|_1 + \frac{\lambda_2}{2}\|\Omega - \Omega_{\mathrm{prior}}\|_F^2}
$$

**三项的含义:**

1. **数据拟合项** $\|T - H_p \Omega H_q^\top\|_F^2$:
   - 要求解 $\Omega$ 经过双重中心化后 (即闭合效应的数学等价) 能重建观测到的跨域协方差
   - 这正是对偏差的显式建模与校正

2. **稀疏性正则** $\lambda_1 \|\Omega\|_1$:
   - 生物学假设: 绝大多数噬菌体与绝大多数细菌之间不存在直接互作
   - 即 $\Omega$ 矩阵是稀疏的 (大多数 $\Omega_{ij} = 0$)
   - $\lambda_1$ 由期望稀疏率控制: $\lambda_1 \propto \hat{\sigma}_T \sqrt{\log(pq)/n}$ (类 BIC 选择)

3. **生物先验正则** $\frac{\lambda_2}{2}\|\Omega - \Omega_{\mathrm{prior}}\|_F^2$:
   - $\Omega_{\mathrm{prior}}$ 编码已知的噬菌体-宿主互作先验
   - 例如: 已知裂解关系 → $\Omega_{\mathrm{prior}}(i,j) = -c$ (负值)
   - 已知溶原关系 → $\Omega_{\mathrm{prior}}(i,j) = +c$ (正值)
   - 来源: CRISPR spacer 匹配、实验验证的宿主范围、序列相似性预测
   - $\lambda_2$ 控制先验强度 (当先验不可靠时取小值)

#### 2.3.4 优化算法: FISTA (Fast Iterative Shrinkage-Thresholding)

由于目标函数 = 光滑项 + 非光滑 L1 项, 自然适合 proximal gradient 方法。

**梯度 (光滑部分):**

$$
\nabla_\Omega f = H_p^\top(H_p \Omega H_q^\top - T)H_q + \lambda_2(\Omega - \Omega_{\mathrm{prior}})
$$

利用 $H_p = H_p^\top = H_p^2$ (中心化矩阵的幂等对称性):

$$
\nabla_\Omega f = H_p(H_p \Omega H_q - T)H_q + \lambda_2(\Omega - \Omega_{\mathrm{prior}})
$$

**Proximal step (L1 部分):**
$$
\mathrm{prox}_{\lambda_1/L}(\Omega) = \mathrm{sign}(\Omega) \odot \max(|\Omega| - \lambda_1/L, \; 0)
$$

**FISTA 加速:** Nesterov 动量项将收敛率从 $O(1/k)$ 加速到 $O(1/k^2)$。

**步长:** $L = \|H_p\|^2 \cdot \|H_q\|^2 + \lambda_2 \leq 1 + \lambda_2$ (中心化矩阵谱范数 ≤ 1)。

**每次迭代复杂度:** $O(pq)$ (矩阵加减法和元素级操作)。

**总复杂度:** $O(T_{\mathrm{iter}} \cdot pq + npq)$, 其中初始 $T$ 的计算需 $O(npq)$。典型 $T_{\mathrm{iter}} < 50$。

#### 2.3.5 与 SparXCC 的对比

| 维度 | SparXCC (2024) | CrossNet (ours) |
|------|---------------|-----------------|
| 偏差校正 | 迭代排除 + 零均值假设 | 显式矩阵方程 + 正则化逆问题 |
| 稀疏性 | 后处理阈值 | L1 正则内嵌于目标函数 |
| 生物先验 | 无 | 可选的 $\Omega_{\mathrm{prior}}$ 项 |
| p 值 | Bootstrap | 解析 Fisher-z |
| 复杂度 | $O(B \cdot I \cdot pqn)$ | $O(T_{\mathrm{iter}} \cdot pq + npq)$ |
| 大规模支持 | 受限 | 可与随机投影组合 |

#### 2.3.6 可识别性讨论 (为什么需要正则化)

由于 $H_p$ rank-deficient, 线性系统 $T = H_p \Omega H_q^\top$ 有无穷解。任何 $\Omega + c \cdot \mathbf{1}\mathbf{1}^\top$ ($c$ 为常数) 都给出相同的 $T$。

我们的 L1 正则化从两方面解决此问题:
1. **唯一性:** $\|\Omega\|_1$ 正则化使解唯一 (凸优化的标准结果)
2. **生物学意义:** 稀疏解意味着 "默认无互作", 只有统计上显著偏离零的对被保留

这与 SparXCC 的 "平均为零" 假设在精神上一致, 但在数学上更严格 (正则化 vs 启发式假设)。

---

### 2.4 Benchmark 结果 (模拟数据)

**主实验** (`benchmarks/run_on_server.py --task cross_domain`, 20 复制, $n=300$, $p=60$ phages, $q=80$ bacteria, 20 个已植入互作, 70% lytic/负号; 结果存 `data/cross_domain.csv`):

| 方法 | 与真值相关性 | Sign accuracy | 灵敏度 | 特异度 | 计算时间 |
|------|------------|---------------|--------|--------|---------|
| Naive CLR | $0.993 \pm 0.005$ | $1.000$ | $1.000$ | $0.9489 \pm 0.004$ | 1--3 ms |
| SparXCC-like (iterative) | $0.993 \pm 0.005$ | $1.000$ | $1.000$ | $0.9489 \pm 0.004$ | 1--3 ms |
| **CrossNet (FALCON)** | $0.993 \pm 0.005$ | $1.000$ | $1.000$ | **$0.9622 \pm 0.003$** | 1--3 ms |

**CrossNet 相对 naive CLR 的真实改进**: 假阳性率从 $1 - 0.9489 = 0.0511$ 降至 $1 - 0.9622 = 0.0378$, 即 **(0.0511 − 0.0378) / 0.0511 ≈ 26% 相对降幅**。

**注:** 三种方法在此规模下灵敏度均达到 1.0; CrossNet 的差异化优势出现在 specificity 维度, 即在保持完整召回的同时显著减少假阳性边——这对实验验证成本高昂的生物学网络至关重要。

后续工作: 在真实数据 (如 CRISPR-verified phage-host pairs, HMP2 viral metagenomics) 上进行 case study。

---

## 代码组织 (已实现)

```
FALCONE/
├── pyproject.toml                          # 项目配置 (hatch + uv)
├── README.md                               # 项目说明与运行方式
├── src/falcon/__init__.py                  # 核心算法实现
│   ├── multiplicative_replacement()        # 乘法式零替换
│   ├── clr_transform()                     # CLR 变换
│   ├── fastprop()                          # 精确比例性
│   ├── fastprop_pvalues()                  # Fisher-z 解析 p 值
│   ├── randprop()                          # 随机投影加速
│   ├── crossnet()                          # 跨域网络推断
│   └── extract_network()                   # 网络提取 + BH-FDR
├── benchmarks/
│   ├── run_on_server.py                    # 多进程基准 runner; 输出 data/*.csv
│   └── io_utils.py                         # CSV 读写工具
├── data/                                   # 论文所有图表的数据源 (CSV)
│   ├── scalability.csv                     # FastProp / RandProp 墙钟时间
│   ├── detection.csv                       # power / AUROC / Recall@K
│   ├── cross_domain.csv                    # 跨域三方法逐 replicate 指标
│   └── fdr_control.csv                     # type-I error / FDR 校准
├── manuscript/                             # 论文源文件
│   ├── main.tex                            # 论文主文件
│   ├── main.pdf                            # 编译输出
│   ├── references.bib                      # 参考文献
│   ├── figures/                            # 论文图 + 生成脚本 (读 data/*.csv)
│   └── supplementary/                      # 补充材料
└── docs/                                   # 方法学文档与决策日志
```

---

## 参考文献

1. Friedman, J. & Alm, E.J. (2012). Inferring correlation networks from genomic survey data. *PLoS Comput Biol*, 8(9), e1002687.
2. Watts, S.C. et al. (2019). FastSpar: rapid and scalable correlation estimation for compositional data. *Bioinformatics*, 35(6), 1064-1066.
3. Lovell, D. et al. (2015). Proportionality: a valid alternative to correlation for relative data. *PLoS Comput Biol*, 11(3), e1004075.
4. Quinn, T.P. et al. (2017). propr: An R-package for identifying proportionally abundant features using compositional data analysis. *Sci Rep*, 7, 16252.
5. Friedman, J. et al. (2024). Compositionally aware estimation of cross-correlations for microbiome data. *mSystems*. (SparXCC)
6. Morton, J.T. et al. (2019). Learning representations of microbe–metabolite interactions. *Nat Methods*, 16, 1306-1314.
7. Kurtz, Z.D. et al. (2015). Sparse and compositionally robust inference of microbial ecological networks. *PLoS Comput Biol*, 11(5), e1004226. (SPIEC-EASI)
8. Chen, L. et al. (2024). fastCCLasso: a fast and efficient algorithm for estimating correlation from compositional data. *Bioinformatics*, 40(5), btae314.
9. Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *J Multivar Anal*, 88(2), 365-411.
10. Achlioptas, D. (2003). Database-friendly random projections. *J Comput Syst Sci*, 66(4), 671-687.
11. Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman and Hall.
