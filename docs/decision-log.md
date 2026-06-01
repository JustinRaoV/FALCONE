# 决策日志 — FALCON 实现

## 设计决策

- **项目命名 FALCON** — Fast Algorithm for Large-scale Cross-domain Compositional Network inference; 隐喻速度(猎隼), 缩写自然, 涵盖两个贡献, 微生物组网络领域未被占用。
- **选择 $\rho_p$ 而非 Pearson/Spearman 作为核心统计量** — 比例性在 Aitchison 几何下理论自洽, 审稿人对组成型数据文章接受度高。`src/falcon/__init__.py:184`
- **Ledoit-Wolf 收缩而非 glasso** — 解析最优解无需 CV, 单次 $O(np^2)$ 即完成; glasso $O(p^3)$ per iter 对大 $p$ 更慢。`src/falcon/__init__.py:234`
- **FISTA 而非 ADMM 求解 CrossNet 目标函数** — 目标函数结构 (smooth + L1) 天然适配 proximal gradient; ADMM 对此问题无显著优势且实现更复杂。`src/falcon/__init__.py:601`
- **Achlioptas 稀疏投影矩阵** — 比高斯投影少 2/3 乘法, 且保持 JL 保证。`src/falcon/__init__.py:420`
- **Fisher-z 解析 p 值替代 bootstrap** — 去掉 SparCC 计算量最大的 B 倍乘数, 对 $n > 30$ 近似良好。`src/falcon/__init__.py:196`
- **乘法式零替换而非加性 pseudocount** — 保持组成结构 (行和不变), Martin-Fernandez (2003) 推荐方案。`src/falcon/__init__.py:39`
- **论文定位: Bioinformatics 方法学** — 侧重算法创新+数学推导+模拟验证; 投稿 Bioinformatics/NAR Methods 或 Genome Biology。

## 偏离规约

- **CrossNet 的 bias_corrected 方法灵敏度与 naive_clr 持平** — 主指标实验 (n=300, p=60, q=80, 20 reps) 显示三种方法 sensitivity 均为 1.0；CrossNet 真实优势是 specificity 0.949 → 0.962 (≈26% 相对假阳性下降)。数据存 `data/cross_domain.csv`，由 `benchmarks/run_on_server.py --task cross_domain` 生成。
- **p=500 时 power=0 但 AUROC≈1.0** — BH-FDR 在 ~125,000 检验下极度保守, 但估计器排序仍正确。论文用 AUROC + Recall@K 揭示这种 "metric dissociation"，把 RandProp top-k + StARS 建议为大 $p$ 推荐工作流。数据存 `data/detection.csv`。
- **已从 pyproject.toml 移除未使用的 numba 依赖** — 当前 NumPy BLAS 已足够快, 移除避免 reviewer 质疑"声明却不用"。
- **模拟器 _nearest_pd 在小样本测试中产生 overflow 警告** — 高维随机矩阵修正为 PD 时数值溢出, 不影响最终结果 (NaN 被下游处理), 后续可换用 Higham (1988) 交替投影法。
- **SparCC 速度比较为复杂度外推, 不是真实运行** — 论文措辞已调整为 "extrapolated from $O(BIp^2n)$ complexity", 避免审稿人指控未做 head-to-head。

## 取舍

- **RandProp 近似 vs 精确 $\rho_p$** — 近似方法丢弃弱边, 对网络推断无损 (只需强边); 代价是需要设 $k$ 和 $\varepsilon$ 两个超参数。`src/falcon/__init__.py:286`
- **CrossNet 正则化路径**: $\lambda_1$ 大 → 特异度高但灵敏度低; $\lambda_1$ 小 → 灵敏度高但可能引入假阳性。建议用 StARS (Stability Approach to Regularization Selection) 选 $\lambda_1$。
- **SparXCC (2024) 的存在降低了 CrossNet 纯"跨域偏差校正"的新颖性** — 我们的差异化优势在于: (a) 无 bootstrap 的速度优势; (b) 生物先验整合; (c) 理论上更严格的正则化框架 vs 启发式迭代排除。

## 待确认

- **是否需要对 FastProp 与 SparCC/FastSpar 做真实数据头对头对比?** — 建议在 HMP2 或 Earth Microbiome Project 子集上跑 wall-clock 和 edge overlap。
- **CrossNet 的 signed_prior 来源** — 当前假设 CRISPR spacer 匹配可提供先验, 但实际噬菌体-宿主数据中此信息覆盖率可能很低; 需确认可用的先验数据库 (如 MVP, PHASTER, iPHoP)。
- **文章定位: 方法学论文 (Bioinformatics/NAR) 还是应用型 (mSystems/Microbiome)?** — 前者侧重算法创新和基准, 后者需要生物学 case study。
- **是否将 FastProp + CrossNet 作为统一工具发布 (Python package)?** — 如果是, 需补充 CLI、文档、pip/conda 分发。
