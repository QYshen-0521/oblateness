## [Project State]

- Current Goal: 系外行星椭率探测初始化
- Active Parameters: TBD
- Chronicler: `archives/` 已就绪（占位）；当前黑板为活跃版本，尚无历史快照迁入

### 数据与文献路径约定（2026-04-16 起）

- **文献（PDF 等）**：保存到 `docs/paper/`（子目录可按主题自拟，黑板只写相对路径）。
- **外部下载数据**（catalog、官方导出表等）：保存到仓库根目录 `data/`（子目录建议按来源/日期命名，例如 `data/nasa_archive/`）。
- **黑板纪律**：每次**新下载或更新**数据文件后，在本文件 **[Session Logs]** 追加一条记录（日期、角色、来源/链接摘要、`data/` 下相对路径、一行用途说明）。不在此粘贴大段表格内容。
- **与 `results/` 分工**：仿真与管线生成物仍以 `results/` 为主（见既有 Session Logs）；`data/` 专放自外源拉取的输入数据。

### squishyplanet `OblateSystem` ↔ 本课题 \((f,\theta)\) ↔ NASA `ps`（keyparams）列

依据 [OblateSystem API](https://squishyplanet.readthedocs.io/en/latest/oblate_system.html)（文档日期以官方为准）。**角度在代码中均为弧度**；NASA 表 `pl_orbincl` 多为度，需换算。

| OblateSystem 参数 | 含义（摘要） | 与用户 \((f,\theta)\) 的对应 | NASA 列（`ps_tran_flag_keyparams_*.csv`）或来源 |
|-------------------|-------------|------------------------------|------------------------------------------------|
| `f1` | \((R_\mathrm{eq}-R_\mathrm{pol})/R_\mathrm{eq}\)，赤道最长轴相对极轴扁平度 | **与用户 \(f\) 同定义**（假定两赤道轴相等，**`f2=0`** 的扁球） | 不来自表；**注入/拟合的物理量** |
| `f2` | 赤道椭圆长短轴相对差 | 本阶段可固定 **0**（轴对称扁球） | — |
| `obliq` | 自转轴与**轨道面法向**夹角（3D） | 仅当不用投影椭圆参数化时使用；**与用户「视线投影 \(\theta\)」不同** | 不直接等同 `pl_trueobliq`（档案定义可能为真倾角/质心） |
| `prec` | 轨道法向为轴的进动/季节角 | 与同步假设、`tidally_locked` 耦合 | 注入/固定；见 API |
| `parameterize_with_projected_ellipse` | 是否用**天球切面上投影椭圆**描述行星轮廊 | **`True`** 时与用户「投影 \(\theta\)」一致 | 建模开关（非 CSV） |
| `projected_f` | 投影椭圆扁平度 | **投影意义下的形状参数**（由 3D \(f\) 与取向共同决定）；可与 \(f\) 联合约定注入规则 | 注入/拟合 |
| `projected_theta` | 投影椭圆半长轴方位角 | **与用户视线投影 \(\theta\) 对齐**（需与文档符号约定一致） | 注入/拟合 |
| `projected_effective_r` | 与投影椭圆同面积圆半径 \([R_\star]\) | 小扁率时可近似 **`pl_ratror`**；精确需由几何换算 | `pl_ratror`（近似起点） |
| `r` | 赤道半径 \([R_\star]\) | `parameterize_with_projected_ellipse=False` 时必填；与 **`pl_ratror`** 对齐 | `pl_ratror` |
| `a` | 半长轴 \([R_\star]\) | — | `pl_ratdor` |
| `period` | 轨道周期 [d] | — | `pl_orbper` |
| `t0` 或 `t_peri` | 凌心时刻 [d] 或 过近点时刻 | 通常用 **`t0`** | `pl_tranmid`（BJD/TDB 等，与 `times` 零点一致） |
| `i` | 轨道倾角 [rad] | — | `pl_orbincl`（**度→rad**） |
| `e` | 偏心率 | — | `pl_orbeccen` |
| `omega` | 近心点幅角 [rad] | **`e=0` 时固定 0** | —（与 `e` 一致） |
| `Omega` | 升交点经度 [rad] | 文档：对凌星光变可通过旋转吸收；可默认 \(\pi\) | 默认/API 默认 |
| `tidally_locked` | 是否潮汐锁定 | **单次凌星内形状不变**可与 `True` 或固定 `prec` 搭配；需显式选择 | 非 NASA；物理假设 |
| `ld_u_coeffs` | 恒星 LD 多项式系数 \(I(\mu)/I_0=1-\sum u_i(1-\mu)^i\) | — | 由 **ExoTiC-LD**（或同类）根据 `st_teff`, `st_logg`, `st_met` + **仪器波段**生成（本管线默认 **JWST** 通带；见 [fit_oblate_transit 教程](https://squishyplanet.readthedocs.io/en/latest/tutorials/fit_oblate_transit.html)） |
| （恒星） | ExoTiC-LD 输入 | — | `st_teff`, `st_logg`, `st_met`（`st_metratio` 为标尺标签如 `[Fe/H]`，数值用 `st_met`） |

**筛选文件说明**：`data/nasa_archive/ps_tran_oblate_inputs_valid_20260416.csv` 自 `ps_tran_flag_keyparams_20260416.csv` 保留「凌星+恒星+轨道」建模直接需要的列且下列字段为**非空数值**且对应 `*lim` 为 **0**（空白的 `lim` 视为 0）：`pl_orbper`, `pl_ratdor`, `pl_ratror`, `pl_orbincl`, `pl_tranmid`, `pl_orbeccen`, `st_teff`, `st_logg`, `st_met`。其余列原样保留便于后续分析；**578** 个行星（+表头）。未要求 `pl_imppar` 等，Executor 可做一致性检查。

### 无噪声反演：二维网格 \(\chi^2\)（Planner 定案 → Executor 实现）

**目标**：在**不加观测噪声**时，从「基准光变 − 椭率光变」的差分曲线恢复注入的 **`projected_f`** 与 **`projected_theta`**（与 `tests/test_nasa_first_row_oblate_lightcurve.py` 中 `parameterize_with_projected_ellipse=True` 一致；\(\theta\) 用弧度进模型）。

**数据定义**（固定时间轴 \(\{t_k\}\)、与单行星测试相同的 NASA+LD+轨道设定）：

- \(F_0(t)\)：`projected_f=0`（圆）的理论流量；  
- \(F_\mathrm{inj}(t)\)：注入 \((f_\mathrm{inj},\theta_\mathrm{inj})\) 的理论流量；  
- **观测差分**（无噪声）：\(D(t_k)=F_0(t_k)-F_\mathrm{inj}(t_k)\)。

**模型差分**：对试探 \((f,\theta)\)，\(m(t_k;f,\theta)=F_0(t_k)-F_\mathrm{obl}(t_k;f,\theta)\)，其中 \(F_\mathrm{obl}\) 与生成 \(F_\mathrm{inj}\) 时除 \((f,\theta)\) 外**同一** `OblateSystem` 状态。

**判据**：\(\chi^2(f,\theta)=\sum_k \bigl(D(t_k)-m(t_k;f,\theta)\bigr)^2\,/\,\sigma^2\)。无噪声实现上取 \(\sigma\equiv 1\) 即最小化 **SSE**；若数值有浮点残差，以 \(\arg\min \chi^2\) 为网格最优，并报告 \(\min\chi^2\) 是否接近机器精度量级。

**网格**：在 \((f,\theta)\) 上矩形扫描（范围/步数可配置）；注意 `projected_theta` 与 API 的**周期/对称性**（必要时限制在如 \([0,\pi)\) 或与 squishyplanet 文档一致），避免重复等价点。

**输出（建议）**：\(\chi^2(f,\theta)\) 二维图或表、网格上最优 \((\hat f,\hat\theta)\) 与注入值对比；为后续「可恢复域」扩展留接口。实现位置与命令由 **Executor** 定（`src/` 或 `tests/`、`results/` 图）。

**等价实现**：\(\sum_k(D_k-m_k)^2=\sum_k(F_{\mathrm{obl},k}-F_{\mathrm{inj},k})^2\)，可不显式计算 \(F_0\)（见代码 `noiseless_grid_chi2_inversion.py`）。

### 批量无噪声反演：注入网格、局部细化、恢复判据与 \(f\)–\(\theta\) 状态图（Planner 定案）

**注入扫参网格**（每一组单独生成 \(F_\mathrm{inj}\) 并做一次反演）：

- \(f_\mathrm{inj}\in\{0.01,0.02,\ldots,0.15\}\)（步长 **0.01**）；  
- \(\theta_\mathrm{inj}\in\{0^\circ,20^\circ,\ldots,160^\circ\}\)（步长 **20°**；模型内用弧度）。

**搜索流程**（每组注入）：

1. **粗网格** \(\chi^2(f,\theta)\)（与现有单点实验同一定义；可等价用 \(\sum_k(F_{\mathrm{obl}}-F_\mathrm{inj})^2\)），得 \((\hat f_0,\hat\theta_0)=\arg\min\chi^2\)。  
2. **局部细化**：在 \((\hat f_0,\hat\theta_0)\) 邻域做**小范围密化**二次网格（范围/倍数由 Executor 配置并写入运行日志），再取全局最优 \((\hat f,\hat\theta)\)。**A1 阈值 \(\epsilon_f,\epsilon_\theta\) 取「细化后」搜索步长的各一半**（\(\epsilon_\theta\) 用度）。

**搜索网格步长（Executor 实现，`batch_noiseless_recovery.py` 中 `COARSE_GRID_CONFIG` / `REFINE_GRID_CONFIG`）**：

- **粗网格**：\(f\in[0,\,0.2]\)，步长 **0.02**；\(\theta\in[0^\circ,\,180^\circ)\)（度），步长 **5°**（入模型时换算为弧度）。  
- **细网格**（粗 \(\arg\min\) 邻域内）：\(f\) 步长 **0.005**；\(\theta\) 步长 **1°**；邻域半宽等仍由 `REFINE_GRID_CONFIG` 可调。

**恢复判据 A1（参数距离）**：同时满足  
\(|\hat f-f_\mathrm{inj}|<\epsilon_f\) 且 \(|\hat\theta-\theta_\mathrm{inj}|<\epsilon_\theta\)（\(\hat\theta\) 与注入均用度比较；注意 \(\theta\) 周期/边界，必要时在 \([0^\circ,180^\circ)\) 上取最短角距）。

**恢复判据 C1（唯一谷底 / 与次极小分离）**：在**该次反演中实际计算过 \(\chi^2\) 的全部离散格点**上，设最小值为 \(\chi^2_{(1)}\)、次小值为 \(\chi^2_{(2)}\)（若多格点并列最小，则 \(\chi^2_{(2)}-\chi^2_{(1)}=0\)，视为不唯一）。要求  
\(\chi^2_{(2)}-\chi^2_{(1)}>\Delta\chi^2_\mathrm{gap}\)  
才记为**非简并**；否则记为「**多解/简并**」，**不算干净恢复**（即使满足 A1）。\(\Delta\chi^2_\mathrm{gap}\) 为 **Executor 可配置常数**（无噪声下建议与浮点噪声量级匹配，如取 `max(1e-12, rtol * max(χ²_{(1)},1))` 等，并在 Session Log / 输出元数据中写明所用值）。

**重要**：C1 所用的 \(\chi^2\) 列表必须先按 **物理格点 \((f,\theta)\) 去重**（同一坐标只保留一个 \(\chi^2\)，例如取 min 或最后一次计算值），再排序取 \(\chi^2_{(1)},\chi^2_{(2)}\)。不得把**粗网格与细网格拼接后**未去重的重复坐标当作两个独立样本——否则无噪声下同一真值点会被算两次、\(\chi^2\) 并列，**gap=0**，误判「简并」。详见下节「C1 实现勘误」。

### C1 实现勘误（`batch_noiseless_recovery.py`）— Agent 2 已修复

**现象（已由 `results/batch_noiseless_recovery.npz` 核对）**：\(f\ge 0.07\) 时 **clean / degenerate 与 \(f_\mathrm{inj}\) 强相关**——\(f_\mathrm{inj}\in\{0.08,0.10,0.12,0.14\}\)（粗 \(f\) 步长 0.02 的格点）**全部**被判为 degenerate；\(f_\mathrm{inj}\in\{0.07,0.09,0.11,0.13,0.15\}\) **全部** clean。degenerate 行 **median(\(\chi^2\) gap)=0**，clean 行 median gap \(\sim 10^{-8}\) 量级。

**原因（实现缺陷，非物理）**：`run_single_injection` 将粗、细两套网格的 \(\chi^2\) **直接 `concatenate`** 后调用 `_c1_gap`。细网格以粗 \(\arg\min\) 为中心展开，**与粗网格在相同 \((f,\theta)\) 上重复计算**。当注入点落在粗格点上时，真值常同时出现在粗、细集合中 → **两个完全相同的 \(\chi^2=0\)** → 排序后 \(\chi^2_{(1)}=\chi^2_{(2)}\) → **C1 假**。注入在粗格点**之间**时，最优解往往只在细网格出现一次 → 无重复并列 → **C1 真**。故状态图出现与 **0.01 注入步长 vs 0.02 粗步长对齐** 相关的条纹，**不可解释为**「大 \(f\) 时椭率可恢复性随 \(f\) 物理振荡」。

**Executor 实现（`C1_DEDUP_CONFIG` + `_dedupe_chi2_by_f_theta`）**：

1. 合并粗、细后，对 **唯一 \((f,\theta)\) 键**（`round` 的 `ndigits_f` / `ndigits_theta_rad` 可配）只保留 **最小** \(\chi^2\)，再调用 `_c1_gap`。  
2. \(\arg\min(\hat f,\hat\theta)\) 仍基于合并前全量 `chi_all`（与去重后最小 \(\chi^2\) 一致）。  
3. 结果中附带 `n_chi2_evaluated`、`n_chi2_unique_c1` 便于核对。  
4. 重跑 `batch_noiseless_recovery` 后更新 `results/batch_noiseless_f_theta_status.png`、`results/batch_noiseless_recovery.npz`；Session Log 见 Executor 条目。

**综合**：一次注入点记为 **「干净恢复」** 当且仅当 **同时满足 A1 与 C1**；否则细分：满足 A1 但违反 C1 → 简并；违反 A1 → 未恢复（或再分子类由 Executor 在图例中说明）。

**交付图**：在 **\(f\)–\(\theta\) 平面**上，对每个注入格点 \((f_\mathrm{inj},\theta_\mathrm{inj})\) 绘**状态**（如颜色或标记：**干净恢复 / 简并 / 未恢复**）。坐标轴与注入扫参网格一致，便于一眼看到可恢复区域；保存至 `results/`（文件名由 Executor 定并在此补记）。

### 多系统批量无噪声测试（Planner 建议 → Agent 2）

**动机**：单行星（`ps_tran_oblate_inputs_valid_*.csv` 固定 `row_index`）上 C1 去重后，用户观测到 **\(f_\mathrm{inj}>0.06\)** 时注入格点可**全部干净恢复**；下一步应在 **不同宿主/轨道/LD** 下检验同一管线，避免结论仅适用于一颗星。

**范围**：对表中多行（每行一颗行星）重复「同一注入网格 + 同一粗/细搜索 + 同一 A1/C1（含去重）+ 同一 JWST LD 模式」。每系统输出：与现 `batch_noiseless_recovery` 同构的 **npz**（或汇总表）+ **每张系统一张** `f`–\(\theta\) 状态图，或 **汇总图**（例如横轴 `pl_name` 或 `row_index`、纵轴干净恢复比例）。

**计算量**：单系统约 **135** 注入 ×（粗+细）\(\chi^2\) 评估；578 系统全跑代价大。**建议分阶段**：(1) **分层小样**（如按 `st_teff` 或 `pl_ratror` 分箱各抽 \(N\) 颗，\(N\) 由 Executor 定，如 10–30）；(2) 再视结果决定是否全表扫描。

**失败/跳过**：ExoTiC-LD 或 `OblateSystem` 个别行失败时，该行记 **skip** 并写日志，不静默吞掉。

**解释边界**：多系统间差异来自 **\(R_p/R_\star,\,a/R_\star,\,i,\,\mathrm{LD},\,T_\mathrm{eff}\)** 等；**小 \(f\)** 仍可能因信号弱系统性 **failed**（与单星结论一致），不应与 C1 实现混淆。

## [Task Board]

- [ ] Planner: 细化椭率探测的数学模型 (Pending)
- [x] Executor: **多系统**无噪声批量：`python -m oblateness.multi_system_batch_noiseless`（`--rows` 或 `MULTI_SYSTEM_CONFIG`）；每系统输出 `results/multi_system/`；超算 `sbatch test.sh`（见 Session Log 2026-04-18）
- [x] Executor: **修复 C1**：粗+细 \(\chi^2\) 合并后对 \((f,\theta)\) **去重**再取 \(\chi^2_{(2)}-\chi^2_{(1)}\)（见「C1 实现勘误」；`C1_DEDUP_CONFIG`）；重跑批量并更新 `results/` 产物
- [x] Executor: 批量无噪声注入网格（\(f\) 0.01–0.15 步 0.01；\(\theta\) 0°–160° 步 20°）+ 粗网格 \(\arg\min\) + **局部细化** + **A1/C1** 判据 + **`f`–\(\theta\) 状态图**（见上节「批量无噪声反演」；实现 `src/oblateness/batch_noiseless_recovery.py`，可调参数见文件顶部各 `*_CONFIG`）
- [x] Executor: 无噪声 \((f,\theta)\) 二维网格 \(\chi^2\) 反演（见上节「无噪声反演」；实现 `src/oblateness/noiseless_grid_chi2_inversion.py`，与 `test_nasa_first_row_oblate_lightcurve` 注入一致）
- [x] Executor: 准备基础仿真环境 (Done — 见 Session Logs 2026-04-16 Executor)

## [Session Logs]

- 2026-04-16: Project initialized by Agent 0.
- 2026-04-16 (Executor): 基础 Python 仿真环境就绪。依赖见 `pyproject.toml`（numpy、scipy、pytest）。包 `src/oblateness/`；物理常数封装 `src/oblateness/constants.py`（CODATA + IAU 2015 标称太阳量）；校验 `tests/test_constants.py`。复现：`python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && pytest`。可选 `pip install -e ".[viz]"`（matplotlib）。输出目录 `results/`（沿用既有 `.gitkeep`）。
- 2026-04-16 (Chronicler): 已对照 `.cursorrules` 核对黑板；Task Board 与 Session Logs 一致（Planner 待办、Executor 已完成并有证据链）。建立 `archives/` 占位；未移动任何 `results/` 产物。
- 2026-04-16 (Planner): 约定文献目录 `docs/paper/`、外部数据目录 `data/`；已创建占位。此后每次下载/更新数据须在 Session Logs 追加一条（路径 + 来源摘要）。`.cursorrules` 中「WORKLOG.md」与本仓库实际文件名 `worklog.md` 视为同一黑板。
- 2026-04-16 (Planner): 自 [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu) TAP 同步下载 Planetary Systems 表（`ps`）全列 CSV：`tran_flag=1` 且 `default_flag=1`（档案对「存在凌星探测」的标记与默认解；见 [TAP 使用说明](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)）。保存路径 `data/nasa_archive/ps_tran_flag_default_20260416.csv`（约 4566 行、含表头）。用途：凌星系外行星参数主样本；正式发表请按档案 [DOI / 致谢](https://exoplanetarchive.ipac.caltech.edu/docs/doi.html) 要求引用。
- 2026-04-16 (Planner): 由 `data/nasa_archive/ps_tran_flag_default_20260416.csv` 抽取轨道/凌星/形状相关与恒星参数等关键列，生成 `data/nasa_archive/ps_tran_flag_keyparams_20260416.csv`（同 4566 行数据；列顺序按用户指定）。用途：轻量分析输入；原始列定义仍以档案 PS 文档为准。
- 2026-04-16 (Planner): 自 `ps_tran_flag_keyparams_20260416.csv` 删除含 ADS/HTML 链接的 `st_refname` 列，余 98 列（行星名、轨道/凌星/恒星物理量及 `*lim` 标志等）；仍覆盖同名文件。文献引用请回查全表 `ps_tran_flag_default_20260416.csv` 或 NASA 档案。
- 2026-04-16 (Planner): 黑板增加 **`OblateSystem` ↔ \((f,\theta)\) ↔ NASA 列** 对照表（见上节）。由 keyparams 筛选 OblateSystem+ExoTiC 直接所需且 `*lim=0` 的 9 组量，输出 `data/nasa_archive/ps_tran_oblate_inputs_valid_20260416.csv`（578 行星 + 表头）。`ld_u_coeffs` 仍须在 Executor 侧按波段生成。
- 2026-04-17 (Executor): 集成测试 `tests/test_nasa_first_row_oblate_lightcurve.py`：自 `data/nasa_archive/ps_tran_oblate_inputs_valid_20260416.csv` 首行（或通过 `USER_CONFIG['row_index']` 指定）读入 NASA 参数，ExoTiC-LD（`exotic_ld`，模式默认 **JWST** `JWST_NIRSpec_Prism`，波长窗与通带一致）生成 quadratic LD，`squishyplanet.OblateSystem` 在 `parameterize_with_projected_ellipse=True` 下比较「投影 f=0」与注入 `(projected_f, projected_theta)` 两条光变；可调参数集中在文件顶部 `USER_CONFIG`。依赖：`pip install -e ".[dev,squishy]"`（见 `pyproject.toml` 中 `[project.optional-dependencies] squishy`）。成功运行后图 `results/nasa_first_row_oblate_theory_compare.png`。`pytest -m integration` 可选中该类测试；缺依赖时 `importorskip` 跳过。
- 2026-04-17 (Planner): 无噪声阶段采用**二维网格 + \(\chi^2\)** 恢复 `projected_f`、`projected_theta`：差分 \(D=F_0-F_\mathrm{inj}\)，模型 \(m=F_0-F_\mathrm{obl}(f,\theta)\)，最小化 \(\sum_k(D_k-m_k)^2\)（数学定义见上节）。委派 **Executor** 实现、出 \(\chi^2\) 图/表与最优格点对比注入值。
- 2026-04-17 (Executor): 无噪声二维网格 \(\chi^2\) 已实现：`src/oblateness/noiseless_grid_chi2_inversion.py`；`PLANET_CONFIG` 与 `tests/test_nasa_first_row_oblate_lightcurve.py` 首行注入（`f=0.12`，\(\theta=40^\circ\)）及 JWST LD 一致；`GRID_CONFIG` 可调。输出 `results/noiseless_chi2_grid_f_theta.png`、`results/noiseless_chi2_grid.npz`。运行：`python -m oblateness.noiseless_grid_chi2_inversion`（需 `.[squishy]`）。
- 2026-04-17 (Planner): 批量无噪声扫参定案：注入 \(f=0.01(0.01)0.15\)、\(\theta=0^\circ(20^\circ)160^\circ\)；每点 **粗网格 \(\arg\min\) + 局部密化**；**A1**：\(|\hat f-f_\mathrm{inj}|<\epsilon_f\)、\(|\hat\theta-\theta_\mathrm{inj}|<\epsilon_\theta\)，\(\epsilon\) 为**细化后步长之半**；**C1**：全评估格点上 \(\chi^2_{(2)}-\chi^2_{(1)}>\Delta\chi^2_\mathrm{gap}\) 否则「多解/简并」；**干净恢复**=A1∧C1。Executor 输出 **`f`–\(\theta\) 注入格点状态图**（见「批量无噪声反演」节）。
- 2026-04-17 (Executor): 批量无噪声管线 `src/oblateness/batch_noiseless_recovery.py`：注入格点、`COARSE_GRID_CONFIG`、`REFINE_GRID_CONFIG`、`DELTA_CHI2_CONFIG`、`BATCH_RUN_CONFIG` 等均在文件顶部集中配置；每注入点粗+细 \(\chi^2\) 合并取全局最优，A1/C1 与状态分类，图 `results/batch_noiseless_f_theta_status.png`、可选 `results/batch_noiseless_recovery.npz`。运行：`python -m oblateness.batch_noiseless_recovery`（需 `.[squishy]`，默认全格点计算量大）。
- 2026-04-17 (Executor): 批量脚本网格更新：粗搜索 \(f\in[0,0.2]\) 步 **0.02**，\(\theta\) 步 **5°**；细搜索步 **0.005**（\(f\)）、**1°**（\(\theta\)）；A1 的 \(\epsilon_f,\epsilon_\theta\) 为细步长之半。黑板「批量无噪声反演」节已同步。
- 2026-04-17 (Planner): **C1 实现勘误**写入黑板：当前 `batch_noiseless_recovery` 粗+细 \(\chi^2\) **未对 \((f,\theta)\) 去重**导致并列最小、gap=0，在 \(f_\mathrm{inj}\) 对齐粗 0.02 网格时出现**假简并**条纹；规格见专节。**Agent 2** 负责去重后 C1、重跑与更新 `results/`；**Agent 1 不修改 `src/`**。
- 2026-04-17 (Executor): **C1 去重已落地**：`batch_noiseless_recovery.py` 中 `_dedupe_chi2_by_f_theta` + `C1_DEDUP_CONFIG`；`_c1_gap` 仅对去重后 \(\chi^2\) 列表调用；输出 `n_chi2_evaluated`、`n_chi2_unique_c1`。烟测（5 注入）次小与最小 \(\chi^2\) 间隙已为 \(\sim 10^{-11}\) 量级（非去重前 gap=0 的假简并）。**全量**重跑后更新 `results/batch_noiseless_f_theta_status.png`、`batch_noiseless_recovery.npz`：`python -m oblateness.batch_noiseless_recovery`。
- 2026-04-18 (Chronicler): 配置 **Cursor `preCompact` 钩子**（`.cursor/hooks.json` → `.cursor/hooks/precompact-worklog-reminder.sh`）：压缩前向用户提示 `@worklog.md` / Agent Read 根目录黑板；**alwaysApply** 规则 `.cursor/rules/worklog-context-recovery.mdc` 约束 Agent 在压缩后或任务延续时先读 `worklog.md`。`.cursorrules` 已增「上下文压缩与黑板恢复」节。
- 2026-04-18 (Planner): C1 去重并全量重跑后，单星批量在 **\(f_\mathrm{inj}>0.06\)** 上可**全部干净恢复**；**小 \(f\)** 仍可能因信号弱失败（物理/分辨率，非条纹假象）。**下一步**同意开展 **多系统**无噪声批量（不同 `row_index`/宿主）；分层小样与交付物见黑板「多系统批量无噪声测试」；委派 **Agent 2**。
- 2026-04-18 (Executor): 多系统入口 `src/oblateness/multi_system_batch_noiseless.py`：对 `--rows`（或默认 `MULTI_SYSTEM_CONFIG['row_indices']`）逐行调用 `run_batch(planet_config_for_row(i))`；每系统 `results/multi_system/status_row*_*.png`、`batch_row*_*.npz`，汇总 `results/multi_system/multi_system_summary.csv`；失败行 stderr 打印并记入 CSV `error`。超算：仓库根 `test.sh`（`#SBATCH`、`WORKDIR=/dssg/home/acct-tdlffb/tdlffb-user1/workspace/RV_astrometry_detect/shen/oblateness`、顶部 `ROWS=...`），`sbatch test.sh`。
