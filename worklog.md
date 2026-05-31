## [Project State]

- Current Goal: 系外行星椭率探测初始化
- Active Parameters: TBD
- Chronicler: `archives/` 已就绪（占位）；当前黑板为活跃版本，尚无历史快照迁入

### 数据与文献路径约定（2026-04-16 起）

- **文献（PDF 等）**：保存到 `docs/paper/`（子目录可按主题自拟，黑板只写相对路径）。
- **外部下载数据**（catalog、官方导出表等）：保存到仓库根目录 `data/`（子目录建议按来源/日期命名，例如 `data/nasa_archive/`）。
- **黑板纪律**：每次**新下载或更新**数据文件后，在本文件 **[Session Logs]** 追加一条记录（日期、角色、来源/链接摘要、`data/` 下相对路径、一行用途说明）。不在此粘贴大段表格内容。
- **与 `results/` 分工**：仿真与管线生成物仍以 `results/` 为主（见既有 Session Logs）；`data/` 专放自外源拉取的输入数据。

### 上下文续接要点（会话压缩 / 新 Chat 恢复）

- **黑板文件名**：根目录 `worklog.md`（小写）与 `.cursorrules` 中的 `WORKLOG.md` 指同一黑板。
- **Tier A / B / 合并表（行数以 `wc -l` 为准）**：`ps_tran_oblate_inputs_valid_20260416.csv` **579** 行（578 数据 + 表头），**未覆盖**；Tier B 仅 `ps_tran_oblate_inputs_tier_b_derived_20260416.csv` **165** 行（164 数据 + 表头）；合并 `ps_tran_oblate_inputs_combined_tierab_20260416.csv` **743** 行（742 数据 + 表头）。生成：`python -m oblateness.build_tier_b_oblate_inputs`。
- **`OBLATE_CSV_PATH`**：若干入口默认 CSV 为 **Tier B 派生表**；重跑**原 578** 多系统或对齐旧 `row_index` 时须显式 `export OBLATE_CSV_PATH=data/nasa_archive/ps_tran_oblate_inputs_valid_20260416.csv`（见 `submit_multi_system_chunks.sh`、`test_chunk.sh`、`test.sh` 注释）。
- **C1 判据**：粗、细网格 \(\chi^2\) 合并后须对 \((f,\theta)\) **去重**再取 \(\chi^2_{(2)}-\chi^2_{(1)}\)，否则无噪声下出现假简并；实现 `C1_DEDUP_CONFIG`、`_dedupe_chi2_by_f_theta`（`batch_noiseless_recovery.py`）。
- **578 多系统 Slurm**：`submit_multi_system_chunks.sh` → `test_chunk.sh` → `merge_multi_system_slurm.sh`；chunk 独占目录，**禁止**并发写同一 `multi_system_summary.csv`。最终汇总：`results/multi_system/multi_system_summary.csv`、`run_timing.txt`。
- **`n_clean≠0` 子集导出**：`data/nasa_archive/multi_system_summary_n_clean_nonzero.csv`（**123** 行 = 122 数据 + 表头，以当前仓库为准）。
- **补跑 Tier B**：对 B 或合并表设置 `OBLATE_CSV_PATH`，用 `oblateness_input_tier`、`tier_a_legacy_row_index`、`keyparams_row_index`、`injection_batch_578_completed` 区分行语义、避免与已完成 578 重复；新宿主 LD：`prefetch_exotic_ld_data`（`--all-csv` 当前针对 valid 578，扩到 B/合并表需 Executor 对齐参数）。
- **Planner 仍 Pending**：Task Board「细化椭率探测的数学模型」。

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

**筛选文件说明（Tier A，沿用）**：`data/nasa_archive/ps_tran_oblate_inputs_valid_20260416.csv` 自 `ps_tran_flag_keyparams_20260416.csv` 保留「凌星+恒星+轨道」建模直接需要的列且下列字段为**非空数值**且对应 `*lim` 为 **0**（空白的 `lim` 视为 0）：`pl_orbper`, `pl_ratdor`, `pl_ratror`, `pl_orbincl`, `pl_tranmid`, `pl_orbeccen`, `st_teff`, `st_logg`, `st_met`。其余列原样保留便于后续分析；**578** 个行星（+表头）。未要求 `pl_imppar` 等，Executor 可做一致性检查。下文称此类行为 **`A_archive_direct`**（档案直接列齐全）。

### NASA 可建模输入：Tier B（派生几何）+ 合并表与防重复注入（Planner → Agent 2）

**问题**：档案 `ps` 中部分系统（如 TOI-2537 b）**有** `pl_orbsmax`（半长轴，常见 **AU**）与 **`st_rad`**（恒星半径，常见 **\(R_\odot\)**），但 **`pl_ratdor`（\(a/R_\star\)）为空**，被 Tier A 误筛；仍可在**单位自洽**下派生 \(a/R_\star\) 参与 `OblateSystem` 与光变生成。

**目标（先松后严）**：在**同一张** `ps_tran_flag_keyparams` 快照上，对 **Tier A 未收录**的行，用**较宽松**规则找回「仅缺 `pl_ratdor`（及可选缺 `pl_ratror`）但可派生」的源；写入**合并输入表**时**必须与 Tier A 区分**，并带**防重复批量注入**字段，避免对已跑完 578 多系统注入测试的系统再提交。

---

**Tier B（`B_derived_geometry`，宽松首版）** 纳入条件（在 **`tran_flag=1` 且 `default_flag=1`** 的 keyparams 子集上操作，与全表下载一致）：

1. **不在 Tier A**：该行不满足 Tier A 的「九列直接有效」定义（通常表现为 **`pl_ratdor` 空或 lim 非 0** 等）。  
2. **其余与 Tier A 同要求的直接列**仍须满足（非空、可解析、`lim=0` 规则与 Tier A 相同）：  
   `pl_orbper`, `pl_orbincl`, `pl_tranmid`, `pl_orbeccen`, `st_teff`, `st_logg`, `st_met`。  
3. **`pl_ratror`**：若档案已有且通过 lim，则用档案值；若为空，**可选（本阶段宽松）** 用 `pl_rade`（地球半径）与 `st_rad`（\(R_\odot\)）派生 \(R_p/R_\star\)，**仅当**两列均有效且 lim 合规；**单位与列定义必须以** [API PS columns](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html) **为准**，Executor 实现前在代码注释或常量处写明换算。  
4. **派生 \(a/R_\star\)**：当 `pl_ratdor` 不可用但 **`pl_orbsmax`** 与 **`st_rad`** 均有效且对应 `*lim` 合规时，  
   \[
   (a/R_\star)_\mathrm{derived} = \frac{a\,[\mathrm{AU}]}{R_\star\,[R_\odot]} \times \frac{1\,\mathrm{AU}}{R_\odot},
   \]  
   其中 \(1\,\mathrm{AU}/R_\odot\) 与仓库 `constants`（IAU/CODATA 一致）对齐；**若档案 `pl_ratdor` 非空且通过 Tier A，禁止覆盖**，以档案为准。  
5. **质量过滤（宽松）**：派生值须为**有限正数**且在合理范围（如 \(a/R_\star>0\)、\(R_p/R_\star\le 1\) 等简单 sanity check）；极端异常行可记日志后丢弃或标 `QC_flag`。

---

**合并输出 CSV（建议路径，Executor 可微调文件名/日期）**  
例如：`data/nasa_archive/ps_tran_oblate_inputs_combined_20260416.csv`（或新日期后缀）。

**必须包含的区分与防重复列**（由 Agent 2 生成，不得手填遗漏）：

| 列名 | 含义 |
|------|------|
| **`oblateness_input_tier`** | **`A_archive_direct`**：与现 `ps_tran_oblate_inputs_valid_20260416.csv` 行一一对应（原 578）；**`B_derived_geometry`**：本轮新找回、至少对 `pl_ratdor` 或 `pl_ratror` 使用了派生。 |
| **`pl_ratdor_source`** | `archive` \| `derived_orbsmax_st_rad`（若派生则同时可在旁路保留档案空列或写入派生列名如 `pl_ratdor_used`，由 Executor 定一种即可，但须在 Session Log 说明）。 |
| **`pl_ratror_source`** | `archive` \| `derived_rade_st_rad` \| `missing_policy`（若仅 archive 则填 archive）。 |
| **`keyparams_row_index`** | 该行在 **`ps_tran_flag_keyparams_*.csv`** 中的 **0-based 数据行索引**（表头下一行为 0），用于回查原始 NASA 行与复现筛选。 |
| **`tier_a_legacy_row_index`** | 若本行属于 Tier A：**在旧文件 `ps_tran_oblate_inputs_valid_20260416.csv` 中的 0-based 行号**（0…577）；若为 Tier B：**留空或 -1**。 |
| **`injection_batch_578_completed`** | **布尔或 0/1**：若该行星**已出现在** `results/multi_system/multi_system_summary.csv`（或已合并的 Slurm 全量汇总）且成功跑过注入网格（可按 `error` 空且存在 `n_clean`/`n_degenerate`/`n_failed` 判定），则为 **1**；否则 **0**。Tier B 新行默认 **0**。更新汇总后由脚本**回填**或单次 join 生成。 |

**Slurm / 多系统提交规则**：

- **仅对** `injection_batch_578_completed == 0` **且** `oblateness_input_tier == B_derived_geometry` **的行**（或用户显式给出的补跑列表）提交新 chunk，避免对已完成 578 系统重复计算。  
- Tier A 行若需重跑，须**显式**改标志或单独 `--rows`，不得与默认「只补 B」混用。

**验收**：合并表行数 = 578 +（新找回的 Tier B 行数）；**578 行的 `oblateness_input_tier` 全为 `A_archive_direct`** 且 **`tier_a_legacy_row_index` 与现有多系统 `row_index` 可对应**；抽查 TOI-2537 b 若被找回应在 Tier B 且 `pl_ratdor_source=derived_orbsmax_st_rad`。

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

**超算 `test.sh`（与 Startrack 作业模板对齐）**：`#SBATCH` 使用分区 `64c512g`、`--account=acct-tdlffb`、`--nodes=1`、`--ntasks-per-node=1`、`--cpus-per-task=4`、`--mem=16G`、`--output=logs/%j.out`；`set -euo pipefail`；`source activate astro_ml`；`WORKDIR` 指向服务器上本仓库路径。**结果路径**：脚本内 **`RESULTS_SUBDIR`**（默认 `multi_system` → 写入 `results/<subdir>/`）或 **`RESULTS_DIR`**（非空时为**绝对路径**，调用 `python -m oblateness.multi_system_batch_noiseless --output-dir`，等价环境变量 `OBLATE_MULTI_OUTPUT_DIR`）。详见 Session Log 2026-04-18（Executor）第二条。

### 多系统 Slurm 并行与汇总（578 系统 ≈ 20 组）— 委派 Agent 2（2026-04-20）

**用户目标**：在超算上**明显加快整批**（`ps_tran_oblate_inputs_valid_*.csv` 共 **578** 条数据行，`row_index` **0…577**），将索引分为约 **20** 组并行 `sbatch`，多作业**同时**运行；最终在**统一路径**交付：

- `results/multi_system/multi_system_summary.csv`
- `results/multi_system/run_timing.txt`

**分组（规格）**

- 组数 **`N_GROUPS ≈ 20`**（可配置常量，如 20；若需整除可改为 17×34 等，以黑板/代码注释为准）。
- **行号集合**：对 0…577 **连续分段**、**无重叠、无遗漏**（例如 `ceil(578/20)=29`，前若干组 29 行、最后一组取余；或尽量均分使每组相差不超过 1）。
- 每组调用与现有多系统入口一致：`python -m oblateness.multi_system_batch_noiseless --rows <逗号分隔>`（或等价环境变量），**不得**在两组中重复同一 `row_index`。

**并行阶段输出目录（必选，避免竞态）**

- **禁止**多个 Slurm 作业**同时**读写**同一个** `results/multi_system/multi_system_summary.csv`。当前实现为「读已有 + 合并 + 写回」，并发会导致**丢失更新/损坏**；**不能**依赖「各组完成时刻几乎不同」作为同步机制。
- 每组作业使用**独占**输出根目录，例如：  
  `results/multi_system/_chunks/chunk_00/` … `chunk_19/`（子目录名与分段规则由实现定，须固定、可复现）。  
  每组在该目录内生成与现行为一致的 per-row `status_*.png`、`batch_*.npz`，以及**本组** `multi_system_summary.csv`（或等价中间表）与（可选）**本组** `run_timing.txt`。

**合并阶段（单次写最终文件）**

- 全部 chunk 作业**成功结束后**，再执行**一次**合并步骤（独立 Slurm 作业 `afterok` 依赖链，或文档化的人工命令二选一，**优先自动化**）：
  1. **汇总 CSV**：将各 chunk 的 summary **按 `row_index` 升序合并**为单一 `results/multi_system/multi_system_summary.csv`（列名与现 CSV 一致；若 chunk 内已有表头，合并时去重表头）；确认 **578 行数据**（或失败行在 `error` 列记录后的总行数）与无重复 `row_index`。
  2. **总计时文件** `results/multi_system/run_timing.txt`：写入**结构化多段**内容，至少包含：每个 chunk 的 `wall_seconds`、`SLURM_JOB_ID`（若有）、`row_indices` 范围或条数；可选增加「合并作业提交时间」「从首作业提交到合并完成」的日历耗时说明。**禁止**依赖「最后一组覆盖写」冒充全表计时。

**提交侧（实现建议）**

- 仓库提供 **shell 驱动脚本**（名称由 Executor 定，如 `submit_multi_system_chunks.sh`）：根据 `N_GROUPS` 打印或提交 20 个 `sbatch`，传入 `CHUNK_ID`、`ROWS`、`OBLATE_MULTI_OUTPUT_DIR`（或 `--output-dir`）。  
- `test.sh` 可改为接受环境变量/参数以适配「单 chunk」模式，或新增 `test_chunk.sh` 避免破坏原单作业用法。  
- 合并可提供 `python -m oblateness.<merge_module>` 或 `tools/merge_multi_system_chunks.py`，**仅读**各 chunk 目录、**写**最终两文件。

**验收**

- 并行 20 作业 + 1 合并（或等价）后，`results/multi_system/multi_system_summary.csv` 覆盖 578 系统；`run_timing.txt` 可追溯各 chunk 墙钟与作业号。  
- Session Log 由 Executor 追加：脚本路径、示例提交命令、合并命令。

### Agent 2 委派：无噪声恢复 **两态化（仅 A1）** + **禁止覆盖既有结果**（用户定案，待实现）

**动机**：无噪声阶段后续将加噪声测试；当前 **C1（次小 \(\chi^2\) 间隙）** 标度与离散网格耦合，易把好系统判成「简并」。先采用**较弱、可解释**的划分，避免 C1 进入主结论。

**判据定案（取代原「clean = A1∧C1」）**

- **成功**：满足 **A1**（与现实现一致）：粗+细网格合并后的全局最优 \((\hat f,\hat\theta)\) 满足 \(|\hat f-f_\mathrm{inj}|<\epsilon_f\)、\(|\hat\theta-\theta_\mathrm{inj}|<\epsilon_\theta\)（\(\epsilon\) 仍为 **细网格**步长之半；\(\theta\) 用既有最短角距规则）。
- **失败**：不满足 A1。
- **不再输出「简并」作为第三态**：原 `degenerate` 取消；若需向后兼容旧汇总列，可令 `n_degenerate` **恒为 0** 并注释废弃，或改为仅两列 **`n_success` / `n_failed`**（推荐后者，由 Executor 统一重命名并更新 `merge_multi_system_chunks.py`、测试与 Slurm 脚本中列期望）。

**C1 / \(\chi^2\) 次小间隙（可选保留）**

- **不得**再参与 `status` 或 success/fail 计数。
- 建议**仍计算**并写入 per-injection 记录（及 npz）：去重后的 \(\chi^2_{(1)},\chi^2_{(2)},\Delta\)、`n_chi2_unique_c1` 等，字段可保留原名或加 `_diagnostic` 后缀，供日后噪声阶段或相对阈值实验使用；`DELTA_CHI2_CONFIG` 可保留但仅用于这些诊断字段，避免删代码过多。

**须修改的代码（路径与要点）**

1. **`src/oblateness/batch_noiseless_recovery.py`**
   - `Status`（或等价类型）：仅 **`success` | `failed`**（若保留字符串 `clean` 表示成功，须在黑板/CSV 列名一处统一说明，避免与旧图例混淆）。
   - `run_single_injection`：`status = success if a1 else failed`；去掉对 C1 的分支。
   - `_save_status_figure`：仅两种颜色/图例（成功 / 失败）；标题或说明改为「A1 only」。
   - `_save_npz`：仍写入 `chi2_1`, `chi2_2`, `chi2_gap`, `A1`；`C1` 可改为恒 `nan` / 省略，或保留数值但**不用于分类**（Executor 选一种并在 Session Log 说明）。
   - `BATCH_RUN_CONFIG`：**默认输出文件名或子目录**改为新路径（见下节），不得仍指向会覆盖旧物的 `batch_noiseless_f_theta_status.png` / `batch_noiseless_recovery.npz`。

2. **`src/oblateness/multi_system_batch_noiseless.py`**
   - 汇总行：按两态统计 `n_success`、`n_failed`（或等价命名）；更新 `fieldnames` 及追加/合并逻辑。
   - 默认 `output_subdir` 或文档要求：指向**新**子目录（见下），避免默认写 `results/multi_system/` 覆盖历史。

3. **`src/oblateness/merge_multi_system_chunks.py`**（若列名变更）
   - 合并时期望的 summary 列与最终写出路径对齐；合并产物同样写入**新文件名或新目录**。

4. **测试**
   - 更新断言 `status` / 计数中不再出现 `degenerate` 的用例（若有）；必要时新增烟测：同一注入在仅 A1 下为 success。

5. **Shell（`test.sh` / `test_chunk.sh` / `submit_multi_system_chunks.sh`）**
   - 默认或注释示例：`RESULTS_SUBDIR`、`OBLATE_MULTI_OUTPUT_DIR`、`RESULTS_DIR` 指向**新**目录（如 `multi_system_a1_only`）；合并脚本输出 **新** `multi_system_summary_*.csv`，**禁止**默认覆盖现有 `results/multi_system/multi_system_summary.csv`。

**输出与版本纪律（硬性：用户要求）**

- 本变更**实施后任何重跑**，产物须写入**新位置**（新子目录和/或带版本/日期后缀的文件名），**不得覆盖**当前仓库中已有结果，包括但不限于：
  - `results/multi_system/multi_system_summary.csv`、`run_timing.txt`、`status_*.png`、`batch_*.npz`、`_chunks/`、`_chunks_tierb/`；
  - `results/batch_noiseless_f_theta_status.png`、`results/batch_noiseless_recovery.npz`。
- **建议约定**（Executor 实现时二选一或组合，并在 Session Log 写明实际路径）：
  - 子目录：`results/multi_system_a1_only/`（chunk 则 `_chunks/` 在其下）；单星批量：`results/batch_noiseless_a1_only/`；或
  - 统一后缀：`*_a1only_20260416`（日期以运行当日为准）。
- 若需对比旧结果，只读旧路径；新汇总与新导出 CSV（如 `data/nasa_archive/multi_system_summary_n_clean_nonzero.csv` 的续表）**另存新文件**，不得覆盖旧表，除非用户显式要求。

### Agent 2 委派：时间采样——**60 s cadence**、按系统计算的 **Ingress/Egress（I/E）**，及**长凌星点数控制**（用户定案，待实现）

**物理与工作假设**

- **先假定**平台（全食）段的扁率与非扁率在流量差分上对当前 \(\chi^2\) **可忽略**；则 \(\chi^2\) **仅须在 I/E 上累加**，与全时标等价（在用户假设下）。
- **`OblateSystem`/`lightcurve` 仅在 I/E 对应的时间向量上调用**：不再对整个「含平坦底」跨度生成致密 `times`，以减轻计算。

**I/E 起止时间与「由参数计算」**

- 每个系统的 ingress / egress **时间边界**必须由**该系统**可用的几何/轨道量算得（如在圆轨、小黑点近似或与小圆模型一致前提下，由 \(a/R_\star\)、\(R_p/R_\star\)、轨道倾角、（若用）偏心率、`t_0`、`P` 等推出**内侧/外侧接触对应的轨道相位**，再换回时间；或对 `squishyplanet` 使用的同一几何定义）。
- NASA 快照中若在 `tran_flag`/keyparams 内存在 **`pl_trandur`、`pl_ratdor`、`pl_imppar` 或接触相关列**，应明确：**档案 `pl_trandur`** 常为 FWHM 或等价全宽表述，是否与「一/四接触」定义的 I/E **一致**必须由 Executor **对照实现与文档**选一个**自洽约定**（Planner 不写死公式）。
- **端点外延（必须）**：在算得的 I/E 段首、末时间点外，各自再外延「适当」时间点，避免触点恰落在离散网格空隙上导致少采样；外延量建议可配置：**至少数个标称 \(\Delta t\)**（见下），或按 ingress 持续时间的小比例外延，写入代码注释并在 Session Log 写一句默认。

**时间与采样（JWST 参考）**

- **相邻样本间距标称**：**60 s**（参考 JWST 曝光量级；实现为常量，若以日传入 `squishyplanet` 则用 \(60/(86400)\,\mathrm{d}\)）。
- **`times`** 仅由 **Ingress 时间段 ∪ Egress 时间段**（含上述外延端点）经均匀或受控抽样得到；**不得在平台段打点**调用光变生成。

**长凌星与点数上限（必须，避免点数爆炸）**

- 仅用 60 s 跨很长 I+E 仍会极多点；必须引入**确定性、可复现**的策略控制总点数 \(N\) 或等价上限，任选或组合，并在 `PLANET_CONFIG` / 专用 config 字典中可调，默认值在 Session Log 说明：
  - **段内点数上限**：对 ingress、egress **各**设 `n_ie_max`，超出则对已生成的均匀序列进行**确定性子采样**（如每隔 \(k\) 点取一点，或对弧长按索引重采样）；
  - **或**：有效间隔设为 \(\Delta t = \max(60\,\mathrm{s},\,\Delta t_\mathrm{adapt})\)，其中 \(\Delta t_\mathrm{adapt}\) 与 ingress+egress 总持续时间挂钩；
  - **或**：总 `times.size` **硬顶**再在 I/E 间按比例分配点数。
- 目标：在长凌星系外行星上仍可在合理墙钟内跑完网格；具体数值由 Executor **回填报表**黑板一行（不写大表）。

**\(\chi^2\) 与实现对齐**

- 若全流程仅构造 I/E 的 `times`，则 \(\chi^2=\sum_k (\cdots)^2\) **只在那些时刻**——与「mask 全时标」在假设下一致而无需在全时标后再 mask。

**不改旧结果的纪律**

- 本特性落地后的管道输出须落在 **新目录/新版本名**（与现有 `*_a1_only`、`multi_system*` 并行），**不得覆盖**已实现跑出的旧路径；Session Log **注明目录名**。

**实现位置（指路）**

- `src/oblateness/noiseless_grid_chi2_inversion.py`、`src/oblateness/batch_noiseless_recovery.py`（`_build_system` 与时间轴）；`tests/test_nasa_first_row_oblate_lightcurve.py` 若复用同一时间构造则同步；必要时抽 **小模块**（如 `transit_windows.py` 或等价）计算 I/E。

## [Task Board]

- [x] Executor: **JWST cadence（60 s）+ 按系统参数的 I/E 时间窗**：仅在该窗内生成光变与 \(\chi^2\)，端点外延可配；长凌星 `times` 有段内上限子抽样；新图/npz：`results/noiseless_ie60/`、`results/batch_noiseless_a1_only_ie60/`（见 `transit_ie_sampling.py`；`OBLATE_TIME_SAMPLING_MODE=uniform_symmetric` 回退均匀窗）
- [x] Executor: **无噪声恢复两态化（仅 A1）**：去掉 C1 对 `status` 的影响；汇总与图例仅 success/failed；C1 相关量保留诊断；单星批量图/npz 写入 `results/batch_noiseless_a1_only_ie60/`（与 I/E 采样同目录；勿覆盖旧的 `results/multi_system/*`）；多系统默认 `results/multi_system_a1_only/`、合并 `multi_system_summary_a1only.csv` / `run_timing_a1only.txt`（见专节）
- [ ] Planner: 细化椭率探测的数学模型 (Pending)
- [x] Executor: **Tier B 派生 + 合并表**：`python -m oblateness.build_tier_b_oblate_inputs` → `data/nasa_archive/ps_tran_oblate_inputs_tier_b_derived_20260416.csv`（164 行 B）、`ps_tran_oblate_inputs_combined_tierab_20260416.csv`（578+164=742）；原 `ps_tran_oblate_inputs_valid_20260416.csv` 未改；`injection_batch_578_completed` 与 `multi_system_summary.csv` join（见 Session Log 2026-04-21）
- [x] Executor: **578 系统 ≈20 组并行 Slurm** + 分 chunk 独占目录 + **单次合并** → `results/multi_system/multi_system_summary.csv` 与 `results/multi_system/run_timing.txt`（`submit_multi_system_chunks.sh` → `test_chunk.sh` → `merge_multi_system_slurm.sh` → `merge_multi_system_chunks`；见 Session Log 2026-04-20）
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
- 2026-04-18 (Executor): `test.sh` 已与集群常用模板对齐（`partition=64c512g`、`account=acct-tdlffb`、`cpus-per-task=4`、`logs/%j.out`、`source activate astro_ml`）。**可改结果目录**：`RESULTS_SUBDIR`（相对 `results/`）或 `RESULTS_DIR`（绝对路径，透传 `--output-dir` / `OBLATE_MULTI_OUTPUT_DIR`）。代码：`multi_system_batch_noiseless.run_multi_system(..., output_dir=...)` 支持仓库外写入；CSV 中路径在仓库外时为绝对路径字符串。
- 2026-04-18 (Executor): `multi_system_summary.csv` 默认**追加**本次运行行（不覆盖历史）；`--summary-overwrite` 整表重写。无外网 HPC：在联网机运行 `python -m oblateness.prefetch_exotic_ld_data --rows ...` 将 ExoTiC-LD 所需文件拉取到 `data/exotic_ld_data/`（与 `exotic_ld` 的 `mps1/.../mps1_spectra.dat`、`Sensitivity_files/..._throughput.csv` 结构一致），再上传至服务器。
- 2026-04-20 (Executor): `prefetch_exotic_ld_data` 增加 `--all-csv`（对 `ps_tran_oblate_inputs_valid_*.csv` 全部 578 行逐星预取；已存在文件跳过）；下载带网络退避重试。全表预取：`PYTHONUNBUFFERED=1 python -m oblateness.prefetch_exotic_ld_data --all-csv 2>&1 | tee logs/prefetch_exotic_ld_all.log`。
- 2026-04-20 (Chronicler): 用户要求 **578 系统分约 20 组并行 `sbatch`**，最终汇总至同一 `results/multi_system/multi_system_summary.csv` 与 `run_timing.txt`。黑板新增专节「多系统 Slurm 并行与汇总」：分 chunk 独占目录、**单次合并**（禁止并发写同一 CSV）；Task Board 新增对应 **Executor** 项。实现委派 **Agent 2**。
- 2026-04-20 (Executor): 实现并行与合并：**分块** `src/oblateness/multi_system_chunks.py`（`partition_indices`、默认 578 行 / 20 组、前 18 组 29 行、后 2 组 28 行；`python -m oblateness.multi_system_chunks --list`）；**单 chunk 作业** `test_chunk.sh`（`CHUNK_ID`+`ROWS` → `results/multi_system/_chunks/chunk_XX/`、`--summary-overwrite`、`OBLATE_CHUNK_ID`）；**合并** `merge_multi_system_chunks.py`（无 `chunk_*` 时拒绝覆盖最终 CSV；`merge_multi_system_slurm.sh`）；**提交** `submit_multi_system_chunks.sh`（`sbatch`×20 + `afterok` 链式 `merge`）。`multi_system_batch_noiseless` 的 `run_timing.txt` 增加 `row_index_min/max`、`chunk_id`（若设 `OBLATE_CHUNK_ID`）。
- 2026-04-21 (Planner): 黑板新增 **Tier B（派生 \(a/R_\star\) 等）** 与 **合并输入表** 规格：`oblateness_input_tier`（`A_archive_direct` / `B_derived_geometry`）、`pl_ratdor_source` / `pl_ratror_source`、`keyparams_row_index`、`tier_a_legacy_row_index`、`injection_batch_578_completed`；多系统补跑**默认仅提交 B 且未完成 578 注入**，避免重复。委派 **Agent 2** 实现筛选脚本与 CSV；**Agent 1 不改 `src/`**。
- 2026-04-21 (Executor): **Tier B + 合并表** 落地：`src/oblateness/build_tier_b_oblate_inputs.py`。`tran_flag=1`∧`default_flag=1`（`ps_tran_flag_default_20260416` join）、非 Tier A 578、七列核心+`pl_orbsmax`/`st_rad` 派生 `pl_ratdor`（`sc.au`/`R_sun`）、`pl_ratror` 取档案或 `pl_rade`（default）×`R_⊕`/(`st_rad`×`R_sun`），`R_⊕=6.3781e6` m。输出：**仅 B** `ps_tran_oblate_inputs_tier_b_derived_20260416.csv`（164）；**A+B** `ps_tran_oblate_inputs_combined_tierab_20260416.csv`（742）；**未修改** `ps_tran_oblate_inputs_valid_20260416.csv`。验收：TOI-2537 b 在 B 且 `pl_ratdor_source=derived_orbsmax_st_rad`。
- 2026-04-21 (Chronicler): 在 **[Project State]** 增补 **「上下文续接要点」**（会话压缩 / 新 Chat 恢复）：黑板文件名、`Tier A/B/合并` 路径与行数、`OBLATE_CSV_PATH` 默认与 578 重跑、C1 去重、Slurm chunk/merge、`n_clean≠0` 导出路径、Tier B 补跑与 LD 预取、Planner 待办指针。
- 2026-04-22 (Planner/User): **无噪声批量判据变更**写入黑板专节「Agent 2 委派：无噪声恢复两态化（仅 A1）」：**成功/失败**仅由 **A1** 决定；**C1 不参与**分类，相关量可保留为诊断。实现后重跑须落在**新路径**，**禁止覆盖**现有 `results/` 已提交产物；Task Board 已增对应 Executor 项。
- 2026-04-22 (Executor): **两态化（仅 A1）+ 新结果路径**：`batch_noiseless_recovery` 的 ``status`` 仅为 ``success``/``failed``（A1）；C1 仍写入每条结果/npz 作诊断；后接 I/E 采样时图/npz 目录为 ``batch_noiseless_a1_only_ie60/``。``multi_system_batch_noiseless`` 默认 ``results/multi_system_a1_only/``，汇总列 ``n_success``/``n_failed``。``merge_multi_system_chunks`` 默认读 ``multi_system_a1_only/_chunks_tierb``，写 ``multi_system_summary_a1only.csv``、``run_timing_a1only.txt``；chunk 内旧三列汇总读入时规范化为 ``n_success=n_clean+n_degenerate``。``test_chunk.sh``/``submit``/``test.sh`` 默认指新目录；``build_tier_b_oblate_inputs`` join 兼容新/旧列。测试：``tests/test_multi_system_chunks.py``。
- 2026-04-22 (Executor): **60 s cadence + I/E 窗**（Planner 专节）：新增 ``src/oblateness/transit_ie_sampling.py``（圆轨接触时刻、b=(a/R_*)cos i、v_sky；ingress/egress 并网、外延 ``edge_pad_cadences``、段内 ``n_ie_segment_max`` 确定性子采样；e 大或几何无效回退均匀窗）。``batch_noiseless_recovery._build_system`` 与 ``noiseless_grid_chi2_inversion.run_grid_inversion`` 经 ``build_lightcurve_time_array_days`` 接 ``PLANET_CONFIG['time_sampling']``；``noiseless`` 图/npz 默认 ``results/noiseless_ie60/``；``OBLATE_TIME_SAMPLING_MODE=uniform_symmetric`` 回退旧对称窗。测试 ``tests/test_transit_ie_sampling.py``; ``test_nasa_first_row_oblate_lightcurve`` 同步。
- 2026-04-22 (Planner/User): 委派 **Executor**：在无噪声 \(\chi^2\) 假定**平台段可不计**前提下，仅在 **ingress/egress** 采样并生成光变；I/E **边界由各系统轨道/几何参数算出**（与档案列定义自洽），段末适当外延打点；间隔标称 **60 s**（JWST 参考）；**长凌星系须有点数上限或可变相隔**以免爆炸（见黑板专节）。
