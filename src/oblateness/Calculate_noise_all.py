from tqdm import tqdm
from scipy.interpolate import interp1d
import copy
import os
from math import ceil
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const
import corner
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from exotic_ld import StellarLimbDarkening
from scipy.interpolate import CubicSpline
from squishyplanet import OblateSystem
import emcee

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LD_DATA_PATH = Path(
    os.environ.get("EXOTIC_LD_DATA_PATH", PROJECT_ROOT / "data/exotic_ld_data")
)
PANDEIA_DATA_PATH = Path(
    os.environ.get("pandeia_refdata", PROJECT_ROOT / "data/pandeia_data")
)
PYSYN_CDBS_PATH = PANDEIA_DATA_PATH / "sed/grp/redcat/trds"
TARGET_TABLE_PATH = (
    PROJECT_ROOT
    / "data/nasa_archive/ps_tran_oblate_inputs_combined_tierab_20260416.csv"
)

if not LD_DATA_PATH.exists():
    raise FileNotFoundError(f"LD data path not found: {LD_DATA_PATH}")

if not PANDEIA_DATA_PATH.exists():
    raise FileNotFoundError(f"Pandeia data path not found: {PANDEIA_DATA_PATH}")

if not (PYSYN_CDBS_PATH / "grid/phoenix/catalog.fits").exists():
    raise FileNotFoundError(f"Phoenix catalog not found: {PYSYN_CDBS_PATH / 'grid/phoenix/catalog.fits'}")

# Pandeia reads this environment variable while importing pandeia.engine modules.
os.environ["pandeia_refdata"] = str(PANDEIA_DATA_PATH)
os.environ["STSYNPHOT_DATA"] = str(PANDEIA_DATA_PATH)
os.environ["PYSYN_CDBS"] = str(PYSYN_CDBS_PATH)

from pandeia.engine.calc_utils import build_default_calc
from pandeia.engine.perform_calculation import perform_calculation


# =========================================================================================================
# 全局设置
# =========================================================================================================
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)

np.random.seed(13)

# New NASA table columns keyed by the old calculate_noise parameter names.
old_to_new_columns = {
    "mh": "st_met",          # stellar metallicity [Fe/H]
    "teff": "st_teff",       # stellar effective temperature [K]
    "logg": "st_logg",       # stellar log(g) [cgs]
    "P": "pl_orbper",        # orbital period [days]
    "a": "pl_ratdor",        # scaled semi-major axis a/R_star, not AU
    "R_p": "pl_radj",        # planet radius [Jupiter radius]
    "R_*": "st_rad",         # stellar radius [Solar radius]
    "i": "pl_orbincl",       # orbital inclination [degree]
    "kmag": "sy_kmag",       # 2MASS K-band magnitude
    "trandur": "pl_trandur", # transit duration [hours]
}

OLD_PARAM_ALIASES = {
    "mh": "M_H",
    "teff": "Teff",
    "logg": "logg",
    "P": "P",
    "a": "a",
    "R_p": "R_p",
    "R_*": "R_*",
    "i": "i",
    "kmag": "kmag",
    "trandur": "trandur",
}

TARGET = "TOI-2537 b"
RUN_ALL_TARGETS = True
NOISE_CACHE_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "results/calculate_noise_cache_combined_tierab_20260416.csv"
RESUME_EXISTING_CACHE = True
BATCH_MAX_TARGETS = None

# ngroup 搜索设置
PREDICT_A = 0.0014419849148
PREDICT_B = 0.920

HARD_MAX_NGROUP = 1024
VERBOSE = True


# =========================================================================================================
# 工具函数
# =========================================================================================================
def get_float(params, key):
    """从 params 字典中读取浮点数，并提供更清楚的报错信息。"""
    if key not in params:
        raise KeyError(f"CSV 中缺少必要列：{key}")

    value = params[key]

    if pd.isna(value):
        raise ValueError(f"参数 {key} 的值为空，无法继续计算。")

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"参数 {key} = {value} 无法转换为 float，请检查 CSV。") from exc


def validate_input_table_columns(df):
    """确认输入表含有 calculate_noise 所需的新表字段。"""
    target_column = "pl_name"
    if target_column not in df.columns:
        raise KeyError(f"CSV 中没有目标名称列 '{target_column}'，请检查新表表头。")

    missing_columns = [
        new_col
        for new_col in old_to_new_columns.values()
        if new_col not in df.columns
    ]
    if missing_columns:
        details = ", ".join(
            f"{old_name}->{new_col}"
            for old_name, new_col in old_to_new_columns.items()
            if new_col in missing_columns
        )
        raise KeyError(f"CSV 中缺少必要的新表字段：{details}")


def row_to_params(row):
    """把新表一行转换成旧 calculate_noise 计算逻辑使用的参数名。"""
    params = {}
    for old_name, new_col in old_to_new_columns.items():
        params[OLD_PARAM_ALIASES[old_name]] = row[new_col]
    return params


def load_target_params(csv_path, target_name):
    """读取目标行星参数。"""
    df = pd.read_csv(csv_path)
    validate_input_table_columns(df)

    target_column = "pl_name"
    matched = df.loc[df[target_column] == target_name]

    if matched.empty:
        raise ValueError(f"目标 {target_name} 不在 CSV 文件中，请检查 {target_column} 是否完全一致。")

    if len(matched) > 1:
        print(f"警告：目标 {target_name} 在 CSV 中出现了 {len(matched)} 次，将使用第一行。")

    row = matched.iloc[0]
    return row_to_params(row)


def print_target_params(
    teff, logg, mh, ksmag,
    period, a_s, r_star, r_jup, r_s, t14,
    b=None, i=None
):
    """打印目标参数。b 和 i 仅用于展示，不参与 Pandeia 噪声计算。"""
    print("=== Stellar & Planetary Parameters ===")
    print(f"Teff       = {teff} K")
    print(f"logg       = {logg}")
    print(f"[M/H]      = {mh}")
    print(f"Kmag       = {ksmag}")
    print(f"Period     = {period} days")
    print(f"a/R*       = {a_s}")
    print(f"R_star     = {r_star} R_sun")
    print(f"R_planet   = {r_jup} R_jup")
    print(f"a/R_star   = {a_s:.4f}")
    print(f"R_p/R_star = {r_s:.4f}")
    print(f"t14        = {t14:.6f} hours")

    if b is not None:
        print(f"impact parameter = {b}")

    if i is not None:
        print(f"inclination      = {i} degree")
        print(f"inclination      = {i * np.pi / 180} rad")

    print("=" * 50)

def predict_ngroup_from_kmag(ksmag, a=PREDICT_A, b=PREDICT_B):
    """
    根据经验关系 n = a * exp(b * Kmag) 粗略预测最大不过曝 ngroup。

    注意：
    这个预测值只作为搜索初值。
    最终 n_f 仍然由 Pandeia saturation check 决定。
    """
    n_pred = a * np.exp(b * float(ksmag))
    n_pred = int(round(n_pred))
    n_pred = max(1, n_pred)
    return n_pred

def check_saturation_cached(base_calculation, ngroup, cache, verbose=False, label=""):
    """
    检查某个 ngroup 是否饱和，并使用 cache 避免重复跑 Pandeia。
    """
    ngroup = int(ngroup)

    if ngroup in cache:
        sat, n_sat = cache[ngroup]
        if verbose:
            status = "过曝" if sat else "不过曝"
            print(f"[缓存{label}] n = {ngroup}, 结果 = {status}")
        return sat, n_sat

    report = run_calculation(base_calculation, ngroup, nint=1)
    sat, n_sat = is_saturated(report)

    cache[ngroup] = (sat, n_sat)

    if verbose:
        if sat:
            print(f"[检查{label}] n = {ngroup}, 结果 = 过曝, 像素数 = {n_sat}")
        else:
            print(f"[检查{label}] n = {ngroup}, 结果 = 不过曝")

    return sat, n_sat


def find_max_unsaturated_ngroup_predictive_walk(
    base_calculation,
    ksmag,
    hard_max_ngroup=1024,
    verbose=True
):
    """
    使用 K-band magnitude 经验公式预测 ngroup 初值，
    然后在预测值附近逐步 +1 或 -1 遍历，寻找最大不过曝 ngroup。

    逻辑：
    1. 由 n = a * exp(b * Kmag) 得到 n_pred；
    2. 先检查 n_pred；
    3. 如果 n_pred 不过曝，则向上逐个检查，直到第一次过曝；
    4. 如果 n_pred 过曝，则向下逐个检查，直到第一次不过曝；
    5. 如果一直向下到 n = 1 仍然过曝，则说明当前配置下无不过曝解。

    注意：
    不再预先检查 ngroup = 1。
    只有当向下遍历真的走到 n = 1 时，才检查它。
    """

    hard_max_ngroup = int(hard_max_ngroup)

    if hard_max_ngroup < 1:
        raise ValueError("hard_max_ngroup 必须 >= 1。")

    cache = {}

    # 根据经验公式预测初值
    n_pred = predict_ngroup_from_kmag(ksmag)
    n_pred = min(max(1, n_pred), hard_max_ngroup)

    if verbose:
        print(f"[经验预测] Kmag = {ksmag:.3f}, n_pred = {n_pred}")

    # 先检查预测值
    sat_pred, n_sat_pred = check_saturation_cached(
        base_calculation=base_calculation,
        ngroup=n_pred,
        cache=cache,
        verbose=verbose,
        label=" 预测值"
    )

    # =====================================================================================
    # 情况 1：预测值不过曝，向上 +1 查找第一次过曝
    # =====================================================================================
    if not sat_pred:
        if verbose:
            print("[搜索方向] 预测值不过曝，向上逐个查找第一次过曝。")

        best_unsaturated = n_pred
        n = n_pred + 1

        while n <= hard_max_ngroup:
            sat, n_sat = check_saturation_cached(
                base_calculation=base_calculation,
                ngroup=n,
                cache=cache,
                verbose=verbose,
                label=" 向上遍历"
            )

            if sat:
                if verbose:
                    print(f"[完成] 第一次过曝 n = {n}")
                    print(f"[完成] 最大不过曝 ngroup = {best_unsaturated}")

                return best_unsaturated

            best_unsaturated = n
            n += 1

        raise RuntimeError(
            f"直到 ngroup = {hard_max_ngroup} 仍未过曝。"
            f"请提高 HARD_MAX_NGROUP，或检查 saturation warning 是否正常返回。"
        )

    # =====================================================================================
    # 情况 2：预测值过曝，向下 -1 查找第一次不过曝
    # =====================================================================================
    if sat_pred:
        if verbose:
            print("[搜索方向] 预测值过曝，向下逐个查找第一次不过曝。")

        n = n_pred - 1

        while n >= 1:
            sat, n_sat = check_saturation_cached(
                base_calculation=base_calculation,
                ngroup=n,
                cache=cache,
                verbose=verbose,
                label=" 向下遍历"
            )

            if not sat:
                if verbose:
                    print(f"[完成] 最大不过曝 ngroup = {n}")

                return n

            n -= 1

        # 如果 n_pred 本身就是 1 且过曝，或者一路减到 1 仍然过曝，就会到这里
        raise RuntimeError(
            "ngroup = 1 时仍然过曝，无法在当前 NIRISS/SOSS 配置下找到不过曝设置。"
        )

def build_niriss_soss_calculation(teff, logg, mh, ksmag):
    """构建 JWST/NIRISS/SOSS 的 Pandeia calculation。"""
    calculation = build_default_calc("jwst", "niriss", "soss")

    calculation["configuration"]["instrument"]["aperture"] = "soss"
    calculation["configuration"]["instrument"]["mode"] = "soss"
    calculation["configuration"]["instrument"]["disperser"] = "gr700xd"
    calculation["configuration"]["instrument"]["filter"] = "clear"

    calculation["configuration"]["detector"]["ngroup"] = 10
    calculation["configuration"]["detector"]["nint"] = 1
    calculation["configuration"]["detector"]["nexp"] = 1
    calculation["configuration"]["detector"]["subarray"] = "substrip256"
    calculation["configuration"]["detector"]["readout_pattern"] = "nisrapid"

    calculation["background"] = "minzodi"
    calculation["background_level"] = "low"

    scene = calculation["scene"][0]

    scene["position"] = {
        "x_offset": 0.0,
        "y_offset": 0.0,
        "orientation": 0.0,
        "position_parameters": ["x_offset", "y_offset", "orientation"],
    }

    scene["shape"] = {"geometry": "point"}

    scene["spectrum"] = {
        "name": "Phoenix Spectrum",
        "spectrum_parameters": ["sed", "normalization"],
    }

    scene["spectrum"]["sed"] = {
        "sed_type": "phoenix",
        "t_eff": teff,
        "log_g": logg,
        # Local Phoenix +0.5 metallicity grid contains invalid spectra; cap positive [M/H] at solar.
        "metallicity": min(float(mh), 0.0),
    }

    scene["spectrum"]["normalization"] = {
        "type": "photsys",
        "bandpass": "2mass,ks",
        "norm_flux": ksmag,
        "norm_fluxunit": "vegamag",
    }

    scene["spectrum"]["lines"] = []

    scene["spectrum"]["extinction"] = {
        "bandpass": "j",
        "law": "mw_rv_31",
        "unit": "mag",
        "value": 0,
    }

    calculation["scene"][0] = scene

    return calculation


def is_saturated(report):
    """检查 Pandeia report 中是否存在部分或完全饱和像素。"""
    warnings = report.get("warnings", {})

    if "full_saturated" in warnings:
        return True, warnings["full_saturated"]

    if "partial_saturated" in warnings:
        return True, warnings["partial_saturated"]

    return False, 0


def run_calculation(base_calculation, ngroup, nint=1):
    """
    运行一次 Pandeia calculation。

    注意：
    这里使用 deepcopy，避免函数调用直接修改原始 calculation。
    这样后面合并代码或批量运行时更安全。
    """
    calc = copy.deepcopy(base_calculation)

    calc["configuration"]["detector"]["ngroup"] = int(ngroup)
    calc["configuration"]["detector"]["nint"] = int(nint)

    return perform_calculation(calc, webapp=False, dict_report=True)


def check_saturation_cached(base_calculation, ngroup, cache, verbose=False, label=""):
    """
    检查某个 ngroup 是否饱和，并使用 cache 避免重复跑 Pandeia。
    """
    ngroup = int(ngroup)

    if ngroup in cache:
        sat, n_sat = cache[ngroup]
        if verbose:
            status = "过曝" if sat else "不过曝"
            print(f"[缓存{label}] n = {ngroup}, 结果 = {status}")
        return sat, n_sat

    report = run_calculation(base_calculation, ngroup, nint=1)
    sat, n_sat = is_saturated(report)

    cache[ngroup] = (sat, n_sat)

    if verbose:
        if sat:
            print(f"[检查{label}] n = {ngroup}, 结果 = 过曝, 像素数 = {n_sat}")
        else:
            print(f"[检查{label}] n = {ngroup}, 结果 = 不过曝")

    return sat, n_sat


def binary_refine_max_unsaturated(
    base_calculation,
    low_unsat,
    high_sat,
    cache,
    verbose=True
):
    """
    已知：
    - low_unsat 不过曝
    - high_sat 过曝

    在二者之间二分，寻找最大不过曝 ngroup。
    """
    low_unsat = int(low_unsat)
    high_sat = int(high_sat)

    left = low_unsat + 1
    right = high_sat - 1
    best_unsaturated = low_unsat
    loop_count = 0

    if verbose:
        print(
            f"[进入二分] 已知不过曝 = {low_unsat}, 已知过曝 = {high_sat}, "
            f"搜索区间 = [{left}, {right}]"
        )

    while left <= right:
        loop_count += 1
        mid = (left + right) // 2

        sat, n_sat = check_saturation_cached(
            base_calculation=base_calculation,
            ngroup=mid,
            cache=cache,
            verbose=verbose,
            label=f" 二分 {loop_count}"
        )

        if sat:
            right = mid - 1
        else:
            best_unsaturated = mid
            left = mid + 1

    if verbose:
        print(f"[完成] 最大不过曝 ngroup = {best_unsaturated}")

    return best_unsaturated


def find_max_unsaturated_ngroup_predictive(
    base_calculation,
    ksmag,
    initial_window=8,
    hard_max_ngroup=1024,
    verbose=True
):
    """
    使用经验公式 + 自适应 bracket + 二分法寻找最大不过曝 ngroup。

    步骤：
    1. 根据 Kmag 预测 n_pred；
    2. 先检查 n_pred ± initial_window；
    3. 如果小窗口已经夹住边界，则直接二分；
    4. 如果小窗口没有夹住，则向外扩展；
    5. 最终在“不过曝 / 过曝”边界内二分精修。

    返回：
    - 最大不过曝 ngroup，即 n_f。
    """

    hard_max_ngroup = int(hard_max_ngroup)
    initial_window = int(initial_window)

    if hard_max_ngroup < 1:
        raise ValueError("hard_max_ngroup 必须 >= 1。")

    if initial_window < 1:
        raise ValueError("initial_window 必须 >= 1。")

    cache = {}

    # ---------------------------------------------------------------------------------
    # 0. 先检查 ngroup = 1
    # ---------------------------------------------------------------------------------
    sat_1, n_sat_1 = check_saturation_cached(
        base_calculation=base_calculation,
        ngroup=1,
        cache=cache,
        verbose=verbose,
        label=" 最小值"
    )

    if sat_1:
        raise RuntimeError(
            "ngroup = 1 时已经过曝，无法在当前 NIRISS/SOSS 配置下找到不过曝设置。"
        )

    # ---------------------------------------------------------------------------------
    # 1. 根据 Kmag 预测初值
    # ---------------------------------------------------------------------------------
    n_pred = predict_ngroup_from_kmag(ksmag)
    n_pred = min(max(1, n_pred), hard_max_ngroup)

    if verbose:
        print(f"[经验预测] Kmag = {ksmag:.3f}, n_pred = {n_pred}")

    # ---------------------------------------------------------------------------------
    # 2. 先检查预测值附近的小窗口
    # ---------------------------------------------------------------------------------
    low = max(1, n_pred - initial_window)
    high = min(hard_max_ngroup, n_pred + initial_window)

    sat_low, n_sat_low = check_saturation_cached(
        base_calculation=base_calculation,
        ngroup=low,
        cache=cache,
        verbose=verbose,
        label=" 窗口下界"
    )

    sat_high, n_sat_high = check_saturation_cached(
        base_calculation=base_calculation,
        ngroup=high,
        cache=cache,
        verbose=verbose,
        label=" 窗口上界"
    )

    # 情况 A：窗口已经成功夹住边界
    if (not sat_low) and sat_high:
        return binary_refine_max_unsaturated(
            base_calculation=base_calculation,
            low_unsat=low,
            high_sat=high,
            cache=cache,
            verbose=verbose
        )

    # ---------------------------------------------------------------------------------
    # 3. 如果窗口下界都过曝，说明预测值偏大，需要向下找不过曝点
    # ---------------------------------------------------------------------------------
    if sat_low:
        if verbose:
            print("[扩展方向] 窗口下界已经过曝，说明预测值偏大，向下寻找不过曝点。")

        high_sat = low
        low_candidate = max(1, low // 2)

        while low_candidate > 1:
            sat_candidate, n_sat_candidate = check_saturation_cached(
                base_calculation=base_calculation,
                ngroup=low_candidate,
                cache=cache,
                verbose=verbose,
                label=" 向下扩展"
            )

            if not sat_candidate:
                return binary_refine_max_unsaturated(
                    base_calculation=base_calculation,
                    low_unsat=low_candidate,
                    high_sat=high_sat,
                    cache=cache,
                    verbose=verbose
                )

            high_sat = low_candidate
            low_candidate = max(1, low_candidate // 2)

        # 前面已经确认 n=1 不过曝，所以如果到这里，直接在 1 和 high_sat 之间二分
        return binary_refine_max_unsaturated(
            base_calculation=base_calculation,
            low_unsat=1,
            high_sat=high_sat,
            cache=cache,
            verbose=verbose
        )

    # ---------------------------------------------------------------------------------
    # 4. 如果窗口上界仍不过曝，说明预测值偏小，需要向上找过曝点
    # ---------------------------------------------------------------------------------
    if not sat_high:
        if verbose:
            print("[扩展方向] 窗口上界仍不过曝，说明预测值偏小，向上寻找过曝点。")

        low_unsat = high
        high_candidate = min(hard_max_ngroup, max(high + 1, high * 2))

        while high_candidate <= hard_max_ngroup:
            sat_candidate, n_sat_candidate = check_saturation_cached(
                base_calculation=base_calculation,
                ngroup=high_candidate,
                cache=cache,
                verbose=verbose,
                label=" 向上扩展"
            )

            if sat_candidate:
                return binary_refine_max_unsaturated(
                    base_calculation=base_calculation,
                    low_unsat=low_unsat,
                    high_sat=high_candidate,
                    cache=cache,
                    verbose=verbose
                )

            low_unsat = high_candidate

            if high_candidate == hard_max_ngroup:
                break

            high_candidate = min(hard_max_ngroup, high_candidate * 2)

        raise RuntimeError(
            f"直到 ngroup = {hard_max_ngroup} 仍未过曝。"
            f"请提高 HARD_MAX_NGROUP，或检查 Pandeia saturation warning 是否正常返回。"
        )

    # 理论上不会到这里；如果到了，说明单调性或判断逻辑异常
    raise RuntimeError(
        "未能建立有效的不过曝/过曝 bracket。"
        "请检查 saturation 判断是否满足随 ngroup 单调变化。"
    )

def get_tgroup(report):
    """从 Pandeia report 中读取单个 group 的时间，单位为秒。"""
    try:
        return float(report["information"]["exposure_specification"]["tgroup"])
    except KeyError as exc:
        raise KeyError("无法从 report 中读取 tgroup，请检查 Pandeia report 结构。") from exc


def integrate_trapezoid(y, x):
    """兼容不同 numpy 版本的梯形积分函数。"""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)

    return np.trapz(y, x)


def compute_snr(report):
    """根据 Pandeia report 计算积分 SNR 和 sigma_frac。"""
    try:
        sn = np.array(report["1d"]["sn"])
    except KeyError as exc:
        raise KeyError("无法从 report['1d']['sn'] 中读取 SNR 数据，请检查 Pandeia report 结构。") from exc

    if sn.ndim != 2 or sn.shape[0] < 2:
        raise ValueError(f"report['1d']['sn'] 结构异常，实际 shape = {sn.shape}")

    # 按原代码逻辑：
    # 第 0 行看作 wavelength
    # 第 1 行看作 SNR
    wave = np.array(sn[0, :], dtype=float)
    snr = np.array(sn[1, :], dtype=float)

    mask = np.isfinite(wave) & np.isfinite(snr)
    wave = wave[mask]
    snr = snr[mask]

    if len(wave) < 2:
        raise ValueError("有效 wavelength 点数不足，无法积分计算 SNR。")

    snr_int = integrate_trapezoid(snr**2, wave)

    if not np.isfinite(snr_int) or snr_int <= 0:
        raise ValueError(f"SNR 积分结果异常：snr_int = {snr_int}")

    total_snr = np.sqrt(snr_int)
    sigma_frac = 1e6 / total_snr

    return float(total_snr), float(sigma_frac)


def print_result(result):
    """统一打印最终结果。"""
    print("\n=== Noise Calculation Result ===")
    print(f"target                      = {result['target']}")
    print(f"最大不饱和 n_f              = {result['n_f']}")
    print(f"最终确定 ngroup             = {result['ngroups']}")
    print(f"最终确定 nint               = {result['nint']}")
    print(f"单个 group 时间 t_group     = {result['t_group_s']:.6f} s")
    print(f"单次曝光时间 t_exp          = {result['t_exp_s']:.2f} s")
    print(f"总曝光时间 t_total          = {result['t_total_s']:.2f} s")
    print(f"总观测时间 t_bond           = {result['t_bond_h']:.2f} h")
    print(f"Integrated SNR              = {result['total_snr']:.6g} (SNR·μm)")
    print(f"sigma_frac                  = {result['sigma_frac_ppm']:.6g} ppm")
    print("=" * 50)


# =========================================================================================================
# 单目标与批量缓存流程
# =========================================================================================================
def calculate_noise_from_params(params, target_name, row_index=None, print_params=False, verbose=VERBOSE):
    """对一个目标执行原 calculate_noise 计算逻辑，并返回可写入缓存表的一行结果。"""
    teff = get_float(params, "Teff")
    logg = get_float(params, "logg")
    mh = get_float(params, "M_H")
    ksmag = get_float(params, "kmag")

    period = get_float(params, "P")
    a_s = get_float(params, "a")
    r_star = get_float(params, "R_*")
    r_jup = get_float(params, "R_p")
    t14 = get_float(params, "trandur")  # 单位：小时
    t14s = t14 * 3600                   # 单位：秒

    r_s = ((r_jup * u.R_jup) / (r_star * u.R_sun)).to(u.dimensionless_unscaled).value

    # 仅用于打印展示，不参与 Pandeia 噪声计算
    b = 0.480
    i = get_float(params, "i")

    if print_params:
        print_target_params(
            teff=teff,
            logg=logg,
            mh=mh,
            ksmag=ksmag,
            period=period,
            a_s=a_s,
            r_star=r_star,
            r_jup=r_jup,
            r_s=r_s,
            t14=t14,
            b=b,
            i=i,
        )

    # -----------------------------------------------------------------------------------------------------
    # 构建 Pandeia calculation
    # -----------------------------------------------------------------------------------------------------
    calculation = build_niriss_soss_calculation(
        teff=teff,
        logg=logg,
        mh=mh,
        ksmag=ksmag,
    )

    # -----------------------------------------------------------------------------------------------------
    # 寻找最大不过曝 ngroup
    # -----------------------------------------------------------------------------------------------------
    n_f = find_max_unsaturated_ngroup_predictive_walk(
        base_calculation=calculation,
        ksmag=ksmag,
        hard_max_ngroup=HARD_MAX_NGROUP,
        verbose=verbose,
    )

    # -----------------------------------------------------------------------------------------------------
    # 最终 ngroup, nint
    # -----------------------------------------------------------------------------------------------------
    # 保持你的原逻辑：最终 ngroup 取 ceil(n_f / 2)
    ngroups = ceil(n_f / 2)

    # 用最终 ngroups 重新跑一次，读取最终配置对应的 tgroup
    report_ngroups = run_calculation(calculation, ngroups, nint=1)

    t_group = get_tgroup(report_ngroups) * u.s
    t_exp = (ngroups * t_group).value  # 单次 integration 曝光时间，单位：秒

    if t_exp <= 0:
        raise ValueError(f"计算得到的 t_exp <= 0，t_exp = {t_exp}")

    # 覆盖整个凌日所需的积分次数
    nint = ceil(t14s / t_exp)

    if nint <= 0:
        raise ValueError(f"计算得到的 nint <= 0，nint = {nint}")

    # 最终正式计算
    final_report = run_calculation(calculation, ngroups, nint=nint)

    total_snr, sigma_frac = compute_snr(final_report)

    # 总曝光时间：nint 次积分，每次积分时间为 t_exp
    t_total = nint * t_exp  # 秒

    # 保留你的原 overhead / bonding 时间写法
    t_bond = ((t_total / 3600) + 1) * 1.2

    result = {
        "row_index": row_index,
        "target": target_name,
        "status": "success",
        "error_message": "",
        "teff_K": float(teff),
        "logg_cgs": float(logg),
        "metallicity_m_h": float(mh),
        "kmag": float(ksmag),
        "period_days": float(period),
        "a_over_rstar": float(a_s),
        "r_star_rsun": float(r_star),
        "r_planet_rjup": float(r_jup),
        "rp_over_rstar": float(r_s),
        "transit_duration_h": float(t14),
        "inclination_deg": float(i),
        "n_f": int(n_f),
        "ngroups": int(ngroups),
        "nint": int(nint),
        "t_group_s": float(t_group.value),
        "t_exp_s": float(t_exp),
        "t_total_s": float(t_total),
        "t_bond_h": float(t_bond),
        "total_snr": float(total_snr),
        "sigma_frac_ppm": float(sigma_frac),
        # Convert ppm to normalized flux units for direct Gaussian light-curve injection.
        "sigma_flux": float(sigma_frac) * 1e-6,
    }

    return result


def calculate_noise_for_target(target_name=TARGET, csv_path=TARGET_TABLE_PATH, print_params=True, verbose=VERBOSE):
    """按目标名读取输入表，并计算该目标的噪声参数。"""
    params = load_target_params(csv_path, target_name)
    result = calculate_noise_from_params(
        params=params,
        target_name=target_name,
        row_index=None,
        print_params=print_params,
        verbose=verbose,
    )
    print_result(result)
    return result


def _load_existing_cache(output_path):
    if not output_path.exists():
        return {}

    df = pd.read_csv(output_path)
    records = {}
    for _, row in df.iterrows():
        key = (int(row["row_index"]), str(row["target"]))
        records[key] = row.to_dict()
    return records


def _write_noise_cache(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records.values())
    if "row_index" in df.columns:
        df = df.sort_values("row_index")
    df.to_csv(output_path, index=False)


def calculate_noise_for_all_targets(
    csv_path=TARGET_TABLE_PATH,
    output_path=NOISE_CACHE_OUTPUT_PATH,
    resume=RESUME_EXISTING_CACHE,
    max_targets=BATCH_MAX_TARGETS,
    verbose=VERBOSE,
):
    """
    遍历 combined tier A+B 输入表，计算每个目标的噪声参数并缓存到一个 CSV。

    每个目标完成后都会重写缓存表；单个目标失败时记录 failed 和错误信息，然后继续下一个目标。
    """
    df = pd.read_csv(csv_path)
    validate_input_table_columns(df)

    records = _load_existing_cache(output_path) if resume else {}
    n_total = len(df) if max_targets is None else min(len(df), int(max_targets))

    for row_index, row in df.iloc[:n_total].iterrows():
        target_name = str(row["pl_name"])
        key = (int(row_index), target_name)

        if resume and key in records and records[key].get("status") == "success":
            print(f"[skip] row_index={row_index}, target={target_name} 已有成功缓存")
            continue

        print(f"[run] row_index={row_index}, target={target_name}")
        try:
            params = row_to_params(row)
            result = calculate_noise_from_params(
                params=params,
                target_name=target_name,
                row_index=int(row_index),
                print_params=False,
                verbose=verbose,
            )
            print_result(result)
        except Exception as exc:  # noqa: BLE001
            result = {
                "row_index": int(row_index),
                "target": target_name,
                "status": "failed",
                "error_message": repr(exc),
            }
            print(f"[failed] row_index={row_index}, target={target_name}: {exc}")

        records[key] = result
        _write_noise_cache(records, output_path)
        print(f"[cache] saved: {output_path}")

    return pd.DataFrame(records.values())


def main():
    if RUN_ALL_TARGETS:
        results = calculate_noise_for_all_targets()
        print(f"\n=== Batch Noise Cache Finished ===")
        print(f"n_cached = {len(results)}")
        print(f"output   = {NOISE_CACHE_OUTPUT_PATH}")
        return results

    return calculate_noise_for_target(TARGET)


if __name__ == "__main__":
    main()
