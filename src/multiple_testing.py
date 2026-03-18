from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


def _two_sample_t_pvalue(y: np.ndarray, z: np.ndarray) -> float:
    treated = y[z == 1]
    control = y[z == 0]
    n1 = treated.shape[0]
    n0 = control.shape[0]
    s1 = float(np.var(treated, ddof=1))
    s0 = float(np.var(control, ddof=1))
    se = float(np.sqrt(s1 / n1 + s0 / n0))
    diff = float(np.mean(treated) - np.mean(control))
    if se == 0.0:
        return 1.0
    t_stat = diff / se
    df_num = (s1 / n1 + s0 / n0) ** 2
    df_den = ((s1 / n1) ** 2) / (n1 - 1) + ((s0 / n0) ** 2) / (n0 - 1)
    if df_den == 0.0:
        return 1.0
    df = df_num / df_den
    return float(2.0 * t.sf(np.abs(t_stat), df=df))


def simulate_null_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under the complete null for L simulations.
    Return columns: sim_id, hypothesis_id, p_value.
    """
    rng = np.random.default_rng(int(config["seed_null"]))
    n = int(config["N"])
    m = int(config["M"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])

    rows: list[dict[str, float | int]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            y = rng.normal(loc=0.0, scale=1.0, size=n)
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def simulate_mixed_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under mixed true and false null hypotheses for L simulations.
    Return columns: sim_id, hypothesis_id, p_value, is_true_null.
    """
    rng = np.random.default_rng(int(config["seed_mixed"]))
    n = int(config["N"])
    m = int(config["M"])
    m0 = int(config["M0"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])
    tau_alt = float(config["tau_alternative"])

    rows: list[dict[str, float | int | bool]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            is_true_null = hypothesis_id >= (m - m0)
            effect = 0.0 if is_true_null else tau_alt
            y = rng.normal(loc=0.0, scale=1.0, size=n) + effect * z
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                    "is_true_null": is_true_null,
                }
            )
    return pd.DataFrame(rows)


def bonferroni_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Bonferroni correction.
    """
    m = p_values.shape[0]
    return p_values <= (alpha / m)


def holm_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Holm step-down correction.
    """
    m = p_values.shape[0]
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    sorted_reject = np.zeros(m, dtype=bool)
    for k in range(m):
        threshold = alpha / (m - k)
        if sorted_p[k] <= threshold:
            sorted_reject[k] = True
        else:
            break

    reject = np.zeros(m, dtype=bool)
    reject[order] = sorted_reject
    return reject


def benjamini_hochberg_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Hochberg correction.
    """
    m = p_values.shape[0]
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * alpha
    passed = sorted_p <= thresholds

    sorted_reject = np.zeros(m, dtype=bool)
    if np.any(passed):
        k_max = int(np.max(np.where(passed)[0]))
        sorted_reject[: k_max + 1] = True

    reject = np.zeros(m, dtype=bool)
    reject[order] = sorted_reject
    return reject


def benjamini_yekutieli_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Yekutieli correction.
    """
    m = p_values.shape[0]
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    adjusted_alpha = alpha / harmonic

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * adjusted_alpha
    passed = sorted_p <= thresholds

    sorted_reject = np.zeros(m, dtype=bool)
    if np.any(passed):
        k_max = int(np.max(np.where(passed)[0]))
        sorted_reject[: k_max + 1] = True

    reject = np.zeros(m, dtype=bool)
    reject[order] = sorted_reject
    return reject


def compute_fwer(rejections_null: np.ndarray) -> float:
    """
    Return family-wise error rate from a [L, M] rejection matrix under the complete null.
    """
    return float(np.mean(np.any(rejections_null, axis=1)))


def compute_fdr(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return FDR for one simulation: false discoveries among all discoveries.
    Use 0.0 when there are no rejections.
    """
    n_rejections = int(np.sum(rejections))
    if n_rejections == 0:
        return 0.0
    false_discoveries = int(np.sum(rejections & is_true_null))
    return float(false_discoveries / n_rejections)


def compute_power(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return power for one simulation: true rejections among false null hypotheses.
    """
    is_false_null = ~is_true_null
    n_false_null = int(np.sum(is_false_null))
    if n_false_null == 0:
        return 0.0
    true_rejections = int(np.sum(rejections & is_false_null))
    return float(true_rejections / n_false_null)


def summarize_multiple_testing(
    null_pvalues: pd.DataFrame,
    mixed_pvalues: pd.DataFrame,
    alpha: float,
) -> dict[str, float]:
    """
    Return summary metrics:
      fwer_uncorrected, fwer_bonferroni, fwer_holm,
      fdr_uncorrected, fdr_bh, fdr_by,
      power_uncorrected, power_bh, power_by.
    """
    null_uncorrected_rows: list[np.ndarray] = []
    null_bonf_rows: list[np.ndarray] = []
    null_holm_rows: list[np.ndarray] = []

    for _, sim_df in null_pvalues.groupby("sim_id"):
        sim_sorted = sim_df.sort_values("hypothesis_id")
        p_values = sim_sorted["p_value"].to_numpy(dtype=float)
        null_uncorrected_rows.append(p_values <= alpha)
        null_bonf_rows.append(bonferroni_rejections(p_values, alpha))
        null_holm_rows.append(holm_rejections(p_values, alpha))

    fwer_uncorrected = compute_fwer(np.vstack(null_uncorrected_rows))
    fwer_bonferroni = compute_fwer(np.vstack(null_bonf_rows))
    fwer_holm = compute_fwer(np.vstack(null_holm_rows))

    fdr_uncorrected_values: list[float] = []
    fdr_bh_values: list[float] = []
    fdr_by_values: list[float] = []

    power_uncorrected_values: list[float] = []
    power_bh_values: list[float] = []
    power_by_values: list[float] = []

    for _, sim_df in mixed_pvalues.groupby("sim_id"):
        sim_sorted = sim_df.sort_values("hypothesis_id")
        p_values = sim_sorted["p_value"].to_numpy(dtype=float)
        is_true_null = sim_sorted["is_true_null"].to_numpy(dtype=bool)

        rejections_uncorrected = p_values <= alpha
        rejections_bh = benjamini_hochberg_rejections(p_values, alpha)
        rejections_by = benjamini_yekutieli_rejections(p_values, alpha)

        fdr_uncorrected_values.append(compute_fdr(rejections_uncorrected, is_true_null))
        fdr_bh_values.append(compute_fdr(rejections_bh, is_true_null))
        fdr_by_values.append(compute_fdr(rejections_by, is_true_null))

        power_uncorrected_values.append(compute_power(rejections_uncorrected, is_true_null))
        power_bh_values.append(compute_power(rejections_bh, is_true_null))
        power_by_values.append(compute_power(rejections_by, is_true_null))

    return {
        "fwer_uncorrected": float(np.mean(fwer_uncorrected)),
        "fwer_bonferroni": float(np.mean(fwer_bonferroni)),
        "fwer_holm": float(np.mean(fwer_holm)),
        "fdr_uncorrected": float(np.mean(fdr_uncorrected_values)),
        "fdr_bh": float(np.mean(fdr_bh_values)),
        "fdr_by": float(np.mean(fdr_by_values)),
        "power_uncorrected": float(np.mean(power_uncorrected_values)),
        "power_bh": float(np.mean(power_bh_values)),
        "power_by": float(np.mean(power_by_values)),
    }
