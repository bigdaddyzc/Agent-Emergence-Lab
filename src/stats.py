"""Small, dependency-light significance tests for the comparison report.

scipy is used when it is installed (Welch's t-test and Mann–Whitney U);
otherwise a pure-Python two-sided permutation test on the difference of means
is used, so the report works on the low-resource setup without extra deps.
"""

from __future__ import annotations

import itertools
import math
import random

try:  # optional dependency
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:  # noqa: BLE001 - any import problem means "not available"
    _scipy_stats = None


def has_scipy() -> bool:
    return _scipy_stats is not None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def permutation_test(a: list[float], b: list[float], *, n_resamples: int = 10000,
                     seed: int = 0) -> float:
    """Two-sided permutation p-value for the difference in means.

    Exact (all splits) when the number of splits is at most ``n_resamples``,
    otherwise Monte Carlo with a fixed seed so reports are reproducible.
    """
    a, b = [float(x) for x in a], [float(x) for x in b]
    if not a or not b:
        return float("nan")
    observed = abs(_mean(a) - _mean(b))
    pooled = a + b
    n_a = len(a)
    total = math.comb(len(pooled), n_a)
    eps = 1e-12
    if total <= n_resamples:
        hits = 0
        for idx in itertools.combinations(range(len(pooled)), n_a):
            chosen = set(idx)
            xa = [pooled[i] for i in chosen]
            xb = [pooled[i] for i in range(len(pooled)) if i not in chosen]
            if abs(_mean(xa) - _mean(xb)) >= observed - eps:
                hits += 1
        return hits / total
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_resamples):
        rng.shuffle(pooled)
        if abs(_mean(pooled[:n_a]) - _mean(pooled[n_a:])) >= observed - eps:
            hits += 1
    return (hits + 1) / (n_resamples + 1)


def welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t statistic and degrees of freedom (pure Python)."""
    na, nb = len(a), len(b)
    ma, mb = _mean(a), _mean(b)
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se2 = va / na + vb / nb
    if se2 == 0:
        return (0.0 if ma == mb else math.copysign(math.inf, ma - mb)), float(na + nb - 2)
    t = (ma - mb) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, df


def compare_samples(a: list[float], b: list[float]) -> dict:
    """Compare two samples; returns test name, p-value and effect size.

    Needs at least 2 values per group; otherwise ``p`` is None and ``test`` is
    ``"n/a"`` (a single run per mode cannot support any significance claim).
    """
    a, b = [float(x) for x in a], [float(x) for x in b]
    result = {"n_a": len(a), "n_b": len(b), "mean_diff": round(_mean(a) - _mean(b), 3),
              "test": "n/a", "p": None, "cliffs_delta": None}
    if len(a) < 2 or len(b) < 2:
        return result
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    result["cliffs_delta"] = round((greater - less) / (len(a) * len(b)), 3)
    if all(x == a[0] for x in a + b):
        result.update(test="identical", p=1.0)
        return result
    if _scipy_stats is not None:
        t = _scipy_stats.ttest_ind(a, b, equal_var=False)
        try:
            u = _scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            result["p_mannwhitney"] = round(float(u.pvalue), 4)
        except ValueError:
            pass
        p = float(t.pvalue)
        result.update(test="welch_t", p=round(p, 4) if not math.isnan(p) else None)
        return result
    result.update(test="permutation", p=round(permutation_test(a, b), 4))
    return result
