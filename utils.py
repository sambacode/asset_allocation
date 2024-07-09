import logging
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

from bwlogger import StyleAdapter

logger = StyleAdapter(logging.getLogger(__name__))


def calculate_weights(
    method: Literal["IV", "ERC", "HRC"],
    cov_matrix: pd.DataFrame,
    risk_contribution: Optional[pd.Series] = None,
    **kwargs,
) -> pd.DataFrame:
    if method == "IV":
        vols = pd.Series(index=cov_matrix.index, data=np.diag(cov_matrix))
        weights = (1 / vols) / (1 / vols).sum()
    elif method == "ERC":
        weights = optmize_risk_budget(cov_matrix, risk_contribution, **kwargs)
    elif method == "HRC":
        raise NotImplementedError(f"method {method} not implemented yet")
    else:
        raise ValueError(f"method {method} not supported")

    return weights


def _exponentially_decaying_weights(
    n: int, alpha: Optional[float] = None, halflife: Optional[int] = None
):
    """
    Exponentially decaying weights for the linear regression.

    """
    if alpha is None:
        if halflife:
            assert 0 > halflife, "Halflife must be positive."
            alpha = 1 - np.exp(-np.log(2) / halflife)
        else:
            raise ValueError("Either alpha or halflife must be specified.")
    assert 0 < alpha < 1, "Alpha must be between 0 and 1."
    return [(alpha) * ((1 - alpha) ** (n - i)) for i in range(0, n)]


def _calculate_parameters(
    y: pd.Series,
    x: pd.Series,
    weights_param: Optional[dict[Literal["alpha", "halflife"], float]] = None,
) -> pd.Series:
    weights = (
        [1] * len(x)
        if weights_param is None
        else _exponentially_decaying_weights(len(x), **weights_param)
    )
    model = sm.WLS(y, sm.add_constant(x), weights=weights).fit()
    s_params = model.params.copy()
    s_params.index = [
        "alpha" if idx == "const" else f"beta_{idx}" for idx in s_params.index
    ]
    return s_params


def _filter_by_period(
    df: pd.DataFrame, period: str, drop_last_period: bool = True
) -> pd.DataFrame:
    df = df.reindex(df.index.to_series().groupby(df.index.to_period(period)).max())
    n = -1 if drop_last_period else len(df.index)
    return df.iloc[:n]


def calculate_returns(
    prices_series: Union[list[Union[pd.Series, pd.DataFrame]], pd.DataFrame],
    type: Literal["log", " simple"] = "log",
    period: Optional[Literal["D", " W", " M", " Q", " Y"]] = None,
    custom_period: Optional[list[pd.Timestamp]] = None,
    timeframe: Optional[int] = 1,
) -> pd.DataFrame:
    if type not in ["log", "simple"]:
        raise ValueError("type must be 'log' or 'simple'")
    logger.info(
        "Calculating returns: type '%s' | period '%s'"
        % (type.upper(), period if not custom_period else "Custom"),
    )

    df_prices = (
        prices_series
        if isinstance(prices_series, pd.DataFrame)
        else pd.concat(prices_series, axis=1, join="outer")
    )
    df_prices = df_prices.sort_index().fillna(method="ffill")
    df_prices.index.name = None

    if custom_period:
        df_prices = df_prices.reindex(custom_period, method="ffill")
    df_period = _filter_by_period(df_prices, period or "D")

    if type == "log":
        df_return = np.log(df_period / df_period.shift(timeframe)).copy()
    else:
        df_return = (df_period / df_period.shift(timeframe) - 1).copy()
    return df_return.dropna().copy()


def calculate_portfolio_var(w: pd.Series, cov: pd.DataFrame) -> float:
    return w.T @ cov @ w


def calculate_risk_contribution(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    vol = calculate_portfolio_var(w, cov) ** 0.5
    mrc = cov @ w
    return mrc * w.T / vol


def risk_budget_objective(
    x: pd.Series, risk_pct: pd.Series, cov: pd.DataFrame
) -> float:
    vol = calculate_portfolio_var(x, cov) ** 0.5
    rc_t = vol * risk_pct

    rc = calculate_risk_contribution(x, cov)
    return np.square(rc - rc_t).sum() * 1e9


def total_weight_constraint(w: pd.Series) -> float:
    return w.sum() - 1.0


def optmize_risk_budget(
    cov: pd.DataFrame,
    rc_t: Optional[pd.Series] = None,
    w0: Optional[pd.Series] = None,
    cons: Optional[list[dict[str, Callable[pd.Series, pd.Series]]]] = [],
    **kwargs,
) -> pd.Series:
    n = len(cov.index)
    if w0 is None:
        w0 = pd.Series([1] * n, index=cov.index) / n
    if rc_t is None:
        rc_t = pd.Series([1] * n, index=cov.index) / n
    cons_sum_1 = {"type": "eq", "fun": total_weight_constraint}
    cons = tuple(cons + [cons_sum_1])

    res = minimize(
        risk_budget_objective,
        w0,
        args=(rc_t, cov),
        constraints=cons,
        **kwargs,
    )
    return pd.Series(res.x, index=cov.index, name="wegihts_target")


def check_weights_in_tolerance(
    weights: pd.Series,
    weights_target: pd.Series,
    tol_by_asset: Optional[float] = None,
    tol_agg: Optional[float] = None,
) -> bool:
    cond1, cond2 = (True, True)
    if tol_by_asset:
        cond1 = (weights - weights_target).abs() <= tol_by_asset
    if tol_agg:
        cond2 = (weights - weights_target).abs().sum() <= tol_agg
    return cond1 and cond2


def get_available_trackers(df: pd.DataFrame, min_data_points: int = 100) -> None:
    s_data_points = (~df.isna()).sum()
    filt = s_data_points >= min_data_points
    return s_data_points[filt].index
