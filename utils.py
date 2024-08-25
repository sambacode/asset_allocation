import logging
from typing import Any, Callable, Literal, Optional, Union, overload

import numpy as np
import pandas as pd
import statsmodels.api as sm
from bwbbgdl import GoGet
from bwlogger import StyleAdapter
from bwutils import TODAY, Date
from scipy.optimize import minimize

logger = StyleAdapter(logging.getLogger(__name__))


def cap_long_only_weights(
    w: pd.Series,
    cap: Optional[float] = None,
) -> pd.Series:
    filt = w >= cap
    w.loc[filt] = cap
    w.loc[~filt] = w.loc[~filt] * (1 - w.loc[filt].sum()) / w.loc[~filt].sum()
    if (w > cap).any():
        return cap_long_only_weights(w, cap=cap)
    else:
        return w


def calculate_weights(
    method: Literal["IV", "ERC", "HRC"],
    cov_matrix: pd.DataFrame,
    risk_contribution: Optional[pd.Series] = None,
    **kwargs,
) -> pd.DataFrame:
    if method == "IV":
        vols = pd.Series(index=cov_matrix.index, data=np.sqrt(np.diag(cov_matrix)))
        weights = (1 / vols) / (1 / vols).sum()
    elif method == "ERC":
        weights = optmize_risk_budget(cov_matrix, risk_contribution, **kwargs)
    elif method == "HRC":
        raise NotImplementedError(f"method {method} not implemented yet")
    else:
        raise ValueError(f"method {method} not supported")

    return weights


@overload
def correlation_to_distance(corr: float) -> float: ...


@overload
def correlation_to_distance(corr: pd.DataFrame) -> pd.DataFrame: ...


def correlation_to_distance(corr):
    return np.sqrt(((1 - corr) / 2))


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


def get_rebalance_dates(
    date_index: pd.DatetimeIndex,
    return_type: Literal["start", " end"] = "start",
    frequency: Literal["D", "W", "M", "Q", "Y"] = "M",
    n_periods: Optional[int] = 1,
) -> list[pd.Timestamp]:
    if return_type not in ["start", "end"]:
        raise ValueError(f"Invalid return_type: {return_type}, must be 'end' or'start'")
    grouper = date_index.to_series().groupby(date_index.to_period(frequency))
    return_func = {"end": grouper.max, "start": grouper.min}
    return return_func[return_type]()[::n_periods].tolist()


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


def _total_weight_constraint(w: pd.Series) -> float:
    return w.sum() - 1.0


def _long_only_constraint(w: pd.Series) -> float:
    return w


def optmize_risk_budget(
    cov: pd.DataFrame,
    rc_t: Optional[pd.Series] = None,
    w0: Optional[pd.Series] = None,
    cons: Optional[list[dict[str, Callable[pd.Series, pd.Series]]]] = [],
    long_only: bool = True,
    **kwargs,
) -> pd.Series:
    n = len(cov.index)
    if w0 is None:
        w0 = pd.Series([1] * n, index=cov.index) / n
    if rc_t is None:
        rc_t = pd.Series([1] * n, index=cov.index) / n
    cons_sum_1 = [{"type": "eq", "fun": _total_weight_constraint}]
    cons_sum_2 = [{"type": "ineq", "fun": _long_only_constraint}] if long_only else []
    cons = tuple(cons + cons_sum_1 + cons_sum_2)

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


def get_available_trackers(df: pd.DataFrame, min_data_points: int = 100) -> pd.Index:
    s_data_points = (~df.isna()).sum()
    filt = s_data_points >= min_data_points
    return s_data_points[filt].index


def load_trackers(
    mapper_ticker: dict[str, str],
    dt_ini: Date = "1990-12-31",
    dt_end: Date = TODAY,
) -> pd.DataFrame:
    inverse_mapper_ticker = {v: k for k, v in mapper_ticker.items()}
    tickers = list(mapper_ticker.values())

    g = GoGet(enforce_strict_matching=True)
    tracker_df: pd.DataFrame = g.fetch(
        tickers=tickers,
        fields="PX_LAST",
        dt_ini=dt_ini,
        dt_end=dt_end,
    )

    tracker_df = tracker_df.pivot_table(index="date", columns="id")
    tracker_df.columns = tracker_df.columns.droplevel(0)
    tracker_df = tracker_df.rename(columns=inverse_mapper_ticker)
    return tracker_df


def _signal_to_rank(signal: pd.Series) -> pd.Series:
    assert not signal.isna().any(), "NaN value in signal"
    rank = signal.rank()
    weight = rank - rank.sum() / rank.count()
    scale = 2 / weight.abs().sum()
    return weight * scale


def _weights_tsmom(returns: pd.Series, vol: pd.Series, **_) -> pd.Series:
    return np.sign(returns) * 1 / vol / (1 / vol).sum()


def _weights_xsmom(returns: pd.Series, **_) -> pd.Series:
    return _signal_to_rank(returns)


def _weights_value_ppp(ppp: pd.Series, **_) -> pd.Series:
    return _signal_to_rank(ppp)


def _weights_value_alpha(alpha: pd.Series, **_) -> pd.Series:
    return _signal_to_rank(alpha)


WEGIHTS: dict[str, Callable] = {
    "tsmom": _weights_tsmom,
    "value_alpha": _weights_value_alpha,
    "value_ppp": _weights_value_ppp,
    "xsmom": _weights_xsmom,
}


@overload
def calculate_factor_weight(
    factor: Literal["tsmom"],
    *,
    returns: pd.Series,
    vol: pd.Series,
    **_,
) -> pd.Series: ...


@overload
def calculate_factor_weight(
    factor: Literal["value_alpha"],
    *,
    alpha: pd.Series,
    **_,
) -> pd.Series: ...


@overload
def calculate_factor_weight(
    factor: Literal["value_ppp"],
    *,
    ppp: pd.Series,
    **_,
) -> pd.Series: ...


@overload
def calculate_factor_weight(
    factor: Literal["xsmom"],
    *,
    returns: pd.Series,
    **_,
) -> pd.Series: ...


def calculate_factor_weight(
    factor: Literal["tsmom", "value_alpha", "value_ppp", "xsmom"], **kwargs
) -> pd.Series:
    if (operator := WEGIHTS.get(factor)) is None:
        raise ValueError(
            f"Unknown data: '{factor}'. " f"Must be one of: {', '.join(WEGIHTS)}."
        )
    return operator(**kwargs)


def inv_vol(vols: pd.Series) -> pd.Series:
    return (1 / vols) / (1 / vols).sum()


def equal_weight(vols: pd.Series, **_) -> pd.Series:
    return (vols * 0 + 1) / vols.count()


def calc_weight(method: Literal["iv", "ew"], vols: pd.Series) -> pd.Series:
    if method == "iv":
        return inv_vol(vols)
    elif method == "ew":
        return equal_weight(vols)
    else:
        raise NotImplementedError("weight method not implemented")


def cov_to_vols(df_vols: pd.DataFrame) -> pd.Series:
    return pd.Series(index=df_vols.index, data=np.sqrt(np.diag(df_vols)))


def calc_covariance(
    df: pd.DataFrame, method: Literal["rolling", "expanding", "ewm"], **kwargs
):
    DEFAULT_PARAM = {"rolling": {"window": 252}, "ewm": {"halflife": 63}}
    params = kwargs | DEFAULT_PARAM.get(method, {})
    return df.__getattr__(method)(**params).cov().loc[df.index[-1]]


class Backtest:
    min_data_points = 252 * 3
    r_wind = 21

    def __init__(
        self, r_wind: Optional[int] = None, min_data_points: Optional[int] = None
    ):
        self.r_wind = r_wind or self.r_wind
        self.min_data_points = r_wind or self.min_data_points

    def run(
        self,
        tracker_df: pd.DataFrame,
        weight_method: Literal["iv", "ew"],
        cov_method: Literal["rolling", "expanding", "ewm"],
        vol_target: float = 0.1,
        cov_params: dict[str, Any] = {},
        details: Optional[bool] = True,
    ) -> Union[pd.DataFrame, pd.Series]:

        df_log_return = np.log(tracker_df).diff(self.r_wind).dropna(how="all")
        t_0, t_1 = df_log_return.iloc[self.min_data_points :].index[:2]
        backtest = pd.Series(index=df_log_return[t_0:].index, dtype="float64")
        pos_open = pd.DataFrame(
            index=df_log_return[t_1:].index,
            columns=df_log_return.columns,
            dtype="float64",
        )
        pos_close = pd.DataFrame(
            index=df_log_return[t_0:].index,
            columns=df_log_return.columns,
            dtype="float64",
        )
        pnl = pd.Series(index=df_log_return[t_1:].index, dtype="float64")
        weights = pd.DataFrame(
            index=df_log_return[t_0:].index,
            columns=df_log_return.columns,
            dtype="float64",
        )

        avaialbe_trackers = get_available_trackers(
            df_log_return.iloc[: self.min_data_points],
            self.min_data_points,
        )
        cov = (
            calc_covariance(df_log_return[avaialbe_trackers], cov_method, **cov_params)
            * 252
            / self.r_wind
        )
        vols = cov_to_vols(cov)

        w_ = calc_weight(weight_method, vols).copy()
        adj_factor = vol_target / np.sqrt(w_ @ cov @ w_).copy()
        weights.loc[t_0] = adj_factor * w_.copy()

        backtest[t_0] = 100.0
        pos_close.loc[t_0] = backtest[t_0] * weights.loc[t_0].copy()
        # pos_open.loc[t_1] = pos_close.loc[t_0]

        for t, tm1 in zip(backtest.index[1:], backtest.index[:-1]):
            pos_open.loc[t] = pos_close.loc[tm1].copy()
            pos_close.loc[t] = (
                (tracker_df.loc[t] / tracker_df.loc[tm1])
            ) * pos_open.loc[t].copy()
            pnl[t] = (pos_close.loc[t] - pos_open.loc[t]).sum()
            backtest[t] = backtest[tm1] + pnl[t]

            # overnight
            if t.month != tm1.month:  # Rebalance on 1st BD
                if tracker_df.loc[:t].shape[0] > 252:
                    avaialbe_trackers = get_available_trackers(
                        df_log_return.loc[:tm1],
                        self.min_data_points,
                    )
                    cov = (
                        calc_covariance(
                            df_log_return.loc[:tm1, avaialbe_trackers],
                            cov_method,
                            **cov_params,
                        )
                        * 252
                        / self.r_wind
                    )
                    vols = cov_to_vols(cov)

                w_ = calc_weight(weight_method, vols).copy()
                adj_factor = vol_target / np.sqrt(w_ @ cov @ w_).copy()
                weights.loc[t] = adj_factor * w_.copy()
                pos_close.loc[t] = (
                    backtest[t] * weights.loc[t].copy()
                )  # rebalance at close of 1st BD

        if details:
            return pd.concat(
                [
                    tracker_df.rename(columns=lambda col: col + "_tracker"),
                    df_log_return.rename(columns=lambda col: col + "_log_return"),
                    pos_close.rename(columns=lambda col: col + "_pos_close"),
                    pos_open.rename(columns=lambda col: col + "_pos_open"),
                    weights.rename(columns=lambda col: col + "_weights"),
                    pnl.to_frame("pnl"),
                    backtest.to_frame("backtest_tracker"),
                ],
                axis=1,
                sort=True,
            )
        return backtest
