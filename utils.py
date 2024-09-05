import inspect
import logging
from typing import Any, Callable, Literal, Optional, Union, overload

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

from bwbbgdl import GoGet
from bwlogger import StyleAdapter
from bwutils import TODAY, Date

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
    signal = signal.dropna()
    assert not signal.isna().any(), "NaN value in signal"
    rank = signal.rank()
    weight = rank - rank.sum() / rank.count()
    scale = 2 / weight.abs().sum()
    return weight * scale


def _weights_tsmom(returns: pd.Series, vols: pd.Series, **_) -> pd.Series:
    return np.sign(returns) * 1 / vols / (1 / vols).sum()


def _weights_xsmom(returns: pd.Series, **_) -> pd.Series:
    return _signal_to_rank(returns)


def _weights_value_ppp(ppp: pd.Series, **_) -> pd.Series:
    return _signal_to_rank(ppp)


def _weights_value_paired(alpha: pd.Series, **_) -> pd.Series:
    return _signal_to_rank(alpha)


WEGIHTS: dict[str, Callable] = {
    "tsmom": _weights_tsmom,
    "value_paired": _weights_value_paired,
    "value_ppp": _weights_value_ppp,
    "xsmom": _weights_xsmom,
}


@overload
def calculate_factor_weight(
    factor: Literal["tsmom"],
    *,
    returns: pd.Series,
    vols: pd.Series,
    **_,
) -> pd.Series: ...


@overload
def calculate_factor_weight(
    factor: Literal["value_paired"],
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
    factor: Literal["tsmom", "value_paired", "value_ppp", "xsmom"], **kwargs
) -> pd.Series:
    if (operator := WEGIHTS.get(factor)) is None:
        raise ValueError(
            f"Unknown data: '{factor}'. " f"Must be one of: {', '.join(WEGIHTS)}."
        )
    return operator(**kwargs).dropna()


def inv_vol(vols: pd.Series) -> pd.Series:
    return (1 / vols) / (1 / vols).sum()


def equal_weight(vols: pd.Series, **_) -> pd.Series:
    return (vols * 0 + 1) / vols.count()


def calculate_alphas_fx_cds_pairs(
    endog: Literal["fx", "cds"], daily_log_returns: pd.DataFrame
) -> pd.Series:
    code = pd.Series([col[:3] for col in daily_log_returns.columns])
    code_unique = code[code.duplicated(keep="first")].to_list()
    pairs = [(f"{code}_fx", f"{code}_cds") for code in code_unique]
    returns = daily_log_returns.rolling(252).sum().iloc[-1]
    cov = calc_covariance(daily_log_returns, "rolling", window=252)
    vols = cov_to_vols(cov)
    betas = pd.Series(
        {(fx, cds): cov.loc[fx, cds] / (vols[cds] ** 2) for fx, cds in pairs}
    )
    alphas = pd.Series(
        {
            idx1: returns[idx1] - returns[idx2] * beta
            for ((idx1, idx2), beta) in betas.iteritems()
        }
    )
    if endog == "fx":
        return alphas
    else:
        return -1 * alphas.rename(index=lambda idx: idx.replace("fx", "cds"))


def calc_weight(
    method: Literal["bn", "iv", "ew", "tsmom", "xsmom", "value_ppp", "value_paired"],
    vols: pd.Series,
    log_returns: Optional[pd.DataFrame] = None,
    n_months: Optional[int] = None,
    endog: Optional[Literal["fx", "cds"]] = None,
    long_short: dict[Literal["long", "short"], str] = None,
) -> pd.Series:
    if method == "iv":
        return inv_vol(vols)
    elif method == "ew":
        return equal_weight(vols)
    elif method == "tsmom":
        returns = log_returns.iloc[-21 * n_months :].sum()
        return calculate_factor_weight(method, vols=vols, returns=returns)
    elif method == "xsmom":
        returns = log_returns.iloc[-21 * n_months :].sum()
        return calculate_factor_weight(method, returns=returns)
    # elif method == "value_ppp":
    #     ppp =
    #     return calculate_factor_weight(method, vols=vols, ppp=ppp)
    elif method == "value_paired":
        alpha = calculate_alphas_fx_cds_pairs(endog, log_returns)
        return calculate_factor_weight(method, alpha=alpha)
    elif method == "bn":
        long = log_returns[long_short["long"]].iloc[-21 * n_months :].rolling(21).sum()
        short = log_returns[long_short["long"]].iloc[-21 * n_months :].rolling(21).sum()
        return pd.Series(
            {
                long.name: 1,
                short.name: -long.cov(short) / short.var(),
            }
        )

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


DataType = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]


class Backtest:
    min_data_points = 252 * 3
    r_wind = 21
    trackers: pd.DataFrame

    def __init__(
        self,
        r_wind: Optional[int] = None,
        min_data_points: Optional[int] = None,
        trackers: Optional[pd.DataFrame] = None,
    ):
        self.r_wind = r_wind or self.r_wind
        self.min_data_points = min_data_points or self.min_data_points
        self.trackers = trackers

    def _prepare_data(self, trackers: pd.DataFrame) -> DataType:
        trackers = trackers.fillna(method="ffill").copy()
        log_returns = np.log(trackers).diff(self.r_wind).dropna(how="all")
        t_0, t_1 = log_returns.iloc[self.min_data_points :].index[:2]
        backtest = pd.Series(index=log_returns[t_0:].index, dtype="float64")
        pos_open = pd.DataFrame(
            index=log_returns[t_1:].index,
            columns=log_returns.columns,
            dtype="float64",
        )
        pos_close = pd.DataFrame(
            index=log_returns[t_0:].index,
            columns=log_returns.columns,
            dtype="float64",
        )
        pnl = pd.Series(index=log_returns[t_1:].index, dtype="float64")
        weights = pd.DataFrame(
            index=log_returns[t_0:].index,
            columns=log_returns.columns,
            dtype="float64",
        )
        return (
            trackers,
            log_returns,
            backtest,
            pos_open,
            pos_close,
            pnl,
            weights,
        )

    @staticmethod
    def concatenate_output(*database) -> pd.DataFrame:
        assert map(lambda data: isinstance(data, (pd.Series,)), database)
        caller_locals = inspect.currentframe().f_back.f_locals
        variable_names = {id(v): k for k, v in caller_locals.items()}
        dataframes = [
            (
                data.rename(columns=lambda col: f"{col}_{variable_names[id(data)]}")
                if isinstance(data, pd.DataFrame)
                else data.to_frame(variable_names[id(data)])
            )
            for data in database
        ]
        return pd.concat(dataframes, axis=1, sort=True)

    def run(
        self,
        trackers: pd.DataFrame,
        weight_method: Literal[
            "bn", "iv", "ew", "tsmom", "xsmom", "value_ppp", "value_paired"
        ],
        cov_method: Literal["rolling", "expanding", "ewm"],
        vol_target: float = 0.1,
        cov_params: dict[str, Any] = {},
        factor_params: dict[str, Any] = {},
        details: Optional[bool] = True,
    ) -> Union[pd.DataFrame, pd.Series]:
        endog = factor_params.get("endog")

        # Prepare Data
        trackers, log_returns, backtest, pos_open, pos_close, pnl, weights = (
            self._prepare_data(trackers)
        )
        t_0 = backtest.index[0]

        # First Setup
        avaialbe_trackers = get_available_trackers(
            log_returns.iloc[: self.min_data_points],
            self.min_data_points,
        )
        cov = (
            calc_covariance(
                (
                    log_returns[avaialbe_trackers].filter(like=endog, axis=1)
                    if endog
                    else log_returns[avaialbe_trackers]
                ),
                cov_method,
                **cov_params,
            )
            * 252
            / self.r_wind
        )
        vols = cov_to_vols(cov)
        w_ = calc_weight(
            weight_method,
            vols,
            log_returns=np.log(trackers[avaialbe_trackers])
            .diff(1)
            .iloc[: self.min_data_points],
            **factor_params,
        ).copy()
        adj_factor = vol_target / np.sqrt(w_ @ cov.loc[w_.index, w_.index] @ w_).copy()
        weights.loc[t_0] = adj_factor * w_.copy()
        # FIXME: not possible to use today's vol to balance the portfolio

        backtest[t_0] = 100.0
        pos_close.loc[t_0] = backtest[t_0] * weights.loc[t_0].copy()

        # Simulate Setup
        for t, tm1 in zip(backtest.index[1:], backtest.index[:-1]):
            pos_open.loc[t] = pos_close.loc[tm1].copy()
            pos_close.loc[t] = ((trackers.loc[t] / trackers.loc[tm1])) * pos_open.loc[
                t
            ].copy()
            pnl[t] = (pos_close.loc[t] - pos_open.loc[t]).sum()
            backtest[t] = backtest[tm1] + pnl[t]

            if t.month != tm1.month:
                avaialbe_trackers = get_available_trackers(
                    log_returns.loc[:tm1],
                    self.min_data_points,
                )
                cov = (
                    calc_covariance(
                        (
                            log_returns.loc[:tm1, avaialbe_trackers].filter(
                                like=endog, axis=1
                            )
                            if endog
                            else log_returns.loc[:tm1, avaialbe_trackers]
                        ),
                        cov_method,
                        **cov_params,
                    )
                    * 252
                    / self.r_wind
                )
                vols = cov_to_vols(cov)
                w_ = calc_weight(
                    weight_method,
                    vols,
                    log_returns=np.log(trackers[avaialbe_trackers]).diff(1).loc[:tm1],
                    **factor_params,
                ).copy()
                adj_factor = (
                    vol_target / np.sqrt(w_ @ cov.loc[w_.index, w_.index] @ w_).copy()
                )
                weights.loc[t] = adj_factor * w_.copy()
                # Rebalance at close of 1st BD
                pos_close.loc[t] = backtest[t] * weights.loc[t].copy()

        if details:
            return Backtest.concatenate_output(
                trackers, log_returns, pos_close, pos_open, weights, pnl, backtest
            )
        return backtest
