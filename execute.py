import argparse
import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from plot import plot_results
from utils import (
    _calculate_parameters,
    calculate_returns,
    calculate_weights,
    get_available_trackers,
    load_trackers,
)

from bwlogger import StyleAdapter, basic_setup
from bwutils import TODAY, Date

logger = StyleAdapter(logging.getLogger(__name__))

APPNAME = ""
NAMESPACE = ""

###############################################################################
SCRIPT_DIR = Path(__file__).parent
DEFAULT_FILE_PATH = SCRIPT_DIR.joinpath("BR CDS and FX.xlsx")
DEFAULT_OUTPUT_PATH = SCRIPT_DIR.joinpath("testing.xlsx")

COL_TIMEFRAME_NBR = "timeframe_nbr"
COL_EXPECTED_RETURN_ACC = "expected_return_acc"
COL_REALIZED_RETURN_ACC = "realized_return_acc"
COL_OUTPERFORMANCE_RETURN_ACC = "outperformance_return_acc"
COL_OUTPERFORMANCE_RETURN_LN_ACC = "outperformance_return_ln_acc"
COL_OUTPERFORMANCE_RETURN_LN = "outperformance_return_ln"
COL_RETURN_LN_ACC_STRATEGY = "return_ln_acc_strategy"
COL_SIGNAL = "signal"

ENDOG_COL = "BRL"
EXOG_COLS = ["Brazil CDS"]
N_MIN = 252

# these are in ER in USD
EM_CDS_TRACKER_DICT = {
    "BRL": "GSCDBRBE Index",
    "CNY": "GSCDCHBE Index",
    "MXN": "GSCDMEBE Index",
    "ZAR": "GSCDSOBE Index",
}

# these are in LOC ER with pnl converted to USD
IRS_TRACKER_DICT = {
    "BRL": "GSSWBRN5 Index",
    "CNY": "GSSWCNN5 Index",
    "MXN": "GSSWMXN5 Index",
    "ZAR": "GSSWZAN5 Index",
}

EQ_TRACKER_DICT = {
    "BRL": "BNPIFBR Index",  # in BRL
    "CNY": "BNPIFCNO Index",  # China onshore but with pnl converted to USD
    "ZAR": "BNPIFSA Index",  # in ZAR
    # "MXN": "???? Index",
}

FX_TRACKER_DICT = {
    "AED": "JPFCTAED Index",
    "ARS": "JPFCTARS Index",
    "BRL": "JPFCTBRL Index",
    "CLP": "JPFCTCLP Index",
    "CNY": "JPFCTCNY Index",
    "COP": "JPFCTCOP Index",
    "CZK": "JPFCTCZK Index",
    "HUF": "JPFCTHUF Index",
    "IDR": "JPFCTIDR Index",
    "INR": "JPFCTINR Index",
    "MXN": "JPFCTMXN Index",
    "MYR": "JPFCTMYR Index",
    "PEN": "JPFCTPEN Index",
    "PHP": "JPFCTPHP Index",
    "PLN": "JPFCTPLN Index",
    "RON": "JPFCTRON Index",
    "RUB": "JPFCTRUB Index",
    "SAR": "JPFCTSAR Index",
    "THB": "JPFCTTHB Index",
    "TRY": "JPFCTTRY Index",
    "ZAR": "JPFCTZAR Index",
}


def country_assets_portfolio(
    vol_target: float = 0.1,
    dt_ini: Date = "1990-12-31",
    dt_end: Date = TODAY,
    method_weights: Literal[
        "ERC",
        "HRC",
        "IV",
    ] = "IV",
    rebalancing_window: tuple[
        Literal["D", "W", "M", "Q", "Y"], Union[int, Literal["start", "end"]]
    ] = ("M", "start"),
    tol_by_asset: Optional[float] = None,
    tol_agg: Optional[float] = None,
    return_period: tuple[Literal["D", "W", "M", "Q", "Y"], int] = ("D", 21),
    return_rolling: bool = True,
    cov_window: Literal["expanding", "rolling"] = "expanding",
    cov_estimate_wegihts: Optional[tuple[Literal["halflife", "alpha"], float]] = None,
) -> pd.DataFrame:
    r_days: int = return_period[1]
    MIN_DATA_POINTS = 252
    tracker_df = load_trackers(FX_TRACKER_DICT)

    backtest = pd.Series(index=tracker_df.index[MIN_DATA_POINTS + r_days :])
    start_backtest = backtest.index.min()
    backtest.iloc[0] = 100.0

    avaialbe_trackers = get_available_trackers(
        tracker_df.iloc[: MIN_DATA_POINTS + r_days], MIN_DATA_POINTS + r_days
    )
    cov = (
        np.log(tracker_df)
        .diff(r_days)[avaialbe_trackers]
        .dropna()
        .iloc[:MIN_DATA_POINTS]
        .cov()
        * 252
        / r_days
    )
    w = calculate_weights(method=method_weights, cov_matrix=cov)
    adj_factor = vol_target / np.sqrt(w @ cov @ w)
    w = adj_factor * w

    weights_rebal = []
    q = backtest.iloc[0] * w / tracker_df.loc[start_backtest]
    s_rebal = q.copy()
    s_rebal.name = start_backtest
    weights_rebal.append(s_rebal)

    for t, tm1 in zip(backtest.index[1:], backtest.index[:-1]):
        pnl = ((tracker_df.loc[t] - tracker_df.loc[tm1]) * q).sum()
        backtest[t] = backtest[tm1] + pnl

        if t.month != tm1.month:
            if tracker_df.loc[:t].shape[0] > 252:
                avaialbe_trackers = get_available_trackers(
                    tracker_df.loc[:tm1], MIN_DATA_POINTS + r_days
                )
                cov = (
                    np.log(tracker_df.loc[:tm1]).diff(r_days)[avaialbe_trackers].cov()
                    * 252
                    / r_days
                )
            w = calculate_weights(method=method_weights, cov_matrix=cov)
            adj_factor = vol_target / np.sqrt(w @ cov @ w)
            w = adj_factor * w
            q = backtest[tm1] * w / tracker_df.loc[tm1]
            s_rebal = q.copy()
            s_rebal.name = t
            weights_rebal.append(s_rebal)

    df_weights = pd.concat(weights_rebal, axis=1).T.reindex(
        backtest.index, method="ffill"
    )
    df_weights.columns = df_weights.columns + "_weights"
    backtest = pd.concat(
        [
            tracker_df,
            df_weights,
            backtest.to_frame("assets"),
        ],
        axis=1,
        join="outer",
        sort=True,
    )

    return backtest


def _calculate_first_signal(
    df: pd.DataFrame, endog_col: str, exog_cols: list[str], cols_betas: list[str]
):
    # timeframe 0 only used to generate first positioning signal
    s_est_betas_0 = df[cols_betas].iloc[-1]
    s_ln_returns_0 = df[exog_cols].sum()
    s_returns_0 = np.exp(s_ln_returns_0) - 1
    expected_return_0 = (s_est_betas_0.values * s_returns_0.values).sum()
    realized_return_0 = np.exp(df[endog_col].sum()) - 1
    outperformance_0 = realized_return_0 - expected_return_0
    first_signal = -1 if outperformance_0 else 1
    logger.info("Singal 0 is %s" % "positive" if outperformance_0 else "negative")
    return first_signal


def _get_rebalance_trades(df: pd.DataFrame, steps: int) -> pd.DatetimeIndex:
    return df.index[::steps].copy()


def generate_parameters_series(
    df_return_ln: pd.DataFrame,
    endog_col: str,
    exog_cols: list[str],
    start: Optional[Union[int, pd.Timestamp]] = None,
) -> pd.DataFrame:
    start = start or N_MIN
    start = start if isinstance(start, int) else df_return_ln.index.get_loc(start) + 1
    end = len(df_return_ln.index)
    logger.info(
        "Estimating parameters for range: %s - %s"
        % (
            df_return_ln.index[start - 1].strftime("%b/%d/%y"),
            df_return_ln.index[end - 1].strftime("%b/%d/%y"),
        )
    )
    aux_params = {}
    for n in range(start, end + 1):
        sub_df = df_return_ln.iloc[:n].copy()
        param = _calculate_parameters(sub_df[endog_col], sub_df[exog_cols])
        ref_date = sub_df.index[-1]
        aux_params[ref_date] = param.to_dict()
    return pd.DataFrame(aux_params).T


def load_data(file_path: Optional[Path] = None) -> tuple[pd.Series, pd.Series]:
    file_path = file_path or DEFAULT_FILE_PATH
    logger.info(f"Loading data from file: {file_path}")
    sheet_name = "BRL"
    fx = pd.read_excel(file_path, index_col=0, sheet_name=sheet_name).iloc[:, 0]
    fx.index = pd.to_datetime(fx.index)
    fx.name = sheet_name

    sheet_name = "Brazil CDS"
    cds = pd.read_excel(file_path, index_col=0, sheet_name=sheet_name).iloc[:, 0]
    cds.index = pd.to_datetime(cds.index)
    cds.name = sheet_name
    return fx, cds


###############################################################################
def backtest_trading_strategy(
    df_params: pd.DataFrame,
    df_return_ln: pd.DataFrame,
    endog_col: str,
    exog_cols: list[str],
    steps: int,
    output_path: Optional[Path] = None,
    fmt: Literal["dataframe", "series"] = "dataframe",
) -> Union[pd.DataFrame, pd.Series]:
    # COL_ALPHA = "alpha"

    df_aux = pd.concat([df_return_ln, df_params], axis=1)
    rebalance_dates = _get_rebalance_trades(df_aux.dropna(), steps=steps)
    df_aux[COL_TIMEFRAME_NBR] = (
        df_aux.index.to_series().isin(rebalance_dates).shift(1).cumsum().fillna(0)
    )
    df_trading_period = df_aux.dropna().copy()

    cols_betas = [f"beta_{col}" for col in exog_cols]
    list_df_outperformance = []
    for timeframe, sub_df in df_trading_period.groupby(COL_TIMEFRAME_NBR):
        if timeframe == 0:
            signal = _calculate_first_signal(
                df_aux[df_aux["timeframe_nbr"] == 0].copy(),
                endog_col,
                exog_cols,
                cols_betas,
            )
            s_est_betas_previous = sub_df[cols_betas].iloc[-1]
            # est_alpha_previous = sub_df["alpha"].iloc[-1]
            continue

        df_ln_returns_acc = sub_df[exog_cols].cumsum().copy()
        df_returns_acc = np.exp(df_ln_returns_acc) - 1
        df_beta_returns_acc = pd.DataFrame(
            df_returns_acc.sort_index(axis=1).values
            * s_est_betas_previous.sort_index().values,
            index=df_returns_acc.index,
            columns=df_returns_acc.columns.sort_values(),
        )
        s_expected_returns_acc = df_beta_returns_acc.sum(
            axis=1
        )  # NOTE: should we incorporate alpha in expected returns?
        s_expected_returns_acc.name = COL_EXPECTED_RETURN_ACC
        s_realized_returns_acc = np.exp(sub_df[endog_col].cumsum()) - 1
        s_realized_returns_acc.name = COL_REALIZED_RETURN_ACC
        s_outperformance_acc = s_realized_returns_acc - s_expected_returns_acc
        s_outperformance_acc.name = COL_OUTPERFORMANCE_RETURN_ACC
        s_outperformance_ln_acc = np.log(1 + s_outperformance_acc)
        s_outperformance_ln_acc.name = COL_OUTPERFORMANCE_RETURN_LN_ACC
        s_outperformance_ln = s_outperformance_ln_acc - s_outperformance_ln_acc.shift(
            1
        ).fillna(0)
        s_outperformance_ln.name = COL_OUTPERFORMANCE_RETURN_LN

        s_est_betas_previous = sub_df[cols_betas].iloc[-1]
        # est_alpha_previous = sub_df[COL_ALPHA].iloc[-1]
        sub_df_outperformance = pd.concat(
            [
                s_expected_returns_acc,
                s_realized_returns_acc,
                s_outperformance_acc,
                s_outperformance_ln_acc,
                s_outperformance_ln,
            ],
            axis=1,
        )
        sub_df_outperformance[COL_SIGNAL] = signal
        list_df_outperformance.append(sub_df_outperformance)
        signal = -1 if s_outperformance_acc.iloc[-1] > 0 else 1

    df_outperformance = pd.concat(list_df_outperformance)
    s_return_ln_acc_strategy = (
        df_outperformance[COL_OUTPERFORMANCE_RETURN_LN] * df_outperformance[COL_SIGNAL]
    ).cumsum()
    s_return_ln_acc_strategy.name = COL_RETURN_LN_ACC_STRATEGY
    df_base = pd.concat([df_aux, df_outperformance, s_return_ln_acc_strategy], axis=1)

    output_path = output_path or DEFAULT_OUTPUT_PATH
    logger.info(f"Saving results to {output_path}")
    df_base.to_excel(output_path)
    return df_base if fmt == "series" else df_base


def main(
    endog_col: str,
    exog_cols: list[str],
    file_path: Optional[Path] = None,
    output_results: Optional[Path] = None,
    output_plots: Optional[Path] = None,
) -> None:
    logger.info("Loading data...")
    prices_series = load_data(file_path)
    logger.info("Data loaded!")
    df_returns = calculate_returns(prices_series, type="log", period="D")
    df_param = generate_parameters_series(df_returns, endog_col, exog_cols, start=252)
    df_strategy = backtest_trading_strategy(
        df_param, df_returns, endog_col, exog_cols, steps=63, output_path=output_results
    )
    plot_results(
        df_returns_ln_acc=df_strategy[[endog_col] + exog_cols].cumsum(),
        s_return_ln_acc_strategy=df_strategy[COL_RETURN_LN_ACC_STRATEGY],
        s_alpha=df_param["alpha"],
        s_beta=df_param[f"beta_{exog_cols[0]}"],
        output_path=output_plots,
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-path",
        required=False,
        help="Path to the file with prices' data.",
    )
    parser.add_argument(
        "--endog-col",
        default=ENDOG_COL,
        help="Path to the file with prices' data.",
    )
    parser.add_argument(
        "--exog-cols",
        nargs="+",
        default=EXOG_COLS,
        help="Path to the file with prices' data.",
    )
    parser.add_argument(
        "--output-results",
        required=False,
        help="Results file path.",
    )
    parser.add_argument(
        "--output-plot",
        required=False,
        help="Plot file path.",
    )
    args = parser.parse_args()

    basic_setup(APPNAME, False, SCRIPT_DIR, NAMESPACE)
    main(
        args.endog_col,
        args.exog_cols,
        args.file_path,
        args.output_results,
        args.output_plot,
    )

    country_assets_portfolio("BRL")
