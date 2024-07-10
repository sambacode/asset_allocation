import argparse
import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from utils import calculate_weights, get_available_trackers, load_trackers

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


def backtest(
    tracker_df: pd.DataFrame,
    vol_target: float = 0.1,
    dt_ini: Date = "1990-12-31",
    dt_end: Date = TODAY,
    method_weights: Literal[
        "ERC",
        "HRC",
        "IV",
    ] = "IV",
    rebal_wind: tuple[
        Literal["D", "W", "M", "Q", "Y"], Union[int, Literal["start", "end"]]
    ] = ("M", "start"),
    tol_by_asset: Optional[float] = None,
    tol_agg: Optional[float] = None,
    return_period: tuple[Literal["D", "W", "M", "Q", "Y"], int] = ("D", 21),
    return_rolling: bool = True,
    cov_window: Literal["expanding", "rolling"] = "expanding",
    cov_estimate_wegihts: Optional[tuple[Literal["halflife", "alpha"], float]] = None,
) -> pd.DataFrame:
    lists_msgs = (
        [
            f"Weights Method: {method_weights}",
            f"Rebalacing Window: {rebal_wind[1]}_{rebal_wind[0]}",
        ]
        + ([f"Rebalacing Tolerance Asset: {tol_by_asset:.%}"] if tol_by_asset else [])
        + ([f"Rebalacing Tolerance Agg.: {tol_agg:.%}"] if tol_agg else [])
        + [
            f"Return Period: {return_period[1]}_{return_period[0]}",
            f"Return Rolling: {str(return_rolling).upper()}",
            f"Covariance Window: {str(cov_window).capitalize()}",
        ]
        + (
            [f"Covariance Weights Paramters: {str(cov_window).capitalize()}"]
            if cov_estimate_wegihts
            else []
        )
    )
    logger.info("Backtest Parameters: %s." % " | ".join(lists_msgs))

    r_days: int = return_period[1]
    MIN_DATA_POINTS = 252

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    basic_setup(APPNAME, False, SCRIPT_DIR, NAMESPACE)
    backtest(
        load_trackers(FX_TRACKER_DICT),
        method_weights="IV",
    )
