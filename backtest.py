import argparse
import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from utils import (
    calculate_weights,
    get_available_trackers,
    load_trackers,
    get_rebalance_dates,
)
from entities import FX_TRACKER_DICT

from bwlogger import StyleAdapter, basic_setup
from bwutils import TODAY, Date, open_file

logger = StyleAdapter(logging.getLogger(__name__))

APPNAME = ""
NAMESPACE = ""
OUTPUT_FOLDER = Path("C:/Users/pcampos/OneDrive - Insper - Instituto de Ensino e Pesquisa/Dissertação Mestrado/Analysis")



def backtest(
    tracker_df: pd.DataFrame,
    vol_target: float = 0.1,
    method_weights: Literal[
        "ERC",
        "HRC",
        "IV",
    ] = "IV",
    rebal_wind: tuple[
        Literal["start", "end"], Literal["D", "W", "M", "Q", "Y"], int
    ] = ("start", "M", 1),
    tol_by_asset: Optional[float] = None,
    tol_agg: Optional[float] = None,
    return_period: tuple[Literal["D", "W", "M", "Q", "Y"], int] = ("D", 21),
    return_rolling: bool = True,
    cov_window: Literal["expanding", "rolling"] = "expanding",
    cov_estimate_wegihts: Optional[tuple[Literal["halflife", "alpha"], float]] = None,
    clipboard: bool = True,
) -> pd.DataFrame:
    lists_msgs = (
        [
            f"Weights Method: {method_weights}",
            "Rebalacing Window: %s" % "_".join(map(str, rebal_wind)),
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

    dates_rebalance = get_rebalance_dates(tracker_df.index, *rebal_wind)

    r_days: int = return_period[1]
    MIN_DATA_POINTS = 252  # TODO: move into parameters

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
        .cov()  # TODO: change covariance method
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

        if t in dates_rebalance:
            if tracker_df.loc[:t].shape[0] > 252:
                avaialbe_trackers = get_available_trackers(
                    tracker_df.loc[:tm1], MIN_DATA_POINTS + r_days
                )
                cov = (
                    np.log(tracker_df.loc[:tm1])
                    .diff(r_days)[avaialbe_trackers]
                    .cov()  # TODO: change covariance method
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
    if clipboard:
        backtest.to_clipboard(excel=True)
    path_output = OUTPUT_FOLDER.joinpath("backtest.xlsx")
    backtest.to_excel(
        path_output,
        index_label="Date",
        sheet_name="Backtest",
    )
    open_file(path_output)
    return backtest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    basic_setup(APPNAME, False, OUTPUT_FOLDER, NAMESPACE)
    backtest(
        load_trackers(FX_TRACKER_DICT),
        method_weights="IV",
    )