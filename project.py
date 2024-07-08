import argparse
import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.cluster.hierarchy import linkage, dendrogram


logger = logging.getLogger(__name__)

DEFAULT_FILE_PATH = r"C:\Users\pcampos\OneDrive - Insper - Instituto de Ensino e Pesquisa\Asset Allocation\Projeto\Projeto Asset Allocation.xlsx"

# NOTE:

def _load_base(file_path: Optional[Path] = None) -> pd.DataFrame:
    return pd.read_excel(
        file_path or DEFAULT_FILE_PATH,
        sheet_name="base_values",
        index_col=0,
        skiprows=1,
    ).astype("float")


def _load_aux(file_path: Optional[Path] = None) -> pd.DataFrame:
    return pd.read_excel(
        file_path or DEFAULT_FILE_PATH,
        sheet_name="aux",
        index_col=0,
        usecols="A:D",
    )


def _calculate_returns(
    data: pd.DataFrame,
    type: Literal["simple", "log"],
    window: Optional[int] = 1,
    frequency: Optional[Union[str, pd.DateOffset]] = "B",
) -> pd.DataFrame:
    assert (
        frequency != "D"
    ), "D frequency is not supported due to weekends, please review"
    data_new = data.copy().asfreq(frequency, method="ffill").dropna(how="all")
    if type == "log":
        df_returns = np.log(data_new / data_new.shift(window))
    else:
        df_returns = data_new / data_new.shift(window) - 1
    return df_returns


def _calculate_volatilities(df_returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
    MIN_PERIODS = 60
    df_vols = df_returns.ewm(**kwargs, min_periods=MIN_PERIODS).vol()
    return df_vols


def _calculate_correlations(df_returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
    MIN_PERIODS = 100
    df_corr = df_returns.ewm(**kwargs, min_periods=MIN_PERIODS).corr()
    return df_corr


def main(
    file_path: Optional[Path] = None,
    window: Optional[int] = 1,
    frequency: Optional[Union[str, pd.DateOffset]] = "B",
    alpha: Optional[float] = 0.06,
) -> None:
    EXCLUDE_COLS = ["LT09TRUU Index", "B1MSBRUS Index"]
    RF_COLS = [
        "BZACCETP Index",
        "G0O1 Index",
    ]
    FX_COLS = [
        "USDEUR Curncy",
        "USDHKD Curncy",
        "USDINR Curncy",
        "USDCNY Curncy",
        "USDJPY Curncy",
        "USDBRL Curncy",
    ]
    df_base = _load_base(file_path)
    ASSET_COLS = list(set(df_base.columns) - set(RF_COLS + FX_COLS + EXCLUDE_COLS))
    df_aux = _load_aux(file_path)
    df_returns = _calculate_returns(
        df_base, type="simple", window=window, frequency=frequency
    )
    df_log_returns = _calculate_returns(
        df_base, type="log", window=window, frequency=frequency
    )
    df_vols = _calculate_volatilities(df_returns[ASSET_COLS], alpha=alpha)
    df_corr = _calculate_correlations(df_returns[ASSET_COLS], alpha=alpha)

    ##### ex-post analysis

    ###### ex-ante aanlysis

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-path",
        required=False,
        help="Path to the file with prices' data.",
    )
    args = parser.parse_args()

    main(
        file_path=args.file_path,
    )
