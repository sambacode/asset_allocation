import argparse
import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib.pyplot import twinx
from matplotlib.ticker import FuncFormatter

from bwlogger import StyleAdapter, basic_setup

logger = StyleAdapter(logging.getLogger(__name__))

APPNAME = ""
NAMESPACE = ""

###############################################################################
SCRIPT_DIR = Path(__file__).parent
DEFAULT_FILE_PATH = SCRIPT_DIR.joinpath("BR CDS and FX.xlsx")
ENDOG_COL = "BRL"
EXOG_COLS = ["Brazil CDS"]
N_MIN = 252

###############################################################################


def load_data(file_path: Optional[Path] = None) -> tuple[pd.Series, pd.Series]:
    file_path = file_path or DEFAULT_FILE_PATH
    sheet_name = "BRL"
    fx = pd.read_excel(file_path, index_col=0, sheet_name=sheet_name).iloc[:, 0]
    fx.index = pd.to_datetime(fx.index)
    fx.name = sheet_name

    sheet_name = "Brazil CDS"
    cds = pd.read_excel(file_path, index_col=0, sheet_name=sheet_name).iloc[:, 0]
    cds.index = pd.to_datetime(cds.index)
    cds.name = sheet_name
    return fx, cds


def _filter_by_period(
    df: pd.DataFrame, period: str, drop_last_period: bool = True
) -> pd.DataFrame:
    df = df.reindex(df.index.to_series().groupby(df.index.to_period(period)).max())
    n = -1 if drop_last_period else len(df.index)
    return df.iloc[:n]


def _calculate_parameters(y: pd.Series, x: pd.Series) -> pd.Series:
    model = sm.OLS(y, sm.add_constant(x)).fit()
    s_params = model.params.copy()
    s_params.index = [
        "alpha" if idx == "const" else f"beta_{idx}" for idx in s_params.index
    ]
    return s_params


def _percentage_formatter(x, _):
    return f"{x * 100:.1f}%"


def calculate_returns(
    prices_series: Union[list[Union[pd.Series, pd.DataFrame]], pd.DataFrame],
    type: Literal["log", " simple"] = "log",
    period: Optional[Literal["D", " W", " M", " Q", " Y"]] = None,
    custom_period: Optional[list[pd.Timestamp]] = None,
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
        df_return = np.log(df_period / df_period.shift(1)).copy()
    else:
        df_return = (df_period / df_period.shift(1) - 1).copy()
    return df_return.dropna().copy()


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


###############################################################################
def trading_strategy(
    df_params: pd.DataFrame, df_return_ln: pd.DataFrame
) -> pd.DataFrame:
    COL_TIMEFRAME_NBR = "timeframe_nbr"
    COL_EXPECTED_RETURN_ACC = "expected_return_acc"
    COL_REALIZED_RETURN_ACC = "realized_return_acc"
    COL_OUTPERFORMANCE_RETURN_ACC = "outperformance_return_acc"
    COL_OUTPERFORMANCE_RETURN_LN_ACC = "outperformance_return_ln_acc"
    COL_OUTPERFORMANCE_RETURN_LN = "outperformance_return_ln"
    COL_SIGNAL = "signal"
    # COL_ALPHA = "alpha"

    def _get_reblance_trades() -> pd.DatetimeIndex:
        N_DAYS = 63
        return df_trading_period.index[::N_DAYS].copy()

    df_aux = pd.concat([df_return_ln, df_params], axis=1)
    rebalance_dates = _get_reblance_trades()
    df_aux[COL_TIMEFRAME_NBR] = (
        df_aux.index.to_series().isin(rebalance_dates).shift(1).cumsum().fillna(0)
    )
    df_trading_period = df_aux.dropna().copy()

    cols_betas = [f"beta_{col}" for col in EXOG_COLS]
    list_df_outperformance = []
    first_signal = 1
    for timeframe, sub_df in df_trading_period.groupby(COL_TIMEFRAME_NBR):
        if timeframe == 0:
            signal = first_signal  # FIXME
            s_est_betas_previous = sub_df[cols_betas].iloc[-1]
            est_alpha_previous = sub_df["alpha"].iloc[-1]
            continue

        # print(
        #     f"Timeframe: #{timeframe:02.0f}
        # |  Signal: {'On' if signal else 'Off': <3}  |  "
        # )
        df_ln_returns_acc = sub_df[EXOG_COLS].cumsum().copy()
        df_returns_acc = np.exp(df_ln_returns_acc) - 1
        df_beta_returns_acc = pd.DataFrame(
            df_returns_acc.sort_index(axis=1).values
            * s_est_betas_previous.sort_index().values,
            index=df_returns_acc.index,
            columns=df_returns_acc.columns.sort_values(),
        )
        s_expected_returns_acc = df_beta_returns_acc.sum(
            axis=1
        )  # TODO: how to incorporate alpha in expected returns?
        s_expected_returns_acc.name = COL_EXPECTED_RETURN_ACC
        s_realized_returns_acc = np.exp(sub_df[ENDOG_COL].cumsum()) - 1
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
        signal = 1 if s_outperformance_acc.iloc[-1] > 0 else -1
        # if timeframe == 1:
        #     break

    df_outperformance = pd.concat(list_df_outperformance)
    df_base = pd.concat([df_aux, df_outperformance], axis=1)
    df_base.to_excel("testing.xlsx")


def main() -> None:
    logger.info("Loading data...")
    prices_series = load_data()
    logger.info("Data loaded!")
    df_returns = calculate_returns(prices_series, type="log", period="D")
    df_param = generate_parameters_series(df_returns, ENDOG_COL, EXOG_COLS, start=252)
    trading_strategy(df_param, df_returns)
    breakpoint()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-path",
        required=False,
        help="Path to the file with prices' data.",
    )
    args = parser.parse_args()

    basic_setup(APPNAME, False, Path().home(), NAMESPACE)
    main()
