import argparse
import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from bwaa.bw_arp_trader.fx.bw_aa_fx_signals import INVERSE_QUOTED, bbg_ticker
from bwbbgdl import GoGet
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

# these are in ER in USD
FX_TRACKER_DICT = {x: f"JPFCT{x} Index" for x in EM_CDS_TRACKER_DICT}

###############################################################################


def country_assets_portfolio(
    ccy: str,
    vol_target: float = 0.1,
    dt_ini: Date = "1990-12-31",
    dt_end: Date = TODAY,
) -> pd.DataFrame:
    assets_ticker_dict = {
        EM_CDS_TRACKER_DICT[ccy]: "5Y CDS",
        IRS_TRACKER_DICT[ccy]: "5Y IRS",
    }

    g = GoGet(enforce_strict_matching=True)

    tracker_df: pd.DataFrame = g.fetch(
        tickers=list(assets_ticker_dict),
        fields="PX_LAST",
        dt_ini=dt_ini,
        dt_end=dt_end,
    )

    tracker_df = tracker_df.pivot_table(index="date", columns="id")
    tracker_df.columns = tracker_df.columns.droplevel(0)
    tracker_df = tracker_df.rename(columns=assets_ticker_dict).dropna()


    backtest = pd.Series(index=tracker_df.index)
    backtest.iloc[0] = 100.0

    cov = np.log(tracker_df).diff(21).cov() * 12.0
    vols = pd.Series(index=cov.index, data=np.diag(cov))
    w = (1 / vols) / (1 / vols).sum()
    adj_factor = vol_target / np.sqrt(w @ cov @ w)
    w = adj_factor * w

    q = backtest.iloc[0] * w / tracker_df.iloc[0]

    for t, tm1 in zip(backtest.index[1:], backtest.index[:-1]):
        pnl = ((tracker_df.loc[t] - tracker_df.loc[tm1]) @ q).sum()
        backtest[t] = backtest[tm1] + pnl
        if t.month != tm1.month:
            if tracker_df.loc[:t].shape[0] > 252:
                cov = np.log(tracker_df.loc[:tm1]).diff(21).cov() * 12.0
                vols = pd.Series(index=cov.index, data=np.diag(cov))

            w = (1 / vols) / (1 / vols).sum()
            adj_factor = vol_target / np.sqrt(w @ cov @ w)
            w = adj_factor * w
            q = backtest[tm1] * w / tracker_df.loc[tm1]

    backtest = pd.concat(
        [tracker_df, backtest.to_frame("assets")],
        axis=1,
        join="outer",
        sort=True,
    )

    return backtest


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


def _calculate_parameters(y: pd.Series, x: pd.Series) -> pd.Series:
    model = sm.OLS(y, sm.add_constant(x)).fit()
    s_params = model.params.copy()
    s_params.index = [
        "alpha" if idx == "const" else f"beta_{idx}" for idx in s_params.index
    ]
    return s_params


def _format_percentage(x, _):
    return f"{x * 100:.1f}%"


def _filter_by_period(
    df: pd.DataFrame, period: str, drop_last_period: bool = True
) -> pd.DataFrame:
    df = df.reindex(df.index.to_series().groupby(df.index.to_period(period)).max())
    n = -1 if drop_last_period else len(df.index)
    return df.iloc[:n]


def _get_rebalance_trades(df: pd.DataFrame, steps: int) -> pd.DatetimeIndex:
    return df.index[::steps].copy()


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


def plot_results(
    df_returns_ln_acc: pd.DataFrame,
    s_return_ln_acc_strategy: pd.Series,
    s_alpha: pd.Series,
    s_beta: pd.Series,
    output_path: Optional[Path] = None,
) -> None:
    x_range = (df_returns_ln_acc.index.min(), df_returns_ln_acc.index.max())

    # plot return
    fig, axes = plt.subplots(figsize=(18, 12), nrows=2)

    axes[0].set_title("BRL x Brazil CDS")
    df_returns_ln_acc.plot(ax=axes[0], color=["blue", "orange"])
    s_return_ln_acc_strategy.plot(ax=axes[0], label="Trading Strategy", color="green")
    axes[0].legend()

    # Add gridlines to the first plot
    axes[0].grid(True, color="gray", linestyle="--")

    # Add a horizontal line at y=0 to the first plot
    axes[0].axhline(y=0, color="black")

    # Move the y-axis label and ticks to the right for the first plot
    axes[0].yaxis.set_label_position("right")
    axes[0].yaxis.tick_right()
    axes[0].set_ylabel("Cumulative Log Returns")

    # Apply the percentage formatter to the first plot
    axes[0].yaxis.set_major_formatter(FuncFormatter(_format_percentage))

    # Set x-axis limits to the first and last data points
    axes[0].set_xlim(x_range)

    # beta and alpha plot
    s_beta.plot(ax=axes[1], color="blue", label="Beta")
    ax2 = axes[1].twinx()
    s_alpha.plot(ax=ax2, color="orange", label="Alpha")

    # Add gridlines to the second plot
    axes[1].grid(True, color="gray", linestyle="--")

    # Set y-axis labels
    axes[1].set_ylabel("Beta")
    ax2.set_ylabel("Alpha")

    # Set x-axis limits to the first and last data points
    axes[1].set_xlim(x_range)

    y_min, y_max = axes[1].get_ylim()
    y_range = y_max - y_min
    y_mid = (y_max + y_min) / 2
    axes[1].set_ylim(y_mid - y_range / 2, y_mid + y_range / 2 * 1.2)

    y_min, y_max = ax2.get_ylim()
    y_range = y_max - y_min
    y_mid = (y_max + y_min) / 2
    ax2.set_ylim(y_mid - y_range / 2, y_mid + y_range / 2 * 1.2)

    # Combine legends from both axes
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=2)

    fig.tight_layout(
        rect=(0, 0, 1, 0.96)
    )  # Adjust the layout to make space for the title

    output_path = output_path or SCRIPT_DIR.joinpath("plot.svg")
    logger.info(f"Saving plot to {output_path}")
    fig.savefig(output_path)
    plt.close()


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
