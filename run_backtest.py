import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from entities import EM_CDS_TRACKER_DICT, FX_TRACKER_DICT
from utils import Backtest, load_trackers

logging.getLogger(__name__)

OUTPUT_FOLDER = Path(
    "C:/Users/pcampos/OneDrive - Insper - Instituto de Ensino e Pesquisa/"
    "Dissertação Mestrado/Analysis/backtests"
)

fx = load_trackers(FX_TRACKER_DICT).rename(columns=lambda col: col + "_fx")
cds = load_trackers(EM_CDS_TRACKER_DICT).rename(columns=lambda col: col + "_cds")
trackers = pd.concat([fx, cds], axis=1, join="outer")


COV_METHOD = "expanding"
VOL_TARGET = 0.1
RETURN_WINDOW = 21
MIN_DATA_POINTS = 252 * 3
bt = Backtest(RETURN_WINDOW, MIN_DATA_POINTS)


def filter_class(trackers: pd.DataFrame, fx_cds: Literal["cds", "fx"]):
    return trackers.loc[:, trackers.columns.str.contains(f"_{fx_cds}")].dropna(
        how="all"
    )


def long_only(
    trackers: pd.DataFrame,
    weight_method: Literal["ew", "iv"],
    fx_cds: Literal["cds", "fx"],
):
    trackers
    return bt.run(
        trackers=trackers,
        weight_method=weight_method,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
    )


def long_only_fx(trackers: pd.DataFrame, weight_method: Literal["ew", "iv"]):
    trackers


def long_only_iv_cds(trackers: pd.DataFrame):
    WEIGHT_METHOD = "iv"

    bt = Backtest()
    data = bt.run(
        trackers=trackers,
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
    )


def tsmom_cds_12m():
    None


def tsmom_cds_6m():
    None


def tsmom_cds_3m():
    None


def tsmom_fx_12m():
    None


def tsmom_fx_6m():
    None


def tsmom_fx_3m():
    None


def tsmom_12m():
    None


def tsmom_6m():
    None


def tsmom_3m():
    None


def value_fx_ppp():
    None


def value_fx_paired():
    None


def value_cds_paired():
    None


def xsmom_fx():
    None


def xsmom_cds():
    None


def xsmom():
    None


def long_cds_iv():
    None


def long_fx_iv():
    None


def long_cds_ew():
    None


def long_fx_ew():
    None


def port_iv_long_short_cds_fx_iv():
    None


def port_iv_long_short_fx_cds_iv():
    None


def port_iv_long_short_cds_fx_beta_neutro():
    None


def port_iv_long_short_fx_cds_beta_neutro():
    None


def port_iv_neutro_long_basket_iv_fx_short_basket_iv_cds():
    None


def port_iv_neutro_long_basket_iv_fx_short_basket_iv_cds():
    None


def port_iv_neutro_long_basket_ew_cds_short_basket_ew_fx():
    None


def port_iv_neutro_long_basket_ew_fx_short_basket_ew_cds():
    None


def port_beta_neutro_long_basket_iv_cds_short_basket_iv_fx():
    None


def port_beta_neutro_long_basket_iv_fx_short_basket_iv_cds():
    None


DICT_BACKTESTS = {
    "TSMOM-CDS-12": tsmom_cds_12m,
    "TSMOM-CDS-6": tsmom_cds_6m,
    "TSMOM-CDS-3": tsmom_cds_3m,
    "TSMOM-FX-12": tsmom_fx_12m,
    "TSMOM-FX-6": tsmom_fx_6m,
    "TSMOM-FX-3": tsmom_fx_3m,
    "TSMOM-12": tsmom_12m,
    "TSMOM-6": tsmom_6m,
    "TSMOM-3": tsmom_3m,
    "VALUE-FX-PPP": value_fx_ppp,
    "VALUE-FX-PAIRED": value_fx_paired,
    "VALUE-CDS-PAIRED": value_cds_paired,
    "XSMOM-FX": xsmom_fx,
    "XSMOM-CDS": xsmom_cds,
    "XSMOM": xsmom,
    "L-CDS-IV": long_cds_iv,
    "L-FX-IV": long_fx_iv,
    "L-CDS-EW": long_cds_ew,
    "L-FX-EW": long_fx_ew,
    "LS-CDS-FX-IV": port_iv_long_short_cds_fx_iv,
    "LS-FX-CDS-IV": port_iv_long_short_fx_cds_iv,
    "LS-CDS-FX-BN-IV": port_iv_long_short_cds_fx_beta_neutro,
    "LS-FX-CDS-BN-IV": port_iv_long_short_fx_cds_beta_neutro,
    "L-FX-S-CDS-IV": port_iv_neutro_long_basket_iv_fx_short_basket_iv_cds,
    "L-CDS-S-FX-IV": port_iv_neutro_long_basket_iv_fx_short_basket_iv_cds,
    "L-FX-S-CDS-EW": port_iv_neutro_long_basket_ew_cds_short_basket_ew_fx,
    "L-CDS-S-FX-EW": port_iv_neutro_long_basket_ew_fx_short_basket_ew_cds,
    "L-CDS-S-FX-IV-BN": port_beta_neutro_long_basket_iv_cds_short_basket_iv_fx,
    "L-FX-S-CDS-IV-BN": port_beta_neutro_long_basket_iv_fx_short_basket_iv_cds,
}

for alias, operator in DICT_BACKTESTS:
    data = operator()
    if data:
        data.to_excel(OUTPUT_FOLDER.joinpath(alias))
