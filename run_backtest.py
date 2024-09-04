import logging
from pathlib import Path
from typing import Literal, Optional

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
trackers = pd.concat([fx, cds], axis=1, join="outer").fillna(method="ffill", limit=5)


COV_METHOD = "expanding"
VOL_TARGET = 0.1
RETURN_WINDOW = 21
MIN_DATA_POINTS = 252 * 3
bt = Backtest(RETURN_WINDOW, MIN_DATA_POINTS)


def filter_class(trackers: pd.DataFrame, fx_cds: Optional[Literal["cds", "fx"]] = None):
    if fx_cds:
        return trackers.loc[:, trackers.columns.str.contains(f"_{fx_cds}")].dropna(
            how="all"
        )
    else:
        return trackers


def tsmom_cds_12m():
    CLASS = "cds"
    WEIGHT_METHOD = "tsmom"
    N_MONTHS = 12

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"n_months": N_MONTHS},
    )


def tsmom_cds_6m():
    CLASS = "cds"
    WEIGHT_METHOD = "tsmom"
    N_MONTHS = 6

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"n_months": N_MONTHS},
    )


def tsmom_cds_3m():
    CLASS = "cds"
    WEIGHT_METHOD = "tsmom"
    N_MONTHS = 3

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"n_months": N_MONTHS},
    )


def tsmom_fx_12m():
    CLASS = "fx"
    WEIGHT_METHOD = "tsmom"
    N_MONTHS = 12

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"n_months": N_MONTHS},
    )


def tsmom_fx_6m():
    CLASS = "fx"
    WEIGHT_METHOD = "tsmom"
    N_MONTHS = 6

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"n_months": N_MONTHS},
    )


def tsmom_fx_3m():
    CLASS = "fx"
    WEIGHT_METHOD = "tsmom"
    N_MONTHS = 3

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"n_months": N_MONTHS},
    )


def tsmom_12m():
    CLASS = None
    WEIGHT_METHOD = "tsmom"
    N_MONTHS = 12

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"n_months": N_MONTHS},
    )


def tsmom_6m():
    CLASS = None
    WEIGHT_METHOD = "tsmom"
    N_MONTHS = 12

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"n_months": N_MONTHS},
    )


def tsmom_3m():
    CLASS = None
    WEIGHT_METHOD = "tsmom"
    N_MONTHS = 3

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"n_months": N_MONTHS},
    )


def value_fx_ppp():
    None


def value_fx_paired():
    CLASS = "fx"
    WEIGHT_METHOD = "value_paired"

    return bt.run(
        trackers=trackers.loc["2008-08-07":],
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"endog": CLASS},
    )


def value_cds_paired():
    CLASS = "cds"
    WEIGHT_METHOD = "value_paired"

    return bt.run(
        trackers=trackers.loc["2008-08-07":],
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"endog": CLASS},
    )


def xsmom_fx():
    CLASS = "fx"
    WEIGHT_METHOD = "xsmom"
    N_MONTHS = 12

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"n_months": N_MONTHS},
    )


def xsmom_cds():
    CLASS = "cds"
    WEIGHT_METHOD = "xsmom"
    N_MONTHS = 12

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"n_months": N_MONTHS},
    )


def xsmom():
    CLASS = None
    WEIGHT_METHOD = "xsmom"
    N_MONTHS = 12

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
        factor_params={"n_months": N_MONTHS},
    )


def long_cds_iv():
    CLASS = "cds"
    WEIGHT_METHOD = "iv"

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
    )


def long_fx_iv():
    CLASS = "fx"
    WEIGHT_METHOD = "iv"

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
    )


def long_cds_ew():
    CLASS = "cds"
    WEIGHT_METHOD = "ew"

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
    )


def long_fx_ew():
    CLASS = "fx"
    WEIGHT_METHOD = "ew"

    return bt.run(
        trackers=filter_class(trackers, CLASS),
        weight_method=WEIGHT_METHOD,
        cov_method=COV_METHOD,
        vol_target=VOL_TARGET,
        details=True,
    )


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


def port_iv_neutro_long_basket_ew_cds_short_basket_ew_fx():
    None


def port_iv_neutro_long_basket_ew_fx_short_basket_ew_cds():
    None


def port_beta_neutro_long_basket_iv_cds_short_basket_iv_fx():
    None


def port_beta_neutro_long_basket_iv_fx_short_basket_iv_cds():
    None


DICT_BACKTESTS = {
    # "TSMOM-CDS-12": tsmom_cds_12m,
    # "TSMOM-CDS-6": tsmom_cds_6m,
    # "TSMOM-CDS-3": tsmom_cds_3m,
    # "TSMOM-FX-12": tsmom_fx_12m,
    # "TSMOM-FX-6": tsmom_fx_6m,
    # "TSMOM-FX-3": tsmom_fx_3m,
    # "TSMOM-12": tsmom_12m,
    # "TSMOM-6": tsmom_6m,
    # "TSMOM-3": tsmom_3m,
    "VALUE-FX-PPP": value_fx_ppp,
    # "VALUE-FX-PAIRED": value_fx_paired,
    # "VALUE-CDS-PAIRED": value_cds_paired,
    # "XSMOM-FX": xsmom_fx,
    # "XSMOM-CDS": xsmom_cds,
    # "XSMOM": xsmom,
    # "L-CDS-IV": long_cds_iv,
    # "L-FX-IV": long_fx_iv,
    # "L-CDS-EW": long_cds_ew,
    # "L-FX-EW": long_fx_ew,
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

if __name__ == "__main__":
    for alias, operator in DICT_BACKTESTS.items():
        data = operator()
        if data is not None:
            data.to_excel(OUTPUT_FOLDER.joinpath(f"{alias}.xlsx"))
