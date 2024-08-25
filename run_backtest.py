import logging

from entities import FX_TRACKER_DICT
from utils import Backtest, load_trackers


logging.getLogger(__name__)

df_fx = load_trackers(FX_TRACKER_DICT)
bt = Backtest()

s_backtest = bt.run(
    tracker_df=df_fx.dropna(axis=1),
    weight_method="iv",
    cov_method="expanding",
    vol_target=0.1,
)
s_backtest.plot()
