import logging

from entities import FX_TRACKER_DICT
from utils import Backtest, load_trackers


logging.getLogger(__name__)

df_fx = load_trackers(FX_TRACKER_DICT)
bt = Backtest()

data = bt.run(
    trackers=df_fx,
    weight_method="iv",
    cov_method="expanding",
    vol_target=0.1,
    details=True
)
data.to_excel("backtest.xlsx")
