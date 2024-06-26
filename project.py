import argparse
import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


logger = logging.getLogger(__name__)

DEFAULT_FILE_PATH = r"\\bwaws01fs\bw\BWGI\Monzú\Projeto Asset Allocation.xlsx"


def _load_base(file_path: Optional[Path] = None) -> pd.DataFrame:
    df_base = pd.read_excel(
        file_path or DEFAULT_FILE_PATH, sheet_name="base_values", index_col=0
    )
    return df_base.iloc[3:].copy()


def main(file_path: Optional[Path] = None) -> None:
    df_base = _load_base(file_path)
    
    
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
