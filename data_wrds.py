"""Example script for fetching and merging WRDS data."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import wrds


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Query WRDS and save merged data.

    Parameters
    ----------
    argv:
        Optional command line argument list for testing.
    """

    parser = argparse.ArgumentParser(description="Fetch data from WRDS")
    parser.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("merged_data.xlsx"),
        help="Output Excel path",
    )
    args = parser.parse_args(argv)

    db = wrds.Connection()

    data = db.raw_sql(
        f"""
        SELECT date, permno, prc, vol
        FROM crsp.dsf
        WHERE date >= '{args.start}' AND date <= '{args.end}'
    """
    )

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(by=["permno", "date"])
    data["log_return"] = np.log(data["prc"] / data.groupby("permno")["prc"].shift(1))
    data["quarter"] = data["date"].dt.to_period("Q")
    quarterly_volatility = data.groupby(["permno", "quarter"])[
        "log_return"
    ].std() * np.sqrt(252)
    quarterly_volatility = quarterly_volatility.reset_index().rename(
        columns={"log_return": "realized_volatility"}
    )

    control_data = db.raw_sql(
        f"""
        SELECT gvkey, datadate, at, sale, ni, invt
        FROM comp.funda
        WHERE datadate >= '{args.start}' AND datadate <= '{args.end}'
    """
    )

    control_data["log_assets"] = np.log(control_data["at"])
    control_data["roa"] = control_data["ni"] / control_data["at"]
    control_data["inventory_ratio"] = control_data["invt"] / control_data["at"]
    control_data["quarter"] = pd.to_datetime(control_data["datadate"]).dt.to_period("Q")

    merged_data = pd.merge(
        quarterly_volatility,
        control_data,
        left_on=["permno", "quarter"],
        right_on=["gvkey", "quarter"],
        how="left",
    )

    merged_data.to_excel(args.output, index=False)
    logger.info("数据已保存为 %s", args.output)


if __name__ == "__main__":
    main()
