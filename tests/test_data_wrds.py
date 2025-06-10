import sys
from pathlib import Path
import types

import pandas as pd

# Ensure the repository root is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))


class FakeConn:
    def __init__(self, price_df, ctrl_df):
        self.price_df = price_df
        self.ctrl_df = ctrl_df

    def raw_sql(self, query):
        if "crsp.dsf" in query:
            return self.price_df.copy()
        if "comp.funda" in query:
            return self.ctrl_df.copy()
        raise ValueError("unexpected query")


def test_main_writes_output(tmp_path):
    price_df = pd.DataFrame(
        {
            "date": ["2010-01-01", "2010-01-02"],
            "permno": [1, 1],
            "prc": [10, 11],
            "vol": [1000, 1100],
        }
    )
    ctrl_df = pd.DataFrame(
        {
            "gvkey": [1],
            "datadate": ["2010-01-01"],
            "at": [100],
            "sale": [50],
            "ni": [10],
            "invt": [20],
        }
    )
    sys.modules["wrds"] = types.SimpleNamespace(
        Connection=lambda: FakeConn(price_df, ctrl_df)
    )
    import data_wrds

    out_file = tmp_path / "out.xlsx"
    data_wrds.main(
        ["--start", "2010-01-01", "--end", "2010-01-02", "--output", str(out_file)]
    )
    assert out_file.exists()
    result = pd.read_excel(out_file)
    assert "realized_volatility" in result.columns
