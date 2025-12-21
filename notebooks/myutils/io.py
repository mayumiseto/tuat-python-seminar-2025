from __future__ import annotations
from pathlib import Path
import pandas as pd

def safe_read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """
    文字コードを自動判定してCSVを安全に読み込む。
    kwargs は pd.read_csv にそのまま渡せます（dtype, usecols, na_values 等）
    """
    path = Path(path)
    try:
        import chardet
        with open(path, "rb") as f:
            enc = chardet.detect(f.read(10000)).get("encoding") or "utf-8-sig"
    except Exception:
        enc = "utf-8-sig"

    try:
        return pd.read_csv(path, encoding=enc, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)

def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """存在する列だけ、指定順のまま抜き出す。"""
    cols = [c for c in columns if c in df.columns]
    return df.reindex(columns=cols)
