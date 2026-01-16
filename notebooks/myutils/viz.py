from __future__ import annotations
from typing import Tuple
import contextlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter

# ---- スタイル適用 ----
def apply_pub_style() -> None:
    """論文っぽい見た目の共通スタイルをグローバルに適用。"""
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 600,
        "savefig.transparent": True,
    })

@contextlib.contextmanager
def use_pub_style():
    """with ブロック内だけスタイルを一時適用（副作用を局所化）。"""
    old = mpl.rcParams.copy()
    try:
        apply_pub_style()
        yield
    finally:
        mpl.rcParams.update(old)

# ---- 可視化：時系列＋右ヒスト ----
def plot_timeseries_with_right_hist(
    x,
    y,
    x_label: str,
    y_label: str,
    date_fmt: str = "%Y-%m-%d",
    figsize: Tuple[float, float] = (5, 2.5),
    save_eps_path: str | None = None,
    bins: int = 20,
):
    """右側にヒストグラムを並べた時系列図。"""
    # Series化・型変換
    x = pd.Series(x)
    y = pd.Series(y)
    if not np.issubdtype(x.dtype, np.datetime64):
        x = pd.to_datetime(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if data.empty:
        raise ValueError("有効なデータがありません（x/y が NaN だらけか、型変換失敗）")

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

    # 左：時系列
    ax_ts = fig.add_subplot(gs[0])
    ax_ts.plot(data["x"], data["y"], color="black", marker="o", ms=2)
    ax_ts.set_xlabel(x_label); ax_ts.set_ylabel(y_label)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    fig.autofmt_xdate(rotation=90, ha="center")
    ax_ts.grid(True, linewidth=0.5, alpha=0.5)

    # 右：ヒスト（横向き）
    ax_hist = fig.add_subplot(gs[1], sharey=ax_ts)
    ax_hist.hist(data["y"].values, bins=bins, orientation="horizontal",
                 color="0.6", edgecolor="black", linewidth=0.5)
    ax_hist.set_xlabel("Count")
    plt.setp(ax_hist.get_yticklabels(), visible=False)
    ax_hist.grid(True, linewidth=0.5, alpha=0.3)

    plt.subplots_adjust(left=0.18, right=0.95, bottom=0.28, top=0.95)

    if save_eps_path:
        fig.savefig(save_eps_path, format="eps", bbox_inches="tight", pad_inches=0.02)
    plt.show()
    return fig, (ax_ts, ax_hist)
    
    
def plot_scatter(
    df: pd.DataFrame,
    x_name: str,
    y_name: str,
    figsize=(5, 5),
    title: str | None = None,
    auto_log_if_wide: bool = True,
    log_threshold_ratio: float = 1e3,
    equal_aspect_if_similar_scale: bool = True,
    save_path: str | None = None,
):
    """
    df の列 x_name, y_name から散布図を作成する。

    特徴
    ----
    - 数値化失敗は NaN → 自動除去
    - ダイナミックレンジが広い場合のみ log 軸に自動切替
    - log 軸使用時は「正の値のみ」を明示的に使用
    - 軸スケールが近い場合のみ equal aspect
    - fig, ax を返す（後処理しやすい）
    """

    # --- 数値化 & 欠損除去 ---
    x = pd.to_numeric(df[x_name], errors="coerce")
    y = pd.to_numeric(df[y_name], errors="coerce")
    data = pd.DataFrame({x_name: x, y_name: y}).dropna()

    if data.empty:
        raise ValueError("有効な数値データがありません（全て NaN です）。")

    # --- log 軸を使うか判定（正の値のみで評価） ---
    use_log = False
    log_data = data

    if auto_log_if_wide:
        pos = data[(data[x_name] > 0) & (data[y_name] > 0)]
        if not pos.empty:
            xr = pos[x_name].max() / pos[x_name].min()
            yr = pos[y_name].max() / pos[y_name].min()
            if (xr >= log_threshold_ratio) or (yr >= log_threshold_ratio):
                use_log = True
                log_data = pos  # log 軸用データを明示

    # --- Figure ---
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        log_data[x_name].values,
        log_data[y_name].values,
        s=20,
        edgecolor="black",
        linewidth=0.5,
    )

    # --- ラベル・タイトル ---
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title if title else f"{y_name} vs {x_name}")

    # --- 軸スケール ---
    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        # log でない場合のみ通常表記（←競合回避）
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style="plain", axis="both")

    # --- 等軸比（線形軸のみ） ---
    if equal_aspect_if_similar_scale and not use_log:
        x_span = log_data[x_name].max() - log_data[x_name].min()
        y_span = log_data[y_name].max() - log_data[y_name].min()
        if x_span > 0 and y_span > 0:
            ratio = max(x_span, y_span) / min(x_span, y_span)
            if ratio < 2:
                ax.set_aspect("equal", adjustable="box")

    # --- 仕上げ ---
    ax.grid(True, linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)

    plt.show()
    return fig, ax