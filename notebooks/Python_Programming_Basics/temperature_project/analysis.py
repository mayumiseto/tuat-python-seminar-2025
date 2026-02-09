# ==============================
# analysis.py
# ==============================

from functions import (
    fetch_temperature_data,
    save_temperature_csv,
    load_temperature_csv,
    calc_month_mean,
    plot_month_temperature
)


def main():
    # ==============================
    # パラメータ設定
    # ==============================

    # 東京
    lat = 35.6895
    lon = 139.6917

    # 解析期間
    start = "1980-08-01"
    end = "2025-08-31"

    # 対象月
    month = 8

    # CSV ファイル
    out_file = "tokyo_daily_mean_temperature_1980_2025.csv"

    # ==============================
    # 処理の流れ
    # ==============================

    # データ取得
    data = fetch_temperature_data(lat, lon, start, end)

    # JSON → CSV
    save_temperature_csv(data, out_file)

    # CSV → DataFrame
    df = load_temperature_csv(out_file)

    # 月平均の計算
    df_month = calc_month_mean(df, month)

    # 可視化
    plot_month_temperature(df_month, month)


if __name__ == "__main__":
    main()
