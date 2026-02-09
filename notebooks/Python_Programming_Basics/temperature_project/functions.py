# ==============================
# functions.py
# ==============================

import requests
import pandas as pd
import matplotlib.pyplot as plt


def fetch_temperature_data(lat, lon, start, end):
    """
    open-meteo API から日平均気温データを取得する
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean",
        "timezone": "Asia/Tokyo"
    }
    url = "https://archive-api.open-meteo.com/v1/archive"

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    return r.json()


def save_temperature_csv(data, out_file):
    """
    API で取得した JSON データを CSV として保存する
    """
    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "temp": data["daily"]["temperature_2m_mean"]
    })

    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_file}")


def load_temperature_csv(path):
    """
    CSV ファイルを読み込み、DataFrame として返す
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def calc_month_mean(df, month):
    """
    指定した月の年平均気温を計算する
    """
    df_month = df[df["date"].dt.month == month].copy()
    df_month["year"] = df_month["date"].dt.year

    years = sorted(df_month["year"].unique())
    year_means = []

    for y in years:
        temps = df_month[df_month["year"] == y]["temp"]
        year_means.append(temps.mean())

    return pd.DataFrame({
        "year": years,
        "ave_temp": year_means
    })


def plot_month_temperature(df, month):
    """
    月平均気温の経年変化をプロットする
    """
    plt.plot(df["year"], df["ave_temp"])
    plt.xlabel("Year")
    plt.ylabel("Average Temperature (°C)")
    plt.title(f"Yearly Average Temperature (Month = {month})")
    plt.grid(True)
    plt.show()
