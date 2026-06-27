import requests
import pandas as pd

cities = [
    {"name": "温州", "lat": 28.01, "lon": 120.67},
    {"name": "杭州", "lat": 30.27, "lon": 120.15},
    {"name": "宁波", "lat": 29.87, "lon": 121.54},
    {"name": "金华", "lat": 29.10, "lon": 119.65},
]

start_date = "2024-06-01"
end_date = "2024-08-31"
all_data = []
for city in cities:
    print(f"正在获取 {city['name']} 的数据...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m",
        "timezone": "Asia/Shanghai"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        hourly = data["hourly"]
        df = pd.DataFrame({
            "time": hourly["time"],
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relativehumidity_2m"],
            "windspeed": hourly["windspeed_10m"]
        })
        df["city"] = city["name"]
        all_data.append(df)
        print(f"✅ {city['name']} 获取成功，共 {len(df)} 行")
    else:
        print(f"❌ {city['name']} 失败")

final_df = pd.concat(all_data, ignore_index=True)
final_df["THI"] = final_df["temperature"] - 0.55 * (1 - final_df["humidity"] / 100) * (final_df["temperature"] - 14.5)
final_df["date"] = pd.to_datetime(final_df["time"]).dt.date
# 保存到 data 文件夹
final_df.to_csv("../data/zhejiang_THI_data.csv", index=False, encoding="utf-8-sig")
print("\n🎉 数据已保存到 ../data/zhejiang_THI_data.csv")
