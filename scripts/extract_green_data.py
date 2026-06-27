import pandas as pd

# 手动整理城市统计特征数据
# 这些数据可以从《浙江统计年鉴》或网上公开资料查到
city_features = pd.DataFrame({
    "city": ["温州", "杭州", "宁波", "金华"],
    "green_rate": [38.5, 40.2, 37.8, 35.2],   # 绿化覆盖率 (%)
    "population_density": [814, 749, 950, 850],  # 人口密度 (人/km²)
    "water_area_ratio": [8.5, 6.2, 7.8, 4.5]   # 水体面积占比 (%)
})

# 保存为CSV，放在data文件夹里
city_features.to_csv("../data/city_features.csv", index=False, encoding="utf-8-sig")
print("✅ 城市特征数据已保存！")