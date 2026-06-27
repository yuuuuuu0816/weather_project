import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

df = pd.read_csv("zhejiang_THI_data.csv")
df["time"] = pd.to_datetime(df["time"])

# 按城市+日期聚合
daily_df = df.groupby(["city", "date"]).agg({
    "temperature": "mean",
    "humidity": "mean",
    "windspeed": "mean",
    "THI": "mean"
}).reset_index()

# 提取用于聚类的特征
features = ["temperature", "humidity", "windspeed", "THI"]
X = daily_df[features]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means聚类（分为3类）
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
daily_df["cluster"] = kmeans.fit_predict(X_scaled)

# 可视化聚类结果
fig = px.scatter(
    daily_df,
    x="temperature",
    y="humidity",
    color="cluster",
    hover_data=["city", "date"],
    title="K-Means聚类：不同热舒适度模式识别",
    labels={"temperature": "气温 (℃)", "humidity": "相对湿度 (%)"}
)
fig.show()

# 查看各类别的特征
print("各类别中心值：")
print(daily_df.groupby("cluster")[features].mean())