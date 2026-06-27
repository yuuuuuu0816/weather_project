import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ========== 1. 读取爬取好的数据 ==========
df = pd.read_csv("zhejiang_THI_data.csv")
df["time"] = pd.to_datetime(df["time"])

# ========== 2. 按天聚合（计算每日平均THI）==========
daily_df = df.groupby(["city", "date"]).agg({
    "temperature": "mean",
    "humidity": "mean",
    "windspeed": "mean",
    "THI": "mean"
}).reset_index()
daily_df["date"] = pd.to_datetime(daily_df["date"])

print("✅ 数据加载完成！")
print(f"总数据量：{len(df)} 行（逐小时）")
print(f"日聚合数据量：{len(daily_df)} 行")

# ========== 3. 图1：各城市THI时间序列（折线图）==========
fig1 = px.line(
    daily_df,
    x="date",
    y="THI",
    color="city",
    title="浙江省各城市夏季热舒适度指数(THI)日变化趋势 (2024年6-8月)",
    labels={"THI": "热舒适度指数", "date": "日期"}
)
fig1.show()

# ========== 4. 图2：各城市THI分布对比（箱线图）==========
fig2 = px.box(
    daily_df,
    x="city",
    y="THI",
    color="city",
    title="各城市THI分布对比",
    labels={"THI": "热舒适度指数", "city": "城市"}
)
fig2.show()

# ========== 5. 图3：温度与湿度对THI的影响（散点图）==========
fig3 = px.scatter(
    df.sample(1000),  # 随机抽取1000个点避免太密集
    x="temperature",
    y="humidity",
    color="THI",
    title="温度与湿度对热舒适度指数的影响",
    labels={"temperature": "气温 (℃)", "humidity": "相对湿度 (%)"},
    color_continuous_scale="RdYlBu_r"
)
fig3.show()

# ========== 6. 保存图表为HTML（用于论文和报告）==========
fig1.write_html("THI_trend.html")
fig2.write_html("THI_boxplot.html")
fig3.write_html("THI_scatter.html")
print("\n✅ 图表已保存为HTML文件，可直接在浏览器中打开查看！")