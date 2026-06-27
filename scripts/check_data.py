import pandas as pd
import os

# 1. 读取气象数据（你的THI数据）
#    ⚠️ 请检查下面这行路径是否正确
thi_df = pd.read_csv(os.path.join("..", "data", "zhejiang_THI_data.csv"))

# 2. 读取城市特征数据
feat_df = pd.read_csv(os.path.join("..", "data", "city_features.csv"))

# 3. 按城市计算平均THI
city_thi = thi_df.groupby("city")["THI"].mean().reset_index()
city_thi.columns = ["city", "avg_THI"]

# 4. 合并数据，用于最终分析
merged_df = pd.merge(city_thi, feat_df, on="city")

# 5. 检查合并结果
print("✅ 数据合并成功！以下是用于分析的数据集：")
print(merged_df)