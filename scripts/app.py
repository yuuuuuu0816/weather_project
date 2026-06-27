import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
from PIL import Image
import io
import base64

# ========== 页面配置 ==========
st.set_page_config(
    page_title="浙江省热舒适度综合评估",
    page_icon="🌊",
    layout="wide"
)

# ========== 加载数据 ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "zhejiang_THI_data.csv")
FEAT_PATH = os.path.join(BASE_DIR, "data", "city_features.csv")
IMAGE_PATH = os.path.join(BASE_DIR, "images", "浙江地图.jpg")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    return df


@st.cache_data
def load_features():
    return pd.read_csv(FEAT_PATH)


df = load_data()
city_features = load_features()

# ========== 界面样式优化（真正可见的毛玻璃效果） ==========
st.markdown("""
<style>
    /* ===== 全局背景（柔和渐变，让毛玻璃可见） ===== */
    .stApp {
        background: linear-gradient(145deg, #eef2f7 0%, #dce3ed 100%);
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    /* ===== 顶部横幅 ===== */
    .banner {
        background: linear-gradient(145deg, #0b1a3a 0%, #1a3a6b 60%, #2a5a8a 100%);
        border-radius: 20px;
        padding: 1.8rem 2.5rem;
        margin-bottom: 1.8rem;
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        box-shadow: 0 8px 32px rgba(10, 30, 60, 0.25);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .banner .text .tag {
        font-size: 0.65rem;
        opacity: 0.5;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.2rem;
    }
    .banner .text h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .banner .text h1 span { color: #7fc8f8; }
    .banner .text .sub {
        opacity: 0.65;
        margin-top: 0.2rem;
        font-weight: 300;
        font-size: 1rem;
    }
    .banner .text .meta {
        display: flex;
        gap: 1.8rem;
        margin-top: 0.6rem;
        font-size: 0.8rem;
        opacity: 0.45;
        flex-wrap: wrap;
    }
    .banner .image img {
        max-height: 110px;
        border-radius: 12px;
        opacity: 0.85;
        object-fit: cover;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }

    /* ===== 研究框架卡片（半透明+毛玻璃） ===== */
    .framework-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0 1.5rem 0;
    }
    .framework-item {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 1rem 1.2rem;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #2a5a8a;
        border: 1px solid rgba(255,255,255,0.4);
        border-left-width: 4px;
        transition: all 0.25s ease;
    }
    .framework-item:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 28px rgba(0,0,0,0.10);
    }
    .framework-item .icon { font-size: 1.8rem; display: block; margin-bottom: 0.1rem; }
    .framework-item .title { font-weight: 600; color: #0b1a3a; font-size: 0.9rem; }
    .framework-item .desc { font-size: 0.75rem; color: #5a6a7a; margin-top: 0.05rem; }

    /* ===== 指标卡片（半透明+毛玻璃） ===== */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0 1.5rem 0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 1.1rem 1.2rem;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        text-align: center;
        border-bottom: 3px solid #2a5a8a;
        border: 1px solid rgba(255,255,255,0.4);
        border-bottom-width: 3px;
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(0,0,0,0.10);
    }
    .metric-card .icon { font-size: 1.6rem; display: block; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #0b1a3a; margin: 0.1rem 0; }
    .metric-card .label { font-size: 0.75rem; color: #5a6a7a; }

    /* ===== 图例说明（半透明+毛玻璃） ===== */
    .legend-box {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid rgba(255,255,255,0.4);
        margin: 1rem 0;
    }
    .legend-box table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .legend-box td, .legend-box th { padding: 0.3rem 0.5rem; border-bottom: 1px solid rgba(0,0,0,0.05); }
    .legend-box .color-dot { display: inline-block; width: 12px; height: 12px; border-radius: 4px; margin-right: 0.4rem; }

    /* ===== 图表卡片（半透明+毛玻璃） ===== */
    .chart-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid rgba(255,255,255,0.4);
        margin-bottom: 1.2rem;
        transition: all 0.2s ease;
    }
    .chart-card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .chart-card .chart-title { font-weight: 600; color: #0b1a3a; font-size: 0.95rem; margin-bottom: 0.5rem; }

    /* ===== 标签页 ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.3rem;
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(8px);
        padding: 0.4rem 0.6rem;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid rgba(255,255,255,0.3);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 30px !important;
        padding: 0.35rem 1.2rem !important;
        font-weight: 500;
        font-size: 0.8rem;
        transition: all 0.2s ease;
        color: #4a5a6a !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.5) !important;
        color: #0b1a3a !important;
    }
    .stTabs [aria-selected="true"] {
        background: #0b1a3a !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(11, 26, 58, 0.25);
    }

    /* ===== 侧边栏 ===== */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px 0 0 16px;
        box-shadow: 2px 0 24px rgba(0,0,0,0.06);
        padding: 1rem 0.5rem;
        border-right: 1px solid rgba(255,255,255,0.3);
    }
    [data-testid="stSidebar"] .stSelectbox label {
        font-weight: 600;
        color: #0b1a3a;
        font-size: 0.85rem;
    }

    /* ===== 通用 ===== */
    .section-title { font-size: 1.15rem; font-weight: 700; color: #0b1a3a; margin: 1.2rem 0 0.8rem 0; }
    .footer {
        text-align: center;
        color: #7a8a9a;
        font-size: 0.7rem;
        padding-top: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.3);
        letter-spacing: 0.3px;
    }

    /* ===== 模型评估指标卡 ===== */
    .metric-eval {
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(8px);
        padding: 0.6rem 0.8rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        border-top: 3px solid #2a5a8a;
        margin: 0.3rem 0;
    }
    .metric-eval .val { font-size: 1.3rem; font-weight: 700; color: #0b1a3a; }
    .metric-eval .lbl { font-size: 0.7rem; color: #5a6a7a; }
</style>
""", unsafe_allow_html=True)

# ========== 顶部横幅 ==========
img_exists = os.path.exists(IMAGE_PATH)
if img_exists:
    try:
        img = Image.open(IMAGE_PATH)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        banner_html = f"""
        <div class="banner">
            <div class="text">
                <div class="tag">📊 数据可视化 · 课程大作业</div>
                <h1>浙江省城市 <span>热舒适度</span> 综合评估</h1>
                <div class="sub">基于 ERA5 再分析数据 · 2024年夏季 · 多维度热环境量化分析</div>
                <div class="meta">
                    <span>📍 4 个主要城市</span>
                    <span>📅 2024年 6-8 月</span>
                    <span>🌡️ 逐小时数据</span>
                    <span>📈 THI 指数</span>
                </div>
            </div>
            <div class="image">
                <img src="data:image/jpeg;base64,{img_base64}" alt="浙江地图">
            </div>
        </div>
        """
    except:
        banner_html = """
        <div class="banner">
            <div class="text">
                <div class="tag">📊 数据可视化 · 课程大作业</div>
                <h1>浙江省城市 <span>热舒适度</span> 综合评估</h1>
                <div class="sub">基于 ERA5 再分析数据 · 2024年夏季 · 多维度热环境量化分析</div>
                <div class="meta">
                    <span>📍 4 个主要城市</span>
                    <span>📅 2024年 6-8 月</span>
                    <span>🌡️ 逐小时数据</span>
                    <span>📈 THI 指数</span>
                </div>
            </div>
        </div>
        """
else:
    banner_html = """
    <div class="banner">
        <div class="text">
            <div class="tag">📊 数据可视化 · 课程大作业</div>
            <h1>浙江省城市 <span>热舒适度</span> 综合评估</h1>
            <div class="sub">基于 ERA5 再分析数据 · 2024年夏季 · 多维度热环境量化分析</div>
            <div class="meta">
                <span>📍 4 个主要城市</span>
                <span>📅 2024年 6-8 月</span>
                <span>🌡️ 逐小时数据</span>
                <span>📈 THI 指数</span>
            </div>
        </div>
    </div>
    """
st.markdown(banner_html, unsafe_allow_html=True)

# ========== 侧边栏 ==========
st.sidebar.markdown("### 🗂️ 数据筛选")
cities = df["city"].unique().tolist()
selected_cities = st.sidebar.multiselect("🏙️ 选择城市", cities, default=cities)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 数据来源")
st.sidebar.markdown("""
- Open-Meteo ERA5（气温 · 湿度 · 风速）
- 中国城市建设统计年鉴（绿化率 · 人口密度）
""")
st.sidebar.caption("📍 覆盖：温州 · 杭州 · 宁波 · 金华")

# ========== 日期筛选 ==========
min_date = df["time"].min().date()
max_date = df["time"].max().date()
if "sel_start" not in st.session_state:
    st.session_state["sel_start"] = min_date
    st.session_state["sel_end"] = max_date

col_left, col_right = st.columns([2, 1])
with col_left:
    st.markdown("#### 📅 时间筛选")
    btn1, btn2, btn3, btn4 = st.columns(4)
    months = {
        "全部": (min_date, max_date),
        "6月": (pd.to_datetime("2024-06-01").date(), pd.to_datetime("2024-06-30").date()),
        "7月": (pd.to_datetime("2024-07-01").date(), pd.to_datetime("2024-07-31").date()),
        "8月": (pd.to_datetime("2024-08-01").date(), pd.to_datetime("2024-08-31").date()),
    }
    with btn1:
        if st.button("📅 全部", use_container_width=True):
            st.session_state["sel_start"], st.session_state["sel_end"] = months["全部"]
    with btn2:
        if st.button("🌱 6月", use_container_width=True):
            st.session_state["sel_start"], st.session_state["sel_end"] = months["6月"]
    with btn3:
        if st.button("☀️ 7月", use_container_width=True):
            st.session_state["sel_start"], st.session_state["sel_end"] = months["7月"]
    with btn4:
        if st.button("🌾 8月", use_container_width=True):
            st.session_state["sel_start"], st.session_state["sel_end"] = months["8月"]

with col_right:
    st.markdown("#### 精确选择")
    date_range = st.date_input(
        "选择日期",
        value=(st.session_state["sel_start"], st.session_state["sel_end"]),
        min_value=min_date,
        max_value=max_date,
        format="YYYY/MM/DD",
        label_visibility="collapsed"
    )
    if len(date_range) == 2:
        st.session_state["sel_start"], st.session_state["sel_end"] = date_range

st.info(
    f"📍 当前查看：**{st.session_state['sel_start'].strftime('%Y年%m月%d日')}** 至 **{st.session_state['sel_end'].strftime('%Y年%m月%d日')}**  |  🏙️ {', '.join(selected_cities)}")

# ========== 数据过滤 ==========
filtered_df = df[df["city"].isin(selected_cities)]
filtered_df = filtered_df[
    (filtered_df["time"].dt.date >= st.session_state["sel_start"]) &
    (filtered_df["time"].dt.date <= st.session_state["sel_end"])
    ]

if filtered_df.empty:
    st.warning("当前筛选条件下没有数据")
    st.stop()

# ========== 研究框架 ==========
st.markdown('<div class="section-title">📊 研究框架</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="framework-item" style="border-left-color:#2a5a8a;">
        <span class="icon">🌡️</span>
        <div class="title">热舒适度 (THI)</div>
        <div class="desc">温湿指数 · 综合温度与湿度</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="framework-item" style="border-left-color:#27ae60;">
        <span class="icon">🌿</span>
        <div class="title">绿化覆盖率</div>
        <div class="desc">城市建成区绿化面积占比</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="framework-item" style="border-left-color:#e67e22;">
        <span class="icon">🏙️</span>
        <div class="title">人口密度</div>
        <div class="desc">城市热源强度指标</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="framework-item" style="border-left-color:#3498db;">
        <span class="icon">🌊</span>
        <div class="title">水体面积占比</div>
        <div class="desc">水体降温效应评估</div>
    </div>
    """, unsafe_allow_html=True)

# ========== 指标卡片 ==========
st.markdown('<div class="section-title">📈 关键指标</div>', unsafe_allow_html=True)
avg_thi = filtered_df["THI"].mean()
avg_temp = filtered_df["temperature"].mean()
avg_hum = filtered_df["humidity"].mean()
if avg_thi < 22:
    comfort_text = "舒适"
elif avg_thi < 26:
    comfort_text = "较为舒适"
elif avg_thi < 30:
    comfort_text = "较不舒适"
else:
    comfort_text = "非常不舒适"

col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 数据条数", f"{len(filtered_df):,}")
col2.metric("🌡️ 平均气温", f"{avg_temp:.1f}℃")
col3.metric("💧 平均湿度", f"{avg_hum:.1f}%")
col4.metric("🔥 平均THI", f"{avg_thi:.1f} · {comfort_text}")

# ========== 图例说明 ==========
st.markdown("""
<div class="legend-box">
    <b>📖 图例与指标说明</b>
    <table>
        <tr><th>指标</th><th>含义</th><th>等级划分</th></tr>
        <tr>
            <td><b>THI（温湿指数）</b></td>
            <td>综合温度与湿度对人体舒适度的影响</td>
            <td>
                <span class="color-dot" style="background:#27ae60;"></span> &lt; 22 = 舒适<br>
                <span class="color-dot" style="background:#f1c40f;"></span> 22–26 = 较为舒适<br>
                <span class="color-dot" style="background:#e67e22;"></span> 26–30 = 较不舒适<br>
                <span class="color-dot" style="background:#e74c3c;"></span> &gt; 30 = 非常不舒适
            </td>
        </tr>
    </table>
    <div style="margin-top:0.5rem; font-size:0.75rem; color:#8a9aaa;">
        THI = T - 0.55 × (1 - RH/100) × (T - 14.5) &nbsp;·&nbsp; T: 气温(°C) &nbsp;·&nbsp; RH: 相对湿度(%)
    </div>
</div>
""", unsafe_allow_html=True)

# ========== 标签页 ==========
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["📈 时序趋势", "🕒 24h变化", "📊 城市对比", "🧠 聚类分析", "📋 城市报告", "🌿 影响因素", "🗺️ 空间分布", "📊 模型评估"]
)

# ---------- Tab1: 时序趋势 ----------
with tab1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📈 THI 日变化趋势</div>', unsafe_allow_html=True)
    daily_df = filtered_df.groupby(["city", filtered_df["time"].dt.date])["THI"].mean().reset_index()
    daily_df.columns = ["city", "date", "THI"]
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    fig = px.line(daily_df, x="date", y="THI", color="city",
                  labels={"THI": "热舒适度指数", "date": "日期"})
    fig.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab2: 24h变化 ----------
with tab2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🕒 24小时平均THI变化</div>', unsafe_allow_html=True)
    hourly = filtered_df.groupby(["city", filtered_df["time"].dt.hour])["THI"].mean().reset_index()
    hourly.columns = ["city", "hour", "THI"]
    fig = px.line(hourly, x="hour", y="THI", color="city",
                  labels={"hour": "小时 (0-23)", "THI": "平均THI"})
    fig.update_xaxes(tickmode="linear", dtick=2)
    fig.update_layout(height=380, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    peak_hour = hourly.groupby("hour")["THI"].mean().idxmax()
    st.info(f"💡 一天中最不舒适的时段是 **{peak_hour}:00**")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab3: 城市对比 ----------
with tab3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📊 各城市平均THI对比</div>', unsafe_allow_html=True)
    city_stats = filtered_df.groupby("city")["THI"].agg(["mean", "std"]).reset_index()
    city_stats.columns = ["city", "mean", "std"]
    fig = go.Figure(go.Bar(
        x=city_stats["city"],
        y=city_stats["mean"],
        error_y=dict(type="data", array=city_stats["std"], color="#2a5a8a"),
        marker_color="#2a5a8a",
        text=[f"{v:.1f}" for v in city_stats["mean"]],
        textposition="outside"
    ))
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab4: 聚类分析 ----------
with tab4:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🧠 天气模式聚类</div>', unsafe_allow_html=True)
    cluster_data = filtered_df[["temperature", "humidity", "windspeed"]].dropna()
    if len(cluster_data) > 10:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(cluster_data)
        labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(scaled)
        cluster_data["cluster"] = labels
        fig = px.scatter(cluster_data, x="temperature", y="humidity", color="cluster",
                         labels={"temperature": "气温 (°C)", "humidity": "湿度 (%)"})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("数据量不足")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab5: 城市报告 ----------
with tab5:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📋 城市热舒适报告</div>', unsafe_allow_html=True)
    report_city = st.selectbox("选择城市", df["city"].unique())
    city_data = df[df["city"] == report_city]
    avg_thi_city = city_data["THI"].mean()
    max_thi_city = city_data["THI"].max()
    avg_temp_city = city_data["temperature"].mean()
    avg_hum_city = city_data["humidity"].mean()

    if avg_thi_city < 22:
        level, color, advice = "🟢 舒适", "#27ae60", "气候宜人，适合户外活动"
    elif avg_thi_city < 26:
        level, color, advice = "🟡 较为舒适", "#f1c40f", "整体舒适，午后注意防晒"
    elif avg_thi_city < 30:
        level, color, advice = "🟠 较不舒适", "#e67e22", "建议避免中午户外活动"
    else:
        level, color, advice = "🔴 非常不舒适", "#e74c3c", "高温高湿，尽量减少户外活动"

    st.markdown(f"""
    <div style="background:#f8f9fa; padding:1.2rem 1.5rem; border-radius:12px; border-left:4px solid {color};">
        <h3 style="margin:0 0 0.5rem 0;">{report_city}</h3>
        <p><b>平均 THI：</b>{avg_thi_city:.1f} · {level}</p>
        <p><b>最高 THI：</b>{max_thi_city:.1f}</p>
        <p><b>平均气温：</b>{avg_temp_city:.1f} °C</p>
        <p><b>平均湿度：</b>{avg_hum_city:.1f} %</p>
        <p>💡 {advice}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab6: 影响因素 ----------
with tab6:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🌿 城市绿化与热舒适度</div>', unsafe_allow_html=True)
    city_thi = filtered_df.groupby("city")["THI"].mean().reset_index()
    city_thi.columns = ["city", "avg_THI"]
    merged_df = pd.merge(city_thi, city_features, on="city")
    if not merged_df.empty and len(merged_df) >= 3:
        fig = px.scatter(
            merged_df,
            x="green_rate",
            y="avg_THI",
            text="city",
            size="population_density",
            labels={"green_rate": "绿化覆盖率 (%)", "avg_THI": "平均THI"},
            trendline="ols",
            trendline_color_override="#e74c3c"
        )
        fig.update_traces(textposition="top center", marker_sizemin=20)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        corr = merged_df["green_rate"].corr(merged_df["avg_THI"])
        if corr < -0.5:
            st.success(f"📊 相关系数 r = {corr:.2f} · 强负相关")
        elif corr < -0.2:
            st.info(f"📊 相关系数 r = {corr:.2f} · 中等负相关")
        else:
            st.warning(f"📊 相关系数 r = {corr:.2f} · 相关性较弱")
    else:
        st.warning("数据不足")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab7: 空间分布 ----------
with tab7:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🗺️ 浙江省热舒适度空间分布</div>', unsafe_allow_html=True)

    city_thi_map = filtered_df.groupby("city")["THI"].mean().reset_index()
    city_thi_map.columns = ["city", "avg_THI"]
    coords = {
        "温州": {"lat": 28.01, "lon": 120.67},
        "杭州": {"lat": 30.27, "lon": 120.15},
        "宁波": {"lat": 29.87, "lon": 121.54},
        "金华": {"lat": 29.10, "lon": 119.65},
    }
    city_thi_map["lat"] = city_thi_map["city"].map(lambda x: coords[x]["lat"])
    city_thi_map["lon"] = city_thi_map["city"].map(lambda x: coords[x]["lon"])

    a, b, c, d = st.columns(4)
    a.metric("🌡️ 平均THI", f"{city_thi_map['avg_THI'].mean():.1f}")
    max_city = city_thi_map.loc[city_thi_map['avg_THI'].idxmax(), 'city']
    b.metric("🔥 最热城市", max_city)
    min_city = city_thi_map.loc[city_thi_map['avg_THI'].idxmin(), 'city']
    c.metric("🍃 最舒适城市", min_city)
    heatwave = filtered_df[filtered_df["THI"] > 30]["time"].dt.date.nunique()
    d.metric("☀️ 热浪天数", f"{heatwave} 天")

    fig_map = px.scatter_mapbox(
        city_thi_map,
        lat="lat", lon="lon",
        color="avg_THI",
        size="avg_THI",
        text="city",
        color_continuous_scale="RdYlBu_r",
        size_max=60,
        zoom=6.5,
        center={"lat": 29.0, "lon": 120.5},
        title="气泡越大 / 颜色越深 = 越不舒适"
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        height=420,
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    fig_map.update_traces(textposition="top center")
    st.plotly_chart(fig_map, use_container_width=True)


    def classify(val):
        if val < 22:
            return "🟢 舒适"
        elif val < 26:
            return "🟡 较为舒适"
        elif val < 30:
            return "🟠 较不舒适"
        else:
            return "🔴 非常不舒适"


    filtered_df["level"] = filtered_df["THI"].apply(classify)
    level_counts = filtered_df["level"].value_counts().reset_index()
    level_counts.columns = ["level", "count"]
    fig_pie = px.pie(
        level_counts, values="count", names="level",
        title="THI 舒适度等级占比",
        color="level",
        color_discrete_map={
            "🟢 舒适": "#27ae60",
            "🟡 较为舒适": "#f1c40f",
            "🟠 较不舒适": "#e67e22",
            "🔴 非常不舒适": "#e74c3c"
        },
        hole=0.4
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=340)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### 📋 城市热舒适度排名")
    ranking = city_thi_map.sort_values("avg_THI", ascending=True)
    ranking["排名"] = range(1, len(ranking) + 1)
    ranking["等级"] = ranking["avg_THI"].apply(classify)
    ranking = ranking[["排名", "city", "avg_THI", "等级"]]
    ranking.columns = ["排名", "城市", "平均THI", "舒适度等级"]
    st.dataframe(ranking, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab8: 模型评估 ----------
with tab8:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📊 模型评估：THI 预测</div>', unsafe_allow_html=True)

    st.markdown("""
    **建模目标**：使用气温、湿度、风速三个特征，预测热舒适度指数 (THI)。  
    **数据**：当前筛选后的所有逐小时数据，共计 {} 条。  
    **模型对比**：线性回归 (基准) 与 随机森林 (非线性)。
    """.format(len(filtered_df)))

    model_df = filtered_df[["temperature", "humidity", "windspeed", "THI"]].dropna()
    if len(model_df) < 50:
        st.warning("数据量不足，无法进行建模评估。")
    else:
        X = model_df[["temperature", "humidity", "windspeed"]]
        y = model_df["THI"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)


        def evaluate(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mask = y_true != 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            return mae, rmse, r2, mape


        mae_lr, rmse_lr, r2_lr, mape_lr = evaluate(y_test, y_pred_lr)
        mae_rf, rmse_rf, r2_rf, mape_rf = evaluate(y_test, y_pred_rf)

        st.markdown("#### 模型性能对比")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**线性回归**")
            st.markdown(f"""
            <div style="background:#f8f9fa; padding:0.8rem; border-radius:10px;">
                <div class="metric-eval"><span class="val">{mae_lr:.2f}</span><span class="lbl"> MAE</span></div>
                <div class="metric-eval"><span class="val">{rmse_lr:.2f}</span><span class="lbl"> RMSE</span></div>
                <div class="metric-eval"><span class="val">{r2_lr:.4f}</span><span class="lbl"> R²</span></div>
                <div class="metric-eval"><span class="val">{mape_lr:.2f}%</span><span class="lbl"> MAPE</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("**随机森林**")
            st.markdown(f"""
            <div style="background:#f8f9fa; padding:0.8rem; border-radius:10px;">
                <div class="metric-eval"><span class="val">{mae_rf:.2f}</span><span class="lbl"> MAE</span></div>
                <div class="metric-eval"><span class="val">{rmse_rf:.2f}</span><span class="lbl"> RMSE</span></div>
                <div class="metric-eval"><span class="val">{r2_rf:.4f}</span><span class="lbl"> R²</span></div>
                <div class="metric-eval"><span class="val">{mape_rf:.2f}%</span><span class="lbl"> MAPE</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 特征重要性 (随机森林)")
        importance = rf.feature_importances_
        feature_names = ["气温", "湿度", "风速"]
        imp_df = pd.DataFrame({"特征": feature_names, "重要性": importance})
        imp_df = imp_df.sort_values("重要性", ascending=True)
        fig_imp = px.bar(imp_df, x="重要性", y="特征", orientation="h",
                         title="各特征对THI预测的相对贡献",
                         color="重要性", color_continuous_scale="Blues")
        fig_imp.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### 预测值 vs 真实值")
        sample_idx = np.random.choice(len(y_test), min(200, len(y_test)), replace=False)
        plot_df = pd.DataFrame({
            "真实值": y_test.iloc[sample_idx],
            "线性回归预测": y_pred_lr[sample_idx],
            "随机森林预测": y_pred_rf[sample_idx]
        })
        fig_pred = px.scatter(plot_df, x="真实值", y=["线性回归预测", "随机森林预测"],
                              title="模型预测效果对比",
                              labels={"value": "THI", "variable": "模型"},
                              color_discrete_map={"线性回归预测": "#2a5a8a", "随机森林预测": "#e67e22"})
        max_val = max(plot_df["真实值"].max(), plot_df["线性回归预测"].max(), plot_df["随机森林预测"].max())
        min_val = min(plot_df["真实值"].min(), plot_df["线性回归预测"].min(), plot_df["随机森林预测"].min())
        fig_pred.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                      mode="lines", name="理想线",
                                      line=dict(dash="dash", color="gray")))
        fig_pred.update_layout(height=400)
        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("#### 残差分布 (随机森林)")
        residuals = y_test - y_pred_rf
        fig_res = px.histogram(residuals, nbins=50, title="随机森林预测残差分布",
                               labels={"value": "残差"})
        fig_res.update_layout(height=300)
        st.plotly_chart(fig_res, use_container_width=True)

        st.markdown("""
        **结论**：随机森林模型在各项指标上均优于线性回归，说明THI与特征之间存在非线性关系。  
        特征重要性显示 **气温** 是影响THI的最主要因素，湿度次之，风速影响较小。
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# ========== 页脚 ==========
st.markdown("""
<div class="footer">
    浙江省热舒适度综合评估 · 数据可视化课程大作业<br>
    数据来源：Open-Meteo ERA5 · 中国城市建设统计年鉴
</div>
""", unsafe_allow_html=True)
