# -*- coding: utf-8 -*-
"""
==========================================================================
 app.py —— 时间序列预测对比工具（网页界面 · 留出法）
==========================================================================
安装依赖：   pip install streamlit plotly
启动应用：   streamlit run app.py
             浏览器会自动打开 http://localhost:8501
停止应用：   在终端按 Ctrl + C

使用思路
--------
把整段数据从"预测起始期"处切成两段：前面若干期只用来建模（训练段），
后面若干期只用来检验模型（留出段）。模型在训练段上估计，一次性向后外推
留出段各期；SSE / SE / Theil U 等指标全部用留出段的"实际值 vs 预测值"计算。

界面说明
--------
  左侧栏 ：选择启用哪些模型、移动平均阶数、SES 的 α、预测起始期数、置信水平
  主区域 ：粘贴数据 → 查看解析结果 → 模型指标对比 → 预测与实际值对照 → 交互图表
计算全部由 ts_core.py 完成，本文件只负责界面与展示。
==========================================================================
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ts_core as core

# 预置示例数据，打开就能直接看到效果，替换成自己的数据即可
SAMPLE = ("112, 118, 109, 121, 125, 117, 130, 128, 137, 134, "
          "141, 146, 139, 152, 148, 157, 161, 154, 166, 163, "
          "172, 175, 169, 181")

# 各模型固定的图表配色
COLORS = {"MA": "#2E7D32", "SES": "#1565C0", "Naive": "#E07B39",
          "Linear": "#8E24AA"}
GREY = "#424242"
GREY_LIGHT = "#9E9E9E"

st.set_page_config(page_title="时间序列预测对比", layout="wide")


# --------------------------------------------------------------------------
# 图表
# --------------------------------------------------------------------------

def _base_layout(fig, height=460):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EAEAEA")
    return fig


def history_forecast_fig(labels, y, train_n, results, z):
    """训练段观测值 + 留出段实际值 + 各模型外推预测值（带误差棒）。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels[:train_n], y=y[:train_n], mode="lines+markers",
        name="观测值·训练段", line=dict(color=GREY, width=2.5),
        marker=dict(size=5)))
    fig.add_trace(go.Scatter(
        x=labels[train_n - 1:], y=y[train_n - 1:], mode="lines+markers",
        name="观测值·留出段", line=dict(color=GREY_LIGHT, width=2.5),
        marker=dict(size=7)))

    for r in results:
        color = COLORS.get(r.id, "#888")
        # 首点接在训练段最后一期上，使折线连续、直观看出从哪里开始外推
        x = [labels[train_n - 1]] + labels[train_n:]
        yv = [y[train_n - 1]] + list(r.forecast)
        err = [0.0] + [z * s for s in r.se_pred]
        fig.add_trace(go.Scatter(
            x=x, y=yv, mode="lines+markers", name=r.name,
            line=dict(color=color, width=1.8, dash="dot"),
            marker=dict(size=7),
            error_y=dict(type="data", array=err, visible=True,
                         thickness=1.2, width=4, color=color),
            hovertemplate="%{x}：%{y:.2f}<extra>" + r.name + "</extra>"))

    fig.update_xaxes(title_text="时点")
    fig.update_yaxes(title_text="观测值")
    return _base_layout(fig)


def train_fit_fig(labels, y, train_n, results):
    """训练段：观测值 vs 各模型的一步向前拟合值。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels[:train_n], y=y[:train_n], mode="lines+markers",
        name="观测值（训练段）", line=dict(color=GREY, width=2.5),
        marker=dict(size=5)))
    for r in results:
        fig.add_trace(go.Scatter(
            x=labels[:train_n], y=r.aligned_train_fitted(len(y)),
            mode="lines+markers", name=r.name,
            line=dict(color=COLORS.get(r.id, "#888"), width=1.6),
            marker=dict(size=4), opacity=0.9, connectgaps=False))
    fig.update_xaxes(title_text="时点")
    fig.update_yaxes(title_text="观测值")
    return _base_layout(fig)


def holdout_error_fig(labels, results):
    """留出段各期预测误差 e_h。"""
    fig = go.Figure()
    for r in results:
        fig.add_trace(go.Scatter(
            x=[labels[t] for t in r.t_index], y=r.errors,
            mode="lines+markers", name=r.name,
            line=dict(color=COLORS.get(r.id, "#888"), width=1.6),
            marker=dict(size=6)))
    fig.add_hline(y=0, line_dash="dash", line_color="#AAAAAA")
    fig.update_xaxes(title_text="留出段时点")
    fig.update_yaxes(title_text="误差  实际值 − 预测值")
    return _base_layout(fig, height=380)


def sse_fig(results):
    """各模型留出段 SSE 柱状对比。"""
    fig = go.Figure(go.Bar(
        x=[r.key for r in results],
        y=[r.stats["sse"] for r in results],
        marker_color=[COLORS.get(r.id, "#888") for r in results],
        text=[f"{r.stats['sse']:.1f}" for r in results],
        textposition="outside"))
    fig.update_yaxes(title_text="SSE（留出段）")
    return _base_layout(fig, height=380)


def _chart(fig):
    """显示 Plotly 图表；兼容不同 Streamlit 版本的宽度写法。"""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# 主区域：① 数据输入
# --------------------------------------------------------------------------

st.title("时间序列预测对比工具")
st.caption("线性趋势 / 移动平均 / 一次指数平滑 / Naive 朴素法　—　"
           "留出法评估：前段建模、后段检验，指标含 SSE、标准误 SE 与 Theil U")

raw = st.text_area(
    "① 粘贴数据（逗号、空格或换行分隔都可以）",
    value=SAMPLE, height=110,
    help="按时间先后顺序，至少 4 期；中间不要有缺失值。"
         "整段数据的最后一段会被留作检验用，不参与建模。")

try:
    y = core.parse_series(raw)
except ValueError as exc:
    st.error(f"数据解析失败：{exc}")
    st.stop()

n = len(y)
if n < 4:
    st.error(f"至少需要 4 期观测值（前段建模 + 后段检验），当前只有 {n} 期。")
    st.stop()

# --------------------------------------------------------------------------
# 侧边栏：模型与参数（放在数据解析之后，滑块范围才能跟随数据长度）
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("模型与参数")

    chosen = st.multiselect(
        "启用哪些模型",
        options=list(core.MODELS),
        default=list(core.MODELS),
        format_func=lambda m: core.MODEL_LABELS[m],
        help="可以只保留一两个模型做对比")

    ma_order = st.slider(
        "移动平均阶数 k", min_value=2, max_value=12, value=3, step=1,
        disabled="MA" not in chosen,
        help="k=3 即三期移动平均（3MA）")

    ses_alpha = st.slider(
        "SES 平滑系数 α", min_value=0.01, max_value=0.99, value=0.30,
        step=0.01, disabled="SES" not in chosen,
        help="越大越重视最新数据；SES 用它从初始值 S0 起逐期递推")

    st.divider()

    min_start, max_start = 3, n - 1
    default_start = max(min_start, min(max_start, n - 5 + 1))
    forecast_start = st.slider(
        "预测起始期数（第几期开始预测）", min_value=min_start,
        max_value=max_start, value=default_start, step=1,
        help="例如设为 6，表示用前 5 期建模，预测第 6 期及其后的所有观测值")

    _tn = forecast_start - 1
    ses_mode = st.radio(
        "SES 初始值 S0 的来源",
        ["自动：取前几期的平均", "手动输入一个数"], index=0,
        help="S0 是第 1 期之前的初始平滑值，SES 从它出发逐期递推求出 S1、S2……"
             "第 t 期的预测值取上一期的平滑值 S_(t−1)。")
    if ses_mode.startswith("自动"):
        ses_avg_n = st.number_input(
            "取前几期的平均", min_value=1, max_value=_tn, value=_tn, step=1)
        ses_s0 = None
        _m = sum(y[:int(ses_avg_n)]) / int(ses_avg_n)
        st.caption(f"S0 = 前 {int(ses_avg_n)} 期实际值的平均 = {_m:.4f}")
    else:
        ses_s0 = st.text_input("S0 的数值", value="", placeholder="例如 13.5768")
        ses_avg_n = None
        st.caption("S0 = 上面填写的数值" if str(ses_s0).strip()
                   else "还没填，S0 会被当成 0，请填一个数")
    confidence = st.select_slider(
        "预测区间置信水平", options=[0.80, 0.90, 0.95, 0.99], value=0.95)

    st.divider()
    st.caption("计算口径：模型只用训练段估计；移动平均一次性外推留出段各期，"
               "SES 从初始值 S0 起按 S_t = α·Y_t + (1−α)·S_(t−1) 逐期递推，"
               "Naive 取上一期实际值滚动预测（其 Theil U2 恒为 1）。"
               "SSE / SE / Theil U 全部按留出段的「实际值 vs 预测值」计算，"
               "残差标准误 SE = √(SSE/(H−k))。")

train_n = forecast_start - 1
horizon = n - train_n

m1, m2, m3, m4 = st.columns(4)
m1.metric("总期数", n)
m2.metric("训练段（建模用）", f"前 {train_n} 期")
m3.metric("留出段（检验用）", f"{horizon} 期")
m4.metric("整体均值", f"{sum(y) / n:.2f}")

with st.expander("查看解析后的数据（已标注用途）"):
    st.dataframe(
        pd.DataFrame({
            "序号": range(1, n + 1),
            "时点": [f"T{i + 1}" for i in range(n)],
            "观测值": y,
            "用途": ["训练段（建模）"] * train_n + ["留出段（检验）"] * horizon,
        }),
        hide_index=True, height=280)

if not chosen:
    st.warning("请至少启用一个模型。")
    st.stop()

try:
    results, info = core.run_holdout(
        y, forecast_start, models=tuple(chosen), ma_order=ma_order,
        ses_s0=ses_s0, ses_avg_n=ses_avg_n, alpha=ses_alpha,
        confidence=confidence)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

z = core.z_value(confidence)
labels = [f"T{i + 1}" for i in range(n)]

# --------------------------------------------------------------------------
# ② 模型指标对比
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# ② 模型公式
# --------------------------------------------------------------------------

st.subheader("② 模型公式")
st.caption("统一记号：t 为期数（1, 2, 3…），y_t 为第 t 期的实际值，"
           "ŷ_t 为第 t 期的预测值；n 为训练段期数，T 为训练段末期数。")

_FORMULA_LATEX = {
    "MA": (r"\hat{y}_t = \frac{1}{k}\sum_{i=1}^{k} y_{t-i}",
           "k 期移动平均：把最近 k 期的平均值作为预测值。"),
    "SES": (r"S_t = \alpha\,Y_t + (1-\alpha)\,S_{t-1},\qquad \hat{y}_t = S_{t-1}",
            "一次指数平滑：S0 由你指定或取前几期实际值的平均，之后逐期递推；"
            "第 t 期的预测值取上一期的平滑值 S_(t−1)。"),
    "Naive": (r"\hat{y}_t = y_{t-1}",
              "朴素法：第 t 期的预测值直接取第 t−1 期的实际值（滚动一步向前）。"),
    "Linear": (r"\hat{y}_t = a + b\,t,\qquad "
               r"b = \frac{\sum_{i=1}^{n}(t_i-\bar{t})(y_i-\bar{y})}"
               r"{\sum_{i=1}^{n}(t_i-\bar{t})^2},\qquad a = \bar{y} - b\,\bar{t}",
               "线性趋势模型：对训练段的 n 期数据做最小二乘拟合，得到截距 a 与斜率 b，"
               "再把期数代入方程向外推。"),
}
_SE_LATEX = {
    "MA": r"SE(h) = SE",
    "SES": r"SE(h) = SE",
    "Naive": r"SE(h) = SE",
    "Linear": r"SE(h) = s\sqrt{1+\frac{1}{n}+\frac{(T+h-\bar{t})^2}"
              r"{\sum_{i=1}^{n}(t_i-\bar{t})^2}}",
}

for r in results:
    st.markdown(f"**{r.name}**　·　{r.params}")
    st.latex(_FORMULA_LATEX[r.id][0])
    st.caption(_FORMULA_LATEX[r.id][1])
    if r.id == "Linear":
        fi = r.fit_info
        sign = "+" if fi["b"] >= 0 else "−"
        st.latex(rf"\hat{{y}}_t = {fi['a']:.4f} {sign} {abs(fi['b']):.4f}\,t"
                 rf"\qquad (t = 1, 2, \ldots)")
        st.caption("把上面这个方程里的 t 依次取训练段之后的各期期数，就得到下面这些预测值。")
    if r.id == "SES":
        fi = r.fit_info
        src = "手动输入" if fi["given"] else f"前 {fi['avg_n']} 期实际值的平均"
        st.latex(rf"S_0 = {fi['s0']:.4f},\qquad \alpha = {fi['alpha']:g}")
        st.caption(f"当前 S0 的来源：{src}。递推从 S0 开始："
                   f"S_t = α·Y_t + (1−α)·S_(t−1)，第 t 期的预测值取 S_(t−1)。")
    st.markdown("预测标准误：")
    st.latex(_SE_LATEX[r.id])
    if r.id == "Linear":
        fi = r.fit_info
        st.caption(f"其中 s = {fi['s']:.4f}（训练段回归残差标准误，自由度 n−2），"
                   f"n = {fi['n']}，t̄ = {fi['xbar']:.4f}，"
                   f"Σ(t−t̄)² = {fi['sxx']:.4f}，h 为外推步长。")
    st.divider()

st.subheader("③ 模型指标对比（留出段）")

best_u2 = min(results, key=lambda r: r.stats["u2"])
rows = []
for r in results:
    s = r.stats
    star = "★ " if r is best_u2 else ""
    rows.append({
        "模型": star + r.name,
        "参数": r.params,
        "留出期数 H": s["n"],
        "SSE": round(s["sse"], 4),
        "MSE": round(s["mse"], 4),
        "RMSE": round(s["rmse"], 4),
        "残差标准误 SE": round(s["se"], 4),
        "自由度": s["df"],
        "Theil U1": round(s["u1"], 4),
        "Theil U2": round(s["u2"], 4),
    })
st.caption(f"训练段 = 第 1 ~ {train_n} 期；留出段 = 第 {info['forecast_start']} ~ {n} 期"
           f"（共 {horizon} 期）。所有指标都在留出段上计算。"
           "SSE / SE / U1 / U2 越小越好；Theil U2 < 1 表示优于朴素法。")
st.dataframe(pd.DataFrame(rows), hide_index=True)

st.success(f"按 Theil U2 判断，当前最优模型：{best_u2.name}"
           f"（U2 = {best_u2.stats['u2']:.4f}）。")

# --------------------------------------------------------------------------
# ③ 留出段：预测值 vs 实际值
# --------------------------------------------------------------------------

st.subheader("④ 留出段预测值与实际值对照")

prows = []
for i, t in enumerate(info["t_index"]):
    row = {"时点": labels[t], "实际值": y[t]}
    for r in results:
        row[f"{r.key} 预测"] = round(r.forecast[i], 4)
        row[f"{r.key} 误差"] = round(r.errors[i], 4)
    prows.append(row)
st.dataframe(pd.DataFrame(prows), hide_index=True)

_ses_res = [r for r in results if r.id == "SES"]
if _ses_res:
    with st.expander(f"查看 SES 递推全过程：S0 ~ S{n}"):
        _S = _ses_res[0].fit_info["s_series"]
        st.caption("S_t = α·Y_t + (1−α)·S_(t−1)；第 t 期的预测值取 S_(t−1)。")
        _srows = []
        for _t in range(len(_S)):
            _srows.append({
                "平滑值": f"S{_t}",
                "计算依据": "初始值" if _t == 0 else f"α·Y{_t} + (1−α)·S{_t - 1}",
                "该期实际值": None if _t == 0 else y[_t - 1],
                "数值": round(_S[_t], 4),
                "用作预测": f"第 {_t + 1} 期" if _t < n else "—",
            })
        st.dataframe(pd.DataFrame(_srows), hide_index=True)

# --------------------------------------------------------------------------
# ④ 预测标准误与预测区间
# --------------------------------------------------------------------------

st.subheader("⑤ 预测标准误与预测区间")
st.caption(f"SE(h) 为第 h 期预测标准误；预测区间 = 点预测 ± z·SE(h)，z = {z:.4f}")

srows = []
for i, t in enumerate(info["t_index"]):
    row = {"时点": labels[t]}
    for r in results:
        fc, se = r.forecast[i], r.se_pred[i]
        row[f"{r.key} SE(h)"] = round(se, 4)
        row[f"{r.key} {confidence:.0%} 区间"] = f"[{fc - z * se:.4f}, {fc + z * se:.4f}]"
    srows.append(row)
st.dataframe(pd.DataFrame(srows), hide_index=True)

# --------------------------------------------------------------------------
# ⑤ 图表
# --------------------------------------------------------------------------

st.subheader("⑥ 图表")
tab1, tab2, tab3 = st.tabs(["整体走势与预测", "训练段拟合", "留出段误差"])

with tab1:
    _chart(history_forecast_fig(labels, y, train_n, results, z))
    st.caption("深色为建模用的训练段，浅色为留作检验的留出段；"
               "虚线是各模型对留出段的预测——Naive 为「上一期实际值」的滚动预测，"
               "竖线段为 ±z·SE(h)。")

with tab2:
    _chart(train_fit_fig(labels, y, train_n, results))
    st.caption("各模型在训练段上的一步向前拟合值；移动平均的前几期没有拟合值。")

with tab3:
    left, right = st.columns(2)
    with left:
        _chart(holdout_error_fig(labels, results))
    with right:
        _chart(sse_fig(results))

with st.expander("计算公式与口径说明"):
    st.markdown(
        """
- 数据被切成两段：**训练段**（前 $T_1$ 期）只用于建模，**留出段**（后 $H$ 期）只用于检验。
- 移动平均与 SES 在训练段末端**一次性向后外推** $H$ 期，各期预测值相同；
  **Naive 为滚动一步向前**：第 $T$ 期的预测值取第 $T-1$ 期的实际值，
  因此它的留出段误差就是逐期差分。
- **残差标准误 SE** $=\\sqrt{SSE/(H-k)}$，$k$ 为待估参数个数（线性趋势模型取 2；SES 自动取平均时取 1，手动指定时取 0；其余取 0）。
- **预测期标准误 SE(h)**：移动平均与 Naive 为常数 $SE$（都是水平 / 一步预测）；
  移动平均、SES 与 Naive 都是常数 $SE$（都是水平 / 一步预测）；
  线性趋势模型按回归预测区间公式 $s\\sqrt{1+1/n+(T+h-\\bar t)^2/\\sum(t-\\bar t)^2}$ 计算，随外推步长增大。
- **Theil U1** $= RMSE / (\\sqrt{\\overline{\\hat y^2}} + \\sqrt{\\overline{y^2}})$，取值 0~1，越小越好。
- **Theil U2** $= \\sqrt{\\sum e_h^2 / \\sum (y_t-y_{t-1})^2}$，分母取留出段上的逐期差分，
  即「每期都用上一期实际值」的朴素预测误差。
        """
    )
