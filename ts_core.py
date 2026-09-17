# -*- coding: utf-8 -*-
"""
==========================================================================
 ts_core.py —— 时间序列预测核心计算模块（留出法 / holdout）
==========================================================================
不依赖 Streamlit / 图形界面，可被 app.py、ts_forecast.py 或你自己的脚本复用。

核心思路（留出法）
------------------
把整段序列从"预测起始期"处切成两段：

    前 train_n 期 —— 训练段，只用来建立模型
    后 horizon 期 —— 留出段，用来检验模型（不参与建模）

模型在训练段上估计，然后一次性向后外推 horizon 期（各期预测值相同，属标准
的样本外多步预测）。所有评价指标都用留出段的"实际值 vs 预测值"计算，因此
衡量的是真正的预测能力，而不是样本内拟合优度。

公开接口
--------
    parse_series(text)              从粘贴的文本解析数值序列
    z_value(confidence)             正态分布双侧分位数
    fit_naive(train, horizon)       Naive 朴素法
    fit_ma(train, order, horizon)   k 期移动平均
    fit_ses(train, horizon, alpha, level)  一次指数平滑（给定起始值后逐期递推）
    fit_linear(train, horizon)      线性趋势模型（最小二乘拟合）
    run_holdout(...)                切分 + 建模 + 留出段评估
    MODELS                          模型标识，按显示顺序：
                                    ("Linear", "MA", "SES", "Naive")

统计口径
--------
  留出段各期预测误差 e_h = y_{train_n+h} - y_hat_h，H = 留出期数，
  k 为该模型待估参数个数。

    SSE = Σ e_h²          MSE = SSE / H          RMSE = √MSE
    残差标准误 SE = √( SSE / (H - k) )            k：Naive、MA 取 0，SES 取 1
    预测期标准误 SE(h)：
        Naive（滚动一步向前）: SE
        MA（水平预测）  : SE
        SES            : SE（每期都是用上一期实际值做一步预测）
        Linear（趋势外推）: s·√( 1 + 1/n + (T+h−t̄)²/Σ(t−t̄)² )，
                           s 为训练段回归残差标准误（随外推步长增大）

  Theil 统计量：
    U1 = RMSE / ( √(mean(y_hat²)) + √(mean(y²)) )     0 ≤ U1 ≤ 1，越小越好
    U2 = √( Σe² / Σ(y_t - y_{t-1})² )                 与"无变化"朴素基准比较，
                                                      <1 优于朴素法，=1 持平，>1 更差
    （U2 的分母同样取留出段上的逐期差分，即"每一步都用上一期实际值"的朴素预测误差）
==========================================================================
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from statistics import NormalDist

__all__ = [
    "MODELS", "MODEL_LABELS", "ModelResult", "parse_series", "z_value",
    "fit_naive", "fit_ma", "fit_ses", "run_holdout", "compute_stats",
]

# 模型标识 —— 顺序即界面/报表中的显示顺序
MODELS = ("Linear", "MA", "SES", "Naive")

# 模型标识 -> 界面显示名
MODEL_LABELS = {
    "MA": "移动平均（MA）",
    "SES": "SES 一次指数平滑",
    "Naive": "Naive 朴素法",
    "Linear": "Linear 线性趋势模型",
}


@dataclass
class ModelResult:
    """单个模型在训练段上的拟合、以及留出段的预测与评估结果。"""

    id: str            # 模型标识："MA" / "SES" / "Naive"
    key: str           # 简写标签，如 "3MA" / "SES" / "Naive"
    name: str          # 完整名称
    params: str        # 参数说明
    k: int             # 待估参数个数（用于 SE 自由度）
    se_kind: str       # 预测期 SE 类型："rw" / "flat" / "ses"
    train_n: int       # 训练段期数
    horizon: int       # 留出段期数
    point: float       # 外推得到的点预测（各期相同；滚动模型的参考值）
    mode: str = "flat"      # 留出段预测方式：flat 各期相同 / rolling 滚动 / trend 沿趋势
    fit_info: dict = field(default_factory=dict)   # 线性趋势模型的拟合参数
    # ---- 训练段：一步向前拟合（仅用于展示拟合效果）----
    train_index: list = field(default_factory=list)    # 原始下标（0 基）
    train_actual: list = field(default_factory=list)
    train_fitted: list = field(default_factory=list)
    train_errors: list = field(default_factory=list)
    # ---- 留出段：预测 vs 实际（评价用）----
    t_index: list = field(default_factory=list)        # 留出段的原始下标
    actual: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    forecast: list = field(default_factory=list)       # 各期预测值
    se_pred: list = field(default_factory=list)        # 各期预测标准误
    stats: dict = field(default_factory=dict)

    def aligned_train_fitted(self, n):
        """把训练段的拟合值对齐到整段序列，其余位置用 None 占位。"""
        out = [None] * n
        for t, f in zip(self.train_index, self.train_fitted):
            out[t] = f
        return out

    def aligned_train_errors(self, n):
        out = [None] * n
        for t, e in zip(self.train_index, self.train_errors):
            out[t] = e
        return out


# --------------------------------------------------------------------------
# 输入解析
# --------------------------------------------------------------------------

def parse_series(text):
    """把粘贴的文本解析为数值列表。

    支持逗号、空格、换行、制表符、分号，以及中文的逗号和顿号作分隔符。
    """
    if text is None:
        raise ValueError("数据为空。")
    tokens = [t for t in re.split(r"[,，、;；\s]+", str(text).strip()) if t]
    if not tokens:
        raise ValueError("没有解析到任何数据，请检查输入内容。")
    values = []
    for tok in tokens:
        try:
            values.append(float(tok))
        except ValueError:
            raise ValueError(f"无法识别的数字：{tok!r}") from None
    return values


def z_value(confidence):
    """正态分布双侧分位数，如 95% 对应 1.95996。"""
    if not 0 < confidence < 1:
        raise ValueError("置信水平必须在 0 和 1 之间。")
    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


# --------------------------------------------------------------------------
# 三个模型（都在训练段上估计，并给出外推点预测）
# --------------------------------------------------------------------------

def fit_naive(train, horizon):
    """朴素法：第 T 期的预测值 = 第 T-1 期的实际值（滚动一步向前）。

    例如训练段为前 5 期，则第 6 期用 y5，第 7 期用 y6，依此类推。
    因此 Naive 在留出段的预测误差就是逐期差分，Theil U2 恒等于 1。
    """
    n = len(train)
    if n < 2:
        raise ValueError("朴素法至少需要 2 期训练数据。")
    ti = list(range(1, n))
    fitted = [train[t - 1] for t in ti]
    errors = [train[t] - train[t - 1] for t in ti]
    return ModelResult(
        id="Naive", key="Naive", name="Naive 朴素法", params="无待估参数",
        k=0, se_kind="flat", mode="rolling",
        train_n=n, horizon=horizon, point=train[-1],
        train_index=ti, train_actual=[train[t] for t in ti],
        train_fitted=fitted, train_errors=errors,
    )


def fit_ma(train, order, horizon):
    """k 期移动平均：外推预测值 = 训练段最后 order 期的平均值。"""
    n = len(train)
    if order < 2:
        raise ValueError("移动平均的阶数至少为 2。")
    if n < order + 1:
        raise ValueError(
            f"{order} 期移动平均至少需要 {order + 1} 期训练数据，"
            f"当前训练段只有 {n} 期；请把预测起始期数调大或减小阶数。")
    ti = list(range(order, n))
    fitted = [sum(train[t - order:t]) / order for t in ti]
    errors = [train[t] - f for t, f in zip(ti, fitted)]
    return ModelResult(
        id="MA", key=f"{order}MA", name=f"{order}期移动平均",
        params=f"阶数 = {order}", k=0, se_kind="flat",
        train_n=n, horizon=horizon, point=sum(train[-order:]) / order,
        train_index=ti, train_actual=[train[t] for t in ti],
        train_fitted=fitted, train_errors=errors,
    )


def fit_linear(train, horizon):
    """线性趋势模型：最小二乘拟合  y_t = a + b·t （t = 1, 2, ... 为期数）。

        b = Σ(t−t̄)(y_t−ȳ) / Σ(t−t̄)²          a = ȳ − b·t̄
        y_hat_t = a + b·t                      （t 取到留出段各期即得外推预测）
    """
    n = len(train)
    if n < 3:
        raise ValueError("线性趋势模型至少需要 3 期训练数据。")
    x = [i + 1 for i in range(n)]              # 期数 1, 2, ..., n
    xbar = sum(x) / n
    ybar = sum(train) / n
    sxx = sum((xi - xbar) ** 2 for xi in x)
    if sxx == 0:
        raise ValueError("训练段期数过少，无法拟合线性趋势。")
    sxy = sum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, train))
    b = sxy / sxx
    a = ybar - b * xbar
    fitted = [a + b * xi for xi in x]
    errors = [yi - f for yi, f in zip(train, fitted)]
    resid_sse = sum(e * e for e in errors)
    s = math.sqrt(resid_sse / (n - 2))         # 回归残差标准误（自由度 n−2）
    return ModelResult(
        id="Linear", key="Linear", name="Linear 线性趋势模型",
        params=f"a = {a:.4f}, b = {b:.4f}", k=2, se_kind="trend",
        mode="trend", train_n=n, horizon=horizon, point=a + b * (n + 1),
        train_index=list(range(n)), train_actual=list(train),
        train_fitted=fitted, train_errors=errors,
        fit_info=dict(a=a, b=b, s=s, n=n, xbar=xbar, sxx=sxx),
    )


def fit_ses(train, horizon, alpha=0.3, level=None):
    """一次指数平滑（SES）：给定初始值后逐期递推。

        S_(t+1) = α·Y_t + (1−α)·S_t          （S_t 表示第 t 期的预测值）

    起始值 level 即第 T−1 期（训练段最后一期）的预测值 S(T−1)：
        · 传入了 level（数字或数字字符串）→ 直接采用该数值；
        · 未传入（None 或空字符串）      → 取训练段 n 期实际值的算术平均。
    递推在 run_holdout 中完成：第 T 期的预测值 =
    α·(第 T−1 期实际值) + (1−α)·S(T−1)，之后每期用当期实际值与当期预测值算出下期预测值。
    待估参数个数 k：起始值由数据估计（取平均）时计 1，手动指定时计 0。
    """
    n = len(train)
    if not 0 < alpha < 1:
        raise ValueError("平滑系数 α 必须满足 0 < α < 1。")
    if n < 2:
        raise ValueError("一次指数平滑至少需要 2 期训练数据。")

    mean = sum(train) / n
    given = level is not None and str(level).strip() != ""
    if given:
        try:
            lvl = float(level)
        except (TypeError, ValueError):
            raise ValueError(f"起始值 S(T−1) 必须是数字，收到：{level!r}") from None
        src, k = "手动指定", 0
    else:
        lvl, src, k = mean, "训练段均值", 1
    return ModelResult(
        id="SES", key="SES", name="SES 一次指数平滑",
        params=f"α = {alpha:g}, S(T−1) = {lvl:.4f}（{src}）",
        k=k, se_kind="flat", mode="ses",
        train_n=n, horizon=horizon, point=lvl,
        fit_info=dict(level=lvl, given=given, mean=mean, n=n, alpha=alpha),
    )








# --------------------------------------------------------------------------
# 统计量
# --------------------------------------------------------------------------

def compute_stats(actual, forecast, errors, k, y, t_index):
    """在留出段上计算 SSE / MSE / RMSE / SE / U1 / U2。

    actual / forecast / errors : 留出段各期实际值、预测值、误差
    y        : 完整序列（U2 的分母要用到 y_{t-1}）
    t_index  : 留出段各期在完整序列中的下标（0 基）
    """
    h = len(actual)
    if h == 0:
        raise ValueError("留出段为空，请检查预测起始期数。")

    sse = sum(e * e for e in errors)
    mse = sse / h
    rmse = math.sqrt(mse)
    df = max(h - k, 1)
    se = math.sqrt(sse / df)

    # Theil U1：Theil 不等系数，0 ≤ U1 ≤ 1，越接近 0 越好
    rms_f = math.sqrt(sum(f * f for f in forecast) / h)
    rms_a = math.sqrt(sum(a * a for a in actual) / h)
    u1 = rmse / (rms_f + rms_a) if (rms_f + rms_a) > 0 else float("nan")

    # Theil U2：以"无变化"朴素法为基准，U2 < 1 说明优于朴素法
    denom = sum((y[t] - y[t - 1]) ** 2 for t in t_index)
    u2 = math.sqrt(sse / denom) if denom > 0 else float("nan")

    return dict(n=h, sse=sse, mse=mse, rmse=rmse, se=se, df=df, u1=u1, u2=u2)


def _se_pred(r, horizon):
    """按模型类型给出留出段各期（h = 1..H）的预测标准误 SE(h)。"""
    se = r.stats["se"]
    if r.se_kind == "flat":                # 水平 / 一步预测：不累积
        return [se] * horizon
    if r.se_kind == "trend":
        # 回归预测区间口径：基于训练段残差标准误，随外推步长增大
        fi = r.fit_info
        return [fi["s"] * math.sqrt(
                    1 + 1.0 / fi["n"] + (r.train_n + h - fi["xbar"]) ** 2 / fi["sxx"])
                for h in range(1, horizon + 1)]
    raise ValueError(f"未知的 SE 类型：{r.se_kind}")




# --------------------------------------------------------------------------
# 主入口：留出法
# --------------------------------------------------------------------------

def run_holdout(y, forecast_start, models=MODELS, ma_order=3, ses_level=None,
                alpha=0.3, confidence=0.95):
    """切分序列、在训练段上建模、在留出段上评估。

    参数
    ----
    y              : 观测值列表（按时间先后）
    forecast_start : 从第几期开始预测（1 基）。例如 6 表示用前 5 期建模、
                     预测第 6 期起的全部观测，因此训练段 = y[:5]。
    models         : 启用哪些模型，MODELS 的子集，如 ("MA", "SES")
    ma_order       : 移动平均阶数 k（默认 3，即 3MA）
    ses_level      : SES 的起始平滑值 S(T−1)；None / 留空表示自动取训练段实际值平均
    alpha          : SES 平滑系数，用于逐期递推
    confidence     : 预测区间置信水平（用于校验）

    返回
    ----
    (results, info) : 各模型结果列表，以及含切分信息的字典
    """
    y = [float(v) for v in y]
    n = len(y)
    if n < 3:
        raise ValueError("至少需要 3 期观测值。")

    try:
        forecast_start = int(forecast_start)
    except (TypeError, ValueError):
        raise ValueError("预测起始期数必须是整数。") from None

    if forecast_start > n:
        raise ValueError(f"预测起始期数不能超过总期数 {n}。")
    if forecast_start < 3:
        raise ValueError("预测起始期数至少要设为 3，即至少用前 2 期建模。")

    train_n = forecast_start - 1
    horizon = n - train_n
    z_value(confidence)                  # 提前校验置信水平

    enabled = [m for m in MODELS if m in models]
    if not enabled:
        raise ValueError("至少要启用一个模型。")

    train = y[:train_n]
    results = []
    for mid in enabled:
        if mid == "MA":
            results.append(fit_ma(train, ma_order, horizon))
        elif mid == "SES":
            results.append(fit_ses(train, horizon, alpha, ses_level))
        elif mid == "Linear":
            results.append(fit_linear(train, horizon))
        else:
            results.append(fit_naive(train, horizon))

    # 留出段：从 train_n 到 n-1（0 基下标）
    t_index = list(range(train_n, n))
    for r in results:
        r.t_index = list(t_index)
        r.actual = [y[t] for t in t_index]
        if r.mode == "rolling":            # 滚动一步向前：取上一期实际值
            r.forecast = [y[t - 1] for t in t_index]
        elif r.mode == "trend":            # 沿线性趋势外推（t 为 1 基期数）
            fi = r.fit_info
            r.forecast = [fi["a"] + fi["b"] * (t + 1) for t in t_index]
        elif r.mode == "ses":              # S_(t+1) = α·Y_t + (1−α)·S_t
            lvl, fc = r.point, []
            for t in t_index:
                lvl = alpha * y[t - 1] + (1 - alpha) * lvl   # 用上一期实际值更新
                fc.append(lvl)                               # 得到本期的预测值
            r.forecast = fc
        else:                              # 一次性外推：各期相同
            r.forecast = [r.point] * horizon
        r.errors = [a - f for a, f in zip(r.actual, r.forecast)]
        r.stats = compute_stats(r.actual, r.forecast, r.errors, r.k, y, t_index)
        r.se_pred = _se_pred(r, horizon)

    info = dict(n=n, train_n=train_n, horizon=horizon,
                forecast_start=train_n + 1, t_index=t_index)
    return results, info
